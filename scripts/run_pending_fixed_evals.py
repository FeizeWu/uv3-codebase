#!/usr/bin/env python3
"""Run fixed cases every checkpoint and distribution metrics at a lower cadence."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def gpus_idle() -> bool:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return not result.stdout.strip()


def external_evaluation_running(run_dir: Path) -> bool:
    """Detect evaluations launched outside this synchronous supervisor."""
    for pid_path in run_dir.glob("eval-step*-*.pid"):
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
        except (OSError, ValueError):
            continue
        return True
    return False


def step_from_snapshot(path: Path) -> int:
    return int(path.stem.removeprefix("ckpt-step"))


def fixed_done(run_dir: Path, step: int) -> bool:
    return (run_dir / "samples" / f"step_{step:08d}" / "manifest.jsonl").is_file()


def distribution_done(run_dir: Path, step: int) -> bool:
    metadata_path = run_dir / "samples" / f"step_{step:08d}" / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metrics = metadata.get("metrics")
    return (
        isinstance(metrics, dict)
        and int(metrics.get("num_generated", 0)) > 0
        and metrics.get("fid") is not None
        and metrics.get("kid_mean") is not None
        and metrics.get("clipscore") is not None
    )


def run_evaluation(
    args: argparse.Namespace,
    checkpoint: Path,
    step: int,
    mode: str,
) -> bool:
    log_path = args.log_dir / f"eval-step{step:08d}-{mode}.log"
    env = os.environ.copy()
    env.update(
        {
            "RUN_NAME": args.run_dir.name,
            "RUN_DIR": str(args.run_dir),
            "TRAIN_CONFIG": str(args.run_dir / "config.yaml"),
            "EVAL_CONFIG": str(args.eval_config),
            "CHECKPOINT": str(checkpoint),
            "REQUIRE_IDLE_GPUS": "1",
            "MASTER_PORT": str(args.master_port),
        }
    )
    if mode == "fixed":
        env["SKIP_DISTRIBUTION"] = "1"
    elif mode == "distribution":
        env["SKIP_FIXED"] = "1"
    else:
        raise ValueError(f"unknown evaluation mode: {mode}")
    print(f"[eval-supervisor] starting mode={mode} step={step}", flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        result = subprocess.run(
            [str(args.repo_root / "scripts" / "run_eval_4b_8gpu.sh")],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    completed = (
        fixed_done(args.run_dir, step)
        if mode == "fixed"
        else distribution_done(args.run_dir, step)
    )
    if result.returncode == 0 and completed:
        print(f"[eval-supervisor] completed mode={mode} step={step}", flush=True)
        return True
    print(
        f"[eval-supervisor] failed mode={mode} step={step} "
        f"status={result.returncode}; retrying later",
        flush=True,
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--eval-config", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-step", type=int, default=10_000)
    parser.add_argument("--distribution-interval", type=int, default=2_000)
    parser.add_argument("--master-port", type=int, default=29685)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    while True:
        snapshots = sorted(
            args.snapshot_dir.glob("ckpt-step????????.pt"),
            key=step_from_snapshot,
        )
        pending_fixed = [
            path for path in snapshots if not fixed_done(args.run_dir, step_from_snapshot(path))
        ]
        pending_distribution = [
            path
            for path in snapshots
            if step_from_snapshot(path) % args.distribution_interval == 0
            and not distribution_done(args.run_dir, step_from_snapshot(path))
        ]
        if (
            (pending_fixed or pending_distribution)
            and not external_evaluation_running(args.run_dir)
            and gpus_idle()
        ):
            # Fixed cases are quick and make every checkpoint visually inspectable.
            # Run them before a long distribution job when both are pending.
            if pending_fixed:
                checkpoint = pending_fixed[0]
                run_evaluation(args, checkpoint, step_from_snapshot(checkpoint), "fixed")
            elif pending_distribution:
                checkpoint = pending_distribution[0]
                run_evaluation(args, checkpoint, step_from_snapshot(checkpoint), "distribution")

        max_snapshot = next(
            (path for path in snapshots if step_from_snapshot(path) >= args.max_step),
            None,
        )
        if max_snapshot is not None:
            max_snapshot_step = step_from_snapshot(max_snapshot)
            final_distribution_required = max_snapshot_step % args.distribution_interval == 0
            if fixed_done(args.run_dir, max_snapshot_step) and (
                not final_distribution_required
                or distribution_done(args.run_dir, max_snapshot_step)
            ):
                print(
                    f"[eval-supervisor] reached max_step={max_snapshot_step}; done",
                    flush=True,
                )
                return
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    main()
