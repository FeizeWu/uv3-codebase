#!/usr/bin/env python3
"""Generate fixed validation cases for every persisted checkpoint when GPUs are idle."""
from __future__ import annotations

import argparse
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


def step_from_snapshot(path: Path) -> int:
    return int(path.stem.removeprefix("ckpt-step"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--eval-config", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-step", type=int, default=10_000)
    parser.add_argument("--master-port", type=int, default=29685)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    while True:
        snapshots = sorted(
            args.snapshot_dir.glob("ckpt-step????????.pt"),
            key=step_from_snapshot,
        )
        pending = [
            path
            for path in snapshots
            if not (
                args.run_dir
                / "samples"
                / f"step_{step_from_snapshot(path):08d}"
                / "manifest.jsonl"
            ).is_file()
        ]
        if pending and gpus_idle():
            checkpoint = pending[0]
            step = step_from_snapshot(checkpoint)
            log_path = args.log_dir / f"eval-step{step:08d}-fixed.log"
            env = os.environ.copy()
            env.update(
                {
                    "RUN_NAME": args.run_dir.name,
                    "RUN_DIR": str(args.run_dir),
                    "TRAIN_CONFIG": str(args.run_dir / "config.yaml"),
                    "EVAL_CONFIG": str(args.eval_config),
                    "CHECKPOINT": str(checkpoint),
                    "SKIP_DISTRIBUTION": "1",
                    "REQUIRE_IDLE_GPUS": "1",
                    "MASTER_PORT": str(args.master_port),
                }
            )
            print(f"[fixed-eval] starting step={step} checkpoint={checkpoint}", flush=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                result = subprocess.run(
                    [str(args.repo_root / "scripts" / "run_eval_4b_8gpu.sh")],
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            manifest = args.run_dir / "samples" / f"step_{step:08d}" / "manifest.jsonl"
            if result.returncode == 0 and manifest.is_file():
                print(f"[fixed-eval] completed step={step}", flush=True)
                if step >= args.max_step:
                    print(f"[fixed-eval] reached max_step={args.max_step}; done", flush=True)
                    return
            else:
                print(
                    f"[fixed-eval] failed step={step} status={result.returncode}; retrying later",
                    flush=True,
                )
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    main()
