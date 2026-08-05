#!/usr/bin/env python3
"""Alternate chunked 8-GPU training and checkpoint evaluation on one machine."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def ensure_gpus_idle() -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    active = [line for line in result.stdout.splitlines() if line.strip()]
    if active:
        raise RuntimeError("GPU compute processes are active:\n" + "\n".join(active))


def checkpoint_step(checkpoint: Path) -> int:
    code = (
        "import torch; print(int(torch.load(" + repr(str(checkpoint))
        + ",map_location='cpu',mmap=True,weights_only=False).get('step',0)))"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def run_logged(command: list[str], env: dict[str, str], log_file) -> None:
    print("[supervisor] exec:", " ".join(command), flush=True)
    log_file.write("[supervisor] exec: " + " ".join(command) + "\n")
    log_file.flush()
    subprocess.run(command, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--skip-initial-idle-check", action="store_true")
    args = parser.parse_args()

    train_raw = yaml.safe_load(args.train_config.read_text())
    eval_raw = yaml.safe_load(args.eval_config.read_text()).get("evaluation", {})
    output_root = Path(train_raw["train"]["output_dir"])
    run_dir = output_root / args.run_name
    checkpoint = run_dir / "ckpt.pt"
    interval = int(eval_raw.get("interval_steps", 1000))
    first_eval = int(eval_raw.get("enabled_after_step", interval))
    if interval <= 0 or args.total_steps <= 0:
        raise ValueError("interval_steps and total_steps must be positive")
    if not args.skip_initial_idle_check:
        ensure_gpus_idle()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "supervisor_state.json"
    current = checkpoint_step(checkpoint) if checkpoint.exists() else 0
    torchrun = str(Path(sys.executable).with_name("torchrun"))
    base_env = dict(os.environ, PYTHONPATH=str(Path.cwd()), UV3_RUN_NAME=args.run_name)
    with (run_dir / "stdout.log").open("a", encoding="utf-8") as log_file:
        while current < args.total_steps:
            segment_end = min(args.total_steps, ((current // interval) + 1) * interval)
            train_env = dict(base_env, UV3_MAX_STEPS=str(segment_end))
            run_logged(
                [
                    torchrun, "--standalone", f"--nproc_per_node={args.nproc}",
                    "-m", "uv3.train.fsdp2_trainer", "--config", str(args.train_config),
                ],
                train_env,
                log_file,
            )
            current = checkpoint_step(checkpoint)
            if current < segment_end:
                raise RuntimeError(f"checkpoint step {current} is behind expected {segment_end}")
            should_eval = current >= first_eval and (
                current % interval == 0 or current == args.total_steps
            )
            if should_eval:
                run_logged(
                    [
                        torchrun, "--standalone", f"--nproc_per_node={args.nproc}",
                        "-m", "uv3.eval.checkpoint_evaluator",
                        "--train-config", str(args.train_config),
                        "--eval-config", str(args.eval_config),
                        "--checkpoint", str(checkpoint),
                        "--run-dir", str(run_dir),
                    ],
                    base_env,
                    log_file,
                )
            state_path.write_text(
                json.dumps({"step": current, "total_steps": args.total_steps}, indent=2) + "\n"
            )
    print(f"[supervisor] DONE step={current} run={run_dir}", flush=True)


if __name__ == "__main__":
    main()
