#!/usr/bin/env python3
"""Persistent 20-minute audit and safe restart watchdog for one overfit run."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def last_jsonl(path: Path) -> dict:
    latest: dict = {}
    try:
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        latest = value
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return latest


def matching_pids(needle: str) -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if needle in command and "monitor_overfit_run.py" not in command:
            found.append(int(entry.name))
    return found


def owned_orphan_pids(run_name: str) -> list[int]:
    """Return only orphaned trainer/evaluator processes belonging to this run."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
            parent = int(next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:")))
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            environ = (entry / "environ").read_bytes().split(b"\0")
        except (OSError, StopIteration, ValueError):
            continue
        has_run = f"UV3_RUN_NAME={run_name}".encode() in environ
        is_worker = any(name in command for name in ("fsdp2_trainer", "checkpoint_evaluator"))
        if parent == 1 and has_run and is_worker:
            found.append(int(entry.name))
    return found


def gpu_rows() -> list[dict]:
    query = "index,memory.used,utilization.gpu,power.draw,temperature.gpu"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    keys = ("index", "memory_mib", "util_pct", "power_w", "temperature_c")
    return [
        dict(zip(keys, (float(value.strip()) for value in line.split(","))))
        for line in result.stdout.splitlines() if line.strip()
    ]


def gpus_idle(rows: list[dict]) -> bool:
    return all(row["memory_mib"] < 1024 for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--interval", type=int, default=1200)
    args = parser.parse_args()
    audit_path = args.run_dir / "watchdog_audit.jsonl"
    restart_count = 0

    while True:
        now = time.time()
        latest = last_jsonl(args.run_dir / "metrics.jsonl")
        rows = gpu_rows()
        supervisor_needle = f"train_eval_supervisor.py --train-config {args.train_config}"
        supervisors = matching_pids(supervisor_needle)
        checkpoint = args.run_dir / "ckpt.pt"
        samples = args.run_dir / "samples"
        record = {
            "timestamp": now,
            "step": latest.get("step"),
            "loss": latest.get("loss"),
            "steps_per_second": latest.get("spd"),
            "metrics_age_seconds": (
                now - (args.run_dir / "metrics.jsonl").stat().st_mtime
                if (args.run_dir / "metrics.jsonl").exists() else None
            ),
            "supervisor_pids": supervisors,
            "checkpoint_exists": checkpoint.is_file(),
            "sample_png_count": sum(1 for _ in samples.rglob("*.png")) if samples.exists() else 0,
            "gpus": rows,
        }
        completed = int(latest.get("step") or 0) >= args.total_steps
        metric_age = record["metrics_age_seconds"]
        if not supervisors and not completed and metric_age is not None and metric_age > args.interval:
            orphans = owned_orphan_pids(args.run_name)
            if orphans:
                for pid in orphans:
                    try:
                        os.kill(pid, 15)
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and any(Path(f"/proc/{pid}").exists() for pid in orphans):
                    time.sleep(1)
                remaining = [pid for pid in orphans if Path(f"/proc/{pid}").exists()]
                record.update({"orphan_term_pids": orphans, "orphan_remaining_pids": remaining})
                rows = gpu_rows()
                record["gpus_after_orphan_term"] = rows
                if remaining:
                    record["action"] = "manual_intervention_required"
        can_restart = record.get("action") != "manual_intervention_required"
        if not supervisors and not completed and can_restart and gpus_idle(rows) and restart_count < 3:
            args.run_dir.mkdir(parents=True, exist_ok=True)
            log = (args.run_dir / "supervisor.log").open("a", encoding="utf-8")
            command = [
                str(args.repo / ".venv-cu128/bin/python"),
                str(args.repo / "scripts/train_eval_supervisor.py"),
                "--train-config", args.train_config,
                "--eval-config", args.eval_config,
                "--run-name", args.run_name,
                "--total-steps", str(args.total_steps), "--nproc", "8",
            ]
            process = subprocess.Popen(
                command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            restart_count += 1
            record.update({"action": "restart", "new_supervisor_pid": process.pid})
        with audit_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if completed:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
