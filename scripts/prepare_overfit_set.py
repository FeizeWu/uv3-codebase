#!/usr/bin/env python3
"""Select a reproducible fixed overfit set from the training-only manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_eval_split import fixed_candidates, read_manifest, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--visual-train", type=int, default=32)
    parser.add_argument("--caption-field", default="caption_qwen3_7_flash")
    parser.add_argument("--salt", default="uv3-overfit-256-v1")
    args = parser.parse_args()

    overfit_path = args.output_dir / "overfit_256.jsonl"
    if overfit_path.exists():
        with overfit_path.open(encoding="utf-8") as file:
            overfit = [json.loads(line) for line in file if line.strip()]
    else:
        shards = read_manifest(args.train_manifest)
        overfit = fixed_candidates(
            shards, "train", args.count, args.caption_field, args.salt
        )
    if len(overfit) != args.count:
        raise RuntimeError(f"only selected {len(overfit)} of {args.count} requested samples")
    visual = overfit[: args.visual_train]
    if not overfit_path.exists():
        write_jsonl(overfit_path, overfit)
    write_jsonl(args.output_dir / "overfit_visual_train_cases.jsonl", visual)
    print(json.dumps({"overfit": len(overfit), "visual": len(visual)}, indent=2))


if __name__ == "__main__":
    main()
