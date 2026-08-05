#!/usr/bin/env python3
"""Create a deterministic shard-level train/holdout split and fixed eval cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def score(value: str, salt: str) -> bytes:
    return hashlib.sha256(f"{salt}\0{value}".encode()).digest()


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def fixed_candidates(
    shards: list[dict], split: str, count: int, caption_field: str, salt: str
) -> list[dict]:
    # A bounded shard subset is enough for a small visual panel and avoids
    # opening tens of thousands of parquet files.
    candidate_shards = sorted(
        shards, key=lambda row: score(row["shard_id"], f"{salt}:{split}:shard")
    )[:64]
    candidates = []
    columns = ["shard_key", "filename", "offset", "size", "width", "height", caption_field]
    for shard in candidate_shards:
        table = pq.read_table(shard["metadata_parquet"], columns=columns)
        for row in table.to_pylist():
            caption = row.get(caption_field)
            if not caption:
                continue
            case_id = f'{shard["shard_id"]}:{row["filename"]}'
            candidates.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "caption": caption,
                    "image_tar": shard["image_tar"],
                    "filename": row["filename"],
                    "offset": row["offset"],
                    "size": row["size"],
                    "width": row["width"],
                    "height": row["height"],
                    "seed": int.from_bytes(score(case_id, f"{salt}:seed")[:4], "big"),
                }
            )
    candidates.sort(key=lambda row: score(row["case_id"], f"{salt}:{split}:case"))
    return candidates[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-shards", type=int, default=64)
    parser.add_argument("--fixed-per-split", type=int, default=16)
    parser.add_argument("--caption-field", default="caption_qwen3_7_flash")
    parser.add_argument("--salt", default="uv3-eval-v1")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if not 0 < args.holdout_shards < len(rows):
        raise ValueError("holdout-shards must be between 1 and manifest size - 1")
    ordered = sorted(rows, key=lambda row: score(row["shard_id"], args.salt))
    holdout_ids = {row["shard_id"] for row in ordered[: args.holdout_shards]}
    holdout = [row for row in rows if row["shard_id"] in holdout_ids]
    train = [row for row in rows if row["shard_id"] not in holdout_ids]

    fixed = fixed_candidates(
        train, "train", args.fixed_per_split, args.caption_field, args.salt
    ) + fixed_candidates(
        holdout, "heldout", args.fixed_per_split, args.caption_field, args.salt
    )
    write_jsonl(args.output_dir / "train_manifest.jsonl", train)
    write_jsonl(args.output_dir / "holdout_manifest.jsonl", holdout)
    write_jsonl(args.output_dir / "fixed_cases.jsonl", fixed)
    summary = {
        "source_manifest": str(args.manifest),
        "salt": args.salt,
        "train_shards": len(train),
        "holdout_shards": len(holdout),
        "estimated_holdout_samples": sum(int(row.get("samples", 0)) for row in holdout),
        "fixed_train_cases": sum(row["split"] == "train" for row in fixed),
        "fixed_heldout_cases": sum(row["split"] == "heldout" for row in fixed),
    }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
