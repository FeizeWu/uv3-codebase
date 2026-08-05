#!/usr/bin/env python3
"""Select eight fixed cases while retaining all production text-length buckets."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    buckets = (512, 640, 768, 896, 1024)
    wanted = {512: 2, 640: 1, 768: 2, 896: 1, 1024: 2}
    grouped = defaultdict(list)
    for row in rows:
        length = len(tokenizer(
            row["caption"], add_special_tokens=True, truncation=True, max_length=1024
        )["input_ids"])
        bucket = next((value for value in buckets if length <= value), buckets[-1])
        grouped[bucket].append(row)
    selected = []
    for bucket in buckets:
        if len(grouped[bucket]) < wanted[bucket]:
            raise RuntimeError(f"bucket {bucket} has only {len(grouped[bucket])} cases")
        selected.extend(grouped[bucket][:wanted[bucket]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"selected": len(selected), "bucket_counts": wanted}))


if __name__ == "__main__":
    main()
