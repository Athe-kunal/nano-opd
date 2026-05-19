"""Export open-r1/DAPO-Math-17k-Processed to JSONL rows matching ``Example`` in dataset.py.

Usage
-----
English + Chinese (default config):

    python -m nanoopd.data.dapo -o datasets/dapo_math.jsonl

English subset only:

    python -m nanoopd.data.dapo -o dapo_math_en.jsonl --config en
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

from datasets import load_dataset

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer in \\boxed{}."
)

def extract_last_boxed(text: str) -> str | None:
    """Return the content of the last brace-balanced \\boxed{...} in text, or None."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None

def export_dapo_math(
    output: Path,
    dataset_id: str = "open-r1/DAPO-Math-17k-Processed",
    config_name: str = "all",
    split: str = "train",
) -> int:
    logger.info(f"{dataset_id=}, {config_name=}, {split=}")
    ds = load_dataset(dataset_id, config_name, split=split)

    records: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        row: Mapping[str, Any]
        prompt_text = (row.get("prompt") or "").strip()
        reward_model = row.get("reward_model") or {}
        top_extra = row.get("extra_info") or {}
        nested_extra = reward_model.get("extra_info") or {}
        raw_id = top_extra.get("index") or nested_extra.get("index")
        row_label = f"dapo_math/{raw_id}" if raw_id and str(raw_id).strip() else f"dapo_math/row_{i}"

        answer = ((row.get("solution") or "").strip()
                  or (reward_model.get("ground_truth") or "").strip())

        if not prompt_text or not answer:
            logger.warning(f"Skipping row {i} ({row_label}): empty prompt or answer")
            continue

        records.append({
            "dataset": "dapo",
            "kind": "math",
            "description": prompt_text,
            "system": DEFAULT_SYSTEM_PROMPT,
            "prompt": prompt_text,
            "answer": answer,
            "tests": None,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"{output=}, num_lines={len(records)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export open-r1/DAPO-Math-17k-Processed to JSONL (Example schema).",
    )
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output .jsonl path.")
    parser.add_argument("--dataset-id", default="open-r1/DAPO-Math-17k-Processed", help="HF dataset id.")
    parser.add_argument("--config", default="all", help="Dataset config (default: all).")
    parser.add_argument("--split", default="train", help="Split name (default: train).")

    args = parser.parse_args(argv)
    return export_dapo_math(args.output, args.dataset_id, args.config, args.split)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))