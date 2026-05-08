from __future__ import annotations
import os

import json
import random
from dataclasses import dataclass
from typing import Iterator, Callable


@dataclass
class Example:
    prompt: str
    kind: str          # "mcq" | "code"
    dataset: str
    description: str
    system: str | None = None   # system prompt; present for sciknoweval
    answer: str | None = None   # ground-truth letter; present for sciknoweval MCQ
    tests: str | None = None    # JSON-encoded test cases; present for lcb_v6


def _adapt_sciknoweval(row: dict) -> Example:
    return Example(
        prompt=row["prompt"],
        kind=row["kind"],
        dataset=row["dataset"],
        description=row["description"],
        system=row.get("system"),
        answer=row.get("answer"),
        tests=row.get("tests"),
    )


def _adapt_lcb(row: dict) -> Example:
    return Example(
        prompt=row["prompt"],
        kind=row["kind"],
        dataset=row["dataset"],
        description=row["description"],
        system=None,
        answer=None,
        tests=row.get("tests"),
    )


_ADAPTERS: dict[str, Callable[[dict], Example]] = {
    "sciknoweval": _adapt_sciknoweval,
    "livecodebench": _adapt_lcb,
}


class JSONLOPDDataset:
    """Loads a JSONL file and exposes Example items."""

    def __init__(self, path: str, adapter: str | None = None):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

        if adapter is None:
            adapter = rows[0].get("dataset", "")

        adapt_fn = _ADAPTERS.get(adapter)
        if adapt_fn is None:
            raise ValueError(f"Unknown adapter '{adapter}'. Known: {list(_ADAPTERS)}")

        self._examples: list[Example] = [adapt_fn(r) for r in rows]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]


class _IndexedOPDDataset:
    """Wraps an existing list of Examples as an indexable dataset."""

    def __init__(self, examples: list[Example]):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]


def distributed_opd_loader(
    dataset: JSONLOPDDataset | _IndexedOPDDataset,
    prompts_per_step: int,
    world_size: int,
    rank: int,
    seed: int = 0,
    resume_state: dict | None = None,
) -> Iterator[tuple[list[Example], dict]]:
    """Yield (list[Example], state_dict) per step."""
    n = len(dataset)
    assert prompts_per_step % world_size == 0
    assert prompts_per_step <= n
    per_rank = prompts_per_step // world_size

    def _epoch_order(epoch_idx: int) -> list[int]:
        rng = random.Random(seed * 1_000_003 + epoch_idx)
        order = list(range(n))
        rng.shuffle(order)
        return order

    if resume_state is not None:
        epoch = resume_state["epoch"]
        cursor = resume_state["cursor"]
    else:
        epoch = 0
        cursor = 0
    order = _epoch_order(epoch)

    while True:
        if cursor + prompts_per_step > n:
            epoch += 1
            cursor = 0
            order = _epoch_order(epoch)
        step_idx = order[cursor:cursor + prompts_per_step]
        rank_idx = step_idx[rank * per_rank:(rank + 1) * per_rank]
        examples = [dataset[i] for i in rank_idx]
        cursor += prompts_per_step
        yield examples, {"epoch": epoch, "cursor": cursor}

OPD_DATASET_PATH = os.environ.get("OPD_DATASET_PATH", "/local-ssd/mh3897/data/rl/dapo_math_17k.jsonl")


def build_opd_dataset() -> JSONLOPDDataset:
    if not os.path.exists(OPD_DATASET_PATH):
        raise FileNotFoundError(f"OPD dataset not found on disk: {OPD_DATASET_PATH}")
    return JSONLOPDDataset(OPD_DATASET_PATH)
