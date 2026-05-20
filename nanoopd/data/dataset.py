from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Iterator, Callable, Literal

from nanoopd.data.base import InputExample, OPDDatasetbase
from nanoopd.data.livecodebench import load_livecodebench
from nanoopd.data.sciknoweval import load_sciknoweval
from nanoopd.data.dapo_dataset import load_dapo_math

DatasetType = Literal["livecodebench", "sciknoweval", "dapo_math"]


def _adapt_row(row: dict) -> InputExample:
    return InputExample(
        prompt=row["prompt"],
        kind=row["kind"],
        dataset=row["dataset"],
        description=row["description"],
        system=row.get("system"),
        metadata=row.get("tests"),
    )


_ADAPTERS: dict[str, Callable[[dict], InputExample]] = {
    "sciknoweval": _adapt_row,
    "livecodebench": _adapt_row,
    "dapo": _adapt_row,
}


class LiveCodeBenchDataset(OPDDatasetbase):
    def save_dataset(self, hf_name: str, path: str) -> None:
        ds = load_livecodebench(dataset_split="train")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_json(path)


class SciKnowEvalDataset(OPDDatasetbase):
    def save_dataset(self, hf_name: str, path: str) -> None:
        ds = load_sciknoweval()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_json(path)


class DapoMathDataset(OPDDatasetbase):
    def save_dataset(self, hf_name: str, path: str) -> None:
        export_dapo_math(output=Path(path), dataset_id=hf_name)


class JSONLOPDDataset:
    """Loads a JSONL file and exposes InputExample items."""

    def __init__(self, path: str, adapter: str | None = None):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

        if adapter is None:
            adapter = rows[0].get("dataset", "")

        adapt_fn = _ADAPTERS.get(adapter)
        if adapt_fn is None:
            raise ValueError(f"Unknown adapter '{adapter}'. Known: {list(_ADAPTERS)}")

        self._examples: list[InputExample] = [adapt_fn(r) for r in rows]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> InputExample:
        return self._examples[idx]


class _IndexedOPDDataset:
    """Wraps an existing list of InputExamples as an indexable dataset."""

    def __init__(self, examples: list[InputExample]):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> InputExample:
        return self._examples[idx]


def distributed_opd_loader(
    dataset: JSONLOPDDataset | _IndexedOPDDataset,
    prompts_per_step: int,
    world_size: int,
    rank: int,
    seed: int = 0,
    resume_state: dict | None = None,
) -> Iterator[tuple[list[InputExample], dict]]:
    """Yield (list[InputExample], state_dict) per step."""
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


def build_opd_dataset(dataset_type: DatasetType) -> _IndexedOPDDataset:
    if dataset_type == "livecodebench":
        rows = [dict(r) for r in load_livecodebench(dataset_split="train")]
    elif dataset_type == "sciknoweval":
        rows = [dict(r) for r in load_sciknoweval()]
    elif dataset_type == "dapo_math":
        rows = load_dapo_math()
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type!r}")
    return _IndexedOPDDataset([_adapt_row(r) for r in rows])
