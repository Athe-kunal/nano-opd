from __future__ import annotations

import random as _random
from typing import Iterator, Literal

from opd.envs.base import OPDEnvBase
from opd.envs.dapo_dataset import DapoMathEnv
from opd.envs.livecodebench import LiveCodeBenchEnv
from opd.envs.sciknoweval import SciKnowEvalEnv

DatasetType = Literal["livecodebench", "sciknoweval", "dapo_math"]


def build_opd_dataset(dataset_type: DatasetType, eval_test_size: float = 0.1, seed: int = 42) -> list[OPDEnvBase]:
    envs: list[OPDEnvBase]
    if dataset_type == "livecodebench":
        envs = LiveCodeBenchEnv.load(dataset_split="train")  # type: ignore[assignment]
    elif dataset_type == "sciknoweval":
        envs = SciKnowEvalEnv.load(test_size=eval_test_size, seed=seed)  # type: ignore[assignment]
    elif dataset_type == "dapo_math":
        envs = DapoMathEnv.load()  # type: ignore[assignment]
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type!r}")
    return envs


def distributed_opd_loader(
    dataset: list[OPDEnvBase],
    prompts_per_step: int,
    world_size: int,
    rank: int,
    seed: int = 0,
    resume_state: dict | None = None,
) -> Iterator[tuple[list[OPDEnvBase], dict]]:
    """Yield (list[OPDEnvBase], state_dict) per step."""
    n = len(dataset)
    assert prompts_per_step % world_size == 0
    assert prompts_per_step <= n
    per_rank = prompts_per_step // world_size

    def _epoch_order(epoch_idx: int) -> list[int]:
        rng = _random.Random(seed * 1_000_003 + epoch_idx)
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
        envs = [dataset[i] for i in rank_idx]
        cursor += prompts_per_step
        yield envs, {"epoch": epoch, "cursor": cursor}
