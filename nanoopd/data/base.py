import abc
from dataclasses import dataclass

from typing import Any, Sequence

@dataclass
class InputExample:
    prompt: str
    kind: str          # "mcq" | "code" | "math"
    dataset: str
    description: str
    system: str | None = None   # system prompt; present for sciknoweval
    metadata: dict[str, Any] | None = None    # JSON-encoded test cases; present for lcb_v6

@dataclass
class OPDOutputExample(InputExample):
    answer: str | None = None   # ground-truth letter; present for sciknoweval MCQ

@dataclass
class FeedBackExample(InputExample):
    answer: str | None = None   # ground-truth letter; present for sciknoweval MCQ
    feedback: str | None = None

class OPDDatasetbase(abc.ABC):
    @abc.abstractmethod
    def preprocess_dataset(self, test_size: float = 0.1, seed: int = 42) -> tuple[list[InputExample], list[InputExample]]:
        """Return (train_examples, test_examples)."""
        ...

class SelfDistillationDatasetbase(abc.ABC):
    @abc.abstractmethod
    def preprocess_dataset(self, test_size: float = 0.1, seed: int = 42) -> tuple[list[InputExample], list[InputExample]]:
        """Return (train_examples, test_examples)."""
        ...

    @abc.abstractmethod
    def get_feedback(self, result: Sequence[FeedBackExample]) -> list[FeedBackExample]:...