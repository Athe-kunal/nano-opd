"""Smoke test for dataset preprocessing and feedback."""
from dataclasses import replace

from nanoopd.data.base import FeedBackExample
from nanoopd.data.dataset import LiveCodeBenchDataset, SciKnowEvalDataset, DapoMathDataset
from nanoopd.data.livecodebench import LiveCodeSelfDistillationDataset
from nanoopd.data.sciknoweval import SciKnowEvalSelfDistillationDataset
from nanoopd.data.dapo_dataset import DapoMathSelfDistillationDataset

N_SHOW = 2  # rows to print per dataset


def show_examples(name: str, train, test):
    print(f"\n{'='*60}")
    print(f"{name}: {len(train)} train / {len(test)} test")
    print(f"{'='*60}")
    for i, ex in enumerate(train[:N_SHOW]):
        print(f"\n--- Train[{i}] ---")
        print(f"  kind     : {ex.kind}")
        print(f"  dataset  : {ex.dataset}")
        print(f"  prompt   : {ex.prompt[:120]!r}...")
        print(f"  metadata : {str(ex.metadata)[:120]}")


FAKE_CODE_RESPONSE = """\
Let me think through this step by step.

```python
def twoSum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
```
"""


def _feedback_examples(split_name: str, examples, dataset_obj):
    print(f"\n  -- {split_name} feedback --")
    ex = examples[0]
    fb_ex = FeedBackExample(
        prompt=ex.prompt, kind=ex.kind, dataset=ex.dataset,
        description=ex.description, system=ex.system, metadata=ex.metadata,
        answer=FAKE_CODE_RESPONSE,
    )
    results = dataset_obj.get_feedback([fb_ex])
    print(f"  feedback:\n{results[0].feedback}")


def test_livecodebench():
    print("\n>>> LiveCodeBench preprocess_dataset")
    train, test = LiveCodeBenchDataset().preprocess_dataset()
    show_examples("LiveCodeBench", train, test)

    sd = LiveCodeSelfDistillationDataset()
    _feedback_examples("train", train, sd)
    _feedback_examples("test", test, sd)


def test_sciknoweval():
    print("\n>>> SciKnowEval preprocess_dataset")
    train, test = SciKnowEvalDataset().preprocess_dataset(test_size=0.1, seed=42)
    show_examples("SciKnowEval", train, test)

    sd = SciKnowEvalSelfDistillationDataset()
    for split_name, examples in [("train", train), ("test", test)]:
        print(f"\n  -- {split_name} feedback --")
        ex = examples[0]
        fb_ex = FeedBackExample(
            prompt=ex.prompt, kind=ex.kind, dataset=ex.dataset,
            description=ex.description, system=ex.system, metadata=ex.metadata,
            answer="<reasoning>I think it's A</reasoning><answer>A</answer>",
        )
        results = sd.get_feedback([fb_ex])
        print(f"  feedback: {results[0].feedback}")


def test_dapo_math():
    print("\n>>> DAPO Math preprocess_dataset")
    train, test = DapoMathDataset().preprocess_dataset(test_size=0.1, seed=42)
    show_examples("DAPO Math", train, test)

    sd = DapoMathSelfDistillationDataset()
    for split_name, examples in [("train", train), ("test", test)]:
        if not examples:
            print(f"\n  -- {split_name} feedback -- (skipped: empty split)")
            continue
        print(f"\n  -- {split_name} feedback --")
        ex = examples[0]
        fb_ex = FeedBackExample(
            prompt=ex.prompt, kind=ex.kind, dataset=ex.dataset,
            description=ex.description, system=ex.system, metadata=ex.metadata,
            answer="Let me reason step by step... \\boxed{42}",
        )
        results = sd.get_feedback([fb_ex])
        print(f"  feedback: {results[0].feedback}")


if __name__ == "__main__":
    test_livecodebench()
    test_sciknoweval()
    test_dapo_math()
    print("\n\nDone.")
