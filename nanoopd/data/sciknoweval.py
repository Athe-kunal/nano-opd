import pathlib
from dataclasses import replace
from datasets import Dataset, load_dataset
from typing import Optional, List, Dict, Sequence

from nanoopd.data.base import FeedBackExample, SelfDistillationDatasetbase

SYSTEM_PROMPT = """
Given a question and four options, please select the right answer. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

For the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else. Do not restate the answer text. For example, if the answer is "A", just output:
<answer>
A
</answer>
"""


def format_choices(choices: Dict[str, str]) -> str:
    texts, labels = choices['text'], choices['label']
    return "\n".join([f"{label}: {text}" for label, text in zip(labels, texts)])


def format(row: dict) -> dict:
    return {
        "dataset": "sciknoweval",
        "system": SYSTEM_PROMPT,
        "prompt": row['question'] + "\n\n" + format_choices(row['choices']) + "\nPlease reason step by step.",
        "answer": row['answerKey'],
        "tests": {"answerKey": row['answerKey']},
        "description": row['question'],
        "kind": "mcq",
        "elo": 1500,
    }


def load_sciknoweval(
    domains: Optional[List[str]] = None,
    levels: Optional[List[str]] = None,
    types: Optional[List[str]] = None,
) -> Dataset:
    ds = load_dataset("hicai-zju/SciKnowEval", split='test')

    if domains:
        ds = ds.filter(lambda x: x['domain'] in domains)
    if levels:
        ds = ds.filter(lambda x: x['details']['level'] in levels)
    if types:
        ds = ds.filter(lambda x: x['type'] in types)

    ds = ds.map(format, remove_columns=ds.column_names, load_from_cache_file=False)

    # The raw dataset generates multiple MCQ variants from the same question stem.
    # Deduplicate by description so each underlying question appears only once.
    seen = set()
    def _is_unique(row):
        if row["description"] in seen:
            return False
        seen.add(row["description"])
        return True

    return ds.filter(_is_unique)





class SciKnowEvalSelfDistillationDataset(SelfDistillationDatasetbase):

    def preprocess_dataset(self, test_size: float = 0.1, seed: int = 42) -> tuple[list, list]:
        from nanoopd.data.base import InputExample
        def _adapt(r): return InputExample(prompt=r["prompt"], kind=r["kind"], dataset=r["dataset"], description=r["description"], system=r.get("system"), metadata=r.get("tests"))
        split = load_sciknoweval().train_test_split(test_size=test_size, seed=seed)
        train = [_adapt(dict(r)) for r in split["train"]]
        test = [_adapt(dict(r)) for r in split["test"]]
        return train, test

    def get_feedback(self, result: Sequence[FeedBackExample]) -> list[FeedBackExample]:
        updated = []
        for ex in result:
            if ex.metadata is None:
                updated.append(replace(ex, feedback="❌ Missing metadata"))
                continue
            answer_key = ex.metadata.get("answerKey", "")
            updated.append(replace(ex, feedback=f"The answer is {answer_key}"))
        return updated


def split_tasks(ds: Dataset, output_dir: str, test_ratio: float = 0.1, seed: int = 42):
    split_ds = ds.train_test_split(test_size=test_ratio, seed=seed)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_ds["train"].to_json(output_dir / "train.json")
    split_ds["test"].to_json(output_dir / "test.json")

    print(f"Train: {len(split_ds['train'])} samples → {output_dir / 'train.json'}")
    print(f"Test:  {len(split_ds['test'])} samples  → {output_dir / 'test.json'}")

if __name__ == '__main__':
    domains = ["Biology", "Chemistry", "Material", "Physics"]

    for domain in domains:
        ds = load_sciknoweval(
            domains=[domain],
            levels=["L3"],
            types=["mcq-4-choices", "mcq-2-choices"],
        )
        output_dir = f"datasets/sciknoweval/{domain.lower()}"
        split_tasks(ds, output_dir=output_dir, test_ratio=0.1, seed=42)
        print(f"{domain}: {len(ds)} samples")