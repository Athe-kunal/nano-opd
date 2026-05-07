import datasets
import pathlib
from datasets import Dataset, load_dataset
from typing import Optional, List, Dict

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
        # dummy fields
        "tests": None,
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

    return ds.map(format, remove_columns=ds.column_names)




def split_tasks(json_path: str, output_dir: str, test_ratio: float = 0.1, seed: int = 42):
    ds = datasets.load_dataset("json", data_files=json_path, split="train")
    split_ds = ds.train_test_split(test_size=test_ratio, seed=seed)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_ds["train"].to_json(output_dir / "train.json")
    split_ds["test"].to_json(output_dir / "test.json")

    print(f"Train: {len(split_ds['train'])} samples → {output_dir / 'train.json'}")
    print(f"Test:  {len(split_ds['test'])} samples  → {output_dir / 'test.json'}")

if __name__ == '__main__':
    from datasets import load_dataset
    import pathlib

    # ── Step 1: Load each domain ──────────────────────────────────────────────────

    def load_sciknoweval(domains=None, levels=None, types=None):
        ds = load_dataset("hicai-zju/SciKnowEval", split="test")
        if domains:
            ds = ds.filter(lambda x: x["domain"] in domains)
        if levels:
            ds = ds.filter(lambda x: x["details"]["level"] in levels)
        if types:
            ds = ds.filter(lambda x: x["type"] in types)
        return ds.map(format, remove_columns=ds.column_names)


    domains = ["Biology", "Chemistry", "Material", "Physics"]

    for domain in domains:
        ds = load_sciknoweval(
            domains=[domain],
            levels=["L3"],
            types=["mcq-4-choices", "mcq-2-choices"],
        )
        out_path = f"datasets/sciknoweval/{domain.lower()}/{domain.lower()}.json"
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_json(out_path)
        print(f"{domain}: {len(ds)} samples saved to {out_path}")


    # ── Step 2: Split each domain into train / test ───────────────────────────────

    for domain in domains:
        json_path = f"datasets/sciknoweval/{domain.lower()}/{domain.lower()}.json"
        output_dir = f"datasets/sciknoweval/{domain.lower()}"
        split_tasks(json_path=json_path, output_dir=output_dir, test_ratio=0.1, seed=42)