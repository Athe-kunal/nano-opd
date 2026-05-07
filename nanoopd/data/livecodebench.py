import os, copy
import numpy as np
import datasets
from datasets import concatenate_datasets, load_dataset, Dataset
import base64, json, pickle, zlib
from datetime import datetime

LCB_TEST_CUTOFF = datetime(2025, 2, 1)
LCB_TRAIN_CUTOFF = datetime(2025, 2, 1)
TIME_LIMIT = 6
PERCENTAGE_TO_KEEP = 0.5


def _parse_signature(starter_code: str) -> str:
    return "def " + starter_code.split("def ")[1].split("Input\n")[0].strip()


def _translate_private_test_cases(encoded_data, fn_name: str):
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    tests = json.loads(original_data)
    return json.dumps({
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
        "testtype": tests[0]["testtype"],
        "fn_name": fn_name,
        "time_limit": TIME_LIMIT,
    }, ensure_ascii=False)


def load_livecodebench(dataset_split: str, until: datetime = None) -> Dataset:
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        revision="refs/pr/6"
    )

    if dataset_split == "train":
        ds = ds.filter(lambda ex: ex["contest_date"] < LCB_TRAIN_CUTOFF)
    else:
        ds = ds.filter(lambda ex: ex["contest_date"] >= LCB_TEST_CUTOFF)

    if until is not None:
        ds = ds.filter(lambda ex: ex["contest_date"] < until)

    def format_prompt(ex):
        problem = ex["question_content"]
        if ex["starter_code"].strip() != "":
            problem += f"\n\nYour solution should have the following signature: ```python\n{_parse_signature(ex['starter_code'])}\n```"

        fn_name = ""
        if ex["metadata"].strip() != "":
            metadata = json.loads(ex["metadata"])
            fn_name = metadata.get("func_name", "")

        return {
            "kind": "code",
            "dataset": "livecodebench",
            "description": problem,
            "problem": problem,
            "tests": _translate_private_test_cases(ex["private_test_cases"], fn_name=fn_name),
        }

    processed_shards = []
    for i in range(4):
        shard = ds.shard(num_shards=4, index=i)
        shard = shard.map(format_prompt, remove_columns=ds.column_names, num_proc=4)
        processed_shards.append(shard)

    return concatenate_datasets(processed_shards)



def sample_tests(example):
    """Keep 50% of tests for the train set."""
    tests = json.loads(example["tests"])
    inputs, outputs = tests["inputs"], tests["outputs"]

    num_tests = len(inputs)
    keep_count = max(1, int(num_tests * PERCENTAGE_TO_KEEP))
    keep_indices = np.sort(np.random.choice(num_tests, size=keep_count, replace=False))

    reduced_tests = copy.deepcopy(tests)
    reduced_tests["inputs"] = [inputs[i] for i in keep_indices]
    reduced_tests["outputs"] = [outputs[i] for i in keep_indices]

    example["tests"] = json.dumps(reduced_tests)
    return example


def main(data_path, output_dir):
    np.random.seed(0)
    os.makedirs(output_dir, exist_ok=True)

    ds = datasets.load_dataset("json", data_files=data_path, split="train")

    # test.json = full dataset (all tests)
    ds.to_json(os.path.join(output_dir, "test.json"))

    # train.json = reduced dataset (50% of tests per problem)
    ds_reduced = ds.map(sample_tests)
    ds_reduced.to_json(os.path.join(output_dir, "train.json"))

    print(f"Test set:  {len(ds)} problems, full tests")
    print(f"Train set: {len(ds_reduced)} problems, {PERCENTAGE_TO_KEEP:.0%} of tests kept")

if __name__ == '__main__':
    from datetime import datetime

    # 1. Load the dataset
    ds = load_livecodebench(dataset_split="test", until=datetime(2025, 5, 1))

    # 2. Save to JSON
    ds.to_json("datasets/lcb_v6.json")

    # 3. Split: test = all tests, train = 50% of tests per problem
    main(data_path="datasets/lcb_v6.json", output_dir="datasets/lcb_v6")
    # → datasets/lcb_v6/train.json
    # → datasets/lcb_v6/test.json