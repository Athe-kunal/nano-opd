import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

import wandb
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DATASETS = {
    "aime24": {
        "hf_path": "HuggingFaceH4/aime_2024",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "answer",
        "id_col": "id",
    },
    "aime25": {
        "hf_path": "MathArena/aime_2025",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "answer",
        "id_col": "problem_idx",
    },
    "hmmt25": {
        "hf_path": "MathArena/hmmt_feb_2025",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "answer",
        "id_col": "problem_idx",
        "trust_remote_code": False,
    },
}


def extract_boxed_answer(text: str) -> str:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None

    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def grade_answer(predicted: str, ground_truth: str) -> bool:
    """Checks whether `predicted` is mathematically equivalent to `ground_truth`.

    Args:
        predicted: Model-extracted boxed answer text, or None if not found.
        ground_truth: The reference answer.

    Returns:
        True if `math_verify` confirms equivalence, or (as a fallback) if the
        normalized strings match exactly.
    """
    if predicted is None:
        return False
    try:
        if "$" not in predicted:
            predicted = f"${predicted}$"
        if "$" not in ground_truth:
            ground_truth = f"${ground_truth}$"
        pred_parsed = parse(predicted, fallback_mode="no_fallback")
        gt_parsed = parse(ground_truth, fallback_mode="no_fallback")
        return verify(gt_parsed, pred_parsed, timeout_seconds=5)
    except Exception:
        # math_verify's LaTeX/sympy parsing can raise many different
        # exception types on malformed model output (and can time out); any
        # failure here falls back to plain normalized string comparison
        # rather than treating the answer as ungraded.
        pred_norm = predicted.replace("$", "").replace(" ", "").lower().strip()
        gt_norm = ground_truth.replace("$", "").replace(" ", "").lower().strip()
        return pred_norm == gt_norm


def load_vllm_model(
    base_model_path: str,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int = None,
    enable_thinking: bool = True,
):
    """Loads a model into vLLM, enabling LoRA if adapter weights are found.

    Args:
        base_model_path: HuggingFace model id or local path to the base model.
        lora_adapter_path: Directory to check for LoRA adapter weights
          (`adapter_model.safetensors` or `.bin`); None disables the check.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        tensor_parallel_size: vLLM's internal tensor-parallel degree.
        max_model_len: Max context length; auto-set from `enable_thinking`
          if None.
        enable_thinking: Whether the model's chat template should enable
          thinking mode; also affects the auto-computed `max_model_len`.

    Returns:
        `(llm, tokenizer)`: the vLLM `LLM` engine and matching tokenizer.
    """
    print(f"Loading model with vLLM from: {base_model_path}")

    if max_model_len is None:
        max_model_len = 40960 if enable_thinking else 32768
        print(f"Auto-setting max_model_len to {max_model_len}")

    llm_config = {
        "model": base_model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "distributed_executor_backend": "mp",
        "enforce_eager": False,
    }

    if lora_adapter_path is not None:
        adapter_path = Path(lora_adapter_path) / "adapter_model.safetensors"
        if not adapter_path.exists():
            adapter_path = Path(lora_adapter_path) / "adapter_model.bin"

        if adapter_path.exists():
            print("LoRA weights found. Enabling LoRA support...")
            llm_config["enable_lora"] = True
            llm_config["max_lora_rank"] = 64
            llm_config["max_loras"] = 1
            llm_config["max_cpu_loras"] = 1
        else:
            print(f"Warning: No LoRA weights found at {lora_adapter_path}. Using base model only.")

    llm = LLM(**llm_config)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"Model dtype: {llm.llm_engine.model_config.dtype}")
    print(f"Quantization: {llm.llm_engine.model_config.quantization}")
    cache_config = getattr(llm.llm_engine, "cache_config", None) or getattr(getattr(llm.llm_engine, "vllm_config", None), "cache_config", None)
    print(f"KV cache dtype: {cache_config.cache_dtype if cache_config else 'unknown'}")
    print("vLLM model loaded successfully!")
    return llm, tokenizer


async def prepare_dataset(
    dataset_name: str,
    tokenizer,
    num_samples: int = None,
    enable_thinking: bool = True,
) -> dict:
    """Loads a dataset and builds chat-templated prompts (I/O-bound, runs concurrently via asyncio.gather).

    Args:
        dataset_name: Key into `DATASETS`.
        tokenizer: Tokenizer used to render the chat template.
        num_samples: If set, cap the dataset to this many problems.
        enable_thinking: Passed to the tokenizer's chat template.

    Returns:
        A dict with `dataset_name`, `prompts`, `gt_answers`, `problems`,
        `question_ids`, and `num_problems`.
    """
    cfg = DATASETS[dataset_name]
    print(f"Loading {dataset_name} from {cfg['hf_path']}...")

    dataset = await asyncio.to_thread(
        load_dataset, cfg["hf_path"], split=cfg["split"],
        trust_remote_code=cfg.get("trust_remote_code", True)
    )

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"  {dataset_name}: {len(dataset)} problems")

    prompts, gt_answers, problems, question_ids = [], [], [], []

    for example in dataset:
        problem = example[cfg["problem_col"]]
        gt_answer = str(example[cfg["answer_col"]])
        question_id = example.get(cfg["id_col"], None)

        user_message = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        messages = [{"role": "user", "content": user_message}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        prompts.append(text)
        gt_answers.append(gt_answer)
        problems.append(problem)
        question_ids.append(question_id)

    return {
        "dataset_name": dataset_name,
        "prompts": prompts,
        "gt_answers": gt_answers,
        "problems": problems,
        "question_ids": question_ids,
        "num_problems": len(dataset),
    }


async def _grade_single_problem(
    idx: int,
    output: Any,
    problem: str,
    gt_answer: str,
    question_id: Any,
    val_n: int,
) -> dict:
    """Grades every sampled generation for one problem and summarizes them.

    Args:
        idx: Problem's position in the dataset, used as a fallback id.
        output: vLLM `RequestOutput` holding `val_n` sampled generations.
        problem: The problem text.
        gt_answer: Ground-truth answer string.
        question_id: Dataset-provided problem id, if any.
        val_n: Number of sampled generations per problem.

    Returns:
        A dict with per-generation grading detail plus this problem's
        pass@n / majority-vote outcome, matching one entry of
        `process_results`'s `results` list.
    """
    generations, predicted_answers, is_correct_list, is_formatted_list = [], [], [], []

    for out in output.outputs:
        predicted_answer = extract_boxed_answer(out.text)
        is_correct = await asyncio.to_thread(grade_answer, predicted_answer, gt_answer)
        is_formatted = predicted_answer is not None

        generations.append(out.text)
        predicted_answers.append(predicted_answer if predicted_answer else "[No boxed answer found]")
        is_correct_list.append(is_correct)
        is_formatted_list.append(is_formatted)

    num_formatted = sum(is_formatted_list)
    has_correct = any(is_correct_list)

    majority_vote_correct = False
    if num_formatted > 0:
        formatted_predictions = [p for p, f in zip(predicted_answers, is_formatted_list) if f]
        most_common = Counter(formatted_predictions).most_common(1)[0][0]
        majority_vote_correct = await asyncio.to_thread(grade_answer, most_common, gt_answer)

    return {
        "problem_id": question_id if question_id is not None else idx,
        "problem": problem,
        "ground_truth": gt_answer,
        "val_n": val_n,
        "generations": [
            {"predicted_answer": pred, "full_generation": gen, "correct": corr, "formatted": fmt}
            for pred, gen, corr, fmt in zip(predicted_answers, generations, is_correct_list, is_formatted_list)
        ],
        "num_correct": sum(is_correct_list),
        "pass_at_n": has_correct,
        "majority_vote_correct": majority_vote_correct,
        "predicted_answer": predicted_answers[0],
        "full_generation": generations[0],
        "correct": is_correct_list[0],
        "formatted": is_formatted_list[0],
    }


def _summarize_and_log_results(
    dataset_name: str,
    results: list[dict],
    val_n: int,
    num_problems: int,
    eval_config: dict,
    output_file: str | None,
    step: int | None,
) -> dict:
    """Aggregates per-problem results into a summary, prints it, and logs/saves it.

    Args:
        dataset_name: Name of the dataset these results belong to.
        results: Per-problem grading dicts from `_grade_single_problem`.
        val_n: Number of sampled generations per problem.
        num_problems: Number of problems graded.
        eval_config: Generation/eval hyperparameters to record in the summary
          (temperature, top_p, top_k, min_p, presence_penalty, max_new_tokens,
          enable_thinking, base_model).
        output_file: If set, write the summary JSON here.
        step: Training step for the W&B x-axis, if logging to an active run.

    Returns:
        The summary dict (also written to `output_file` and logged to W&B).
    """
    total = num_problems * val_n
    formatted_count = sum(g["formatted"] for r in results for g in r["generations"])
    total_correct_per_problem = sum(r["num_correct"] for r in results)
    pass_at_n = sum(1 for r in results if r["pass_at_n"])
    majority_vote_correct_count = sum(1 for r in results if r["majority_vote_correct"])

    format_rate = formatted_count / total * 100
    pass_at_n_pct = pass_at_n / num_problems * 100
    average_at_n_pct = total_correct_per_problem / total * 100
    majority_vote_at_n_pct = majority_vote_correct_count / num_problems * 100

    print(f"\n{'='*70}")
    print(f"RESULTS: {dataset_name.upper()}")
    print(f"  Pass@{val_n}:          {pass_at_n_pct:.2f}% ({pass_at_n}/{num_problems})")
    print(f"  Average@{val_n}:       {average_at_n_pct:.2f}% ({total_correct_per_problem}/{total})")
    print(f"  Majority Vote@{val_n}: {majority_vote_at_n_pct:.2f}% ({majority_vote_correct_count}/{num_problems})")
    print(f"  Format rate:     {format_rate:.2f}%")
    print(f"{'='*70}")

    if wandb.run is not None:
        log_data = {
            f"{dataset_name}/pass_at_{val_n}": pass_at_n_pct,
            f"{dataset_name}/average_at_{val_n}": average_at_n_pct,
            f"{dataset_name}/majority_vote_at_{val_n}": majority_vote_at_n_pct,
            f"{dataset_name}/format_rate": format_rate,
        }
        wandb.log(log_data, step=step)

    summary = {
        **eval_config,
        "dataset": dataset_name,
        "val_n": val_n,
        "num_problems": num_problems,
        "total_solutions": total,
        "pass_at_n": pass_at_n,
        "pass_at_n_pct": pass_at_n_pct,
        "average_at_n": total_correct_per_problem,
        "average_at_n_pct": average_at_n_pct,
        "majority_vote_at_n": majority_vote_correct_count,
        "majority_vote_at_n_pct": majority_vote_at_n_pct,
        "formatted_count": formatted_count,
        "format_rate": format_rate,
        "results": results,
    }

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")

    return summary


async def process_results(
    dataset_info: dict,
    outputs: list,
    val_n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    enable_thinking: bool,
    base_model_name: str,
    output_file: str = None,
    step: int = None,
) -> dict:
    """Grades every problem's outputs and returns the aggregated summary.

    Args:
        dataset_info: Dict from `prepare_dataset` (problems, gt_answers, etc).
        outputs: vLLM `RequestOutput` list, one per problem, `val_n` samples each.
        val_n: Number of sampled generations per problem.
        max_new_tokens: Max generation length, recorded in the summary.
        temperature: Sampling temperature, recorded in the summary.
        top_p: Sampling top-p, recorded in the summary.
        top_k: Sampling top-k, recorded in the summary.
        min_p: Sampling min-p, recorded in the summary.
        presence_penalty: Sampling presence penalty, recorded in the summary.
        enable_thinking: Whether thinking mode was enabled, recorded in the summary.
        base_model_name: Model name, recorded in the summary.
        output_file: If set, write the summary JSON here.
        step: Training step for the W&B x-axis, if logging to an active run.

    Returns:
        The aggregated summary dict for this dataset (CPU-bound grading runs
        concurrently with other datasets via the caller's `asyncio.gather`).
    """
    dataset_name = dataset_info["dataset_name"]
    problems = dataset_info["problems"]
    gt_answers = dataset_info["gt_answers"]
    question_ids = dataset_info["question_ids"]
    num_problems = dataset_info["num_problems"]

    print(f"\nProcessing results for {dataset_name}...")

    results = [
        await _grade_single_problem(idx, output, problem, gt_answer, question_id, val_n)
        for idx, (output, problem, gt_answer, question_id) in enumerate(
            zip(outputs, problems, gt_answers, question_ids)
        )
    ]

    eval_config = {
        "base_model": base_model_name,
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "max_new_tokens": max_new_tokens,
    }
    return _summarize_and_log_results(
        dataset_name, results, val_n, num_problems, eval_config, output_file, step
    )


def build_output_path(base_model, checkpoint_dir, dataset_name, enable_thinking, temperature, val_n):
    parts = ["eval_results", dataset_name, Path(base_model).name]
    if checkpoint_dir:
        cp = Path(checkpoint_dir)
        parts += [cp.parent.name, cp.name]
    parts += [
        "thinking" if enable_thinking else "nonthinking",
        f"temp{temperature}",
        f"valn{val_n}",
    ]
    return str(Path("eval_results") / ("_".join(parts) + ".json"))


async def run_evaluation(args, llm, tokenizer, lora_request, step: int = None):
    print(f"\n{'='*70}")
    print("PHASE 1: Loading all datasets concurrently...")
    print(f"{'='*70}")
    dataset_infos = await asyncio.gather(*[
        prepare_dataset(name, tokenizer, args.num_samples, args.enable_thinking)
        for name in args.datasets
    ])

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_new_tokens,
        presence_penalty=args.presence_penalty,
        n=args.val_n,
    )

    all_prompts = [p for info in dataset_infos for p in info["prompts"]]
    dataset_sizes = [len(info["prompts"]) for info in dataset_infos]

    print(f"\n{'='*70}")
    print(f"PHASE 2: Single vLLM generate call — {len(all_prompts)} total prompts")
    print(f"  " + ", ".join(f"{info['dataset_name']}: {n}" for info, n in zip(dataset_infos, dataset_sizes)))
    print(f"  LoRA: {lora_request is not None}")
    print(f"{'='*70}\n")

    if lora_request is not None:
        if lora_request.lora_path is None:
            raise ValueError("LoRA request has no path; may be a zero3+peft issue — try zero2")
        all_outputs = await asyncio.to_thread(
            llm.generate, all_prompts, sampling_params, lora_request=lora_request, use_tqdm=True
        )
    else:
        all_outputs = await asyncio.to_thread(
            llm.generate, all_prompts, sampling_params, use_tqdm=True
        )

    # Split outputs back per dataset
    split_outputs = []
    offset = 0
    for size in dataset_sizes:
        split_outputs.append(all_outputs[offset : offset + size])
        offset += size

    print(f"\n{'='*70}")
    print("PHASE 3: Grading results for all datasets concurrently...")
    print(f"{'='*70}")

    output_files = [
        build_output_path(
            args.base_model, args.checkpoint_dir, info["dataset_name"],
            args.enable_thinking, args.temperature, args.val_n
        )
        for info in dataset_infos
    ]

    summaries = await asyncio.gather(*[
        process_results(
            dataset_info=info,
            outputs=outputs,
            val_n=args.val_n,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            enable_thinking=args.enable_thinking,
            base_model_name=args.base_model,
            output_file=out_file,
            step=step,
        )
        for info, outputs, out_file in zip(dataset_infos, split_outputs, output_files)
    ])

    return {s["dataset"]: s for s in summaries}


def _apply_arg_defaults(args: argparse.Namespace) -> None:
    """Applies smoke-test overrides and auto-computed defaults to parsed args, in place.

    Args:
        args: Parsed CLI args, mutated in place.

    Raises:
        SystemExit: If `--checkpoint_dir` is set but doesn't exist.
    """
    if args.smoke_test:
        args.num_samples = 1
        args.val_n = 4
        print("SMOKE TEST MODE: 1 sample per dataset, val_n=1")

    if args.checkpoint_dir is not None and not Path(args.checkpoint_dir).exists():
        print(f"ERROR: Checkpoint directory does not exist: {args.checkpoint_dir}")
        exit(1)

    if args.top_p is None:
        args.top_p = 0.95 if args.enable_thinking else 0.8
        print(f"Auto-setting top_p to {args.top_p}")

    if args.enable_thinking and args.temperature == 0.0:
        print("WARNING: greedy decoding in thinking mode may cause repetitions; Qwen3 recommends temp=0.6")


def _build_lora_request(checkpoint_dir: str | None) -> Any:
    """Builds a vLLM `LoRARequest` if `checkpoint_dir` holds LoRA adapter weights.

    Args:
        checkpoint_dir: Path to a checkpoint directory, or None to skip LoRA.

    Returns:
        A `LoRARequest`, or None if no checkpoint dir was given, no adapter
        weights were found there, or `vllm.lora` could not be imported.
    """
    if checkpoint_dir is None:
        return None
    try:
        from vllm.lora.request import LoRARequest

        adapter_safetensors = Path(checkpoint_dir) / "adapter_model.safetensors"
        adapter_bin = Path(checkpoint_dir) / "adapter_model.bin"

        if adapter_safetensors.exists() or adapter_bin.exists():
            print(f"✓ LoRA request created for: {checkpoint_dir}")
            return LoRARequest("checkpoint_lora", 1, checkpoint_dir)
        print(f"Warning: No LoRA weights found at {checkpoint_dir}. Using base model only.")
    except ImportError:
        print("Warning: Could not import LoRARequest. Running without LoRA.")
    except Exception as e:
        print(f"Warning: Could not create LoRA request: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on AIME 2024, AIME 2025, and HMMT 2025")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/infra/old-home/home/siyanzhao/models/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aime24", "aime25", "hmmt25"],
        choices=list(DATASETS.keys()),
    )
    parser.add_argument("--max_new_tokens", type=int, default=38912)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true", default=False, help="Run with 1 sample per dataset to verify the pipeline end-to-end")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--val_n", type=int, default=6)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--step", type=int, default=None, help="Training step for x-axis in W&B plots")

    args = parser.parse_args()
    _apply_arg_defaults(args)

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    llm, tokenizer = load_vllm_model(
        args.base_model,
        args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )

    lora_request = _build_lora_request(args.checkpoint_dir)

    all_summaries = asyncio.run(run_evaluation(args, llm, tokenizer, lora_request, step=args.step))

    print(f"\n{'='*70}")
    print("ALL EVALUATIONS COMPLETE")
    print(f"{'='*70}")
    for name, s in all_summaries.items():
        n = s["val_n"]
        print(
            f"  {name.upper():<8} Pass@{n}: {s['pass_at_n_pct']:.1f}% | "
            f"Avg@{n}: {s['average_at_n_pct']:.1f}% | "
            f"MajVote@{n}: {s['majority_vote_at_n_pct']:.1f}%"
        )
    print(f"{'='*70}")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
