"""Smoke tests for OPDEnvBase subclasses (init / step / compute_reward / get_feedback)."""

from opd.envs.dapo_dataset import DapoMathEnv
from opd.envs.livecodebench import LiveCodeBenchEnv
from opd.envs.sciknoweval import SciKnowEvalEnv

# ---------------------------------------------------------------------------
# Rich helpers (used by the vLLM live test at the bottom of this file)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    _rich_console = Console()
except ImportError:
    _rich_console = None  # type: ignore[assignment]

N_SHOW = 2

FAKE_MATH_CORRECT = r"Let me think... \boxed{42}"
FAKE_MATH_WRONG   = r"Let me think... \boxed{99}"

FAKE_CODE_CORRECT = """\
Here is my solution.
```python
def twoSum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
```
"""
FAKE_CODE_WRONG = "I don't know how to solve this."

FAKE_MCQ_CORRECT = "<reasoning>It must be A</reasoning><answer>A</answer>"
FAKE_MCQ_WRONG   = "<reasoning>I'll guess B</reasoning><answer>B</answer>"


def _show_step(env, action: str, label: str):
    env.init([])
    result = env.step(action)
    print(f"  [{label}]")
    print(f"    reward  : {result['reward']}")
    print(f"    done    : {result['done']}")
    print(f"    feedback: {result['metadata']['feedback']}")


def test_dapo_math():
    print("\n" + "=" * 60)
    print("DapoMathEnv")
    print("=" * 60)
    envs = DapoMathEnv.load()
    print(f"  loaded {len(envs)} envs")

    for i, env in enumerate(envs[:N_SHOW]):
        print(f"\n  env[{i}] prompt: {env.prompt[:80]!r}...")
        _show_step(env, FAKE_MATH_CORRECT.replace("42", env.answer), "correct answer")
        _show_step(env, FAKE_MATH_WRONG, "wrong answer")


def test_livecodebench():
    print("\n" + "=" * 60)
    print("LiveCodeBenchEnv")
    print("=" * 60)
    envs = LiveCodeBenchEnv.load(dataset_split="train")
    print(f"  loaded {len(envs)} envs")

    for i, env in enumerate(envs[:N_SHOW]):
        print(f"\n  env[{i}] prompt: {env.prompt[:80]!r}...")
        _show_step(env, FAKE_CODE_CORRECT, "fake correct code")
        _show_step(env, FAKE_CODE_WRONG, "no code block")


def test_sciknoweval():
    print("\n" + "=" * 60)
    print("SciKnowEvalEnv")
    print("=" * 60)
    # load train split only (exclude held-out 10% test used by evaluate())
    envs = SciKnowEvalEnv.load(test_size=0.1, seed=42)
    print(f"  loaded {len(envs)} train envs")

    for i, env in enumerate(envs[:N_SHOW]):
        print(f"\n  env[{i}] prompt : {env.prompt[:80]!r}...")
        print(f"           answer : {env.answer_key}")
        correct_action = f"<reasoning>must be {env.answer_key}</reasoning><answer>{env.answer_key}</answer>"
        _show_step(env, correct_action, "correct answer")
        wrong_key = "B" if env.answer_key != "B" else "A"
        wrong_action = f"<reasoning>guessing</reasoning><answer>{wrong_key}</answer>"
        _show_step(env, wrong_action, "wrong answer")


def _render_trace(
    console: "Console",
    env_name: str,
    idx: int,
    prompt: str,
    completion: str,
    reward: float,
    feedback: str,
) -> None:
    """Pretty-print a single prompt→completion→reward trace with Rich."""
    color = "green" if reward > 0 else "red"
    reward_badge = Text(f"  reward={reward:.2f}  ", style=f"bold white on {color}")

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("field", style="bold cyan", no_wrap=True)
    table.add_column("value", overflow="fold")
    table.add_row("prompt", prompt[:300] + ("…" if len(prompt) > 300 else ""))
    table.add_row("completion", completion[:600] + ("…" if len(completion) > 600 else ""))
    table.add_row("feedback", feedback)

    title = Text()
    title.append(f"{env_name} ", style="bold yellow")
    title.append(f"[{idx}] ")
    title.append_text(reward_badge)

    console.print(Panel(table, title=title, border_style="dim"))


def test_vllm_server(
    base_url: str = "http://localhost:8000",
    model: str | None = None,
    n_prompts: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> None:
    """
    Run a live end-to-end trace against a running vLLM server (OpenAI-compatible).

    For each environment type (DapoMath, LiveCodeBench, SciKnowEval) we:
      1. Sample `n_prompts` environments from the dataset.
      2. Call env.init([]) to get the opening conversation.
      3. Send that conversation to the vLLM server via the OpenAI chat-completions API.
      4. Feed the completion back through env.step().
      5. Print the full trace (prompt, completion, reward, feedback) with Rich.

    Args:
        base_url:    Base URL of the vLLM server, e.g. "http://localhost:8000".
        model:       Model name as registered in vLLM (auto-detected if None).
        n_prompts:   How many envs to sample per dataset.
        temperature: Sampling temperature sent to vLLM.
        max_tokens:  Max new tokens per completion.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai to use test_vllm_server") from e

    if _rich_console is None:
        raise ImportError("pip install rich to use test_vllm_server")

    console = _rich_console
    client = OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY")

    # Auto-detect model name from the server if not provided
    if model is None:
        models = client.models.list().data
        if not models:
            raise RuntimeError(f"No models served at {base_url}")
        model = models[0].id
        console.print(f"[dim]Auto-detected model: {model}[/dim]")

    def _generate(messages: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    # ---- DapoMathEnv -------------------------------------------------------
    console.rule("[bold yellow]DapoMathEnv — live vLLM traces[/bold yellow]")
    math_envs = DapoMathEnv.load()[:n_prompts]
    for i, env in enumerate(math_envs):
        messages, _ = env.init([])
        completion = _generate(messages)
        result = env.step(completion)
        _render_trace(
            console, "DapoMathEnv", i,
            prompt=env.prompt,
            completion=completion,
            reward=result["reward"],
            feedback=result["metadata"]["feedback"],
        )

    # ---- LiveCodeBenchEnv --------------------------------------------------
    console.rule("[bold yellow]LiveCodeBenchEnv — live vLLM traces[/bold yellow]")
    code_envs = LiveCodeBenchEnv.load(dataset_split="train")[:n_prompts]
    for i, env in enumerate(code_envs):
        messages, _ = env.init([])
        completion = _generate(messages)
        result = env.step(completion)
        _render_trace(
            console, "LiveCodeBenchEnv", i,
            prompt=env.prompt,
            completion=completion,
            reward=result["reward"],
            feedback=result["metadata"]["feedback"],
        )

    # ---- SciKnowEvalEnv ----------------------------------------------------
    console.rule("[bold yellow]SciKnowEvalEnv — live vLLM traces[/bold yellow]")
    sci_envs = SciKnowEvalEnv.load(test_size=0.1, seed=42)[:n_prompts]
    for i, env in enumerate(sci_envs):
        messages, _ = env.init([])
        completion = _generate(messages)
        result = env.step(completion)
        _render_trace(
            console, "SciKnowEvalEnv", i,
            prompt=env.prompt,
            completion=completion,
            reward=result["reward"],
            feedback=result["metadata"]["feedback"],
        )

    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset env smoke tests + optional live vLLM trace")
    parser.add_argument("--vllm", action="store_true", help="Run live vLLM server test instead of mock tests")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM server base URL")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument("--n-prompts", type=int, default=3, help="Prompts per dataset")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.vllm:
        test_vllm_server(
            base_url=args.base_url,
            model=args.model,
            n_prompts=args.n_prompts,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        test_dapo_math()
        test_livecodebench()
        test_sciknoweval()
        print("\n\nDone.")
