"""Smoke tests for OPDEnvBase subclasses (init / step / compute_reward / get_feedback)."""

from opd.envs.dapo_dataset import DapoMathEnv
from opd.envs.livecodebench import LiveCodeBenchEnv
from opd.envs.sciknoweval import SciKnowEvalEnv

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


if __name__ == "__main__":
    test_dapo_math()
    test_livecodebench()
    test_sciknoweval()
    print("\n\nDone.")
