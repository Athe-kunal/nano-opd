"""
Rubric extractor for the ContextualAI/ultrafeedback_clair_32k dataset.

Calls an async OpenAI client to extract rubrics from the `rational` column.

Two rational formats are handled:
  - Numbered (1. **Title** ...): one rubric per numbered item
  - Free-form: a single list of rubrics covering the whole rational

Results are saved to RESULTS_PATH (jsonl) keyed by prompt so re-runs skip
already-processed rows.

Env vars:
  OPENAI_API_KEY   required
  OPENAI_BASE_URL  optional, defaults to OpenAI
  OPENAI_MODEL     optional, defaults to gpt-4o-mini

Usage:
  python rubric_extractor.py            # process full dataset
  python rubric_extractor.py --smoke    # run smoke test on two hardcoded rationals
  python rubric_extractor.py --samples 100  # process first 100 rows
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from pydantic import BaseModel

RESULTS_PATH = Path(__file__).parent / "rubric_results.jsonl"

# ── sample rationals for smoke testing ────────────────────────────────────────

FREE_FORM_RATIONAL = (
    "The student's answer correctly identifies the duration for steaming the white "
    "asparagus as 6-8 minutes, which matches the instructions given in the recipe. "
    "However, to make the answer clearer and more engaging, we can add a little more "
    "context about why steaming for this specific amount of time is important, "
    'emphasizing the goal of achieving a "crisp-tender" texture. This not only '
    "reinforces the cooking technique but also educates on the desired outcome, making "
    "it more informative and engaging for someone following the recipe."
)

NUMBERED_RATIONAL = """\
To enhance the student's answer, we should focus on clarifying and enriching the \
information provided, making it more engaging and informative. Here are some steps \
we can take:

1. **Clarify and Expand Examples**: While the student lists dishes that use red chili \
powder, we can provide brief descriptions or details to make it clearer how red chili \
powder contributes to each dish's flavor profile.
2. **Enhance Engagement**: Adding an introductory sentence that invites the reader to \
explore the versatility of red chili powder can make the text more engaging. We might \
also include a sentence or two about the cultural significance of red chili powder in \
various cuisines.
3. **Practical Tips**: The advice on using red chili powder is sound, but we can \
enhance it by suggesting specific amounts to start with and mentioning how its flavor \
can change with cooking time.
4. **Correct and Improve Language**: Maintain the student's informal and friendly tone \
but make slight adjustments for readability and clarity. Ensure that all sentences flow \
well together."""

# ── config ────────────────────────────────────────────────────────────────────


def _make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL"),  # None → OpenAI default
    )


def _default_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ── pydantic response schema ──────────────────────────────────────────────────


class RubricOutput(BaseModel):
    rubric: list[str]


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a concise rubric extractor. Given feedback about an answer, extract a list of rubrics.
Each rubric is a short, actionable criterion (one sentence) that a good answer should satisfy.
Return only the rubric list — no explanation."""

CHUNK_PROMPT = """\
Feedback chunk:
\"\"\"
{chunk}
\"\"\"

Extract a single rubric (one sentence) from this feedback chunk."""

FREE_FORM_PROMPT = """\
Feedback:
\"\"\"
{rational}
\"\"\"

Extract a list of rubrics from this feedback. Each rubric is one short, actionable sentence."""

# ── parsing ───────────────────────────────────────────────────────────────────


def _split_numbered_chunks(rational: str) -> list[str] | None:
    """Return chunks for numbered rationals (≥2 items), else None."""
    pattern = re.compile(r"(?m)^(?:\d+\.\s)")
    positions = [m.start() for m in pattern.finditer(rational)]
    if len(positions) < 2:
        return None
    chunks = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(rational)
        chunks.append(rational[start:end].strip())
    return chunks


# ── API calls ─────────────────────────────────────────────────────────────────


async def _extract_rubric_for_chunk(client: AsyncOpenAI, model: str, chunk: str) -> str:
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": CHUNK_PROMPT.format(chunk=chunk)},
        ],
        response_format=RubricOutput,
        max_tokens=1024,
    )
    result = response.choices[0].message.parsed
    return result.rubric[0] if result.rubric else ""


async def _extract_rubrics_free_form(client: AsyncOpenAI, model: str, rational: str) -> list[str]:
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FREE_FORM_PROMPT.format(rational=rational)},
        ],
        response_format=RubricOutput,
        max_tokens=1024,
    )
    result = response.choices[0].message.parsed
    return result.rubric


async def extract_rubrics(client: AsyncOpenAI, model: str, rational: str) -> RubricOutput:
    """
    Return a RubricOutput: one rubric per numbered chunk, or a list for free-form.
    """
    chunks = _split_numbered_chunks(rational)
    if chunks:
        rubrics = await asyncio.gather(
            *[_extract_rubric_for_chunk(client, model, chunk) for chunk in chunks]
        )
        return RubricOutput(rubric=list(rubrics))
    rubrics = await _extract_rubrics_free_form(client, model, rational)
    return RubricOutput(rubric=rubrics)


# ── persistence ───────────────────────────────────────────────────────────────


def _load_done(path: Path) -> dict[str, list[str]]:
    """Return {prompt: rubric_list} for all already-saved results."""
    done: dict[str, list[str]] = {}
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                done[row["prompt"]] = row["rubric"]
    return done


def _append_result(path: Path, prompt: str, rubric: list[str]) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"prompt": prompt, "rubric": rubric}) + "\n")


# ── dataset processing ────────────────────────────────────────────────────────


async def process_dataset(
    model: str | None = None,
    max_concurrent: int = 64,
    num_samples: int | None = None,
    results_path: Path = RESULTS_PATH,
) -> list[dict[str, Any]]:
    """
    Load the dataset, skip already-processed prompts, call the API for the rest,
    and append new results to results_path.
    """
    model = model or _default_model()
    ds = load_dataset("ContextualAI/ultrafeedback_clair_32k", split="train")
    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    done = _load_done(results_path)
    print(f"Already done: {len(done)} / {len(ds)}")

    client = _make_client()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded(row: dict[str, Any]) -> dict[str, Any] | None:
        prompt = row["prompt"]
        if prompt in done:
            return None
        async with semaphore:
            rubric_out = await extract_rubrics(client, model, row["rational"])
            _append_result(results_path, prompt, rubric_out.rubric)
            return {"prompt": prompt, "rubric": rubric_out.rubric}

    tasks = [_bounded(row) for row in ds]
    new_results = [
        r
        for r in await tqdm.gather(*tasks, desc="Extracting rubrics", unit="prompt")
        if r is not None
    ]
    print(f"Newly processed: {len(new_results)}")

    # Return everything (old + new)
    all_done = _load_done(results_path)
    return [{"prompt": p, "rubric": r} for p, r in all_done.items()]


# ── smoke test ────────────────────────────────────────────────────────────────


async def _smoke_test() -> None:
    client = _make_client()
    model = _default_model()
    print(f"Smoke test using model: {model}\n")

    # --- unit checks (no API) ---
    assert _split_numbered_chunks(FREE_FORM_RATIONAL) is None, "free-form should return None"
    chunks = _split_numbered_chunks(NUMBERED_RATIONAL)
    assert chunks is not None and len(chunks) == 4, f"expected 4 chunks, got {chunks}"
    assert _split_numbered_chunks("1. **Only item**: text") is None, "single item should be None"
    print("Parsing checks passed.\n")

    # --- live API: free-form ---
    print("Testing free-form rational...")
    result_ff = await extract_rubrics(client, model, FREE_FORM_RATIONAL)
    assert isinstance(result_ff.rubric, list) and len(result_ff.rubric) >= 1
    print(f"  rubrics ({len(result_ff.rubric)}): {result_ff.rubric}\n")

    # --- live API: numbered ---
    print("Testing numbered rational...")
    result_num = await extract_rubrics(client, model, NUMBERED_RATIONAL)
    assert len(result_num.rubric) == 4, f"expected 4 rubrics, got {len(result_num.rubric)}"
    for i, r in enumerate(result_num.rubric, 1):
        print(f"  [{i}] {r}")

    print("\nSmoke test passed.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    parser.add_argument("--samples", type=int, default=None, help="Limit to N rows")
    args = parser.parse_args()

    if args.smoke:
        asyncio.run(_smoke_test())
    else:
        results = asyncio.run(process_dataset(num_samples=args.samples))
        print(f"Total results: {len(results)}")
