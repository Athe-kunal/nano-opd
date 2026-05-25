"""
Streamlit app: Top-K logprobs per token, conditioned on rubrics from CLAIR dataset.

For each token position in the chosen/rejected response, shows the top-k candidate
tokens the model predicts when the rubric is appended to the user prompt:

    [User]: <original prompt>

    Rubric: <rubric text>
    [Assistant]: <chosen or rejected response>

Usage:
    streamlit run opd/apo-opd/app.py
"""

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

RUBRIC_PATH = Path(__file__).parent / "rubric_results.jsonl"
JUDGE_RUBRIC_PATH = Path(__file__).parent / "judge_rubrics.jsonl"
DATASET_NAME = "ContextualAI/ultrafeedback_clair_32k"


# ── data loading ──────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading rubric results...")
def load_rubrics() -> dict[str, dict[str, list[str]]]:
    """Returns {prompt: {"clair": [...], "judge": [...]}}."""
    rubrics: dict[str, dict[str, list[str]]] = {}

    def _load(path: Path, key: str):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    p = row["prompt"]
                    rubrics.setdefault(p, {"clair": [], "judge": []})
                    rubrics[p][key] = row["rubric"]

    _load(RUBRIC_PATH, "clair")
    if JUDGE_RUBRIC_PATH.exists():
        _load(JUDGE_RUBRIC_PATH, "judge")

    return rubrics


@st.cache_data(show_spinner="Loading CLAIR dataset...")
def load_clair() -> list[dict]:
    ds = load_dataset(DATASET_NAME, split="train")
    rows = []
    for row in ds:
        rows.append(
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
        )
    return rows


# ── model loading ─────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model and tokenizer...")
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


# ── logprob computation ───────────────────────────────────────────────────────


def _build_input(
    tokenizer,
    prompt: str,
    rubric: str,
    response: str,
) -> tuple[list[int], int]:
    """
    Returns (input_ids, response_start_idx).

    Applies the chat template if available; falls back to a plain text format.
    response_start_idx is the first token index that belongs to the response.
    """
    user_content = f"{prompt}\n\nRubric: {rubric}" if rubric else prompt

    if tokenizer.chat_template is not None:
        # Build prompt-only first to find where the response starts.
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=True,
            add_generation_prompt=True,
        )
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        full_ids = prompt_only + response_ids
        response_start = len(prompt_only)
    else:
        prompt_text = f"User: {user_content}\nAssistant: "
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        full_ids = prompt_ids + response_ids
        response_start = len(prompt_ids)

    return full_ids, response_start


@torch.inference_mode()
def get_topk_logprobs(
    model,
    tokenizer,
    prompt: str,
    rubric: str,
    response: str,
    top_k: int,
) -> list[dict]:
    """
    Run a single forward pass and return top-k predictions for each response
    token position.

    Returns a list of dicts, one per response token:
        {
            "position": int,
            "actual_token": str,
            "actual_token_id": int,
            "topk": [{"token": str, "token_id": int, "logprob": float}, ...]
        }
    """
    input_ids, response_start = _build_input(tokenizer, prompt, rubric, response)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(model.device)

    outputs = model(input_tensor)
    # logits: [1, seq_len, vocab]
    logits = outputs.logits[0]  # [seq_len, vocab]

    results = []
    # The logit at position t predicts token t+1.
    # Response tokens start at index response_start.
    # So response token i is at full_ids[response_start + i],
    # predicted by logit at response_start + i - 1.
    for i, token_id in enumerate(input_ids[response_start:]):
        logit_pos = response_start + i - 1
        if logit_pos < 0:
            continue

        log_probs = torch.log_softmax(logits[logit_pos], dim=-1)
        topk_vals, topk_ids = torch.topk(log_probs, k=min(100, log_probs.shape[-1]))

        topk_list = [
            {
                "token": tokenizer.decode([tid.item()]),
                "token_id": tid.item(),
                "logprob": val.item(),
            }
            for tid, val in zip(topk_ids, topk_vals)
        ]

        results.append(
            {
                "position": i,
                "actual_token": tokenizer.decode([token_id]),
                "actual_token_id": token_id,
                "topk": topk_list,
            }
        )

    return results


# ── UI helpers ────────────────────────────────────────────────────────────────


def _score_color(t: float) -> str:
    """White (low) → red (high). t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    r, g, b = 220, int(220 * (1 - t)), int(220 * (1 - t))
    return f"rgba({r},{g},{b},0.55)"


def _topk_probs(topk: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (token_ids, renormalized probs) tensors for a stored topk list."""
    ids = torch.tensor([t["token_id"] for t in topk])
    probs = torch.softmax(torch.tensor([t["logprob"] for t in topk]), dim=0)
    return ids, probs


def _kl_at_position(with_rubric: list[dict], without_rubric: list[dict]) -> float:
    """
    KL(p_with_rubric || p_without_rubric) over the shared top-100 vocabulary.

    Both distributions are renormalized over their stored top-100 tokens.
    For tokens in p_with_rubric that are missing from p_without_rubric we
    assign a small epsilon probability to avoid -inf.
    """
    ids_r, p_r = _topk_probs(with_rubric)
    ids_nr, p_nr = _topk_probs(without_rubric)

    # Build a lookup: token_id -> prob for the no-rubric distribution.
    nr_lookup = {tid.item(): prob.item() for tid, prob in zip(ids_nr, p_nr)}
    eps = 1e-9

    kl = 0.0
    for tid, pr in zip(ids_r, p_r):
        q = nr_lookup.get(tid.item(), eps)
        kl += pr.item() * (torch.log(pr) - torch.log(torch.tensor(q))).item()
    return kl


def _render_highlighted_response(
    token_results: list[dict],
    top_k: int,
    reference_results: list[dict] | None = None,
) -> str:
    """
    Highlight the top-k most rubric-influenced token positions using
    KL(p_with_rubric || p_without_rubric).

    If reference_results is None, falls back to entropy as the score.
    """
    if reference_results is not None and len(reference_results) == len(token_results):
        scores = [
            _kl_at_position(r["topk"], ref["topk"])
            for r, ref in zip(token_results, reference_results)
        ]
        tooltip_label = "KL divergence"
    else:
        def _entropy(topk):
            probs = torch.softmax(torch.tensor([t["logprob"] for t in topk]), dim=0)
            return -torch.sum(probs * torch.log(probs + 1e-10)).item()
        scores = [_entropy(e["topk"]) for e in token_results]
        tooltip_label = "entropy"

    top_indices = set(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    )
    highlighted = [scores[i] for i in top_indices]
    min_s = min(highlighted) if highlighted else 0.0
    max_s = max(highlighted) if highlighted else 1.0

    parts = []
    for idx, (entry, score) in enumerate(zip(token_results, scores)):
        display = (
            entry["actual_token"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
            .replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
        )
        if idx in top_indices:
            t = (score - min_s) / (max_s - min_s) if max_s > min_s else 1.0
            color = _score_color(t)
            parts.append(
                f'<span style="background:{color};border-radius:3px;padding:1px 2px;'
                f'cursor:default;" title="{tooltip_label}: {score:.4f}">{display}</span>'
            )
        else:
            parts.append(display)

    return "".join(parts)




# ── main app ──────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="APO Logprob Explorer", layout="wide")
    st.title("APO Logprob Explorer")
    st.caption(
        "Inspect top-k token predictions conditioned on a rubric from the CLAIR dataset."
    )

    # ── sidebar: model config ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("Model")
        model_id = st.text_input(
            "HuggingFace model ID",
            value="Qwen/Qwen2.5-7B-Instruct",
        )
        load_btn = st.button("Load model", type="primary")

        st.divider()
        st.header("High-entropy tokens")
        top_k = st.slider("Top-k highest entropy tokens to highlight", min_value=1, max_value=100, value=20)

    if load_btn:
        with st.spinner(f"Loading {model_id} on GPU (fp32)..."):
            try:
                load_model(model_id)
                st.sidebar.success("Model loaded.")
            except Exception as e:
                st.sidebar.error(f"Failed to load model: {e}")
                return

    # ── load data ──────────────────────────────────────────────────────────────
    rubrics_map = load_rubrics()
    clair_rows = load_clair()

    # Filter to rows that have rubrics (from either source).
    rows_with_rubrics = [r for r in clair_rows if r["prompt"] in rubrics_map]
    st.info(
        f"Dataset: {len(clair_rows)} rows total, "
        f"{len(rows_with_rubrics)} have rubrics extracted."
    )

    # ── prompt navigation (prev / next buttons) ────────────────────────────────
    st.subheader("1. Select a prompt")
    total = len(rows_with_rubrics)

    if "prompt_idx" not in st.session_state:
        st.session_state["prompt_idx"] = 0

    col_prev, col_counter, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("← Prev", disabled=st.session_state["prompt_idx"] == 0):
            st.session_state["prompt_idx"] -= 1
            st.session_state.pop("last_results", None)
    with col_next:
        if st.button("Next →", disabled=st.session_state["prompt_idx"] == total - 1):
            st.session_state["prompt_idx"] += 1
            st.session_state.pop("last_results", None)
    with col_counter:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px;'>"
            f"Prompt <b>{st.session_state['prompt_idx'] + 1}</b> / {total}</div>",
            unsafe_allow_html=True,
        )

    selected_row = rows_with_rubrics[st.session_state["prompt_idx"]]
    prompt = selected_row["prompt"]

    user_prompt = selected_row["rejected"][0]["content"]
    st.markdown("**User prompt**")
    st.write(user_prompt)

    # ── responses side by side ─────────────────────────────────────────────────
    # st.subheader("2. Responses")

    def _extract_response(raw) -> str:
        if isinstance(raw, list):
            return next(
                (turn["content"] for turn in raw if turn.get("role") == "assistant"),
                str(raw),
            )
        return str(raw)

    chosen_text = _extract_response(selected_row["chosen"])
    rejected_text = _extract_response(selected_row["rejected"])

    # col_chosen, col_rejected = st.columns(2)
    # with col_chosen:
    #     st.markdown("**Chosen**")
    #     with st.expander("Show chosen response", expanded=True):
    #         st.write(chosen_text)
    # with col_rejected:
    #     st.markdown("**Rejected**")
    #     with st.expander("Show rejected response", expanded=True):
    #         st.write(rejected_text)

    # ── rubric selection ───────────────────────────────────────────────────────
    st.subheader("2. Select or write a rubric")
    sources = rubrics_map.get(prompt, {"clair": [], "judge": []})
    # Build a flat list of (source_label, rubric_text) tuples for the dropdown.
    rubric_options: list[tuple[str, str]] = []
    for i, r in enumerate(sources.get("clair", [])):
        rubric_options.append((f"[CLAIR {i+1}] {r}", r))
    for i, r in enumerate(sources.get("judge", [])):
        rubric_options.append((f"[Judge {i+1}] {r}", r))

    rubric_text_idx = st.selectbox(
        "Preset rubrics",
        options=range(len(rubric_options)),
        format_func=lambda i: rubric_options[i][0],
    )
    preset_rubric = rubric_options[rubric_text_idx][1] if rubric_options else ""

    rubric_text = st.text_area(
        "Rubric (edit freely — this is what gets passed to the model)",
        value=preset_rubric,
        height=80,
        key=f"rubric_text_{rubric_text_idx}_{st.session_state['prompt_idx']}",
    )

    # ── forward pass for both responses ───────────────────────────────────────
    st.subheader("3. Compute top-k logprobs")

    try:
        model, tokenizer = load_model(model_id)
    except Exception:
        st.warning("Load a model first using the sidebar.")
        return

    col_run_base, col_run_rubric = st.columns(2)
    with col_run_base:
        run_base = st.button("Run baseline passes (chosen + rejected, no rubric)", type="secondary")
    with col_run_rubric:
        run_rubric = st.button("Run rejected + rubric", type="primary")

    if run_base:
        with st.spinner("Running 2 baseline forward passes..."):
            try:
                st.session_state["chosen_results"] = get_topk_logprobs(
                    model, tokenizer, prompt, "", chosen_text, top_k
                )
                st.session_state["rejected_no_rubric_results"] = get_topk_logprobs(
                    model, tokenizer, prompt, "", rejected_text, top_k
                )
                st.success("Baseline passes done.")
            except Exception as e:
                st.error(f"Error during baseline pass: {e}")
                raise

    if run_rubric:
        with st.spinner("Running rejected + rubric forward pass..."):
            try:
                st.session_state["rejected_rubric_results"] = get_topk_logprobs(
                    model, tokenizer, prompt, rubric_text, rejected_text, top_k
                )
                st.success("Done.")
            except Exception as e:
                st.error(f"Error during rubric pass: {e}")
                raise

    if "chosen_results" in st.session_state:
        chosen_results = st.session_state["chosen_results"]
        rejected_rubric_results = st.session_state["rejected_rubric_results"]
        rejected_no_rubric_results = st.session_state["rejected_no_rubric_results"]

        legend_items = "".join(
            f'<span style="background:{_score_color(v)};'
            f'border-radius:3px;padding:1px 8px;">{label}</span>'
            for v, label in [(0.0, "low influence"), (0.5, "mid"), (1.0, "high influence")]
        )
        div_style = (
            "font-family:monospace;font-size:0.9em;"
            "line-height:1.8;word-break:break-word;"
            "padding:8px;border:1px solid #e0e0e0;border-radius:6px;margin-top:6px;"
        )
        header_style = "font-size:1em;font-weight:bold;margin-bottom:4px;color:#333;"

        # KL highlight for rejected+rubric vs rejected baseline.
        rejected_rubric_html = _render_highlighted_response(
            rejected_rubric_results, top_k, reference_results=rejected_no_rubric_results
        )
        # Entropy fallback for the two no-rubric responses.
        chosen_html = _render_highlighted_response(chosen_results, top_k)
        rejected_no_rubric_html = _render_highlighted_response(rejected_no_rubric_results, top_k)

        rubric_escaped = rubric_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        full_html = f"""
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;font-size:0.85em;">
          {legend_items}
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
          <div>
            <div style="{header_style}">Rejected (no rubric)</div>
            <div style="{div_style}">{rejected_no_rubric_html}</div>
          </div>
          <div>
            <div style="{header_style}">Rejected + rubric: <em>{rubric_escaped}</em></div>
            <div style="{div_style}">{rejected_rubric_html}</div>
          </div>
          <div>
            <div style="{header_style}">Chosen (no rubric)</div>
            <div style="{div_style}">{chosen_html}</div>
          </div>
        </div>
        """
        height = max(len(chosen_text), len(rejected_text)) // 2 + 500
        components.html(full_html, height=min(height, 1600), scrolling=True)


if __name__ == "__main__":
    main()
