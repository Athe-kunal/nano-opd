import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import pydantic

from datasets import load_dataset

clair_ds = load_dataset("ContextualAI/ultrafeedback_clair_32k")



class RejectedChosen(pydantic.BaseModel):
    prompt: str
    chosen: str
    rejected: str


responses = []
for i in clair_ds['train']:
    responses.append(
        RejectedChosen(
            prompt = i['prompt'],
            chosen = i['chosen'][-1]['content'],
            rejected = i['rejected'][-1]['content'],
        )
    )

# ── load rubrics ──────────────────────────────────────────────────────────────
RUBRIC_PATH       = Path("opd/apo-opd/rubric_results.jsonl")
JUDGE_RUBRIC_PATH = Path("opd/apo-opd/judge_rubrics.jsonl")

kl_rubrics_map: dict[str, dict[str, list[str]]] = {}

def _load_rubric_file(path: Path, key: str):
    if not path.exists():
        print(f"Warning: {path} not found, skipping.")
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                p = row["prompt"]
                kl_rubrics_map.setdefault(p, {"clair": [], "judge": []})
                kl_rubrics_map[p][key] = row["rubric"]

_load_rubric_file(RUBRIC_PATH, "clair")
_load_rubric_file(JUDGE_RUBRIC_PATH, "judge")
print(f"Loaded rubrics for {len(kl_rubrics_map)} prompts.")

# ── load model on cuda:1 ──────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-8B"
DEVICE   = "cuda:1"

kl_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
kl_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map=DEVICE,
)
kl_model.eval()
print("Model loaded.")

# ── helpers ───────────────────────────────────────────────────────────────────

def _build_ids(prompt: str, rubric: str, response: str) -> tuple[list[int], int]:
    """Returns (full_input_ids, response_start_idx)."""
    user_content = f"{prompt}\n\nRubric: {rubric}" if rubric else prompt
    if kl_tokenizer.chat_template is not None:
        prompt_ids = kl_tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        prompt_ids = kl_tokenizer.encode(
            f"User: {user_content}\nAssistant: ", add_special_tokens=True
        )
    response_ids = kl_tokenizer.encode(response, add_special_tokens=False)
    return prompt_ids + response_ids, len(prompt_ids)


def _pad_and_run(ids_list: list[list[int]]) -> torch.Tensor:
    """Pad ids_list to uniform length and run one forward pass. Returns logits [B, T, V]."""
    pad_id  = kl_tokenizer.pad_token_id or 0
    max_len = max(len(ids) for ids in ids_list)
    padded  = [ids + [pad_id] * (max_len - len(ids)) for ids in ids_list]
    attn    = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in ids_list]
    input_t = torch.tensor(padded, dtype=torch.long, device=DEVICE)
    attn_t  = torch.tensor(attn,   dtype=torch.long, device=DEVICE)
    return kl_model(input_t, attention_mask=attn_t).logits  # [B, T, V]


def _resp_positions(rs: int, resp_len: int) -> torch.Tensor:
    """Response token positions in the full input: logit at rs+i-1 predicts token i."""
    return torch.arange(resp_len, device=DEVICE) + max(rs - 1, 0)


@torch.inference_mode()
def compute_kl_per_token(
    no_rubric_ids:   list[list[int]],
    no_rs:           list[int],
    with_rubric_ids: list[list[int]],
    with_rs:         list[int],
    resp_lens:       list[int],
    top_k:           int = 500,
) -> list[list[float]]:
    """
    Two sequential forward passes so peak GPU memory is O(B*T_with*V) then
    O(B*T_no*V), never both at once.

    Pass 1 (with_rubric): extract top-K logit values + indices per response
    token → move [resp_len, K] tensors to CPU, free [B, T, V] on GPU.

    Pass 2 (no_rubric): gather logits at exactly those same K indices → move
    [resp_len, K] to CPU, free [B, T, V] on GPU.

    KL is then computed on CPU over small [resp_len, K] tensors only.
    """
    B = len(resp_lens)

    # ── Pass 1: with_rubric → top-K vals and indices ──────────────────────────
    logits_with = _pad_and_run(with_rubric_ids)   # [B, T_with, V]
    topk_vals_with: list[torch.Tensor] = []       # per item: [resp_len, K] on CPU
    topk_idx:       list[torch.Tensor] = []       # per item: [resp_len, K] on CPU

    for b in range(B):
        if resp_lens[b] == 0:
            topk_vals_with.append(torch.empty(0, top_k))
            topk_idx.append(torch.empty(0, top_k, dtype=torch.long))
            continue
        pos  = _resp_positions(with_rs[b], resp_lens[b])
        lgt  = logits_with[b, pos].float()                             # [resp_len, V]
        v, idx = torch.topk(lgt, k=min(top_k, lgt.shape[-1]), dim=-1) # [resp_len, K]
        topk_vals_with.append(v.cpu())
        topk_idx.append(idx.cpu())

    del logits_with
    torch.cuda.empty_cache()

    # ── Pass 2: no_rubric → gather logits at with_rubric's top-K indices ──────
    logits_no = _pad_and_run(no_rubric_ids)       # [B, T_no, V]
    topk_vals_no: list[torch.Tensor] = []         # per item: [resp_len, K] on CPU

    for b in range(B):
        if resp_lens[b] == 0:
            topk_vals_no.append(torch.empty(0, top_k))
            continue
        pos    = _resp_positions(no_rs[b], resp_lens[b])
        lgt    = logits_no[b, pos].float()                             # [resp_len, V]
        idx_gpu = topk_idx[b].to(DEVICE)
        gathered = lgt.gather(1, idx_gpu)                              # [resp_len, K]
        topk_vals_no.append(gathered.cpu())

    del logits_no
    torch.cuda.empty_cache()

    # ── KL on CPU over [resp_len, K] tensors ──────────────────────────────────
    results: list[list[float]] = []
    for b in range(B):
        if resp_lens[b] == 0:
            results.append([])
            continue

        log_p = F.log_softmax(topk_vals_with[b], dim=-1)  # [resp_len, K]
        log_q = F.log_softmax(topk_vals_no[b],   dim=-1)  # [resp_len, K]
        kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1).clamp(min=0.0)  # [resp_len]

        if no_rs[b] == 0 or with_rs[b] == 0:
            kl[0] = 0.0

        results.append(kl.tolist())

    return results


def _sentence_token_ranges(response: str) -> list[tuple[int, int, str]]:
    """
    Returns [(tok_start, tok_end, label), ...] where label is 'start'/'middle'/'end'
    relative to the paragraph the sentence belongs to.
    Indices are into the response token array (0-indexed).
    """
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    ranges: list[tuple[int, int, str]] = []
    tok_offset = 0
    for para in paragraphs:
        sents = sent_tokenize(para)
        n = len(sents)
        for i, sent in enumerate(sents):
            sent_len = len(kl_tokenizer.encode(sent, add_special_tokens=False))
            if n == 1:
                label = "start"  # single-sentence paragraph: counts as start (and end below)
            elif i == 0:
                label = "start"
            elif i == n - 1:
                label = "end"
            else:
                label = "middle"
            ranges.append((tok_offset, tok_offset + sent_len, label))
            tok_offset += sent_len
    return ranges


# ── main loop ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 32    # raise/lower depending on free VRAM
TOP_K_KL   = 500   # vocab tokens used for KL approximation

items_with_rubrics = [
    (item, kl_rubrics_map[item.prompt])
    for item in responses
    if item.prompt in kl_rubrics_map
]
print(f"{len(items_with_rubrics)} / {len(responses)} items have rubrics.")

# Sort by approximate sequence length so each batch is internally uniform and
# padding waste is minimized. Order does not affect the per-item KL aggregation.
def _approx_len(item) -> int:
    return len(kl_tokenizer.encode(item.prompt + item.rejected, add_special_tokens=False))

items_with_rubrics.sort(key=lambda x: _approx_len(x[0]))

start_kls, middle_kls, end_kls = [], [], []

for batch_start in tqdm(range(0, min(len(items_with_rubrics), 300 * BATCH_SIZE), BATCH_SIZE), desc="KL forward passes"):
    batch = items_with_rubrics[batch_start : batch_start + BATCH_SIZE]

    no_rubric_ids,   no_rs   = [], []
    with_rubric_ids, with_rs = [], []
    resp_lens = []

    for item, sources in batch:
        judge_rubrics   = sources.get("judge", [])
        all_rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(judge_rubrics))

        ids_no,   rs_no   = _build_ids(item.prompt, "",             item.rejected)
        ids_with, rs_with = _build_ids(item.prompt, all_rubric_text, item.rejected)

        no_rubric_ids.append(ids_no);     no_rs.append(rs_no)
        with_rubric_ids.append(ids_with); with_rs.append(rs_with)
        resp_lens.append(len(kl_tokenizer.encode(item.rejected, add_special_tokens=False)))

    # One fused forward pass for both conditions; top-K KL computed on GPU.
    kl_per_tok_batch = compute_kl_per_token(
        no_rubric_ids, no_rs, with_rubric_ids, with_rs, resp_lens
    )

    for b, (item, _) in enumerate(batch):
        kl_per_tok = kl_per_tok_batch[b]

        for tok_start, tok_end, label in _sentence_token_ranges(item.rejected):
            sent_toks = kl_per_tok[tok_start:tok_end]
            if not sent_toks:
                continue
            mean_sent_kl = sum(sent_toks) / len(sent_toks)
            if label == "start":
                start_kls.append(mean_sent_kl)
            elif label == "middle":
                middle_kls.append(mean_sent_kl)
            else:
                end_kls.append(mean_sent_kl)

# ── results ───────────────────────────────────────────────────────────────────
import numpy as np
from scipy import stats

def summarize(name: str, vals: list[float]):
    if len(vals) < 2:
        print(f"  {name}: insufficient data (n={len(vals)})")
        return
    a = np.array(vals, dtype=np.float64)
    mean = a.mean()
    std  = a.std(ddof=1)
    se   = std / np.sqrt(len(a))
    # 95% CI via t-distribution (exact for any n, converges to z for large n)
    lo, hi = stats.t.interval(0.95, df=len(a) - 1, loc=mean, scale=se)
    print(f"  {name:6s}: mean={mean:.5f}  std={std:.5f}  95% CI=[{lo:.5f}, {hi:.5f}]  (n={len(a)})")

print(f"\nMean per-token KL(rejected+rubric || rejected)")
summarize("Start",  start_kls)
summarize("Middle", middle_kls)
summarize("End",    end_kls)

# Pairwise Mann-Whitney U tests (non-parametric: KL values are skewed/non-normal)
print("\nMann-Whitney U tests (one-sided: row > col)")
pairs = [("Start", start_kls), ("Middle", middle_kls), ("End", end_kls)]
for (n1, v1), (n2, v2) in [(pairs[i], pairs[j]) for i in range(len(pairs)) for j in range(i+1, len(pairs))]:
    stat, p = stats.mannwhitneyu(v1, v2, alternative="greater")
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {n1} > {n2}: p={p:.2e}  {sig}")