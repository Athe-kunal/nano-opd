import torch

def _masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """token-mean: sum over all response tokens / count of response tokens."""
    return (values * mask).sum() / mask.sum().clamp(min=1)

def compute_reverse_kl_loss(
    student_logprobs: torch.Tensor,   # [B, seq_len-1, V] — log probs
    teacher_logprobs: torch.Tensor,   # [B, seq_len-1, V] — log probs
    response_mask: torch.Tensor,      # [B, seq_len-1]
    top_k: int = -1,
) -> torch.Tensor:
    # Convert log probs to probs for the student (needed to weight the KL)
    student_probs = student_logprobs.exp()  # [B, seq_len-1, V]

    if top_k > 0:
        # Find top-k tokens by student probability at each position
        topk_vals, topk_idx = student_probs.topk(top_k, dim=-1)  # [B, seq_len-1, top_k]

        # Gather log probs for those indices from both models
        s_topk_logprobs = student_logprobs.gather(-1, topk_idx)  # [B, seq_len-1, top_k]
        t_topk_logprobs = teacher_logprobs.gather(-1, topk_idx)  # [B, seq_len-1, top_k]

        # Renormalize student probs over top-k (so they sum to 1 within the subset)
        s_topk_probs = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # KL over top-k vocab: sum_v pi(v) * [log pi(v) - log q(v)]
        per_token_kl = (s_topk_probs * (s_topk_logprobs - t_topk_logprobs)).sum(dim=-1)  # [B, seq_len-1]

    else:
        # Full vocab KL
        per_token_kl = (student_probs * (student_logprobs - teacher_logprobs)).sum(dim=-1)  # [B, seq_len-1]

    return _masked_token_mean(per_token_kl, response_mask)
