import torch

from opd.fsdp.algorithms import student_logprobs_at_indices


def compute_overlap_ratio(student_topk_idx: torch.Tensor, #[B,T,K]
                        teacher_topk_idx: torch.Tensor #[B,T,K]
                        ) -> torch.Tensor: #scalar
    student_topk = student_topk_idx.unsqueeze(dim=-1) #[B,T,K,1]
    teacher_topk = teacher_topk_idx.unsqueeze(dim=-2) #[B,T,1,K]
    num_overlap = (student_topk == teacher_topk).any(dim=-1)
    K = student_topk_idx.shape[-1]
    overlap_per_position = num_overlap.float().sum(dim=-1) / K #[B,T]
    return overlap_per_position.mean()

    

def compute_overlap_token_advantage(
    student_topk_idx: torch.Tensor,  # [B, T, K]
    teacher_topk_idx: torch.Tensor,  # [B, T, K]
    s_logprobs: torch.Tensor,        # [B, T, K] — at student top-K
    t_logprobs: torch.Tensor,        # [B, T, K] — at student top-K
) -> torch.Tensor:                   # scalar
    in_intersection = (
        student_topk_idx.unsqueeze(-1) == teacher_topk_idx.unsqueeze(-2)
    ).any(dim=-1)                                                        # [B, T, K]

    # Renormalize over intersection tokens only
    s_lp = s_logprobs.masked_fill(~in_intersection, float('-inf'))
    t_lp = t_logprobs.masked_fill(~in_intersection, float('-inf'))
    s_lp = s_lp - torch.logsumexp(s_lp, dim=-1, keepdim=True)          # [B, T, K]
    t_lp = t_lp - torch.logsumexp(t_lp, dim=-1, keepdim=True)          # [B, T, K]

    # A_t(v) = p̄_t(v) * (log q̄_t(v) - log p̄_t(v))
    advantage = s_lp.exp() * (t_lp - s_lp)                             # [B, T, K]

    # Average over intersection tokens, then over (B, T)
    advantage = advantage.masked_fill(~in_intersection, 0.0)
    n_overlap = in_intersection.float().sum(dim=-1).clamp(min=1)        # [B, T]
    return (advantage.sum(dim=-1) / n_overlap).mean()


def compute_entropy_gap(
    student_topk_logprobs: torch.Tensor,  # [B, T, K] — at student top-K
    teacher_topk_logprobs: torch.Tensor,  # [B, T, K] — at teacher top-K
) -> torch.Tensor:                        # scalar
    s_lp = student_topk_logprobs - torch.logsumexp(student_topk_logprobs, dim=-1, keepdim=True)
    t_lp = teacher_topk_logprobs - torch.logsumexp(teacher_topk_logprobs, dim=-1, keepdim=True)
    s_entropy = -(s_lp.exp() * s_lp).sum(dim=-1)  # [B, T]
    t_entropy = -(t_lp.exp() * t_lp).sum(dim=-1)  # [B, T]
    return torch.abs(t_entropy - s_entropy).mean()


def compute_topk_health_metrics(student_logits, topk, student_chunk_size: int = -1):
    """Overlap ratio / overlap-token advantage / entropy gap between the student
    and teacher top-K distributions in a TopKExchange (see distillation_utils.py).

    Loss-agnostic: these are diagnostics about how aligned the two policies are
    at each token, independent of which loss formula produced the top-K data —
    usable for reverse/forward KL, JSD, or the MOPD-PG form alike.

    Returns (overlap_ratio, overlap_advantage, entropy_gap) as plain floats.
    """
    s_lp = student_logprobs_at_indices(student_logits, topk.student_topk_idx, student_chunk_size)
    overlap_ratio = compute_overlap_ratio(topk.student_topk_idx, topk.teacher_topk_idx).item()
    overlap_advantage = compute_overlap_token_advantage(
        topk.student_topk_idx, topk.teacher_topk_idx, s_lp, topk.t_logprobs_at_student
    ).item()
    entropy_gap = compute_entropy_gap(s_lp, topk.teacher_own_logprobs).item()
    return overlap_ratio, overlap_advantage, entropy_gap
