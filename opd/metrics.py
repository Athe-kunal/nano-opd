import torch

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
