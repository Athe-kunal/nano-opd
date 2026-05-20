import torch
from einops import reduce


def _masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (
        torch.einsum("bt,bt->", values, mask)
        / mask.sum().clamp(min=1)
    )


def compute_reverse_kl_loss(
    student_logprobs: torch.Tensor,   # [B, T, K]
    teacher_logprobs: torch.Tensor,   # [B, T, K]
    response_mask: torch.Tensor,      # [B, T]
    renormalize: bool = True,
) -> torch.Tensor:
    """KL(p_student || p_teacher), truncated to top-K (selected by student)."""
    student_probs = student_logprobs.exp()
    if renormalize:
        student_probs = student_probs / reduce(student_probs, "b t k -> b t 1", "sum").clamp(min=1e-8)
    per_token_kl = torch.einsum("btk,btk->bt", student_probs, student_logprobs - teacher_logprobs)
    return _masked_token_mean(per_token_kl, response_mask)


def compute_forward_kl_loss(
    student_logprobs: torch.Tensor,   # [B, T, K]
    teacher_logprobs: torch.Tensor,   # [B, T, K]
    response_mask: torch.Tensor,      # [B, T]
    renormalize: bool = True,
) -> torch.Tensor:
    """KL(p_teacher || p_student), truncated to top-K (ideally selected by teacher)."""
    teacher_probs = teacher_logprobs.exp()
    if renormalize:
        teacher_probs = teacher_probs / reduce(teacher_probs, "b t k -> b t 1", "sum").clamp(min=1e-8)
    per_token_kl = torch.einsum("btk,btk->bt", teacher_probs, teacher_logprobs - student_logprobs)
    return _masked_token_mean(per_token_kl, response_mask)


def compute_jsd_loss(
    student_logprobs: torch.Tensor,   # [B, T, K]
    teacher_logprobs: torch.Tensor,   # [B, T, K]
    response_mask: torch.Tensor,      # [B, T]
    jsd_alpha: float = 0.5,
    renormalize: bool = True,
) -> torch.Tensor:
    """JSD(student || teacher) with mixture weight jsd_alpha (top-K selected by student).

    jsd_alpha=0.5 → symmetric JSD (SDPO default).
    jsd_alpha=0.0 → forward KL.  jsd_alpha=1.0 → reverse KL.
    """
    student_probs = student_logprobs.exp()
    teacher_probs = teacher_logprobs.exp()
    if renormalize:
        student_probs = student_probs / reduce(student_probs, "b t k -> b t 1", "sum").clamp(min=1e-8)
        teacher_probs = teacher_probs / reduce(teacher_probs, "b t k -> b t 1", "sum").clamp(min=1e-8)
    M = jsd_alpha * student_probs + (1.0 - jsd_alpha) * teacher_probs
    log_M = M.clamp(min=1e-8).log()
    kl_s = torch.einsum("btk,btk->bt", student_probs, student_logprobs - log_M)
    kl_t = torch.einsum("btk,btk->bt", teacher_probs, teacher_logprobs - log_M)
    per_token_jsd = jsd_alpha * kl_s + (1.0 - jsd_alpha) * kl_t
    return _masked_token_mean(per_token_jsd, response_mask)


ALGORITHMS = {
    "reverse_kl": compute_reverse_kl_loss,
    "forward_kl": compute_forward_kl_loss,
    "jsd": compute_jsd_loss,
}
