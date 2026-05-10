import torch


def _masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1)


def compute_reverse_kl_loss(
    student_logprobs: torch.Tensor,   # [B, T, K]
    teacher_logprobs: torch.Tensor,   # [B, T, K]
    response_mask: torch.Tensor,      # [B, T]
    renormalize: bool = True,
) -> torch.Tensor:
    """KL(p_student || p_teacher), truncated to top-K (selected by student)."""
    student_probs = student_logprobs.exp()
    if renormalize:
        student_probs = student_probs / student_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    per_token_kl = (student_probs * (student_logprobs - teacher_logprobs)).sum(dim=-1)
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
        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    per_token_kl = (teacher_probs * (teacher_logprobs - student_logprobs)).sum(dim=-1)
    return _masked_token_mean(per_token_kl, response_mask)


ALGORITHMS = {
    "reverse_kl": compute_reverse_kl_loss,
    "forward_kl": compute_forward_kl_loss,
}