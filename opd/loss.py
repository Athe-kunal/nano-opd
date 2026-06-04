import torch


def _masked_token_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    tis_weights: torch.Tensor | None = None,
    kl_clip: float | None = None,
) -> torch.Tensor:
    mask = mask.to(values.dtype)
    # Per-token pointwise KL clipping (OPSD paper Section 3.2).
    # Stylistic tokens can dominate the gradient signal with very high divergence
    # values. Clamping each token's contribution to τ before the masked mean
    # prevents this heavy-tail effect from overwhelming math-relevant tokens.
    if kl_clip is not None:
        values = values.clamp(max=kl_clip)
    if tis_weights is not None:
        values = values * tis_weights
    return (
        torch.einsum("bt,bt->", values, mask)
        / mask.sum().clamp(min=1)
    )


def _tail_probs(probs: torch.Tensor) -> torch.Tensor:
    """Probability mass outside the top-K support: 1 - Σ_K p(k)."""
    return (1.0 - probs.sum(dim=-1)).clamp(min=1e-8)  # [B, T]


def compute_reverse_kl_loss(
    student_logprobs: torch.Tensor,         # [B, T, K]
    teacher_logprobs: torch.Tensor,         # [B, T, K]
    response_mask: torch.Tensor,            # [B, T]
    tis_weights: torch.Tensor | None = None,  # [B, T]
    kl_clip: float | None = None,
) -> torch.Tensor:
    """KL(p_student || p_teacher), top-K truncated with tail term (Eq. 11, Appendix A.3).

    The tail term accounts for all probability mass outside the top-K support:
        tail = (1 - Σ_K p_s) · [log(1 - Σ_K p_s) - log(1 - Σ_K p_t)]
    This preserves the signal when the teacher wants to upweight a token the
    student ranked below K — exactly where feedback-driven corrections live.
    """
    student_logprobs = student_logprobs.float()
    teacher_logprobs = teacher_logprobs.float()
    student_probs = student_logprobs.exp()
    teacher_probs = teacher_logprobs.exp()

    s_tail = _tail_probs(student_probs)
    t_tail = _tail_probs(teacher_probs)

    per_token_kl = torch.einsum("btk,btk->bt", student_probs, student_logprobs - teacher_logprobs)
    per_token_kl = per_token_kl + s_tail * (s_tail.log() - t_tail.log())
    return _masked_token_mean(per_token_kl, response_mask, tis_weights, kl_clip)


def compute_forward_kl_loss(
    student_logprobs: torch.Tensor,         # [B, T, K]
    teacher_logprobs: torch.Tensor,         # [B, T, K]
    response_mask: torch.Tensor,            # [B, T]
    tis_weights: torch.Tensor | None = None,  # [B, T]
    kl_clip: float | None = None,
) -> torch.Tensor:
    """KL(p_teacher || p_student), top-K truncated with tail term (Eq. 11, Appendix A.3).

    Tail term:
        tail = (1 - Σ_K p_t) · [log(1 - Σ_K p_t) - log(1 - Σ_K p_s)]
    """
    student_logprobs = student_logprobs.float()
    teacher_logprobs = teacher_logprobs.float()
    student_probs = student_logprobs.exp()
    teacher_probs = teacher_logprobs.exp()

    s_tail = _tail_probs(student_probs)
    t_tail = _tail_probs(teacher_probs)

    per_token_kl = torch.einsum("btk,btk->bt", teacher_probs, teacher_logprobs - student_logprobs)
    per_token_kl = per_token_kl + t_tail * (t_tail.log() - s_tail.log())
    return _masked_token_mean(per_token_kl, response_mask, tis_weights, kl_clip)


def compute_jsd_loss(
    student_logprobs: torch.Tensor,         # [B, T, K]
    teacher_logprobs: torch.Tensor,         # [B, T, K]
    response_mask: torch.Tensor,            # [B, T]
    tis_weights: torch.Tensor | None = None,  # [B, T]
    kl_clip: float | None = None,
    jsd_alpha: float = 0.5,
) -> torch.Tensor:
    """JSD(student || teacher) with tail term (Eq. 11, Appendix A.3).

    jsd_alpha=0.5 → symmetric JSD (SDPO default).
    jsd_alpha=0.0 → forward KL.  jsd_alpha=1.0 → reverse KL.

    The mixture M is formed over both the top-K support and the tail bucket:
        M_tail = α · (1 - Σ_K p_s) + (1-α) · (1 - Σ_K p_t)
    """
    student_logprobs = student_logprobs.float()
    teacher_logprobs = teacher_logprobs.float()
    student_probs = student_logprobs.exp()
    teacher_probs = teacher_logprobs.exp()

    s_tail = _tail_probs(student_probs)                                   # [B, T]
    t_tail = _tail_probs(teacher_probs)

    M = jsd_alpha * student_probs + (1.0 - jsd_alpha) * teacher_probs    # [B, T, K]
    M_tail = jsd_alpha * s_tail + (1.0 - jsd_alpha) * t_tail            # [B, T]
    log_M = M.clamp(min=1e-8).log()
    log_M_tail = M_tail.log()

    kl_s = torch.einsum("btk,btk->bt", student_probs, student_logprobs - log_M)
    kl_s = kl_s + s_tail * (s_tail.log() - log_M_tail)

    kl_t = torch.einsum("btk,btk->bt", teacher_probs, teacher_logprobs - log_M)
    kl_t = kl_t + t_tail * (t_tail.log() - log_M_tail)

    per_token_jsd = jsd_alpha * kl_s + (1.0 - jsd_alpha) * kl_t
    return _masked_token_mean(per_token_jsd, response_mask, tis_weights, kl_clip)


ALGORITHMS = {
    "reverse_kl": compute_reverse_kl_loss,
    "forward_kl": compute_forward_kl_loss,
    "jsd": compute_jsd_loss,
}
