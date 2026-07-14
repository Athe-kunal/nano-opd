import torch

from opd.fsdp.algorithms import student_logprob_at_sampled_tokens


def compute_tis_weights(
    student_logits: torch.Tensor,           # [B, T, V]
    sampled_ids: torch.Tensor,              # [B, T]
    inf_lp_shifted: torch.Tensor | None,    # [B, T], vLLM inference log-probs
    tis_clip: float,
) -> torch.Tensor | None:
    """Per-token Truncated Importance Sampling (TIS) weight.

        w_t = exp(log π_train(y_t) − log π_vllm(y_t)), clipped to `tis_clip`.

    Corrects for the numerical gap between vLLM's inference-time log-probs
    and the training-time forward pass — without this correction, that gap
    silently biases the distillation gradient toward whichever direction the
    two kernels happen to disagree.

    Args:
        student_logits: Student logits at each position, `[B, T, V]`.
        sampled_ids: The sampled token id at each position, `[B, T]`.
        inf_lp_shifted: vLLM inference log-probs of the sampled tokens,
          `[B, T]`. Ignored (and may be `None`) when `tis_clip <= 0`.
        tis_clip: Clip bound C for the importance weight. `<= 0` disables
          TIS entirely (returns `None`).

    Returns:
        The per-token TIS weight, `[B, T]`, or `None` if `tis_clip <= 0`.
    """
    if tis_clip <= 0.0:
        return None
    s_lp_sampled = student_logprob_at_sampled_tokens(student_logits, sampled_ids)
    inf_lp_shifted = inf_lp_shifted.to(s_lp_sampled.dtype)
    return (s_lp_sampled - inf_lp_shifted).exp().clamp(max=tis_clip)


def _masked_token_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    tis_weights: torch.Tensor | None = None,
    kl_clip: float | None = None,
) -> torch.Tensor:
    """Averages per-token `values` over the response tokens marked in `mask`.

    Args:
        values: Per-token loss/divergence values, `[B, T]`.
        mask: Response mask, `[B, T]` (1 for response tokens, 0 for prompt).
        tis_weights: Optional per-token importance weights, `[B, T]`.
        kl_clip: If set, clamp `values` to this max before averaging.

    Returns:
        A scalar: the mask-weighted mean of `values`.
    """
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
    return (1.0 - torch.einsum("btk->bt", probs)).clamp(min=1e-8)  # [B, T]


def _student_teacher_probs(
    student_logprobs: torch.Tensor, teacher_logprobs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Casts top-K log-probs to float and exponentiates them, for both policies.

    Every top-K loss needs both the float log-probs (for the divergence's
    log-ratio term) and the plain probabilities (for the weighting term and,
    where applicable, `_tail_probs`).

    Returns:
        `(student_logprobs, teacher_logprobs, student_probs, teacher_probs)`.
    """
    student_logprobs = student_logprobs.float()
    teacher_logprobs = teacher_logprobs.float()
    return student_logprobs, teacher_logprobs, student_logprobs.exp(), teacher_logprobs.exp()


def compute_reverse_kl_loss(
    student_logprobs: torch.Tensor,         # [B, T, K]
    teacher_logprobs: torch.Tensor,         # [B, T, K]
    response_mask: torch.Tensor,            # [B, T]
    tis_weights: torch.Tensor | None = None,  # [B, T]
    kl_clip: float | None = None,
) -> torch.Tensor:
    """KL(p_student || p_teacher), top-K truncated with tail term.

    The tail term accounts for all probability mass outside the top-K support:
        tail = (1 - Σ_K p_s) · [log(1 - Σ_K p_s) - log(1 - Σ_K p_t)]
    This preserves the signal when the teacher wants to upweight a token the
    student ranked below K — exactly where feedback-driven corrections live.

    Args:
        student_logprobs: Student log-probs at the top-K indices, `[B, T, K]`.
        teacher_logprobs: Teacher log-probs at the same indices, `[B, T, K]`.
        response_mask: Response mask, `[B, T]`.
        tis_weights: Optional per-token TIS importance weights, `[B, T]`.
        kl_clip: If set, clamp each token's KL contribution to this max.

    Returns:
        A scalar loss, averaged over response tokens.
    """
    student_logprobs, teacher_logprobs, student_probs, teacher_probs = _student_teacher_probs(
        student_logprobs, teacher_logprobs
    )
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
    """KL(p_teacher || p_student), top-K truncated with tail term.

    Tail term:
        tail = (1 - Σ_K p_t) · [log(1 - Σ_K p_t) - log(1 - Σ_K p_s)]

    Args:
        student_logprobs: Student log-probs at the top-K indices, `[B, T, K]`.
        teacher_logprobs: Teacher log-probs at the same indices, `[B, T, K]`.
        response_mask: Response mask, `[B, T]`.
        tis_weights: Optional per-token TIS importance weights, `[B, T]`.
        kl_clip: If set, clamp each token's KL contribution to this max.

    Returns:
        A scalar loss, averaged over response tokens.
    """
    student_logprobs, teacher_logprobs, student_probs, teacher_probs = _student_teacher_probs(
        student_logprobs, teacher_logprobs
    )
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
    """JSD(student || teacher) with tail term.

    jsd_alpha=0.5 → symmetric JSD (SDPO default).
    jsd_alpha=0.0 → forward KL.  jsd_alpha=1.0 → reverse KL.

    The mixture M is formed over both the top-K support and the tail bucket:
        M_tail = α · (1 - Σ_K p_s) + (1-α) · (1 - Σ_K p_t)

    Args:
        student_logprobs: Student log-probs at the top-K indices, `[B, T, K]`.
        teacher_logprobs: Teacher log-probs at the same indices, `[B, T, K]`.
        response_mask: Response mask, `[B, T]`.
        tis_weights: Optional per-token TIS importance weights, `[B, T]`.
        kl_clip: If set, clamp each token's JSD contribution to this max.
        jsd_alpha: Mixture weight on the student distribution.

    Returns:
        A scalar loss, averaged over response tokens.
    """
    student_logprobs, teacher_logprobs, student_probs, teacher_probs = _student_teacher_probs(
        student_logprobs, teacher_logprobs
    )
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


def compute_mopd_loss(
    student_logprobs: torch.Tensor,         # [B, T, K]
    teacher_logprobs: torch.Tensor,         # [B, T, K]
    response_mask: torch.Tensor,            # [B, T]
    tis_weights: torch.Tensor | None = None,  # [B, T]
    kl_clip: float | None = None,
) -> torch.Tensor:
    """Top-K MOPD loss: generalized (unnormalized) reverse KL.

    Reference: https://arxiv.org/pdf/2606.30406
    MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training

    Teacher selects the top-K indices (see select_topk_by="teacher" wiring).
    Per-token: Σ_k p_s·(log p_s − log p_t) − p_s + p_t.
    The −p_s + p_t term makes this a Bregman divergence in p_s (min at p_s=p_t
    pointwise, for any k independently) so, unlike plain top-K reverse KL, no
    tail/renormalization correction is needed — the K-token sum alone is
    already unbiased.

    Args:
        student_logprobs: Student log-probs at the top-K indices, `[B, T, K]`.
        teacher_logprobs: Teacher log-probs at the same indices, `[B, T, K]`.
        response_mask: Response mask, `[B, T]`.
        tis_weights: Optional per-token TIS importance weights, `[B, T]`.
        kl_clip: If set, clamp each token's contribution to this max.

    Returns:
        A scalar loss, averaged over response tokens.
    """
    student_logprobs, teacher_logprobs, student_probs, teacher_probs = _student_teacher_probs(
        student_logprobs, teacher_logprobs
    )
    per_token = torch.einsum("btk,btk->bt", student_probs, student_logprobs - teacher_logprobs)
    per_token = per_token - torch.einsum("btk->bt", student_probs) + torch.einsum("btk->bt", teacher_probs)
    return _masked_token_mean(per_token, response_mask, tis_weights, kl_clip)


def compute_mopd_pg_loss(
    student_logprob: torch.Tensor,          # [B, T], grad-carrying log π_θ(y_t)
    teacher_logprob: torch.Tensor,          # [B, T], no-grad log π_φd(y_t)
    response_mask: torch.Tensor,            # [B, T]
    tis_weights: torch.Tensor | None = None,  # [B, T]
    adv_clip: float | None = 5.0,
) -> torch.Tensor:
    """Policy-gradient MOPD loss on the sampled token only.

    No top-K exchange needed at all — only the sampled token's log-prob under
    each policy. The teacher–student log-diff is a stop-gradient advantage;
    it reweights −log π_θ(y_t) (standard NLL) instead of appearing inside a
    divergence. Two-sided clip (A_max=5 in the paper) bounds how much a single
    token can dominate the gradient, same motivation as kl_clip elsewhere.

    Args:
        student_logprob: Grad-carrying student log-prob of the sampled
          token, `[B, T]`.
        teacher_logprob: No-grad teacher log-prob of the same sampled
          token, `[B, T]`.
        response_mask: Response mask, `[B, T]`.
        tis_weights: Optional per-token TIS importance weights, `[B, T]`.
        adv_clip: Two-sided clip bound for the stop-gradient advantage.

    Returns:
        A scalar loss, averaged over response tokens.
    """
    student_logprob = student_logprob.float()
    teacher_logprob = teacher_logprob.float().detach()

    advantage = (teacher_logprob - student_logprob.detach())
    if adv_clip is not None:
        advantage = advantage.clamp(min=-adv_clip, max=adv_clip)

    per_token = -advantage * student_logprob
    return _masked_token_mean(per_token, response_mask, tis_weights, kl_clip=None)


ALGORITHMS = {
    "reverse_kl": compute_reverse_kl_loss,
    "forward_kl": compute_forward_kl_loss,
    "jsd": compute_jsd_loss,
    "mopd_loss": compute_mopd_loss,
    "mopd_pg_loss": compute_mopd_pg_loss,
}
