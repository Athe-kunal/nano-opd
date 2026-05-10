import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch.distributed.device_mesh import DeviceMesh
from typing import Literal, Optional


def _distributed_logsumexp(
    logits: torch.Tensor, process_group: dist.ProcessGroup, chunk_size: int = 1024,
) -> torch.Tensor:
    parts = []
    for start in range(0, logits.shape[1], chunk_size):
        local = torch.logsumexp(logits[:, start:start + chunk_size], dim=-1)
        gathered = dist_nn.all_gather(local, group=process_group)
        parts.append(torch.logsumexp(torch.stack(gathered, dim=-1), dim=-1))
    return torch.cat(parts, dim=1)


def _topk_logprobs_at_indices(
    logits: torch.Tensor, lse: torch.Tensor, topk_idx_global: torch.Tensor,
    process_group: dist.ProcessGroup, tp_rank: int,
    requires_grad: bool,
) -> torch.Tensor:
    V_shard = logits.shape[-1]
    shard_offset = tp_rank * V_shard
    owner_rank = topk_idx_global // V_shard
    local_idx = topk_idx_global - shard_offset
    on_this_rank = (owner_rank == tp_rank)
    safe_idx = torch.where(on_this_rank, local_idx, torch.zeros_like(local_idx))
    vals = logits.gather(-1, safe_idx) * on_this_rank
    if requires_grad:
        vals = dist_nn.all_reduce(vals, group=process_group)
    else:
        dist.all_reduce(vals, group=process_group)
    return vals - lse.unsqueeze(-1)


def _select_topk_global(
    logits: torch.Tensor,         # [B, T, V_shard]
    pg: dist.ProcessGroup,
    tp_size: int,
    tp_rank: int,
    top_k: int,
) -> torch.Tensor:
    """Select global top-K vocab indices from TP-sharded logits. Returns [B, T, K] int64."""
    V_shard = logits.shape[-1]
    shard_offset = tp_rank * V_shard
    k_local = min(top_k, V_shard)

    local_topk_vals, local_topk_idx = logits.topk(k_local, dim=-1)
    local_topk_idx_global = local_topk_idx + shard_offset

    vals_parts = [torch.empty_like(local_topk_vals) for _ in range(tp_size)]
    dist.all_gather(vals_parts, local_topk_vals, group=pg)
    idx_parts = [torch.empty_like(local_topk_idx_global) for _ in range(tp_size)]
    dist.all_gather(idx_parts, local_topk_idx_global, group=pg)

    cand_vals = torch.cat(vals_parts, dim=-1)
    cand_idx = torch.cat(idx_parts, dim=-1)
    _, sel = cand_vals.topk(top_k, dim=-1)
    return cand_idx.gather(-1, sel)


def compute_topk_logprobs_for_distillation(
    student_logits: torch.Tensor,   # [B, T, V]
    teacher_logits: torch.Tensor,   # [B, T, V]
    top_k: int = 100,
    chunk_size: int = 1024,
    select_topk_by: Literal["student", "teacher"] = "student",
):
    """
    Returns:
        s_topk_logprobs: [B, T, K]
        t_topk_logprobs: [B, T, K]
        topk_idx:        [B, T, K]
    """
    s_lse = torch.logsumexp(student_logits, dim=-1)         # [B, T]
    with torch.no_grad():
        t_lse = torch.logsumexp(teacher_logits, dim=-1)     # [B, T]

    if select_topk_by == "student":
        _, topk_idx = student_logits.topk(top_k, dim=-1)   # [B, T, K]
    else:
        with torch.no_grad():
            _, topk_idx = teacher_logits.topk(top_k, dim=-1)

    s_topk_logprobs = student_logits.gather(-1, topk_idx) - s_lse.unsqueeze(-1)
    with torch.no_grad():
        t_topk_logprobs = teacher_logits.gather(-1, topk_idx) - t_lse.unsqueeze(-1)

    return s_topk_logprobs, t_topk_logprobs, topk_idx


def compute_topk_logprobs_for_distillation_tp(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_mesh: DeviceMesh,
    teacher_mesh: DeviceMesh,
    top_k: int = 100,
    chunk_size: int = 1024,
    select_topk_by: Literal["student", "teacher"] = "student",
    bridge_broadcast: Optional[dict] = None,
):
    """
    Args:
        select_topk_by:
            "student" — top-K by student probs. Use for REVERSE KL.
            "teacher" — top-K by teacher probs. Use for FORWARD KL.
        bridge_broadcast: optional dict for cross-mesh index handoff when student
            and teacher live on disjoint ranks. Keys:
                "src_rank":        global rank that holds student-selected indices
                "src_rank_teacher" global rank that holds teacher-selected indices
                "group":           a process group spanning all student+teacher ranks
                "is_src":          bool, True on the student source rank
                "is_src_teacher":  bool, True on the teacher source rank
                "shape":           index tensor shape [B, T, K]
                "device":          device for pre-allocated receive buffers
            If None, assumes co-located meshes (every rank already participates
            in both student and teacher process groups).

    Returns:
        s_topk_logprobs: [B, T, K]
        t_topk_logprobs: [B, T, K]
        topk_idx_global: [B, T, K]
    """
    s_tp = student_mesh["tp"]
    t_tp = teacher_mesh["tp"]
    s_pg, s_size, s_rank = s_tp.get_group(), s_tp.size(), s_tp.get_local_rank()
    t_pg, t_size, t_rank = t_tp.get_group(), t_tp.size(), t_tp.get_local_rank()

    is_student_rank = student_logits.numel() > 0
    is_teacher_rank = teacher_logits.numel() > 0

    # ---- 1. Full-vocab logsumexp on each mesh.
    s_lse = _distributed_logsumexp(student_logits, s_pg, chunk_size) if is_student_rank else None
    with torch.no_grad():
        t_lse = _distributed_logsumexp(teacher_logits, t_pg, chunk_size) if is_teacher_rank else None

    # ---- 2. Select global top-K indices on the appropriate mesh(es).
    if select_topk_by == "student" and is_student_rank:
        topk_idx_global = _select_topk_global(student_logits, s_pg, s_size, s_rank, top_k)
    else:
        with torch.no_grad():
            topk_idx_global = _select_topk_global(teacher_logits, t_pg, t_size, t_rank, top_k) if is_teacher_rank else None

    # ---- 3. Cross-mesh broadcast of indices if needed.
    if bridge_broadcast is not None:
        bg = bridge_broadcast["group"]
        if select_topk_by == "student":
            buf = topk_idx_global if bridge_broadcast["is_src"] else torch.empty(
                bridge_broadcast["shape"], dtype=torch.int64,
                device=bridge_broadcast["device"],
            )
            dist.broadcast(buf, src=bridge_broadcast["src_rank"], group=bg)
            if not is_student_rank:
                topk_idx_global = buf
        else:
            buf = topk_idx_global if bridge_broadcast.get("is_src_teacher", False) else torch.empty(
                bridge_broadcast["shape"], dtype=torch.int64,
                device=bridge_broadcast["device"],
            )
            dist.broadcast(buf, src=bridge_broadcast["src_rank_teacher"], group=bg)
            if not is_teacher_rank:
                topk_idx_global = buf

    # ---- 4. Look up log-probs on each mesh.
    s_topk_logprobs = _topk_logprobs_at_indices(
        student_logits, s_lse, topk_idx_global, s_pg, s_rank,
        requires_grad=True,
    ) if is_student_rank else None

    with torch.no_grad():
        t_topk_logprobs = _topk_logprobs_at_indices(
            teacher_logits, t_lse, topk_idx_global, t_pg, t_rank,
            requires_grad=False,
        ) if is_teacher_rank else None

    return s_topk_logprobs, t_topk_logprobs, topk_idx_global