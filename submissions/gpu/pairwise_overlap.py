"""
Optional Triton-accelerated **forward** for normalized pairwise macro overlap area.

The GpuPlacer overlap term is O(n²) in the number of macros. When Triton is available
and ``MACRO_PLACE_GPU_OVERLAP_TRITON=1``, the forward pass uses a small fused kernel
(one program per eligible pair, atomic add into a scalar).

**Backward** still uses the original PyTorch graph (recomputed inside
``torch.autograd.Function.backward``) so training remains correct without hand-derived
subgradients through ReLU kinks.

Windows: install a Triton build compatible with your PyTorch/CUDA stack, e.g.
``uv pip install -U triton-windows`` (third-party; YMMV vs Linux ``triton`` wheels).

Env:
    MACRO_PLACE_GPU_OVERLAP_TRITON — ``1``/``true`` to enable when CUDA + float32 + Triton import succeeds.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

_TRITON_WARNED = False


def _env_overlap_triton_requested() -> bool:
    raw = (os.environ.get("MACRO_PLACE_GPU_OVERLAP_TRITON") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def overlap_triton_available() -> bool:
    """True if Triton can be imported (Linux wheels or ``triton-windows``, etc.)."""
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def want_triton_overlap(device: torch.device, dtype: torch.dtype) -> bool:
    """CUDA float32 only — keeps first kernel version small and numerically predictable."""
    if not _env_overlap_triton_requested():
        return False
    if device.type != "cuda":
        return False
    if dtype != torch.float32:
        return False
    if not overlap_triton_available():
        global _TRITON_WARNED
        if not _TRITON_WARNED:
            print(
                "[gpu_pairwise_overlap] MACRO_PLACE_GPU_OVERLAP_TRITON=1 but triton "
                "is not installed; using PyTorch overlap. "
                "(Linux: triton via PyTorch deps; Windows: e.g. uv pip install -U triton-windows)",
                flush=True,
            )
            _TRITON_WARNED = True
        return False
    return True


def build_overlap_pair_indices(
    fixed_mask: torch.Tensor,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Upper-triangle pairs (i,j), excluding fixed–fixed pairs."""
    n = int(fixed_mask.shape[0])
    ii: list[int] = []
    jj: list[int] = []
    fm = fixed_mask.detach()
    for i in range(n):
        fi = bool(fm[i].item())
        for j in range(i + 1, n):
            if fi and bool(fm[j].item()):
                continue
            ii.append(i)
            jj.append(j)
    if not ii:
        z = torch.zeros(0, dtype=torch.int32, device=device)
        return z, z
    return (
        torch.tensor(ii, dtype=torch.int32, device=device),
        torch.tensor(jj, dtype=torch.int32, device=device),
    )


def pairwise_overlap_sum_normalized_torch(
    full_pos: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    eps_canvas: torch.Tensor,
) -> torch.Tensor:
    """Soft overlap penalty: sum of rectangle intersection areas over i<j / canvas area."""
    if full_pos.shape[0] <= 1:
        return torch.zeros((), device=full_pos.device, dtype=full_pos.dtype)
    cx = full_pos[:, 0]
    cy = full_pos[:, 1]
    w = sizes[:, 0]
    h = sizes[:, 1]
    lx = cx - 0.5 * w
    rx = cx + 0.5 * w
    by = cy - 0.5 * h
    ty = cy + 0.5 * h

    ix = torch.relu(torch.minimum(rx[:, None], rx[None, :]) - torch.maximum(lx[:, None], lx[None, :]))
    iy = torch.relu(torch.minimum(ty[:, None], ty[None, :]) - torch.maximum(by[:, None], by[None, :]))
    area = ix * iy
    pair = torch.triu(area, diagonal=1)

    if fixed_mask.numel() == full_pos.shape[0]:
        fixed_pair = (fixed_mask[:, None] & fixed_mask[None, :]).to(pair.dtype)
        pair = pair * (1.0 - torch.triu(fixed_pair, diagonal=1))

    return pair.sum() / eps_canvas


def _launch_triton_overlap_fwd(
    cx: torch.Tensor,
    cy: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    inv_eps: float,
) -> torch.Tensor:
    import triton
    import triton.language as tl

    n_pairs = int(pair_i.shape[0])
    out = torch.zeros(1, device=cx.device, dtype=torch.float32)

    if n_pairs == 0:
        return out.to(dtype=cx.dtype)

    @triton.jit
    def _kernel(
        cx_ptr,
        cy_ptr,
        w_ptr,
        h_ptr,
        pi_ptr,
        pj_ptr,
        out_ptr,
        inv_eps,
    ):
        pid = tl.program_id(0)
        i = tl.load(pi_ptr + pid)
        j = tl.load(pj_ptr + pid)

        cxi = tl.load(cx_ptr + i)
        cxj = tl.load(cx_ptr + j)
        cyi = tl.load(cy_ptr + i)
        cyj = tl.load(cy_ptr + j)
        wi = tl.load(w_ptr + i)
        wj = tl.load(w_ptr + j)
        hi = tl.load(h_ptr + i)
        hj = tl.load(h_ptr + j)

        lx_i = cxi - 0.5 * wi
        rx_i = cxi + 0.5 * wi
        lx_j = cxj - 0.5 * wj
        rx_j = cxj + 0.5 * wj
        by_i = cyi - 0.5 * hi
        ty_i = cyi + 0.5 * hi
        by_j = cyj - 0.5 * hj
        ty_j = cyj + 0.5 * hj

        ix_max = tl.minimum(rx_i, rx_j) - tl.maximum(lx_i, lx_j)
        iy_max = tl.minimum(ty_i, ty_j) - tl.maximum(by_i, by_j)
        ix = tl.maximum(ix_max, 0.0)
        iy = tl.maximum(iy_max, 0.0)
        area = ix * iy
        contrib = area * inv_eps
        tl.atomic_add(out_ptr, contrib)

    grid = (n_pairs,)
    _kernel[grid](
        cx,
        cy,
        w,
        h,
        pair_i,
        pair_j,
        out,
        float(inv_eps),
    )
    return out.view(()).to(dtype=cx.dtype)


class _PairwiseOverlapNorm(torch.autograd.Function):
    """Triton forward + PyTorch backward (recompute analytic torch graph)."""

    @staticmethod
    def forward(
        ctx,
        full_pos: torch.Tensor,
        sizes: torch.Tensor,
        fixed_mask: torch.Tensor,
        eps_canvas: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(full_pos, sizes)
        ctx.fixed_mask = fixed_mask
        ctx.eps_canvas = eps_canvas

        cx = full_pos[:, 0].contiguous()
        cy = full_pos[:, 1].contiguous()
        w = sizes[:, 0].contiguous()
        h = sizes[:, 1].contiguous()
        inv_eps = float((1.0 / eps_canvas).item())

        if pair_i.numel() == 0:
            return torch.zeros((), device=full_pos.device, dtype=full_pos.dtype)

        out = _launch_triton_overlap_fwd(cx, cy, w, h, pair_i, pair_j, inv_eps)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        full_pos, sizes = ctx.saved_tensors
        fixed_mask = ctx.fixed_mask
        eps_canvas = ctx.eps_canvas

        # Recompute smooth torch overlap for correct autograd through movable macros.
        fp = full_pos.detach().requires_grad_(True)
        y = pairwise_overlap_sum_normalized_torch(fp, sizes, fixed_mask, eps_canvas)
        g_fp, = torch.autograd.grad(y, fp, grad_outputs=grad_out, retain_graph=False)
        return g_fp, None, None, None, None, None


def pairwise_overlap_sum_normalized(
    full_pos: torch.Tensor,
    sizes: torch.Tensor,
    fixed_mask: torch.Tensor,
    eps_canvas: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    *,
    use_triton: bool,
) -> torch.Tensor:
    """Normalized pairwise overlap scalar; Triton path uses custom autograd."""
    if not use_triton:
        return pairwise_overlap_sum_normalized_torch(full_pos, sizes, fixed_mask, eps_canvas)
    return _PairwiseOverlapNorm.apply(full_pos, sizes, fixed_mask, eps_canvas, pair_i, pair_j)
