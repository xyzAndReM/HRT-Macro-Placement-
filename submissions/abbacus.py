"""
Abbacus / Abacus-style hard-macro legalization.

The original **Abacus** algorithm row-legalizes standard cells by visiting macros in
order and placing each at the minimum-displacement feasible site along the row.
Hard macros here live on a 2D canvas with varying widths/heights, so we adapt the
same idea:

1. Sort **movable** hard macros by **decreasing area** (large macros first).
2. **Gauss–Seidel sweeps**: for each macro in that order, repeatedly separate it from
   any overlapping partner by shifting **only that macro** along the cheaper axis
   (minimum overlap depth), then clamp to the canvas.
3. A final **pairwise residual push** (shared with ``submissions/qp.py``) with
   ``alpha=0.3`` (default) when single-sided pushes stagnate.
4. If overlaps remain, one **cooperative Gauss–Seidel** sweep: both partners move
   **50/50** along the cheaper separation axis (trapped clusters).

No QP/cvxpy — only NumPy. Optional gap and iteration caps via env (see below).

Env:
    MACRO_PLACE_ABBACUS_GAP — minimum separation margin (default: same as QP, ``0.005`` µm)
    MACRO_PLACE_ABBACUS_OUTER — max outer sweeps (default ``240``)
    MACRO_PLACE_ABBACUS_INNER — max inner separation attempts per macro per sweep (default ``120``)
    MACRO_PLACE_ABBACUS_FALLBACK_ITERS — fallback push iterations (default ``max(12000, 50×n_hard)``)
    MACRO_PLACE_ABBACUS_FALLBACK_ALPHA — fallback step scale (default ``0.3``)
    MACRO_PLACE_ABBACUS_STAGNANT_OUTER — consecutive outer sweeps with no overlap shrink before giving up
       on the GS phase (default ``16``); fallback push still runs afterward

Usage:
    uv run evaluate submissions/abbacus.py -b ibm01
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.qp import (  # noqa: E402
    _GAP,
    _detect_overlaps,
    _fallback_pairwise_push,
    _overlaps_xy,
)

_DEFAULT_ABBACUS_OUTER = 240
_DEFAULT_ABBACUS_INNER = 120
# Quit outer GS early only after this many consecutive sweeps with no overlap-count reduction.
_DEFAULT_ABBACUS_STAGNANT_OUTER = 16


def _fallback_iters_default(n_hard: int) -> int:
    """``max(12000, 50×n_hard)`` — higher floor and scale for tight benchmarks (e.g. ibm06)."""
    return max(12000, 50 * max(1, int(n_hard)))


def _sep_push_i_only(
    i: int,
    j: int,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    cw: float,
    ch: float,
    gap: float,
) -> bool:
    """Move macro i away from j along the smaller overlap depth; returns True if i moved."""
    dx_need = (w[i] + w[j]) * 0.5 + gap - abs(x[i] - x[j])
    dy_need = (h[i] + h[j]) * 0.5 + gap - abs(y[i] - y[j])
    if dx_need <= 0.0 or dy_need <= 0.0:
        return False
    if dx_need <= dy_need:
        sign = 1.0 if x[i] >= x[j] else -1.0
        x[i] = min(max(x[i] + sign * dx_need, w[i] * 0.5), cw - w[i] * 0.5)
    else:
        sign = 1.0 if y[i] >= y[j] else -1.0
        y[i] = min(max(y[i] + sign * dy_need, h[i] * 0.5), ch - h[i] * 0.5)
    return True


def _sep_push_cooperative(
    i: int,
    j: int,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    fixed: np.ndarray,
    cw: float,
    ch: float,
    gap: float,
) -> bool:
    """Separate i and j along the smaller overlap depth; movable macros share the push 50/50."""
    dx_need = (w[i] + w[j]) * 0.5 + gap - abs(x[i] - x[j])
    dy_need = (h[i] + h[j]) * 0.5 + gap - abs(y[i] - y[j])
    if dx_need <= 0.0 or dy_need <= 0.0:
        return False
    if fixed[i] and fixed[j]:
        return False

    moved = False
    if dx_need <= dy_need:
        sign = 1.0 if x[i] >= x[j] else -1.0
        if fixed[i]:
            x[j] = min(max(x[j] - sign * dx_need, w[j] * 0.5), cw - w[j] * 0.5)
        elif fixed[j]:
            x[i] = min(max(x[i] + sign * dx_need, w[i] * 0.5), cw - w[i] * 0.5)
        else:
            half = 0.5 * dx_need
            x[i] = min(max(x[i] + sign * half, w[i] * 0.5), cw - w[i] * 0.5)
            x[j] = min(max(x[j] - sign * half, w[j] * 0.5), cw - w[j] * 0.5)
        moved = True
    else:
        sign = 1.0 if y[i] >= y[j] else -1.0
        if fixed[i]:
            y[j] = min(max(y[j] - sign * dy_need, h[j] * 0.5), ch - h[j] * 0.5)
        elif fixed[j]:
            y[i] = min(max(y[i] + sign * dy_need, h[i] * 0.5), ch - h[i] * 0.5)
        else:
            half = 0.5 * dy_need
            y[i] = min(max(y[i] + sign * half, h[i] * 0.5), ch - h[i] * 0.5)
            y[j] = min(max(y[j] - sign * half, h[j] * 0.5), ch - h[j] * 0.5)
        moved = True
    return moved


def _cooperative_gs_sweep(
    order: list[int],
    n_hard: int,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    fixed: np.ndarray,
    cw: float,
    ch: float,
    gap: float,
    bucket: float,
    max_inner: int,
) -> None:
    """One Gauss–Seidel pass with 50/50 two-sided separation (post-fallback cleanup)."""
    if _detect_overlaps(x, y, w, h, gap, bucket):
        for i in order:
            if fixed[i]:
                continue
            for _ in range(max_inner):
                partner: int | None = None
                for j in range(n_hard):
                    if j == i:
                        continue
                    if _overlaps_xy(x[i], y[i], w[i], h[i], x[j], y[j], w[j], h[j], gap):
                        partner = j
                        break
                if partner is None:
                    break
                _sep_push_cooperative(i, partner, x, y, w, h, fixed, cw, ch, gap)


class AbbacusLegalizer:
    """
    Abacus-style iterative legalization for hard macros.

    API matches ``QPLegalizer.place`` so you can swap implementations in pipelines.
    """

    def place(
        self,
        benchmark: Benchmark,
        *,
        initial_macro_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if initial_macro_positions is not None:
            placement = initial_macro_positions.to(
                device=benchmark.macro_positions.device,
                dtype=benchmark.macro_positions.dtype,
            ).clone()
        else:
            placement = benchmark.macro_positions.clone()

        n_hard = int(benchmark.num_hard_macros)
        if n_hard <= 1:
            return placement

        gap = float(os.environ.get("MACRO_PLACE_ABBACUS_GAP", str(_GAP)) or _GAP)
        max_outer = max(
            1,
            int(
                os.environ.get(
                    "MACRO_PLACE_ABBACUS_OUTER", str(_DEFAULT_ABBACUS_OUTER)
                )
                or str(_DEFAULT_ABBACUS_OUTER)
            ),
        )
        max_inner = max(
            1,
            int(
                os.environ.get(
                    "MACRO_PLACE_ABBACUS_INNER", str(_DEFAULT_ABBACUS_INNER)
                )
                or str(_DEFAULT_ABBACUS_INNER)
            ),
        )
        stagnant_cap = max(
            1,
            int(
                os.environ.get(
                    "MACRO_PLACE_ABBACUS_STAGNANT_OUTER",
                    str(_DEFAULT_ABBACUS_STAGNANT_OUTER),
                )
                or str(_DEFAULT_ABBACUS_STAGNANT_OUTER)
            ),
        )
        fb_raw = (os.environ.get("MACRO_PLACE_ABBACUS_FALLBACK_ITERS") or "").strip()
        fb_iters = max(
            1,
            int(fb_raw) if fb_raw else _fallback_iters_default(n_hard),
        )
        fb_alpha = float(os.environ.get("MACRO_PLACE_ABBACUS_FALLBACK_ALPHA", "0.3") or "0.3")

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        grid_rows = max(int(benchmark.grid_rows), 1)
        grid_cols = max(int(benchmark.grid_cols), 1)
        cell_w = cw / grid_cols
        cell_h = ch / grid_rows
        bucket = max(cell_w, cell_h)

        x = placement[:n_hard, 0].detach().cpu().numpy().astype(np.float64)
        y = placement[:n_hard, 1].detach().cpu().numpy().astype(np.float64)
        w = benchmark.macro_sizes[:n_hard, 0].detach().cpu().numpy().astype(np.float64)
        h = benchmark.macro_sizes[:n_hard, 1].detach().cpu().numpy().astype(np.float64)
        fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)

        movable_ids = [i for i in range(n_hard) if not fixed[i]]
        if not movable_ids:
            return placement

        # Decreasing area — mirrors “difficult / large objects first” in row Abacus.
        order = sorted(movable_ids, key=lambda i: -(w[i] * h[i]))

        stagnant = 0
        for _ in range(max_outer):
            ov_list = _detect_overlaps(x, y, w, h, gap, bucket)
            if not ov_list:
                break
            before = len(ov_list)

            for i in order:
                if fixed[i]:
                    continue
                for _ in range(max_inner):
                    partner: int | None = None
                    for j in range(n_hard):
                        if j == i:
                            continue
                        if _overlaps_xy(x[i], y[i], w[i], h[i], x[j], y[j], w[j], h[j], gap):
                            partner = j
                            break
                    if partner is None:
                        break
                    _sep_push_i_only(i, partner, x, y, w, h, cw, ch, gap)

            after = len(_detect_overlaps(x, y, w, h, gap, bucket))
            if after >= before:
                stagnant += 1
                if stagnant >= stagnant_cap:
                    break
            else:
                stagnant = 0

        if _detect_overlaps(x, y, w, h, gap, bucket):
            _fallback_pairwise_push(
                x,
                y,
                w,
                h,
                fixed,
                cw,
                ch,
                bucket,
                max_iters=fb_iters,
                alpha=fb_alpha,
                gap=gap,
            )

        if _detect_overlaps(x, y, w, h, gap, bucket):
            _cooperative_gs_sweep(
                order,
                n_hard,
                x,
                y,
                w,
                h,
                fixed,
                cw,
                ch,
                gap,
                bucket,
                max_inner,
            )

        for i in range(n_hard):
            placement[i, 0] = float(x[i])
            placement[i, 1] = float(y[i])

        return placement


if __name__ == "__main__":
    from macro_place.loader import load_benchmark_from_dir

    root = _ROOT
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    AbbacusLegalizer().place(b)
