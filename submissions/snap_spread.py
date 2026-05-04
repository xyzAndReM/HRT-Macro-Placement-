"""
Snap-spread placer — snap to routing-grid cell centers with margin-aware legality.

Candidate sites are the **centers** of the benchmark placement grid
(``grid_cols`` × ``grid_rows``): ``((c+0.5) * cell_w, (r+0.5) * cell_h)``.

For each movable hard macro (positional order: top-left → bottom-right, y up), we
snap to the **Euclidean-nearest** legal routing-grid center. Candidates are all
cell centers in that macro’s valid index rectangle on ``grid_cols`` × ``grid_rows``
(typically ~1–2k sites): the same set you would visit by **expanding Chebyshev
radius** from the seed indices nearest the macro, without rescanning the
rectangle per ring.

A site is **available** if the macro stays on-chip and has at least
``inter_macro_margin`` edge clearance vs fixed macros and movables already
snapped.

Soft macros are unchanged.

Usage:
    uv run evaluate submissions/snap_spread.py -b ibm01
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch

from macro_place.benchmark import Benchmark

_GAP = 0.001


def _sep_x(w0: float, w1: float, margin: float) -> float:
    return 0.5 * (w0 + w1) + margin + _GAP


def _sep_y(h0: float, h1: float, margin: float) -> float:
    return 0.5 * (h0 + h1) + margin + _GAP


def _overlap_pair(
    cx: float,
    cy: float,
    w: float,
    h: float,
    ox: float,
    oy: float,
    ow: float,
    oh: float,
    margin: float,
) -> bool:
    dx = abs(cx - ox)
    dy = abs(cy - oy)
    return dx < _sep_x(w, ow, margin) and dy < _sep_y(h, oh, margin)


def _valid_cr_bounds(
    w: float,
    h: float,
    cw: float,
    ch: float,
    cell_w: float,
    cell_h: float,
    nc: int,
    nr: int,
) -> Tuple[int, int, int, int]:
    """Inclusive index ranges (c, r) whose cell center fits macro (w,h) on canvas."""
    c_lo = max(0, int(math.ceil(w / (2.0 * cell_w) - 0.5 - 1e-9)))
    c_hi = min(nc - 1, int(math.floor((cw - w / 2.0) / cell_w - 0.5 + 1e-9)))
    r_lo = max(0, int(math.ceil(h / (2.0 * cell_h) - 0.5 - 1e-9)))
    r_hi = min(nr - 1, int(math.floor((ch - h / 2.0) / cell_h - 0.5 + 1e-9)))
    if c_lo > c_hi or r_lo > r_hi:
        return 0, -1, 0, -1
    return c_lo, c_hi, r_lo, r_hi


def _center_xy(c: int, r: int, cell_w: float, cell_h: float) -> Tuple[float, float]:
    return (c + 0.5) * cell_w, (r + 0.5) * cell_h


class SnapSpreadPlacer:
    """
    Snap movable hard macros to routing-grid centers; ring search, nearest legal.

    Args:
        inter_macro_margin: Minimum extra edge-to-edge gap (μm) between hard macros.
    """

    def __init__(self, inter_macro_margin: float = 0.08):
        self.inter_macro_margin = float(inter_macro_margin)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        n_hard = int(benchmark.num_hard_macros)
        if n_hard == 0:
            return placement

        margin = self.inter_macro_margin
        nc = max(int(benchmark.grid_cols), 1)
        nr = max(int(benchmark.grid_rows), 1)
        cell_w = cw / nc
        cell_h = ch / nr

        movable_mask = benchmark.get_movable_mask()[:n_hard] & benchmark.get_hard_macro_mask()[
            :n_hard
        ]
        fixed = benchmark.macro_fixed[:n_hard]

        sizes = [
            (float(benchmark.macro_sizes[i, 0]), float(benchmark.macro_sizes[i, 1]))
            for i in range(n_hard)
        ]

        obstacles: List[Tuple[float, float, float, float]] = []
        for i in range(n_hard):
            if bool(fixed[i].item()):
                ox, oy = float(placement[i, 0]), float(placement[i, 1])
                w, h = sizes[i]
                obstacles.append((ox, oy, w, h))

        movable_idx = [i for i in range(n_hard) if bool(movable_mask[i].item())]
        movable_idx.sort(key=lambda i: (-placement[i, 1].item(), placement[i, 0].item()))

        for i in movable_idx:
            w, h = sizes[i]
            ox, oy = float(placement[i, 0]), float(placement[i, 1])

            c_lo, c_hi, r_lo, r_hi = _valid_cr_bounds(w, h, cw, ch, cell_w, cell_h, nc, nr)
            if c_lo > c_hi or r_lo > r_hi:
                continue

            best: Tuple[float, float] | None = None
            best_d2 = float("inf")

            for c in range(c_lo, c_hi + 1):
                for r in range(r_lo, r_hi + 1):
                    cx, cy = _center_xy(c, r, cell_w, cell_h)
                    if cx < w * 0.5 - 1e-9 or cx > cw - w * 0.5 + 1e-9:
                        continue
                    if cy < h * 0.5 - 1e-9 or cy > ch - h * 0.5 + 1e-9:
                        continue
                    bad = False
                    for ox2, oy2, ow, oh2 in obstacles:
                        if _overlap_pair(cx, cy, w, h, ox2, oy2, ow, oh2, margin):
                            bad = True
                            break
                    if bad:
                        continue
                    d2 = (cx - ox) ** 2 + (cy - oy) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (cx, cy)

            if best is not None:
                cx, cy = best
                placement[i, 0] = cx
                placement[i, 1] = cy
                obstacles.append((cx, cy, w, h))

        return placement
