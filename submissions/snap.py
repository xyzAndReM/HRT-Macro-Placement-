"""
Grid snap placer — snaps movable hard macros onto the routing grid and legalizes overlaps.

Multi-resolution legalization:

- For each lattice resolution (1×, 2×, 4×, 8× the routing grid), we try to place every
  *remaining* movable hard macro at the **nearest free cell center** to its original
  position (Manhattan-ring expansion, Euclidean tie-breaks within the ring).
- Placement is **carried forward** across resolutions: macros placed at a coarser lattice
  stay where they are, and become obstacles for finer-lattice retries. Only macros that
  could not be placed are retried with more candidate cells.
- Soft macros are unchanged. Fixed macros are obstacles at their original positions.

If after the finest lattice some macros still cannot be placed, they are left at their
original positions (which may overlap). Use a stronger fallback if that becomes a problem.

Usage:
    uv run evaluate submissions/snap.py
    uv run evaluate submissions/snap.py -b ibm03
    uv run evaluate submissions/snap.py --all
"""

from __future__ import annotations

import math
import torch

from macro_place.benchmark import Benchmark

# Match greedy_row_placer / validator tolerance so touching edges are not false overlaps
_GAP = 0.001

_LATTICE_MULTIPLIERS = (1, 2, 4, 8)

# Per-macro search caps. A Manhattan ring of radius d contains 4*d cells, so cumulative
# candidates grow as O(d^2). We cap both the radius and the number of candidates examined
# so that hard-to-place macros escalate to the next finer lattice quickly instead of
# thrashing on the current one.
_MAX_RADIUS_CELLS = 64       # absolute cap independent of lattice size
_CANDIDATE_BUDGET = 4096     # max cells examined per macro per lattice attempt


def _pair_overlap(
    cx: float,
    cy: float,
    w: float,
    h: float,
    ox: float,
    oy: float,
    ow: float,
    oh: float,
) -> bool:
    dx = abs(cx - ox)
    dy = abs(cy - oy)
    sep_x = (w + ow) * 0.5 + _GAP
    sep_y = (h + oh) * 0.5 + _GAP
    return dx < sep_x and dy < sep_y


def _ring_cells(r0: int, c0: int, d: int):
    """Yield (r,c) at Manhattan distance exactly d from (r0,c0)."""
    for dr in range(-d, d + 1):
        dc = d - abs(dr)
        r = r0 + dr
        c = c0 + dc
        yield r, c
        if dc != 0:
            yield r, c0 - dc


class SnapPlacer:
    """Snap movable hard macros to a discrete grid with overlap-free legalization.

    Carries the placement forward across lattice resolutions and only retries the
    macros that failed at the previous (coarser) resolution.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        grid_rows = max(int(benchmark.grid_rows), 1)
        grid_cols = max(int(benchmark.grid_cols), 1)

        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        if not movable.any():
            return placement

        n_hard = benchmark.num_hard_macros
        fixed = benchmark.macro_fixed[:n_hard]
        sizes = benchmark.macro_sizes[:n_hard]

        # Cache hot tensor reads as Python floats / lists once.
        sizes_list = [
            (float(sizes[i, 0].item()), float(sizes[i, 1].item())) for i in range(n_hard)
        ]
        fixed_list = [bool(fixed[i].item()) for i in range(n_hard)]
        orig = benchmark.macro_positions[:n_hard]
        orig_list = [
            (float(orig[i, 0].item()), float(orig[i, 1].item())) for i in range(n_hard)
        ]

        movable_idx = torch.where(movable[:n_hard])[0].tolist()
        # Largest-first improves legalization success rate.
        movable_idx.sort(key=lambda i: sizes_list[i][0] * sizes_list[i][1], reverse=True)

        max_half_w = max((w * 0.5 for (w, _h) in sizes_list), default=0.0)
        max_half_h = max((h * 0.5 for (_w, h) in sizes_list), default=0.0)

        fixed_indices = [i for i in range(n_hard) if fixed_list[i]]

        # Persistent across resolutions: which movable macros are already placed,
        # and where (positions live in `placement`).
        placed_set: set[int] = set()
        # remaining is the queue of movable macros still to place; we keep largest-first order
        remaining = list(movable_idx)

        for mult in _LATTICE_MULTIPLIERS:
            if not remaining:
                break

            nx = grid_cols * mult
            ny = grid_rows * mult
            cell_w = cw / nx
            cell_h = ch / ny

            # Spatial index for this lattice; rebuilt per resolution since cell sizes change.
            buckets: dict[tuple[int, int], list[int]] = {}

            def to_rc(x: float, y: float) -> tuple[int, int]:
                col = int(math.floor(x / cell_w))
                row = int(math.floor(y / cell_h))
                col = max(0, min(nx - 1, col))
                row = max(0, min(ny - 1, row))
                return row, col

            # Seed buckets with fixed macros (obstacles).
            for j in fixed_indices:
                fx, fy = orig_list[j]
                buckets.setdefault(to_rc(fx, fy), []).append(j)

            # Seed buckets with macros already placed at coarser resolutions.
            # Their positions stay; they become obstacles for the rest.
            for j in placed_set:
                px = float(placement[j, 0].item())
                py = float(placement[j, 1].item())
                buckets.setdefault(to_rc(px, py), []).append(j)

            def overlaps_local(cx: float, cy: float, w: float, h: float, self_idx: int) -> bool:
                row, col = to_rc(cx, cy)
                rad_c = int(math.ceil((w * 0.5 + max_half_w + _GAP) / cell_w))
                rad_r = int(math.ceil((h * 0.5 + max_half_h + _GAP) / cell_h))

                r0 = max(0, row - rad_r)
                r1 = min(ny - 1, row + rad_r)
                c0 = max(0, col - rad_c)
                c1 = min(nx - 1, col + rad_c)

                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        js = buckets.get((rr, cc))
                        if not js:
                            continue
                        for j in js:
                            if j == self_idx:
                                continue
                            if fixed_list[j]:
                                ox, oy = orig_list[j]
                            else:
                                ox = float(placement[j, 0].item())
                                oy = float(placement[j, 1].item())
                            ow, oh = sizes_list[j]
                            if _pair_overlap(cx, cy, w, h, ox, oy, ow, oh):
                                return True
                return False

            still_failing: list[int] = []
            # Bound the ring expansion.
            max_d = min(_MAX_RADIUS_CELLS, max(nx, ny))

            for idx in remaining:
                w, h = sizes_list[idx]
                ox, oy = orig_list[idx]
                tr, tc = to_rc(ox, oy)
                placed = False
                examined = 0

                for d in range(max_d + 1):
                    if placed:
                        break
                    if examined >= _CANDIDATE_BUDGET:
                        break
                    ring = []
                    for rr, cc in _ring_cells(tr, tc, d):
                        if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                            continue
                        cx = (cc + 0.5) * cell_w
                        cy = (rr + 0.5) * cell_h
                        ring.append(((cx - ox) ** 2 + (cy - oy) ** 2, rr, cc, cx, cy))
                    ring.sort(key=lambda t: t[0])
                    examined += len(ring)

                    for _, rr, cc, cx, cy in ring:
                        if (
                            cx < w * 0.5
                            or cx > cw - w * 0.5
                            or cy < h * 0.5
                            or cy > ch - h * 0.5
                        ):
                            continue
                        if overlaps_local(cx, cy, w, h, idx):
                            continue

                        placement[idx, 0] = cx
                        placement[idx, 1] = cy
                        placed_set.add(idx)
                        buckets.setdefault((rr, cc), []).append(idx)
                        placed = True
                        break

                if not placed:
                    still_failing.append(idx)

            remaining = still_failing

        # Anything still in `remaining` could not be legalized at any lattice.
        # Leave them at their original positions (may remain illegal).
        return placement

