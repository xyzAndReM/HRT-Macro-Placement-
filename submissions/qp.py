"""
QP-based hard-macro legalizer (iterative constraint generation).

Objective: minimize total squared displacement from the initial macro centers while
enforcing non-overlap and boundary constraints for hard macros.

We handle the non-overlap disjunction by iteratively adding a *single* separation
constraint per currently-overlapping pair: the minimum-separation-direction (MSD)
inequality that requires the smallest change at the current solution.

Solver: cvxpy with OSQP backend.

Usage:
    uv run evaluate submissions/qp.py
    uv run evaluate submissions/qp.py -b ibm06
    uv run evaluate submissions/qp.py --all
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

from macro_place.benchmark import Benchmark

try:
    import cvxpy as cp
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "cvxpy is required for submissions/qp.py. Install with: `uv pip install cvxpy osqp` "
        "or add it to project dependencies."
    ) from e


_GAP = 0.005
# Tighter caps keep runtime bounded on hard benchmarks (e.g. ibm06) where the QP +
# residual push would otherwise run for a long time without fully legalizing.
_MAX_QP_ITERS = 25
_MAX_NEW_CONSTRAINTS_PER_ITER = 200
# Outer rounds of bucket pairwise push; each round is O(n · local_degree).
_MAX_FALLBACK_OUTER_ITERS = 1_500


def _sep_key(i: int, j: int, kind: str) -> tuple[int, int, str]:
    """Hashable registry key for a separation constraint."""
    return (i, j, kind)


def _overlaps_xy(
    xi: float,
    yi: float,
    wi: float,
    hi: float,
    xj: float,
    yj: float,
    wj: float,
    hj: float,
    gap: float,
) -> bool:
    dx = (wi + wj) * 0.5 + gap - abs(xi - xj)
    dy = (hi + hj) * 0.5 + gap - abs(yi - yj)
    return dx > 0.0 and dy > 0.0


def _detect_overlaps(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    gap: float,
    bucket: float,
) -> list[tuple[float, int, int]]:
    # Spatial hash: only test pairs that share at least one bucket cell.
    n = int(x.shape[0])

    def cells_touched(i: int) -> Iterable[tuple[int, int]]:
        x0 = math.floor((x[i] - w[i] * 0.5) / bucket)
        x1 = math.floor((x[i] + w[i] * 0.5) / bucket)
        y0 = math.floor((y[i] - h[i] * 0.5) / bucket)
        y1 = math.floor((y[i] + h[i] * 0.5) / bucket)
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                yield (cx, cy)

    buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        for cell in cells_touched(i):
            buckets.setdefault(cell, []).append(i)

    seen: set[tuple[int, int]] = set()
    out: list[tuple[float, int, int]] = []
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        # Quadratic in local occupancy only.
        for a in range(len(ids)):
            i = ids[a]
            for b in range(a + 1, len(ids)):
                j = ids[b]
                ii, jj = (i, j) if i < j else (j, i)
                if (ii, jj) in seen:
                    continue
                seen.add((ii, jj))
                dx = (w[ii] + w[jj]) * 0.5 + gap - abs(x[ii] - x[jj])
                dy = (h[ii] + h[jj]) * 0.5 + gap - abs(y[ii] - y[jj])
                if dx > 0.0 and dy > 0.0:
                    out.append((-(dx * dy), ii, jj))
    return out


def _msd_choice(
    xi: float,
    yi: float,
    wi: float,
    hi: float,
    xj: float,
    yj: float,
    wj: float,
    hj: float,
    gap: float,
) -> str:
    """Return which of the 4 separation inequalities requires the smallest adjustment."""
    rhs_x = -((wi + wj) * 0.5 + gap)
    rhs_y = -((hi + hj) * 0.5 + gap)

    dijx = xi - xj
    dijy = yi - yj

    need_i_left = max(0.0, dijx - rhs_x)     # enforce xi - xj <= rhs_x
    need_j_left = max(0.0, -dijx - rhs_x)    # enforce xj - xi <= rhs_x
    need_i_below = max(0.0, dijy - rhs_y)    # enforce yi - yj <= rhs_y
    need_j_below = max(0.0, -dijy - rhs_y)   # enforce yj - yi <= rhs_y

    # Choose minimum required correction.
    needs = [
        (need_i_left, "iL"),
        (need_j_left, "jL"),
        (need_i_below, "iB"),
        (need_j_below, "jB"),
    ]
    needs.sort(key=lambda t: t[0])
    return needs[0][1]


def _build_bucket_index(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    n: int,
    bucket: float,
) -> dict[tuple[int, int], list[int]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        x0 = int(math.floor((x[i] - w[i] * 0.5) / bucket))
        x1 = int(math.floor((x[i] + w[i] * 0.5) / bucket))
        y0 = int(math.floor((y[i] - h[i] * 0.5) / bucket))
        y1 = int(math.floor((y[i] + h[i] * 0.5) / bucket))
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                buckets.setdefault((cx, cy), []).append(i)
    return buckets


def _neighbor_indices(i: int, buckets: dict[tuple[int, int], list[int]], bucket: float, x, y, w, h) -> list[int]:
    seen: set[int] = {i}
    out: list[int] = []
    x0 = int(math.floor((x[i] - w[i] * 0.5) / bucket))
    x1 = int(math.floor((x[i] + w[i] * 0.5) / bucket))
    y0 = int(math.floor((y[i] - h[i] * 0.5) / bucket))
    y1 = int(math.floor((y[i] + h[i] * 0.5) / bucket))
    for cx in range(x0, x1 + 1):
        for cy in range(y0, y1 + 1):
            for j in buckets.get((cx, cy), ()):
                if j not in seen:
                    seen.add(j)
                    out.append(j)
    return out


def _fallback_pairwise_push(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    fixed: np.ndarray,
    cw: float,
    ch: float,
    bucket: float,
    max_iters: int = _MAX_FALLBACK_OUTER_ITERS,
    alpha: float = 0.5,
    gap: float = _GAP,
) -> None:
    """Residual pairwise push using a bucket index (O(n) neighbors per macro, not O(n²) all pairs)."""
    n = int(x.shape[0])
    for _ in range(max_iters):
        buckets = _build_bucket_index(x, y, w, h, n, bucket)
        moved = False
        for i in range(n):
            for j in _neighbor_indices(i, buckets, bucket, x, y, w, h):
                if j <= i:
                    continue
                dx = (w[i] + w[j]) * 0.5 + gap - abs(x[i] - x[j])
                dy = (h[i] + h[j]) * 0.5 + gap - abs(y[i] - y[j])
                if dx <= 0.0 or dy <= 0.0:
                    continue
                if fixed[i] and fixed[j]:
                    continue
                if dx <= dy:
                    sign = 1.0 if x[i] >= x[j] else -1.0
                    di = sign * dx
                    dj = -sign * dx
                    if fixed[i]:
                        di = 0.0
                    if fixed[j]:
                        dj = 0.0
                    if not fixed[i] and not fixed[j]:
                        ai = w[i] * h[i]
                        aj = w[j] * h[j]
                        tot = ai + aj
                        fi = aj / tot if tot > 0 else 0.5
                        fj = ai / tot if tot > 0 else 0.5
                        di *= fi
                        dj *= fj
                    if not fixed[i]:
                        x[i] = min(max(x[i] + alpha * di, w[i] * 0.5), cw - w[i] * 0.5)
                    if not fixed[j]:
                        x[j] = min(max(x[j] + alpha * dj, w[j] * 0.5), cw - w[j] * 0.5)
                else:
                    sign = 1.0 if y[i] >= y[j] else -1.0
                    di = sign * dy
                    dj = -sign * dy
                    if fixed[i]:
                        di = 0.0
                    if fixed[j]:
                        dj = 0.0
                    if not fixed[i] and not fixed[j]:
                        ai = w[i] * h[i]
                        aj = w[j] * h[j]
                        tot = ai + aj
                        fi = aj / tot if tot > 0 else 0.5
                        fj = ai / tot if tot > 0 else 0.5
                        di *= fi
                        dj *= fj
                    if not fixed[i]:
                        y[i] = min(max(y[i] + alpha * di, h[i] * 0.5), ch - h[i] * 0.5)
                    if not fixed[j]:
                        y[j] = min(max(y[j] + alpha * dj, h[j] * 0.5), ch - h[j] * 0.5)
                moved = True
        if not moved:
            break


def _try_snap_one(
    idx: int,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    fixed: np.ndarray,
    cw: float,
    ch: float,
    cell_w: float,
    cell_h: float,
    gap: float,
) -> bool:
    if fixed[idx]:
        return False
    nx = round(x[idx] / cell_w) * cell_w
    ny = round(y[idx] / cell_h) * cell_h
    nx = min(max(nx, w[idx] * 0.5), cw - w[idx] * 0.5)
    ny = min(max(ny, h[idx] * 0.5), ch - h[idx] * 0.5)
    if nx == x[idx] and ny == y[idx]:
        return False
    oldx, oldy = x[idx], y[idx]
    x[idx], y[idx] = nx, ny
    for j in range(int(x.shape[0])):
        if j == idx:
            continue
        if _overlaps_xy(x[idx], y[idx], w[idx], h[idx], x[j], y[j], w[j], h[j], gap):
            x[idx], y[idx] = oldx, oldy
            return False
    return True


class QPLegalizer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()

        n_hard = int(benchmark.num_hard_macros)
        if n_hard == 0:
            return placement

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        grid_rows = max(int(benchmark.grid_rows), 1)
        grid_cols = max(int(benchmark.grid_cols), 1)
        cell_w = cw / grid_cols
        cell_h = ch / grid_rows
        bucket = max(cell_w, cell_h)

        pos = benchmark.macro_positions[:n_hard].detach().cpu().numpy().astype(np.float64)
        sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
        fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)

        x0 = pos[:, 0].copy()
        y0 = pos[:, 1].copy()
        w = sizes[:, 0].copy()
        h = sizes[:, 1].copy()

        movable_ids = [i for i in range(n_hard) if not fixed[i]]
        m = len(movable_ids)
        if m == 0:
            return placement

        to_var = {mid: k for k, mid in enumerate(movable_ids)}
        x0m = np.array([x0[mid] for mid in movable_ids], dtype=np.float64)
        y0m = np.array([y0[mid] for mid in movable_ids], dtype=np.float64)
        wm = np.array([w[mid] for mid in movable_ids], dtype=np.float64)
        hm = np.array([h[mid] for mid in movable_ids], dtype=np.float64)

        x = cp.Variable(m)
        y = cp.Variable(m)

        objective = cp.sum_squares(x - x0m) + cp.sum_squares(y - y0m)
        constraints: list[cp.Constraint] = []

        # Boundary constraints.
        constraints += [
            x >= wm * 0.5,
            x <= cw - wm * 0.5,
            y >= hm * 0.5,
            y <= ch - hm * 0.5,
        ]

        registry: set[tuple[int, int, str]] = set()

        prob = cp.Problem(cp.Minimize(objective), constraints)

        x_prev = x0m.copy()
        y_prev = y0m.copy()

        osqp_kw = dict(
            warm_start=True,
            verbose=False,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=2500,
            polish=True,
        )

        for it in range(_MAX_QP_ITERS):
            x.value = x_prev
            y.value = y_prev
            prob.solve(solver=cp.OSQP, **osqp_kw)
            if x.value is None or y.value is None:
                break

            x_prev = np.asarray(x.value, dtype=np.float64).copy()
            y_prev = np.asarray(y.value, dtype=np.float64).copy()

            # Build current full hard-macro placement.
            xcur = x0.copy()
            ycur = y0.copy()
            for mid in movable_ids:
                k = to_var[mid]
                xcur[mid] = x_prev[k]
                ycur[mid] = y_prev[k]

            overlaps = _detect_overlaps(xcur, ycur, w, h, _GAP, bucket)
            if not overlaps:
                break
            overlaps.sort()  # most negative area first
            added = 0

            for _neg_area, i, j in overlaps[:_MAX_NEW_CONSTRAINTS_PER_ITER]:
                if fixed[i] and fixed[j]:
                    continue
                kind = _msd_choice(xcur[i], ycur[i], w[i], h[i], xcur[j], ycur[j], w[j], h[j], _GAP)
                key = _sep_key(min(i, j), max(i, j), kind)
                if key in registry:
                    continue

                rhs_x = -((w[i] + w[j]) * 0.5 + _GAP)
                rhs_y = -((h[i] + h[j]) * 0.5 + _GAP)

                # Add the chosen inequality, expressed in variables (movables) and constants (fixed).
                if kind == "iL":
                    # xi - xj <= rhs_x
                    if not fixed[i] and not fixed[j]:
                        constraints.append(x[to_var[i]] - x[to_var[j]] <= rhs_x)
                    elif not fixed[i] and fixed[j]:
                        constraints.append(x[to_var[i]] <= x0[j] + rhs_x)
                    elif fixed[i] and not fixed[j]:
                        constraints.append(-x[to_var[j]] <= -x0[i] + rhs_x)
                    else:
                        continue
                elif kind == "jL":
                    # xj - xi <= rhs_x
                    if not fixed[i] and not fixed[j]:
                        constraints.append(x[to_var[j]] - x[to_var[i]] <= rhs_x)
                    elif fixed[i] and not fixed[j]:
                        constraints.append(x[to_var[j]] <= x0[i] + rhs_x)
                    elif not fixed[i] and fixed[j]:
                        constraints.append(-x[to_var[i]] <= -x0[j] + rhs_x)
                    else:
                        continue
                elif kind == "iB":
                    # yi - yj <= rhs_y
                    if not fixed[i] and not fixed[j]:
                        constraints.append(y[to_var[i]] - y[to_var[j]] <= rhs_y)
                    elif not fixed[i] and fixed[j]:
                        constraints.append(y[to_var[i]] <= y0[j] + rhs_y)
                    elif fixed[i] and not fixed[j]:
                        constraints.append(-y[to_var[j]] <= -y0[i] + rhs_y)
                    else:
                        continue
                else:  # "jB"
                    # yj - yi <= rhs_y
                    if not fixed[i] and not fixed[j]:
                        constraints.append(y[to_var[j]] - y[to_var[i]] <= rhs_y)
                    elif fixed[i] and not fixed[j]:
                        constraints.append(y[to_var[j]] <= y0[i] + rhs_y)
                    elif not fixed[i] and fixed[j]:
                        constraints.append(-y[to_var[i]] <= -y0[j] + rhs_y)
                    else:
                        continue

                registry.add(key)
                added += 1

            if added == 0:
                break

            # Rebuild the problem with the expanded constraint list.
            prob = cp.Problem(cp.Minimize(objective), constraints)

        # Final full placement after QP.
        xcur = x0.copy()
        ycur = y0.copy()
        for mid in movable_ids:
            k = to_var[mid]
            xcur[mid] = x_prev[k]
            ycur[mid] = y_prev[k]

        # If still overlapping, briefly try residual pairwise push (capped).
        if len(_detect_overlaps(xcur, ycur, w, h, 0.0, bucket)) > 0:
            _fallback_pairwise_push(
                xcur,
                ycur,
                w,
                h,
                fixed,
                cw,
                ch,
                bucket,
                max_iters=_MAX_FALLBACK_OUTER_ITERS,
            )

        # Optional snap-to-grid (safe snap).
        for mid in movable_ids:
            _try_snap_one(mid, xcur, ycur, w, h, fixed, cw, ch, cell_w, cell_h, _GAP)

        # Write back into torch placement; keep soft macros unchanged.
        for i in range(n_hard):
            placement[i, 0] = float(xcur[i])
            placement[i, 1] = float(ycur[i])
        return placement

