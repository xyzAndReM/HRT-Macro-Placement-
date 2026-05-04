"""
Simple iterative penetration-resolution placer.

Treats macros as soft bodies and resolves overlaps by repeatedly pushing the
heaviest-overlap pair apart along its cheaper separation axis (the axis of
smaller penetration depth), proportional to size — large macros move less,
fixed macros do not move at all. After continuous-space overlaps are resolved,
each movable hard macro is snapped to the nearest routing-grid center and a
final cleanup pass fixes any snap-induced overlaps (which are at most one
grid pitch deep).

Phases:
  0. Bucket spatial index (each macro registered in every bucket its bbox
     touches) for fast neighbor queries.
  1. Detect all overlapping pairs into a max-heap keyed by overlap area.
  2. Pop the heaviest overlap, choose the cheaper separation axis.
  3. Apply damped (ALPHA) displacement to each side and clamp to canvas.
  4. Re-queue any newly-overlapping neighbors of the moved macros. Loop.
  5. Overlap-aware snap: each movable hard macro is rounded to the nearest
     routing-grid center, but only if doing so doesn't introduce a new overlap.
     (Naïve snap stacks several macros on the same cell and the cleanup loop
     turns into long chain-reaction work.)

Soft macros are not moved (they are clusters of standard cells, not the target
of this placer). Fixed hard macros stay put and act as obstacles.

Usage:
    uv run evaluate submissions/simple.py
    uv run evaluate submissions/simple.py -b ibm06
    uv run evaluate submissions/simple.py --all
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

import torch

from macro_place.benchmark import Benchmark


# Target separation margin. The validator treats any positive overlap as illegal,
# so we keep a small positive gap to avoid ending up with tiny residual overlaps
# after damped updates and chain reactions.
_GAP = 0.005
# Damping factor on each pair-resolution step. <1 prevents oscillation when a
# macro's neighbours simultaneously push it from both sides.
_ALPHA = 0.5
# Iteration cap per fix-overlaps pass. Each pop reduces a pair's penetration
# by (1 - ALPHA); chain reactions can keep a few pairs oscillating, so we
# bound the total work and accept whatever leftover the resolver couldn't fix.
_MAX_ITER = 100_000
# Stuck handling: periodically check strict (GAP=0) overlaps; if the count does
# not improve for several checks, apply one group push to the worst connected
# component of the strict-overlap graph, then resume pairwise.
_CHECK_EVERY = 1024
_STAGNANT_CHECKS = 1
_MAX_COMPONENT_STEPS = 16
_COMP_ALPHA = 1.0
_COMP_MAX_STEP = 5.0  # microns, cap per-axis group move per macro per step


# ── Macro record ────────────────────────────────────────────────────────────

class _Macro:
    __slots__ = ("id", "cx", "cy", "w", "h", "fixed")

    def __init__(self, mid: int, cx: float, cy: float, w: float, h: float, fixed: bool):
        self.id = mid
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.fixed = fixed


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ── Phase 0: bucket spatial index ───────────────────────────────────────────

def _cells_touched(m: _Macro, bucket: float):
    x0 = math.floor((m.cx - m.w * 0.5) / bucket)
    x1 = math.floor((m.cx + m.w * 0.5) / bucket)
    y0 = math.floor((m.cy - m.h * 0.5) / bucket)
    y1 = math.floor((m.cy + m.h * 0.5) / bucket)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            yield (x, y)


def _build_index(macros: list[_Macro], bucket: float) -> dict[tuple[int, int], list[int]]:
    idx: dict[tuple[int, int], list[int]] = defaultdict(list)
    for m in macros:
        for cell in _cells_touched(m, bucket):
            idx[cell].append(m.id)
    return idx


def _insert(idx: dict[tuple[int, int], list[int]], m: _Macro, bucket: float) -> None:
    for cell in _cells_touched(m, bucket):
        idx.setdefault(cell, []).append(m.id)


def _remove(idx: dict[tuple[int, int], list[int]], m: _Macro, bucket: float) -> None:
    for cell in _cells_touched(m, bucket):
        lst = idx.get(cell)
        if lst is None:
            continue
        try:
            lst.remove(m.id)
        except ValueError:
            pass
        if not lst:
            del idx[cell]


def _neighbors(m: _Macro, idx: dict[tuple[int, int], list[int]],
               macros: list[_Macro], bucket: float):
    seen: set[int] = set()
    for cell in _cells_touched(m, bucket):
        for mid in idx.get(cell, ()):
            if mid != m.id and mid not in seen:
                seen.add(mid)
                yield macros[mid]


# ── Phase 1: overlap detection ──────────────────────────────────────────────

def _overlap_depth(a: _Macro, b: _Macro) -> tuple[float, float]:
    """Penetration depth on each axis. Both positive ⇒ overlapping."""
    dx = (a.w + b.w) * 0.5 + _GAP - abs(a.cx - b.cx)
    dy = (a.h + b.h) * 0.5 + _GAP - abs(a.cy - b.cy)
    return dx, dy


def _overlap_depth_gap0(a: _Macro, b: _Macro) -> tuple[float, float]:
    """Penetration depth with GAP=0 (matches validator strict overlap check)."""
    dx = (a.w + b.w) * 0.5 - abs(a.cx - b.cx)
    dy = (a.h + b.h) * 0.5 - abs(a.cy - b.cy)
    return dx, dy


def _has_true_overlaps(macros: list[_Macro],
                       idx: dict[tuple[int, int], list[int]],
                       bucket: float) -> bool:
    """Return True iff any hard-macro pair strictly overlaps (GAP=0)."""
    seen: set[tuple[int, int]] = set()
    for m in macros:
        for nb in _neighbors(m, idx, macros, bucket):
            i, j = (m.id, nb.id) if m.id < nb.id else (nb.id, m.id)
            pair = (i, j)
            if pair in seen:
                continue
            seen.add(pair)
            dx0, dy0 = _overlap_depth_gap0(macros[i], macros[j])
            if dx0 > 0.0 and dy0 > 0.0:
                return True
    return False


def _true_overlap_edges(macros: list[_Macro]) -> list[tuple[int, int, float, float, float]]:
    """Return strict (GAP=0) overlap edges via brute-force: (i, j, area, dx0, dy0).

    This intentionally mirrors the validator semantics (strict intersection)
    and avoids relying on the bucket index, which can be inconsistent if the
    incremental remove/insert bookkeeping drifts.
    """
    n = len(macros)
    edges: list[tuple[int, int, float, float, float]] = []
    for i in range(n):
        a = macros[i]
        ax0 = a.cx - a.w * 0.5
        ax1 = a.cx + a.w * 0.5
        ay0 = a.cy - a.h * 0.5
        ay1 = a.cy + a.h * 0.5
        for j in range(i + 1, n):
            b = macros[j]
            bx0 = b.cx - b.w * 0.5
            bx1 = b.cx + b.w * 0.5
            by0 = b.cy - b.h * 0.5
            by1 = b.cy + b.h * 0.5
            # Strict overlap: intervals intersect with positive length on both axes.
            if ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0:
                dx0, dy0 = _overlap_depth_gap0(a, b)
                if dx0 > 0.0 and dy0 > 0.0:
                    edges.append((i, j, dx0 * dy0, dx0, dy0))
    return edges


def _count_true_overlaps(macros: list[_Macro],
                         idx: dict[tuple[int, int], list[int]],
                         bucket: float) -> int:
    """Count strict (GAP=0) overlapping pairs."""
    return len(_true_overlap_edges(macros))


def _worst_overlap_component(macros: list[_Macro],
                             idx: dict[tuple[int, int], list[int]],
                             bucket: float):
    """Find the connected component with largest total strict-overlap area."""
    edges = _true_overlap_edges(macros)
    if not edges:
        return set(), []

    adj: dict[int, list[int]] = defaultdict(list)
    for i, j, _area, _dx0, _dy0 in edges:
        adj[i].append(j)
        adj[j].append(i)

    edge_map: dict[tuple[int, int], tuple[float, float, float]] = {}
    for i, j, area, dx0, dy0 in edges:
        edge_map[(i, j)] = (area, dx0, dy0)

    seen: set[int] = set()
    best_nodes: set[int] = set()
    best_edges: list[tuple[int, int, float, float, float]] = []
    best_score = -1.0

    for start in adj.keys():
        if start in seen:
            continue
        stack = [start]
        nodes: set[int] = set()
        seen.add(start)
        while stack:
            u = stack.pop()
            nodes.add(u)
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)

        comp_edges: list[tuple[int, int, float, float, float]] = []
        score = 0.0
        for u in nodes:
            for v in adj.get(u, ()):
                if u < v:
                    area, dx0, dy0 = edge_map[(u, v)]
                    score += area
                    comp_edges.append((u, v, area, dx0, dy0))
        if score > best_score:
            best_score = score
            best_nodes = nodes
            best_edges = comp_edges

    return best_nodes, best_edges


def _apply_with_alpha(m: _Macro, axis: str, delta: float, cw: float, ch: float, alpha: float) -> None:
    if m.fixed or delta == 0.0:
        return
    delta *= alpha
    if axis == "x":
        m.cx = _clamp(m.cx + delta, m.w * 0.5, cw - m.w * 0.5)
    else:
        m.cy = _clamp(m.cy + delta, m.h * 0.5, ch - m.h * 0.5)


def _detect_all_overlaps(macros: list[_Macro],
                         idx: dict[tuple[int, int], list[int]],
                         bucket: float) -> list[tuple[float, int, int]]:
    heap: list[tuple[float, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for m in macros:
        for nb in _neighbors(m, idx, macros, bucket):
            i, j = (m.id, nb.id) if m.id < nb.id else (nb.id, m.id)
            pair = (i, j)
            if pair in seen:
                continue
            seen.add(pair)
            dx, dy = _overlap_depth(macros[i], macros[j])
            if dx > 0.0 and dy > 0.0:
                heapq.heappush(heap, (-dx * dy, i, j))
    return heap


# ── Phase 2: separation axis & displacement split ───────────────────────────

def _separation_axis(a: _Macro, b: _Macro, dx: float, dy: float):
    """Pick cheaper axis; return (axis, delta_a, delta_b) signed.

    Movable-vs-movable: split displacement inversely with area (small moves more).
    Movable-vs-fixed:   movable absorbs the entire push.
    Fixed-vs-fixed:     no movement possible (caller should skip).
    """
    if a.fixed and b.fixed:
        return "x", 0.0, 0.0

    if a.fixed:
        if dx <= dy:
            sign = 1.0 if a.cx >= b.cx else -1.0
            return "x", 0.0, -sign * dx
        sign = 1.0 if a.cy >= b.cy else -1.0
        return "y", 0.0, -sign * dy

    if b.fixed:
        if dx <= dy:
            sign = 1.0 if a.cx >= b.cx else -1.0
            return "x", sign * dx, 0.0
        sign = 1.0 if a.cy >= b.cy else -1.0
        return "y", sign * dy, 0.0

    area_a = a.w * a.h
    area_b = b.w * b.h
    total = area_a + area_b
    if total <= 0.0:
        frac_a = frac_b = 0.5
    else:
        frac_a = area_b / total
        frac_b = area_a / total

    if dx <= dy:
        sign = 1.0 if a.cx >= b.cx else -1.0
        return "x", sign * dx * frac_a, -sign * dx * frac_b
    sign = 1.0 if a.cy >= b.cy else -1.0
    return "y", sign * dy * frac_a, -sign * dy * frac_b


# ── Phase 3: apply damped, clamped displacement ─────────────────────────────

def _apply(m: _Macro, axis: str, delta: float, cw: float, ch: float) -> None:
    _apply_with_alpha(m, axis, delta, cw, ch, _ALPHA)


# ── Phase 4: main fix-overlaps loop ─────────────────────────────────────────

def _fix_overlaps(macros: list[_Macro],
                  idx: dict[tuple[int, int], list[int]],
                  cw: float, ch: float, bucket: float) -> int:
    def component_shove_once() -> bool:
        nodes, edges = _worst_overlap_component(macros, idx, bucket)
        if not nodes or not edges:
            return False

        dx_acc: dict[int, float] = {n: 0.0 for n in nodes}
        dy_acc: dict[int, float] = {n: 0.0 for n in nodes}

        for u, v, _area, dx0, dy0 in edges:
            a = macros[u]
            b = macros[v]
            dxr = dx0 + _GAP
            dyr = dy0 + _GAP
            axis, da, db = _separation_axis(a, b, dxr, dyr)

            if da > _COMP_MAX_STEP:
                da = _COMP_MAX_STEP
            elif da < -_COMP_MAX_STEP:
                da = -_COMP_MAX_STEP
            if db > _COMP_MAX_STEP:
                db = _COMP_MAX_STEP
            elif db < -_COMP_MAX_STEP:
                db = -_COMP_MAX_STEP

            if axis == "x":
                dx_acc[u] += da
                dx_acc[v] += db
            else:
                dy_acc[u] += da
                dy_acc[v] += db

        for n in nodes:
            _remove(idx, macros[n], bucket)
        for n in nodes:
            if dx_acc[n] != 0.0:
                _apply_with_alpha(macros[n], "x", dx_acc[n], cw, ch, _COMP_ALPHA)
            if dy_acc[n] != 0.0:
                _apply_with_alpha(macros[n], "y", dy_acc[n], cw, ch, _COMP_ALPHA)
        for n in nodes:
            _insert(idx, macros[n], bucket)

        return True

    heap = _detect_all_overlaps(macros, idx, bucket)
    in_queue: set[tuple[int, int]] = {(i, j) for (_, i, j) in heap}

    iters = 0
    best_true = _count_true_overlaps(macros, idx, bucket)
    stagnant = 0
    comp_steps = 0
    while heap and iters < _MAX_ITER:
        iters += 1
        _, i, j = heapq.heappop(heap)
        in_queue.discard((i, j))

        a = macros[i]
        b = macros[j]
        dx, dy = _overlap_depth(a, b)
        if not (dx > 0.0 and dy > 0.0):
            continue
        if a.fixed and b.fixed:
            continue

        axis, da, db = _separation_axis(a, b, dx, dy)

        _remove(idx, a, bucket)
        _remove(idx, b, bucket)
        _apply(a, axis, da, cw, ch)
        _apply(b, axis, db, cw, ch)
        _insert(idx, a, bucket)
        _insert(idx, b, bucket)

        for m in (a, b):
            for nb in _neighbors(m, idx, macros, bucket):
                ddx, ddy = _overlap_depth(m, nb)
                if ddx > 0.0 and ddy > 0.0:
                    p, q = (m.id, nb.id) if m.id < nb.id else (nb.id, m.id)
                    if (p, q) not in in_queue:
                        heapq.heappush(heap, (-ddx * ddy, p, q))
                        in_queue.add((p, q))

        if (iters & (_CHECK_EVERY - 1)) == 0:
            true_now = _count_true_overlaps(macros, idx, bucket)
            if true_now == 0:
                break
            if true_now < best_true:
                best_true = true_now
                stagnant = 0
            else:
                stagnant += 1

            if stagnant >= _STAGNANT_CHECKS and comp_steps < _MAX_COMPONENT_STEPS:
                if component_shove_once():
                    heap = _detect_all_overlaps(macros, idx, bucket)
                    in_queue = {(i, j) for (_, i, j) in heap}
                    comp_steps += 1
                    stagnant = 0

    while comp_steps < _MAX_COMPONENT_STEPS and _has_true_overlaps(macros, idx, bucket):
        if not component_shove_once():
            break
        comp_steps += 1
        heap = _detect_all_overlaps(macros, idx, bucket)
        in_queue = {(i, j) for (_, i, j) in heap}
        stagnant = 0
        burst = 0
        while heap and iters < _MAX_ITER and burst < _CHECK_EVERY:
            burst += 1
            iters += 1
            _, i, j = heapq.heappop(heap)
            in_queue.discard((i, j))
            a = macros[i]
            b = macros[j]
            dx, dy = _overlap_depth(a, b)
            if not (dx > 0.0 and dy > 0.0):
                continue
            if a.fixed and b.fixed:
                continue
            axis, da, db = _separation_axis(a, b, dx, dy)
            _remove(idx, a, bucket)
            _remove(idx, b, bucket)
            _apply(a, axis, da, cw, ch)
            _apply(b, axis, db, cw, ch)
            _insert(idx, a, bucket)
            _insert(idx, b, bucket)
            for m in (a, b):
                for nb in _neighbors(m, idx, macros, bucket):
                    ddx, ddy = _overlap_depth(m, nb)
                    if ddx > 0.0 and ddy > 0.0:
                        p, q = (m.id, nb.id) if m.id < nb.id else (nb.id, m.id)
                        if (p, q) not in in_queue:
                            heapq.heappush(heap, (-ddx * ddy, p, q))
                            in_queue.add((p, q))

            if not _has_true_overlaps(macros, idx, bucket):
                break

    return iters


# ── Phase 5: snap to routing grid (overlap-aware) ───────────────────────────

def _try_snap_to_grid(m: _Macro,
                      idx: dict[tuple[int, int], list[int]],
                      macros: list[_Macro],
                      cell_w: float, cell_h: float,
                      cw: float, ch: float, bucket: float) -> bool:
    """Snap m to its nearest grid center if doing so doesn't overlap a neighbor.

    Returns True if the snap was applied.
    """
    if m.fixed:
        return False
    nx = round(m.cx / cell_w) * cell_w
    ny = round(m.cy / cell_h) * cell_h
    nx = _clamp(nx, m.w * 0.5, cw - m.w * 0.5)
    ny = _clamp(ny, m.h * 0.5, ch - m.h * 0.5)

    old_cx, old_cy = m.cx, m.cy
    _remove(idx, m, bucket)
    m.cx, m.cy = nx, ny
    bad = False
    for nb in _neighbors(m, idx, macros, bucket):
        dx, dy = _overlap_depth(m, nb)
        if dx > 0.0 and dy > 0.0:
            bad = True
            break
    if bad:
        m.cx, m.cy = old_cx, old_cy
        _insert(idx, m, bucket)
        return False
    _insert(idx, m, bucket)
    return True


# ── Placer entry point ──────────────────────────────────────────────────────

class SimplePlacer:
    """Iterative pairwise overlap resolver + snap-to-grid + cleanup."""

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        grid_rows = max(int(benchmark.grid_rows), 1)
        grid_cols = max(int(benchmark.grid_cols), 1)
        cell_w = cw / grid_cols
        cell_h = ch / grid_rows

        n_hard = benchmark.num_hard_macros
        if n_hard == 0:
            return placement

        positions = benchmark.macro_positions[:n_hard]
        sizes = benchmark.macro_sizes[:n_hard]
        fixed = benchmark.macro_fixed[:n_hard]

        macros: list[_Macro] = [
            _Macro(
                i,
                float(positions[i, 0].item()),
                float(positions[i, 1].item()),
                float(sizes[i, 0].item()),
                float(sizes[i, 1].item()),
                bool(fixed[i].item()),
            )
            for i in range(n_hard)
        ]

        bucket = max(cell_w, cell_h)
        idx = _build_index(macros, bucket)

        _fix_overlaps(macros, idx, cw, ch, bucket)

        for m in macros:
            _try_snap_to_grid(m, idx, macros, cell_w, cell_h, cw, ch, bucket)

        for m in macros:
            placement[m.id, 0] = m.cx
            placement[m.id, 1] = m.cy

        return placement

