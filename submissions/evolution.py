"""
EGPlace-style building blocks for this repo (ICCAD04 / ``Benchmark`` tensors).

This file implements EGPlace / WireMask-style **masks** on the placement grid:

- **Bound mask** — ``1`` if macro ``m`` with bottom-left at bin ``(r,c)`` would
  **cross the die** (EGPlace Fig. 2: bound = 1 means out of region), else ``0``.
- **Overlap mask** — **total overlap area** (μm²) between trial ``m`` at bin
  ``(r,c)`` and other **hard** macros (default), via **1D × 1D** factors per
  neighbor: ``area = ox[c] · oy[r]`` summed over ``j ≠ m`` (EGPlace Alg. 2 idea).
- **Wire mask** — **weighted ΔHPWL** for the same bottom-left convention
  (illegal bins get ``invalid``, default ``inf``).

- Uses **pin-level** HPWL when ``benchmark.net_pin_nodes`` is populated
  (loader does this for ICCAD04); otherwise falls back to one point per macro
  (macro center) via ``net_nodes``.

**Fitness & module scores** (EGPlace §4–5):

- **Objective** (Eq. 1): ``HPWL + λ1 · RUDY_max + λ2 · overlap_rate`` with
  ``overlap_rate = (hard–hard overlap area) / chip area``.
- **Fitness** ``f_L = -objective`` (higher is better).
- **Selection** (Eq. 2): ``p(L_i) ∝ exp(f_{L_i})`` via ``layout_selection_probs``.
- **Module score** (Eq. 3–5): ``(wirelen_m + λ1·cong_m + λ2·overlap_m_raw) / (w_m·h_m)``
  with ``wirelen_m`` = mean pin–net-center Manhattan distance, ``cong_m`` pin
  counts for hot nets (``rudy(net) > r_frac·RUDY_max``), ``overlap_m_raw`` =
  hard–hard overlap area for ``m``.

A full EGPlace loop (population, mutation, greedy reposition) can call these.

Usage:
    uv run python submissions/evolution.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _macro_to_nets(benchmark: Benchmark) -> list[list[int]]:
    n_macros = int(benchmark.num_macros)
    out: list[list[int]] = [[] for _ in range(n_macros)]
    for ni in range(int(benchmark.num_nets)):
        for v in benchmark.net_nodes[ni].tolist():
            v = int(v)
            if 0 <= v < n_macros:
                out[v].append(ni)
    return out


def _pin_xy(
    owner: int,
    pin_slot: int,
    pos: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
) -> tuple[float, float]:
    if owner < n_macros:
        ox = oy = 0.0
        offs_list = benchmark.macro_pin_offsets
        if (
            owner < len(offs_list)
            and isinstance(offs_list[owner], torch.Tensor)
            and offs_list[owner].numel() > 0
        ):
            po = offs_list[owner]
            if pin_slot < int(po.shape[0]):
                ox, oy = float(po[pin_slot, 0]), float(po[pin_slot, 1])
        return float(pos[owner, 0]) + ox, float(pos[owner, 1]) + oy
    pi = owner - n_macros
    if 0 <= pi < n_ports:
        return float(ports_np[pi, 0]), float(ports_np[pi, 1])
    return 0.0, 0.0


def _single_net_hpwl(
    ni: int,
    pos: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
) -> float:
    """Half-perimeter wirelength for net ``ni`` (weighted nets use same geometry)."""
    if benchmark.net_pin_nodes and len(benchmark.net_pin_nodes) > ni:
        pnodes = benchmark.net_pin_nodes[ni]
        if pnodes.numel() < 2:
            return 0.0
        xs: list[float] = []
        ys: list[float] = []
        for row in pnodes.tolist():
            owner, slot = int(row[0]), int(row[1])
            x, y = _pin_xy(owner, slot, pos, benchmark, ports_np, n_macros, n_ports)
            xs.append(x)
            ys.append(y)
        if len(xs) < 2:
            return 0.0
        return max(xs) - min(xs) + max(ys) - min(ys)

    nodes = benchmark.net_nodes[ni]
    if nodes.numel() < 2:
        return 0.0
    xs, ys = [], []
    for v in nodes.tolist():
        v = int(v)
        if v < n_macros:
            xs.append(float(pos[v, 0]))
            ys.append(float(pos[v, 1]))
        else:
            pi = v - n_macros
            if 0 <= pi < n_ports:
                xs.append(float(ports_np[pi, 0]))
                ys.append(float(ports_np[pi, 1]))
    if len(xs) < 2:
        return 0.0
    return max(xs) - min(xs) + max(ys) - min(ys)


def _net_pin_bbox(
    ni: int,
    pos: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
) -> tuple[float, float, float, float, float, float] | None:
    """Bounding box of net ``ni`` from pin positions; returns ``lx, rx, by, ty, cx, cy``."""
    if benchmark.net_pin_nodes and len(benchmark.net_pin_nodes) > ni:
        pnodes = benchmark.net_pin_nodes[ni]
        if pnodes.numel() < 2:
            return None
        xs: list[float] = []
        ys: list[float] = []
        for row in pnodes.tolist():
            owner, slot = int(row[0]), int(row[1])
            x, y = _pin_xy(owner, slot, pos, benchmark, ports_np, n_macros, n_ports)
            xs.append(x)
            ys.append(y)
    else:
        nodes = benchmark.net_nodes[ni]
        if nodes.numel() < 2:
            return None
        xs, ys = [], []
        for v in nodes.tolist():
            v = int(v)
            if v < n_macros:
                xs.append(float(pos[v, 0]))
                ys.append(float(pos[v, 1]))
            else:
                pi = v - n_macros
                if 0 <= pi < n_ports:
                    xs.append(float(ports_np[pi, 0]))
                    ys.append(float(ports_np[pi, 1]))
    if len(xs) < 2:
        return None
    lx, rx = min(xs), max(xs)
    by, ty = min(ys), max(ys)
    return lx, rx, by, ty, 0.5 * (lx + rx), 0.5 * (by + ty)


def compute_rudy_grid(
    pos: np.ndarray,
    benchmark: Benchmark,
    *,
    nr: int | None = None,
    nc: int | None = None,
) -> np.ndarray:
    """
    RUDY grid (Spindler & Johannes / EGPlace Appendix B): each bin accumulates
    ``Σ_nets (1/h_bb + 1/w_bb)`` for every net whose **pin bounding box**
    intersects the bin (``w_bb``, ``h_bb`` in μm, floored away from zero).
    """
    cw, ch, nr, nc, cell_w, cell_h = _placement_grid(benchmark, nr, nc)
    grid = np.zeros((nr, nc), dtype=np.float64)
    ports_np = benchmark.port_positions.detach().cpu().numpy()
    n_macros = int(benchmark.num_macros)
    n_ports = int(ports_np.shape[0])

    # Minimum bbox extent (μm) so 1/w + 1/h stays bounded for collinear / tiny nets.
    min_bb = max(0.01, 0.001 * min(cell_w, cell_h))
    for ni in range(int(benchmark.num_nets)):
        bb = _net_pin_bbox(ni, pos, benchmark, ports_np, n_macros, n_ports)
        if bb is None:
            continue
        lx, rx, by, ty, _, _ = bb
        w_bb = max(rx - lx, min_bb)
        h_bb = max(ty - by, min_bb)
        contrib = 1.0 / h_bb + 1.0 / w_bb
        c0 = max(0, int(np.floor(lx / cell_w)))
        c1 = min(nc - 1, int(np.floor(rx / cell_w)))
        r0 = max(0, int(np.floor(by / cell_h)))
        r1 = min(nr - 1, int(np.floor(ty / cell_h)))
        if c0 <= c1 and r0 <= r1:
            grid[r0 : r1 + 1, c0 : c1 + 1] += contrib
    return grid


def layout_rudy_max(rudy_grid: np.ndarray) -> float:
    return float(np.max(rudy_grid)) if rudy_grid.size else 0.0


def _net_max_rudy_in_bbox(
    rudy_grid: np.ndarray,
    lx: float,
    rx: float,
    by: float,
    ty: float,
    cell_w: float,
    cell_h: float,
    nr: int,
    nc: int,
) -> float:
    c0 = max(0, int(np.floor(lx / cell_w)))
    c1 = min(nc - 1, int(np.floor(rx / cell_w)))
    r0 = max(0, int(np.floor(by / cell_h)))
    r1 = min(nr - 1, int(np.floor(ty / cell_h)))
    if c0 > c1 or r0 > r1:
        return 0.0
    return float(np.max(rudy_grid[r0 : r1 + 1, c0 : c1 + 1]))


def compute_total_hard_overlap_area(
    pos: np.ndarray,
    benchmark: Benchmark,
) -> float:
    """Sum of pairwise **overlap areas** (μm²) over hard macros ``i < j``."""
    n_hard = int(benchmark.num_hard_macros)
    s = 0.0
    for i in range(n_hard):
        ci_x, ci_y = float(pos[i, 0]), float(pos[i, 1])
        wi, hi = float(benchmark.macro_sizes[i, 0]), float(benchmark.macro_sizes[i, 1])
        li, ri = ci_x - 0.5 * wi, ci_x + 0.5 * wi
        bi, ti = ci_y - 0.5 * hi, ci_y + 0.5 * hi
        for j in range(i + 1, n_hard):
            cj_x, cj_y = float(pos[j, 0]), float(pos[j, 1])
            wj, hj = float(benchmark.macro_sizes[j, 0]), float(benchmark.macro_sizes[j, 1])
            lj, rj = cj_x - 0.5 * wj, cj_x + 0.5 * wj
            bj, tj = cj_y - 0.5 * hj, cj_y + 0.5 * hj
            ix = max(0.0, min(ri, rj) - max(li, lj))
            iy = max(0.0, min(ti, tj) - max(bi, bj))
            s += ix * iy
    return s


def layout_objective(
    pos: np.ndarray,
    benchmark: Benchmark,
    *,
    lam1: float = 0.0,
    lam2: float = 0.1,
    nr: int | None = None,
    nc: int | None = None,
    rudy_grid: np.ndarray | None = None,
) -> dict[str, float]:
    """
    EGPlace Eq. (1): ``HPWL + λ1·RUDY_max + λ2·overlap_rate``.

    ``HPWL`` is **weighted** pin-accurate half-perimeter sum (μm). ``overlap_rate``
    is total hard–hard overlap area divided by ``canvas_width * canvas_height``.

    Pass ``rudy_grid`` to reuse a precomputed grid (e.g. share with
    ``compute_module_scores``).
    """
    cw, ch, nr, nc, cell_w, cell_h = _placement_grid(benchmark, nr, nc)
    chip_area = cw * ch
    ports_np = benchmark.port_positions.detach().cpu().numpy()
    n_macros = int(benchmark.num_macros)
    n_ports = int(ports_np.shape[0])

    hpwl = _weighted_hpwl_total(pos, benchmark, ports_np, n_macros, n_ports)
    rudy_g = rudy_grid if rudy_grid is not None else compute_rudy_grid(pos, benchmark, nr=nr, nc=nc)
    r_max = layout_rudy_max(rudy_g)
    ov_area = compute_total_hard_overlap_area(pos, benchmark)
    ov_rate = ov_area / max(chip_area, 1e-18)
    obj = float(hpwl + lam1 * r_max + lam2 * ov_rate)
    return {
        "hpwl_weighted": float(hpwl),
        "rudy_max": r_max,
        "overlap_area_hard": float(ov_area),
        "overlap_rate": float(ov_rate),
        "objective": obj,
        "lambda1": float(lam1),
        "lambda2": float(lam2),
    }


def layout_fitness(
    pos: np.ndarray,
    benchmark: Benchmark,
    *,
    lam1: float = 0.0,
    lam2: float = 0.1,
    nr: int | None = None,
    nc: int | None = None,
    rudy_grid: np.ndarray | None = None,
) -> float:
    """``f_L = -objective`` (higher is better)."""
    m = layout_objective(
        pos, benchmark, lam1=lam1, lam2=lam2, nr=nr, nc=nc, rudy_grid=rudy_grid
    )
    return -float(m["objective"])


def layout_selection_probs(fitness: np.ndarray) -> np.ndarray:
    """Eq. (2): ``p_i ∝ exp(f_i)``, numerically stable."""
    f = np.asarray(fitness, dtype=np.float64)
    if f.size == 0:
        return f
    z = f - np.max(f)
    w = np.exp(z)
    s = np.sum(w)
    if s <= 0.0:
        return np.ones_like(f) / max(f.size, 1)
    return w / s


def module_wirelen_term(
    m: int,
    pos: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
) -> float:
    """Eq. (4): mean Manhattan distance from each pin on ``m`` to its net's bbox center."""
    if benchmark.net_pin_nodes and len(benchmark.net_pin_nodes) == int(benchmark.num_nets):
        tot = 0.0
        cnt = 0
        for ni in range(int(benchmark.num_nets)):
            bb = _net_pin_bbox(ni, pos, benchmark, ports_np, n_macros, n_ports)
            if bb is None:
                continue
            _, _, _, _, cx, cy = bb
            pnodes = benchmark.net_pin_nodes[ni]
            for row in pnodes.tolist():
                owner, slot = int(row[0]), int(row[1])
                if owner != m:
                    continue
                px, py = _pin_xy(owner, slot, pos, benchmark, ports_np, n_macros, n_ports)
                tot += abs(px - cx) + abs(py - cy)
                cnt += 1
        return tot / max(cnt, 1)

    tot = 0.0
    cnt = 0
    macro_to_nets = _macro_to_nets(benchmark)
    for ni in macro_to_nets[m]:
        bb = _net_pin_bbox(ni, pos, benchmark, ports_np, n_macros, n_ports)
        if bb is None:
            continue
        _, _, _, _, cx, cy = bb
        px, py = float(pos[m, 0]), float(pos[m, 1])
        tot += abs(px - cx) + abs(py - cy)
        cnt += 1
    return tot / max(cnt, 1)


def module_cong_term(
    m: int,
    pos: np.ndarray,
    benchmark: Benchmark,
    net_rudy_peak: np.ndarray,
    layout_rudy_m: float,
    *,
    r_frac: float = 0.98,
) -> int:
    """Eq. (5): count pins on ``m`` whose net's peak RUDY exceeds ``r_frac·RUDY(L)``."""
    if layout_rudy_m <= 0.0:
        return 0
    thr = r_frac * layout_rudy_m
    cnt = 0
    ports_np = benchmark.port_positions.detach().cpu().numpy()
    n_macros = int(benchmark.num_macros)
    n_ports = int(ports_np.shape[0])

    if benchmark.net_pin_nodes and len(benchmark.net_pin_nodes) == int(benchmark.num_nets):
        for ni in range(int(benchmark.num_nets)):
            if net_rudy_peak[ni] <= thr:
                continue
            pnodes = benchmark.net_pin_nodes[ni]
            for row in pnodes.tolist():
                if int(row[0]) == m:
                    cnt += 1
        return cnt

    macro_to_nets = _macro_to_nets(benchmark)
    for ni in macro_to_nets[m]:
        if net_rudy_peak[ni] <= thr:
            continue
        cnt += 1
    return cnt


def module_overlap_area_with_others(
    m: int,
    pos: np.ndarray,
    benchmark: Benchmark,
    *,
    hard_only: bool = True,
) -> float:
    """Total overlap area (μm²) between macro ``m`` and other macros."""
    n_macros = int(benchmark.num_macros)
    n_hard = int(benchmark.num_hard_macros)
    j_hi = n_hard if hard_only else n_macros
    cm_x, cm_y = float(pos[m, 0]), float(pos[m, 1])
    wm, hm = float(benchmark.macro_sizes[m, 0]), float(benchmark.macro_sizes[m, 1])
    lm, rm = cm_x - 0.5 * wm, cm_x + 0.5 * wm
    bm, tm = cm_y - 0.5 * hm, cm_y + 0.5 * hm
    s = 0.0
    for j in range(j_hi):
        if j == m:
            continue
        cj_x, cj_y = float(pos[j, 0]), float(pos[j, 1])
        wj, hj = float(benchmark.macro_sizes[j, 0]), float(benchmark.macro_sizes[j, 1])
        lj, rj = cj_x - 0.5 * wj, cj_x + 0.5 * wj
        bj, tj = cj_y - 0.5 * hj, cj_y + 0.5 * hj
        ix = max(0.0, min(rm, rj) - max(lm, lj))
        iy = max(0.0, min(tm, tj) - max(bm, bj))
        s += ix * iy
    return s


def compute_net_rudy_peaks(
    pos: np.ndarray,
    benchmark: Benchmark,
    rudy_grid: np.ndarray,
    *,
    nr: int | None = None,
    nc: int | None = None,
) -> np.ndarray:
    """``net_rudy_peak[ni]`` = max RUDY in bins intersecting net ``ni``'s pin bbox."""
    cw, ch, nr, nc, cell_w, cell_h = _placement_grid(benchmark, nr, nc)
    ports_np = benchmark.port_positions.detach().cpu().numpy()
    n_macros = int(benchmark.num_macros)
    n_ports = int(ports_np.shape[0])
    nn = int(benchmark.num_nets)
    out = np.zeros(nn, dtype=np.float64)
    for ni in range(nn):
        bb = _net_pin_bbox(ni, pos, benchmark, ports_np, n_macros, n_ports)
        if bb is None:
            continue
        lx, rx, by, ty, _, _ = bb
        out[ni] = _net_max_rudy_in_bbox(rudy_grid, lx, rx, by, ty, cell_w, cell_h, nr, nc)
    return out


def compute_module_scores(
    pos: np.ndarray,
    benchmark: Benchmark,
    *,
    lam1: float = 0.0,
    lam2: float = 0.1,
    r_frac: float = 0.98,
    nr: int | None = None,
    nc: int | None = None,
    hard_only_overlap: bool = True,
    rudy_grid: np.ndarray | None = None,
) -> np.ndarray:
    """
    EGPlace Eq. (3): ``score_m = (wirelen_m + λ1·cong_m + λ2·overlap_m_raw) / (w_m·h_m)``.

    Returns a vector of length ``num_macros`` (soft macros get scores too if you
    use them in selection).

    Pass ``rudy_grid`` to reuse a grid built for ``layout_objective``.
    """
    n_macros = int(benchmark.num_macros)
    ports_np = benchmark.port_positions.detach().cpu().numpy()
    n_ports = int(ports_np.shape[0])

    rudy_g = rudy_grid if rudy_grid is not None else compute_rudy_grid(pos, benchmark, nr=nr, nc=nc)
    r_layout = layout_rudy_max(rudy_g)
    net_peak = compute_net_rudy_peaks(pos, benchmark, rudy_g, nr=nr, nc=nc)

    scores = np.zeros(n_macros, dtype=np.float64)
    for m in range(n_macros):
        wl = module_wirelen_term(m, pos, benchmark, ports_np, n_macros, n_ports)
        cg = module_cong_term(m, pos, benchmark, net_peak, r_layout, r_frac=r_frac)
        ov = module_overlap_area_with_others(m, pos, benchmark, hard_only=hard_only_overlap)
        w = float(benchmark.macro_sizes[m, 0])
        h = float(benchmark.macro_sizes[m, 1])
        area = max(w * h, 1e-18)
        scores[m] = (wl + lam1 * float(cg) + lam2 * ov) / area
    return scores


def module_selection_probs(scores: np.ndarray) -> np.ndarray:
    """``p_m ∝ score_m`` for nonnegative scores (normalize); add ε if needed."""
    s = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
    tot = np.sum(s)
    if tot <= 0.0:
        n = s.size
        return np.ones(n, dtype=np.float64) / max(n, 1)
    return s / tot


def _placement_grid(
    benchmark: Benchmark,
    nr: int | None,
    nc: int | None,
) -> tuple[float, float, int, int, float, float]:
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    nr = max(1, int(benchmark.grid_rows) if nr is None else int(nr))
    nc = max(1, int(benchmark.grid_cols) if nc is None else int(nc))
    return cw, ch, nr, nc, cw / nc, ch / nr


def compute_bound_mask(
    m: int,
    benchmark: Benchmark,
    *,
    nr: int | None = None,
    nc: int | None = None,
) -> np.ndarray:
    """
    **Bound mask** (EGPlace): for each bin ``(r, c)``, place macro ``m`` with
    **bottom-left** at ``(col * cell_w, row * cell_h)``.

    Returns a float array of shape ``(nr, nc)`` with:

    - ``1.0`` — placement **violates** the chip rectangle (any part outside).
    - ``0.0`` — macro **fits entirely** inside ``[0, cw] × [0, ch]``.

    Does not depend on other macros or on ``pos`` (pure geometry).
    """
    n_macros = int(benchmark.num_macros)
    if m < 0 or m >= n_macros:
        raise ValueError(f"macro index {m} out of range [0, {n_macros})")

    cw, ch, nr, nc, cell_w, cell_h = _placement_grid(benchmark, nr, nc)
    w_m = float(benchmark.macro_sizes[m, 0])
    h_m = float(benchmark.macro_sizes[m, 1])

    out = np.ones((nr, nc), dtype=np.float64)
    for r in range(nr):
        for c in range(nc):
            bl_x = c * cell_w
            bl_y = r * cell_h
            cx = bl_x + 0.5 * w_m
            cy = bl_y + 0.5 * h_m
            if cx < w_m * 0.5 - 1e-9 or cx > cw - w_m * 0.5 + 1e-9:
                continue
            if cy < h_m * 0.5 - 1e-9 or cy > ch - h_m * 0.5 + 1e-9:
                continue
            out[r, c] = 0.0
    return out


def _overlap_x_profile(
    w_m: float,
    lx_n: float,
    rx_n: float,
    cell_w: float,
    nc: int,
) -> np.ndarray:
    """Overlap length along x when ``m`` has bottom-left ``bl_x = c * cell_w``."""
    c = np.arange(nc, dtype=np.float64)
    bl_x = c * cell_w
    return np.maximum(0.0, np.minimum(bl_x + w_m, rx_n) - np.maximum(bl_x, lx_n))


def _overlap_y_profile(
    h_m: float,
    by_n: float,
    ty_n: float,
    cell_h: float,
    nr: int,
) -> np.ndarray:
    """Overlap length along y when ``m`` has bottom-left ``bl_y = r * cell_h``."""
    r = np.arange(nr, dtype=np.float64)
    bl_y = r * cell_h
    return np.maximum(0.0, np.minimum(bl_y + h_m, ty_n) - np.maximum(bl_y, by_n))


def compute_overlap_mask(
    m: int,
    pos: np.ndarray,
    benchmark: Benchmark,
    *,
    nr: int | None = None,
    nc: int | None = None,
    hard_only: bool = True,
    outside: float = math.inf,
) -> np.ndarray:
    """
    **Overlap mask** for macro ``m``: ``mask[r, c]`` is the **sum of overlap
    areas** (μm²) between ``m`` placed with bottom-left at bin ``(r,c)`` and
    every other macro ``j ≠ m`` (centers taken from ``pos``).

    For axis-aligned rectangles, each pairwise area factors as
    ``ox_j[c] · oy_j[r]`` with 1D profiles of length ``nc`` and ``nr``, then
    summed over ``j`` via ``np.outer`` — **O((nr+nc)·J + nr·nc·J)** with small
    constants (same structural trick as EGPlace Appendix Alg. 2 / outer product).

    Parameters:
        hard_only: If ``True`` (default), only **hard** macros ``j < num_hard``
            count (matches validator hard–hard overlap). If ``False``, all
            ``j ≠ m`` are included.
        outside: Value for bins where the **bound mask** is violated (default
            ``inf``).
    """
    n_macros = int(benchmark.num_macros)
    n_hard = int(benchmark.num_hard_macros)
    if m < 0 or m >= n_macros:
        raise ValueError(f"macro index {m} out of range [0, {n_macros})")

    cw, ch, nr, nc, cell_w, cell_h = _placement_grid(benchmark, nr, nc)
    w_m = float(benchmark.macro_sizes[m, 0])
    h_m = float(benchmark.macro_sizes[m, 1])

    j_hi = n_hard if hard_only else n_macros
    mask = np.zeros((nr, nc), dtype=np.float64)
    for j in range(j_hi):
        if j == m:
            continue
        cx = float(pos[j, 0])
        cy = float(pos[j, 1])
        wj = float(benchmark.macro_sizes[j, 0])
        hj = float(benchmark.macro_sizes[j, 1])
        lx_n, rx_n = cx - 0.5 * wj, cx + 0.5 * wj
        by_n, ty_n = cy - 0.5 * hj, cy + 0.5 * hj
        ox = _overlap_x_profile(w_m, lx_n, rx_n, cell_w, nc)
        oy = _overlap_y_profile(h_m, by_n, ty_n, cell_h, nr)
        mask += np.outer(oy, ox)

    bmask = compute_bound_mask(m, benchmark, nr=nr, nc=nc)
    mask = np.where(bmask >= 0.5, outside, mask)
    return mask


def _weighted_hpwl_total(
    pos: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
) -> float:
    w = benchmark.net_weights.detach().cpu().numpy().astype(np.float64)
    s = 0.0
    for ni in range(int(benchmark.num_nets)):
        s += float(w[ni]) * _single_net_hpwl(ni, pos, benchmark, ports_np, n_macros, n_ports)
    return s


def compute_wire_mask_hpwl_delta(
    m: int,
    pos: np.ndarray,
    benchmark: Benchmark,
    *,
    nr: int | None = None,
    nc: int | None = None,
    invalid: float = math.inf,
) -> np.ndarray:
    """
    ``mask[r, c] = Σ_e w_e · (HPWL_e(pos with m moved) - HPWL_e(pos))``.

    The trial center of macro ``m`` is implied by placing its **bottom-left**
    at ``(col * cell_w, row * cell_h)`` so
    ``center = (bl_x + w/2, bl_y + h/2)`` — consistent with EGPlace Algorithm 1
    (“bottom left corner … into bin”).

    Bins where the macro would cross the die boundary are set to ``invalid``.
    """
    n_macros = int(benchmark.num_macros)
    if m < 0 or m >= n_macros:
        raise ValueError(f"macro index {m} out of range [0, {n_macros})")

    cw, ch, nr, nc, cell_w, cell_h = _placement_grid(benchmark, nr, nc)
    w_m = float(benchmark.macro_sizes[m, 0])
    h_m = float(benchmark.macro_sizes[m, 1])

    ports_np = benchmark.port_positions.detach().cpu().numpy()
    n_ports = int(ports_np.shape[0])
    weights = benchmark.net_weights.detach().cpu().numpy().astype(np.float64)
    macro_to_nets = _macro_to_nets(benchmark)

    nets_m = macro_to_nets[m]
    if not nets_m:
        return np.zeros((nr, nc), dtype=np.float64)

    base_hpwl = {ni: _single_net_hpwl(ni, pos, benchmark, ports_np, n_macros, n_ports) for ni in nets_m}

    mask = np.full((nr, nc), invalid, dtype=np.float64)
    work = pos.copy()

    for r in range(nr):
        for c in range(nc):
            bl_x = c * cell_w
            bl_y = r * cell_h
            cx = bl_x + 0.5 * w_m
            cy = bl_y + 0.5 * h_m
            if cx < w_m * 0.5 - 1e-9 or cx > cw - w_m * 0.5 + 1e-9:
                continue
            if cy < h_m * 0.5 - 1e-9 or cy > ch - h_m * 0.5 + 1e-9:
                continue

            ox, oy = float(work[m, 0]), float(work[m, 1])
            work[m, 0], work[m, 1] = cx, cy
            delta = 0.0
            for ni in nets_m:
                new_h = _single_net_hpwl(ni, work, benchmark, ports_np, n_macros, n_ports)
                delta += float(weights[ni]) * (new_h - base_hpwl[ni])
            work[m, 0], work[m, 1] = ox, oy
            mask[r, c] = delta

    return mask


def _demo() -> None:
    from macro_place.loader import load_benchmark_from_dir

    root = _repo_root()
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    pos = b.macro_positions.detach().cpu().numpy().copy()
    m = 0
    bmask = compute_bound_mask(m, b)
    n_legal = int(np.sum(bmask == 0.0))
    print(f"{b.name}: bound_mask for macro {m} shape {bmask.shape}  legal bins {n_legal}")
    omask = compute_overlap_mask(m, pos, b, hard_only=True)
    leg = omask[np.isfinite(omask) & (bmask < 0.5)]
    print(
        f"{b.name}: overlap_mask (hard-only) for macro {m}  "
        f"legal min {leg.min():.4f} max {leg.max():.4f} μm²"
        if leg.size
        else f"{b.name}: overlap_mask empty"
    )
    mask = compute_wire_mask_hpwl_delta(m, pos, b)
    finite = mask[np.isfinite(mask)]
    print(f"{b.name}: wire_mask for macro {m} shape {mask.shape}")
    if finite.size:
        print(f"  finite cells: {finite.size}  min ΔHPWL {finite.min():.6f}  max {finite.max():.6f}")
    else:
        print("  no finite cells")

    lam1, lam2 = 0.5, 0.1
    rudy = compute_rudy_grid(pos, b)
    metrics = layout_objective(pos, b, lam1=lam1, lam2=lam2, rudy_grid=rudy)
    fit = layout_fitness(pos, b, lam1=lam1, lam2=lam2, rudy_grid=rudy)
    print(
        f"{b.name}: objective={metrics['objective']:.6f} fitness={fit:.6f} "
        f"(HPWL={metrics['hpwl_weighted']:.4f} RUDY_max={metrics['rudy_max']:.6f} "
        f"overlap_rate={metrics['overlap_rate']:.6e})"
    )
    fvec = np.array([fit, fit - 1.0, fit + 0.5])
    p = layout_selection_probs(fvec)
    print(f"  softmax(selection) example p={np.array2string(p, precision=4)}")
    scores = compute_module_scores(pos, b, lam1=lam1, lam2=lam2, rudy_grid=rudy)
    top = np.argsort(-scores)[: min(5, scores.size)]
    print(f"  top module scores (m, score): {[(int(i), float(scores[i])) for i in top]}")


class EvolutionPlacer:
    """Placeholder placer for the harness; full EGPlace not wired yet."""

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return benchmark.macro_positions.clone()


if __name__ == "__main__":
    _demo()
