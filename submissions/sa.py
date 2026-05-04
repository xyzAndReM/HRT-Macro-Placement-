"""
Simulated annealing from the **initial** placement using a **fast surrogate**
aligned with the evaluator proxy weights:

    Surrogate = 1.0 × wirelength_cost + 0.5 × density_cost + 0.5 × rudy_cost

where:
- **wirelength_cost** matches ``PlacementCost.get_cost`` normalization
  (weighted HPWL / ((W+H)·∑w)).
- **density_cost** matches ``PlacementCost.get_density_cost`` (top-10% occupied
  cells, averaged, then ×0.5).
- **rudy_cost** is a lightweight **RUDY** congestion proxy on the placement grid,
  scored like ``abu(x, 0.05)`` (top-5% average).

**Hard–hard overlaps** must **not increase** (pairwise overlap area vs the
current placement; same tolerance as ``free.py``). **Soft** macro moves skip this
check (the validator only flags **hard–hard** overlaps). **Canvas** bounds still
apply to every move.

**Pool:** every **movable** macro (hard and soft). Proposals move **one macro**
by ``±delta_um`` on one axis. If a macro proposes an **illegal** move (out of
canvas, or for **hard** macros: overlap would grow) **four** times in a row, it
is removed from the pool.

Usage:
    uv run evaluate submissions/sa.py -b ibm01
    uv run python submissions/sa.py
    uv run python scripts/sa_before_after.py ibm06
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark

_GAP_AREA = 1e-6


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _macro_to_nets(benchmark: Benchmark) -> list[list[int]]:
    n_macros = int(benchmark.num_macros)
    out: list[list[int]] = [[] for _ in range(n_macros)]
    for ni in range(int(benchmark.num_nets)):
        nodes = benchmark.net_nodes[ni]
        for v in nodes.tolist():
            v = int(v)
            if 0 <= v < n_macros:
                out[v].append(ni)
    return out


def _bbox(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return cx - 0.5 * w, cx + 0.5 * w, cy - 0.5 * h, cy + 0.5 * h


def _overlap_area(
    lx1: float,
    rx1: float,
    ly1: float,
    ry1: float,
    lx2: float,
    rx2: float,
    ly2: float,
    ry2: float,
) -> float:
    ix = max(0.0, min(rx1, rx2) - max(lx1, lx2))
    iy = max(0.0, min(ry1, ry2) - max(ly1, ly2))
    return ix * iy


def _pair_overlaps_current(
    pos: np.ndarray,
    sizes: np.ndarray,
    n_hard: int,
) -> np.ndarray:
    o = np.zeros((n_hard, n_hard), dtype=np.float64)
    for ii in range(n_hard):
        wi, hi = float(sizes[ii, 0]), float(sizes[ii, 1])
        li, ri, bi, ti = _bbox(float(pos[ii, 0]), float(pos[ii, 1]), wi, hi)
        for jj in range(ii + 1, n_hard):
            wj, hj = float(sizes[jj, 0]), float(sizes[jj, 1])
            lj, rj, bj, tj = _bbox(float(pos[jj, 0]), float(pos[jj, 1]), wj, hj)
            a = _overlap_area(li, ri, bi, ti, lj, rj, bj, tj)
            o[ii, jj] = a
            o[jj, ii] = a
    return o


def _refresh_overlap_row(
    i: int,
    pos: np.ndarray,
    sizes: np.ndarray,
    n_hard: int,
    o_mat: np.ndarray,
) -> None:
    wi, hi = float(sizes[i, 0]), float(sizes[i, 1])
    li, ri, bi, ti = _bbox(float(pos[i, 0]), float(pos[i, 1]), wi, hi)
    for j in range(n_hard):
        if j == i:
            o_mat[i, j] = 0.0
            continue
        wj, hj = float(sizes[j, 0]), float(sizes[j, 1])
        lj, rj, bj, tj = _bbox(float(pos[j, 0]), float(pos[j, 1]), wj, hj)
        a = _overlap_area(li, ri, bi, ti, lj, rj, bj, tj)
        o_mat[i, j] = a
        o_mat[j, i] = a


def _hard_move_overlap_ok(
    i: int,
    dcx: float,
    dcy: float,
    pos: np.ndarray,
    sizes: np.ndarray,
    n_hard: int,
    o_mat: np.ndarray,
) -> bool:
    """False if moving hard macro i increases overlap area with any other hard macro."""
    if i < 0 or i >= n_hard:
        return True
    cx = float(pos[i, 0]) + dcx
    cy = float(pos[i, 1]) + dcy
    w, h = float(sizes[i, 0]), float(sizes[i, 1])
    li, ri, bi, ti = _bbox(cx, cy, w, h)
    for j in range(n_hard):
        if j == i:
            continue
        wj, hj = float(sizes[j, 0]), float(sizes[j, 1])
        lj, rj, bj, tj = _bbox(float(pos[j, 0]), float(pos[j, 1]), wj, hj)
        new_a = _overlap_area(li, ri, bi, ti, lj, rj, bj, tj)
        if new_a > float(o_mat[i, j]) + _GAP_AREA:
            return False
    return True


def _canvas_ok_single(
    pos: np.ndarray,
    benchmark: Benchmark,
    i: int,
    dcx: float,
    dcy: float,
    cw: float,
    ch: float,
) -> bool:
    w = float(benchmark.macro_sizes[i, 0])
    h = float(benchmark.macro_sizes[i, 1])
    cx = float(pos[i, 0]) + dcx
    cy = float(pos[i, 1]) + dcy
    if cx < w * 0.5 - 1e-9 or cx > cw - w * 0.5 + 1e-9:
        return False
    if cy < h * 0.5 - 1e-9 or cy > ch - h * 0.5 + 1e-9:
        return False
    return True


def _single_net_hpwl(
    ni: int,
    pos_np: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
) -> float:
    nodes = benchmark.net_nodes[ni]
    if nodes.numel() < 2:
        return 0.0
    xs: list[float] = []
    ys: list[float] = []
    for v in nodes.tolist():
        v = int(v)
        if v < n_macros:
            xs.append(float(pos_np[v, 0]))
            ys.append(float(pos_np[v, 1]))
        else:
            pi = v - n_macros
            if 0 <= pi < n_ports:
                xs.append(float(ports_np[pi, 0]))
                ys.append(float(ports_np[pi, 1]))
    if len(xs) < 2:
        return 0.0
    return max(xs) - min(xs) + max(ys) - min(ys)


def _init_net_hpwl(
    pos_np: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
) -> np.ndarray:
    nn = int(benchmark.num_nets)
    out = np.zeros(nn, dtype=np.float64)
    for ni in range(nn):
        out[ni] = _single_net_hpwl(ni, pos_np, benchmark, ports_np, n_macros, n_ports)
    return out


def _hpwl_delta_single_move(
    i: int,
    ncx: float,
    ncy: float,
    pos_full: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
    net_hpwl: np.ndarray,
    macro_to_nets: list[list[int]],
    weights: np.ndarray,
) -> tuple[float, list[tuple[int, float]]]:
    ox, oy = float(pos_full[i, 0]), float(pos_full[i, 1])
    pos_full[i, 0], pos_full[i, 1] = ncx, ncy
    delta = 0.0
    updates: list[tuple[int, float]] = []
    for ni in macro_to_nets[i]:
        old = float(net_hpwl[ni])
        new = _single_net_hpwl(ni, pos_full, benchmark, ports_np, n_macros, n_ports)
        delta += float(weights[ni]) * (new - old)
        updates.append((ni, new))
    pos_full[i, 0], pos_full[i, 1] = ox, oy
    return delta, updates


def _apply_macro_to_density(
    dens: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
    sign: float,
) -> None:
    cell_w = cw / nc
    cell_h = ch / nr
    x0, x1 = cx - 0.5 * w, cx + 0.5 * w
    y0, y1 = cy - 0.5 * h, cy + 0.5 * h
    c0 = max(0, int(math.floor(x0 / cell_w)))
    c1 = min(nc - 1, int(math.floor(x1 / cell_w)))
    r0 = max(0, int(math.floor(y0 / cell_h)))
    r1 = min(nr - 1, int(math.floor(y1 / cell_h)))
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            bx0, bx1 = c * cell_w, (c + 1) * cell_w
            by0, by1 = r * cell_h, (r + 1) * cell_h
            ix0, ix1 = max(x0, bx0), min(x1, bx1)
            iy0, iy1 = max(y0, by0), min(y1, by1)
            if ix1 > ix0 and iy1 > iy0:
                contrib = (ix1 - ix0) * (iy1 - iy0) / (cell_w * cell_h)
                dens[r, c] += sign * contrib


def _move_macro_in_density_inplace(
    dens: np.ndarray,
    cx0: float,
    cy0: float,
    cx1: float,
    cy1: float,
    w: float,
    h: float,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
) -> None:
    _apply_macro_to_density(dens, cx0, cy0, w, h, cw, ch, nr, nc, -1.0)
    _apply_macro_to_density(dens, cx1, cy1, w, h, cw, ch, nr, nc, +1.0)


def _build_area_density_map(
    placement: np.ndarray,
    benchmark: Benchmark,
) -> np.ndarray:
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    nr = max(int(benchmark.grid_rows), 1)
    nc = max(int(benchmark.grid_cols), 1)
    cell_w = cw / nc
    cell_h = ch / nr
    dens = np.zeros((nr, nc), dtype=np.float64)
    n = int(benchmark.num_macros)
    for i in range(n):
        cx, cy = float(placement[i, 0]), float(placement[i, 1])
        w = float(benchmark.macro_sizes[i, 0])
        h = float(benchmark.macro_sizes[i, 1])
        x0, x1 = cx - 0.5 * w, cx + 0.5 * w
        y0, y1 = cy - 0.5 * h, cy + 0.5 * h
        c0 = max(0, int(math.floor(x0 / cell_w)))
        c1 = min(nc - 1, int(math.floor(x1 / cell_w)))
        r0 = max(0, int(math.floor(y0 / cell_h)))
        r1 = min(nr - 1, int(math.floor(y1 / cell_h)))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                bx0, bx1 = c * cell_w, (c + 1) * cell_w
                by0, by1 = r * cell_h, (r + 1) * cell_h
                ix0, ix1 = max(x0, bx0), min(x1, bx1)
                iy0, iy1 = max(y0, by0), min(y1, by1)
                if ix1 > ix0 and iy1 > iy0:
                    dens[r, c] += (ix1 - ix0) * (iy1 - iy0) / (cell_w * cell_h)
    return dens


def _plc_style_density_cost(dens: np.ndarray) -> float:
    """Match ``PlacementCost.get_density_cost`` (top 10% occupied cells, ×0.5)."""
    flat = dens.ravel()
    ncells = flat.size
    density_cnt = max(1, int(math.floor(ncells * 0.1)))
    nz = flat[flat > 0.0]
    if ncells < 10:
        if nz.size == 0:
            return 0.0
        return 0.5 * float(np.sum(nz) / nz.size)
    if nz.size == 0:
        return 0.0
    k = min(density_cnt, nz.size)
    top = np.partition(nz, nz.size - k)[nz.size - k :]
    s = float(np.sum(top))
    return 0.5 * (s / density_cnt)


def _abu_like_cost(grid: np.ndarray, frac: float) -> float:
    """Average of the top frac of entries (like Plc_client abu())."""
    flat = grid.ravel()
    n = flat.size
    if n == 0:
        return 0.0
    k = int(math.floor(n * frac))
    if k <= 0:
        return float(np.max(flat))
    top = np.partition(flat, n - k)[n - k :]
    return float(np.sum(top) / k)


def _add_net_to_rudy(
    rudy: np.ndarray,
    bbox: tuple[float, float, float, float],
    demand: float,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
    sign: float,
) -> None:
    """Uniformly smear demand over bins intersecting bbox using area fractions."""
    lx, rx, by, ty = bbox
    if rx <= lx or ty <= by:
        return
    cell_w = cw / nc
    cell_h = ch / nr
    c0 = max(0, int(math.floor(lx / cell_w)))
    c1 = min(nc - 1, int(math.floor(rx / cell_w)))
    r0 = max(0, int(math.floor(by / cell_h)))
    r1 = min(nr - 1, int(math.floor(ty / cell_h)))
    if c1 < c0 or r1 < r0:
        return
    bin_area = cell_w * cell_h
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            bx0, bx1 = c * cell_w, (c + 1) * cell_w
            by0, by1 = r * cell_h, (r + 1) * cell_h
            ix0, ix1 = max(lx, bx0), min(rx, bx1)
            iy0, iy1 = max(by, by0), min(ty, by1)
            if ix1 > ix0 and iy1 > iy0:
                frac = (ix1 - ix0) * (iy1 - iy0) / bin_area
                rudy[r, c] += sign * (demand * frac)


def _net_bbox(
    ni: int,
    pos_full: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
) -> tuple[float, float, float, float] | None:
    nodes = benchmark.net_nodes[ni]
    if nodes.numel() < 2:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for v in nodes.tolist():
        v = int(v)
        if v < n_macros:
            xs.append(float(pos_full[v, 0]))
            ys.append(float(pos_full[v, 1]))
        else:
            pi = v - n_macros
            if 0 <= pi < n_ports:
                xs.append(float(ports_np[pi, 0]))
                ys.append(float(ports_np[pi, 1]))
    if len(xs) < 2:
        return None
    lx, rx = min(xs), max(xs)
    by, ty = min(ys), max(ys)
    return lx, rx, by, ty


def _build_rudy_map(
    pos_full: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
    weights: np.ndarray,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
) -> tuple[np.ndarray, list[tuple[float, float, float, float] | None], np.ndarray]:
    """Return (rudy_grid, per-net bbox, per-net demand)."""
    nn = int(benchmark.num_nets)
    rudy = np.zeros((nr, nc), dtype=np.float64)
    bboxes: list[tuple[float, float, float, float] | None] = [None] * nn
    demands = np.zeros(nn, dtype=np.float64)
    for ni in range(nn):
        bb = _net_bbox(ni, pos_full, benchmark, ports_np, n_macros, n_ports)
        bboxes[ni] = bb
        if bb is None:
            continue
        lx, rx, by, ty = bb
        area = max((rx - lx) * (ty - by), 1e-12)
        dem = float(weights[ni]) / area
        demands[ni] = dem
        _add_net_to_rudy(rudy, bb, dem, cw, ch, nr, nc, +1.0)
    return rudy, bboxes, demands


def _wl_cost_term(wl_weighted_sum: float, cw: float, ch: float, net_norm: float) -> float:
    return wl_weighted_sum / ((cw + ch) * max(net_norm, 1.0))


class SAPlacer:
    """
    Proxy-style SA: minimize ``wl_cost + 0.5*density_cost + 0.5*rudy_cost`` per
    inner step (PLC-normalized WL + density; RUDY for congestion).

    Args:
        delta_um: Translation step on one axis (μm).
        max_iters: Metropolis trials.
        t0_factor: ``T0 = t0_factor * max(initial_surrogate, 1e-12)``.
        t_min: Terminal temperature.
        strike_limit: Illegal proposals in a row before dropping pivot.
        seed: RNG seed.
    """

    def __init__(
        self,
        delta_um: float = 0.001,
        max_iters: int = 80_000,
        t0_factor: float = 0.02,
        t_min: float = 1e-9,
        strike_limit: int = 4,
        seed: int = 0,
    ):
        self.delta_um = float(delta_um)
        self.max_iters = int(max_iters)
        self.t0_factor = float(t0_factor)
        self.t_min = float(t_min)
        self.strike_limit = int(strike_limit)
        self.seed = int(seed)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        n_macros = int(benchmark.num_macros)
        n_hard = int(benchmark.num_hard_macros)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        fixed = benchmark.macro_fixed.detach().cpu().numpy().astype(bool)
        sizes_hard = benchmark.macro_sizes[:n_hard].detach().cpu().numpy()

        pool_list = [i for i in range(n_macros) if not fixed[i]]
        pool: set[int] = set(pool_list)
        strikes = {int(i): 0 for i in pool}

        macro_to_nets = _macro_to_nets(benchmark)
        pos_full = placement.detach().cpu().numpy().copy()
        o_mat = _pair_overlaps_current(pos_full, sizes_hard, n_hard)
        ports_np = benchmark.port_positions.detach().cpu().numpy()
        n_ports = int(ports_np.shape[0])
        weights = benchmark.net_weights.detach().cpu().numpy().astype(np.float64)
        net_norm = max(1.0, float(np.sum(weights)))

        net_hpwl = _init_net_hpwl(pos_full, benchmark, ports_np, n_macros, n_ports)
        wl_sum = float(np.dot(net_hpwl, weights))

        nr = max(int(benchmark.grid_rows), 1)
        nc = max(int(benchmark.grid_cols), 1)
        dens = _build_area_density_map(pos_full, benchmark)
        den_cost = _plc_style_density_cost(dens)

        rudy, rudy_bboxes, rudy_demands = _build_rudy_map(
            pos_full,
            benchmark,
            ports_np,
            n_macros,
            n_ports,
            weights,
            cw,
            ch,
            nr,
            nc,
        )
        rudy_cost = _abu_like_cost(rudy, 0.05)

        cur_e = _wl_cost_term(wl_sum, cw, ch, net_norm) + 0.5 * den_cost + 0.5 * rudy_cost

        rng = random.Random(self.seed)
        d = self.delta_um
        dirs = ((d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d))
        t0 = max(self.t_min * 10, self.t0_factor * max(cur_e, 1e-12))
        t_min = self.t_min
        n_it = max(1, self.max_iters)

        for t in range(n_it):
            if not pool:
                break
            T = t0 * (t_min / t0) ** (t / (n_it - 1)) if n_it > 1 else t_min

            i = rng.choice(list(pool))
            dcx, dcy = rng.choice(dirs)

            if not _canvas_ok_single(pos_full, benchmark, i, dcx, dcy, cw, ch):
                strikes[i] = strikes.get(i, 0) + 1
                if strikes[i] >= self.strike_limit:
                    pool.discard(i)
                continue

            if i < n_hard and not _hard_move_overlap_ok(
                i, dcx, dcy, pos_full, sizes_hard, n_hard, o_mat
            ):
                strikes[i] = strikes.get(i, 0) + 1
                if strikes[i] >= self.strike_limit:
                    pool.discard(i)
                continue

            strikes[i] = 0

            ncx, ncy = float(pos_full[i, 0]) + dcx, float(pos_full[i, 1]) + dcy

            d_wl_raw, hp_updates = _hpwl_delta_single_move(
                i,
                ncx,
                ncy,
                pos_full,
                benchmark,
                ports_np,
                n_macros,
                n_ports,
                net_hpwl,
                macro_to_nets,
                weights,
            )
            d_wl_cost = d_wl_raw / ((cw + ch) * net_norm)

            # density delta (temporarily apply to dens and rollback)
            ox, oy = float(pos_full[i, 0]), float(pos_full[i, 1])
            w = float(benchmark.macro_sizes[i, 0])
            h = float(benchmark.macro_sizes[i, 1])
            before_den = _plc_style_density_cost(dens)
            _move_macro_in_density_inplace(dens, ox, oy, ncx, ncy, w, h, cw, ch, nr, nc)
            after_den = _plc_style_density_cost(dens)
            _move_macro_in_density_inplace(dens, ncx, ncy, ox, oy, w, h, cw, ch, nr, nc)
            d_den = after_den - before_den

            # rudy delta (incremental on touched nets) — implement directly here to keep weights available
            before_rudy = _abu_like_cost(rudy, 0.05)
            # remove old touched-net contributions
            for ni in macro_to_nets[i]:
                bb0 = rudy_bboxes[ni]
                if bb0 is not None:
                    _add_net_to_rudy(rudy, bb0, float(rudy_demands[ni]), cw, ch, nr, nc, -1.0)
            # apply position
            pos_full[i, 0], pos_full[i, 1] = ncx, ncy
            # add new contributions
            for ni in macro_to_nets[i]:
                bb1 = _net_bbox(ni, pos_full, benchmark, ports_np, n_macros, n_ports)
                rudy_bboxes[ni] = bb1
                if bb1 is None:
                    rudy_demands[ni] = 0.0
                    continue
                lx, rx, by, ty = bb1
                area = max((rx - lx) * (ty - by), 1e-12)
                dem1 = float(weights[ni]) / area
                rudy_demands[ni] = dem1
                _add_net_to_rudy(rudy, bb1, dem1, cw, ch, nr, nc, +1.0)
            after_rudy = _abu_like_cost(rudy, 0.05)
            d_rudy = after_rudy - before_rudy
            # rollback position and rudy/bboxes/demands
            for ni in macro_to_nets[i]:
                bb1 = rudy_bboxes[ni]
                if bb1 is not None:
                    _add_net_to_rudy(rudy, bb1, float(rudy_demands[ni]), cw, ch, nr, nc, -1.0)
            pos_full[i, 0], pos_full[i, 1] = ox, oy
            for ni in macro_to_nets[i]:
                bb0 = _net_bbox(ni, pos_full, benchmark, ports_np, n_macros, n_ports)
                rudy_bboxes[ni] = bb0
                if bb0 is None:
                    rudy_demands[ni] = 0.0
                    continue
                lx, rx, by, ty = bb0
                area0 = max((rx - lx) * (ty - by), 1e-12)
                dem0 = float(weights[ni]) / area0
                rudy_demands[ni] = dem0
                _add_net_to_rudy(rudy, bb0, dem0, cw, ch, nr, nc, +1.0)

            delta_e = d_wl_cost + 0.5 * d_den + 0.5 * d_rudy

            accept = delta_e <= 0.0 or (T > 0 and rng.random() < math.exp(-delta_e / T))

            if accept:
                # commit: update density
                _move_macro_in_density_inplace(dens, ox, oy, ncx, ncy, w, h, cw, ch, nr, nc)
                # commit: update rudy touched nets
                for ni in macro_to_nets[i]:
                    bb0 = rudy_bboxes[ni]
                    if bb0 is not None:
                        _add_net_to_rudy(rudy, bb0, float(rudy_demands[ni]), cw, ch, nr, nc, -1.0)
                pos_full[i, 0], pos_full[i, 1] = ncx, ncy
                for ni in macro_to_nets[i]:
                    bb1 = _net_bbox(ni, pos_full, benchmark, ports_np, n_macros, n_ports)
                    rudy_bboxes[ni] = bb1
                    if bb1 is None:
                        rudy_demands[ni] = 0.0
                        continue
                    lx, rx, by, ty = bb1
                    area = max((rx - lx) * (ty - by), 1e-12)
                    dem1 = float(weights[ni]) / area
                    rudy_demands[ni] = dem1
                    _add_net_to_rudy(rudy, bb1, dem1, cw, ch, nr, nc, +1.0)

                placement[i, 0] = float(ncx)
                placement[i, 1] = float(ncy)
                for ni, wn in hp_updates:
                    net_hpwl[ni] = wn
                wl_sum += d_wl_raw
                cur_e += delta_e
                if i < n_hard:
                    _refresh_overlap_row(i, pos_full, sizes_hard, n_hard, o_mat)

        return placement


def _cli_main() -> None:
    from macro_place.loader import load_benchmark_from_dir

    case = _repo_root() / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    SAPlacer().place(b)
    print("SAPlacer finished on ibm01.")


if __name__ == "__main__":
    _cli_main()
