"""
Density / congestion relaxation placer (no overlap legalization).

Phase 1 — aggressive density relaxation (wirelength allowed to suffer):
  Large steps (~few× cell size), no step cooling, optional large displacement cap
  from *input* placement (``None`` = canvas clamp only). Accepts any move that
  strictly lowers local (footprint) peak density; HPWL is not considered.

Phase 2 — gentle wirelength repair:
  Small moves from Phase~1 end positions with a tight displacement cap.
  Tries axis/diagonal probes; accepts only strict HPWL improvement while keeping
  footprint peak density at or below a percentile cap from the Phase~2 start map
  (no new hotspots).

Phase 3 — optional congestion refinement (PlacementCost):
  Swap-based cleanup as before (RUDY prefilter, cheap HPWL gate, PLC congestion).

Usage:
    uv run evaluate submissions/relax.py -b ibm01
    uv run evaluate submissions/relax.py --all
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import _set_placement


def _macro_to_nets(benchmark: Benchmark) -> list[list[int]]:
    """For each macro index, list of net indices that touch that macro (from net_nodes)."""
    n_macros = int(benchmark.num_macros)
    out: list[list[int]] = [[] for _ in range(n_macros)]
    for ni in range(int(benchmark.num_nets)):
        nodes = benchmark.net_nodes[ni]
        for v in nodes.tolist():
            v = int(v)
            if 0 <= v < n_macros:
                out[v].append(ni)
    return out


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


def _init_net_hpwl(pos_np: np.ndarray, benchmark: Benchmark, ports_np: np.ndarray, n_macros: int, n_ports: int) -> np.ndarray:
    nn = int(benchmark.num_nets)
    out = np.zeros(nn, dtype=np.float64)
    for ni in range(nn):
        out[ni] = _single_net_hpwl(ni, pos_np, benchmark, ports_np, n_macros, n_ports)
    return out


def _macro_level_hpwl_from_cache(net_hpwl: np.ndarray) -> float:
    return float(np.sum(net_hpwl))


def _hpwl_delta_move_one_macro(
    i: int,
    ncx: float,
    ncy: float,
    pos_np: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
    net_hpwl: np.ndarray,
    macro_to_nets: list[list[int]],
) -> tuple[float, list[tuple[int, float]]]:
    """HPWL change if macro i moves to (ncx,ncy); does not mutate pos_np or net_hpwl."""
    ox, oy = float(pos_np[i, 0]), float(pos_np[i, 1])
    pos_np[i, 0], pos_np[i, 1] = ncx, ncy
    delta = 0.0
    updates: list[tuple[int, float]] = []
    for ni in macro_to_nets[i]:
        old = float(net_hpwl[ni])
        new = _single_net_hpwl(ni, pos_np, benchmark, ports_np, n_macros, n_ports)
        delta += new - old
        updates.append((ni, new))
    pos_np[i, 0], pos_np[i, 1] = ox, oy
    return delta, updates


def _hpwl_delta_swap(
    hi: int,
    ci: int,
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    pos_np: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
    net_hpwl: np.ndarray,
    macro_to_nets: list[list[int]],
) -> tuple[float, list[tuple[int, float]]]:
    """Delta total HPWL and per-net new values for swapping hi→(sx,sy), ci→(tx,ty)."""
    nets_touch = set(macro_to_nets[hi]) | set(macro_to_nets[ci])
    px0, py0 = float(pos_np[hi, 0]), float(pos_np[hi, 1])
    px1, py1 = float(pos_np[ci, 0]), float(pos_np[ci, 1])
    pos_np[hi, 0], pos_np[hi, 1] = sx, sy
    pos_np[ci, 0], pos_np[ci, 1] = tx, ty
    delta = 0.0
    updates: list[tuple[int, float]] = []
    for ni in nets_touch:
        old = float(net_hpwl[ni])
        new = _single_net_hpwl(ni, pos_np, benchmark, ports_np, n_macros, n_ports)
        delta += new - old
        updates.append((ni, new))
    pos_np[hi, 0], pos_np[hi, 1] = px0, py0
    pos_np[ci, 0], pos_np[ci, 1] = px1, py1
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
) -> float:
    """Add sign * (fractional area overlap) to each covered cell. Returns sum of signed changes."""
    cell_w = cw / nc
    cell_h = ch / nr
    x0, x1 = cx - 0.5 * w, cx + 0.5 * w
    y0, y1 = cy - 0.5 * h, cy + 0.5 * h
    c0 = max(0, int(math.floor(x0 / cell_w)))
    c1 = min(nc - 1, int(math.floor(x1 / cell_w)))
    r0 = max(0, int(math.floor(y0 / cell_h)))
    r1 = min(nr - 1, int(math.floor(y1 / cell_h)))
    delta_sum = 0.0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            bx0, bx1 = c * cell_w, (c + 1) * cell_w
            by0, by1 = r * cell_h, (r + 1) * cell_h
            ix0, ix1 = max(x0, bx0), min(x1, bx1)
            iy0, iy1 = max(y0, by0), min(y1, by1)
            if ix1 > ix0 and iy1 > iy0:
                contrib = (ix1 - ix0) * (iy1 - iy0) / (cell_w * cell_h)
                d = sign * contrib
                dens[r, c] += d
                delta_sum += d
    return delta_sum


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
) -> float:
    """Remove macro at (cx0,cy0), add at (cx1,cy1). Returns net change in sum(dens)."""
    d0 = _apply_macro_to_density(dens, cx0, cy0, w, h, cw, ch, nr, nc, -1.0)
    d1 = _apply_macro_to_density(dens, cx1, cy1, w, h, cw, ch, nr, nc, +1.0)
    return d0 + d1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _try_load_plc(benchmark: Benchmark):
    """Reload testcase to obtain PlacementCost for WL / density / congestion metrics."""
    case_dir = _repo_root() / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / benchmark.name
    if not case_dir.is_dir():
        return None
    try:
        _b2, plc = load_benchmark_from_dir(str(case_dir))
        return plc
    except Exception:
        return None


def _clamp_center(cx: float, cy: float, w: float, h: float, cw: float, ch: float) -> tuple[float, float]:
    return (
        max(w * 0.5, min(cx, cw - w * 0.5)),
        max(h * 0.5, min(cy, ch - h * 0.5)),
    )


def _build_area_density_map(
    placement: np.ndarray,
    benchmark: Benchmark,
    include_soft: bool = True,
) -> np.ndarray:
    """Per-cell occupancy = (macro area intersect cell) / cell area. Shape (nr, nc)."""
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    nr = max(int(benchmark.grid_rows), 1)
    nc = max(int(benchmark.grid_cols), 1)
    cell_w = cw / nc
    cell_h = ch / nr
    dens = np.zeros((nr, nc), dtype=np.float64)

    n = benchmark.num_macros if include_soft else benchmark.num_hard_macros
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


def _local_max_density_in_footprint(
    dens: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
) -> float:
    cell_w = cw / nc
    cell_h = ch / nr
    x0, x1 = cx - 0.5 * w, cx + 0.5 * w
    y0, y1 = cy - 0.5 * h, cy + 0.5 * h
    c0 = max(0, int(math.floor(x0 / cell_w)))
    c1 = min(nc - 1, int(math.floor(x1 / cell_w)))
    r0 = max(0, int(math.floor(y0 / cell_h)))
    r1 = min(nr - 1, int(math.floor(y1 / cell_h)))
    best = 0.0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            best = max(best, dens[r, c])
    return best


def _sample_gradient_direction(
    dens: np.ndarray,
    cx: float,
    cy: float,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
) -> tuple[float, float]:
    """Finite-difference gradient of density field at (cx, cy); points uphill."""
    cell_w = cw / nc
    cell_h = ch / nr
    # Bilinear sample density at float cell coords
    fx = cx / cell_w - 0.5
    fy = cy / cell_h - 0.5

    def sample(frow: float, fcol: float) -> float:
        fcol = max(0.0, min(nc - 1.001, fcol))
        frow = max(0.0, min(nr - 1.001, frow))
        c0 = int(math.floor(fcol))
        r0 = int(math.floor(frow))
        c1 = min(c0 + 1, nc - 1)
        r1 = min(r0 + 1, nr - 1)
        tx, ty = fcol - c0, frow - r0
        z00 = dens[r0, c0]
        z01 = dens[r0, c1]
        z10 = dens[r1, c0]
        z11 = dens[r1, c1]
        return (1 - ty) * ((1 - tx) * z00 + tx * z01) + ty * ((1 - tx) * z10 + tx * z11)

    eps_c = 0.5
    eps_r = 0.5
    gx = (sample(fy, fx + eps_c) - sample(fy, fx - eps_c)) / (2 * eps_c * cell_w)
    gy = (sample(fy + eps_r, fx) - sample(fy - eps_r, fx)) / (2 * eps_r * cell_h)
    nrm = math.hypot(gx, gy)
    if nrm < 1e-12:
        return 0.0, 0.0
    return gx / nrm, gy / nrm


def _unit_directions_8() -> list[tuple[float, float]]:
    d = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    s = 1.0 / math.sqrt(2.0)
    d.extend([(s, s), (s, -s), (-s, s), (-s, -s)])
    return d


def _macros_overlapping_bin(
    placement: np.ndarray,
    benchmark: Benchmark,
    n_hard: int,
    movable: np.ndarray,
    r: int,
    c: int,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
) -> list[int]:
    cell_w = cw / nc
    cell_h = ch / nr
    bx0, bx1 = c * cell_w, (c + 1) * cell_w
    by0, by1 = r * cell_h, (r + 1) * cell_h
    out: list[int] = []
    for i in range(n_hard):
        if not movable[i]:
            continue
        cx, cy = float(placement[i, 0]), float(placement[i, 1])
        w = float(benchmark.macro_sizes[i, 0])
        h = float(benchmark.macro_sizes[i, 1])
        mx0, mx1 = cx - 0.5 * w, cx + 0.5 * w
        my0, my1 = cy - 0.5 * h, cy + 0.5 * h
        if mx1 > bx0 and mx0 < bx1 and my1 > by0 and my0 < by1:
            out.append(i)
    return out


class RelaxPlacer:
    """Three-phase relax: aggressive density → WL repair → optional congestion swaps."""

    def __init__(
        self,
        phase1_iters: int = 75,
        phase1_max_displacement_frac: Optional[float] = 0.40,
        phase1_step_cell_mult: float = 3.5,
        phase1_dens_rebuild_interval: int = 10,
        phase1_target_density: Optional[float] = None,
        phase2_iters: int = 28,
        phase2_max_displacement_frac: float = 0.04,
        phase2_step_frac: float = 0.14,
        phase2_dens_rebuild_interval: int = 7,
        phase2_hotspot_percentile: float = 78.0,
        phase2_hotspot_abs_margin: float = 0.06,
        phase3_enable: bool = True,
        phase3_outer_passes: int = 12,
        phase3_pair_budget: int = 120,
        phase3_wl_slack: float = 0.02,
        top_cong_frac: float = 0.10,
        bot_cong_frac: float = 0.10,
    ):
        self.phase1_iters = int(phase1_iters)
        self.phase1_max_displacement_frac = phase1_max_displacement_frac
        self.phase1_step_cell_mult = float(phase1_step_cell_mult)
        self.phase1_dens_rebuild_interval = max(1, int(phase1_dens_rebuild_interval))
        self.phase1_target_density = phase1_target_density
        self.phase2_iters = int(phase2_iters)
        self.phase2_max_displacement_frac = float(phase2_max_displacement_frac)
        self.phase2_step_frac = float(phase2_step_frac)
        self.phase2_dens_rebuild_interval = max(1, int(phase2_dens_rebuild_interval))
        self.phase2_hotspot_percentile = float(phase2_hotspot_percentile)
        self.phase2_hotspot_abs_margin = float(phase2_hotspot_abs_margin)
        self.phase3_enable = bool(phase3_enable)
        self.phase3_outer_passes = int(phase3_outer_passes)
        self.phase3_pair_budget = int(phase3_pair_budget)
        self.phase3_wl_slack = float(phase3_wl_slack)
        self.top_cong_frac = top_cong_frac
        self.bot_cong_frac = bot_cong_frac

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        nr = max(int(benchmark.grid_rows), 1)
        nc = max(int(benchmark.grid_cols), 1)
        n_hard = int(benchmark.num_hard_macros)
        n_macros = int(benchmark.num_macros)

        pos_np = placement.detach().cpu().numpy()
        fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)
        movable = (~fixed) & benchmark.get_hard_macro_mask()[:n_hard].detach().cpu().numpy().astype(bool)

        input_orig_xy = pos_np[:n_hard, :2].copy()
        phase1_cap = None
        if self.phase1_max_displacement_frac is not None:
            phase1_cap = float(self.phase1_max_displacement_frac) * min(cw, ch)

        ports_np = benchmark.port_positions.detach().cpu().numpy()
        n_ports = int(ports_np.shape[0])
        macro_to_nets = _macro_to_nets(benchmark)
        net_hpwl = _init_net_hpwl(pos_np, benchmark, ports_np, n_macros, n_ports)
        wl_current = _macro_level_hpwl_from_cache(net_hpwl)

        order_small = [i for i in range(n_hard) if movable[i]]
        order_small.sort(
            key=lambda i: float(benchmark.macro_sizes[i, 0] * benchmark.macro_sizes[i, 1])
        )

        # ── Phase 1: aggressive density (local peak down; ignore HPWL) ───
        dens = _build_area_density_map(pos_np, benchmark, include_soft=True)
        sum_dens = float(np.sum(dens))
        n_cells = float(nr * nc)
        cell_w = cw / nc
        cell_h = ch / nr
        base_step_p1 = self.phase1_step_cell_mult * min(cell_w, cell_h)

        for it in range(self.phase1_iters):
            if it > 0 and it % self.phase1_dens_rebuild_interval == 0:
                dens = _build_area_density_map(pos_np, benchmark, include_soft=True)
                sum_dens = float(np.sum(dens))

            mean_iter_start = sum_dens / n_cells
            p75 = float(np.percentile(dens, 75))
            target = (
                self.phase1_target_density
                if self.phase1_target_density is not None
                else max(mean_iter_start, 1e-9)
            )
            hot_cut = max(target, 0.5 * mean_iter_start + 0.5 * p75)

            for i in order_small:
                w = float(benchmark.macro_sizes[i, 0])
                h = float(benchmark.macro_sizes[i, 1])
                cx, cy = float(pos_np[i, 0]), float(pos_np[i, 1])
                loc = _local_max_density_in_footprint(dens, cx, cy, w, h, cw, ch, nr, nc)
                if loc <= hot_cut:
                    continue

                gx, gy = _sample_gradient_direction(dens, cx, cy, cw, ch, nr, nc)
                if gx == 0.0 and gy == 0.0:
                    continue
                ncx = cx - gx * base_step_p1
                ncy = cy - gy * base_step_p1
                ncx, ncy = _clamp_center(ncx, ncy, w, h, cw, ch)
                if phase1_cap is not None:
                    ox, oy = float(input_orig_xy[i, 0]), float(input_orig_xy[i, 1])
                    dx, dy = ncx - ox, ncy - oy
                    dist = math.hypot(dx, dy)
                    if dist > phase1_cap:
                        s = phase1_cap / dist
                        ncx, ncy = ox + dx * s, oy + dy * s
                        ncx, ncy = _clamp_center(ncx, ncy, w, h, cw, ch)

                d_sum = _move_macro_in_density_inplace(dens, cx, cy, ncx, ncy, w, h, cw, ch, nr, nc)
                sum_dens += d_sum
                loc2 = _local_max_density_in_footprint(dens, ncx, ncy, w, h, cw, ch, nr, nc)

                wl_delta, net_updates = _hpwl_delta_move_one_macro(
                    i,
                    ncx,
                    ncy,
                    pos_np,
                    benchmark,
                    ports_np,
                    n_macros,
                    n_ports,
                    net_hpwl,
                    macro_to_nets,
                )

                accept = loc2 < loc - 1e-12

                if accept:
                    pos_np[i, 0], pos_np[i, 1] = ncx, ncy
                    placement[i, 0], placement[i, 1] = float(ncx), float(ncy)
                    for ni, wn in net_updates:
                        net_hpwl[ni] = wn
                    wl_current += wl_delta
                else:
                    rev = _move_macro_in_density_inplace(
                        dens, ncx, ncy, cx, cy, w, h, cw, ch, nr, nc
                    )
                    sum_dens += rev

        # ── Phase 2: gentle HPWL repair (tight cap from Phase 1 result) ───
        repair_xy = pos_np[:n_hard, :2].copy()
        max_disp_p2 = self.phase2_max_displacement_frac * min(cw, ch)
        dens_p2 = _build_area_density_map(pos_np, benchmark, include_soft=True)
        sum_dens = float(np.sum(dens_p2))
        hot_cap = float(np.percentile(dens_p2, self.phase2_hotspot_percentile)) + self.phase2_hotspot_abs_margin
        step_p2 = self.phase2_step_frac * min(cell_w, cell_h)
        dirs8 = _unit_directions_8()

        order_large = list(order_small)
        order_large.reverse()

        for it in range(self.phase2_iters):
            if it > 0 and it % self.phase2_dens_rebuild_interval == 0:
                dens_p2 = _build_area_density_map(pos_np, benchmark, include_soft=True)
                sum_dens = float(np.sum(dens_p2))

            for i in order_large:
                w = float(benchmark.macro_sizes[i, 0])
                h = float(benchmark.macro_sizes[i, 1])
                cx, cy = float(pos_np[i, 0]), float(pos_np[i, 1])

                best_delta = 0.0
                best_pos: Optional[tuple[float, float]] = None
                best_updates: list[tuple[int, float]] = []

                for ux, uy in dirs8:
                    ncx = cx + ux * step_p2
                    ncy = cy + uy * step_p2
                    ncx, ncy = _clamp_center(ncx, ncy, w, h, cw, ch)
                    rx, ry = float(repair_xy[i, 0]), float(repair_xy[i, 1])
                    dx, dy = ncx - rx, ncy - ry
                    if math.hypot(dx, dy) > max_disp_p2:
                        continue

                    d_sum = _move_macro_in_density_inplace(
                        dens_p2, cx, cy, ncx, ncy, w, h, cw, ch, nr, nc
                    )
                    sum_dens += d_sum
                    loc2 = _local_max_density_in_footprint(
                        dens_p2, ncx, ncy, w, h, cw, ch, nr, nc
                    )

                    wl_delta, net_updates = _hpwl_delta_move_one_macro(
                        i,
                        ncx,
                        ncy,
                        pos_np,
                        benchmark,
                        ports_np,
                        n_macros,
                        n_ports,
                        net_hpwl,
                        macro_to_nets,
                    )

                    ok = wl_delta < -1e-12 and loc2 <= hot_cap + 1e-12
                    if ok and wl_delta < best_delta:
                        best_delta = wl_delta
                        best_pos = (ncx, ncy)
                        best_updates = list(net_updates)

                    rev = _move_macro_in_density_inplace(
                        dens_p2, ncx, ncy, cx, cy, w, h, cw, ch, nr, nc
                    )
                    sum_dens += rev

                if best_pos is not None:
                    ncx, ncy = best_pos
                    d_applied = _move_macro_in_density_inplace(
                        dens_p2, cx, cy, ncx, ncy, w, h, cw, ch, nr, nc
                    )
                    sum_dens += d_applied
                    pos_np[i, 0], pos_np[i, 1] = ncx, ncy
                    placement[i, 0], placement[i, 1] = float(ncx), float(ncy)
                    for ni, wn in best_updates:
                        net_hpwl[ni] = wn
                    wl_current += best_delta

        # ── Phase 3: congestion refinement (optional) ─────────────────────
        plc = _try_load_plc(benchmark)
        if plc is not None and self.phase3_enable and self.phase3_outer_passes > 0:
            rudy = _build_area_density_map(pos_np, benchmark, include_soft=True)

            _set_placement(plc, placement, benchmark)
            wl_ref_plc = float(plc.get_cost())
            cong_ref = float(plc.get_congestion_cost())
            wl_ref_fast = float(np.sum(net_hpwl))

            pairs_tried = 0
            stop_phase3 = False

            for _pass in range(self.phase3_outer_passes):
                if stop_phase3:
                    break
                _set_placement(plc, placement, benchmark)
                plc.get_congestion_cost()
                H = np.asarray(plc.H_routing_cong, dtype=np.float64).reshape(nr, nc)
                V = np.asarray(plc.V_routing_cong, dtype=np.float64).reshape(nr, nc)
                cong = np.maximum(H, V)
                flat = cong.ravel()
                n_bins = flat.size
                k_top = max(1, int(math.ceil(self.top_cong_frac * n_bins)))
                k_bot = max(1, int(math.ceil(self.bot_cong_frac * n_bins)))
                top_idx = np.argsort(flat)[::-1][: min(k_top, 32)]
                bot_idx = np.argsort(flat)[: min(k_bot, 32)]
                bot_set = set(int(x) for x in bot_idx.tolist())

                improved = False
                for ti in top_idx:
                    r, c = divmod(int(ti), nc)
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                            continue
                        ni = rr * nc + cc
                        if ni not in bot_set:
                            continue
                        hot_macros = _macros_overlapping_bin(
                            pos_np, benchmark, n_hard, movable, r, c, cw, ch, nr, nc
                        )
                        cool_macros = _macros_overlapping_bin(
                            pos_np, benchmark, n_hard, movable, rr, cc, cw, ch, nr, nc
                        )
                        if not hot_macros or not cool_macros:
                            continue
                        hot_macros.sort(
                            key=lambda i: -float(
                                benchmark.macro_sizes[i, 0] * benchmark.macro_sizes[i, 1]
                            )
                        )
                        cool_macros.sort(
                            key=lambda i: float(
                                benchmark.macro_sizes[i, 0] * benchmark.macro_sizes[i, 1]
                            )
                        )
                        for hi in hot_macros[:4]:
                            for ci in cool_macros[:4]:
                                if pairs_tried >= self.phase3_pair_budget:
                                    stop_phase3 = True
                                    break
                                if hi == ci:
                                    continue
                                pairs_tried += 1
                                px0, py0 = float(pos_np[hi, 0]), float(pos_np[hi, 1])
                                px1, py1 = float(pos_np[ci, 0]), float(pos_np[ci, 1])
                                w0 = float(benchmark.macro_sizes[hi, 0])
                                h0 = float(benchmark.macro_sizes[hi, 1])
                                w1 = float(benchmark.macro_sizes[ci, 0])
                                h1 = float(benchmark.macro_sizes[ci, 1])
                                sx, sy = px1, py1
                                tx, ty = px0, py0
                                sx, sy = _clamp_center(sx, sy, w0, h0, cw, ch)
                                tx, ty = _clamp_center(tx, ty, w1, h1, cw, ch)

                                rh = _local_max_density_in_footprint(
                                    rudy, px0, py0, w0, h0, cw, ch, nr, nc
                                )
                                rc = _local_max_density_in_footprint(
                                    rudy, px1, py1, w1, h1, cw, ch, nr, nc
                                )
                                if rh <= rc + 1e-9:
                                    continue

                                d_wl, swap_updates = _hpwl_delta_swap(
                                    hi,
                                    ci,
                                    sx,
                                    sy,
                                    tx,
                                    ty,
                                    pos_np,
                                    benchmark,
                                    ports_np,
                                    n_macros,
                                    n_ports,
                                    net_hpwl,
                                    macro_to_nets,
                                )
                                if wl_ref_fast + d_wl > wl_ref_fast * (1.0 + self.phase3_wl_slack):
                                    continue

                                pos_np[hi, 0], pos_np[hi, 1] = sx, sy
                                pos_np[ci, 0], pos_np[ci, 1] = tx, ty
                                placement[hi, 0], placement[hi, 1] = float(sx), float(sy)
                                placement[ci, 0], placement[ci, 1] = float(tx), float(ty)

                                _set_placement(plc, placement, benchmark)
                                wl_new = float(plc.get_cost())
                                cong_new = float(plc.get_congestion_cost())

                                ok = cong_new < cong_ref and wl_new <= wl_ref_plc * (
                                    1.0 + self.phase3_wl_slack
                                )
                                if ok:
                                    cong_ref = cong_new
                                    wl_ref_plc = wl_new
                                    for nii, wn in swap_updates:
                                        net_hpwl[nii] = wn
                                    wl_ref_fast = float(np.sum(net_hpwl))
                                    improved = True
                                    rudy = _build_area_density_map(
                                        pos_np, benchmark, include_soft=True
                                    )
                                    break
                                pos_np[hi, 0], pos_np[hi, 1] = px0, py0
                                pos_np[ci, 0], pos_np[ci, 1] = px1, py1
                                placement[hi, 0], placement[hi, 1] = px0, py0
                                placement[ci, 0], placement[ci, 1] = px1, py1
                            if improved or stop_phase3:
                                break
                        if improved or stop_phase3:
                            break
                    if improved or stop_phase3:
                        break
                if not improved:
                    break

        return placement
