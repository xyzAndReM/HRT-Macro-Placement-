"""
Simulated annealing with **rigid cluster moves** on movable hard macros.

Clusters come from ``cluster_movable_hard_macros`` (net-weight graph + greedy merge).
Each Metropolis proposal translates **every macro in one cluster** by the same
``(dx, dy)``. Movable **soft** macros are still proposed **one at a time** (same as
``sa.py``), so the surrogate can optimize full-chip WL/density/RUDY.

Surrogate (matches ``sa.py``):

    wl_cost + 0.5 × density_cost + 0.5 × rudy_cost

Hard–hard overlaps are **allowed** to increase (exploration); only **canvas** bounds apply
to proposed moves.

Usage:
    uv run evaluate submissions/cluster_sa.py -b ibm01
    uv run python submissions/cluster_sa.py
"""

from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_sa_helpers():
    """Load ``sa.py`` in this package for shared surrogate / overlap / grid code."""
    sa_path = Path(__file__).resolve().parent / "sa.py"
    spec = importlib.util.spec_from_file_location("_macro_sa_helpers", sa_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_cluster_helpers():
    cluster_path = Path(__file__).resolve().parent / "cluster.py"
    spec = importlib.util.spec_from_file_location("_macro_cluster_helpers", cluster_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_SA = _load_sa_helpers()
_CLUSTER = _load_cluster_helpers()
cluster_movable_hard_macros = _CLUSTER.cluster_movable_hard_macros


def _canvas_ok_cluster(
    pos_full: np.ndarray,
    benchmark: Benchmark,
    ids: list[int],
    dcx: float,
    dcy: float,
    cw: float,
    ch: float,
) -> bool:
    for i in ids:
        if not _SA._canvas_ok_single(pos_full, benchmark, i, dcx, dcy, cw, ch):
            return False
    return True


def _nets_touched_by_macros(
    macro_to_nets: list[list[int]], ids: list[int]
) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for i in ids:
        for ni in macro_to_nets[i]:
            if ni not in seen:
                seen.add(ni)
                out.append(ni)
    return out


def _hpwl_delta_cluster_move(
    ids: list[int],
    dcx: float,
    dcy: float,
    pos_full: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
    net_hpwl: np.ndarray,
    macro_to_nets: list[list[int]],
    weights: np.ndarray,
) -> tuple[float, list[tuple[int, float]]]:
    """Apply rigid shift to all ``ids``, compute weighted HPWL delta, rollback."""
    old_xy = [(float(pos_full[i, 0]), float(pos_full[i, 1])) for i in ids]
    for i in ids:
        pos_full[i, 0] += dcx
        pos_full[i, 1] += dcy

    touched = _nets_touched_by_macros(macro_to_nets, ids)
    delta = 0.0
    updates: list[tuple[int, float]] = []
    for ni in touched:
        old = float(net_hpwl[ni])
        new = _SA._single_net_hpwl(ni, pos_full, benchmark, ports_np, n_macros, n_ports)
        delta += float(weights[ni]) * (new - old)
        updates.append((ni, new))

    for i, (ox, oy) in zip(ids, old_xy):
        pos_full[i, 0] = ox
        pos_full[i, 1] = oy
    return delta, updates


def _density_delta_cluster(
    ids: list[int],
    dcx: float,
    dcy: float,
    pos_full: np.ndarray,
    benchmark: Benchmark,
    dens: np.ndarray,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
) -> float:
    before = _SA._plc_style_density_cost(dens)
    for i in ids:
        ox, oy = float(pos_full[i, 0]), float(pos_full[i, 1])
        ncx, ncy = ox + dcx, oy + dcy
        w = float(benchmark.macro_sizes[i, 0])
        h = float(benchmark.macro_sizes[i, 1])
        _SA._move_macro_in_density_inplace(dens, ox, oy, ncx, ncy, w, h, cw, ch, nr, nc)
    after = _SA._plc_style_density_cost(dens)
    for i in reversed(ids):
        ox, oy = float(pos_full[i, 0]), float(pos_full[i, 1])
        ncx, ncy = ox + dcx, oy + dcy
        w = float(benchmark.macro_sizes[i, 0])
        h = float(benchmark.macro_sizes[i, 1])
        _SA._move_macro_in_density_inplace(dens, ncx, ncy, ox, oy, w, h, cw, ch, nr, nc)
    return after - before


def _rudy_delta_cluster(
    ids: list[int],
    dcx: float,
    dcy: float,
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
    rudy: np.ndarray,
    rudy_bboxes: list[tuple[float, float, float, float] | None],
    rudy_demands: np.ndarray,
    macro_to_nets: list[list[int]],
) -> float:
    touched = _nets_touched_by_macros(macro_to_nets, ids)
    before_rudy = _SA._abu_like_cost(rudy, 0.05)
    for ni in touched:
        bb0 = rudy_bboxes[ni]
        if bb0 is not None:
            _SA._add_net_to_rudy(
                rudy, bb0, float(rudy_demands[ni]), cw, ch, nr, nc, -1.0
            )
    for i in ids:
        pos_full[i, 0] += dcx
        pos_full[i, 1] += dcy
    for ni in touched:
        bb1 = _SA._net_bbox(ni, pos_full, benchmark, ports_np, n_macros, n_ports)
        rudy_bboxes[ni] = bb1
        if bb1 is None:
            rudy_demands[ni] = 0.0
            continue
        lx, rx, by, ty = bb1
        area = max((rx - lx) * (ty - by), 1e-12)
        dem1 = float(weights[ni]) / area
        rudy_demands[ni] = dem1
        _SA._add_net_to_rudy(rudy, bb1, dem1, cw, ch, nr, nc, +1.0)
    after_rudy = _SA._abu_like_cost(rudy, 0.05)
    d_rudy = after_rudy - before_rudy
    # rollback rudy state
    for ni in touched:
        bb1 = rudy_bboxes[ni]
        if bb1 is not None:
            _SA._add_net_to_rudy(
                rudy, bb1, float(rudy_demands[ni]), cw, ch, nr, nc, -1.0
            )
    for i in ids:
        pos_full[i, 0] -= dcx
        pos_full[i, 1] -= dcy
    for ni in touched:
        bb0 = _SA._net_bbox(ni, pos_full, benchmark, ports_np, n_macros, n_ports)
        rudy_bboxes[ni] = bb0
        if bb0 is None:
            rudy_demands[ni] = 0.0
            continue
        lx, rx, by, ty = bb0
        area0 = max((rx - lx) * (ty - by), 1e-12)
        dem0 = float(weights[ni]) / area0
        rudy_demands[ni] = dem0
        _SA._add_net_to_rudy(rudy, bb0, dem0, cw, ch, nr, nc, +1.0)
    return d_rudy


def _commit_cluster_move_density(
    ids: list[int],
    dcx: float,
    dcy: float,
    pos_full: np.ndarray,
    benchmark: Benchmark,
    dens: np.ndarray,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
) -> None:
    for i in ids:
        ox, oy = float(pos_full[i, 0]), float(pos_full[i, 1])
        ncx, ncy = ox + dcx, oy + dcy
        w = float(benchmark.macro_sizes[i, 0])
        h = float(benchmark.macro_sizes[i, 1])
        _SA._move_macro_in_density_inplace(dens, ox, oy, ncx, ncy, w, h, cw, ch, nr, nc)


def _commit_cluster_move_rudy(
    touched: list[int],
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
    rudy: np.ndarray,
    rudy_bboxes: list[tuple[float, float, float, float] | None],
    rudy_demands: np.ndarray,
) -> None:
    for ni in touched:
        bb0 = rudy_bboxes[ni]
        if bb0 is not None:
            _SA._add_net_to_rudy(
                rudy, bb0, float(rudy_demands[ni]), cw, ch, nr, nc, -1.0
            )
    for ni in touched:
        bb1 = _SA._net_bbox(ni, pos_full, benchmark, ports_np, n_macros, n_ports)
        rudy_bboxes[ni] = bb1
        if bb1 is None:
            rudy_demands[ni] = 0.0
            continue
        lx, rx, by, ty = bb1
        area = max((rx - lx) * (ty - by), 1e-12)
        dem1 = float(weights[ni]) / area
        rudy_demands[ni] = dem1
        _SA._add_net_to_rudy(rudy, bb1, dem1, cw, ch, nr, nc, +1.0)


class ClusterSAPlacer:
    """
    SA with rigid cluster moves (hard) + single-macro moves (soft), same surrogate as SAPlacer.

    Overlap is not constrained (only canvas bounds). Default ``delta_um`` is 0.1 μm.
    Clustering kwargs are passed to ``cluster_movable_hard_macros``.
    """

    def __init__(
        self,
        delta_um: float = 0.1,
        max_iters: int = 80_000,
        t0_factor: float = 0.02,
        t_min: float = 1e-9,
        strike_limit: int = 4,
        seed: int = 0,
        min_edge_weight: float = 1e-4,
        max_macros_per_cluster: int = 8,
        area_balance_rel_tol: float = 0.35,
        cluster_eps_um: float = 0.5,
    ):
        self.delta_um = float(delta_um)
        self.max_iters = int(max_iters)
        self.t0_factor = float(t0_factor)
        self.t_min = float(t_min)
        self.strike_limit = int(strike_limit)
        self.seed = int(seed)
        self.min_edge_weight = float(min_edge_weight)
        self.max_macros_per_cluster = int(max_macros_per_cluster)
        self.area_balance_rel_tol = float(area_balance_rel_tol)
        self.cluster_eps_um = float(cluster_eps_um)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        n_macros = int(benchmark.num_macros)
        n_hard = int(benchmark.num_hard_macros)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        fixed = benchmark.macro_fixed.detach().cpu().numpy().astype(bool)

        clusters_g, _, _ = cluster_movable_hard_macros(
            benchmark,
            min_edge_weight=self.min_edge_weight,
            max_macros_per_cluster=self.max_macros_per_cluster,
            area_balance_rel_tol=self.area_balance_rel_tol,
            eps_um=self.cluster_eps_um,
        )

        # Pivots: each hard cluster (rigid), plus each movable soft macro alone
        pivots: list[list[int]] = [list(c) for c in clusters_g]
        for i in range(n_hard, n_macros):
            if not fixed[i]:
                pivots.append([i])

        pool: set[int] = set(range(len(pivots)))
        strikes = {p: 0 for p in pool}

        macro_to_nets = _SA._macro_to_nets(benchmark)
        pos_full = placement.detach().cpu().numpy().copy()
        ports_np = benchmark.port_positions.detach().cpu().numpy()
        n_ports = int(ports_np.shape[0])
        weights = benchmark.net_weights.detach().cpu().numpy().astype(np.float64)
        net_norm = max(1.0, float(np.sum(weights)))

        net_hpwl = _SA._init_net_hpwl(
            pos_full, benchmark, ports_np, n_macros, n_ports
        )
        wl_sum = float(np.dot(net_hpwl, weights))

        nr = max(int(benchmark.grid_rows), 1)
        nc = max(int(benchmark.grid_cols), 1)
        dens = _SA._build_area_density_map(pos_full, benchmark)
        den_cost = _SA._plc_style_density_cost(dens)

        rudy, rudy_bboxes, rudy_demands = _SA._build_rudy_map(
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
        rudy_cost = _SA._abu_like_cost(rudy, 0.05)

        cur_e = (
            _SA._wl_cost_term(wl_sum, cw, ch, net_norm)
            + 0.5 * den_cost
            + 0.5 * rudy_cost
        )

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

            pid = rng.choice(list(pool))
            ids = pivots[pid]
            dcx, dcy = rng.choice(dirs)

            if not _canvas_ok_cluster(pos_full, benchmark, ids, dcx, dcy, cw, ch):
                strikes[pid] = strikes.get(pid, 0) + 1
                if strikes[pid] >= self.strike_limit:
                    pool.discard(pid)
                continue

            strikes[pid] = 0

            if len(ids) == 1:
                i = ids[0]
                ncx, ncy = float(pos_full[i, 0]) + dcx, float(pos_full[i, 1]) + dcy
                d_wl_raw, hp_updates = _SA._hpwl_delta_single_move(
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
            else:
                d_wl_raw, hp_updates = _hpwl_delta_cluster_move(
                    ids,
                    dcx,
                    dcy,
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

            if len(ids) == 1:
                i = ids[0]
                ox, oy = float(pos_full[i, 0]), float(pos_full[i, 1])
                w = float(benchmark.macro_sizes[i, 0])
                h = float(benchmark.macro_sizes[i, 1])
                ncx, ncy = ox + dcx, oy + dcy
                before_den = _SA._plc_style_density_cost(dens)
                _SA._move_macro_in_density_inplace(
                    dens, ox, oy, ncx, ncy, w, h, cw, ch, nr, nc
                )
                after_den = _SA._plc_style_density_cost(dens)
                _SA._move_macro_in_density_inplace(
                    dens, ncx, ncy, ox, oy, w, h, cw, ch, nr, nc
                )
                d_den = after_den - before_den

                before_rudy = _SA._abu_like_cost(rudy, 0.05)
                for ni in macro_to_nets[i]:
                    bb0 = rudy_bboxes[ni]
                    if bb0 is not None:
                        _SA._add_net_to_rudy(
                            rudy,
                            bb0,
                            float(rudy_demands[ni]),
                            cw,
                            ch,
                            nr,
                            nc,
                            -1.0,
                        )
                pos_full[i, 0], pos_full[i, 1] = ncx, ncy
                for ni in macro_to_nets[i]:
                    bb1 = _SA._net_bbox(
                        ni, pos_full, benchmark, ports_np, n_macros, n_ports
                    )
                    rudy_bboxes[ni] = bb1
                    if bb1 is None:
                        rudy_demands[ni] = 0.0
                        continue
                    lx, rx, by, ty = bb1
                    area = max((rx - lx) * (ty - by), 1e-12)
                    dem1 = float(weights[ni]) / area
                    rudy_demands[ni] = dem1
                    _SA._add_net_to_rudy(rudy, bb1, dem1, cw, ch, nr, nc, +1.0)
                after_rudy = _SA._abu_like_cost(rudy, 0.05)
                d_rudy = after_rudy - before_rudy
                for ni in macro_to_nets[i]:
                    bb1 = rudy_bboxes[ni]
                    if bb1 is not None:
                        _SA._add_net_to_rudy(
                            rudy,
                            bb1,
                            float(rudy_demands[ni]),
                            cw,
                            ch,
                            nr,
                            nc,
                            -1.0,
                        )
                pos_full[i, 0], pos_full[i, 1] = ox, oy
                for ni in macro_to_nets[i]:
                    bb0 = _SA._net_bbox(
                        ni, pos_full, benchmark, ports_np, n_macros, n_ports
                    )
                    rudy_bboxes[ni] = bb0
                    if bb0 is None:
                        rudy_demands[ni] = 0.0
                        continue
                    lx, rx, by, ty = bb0
                    area0 = max((rx - lx) * (ty - by), 1e-12)
                    dem0 = float(weights[ni]) / area0
                    rudy_demands[ni] = dem0
                    _SA._add_net_to_rudy(rudy, bb0, dem0, cw, ch, nr, nc, +1.0)
            else:
                d_den = _density_delta_cluster(
                    ids, dcx, dcy, pos_full, benchmark, dens, cw, ch, nr, nc
                )
                d_rudy = _rudy_delta_cluster(
                    ids,
                    dcx,
                    dcy,
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
                    rudy,
                    rudy_bboxes,
                    rudy_demands,
                    macro_to_nets,
                )

            delta_e = d_wl_cost + 0.5 * d_den + 0.5 * d_rudy

            accept = delta_e <= 0.0 or (T > 0 and rng.random() < math.exp(-delta_e / T))

            if accept:
                if len(ids) == 1:
                    i = ids[0]
                    ox, oy = float(pos_full[i, 0]), float(pos_full[i, 1])
                    w = float(benchmark.macro_sizes[i, 0])
                    h = float(benchmark.macro_sizes[i, 1])
                    ncx, ncy = ox + dcx, oy + dcy
                    _SA._move_macro_in_density_inplace(
                        dens, ox, oy, ncx, ncy, w, h, cw, ch, nr, nc
                    )
                    for ni in macro_to_nets[i]:
                        bb0 = rudy_bboxes[ni]
                        if bb0 is not None:
                            _SA._add_net_to_rudy(
                                rudy,
                                bb0,
                                float(rudy_demands[ni]),
                                cw,
                                ch,
                                nr,
                                nc,
                                -1.0,
                            )
                    pos_full[i, 0], pos_full[i, 1] = ncx, ncy
                    for ni in macro_to_nets[i]:
                        bb1 = _SA._net_bbox(
                            ni, pos_full, benchmark, ports_np, n_macros, n_ports
                        )
                        rudy_bboxes[ni] = bb1
                        if bb1 is None:
                            rudy_demands[ni] = 0.0
                            continue
                        lx, rx, by, ty = bb1
                        area = max((rx - lx) * (ty - by), 1e-12)
                        dem1 = float(weights[ni]) / area
                        rudy_demands[ni] = dem1
                        _SA._add_net_to_rudy(rudy, bb1, dem1, cw, ch, nr, nc, +1.0)

                    placement[i, 0] = float(ncx)
                    placement[i, 1] = float(ncy)
                    for ni, wn in hp_updates:
                        net_hpwl[ni] = wn
                    wl_sum += d_wl_raw
                    cur_e += delta_e
                else:
                    _commit_cluster_move_density(
                        ids, dcx, dcy, pos_full, benchmark, dens, cw, ch, nr, nc
                    )
                    touched = _nets_touched_by_macros(macro_to_nets, ids)
                    for i in ids:
                        pos_full[i, 0] += dcx
                        pos_full[i, 1] += dcy
                        placement[i, 0] = float(pos_full[i, 0])
                        placement[i, 1] = float(pos_full[i, 1])
                    _commit_cluster_move_rudy(
                        touched,
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
                        rudy,
                        rudy_bboxes,
                        rudy_demands,
                    )
                    for ni, wn in hp_updates:
                        net_hpwl[ni] = wn
                    wl_sum += d_wl_raw
                    cur_e += delta_e

        return placement


def _cli_main() -> None:
    from macro_place.loader import load_benchmark_from_dir

    case = (
        _repo_root()
        / "external"
        / "MacroPlacement"
        / "Testcases"
        / "ICCAD04"
        / "ibm01"
    )
    b, _ = load_benchmark_from_dir(str(case))
    ClusterSAPlacer().place(b)
    print("ClusterSAPlacer finished on ibm01.")


if __name__ == "__main__":
    _cli_main()
