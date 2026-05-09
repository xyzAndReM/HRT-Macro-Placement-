"""
Genetic placement with **guided mutation**: local greedy search on one random macro per
mutation event, scoring candidates with the same **fast surrogate** as
`submissions/sa.py`:

    wl_cost + 0.5 * density_cost + 0.5 * rudy_abu

where **congestion** is **RUDY demand** on the placement grid reduced like top-5% ABU —
not PLC routing (too expensive for inner candidate loops). Optional
``use_proxy_fitness`` uses ``compute_proxy_cost`` every ``proxy_fitness_interval``
generations for selection pressure closer to the evaluator.

Usage:
    uv run evaluate submissions/genetic.py -b ibm01
    uv run python submissions/genetic.py
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_sa_module():
    """Load ``sa.py`` from this directory (submission evaluate loads files standalone)."""
    path = Path(__file__).resolve().parent / "sa.py"
    spec = importlib.util.spec_from_file_location("genetic_sa_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sa.py from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_sa = _load_sa_module()

_macro_to_nets = _sa._macro_to_nets
_pair_overlaps_current = _sa._pair_overlaps_current
_refresh_overlap_row = _sa._refresh_overlap_row
_hard_move_overlap_ok = _sa._hard_move_overlap_ok
_canvas_ok_single = _sa._canvas_ok_single
_init_net_hpwl = _sa._init_net_hpwl
_hpwl_delta_single_move = _sa._hpwl_delta_single_move
_build_area_density_map = _sa._build_area_density_map
_plc_style_density_cost = _sa._plc_style_density_cost
_build_rudy_map = _sa._build_rudy_map
_abu_like_cost = _sa._abu_like_cost
_add_net_to_rudy = _sa._add_net_to_rudy
_net_bbox = _sa._net_bbox
_wl_cost_term = _sa._wl_cost_term
_move_macro_in_density_inplace = _sa._move_macro_in_density_inplace


def _try_load_plc_iccad04(benchmark: Benchmark):
    root = _repo_root()
    case_dir = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / benchmark.name
    if not (case_dir / "netlist.pb.txt").is_file():
        return None
    _, plc = load_benchmark_from_dir(str(case_dir))
    return plc


def _clamp_macro_to_canvas(
    pos: np.ndarray,
    i: int,
    benchmark: Benchmark,
    cw: float,
    ch: float,
) -> None:
    w = float(benchmark.macro_sizes[i, 0])
    h = float(benchmark.macro_sizes[i, 1])
    lo_x = w * 0.5
    hi_x = cw - w * 0.5
    lo_y = h * 0.5
    hi_y = ch - h * 0.5
    pos[i, 0] = float(min(max(float(pos[i, 0]), lo_x), hi_x))
    pos[i, 1] = float(min(max(float(pos[i, 1]), lo_y), hi_y))


def _surrogate_energy(
    pos_full: np.ndarray,
    benchmark: Benchmark,
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
    weights: np.ndarray,
    cw: float,
    ch: float,
    net_norm: float,
    nr: int,
    nc: int,
) -> float:
    net_hpwl = _init_net_hpwl(pos_full, benchmark, ports_np, n_macros, n_ports)
    wl_sum = float(np.dot(net_hpwl, weights))
    dens = _build_area_density_map(pos_full, benchmark)
    den_cost = _plc_style_density_cost(dens)
    rudy, _, _ = _build_rudy_map(
        pos_full, benchmark, ports_np, n_macros, n_ports, weights, cw, ch, nr, nc
    )
    rudy_cost = _abu_like_cost(rudy, 0.05)
    return float(
        _wl_cost_term(wl_sum, cw, ch, net_norm) + 0.5 * den_cost + 0.5 * rudy_cost
    )


def _move_delta_energy(
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
    cw: float,
    ch: float,
    net_norm: float,
    dens: np.ndarray,
    nr: int,
    nc: int,
    rudy: np.ndarray,
    rudy_bboxes: list,
    rudy_demands: np.ndarray,
    o_mat: np.ndarray,
    n_hard: int,
    sizes_hard: np.ndarray,
) -> float | None:
    """Return Δ surrogate energy for moving macro ``i`` to (ncx,ncy); ``None`` if illegal."""
    dcx = ncx - float(pos_full[i, 0])
    dcy = ncy - float(pos_full[i, 1])
    if not _canvas_ok_single(pos_full, benchmark, i, dcx, dcy, cw, ch):
        return None

    if i < n_hard and not _hard_move_overlap_ok(
        i, dcx, dcy, pos_full, sizes_hard, n_hard, o_mat
    ):
        return None

    d_wl_raw, _ = _hpwl_delta_single_move(
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

    ox, oy = float(pos_full[i, 0]), float(pos_full[i, 1])
    w = float(benchmark.macro_sizes[i, 0])
    h = float(benchmark.macro_sizes[i, 1])
    before_den = _plc_style_density_cost(dens)
    _move_macro_in_density_inplace(dens, ox, oy, ncx, ncy, w, h, cw, ch, nr, nc)
    after_den = _plc_style_density_cost(dens)
    _move_macro_in_density_inplace(dens, ncx, ncy, ox, oy, w, h, cw, ch, nr, nc)
    d_den = after_den - before_den

    before_rudy = _abu_like_cost(rudy, 0.05)
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
    after_rudy = _abu_like_cost(rudy, 0.05)
    d_rudy = after_rudy - before_rudy
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

    return float(d_wl_cost + 0.5 * d_den + 0.5 * d_rudy)


def _commit_move(
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
    cw: float,
    ch: float,
    dens: np.ndarray,
    nr: int,
    nc: int,
    rudy: np.ndarray,
    rudy_bboxes: list,
    rudy_demands: np.ndarray,
    o_mat: np.ndarray | None,
    n_hard: int,
    sizes_hard: np.ndarray,
) -> None:
    ox, oy = float(pos_full[i, 0]), float(pos_full[i, 1])
    w = float(benchmark.macro_sizes[i, 0])
    h = float(benchmark.macro_sizes[i, 1])
    _move_macro_in_density_inplace(dens, ox, oy, ncx, ncy, w, h, cw, ch, nr, nc)
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
    for ni in macro_to_nets[i]:
        net_hpwl[ni] = _sa._single_net_hpwl(
            ni, pos_full, benchmark, ports_np, n_macros, n_ports
        )
    if o_mat is not None and i < n_hard:
        _refresh_overlap_row(i, pos_full, sizes_hard, n_hard, o_mat)


def _guided_mutation(
    pos_full: np.ndarray,
    benchmark: Benchmark,
    rng: random.Random,
    movable: list[int],
    macro_to_nets: list[list[int]],
    ports_np: np.ndarray,
    n_macros: int,
    n_ports: int,
    weights: np.ndarray,
    cw: float,
    ch: float,
    net_norm: float,
    nr: int,
    nc: int,
    n_hard: int,
    sizes_hard: np.ndarray,
    guided_candidates: int,
    candidate_radius_um: float,
) -> None:
    net_hpwl = _init_net_hpwl(pos_full, benchmark, ports_np, n_macros, n_ports)
    dens = _build_area_density_map(pos_full, benchmark)
    rudy, rudy_bboxes, rudy_demands = _build_rudy_map(
        pos_full, benchmark, ports_np, n_macros, n_ports, weights, cw, ch, nr, nc
    )
    o_mat = _pair_overlaps_current(pos_full, sizes_hard, n_hard)

    i = int(rng.choice(movable))
    ox = float(pos_full[i, 0])
    oy = float(pos_full[i, 1])
    w_m = float(benchmark.macro_sizes[i, 0])
    h_m = float(benchmark.macro_sizes[i, 1])
    lo_x = w_m * 0.5
    hi_x = cw - w_m * 0.5
    lo_y = h_m * 0.5
    hi_y = ch - h_m * 0.5

    best_delta = 0.0
    best_ncx, best_ncy = ox, oy
    rad = max(candidate_radius_um, 1e-12)

    for _ in range(max(1, guided_candidates)):
        ncx = ox + rng.uniform(-rad, rad)
        ncy = oy + rng.uniform(-rad, rad)
        ncx = float(min(max(ncx, lo_x), hi_x))
        ncy = float(min(max(ncy, lo_y), hi_y))
        de = _move_delta_energy(
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
            cw,
            ch,
            net_norm,
            dens,
            nr,
            nc,
            rudy,
            rudy_bboxes,
            rudy_demands,
            o_mat,
            n_hard,
            sizes_hard,
        )
        if de is None:
            continue
        if de < best_delta:
            best_delta = de
            best_ncx, best_ncy = ncx, ncy

    if best_delta < -1e-15:
        _commit_move(
            i,
            best_ncx,
            best_ncy,
            pos_full,
            benchmark,
            ports_np,
            n_macros,
            n_ports,
            net_hpwl,
            macro_to_nets,
            weights,
            cw,
            ch,
            dens,
            nr,
            nc,
            rudy,
            rudy_bboxes,
            rudy_demands,
            o_mat,
            n_hard,
            sizes_hard,
        )


def _crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: random.Random,
    benchmark: Benchmark,
    fixed: np.ndarray,
    cw: float,
    ch: float,
) -> np.ndarray:
    n = p1.shape[0]
    child = p1.copy()
    for i in range(n):
        if bool(fixed[i]):
            continue
        if rng.random() < 0.5:
            child[i, 0] = float(p2[i, 0])
            child[i, 1] = float(p2[i, 1])
        _clamp_macro_to_canvas(child, i, benchmark, cw, ch)
    return child


def _tournament(
    fitness: list[float],
    rng: random.Random,
    k: int = 3,
) -> int:
    idxs = [rng.randrange(len(fitness)) for _ in range(max(2, k))]
    return min(idxs, key=lambda j: fitness[j])


class GeneticPlacer:
    """
    Genetic algorithm on macro centers with SA-aligned surrogate fitness and guided
    mutation (greedy local search using incremental WL / density / RUDY).
    """

    def __init__(
        self,
        population_size: int = 24,
        generations: int = 40,
        mutation_rate: float = 0.75,
        crossover_rate: float = 0.85,
        tournament_k: int = 3,
        guided_candidates: int = 32,
        candidate_radius_um: float = 0.002,
        init_noise_um: float = 0.001,
        seed: int = 0,
        use_proxy_fitness: bool = False,
        proxy_fitness_interval: int = 10,
    ):
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)
        self.tournament_k = int(tournament_k)
        self.guided_candidates = int(guided_candidates)
        self.candidate_radius_um = float(candidate_radius_um)
        self.init_noise_um = float(init_noise_um)
        self.seed = int(seed)
        self.use_proxy_fitness = bool(use_proxy_fitness)
        self.proxy_fitness_interval = max(1, int(proxy_fitness_interval))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        rng = random.Random(self.seed)
        n_macros = int(benchmark.num_macros)
        n_hard = int(benchmark.num_hard_macros)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        fixed = benchmark.macro_fixed.detach().cpu().numpy().astype(bool)
        movable = [i for i in range(n_macros) if not fixed[i]]
        if not movable:
            return benchmark.macro_positions.clone()

        ports_np = benchmark.port_positions.detach().cpu().numpy()
        n_ports = int(ports_np.shape[0])
        weights = benchmark.net_weights.detach().cpu().numpy().astype(np.float64)
        net_norm = max(1.0, float(np.sum(weights)))
        macro_to_nets = _macro_to_nets(benchmark)
        nr = max(int(benchmark.grid_rows), 1)
        nc = max(int(benchmark.grid_cols), 1)
        sizes_hard = benchmark.macro_sizes[:n_hard].detach().cpu().numpy()

        base = benchmark.macro_positions.detach().cpu().numpy().copy()
        plc = _try_load_plc_iccad04(benchmark) if self.use_proxy_fitness else None
        use_proxy = bool(self.use_proxy_fitness and plc is not None)

        pop: list[np.ndarray] = []
        noise = max(0.0, self.init_noise_um)
        for _ in range(self.population_size):
            ind = base.copy()
            for i in movable:
                if noise > 0.0:
                    ind[i, 0] += rng.gauss(0.0, noise)
                    ind[i, 1] += rng.gauss(0.0, noise)
                _clamp_macro_to_canvas(ind, i, benchmark, cw, ch)
            pop.append(ind)

        def fitness_surrogate(pos: np.ndarray) -> float:
            return _surrogate_energy(
                pos, benchmark, ports_np, n_macros, n_ports, weights, cw, ch, net_norm, nr, nc
            )

        def fitness_proxy(pos: np.ndarray) -> float:
            assert plc is not None
            t = torch.from_numpy(pos.astype(np.float64)).to(
                dtype=benchmark.macro_positions.dtype,
                device=benchmark.macro_positions.device,
            )
            return float(compute_proxy_cost(t, benchmark, plc)["proxy_cost"])

        for gen in range(self.generations):
            use_px = use_proxy and (gen % self.proxy_fitness_interval == 0)
            fit: list[float] = []
            for ind in pop:
                if use_px:
                    fit.append(fitness_proxy(ind))
                else:
                    fit.append(fitness_surrogate(ind))

            best_i = int(min(range(len(fit)), key=lambda j: fit[j]))
            elite = pop[best_i].copy()

            next_pop: list[np.ndarray] = []
            while len(next_pop) < self.population_size - 1:
                i1 = _tournament(fit, rng, self.tournament_k)
                i2 = _tournament(fit, rng, self.tournament_k)
                p1, p2 = pop[i1], pop[i2]
                if rng.random() < self.crossover_rate:
                    child = _crossover(p1, p2, rng, benchmark, fixed, cw, ch)
                else:
                    child = p1.copy()
                if rng.random() < self.mutation_rate:
                    _guided_mutation(
                        child,
                        benchmark,
                        rng,
                        movable,
                        macro_to_nets,
                        ports_np,
                        n_macros,
                        n_ports,
                        weights,
                        cw,
                        ch,
                        net_norm,
                        nr,
                        nc,
                        n_hard,
                        sizes_hard,
                        self.guided_candidates,
                        self.candidate_radius_um,
                    )
                next_pop.append(child)
            next_pop.append(elite)
            pop = next_pop

        # Final best by surrogate (cheap); optional proxy tie-break not required
        final_fit = [fitness_surrogate(p) for p in pop]
        best_j = int(min(range(len(final_fit)), key=lambda j: final_fit[j]))
        best_pos = pop[best_j]

        out = benchmark.macro_positions.clone()
        out.copy_(
            torch.from_numpy(best_pos.astype(np.float64)).to(
                device=out.device, dtype=out.dtype
            )
        )
        return out


def _cli_main() -> None:
    case = _repo_root() / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    GeneticPlacer().place(b)
    print("GeneticPlacer finished on ibm01.")


if __name__ == "__main__":
    _cli_main()
