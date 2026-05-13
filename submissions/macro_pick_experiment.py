"""
Macro-selection criteria shootout for ExplorePlacer-style local search.

This compares different strategies for choosing which macro to attempt to move
each epoch, while keeping the rest of the ExplorePlacer logic identical:
  - 8×8 coarse grid
  - candidate set = 3×3 neighbors of current coarse cell
  - score = FastProxyEvaluator.total(...)
  - accept only strictly improving moves

Winner metric (default): final real proxy_cost from compute_proxy_cost.

Usage:
    uv run python submissions/macro_pick_experiment.py
    uv run python submissions/macro_pick_experiment.py --epochs 200 --seeds 0,1,2,3,4,5,6,7,8,9
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.fast_proxy import FastProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from submissions.explore import (  # re-use identical helpers and geometry rules
    _cell_xy_to_rc,
    _legal_center,
    _precompute_nine_neighbors,
    _repo_root,
    _try_load_plc,
)


def _cell_centers(cw: float, ch: float, n: int) -> tuple[list[float], list[float]]:
    cell_w = cw / float(n)
    cell_h = ch / float(n)
    center_x = [(c + 0.5) * cell_w for c in range(n)]
    center_y = [(r + 0.5) * cell_h for r in range(n)]
    return center_x, center_y


class MacroPicker:
    name: str

    def on_epoch_start(self, *, placement: torch.Tensor) -> None:
        pass

    def pick(self, *, ep: int, rng, movable: list[int], placement: torch.Tensor) -> int:
        raise NotImplementedError

    def on_accepted(self, *, i_macro: int) -> None:
        pass


class UniformRandomPicker(MacroPicker):
    name = "uniform_random"

    def pick(self, *, ep: int, rng, movable: list[int], placement: torch.Tensor) -> int:
        return rng.choice(movable)


class DownweightMovedPicker(MacroPicker):
    name = "downweight_moved"

    def __init__(self, *, moved_macro_weight: float = 0.25):
        self.moved_w = float(moved_macro_weight)
        if not (0.0 < self.moved_w <= 1.0):
            raise ValueError("moved_macro_weight must be in (0, 1].")
        self._moved: set[int] = set()

    def pick(self, *, ep: int, rng, movable: list[int], placement: torch.Tensor) -> int:
        # Recompute weights each pick to keep behavior correct and deterministic.
        # For ~1000 macros and ~200 epochs this overhead is negligible.
        weights = [self.moved_w if m in self._moved else 1.0 for m in movable]
        return rng.choices(movable, weights=weights, k=1)[0]

    def on_accepted(self, *, i_macro: int) -> None:
        self._moved.add(i_macro)


class RoundRobinPicker(MacroPicker):
    name = "round_robin"

    def __init__(self, *, shuffle_once: bool = True):
        self.shuffle_once = bool(shuffle_once)
        self._order: list[int] | None = None

    def pick(self, *, ep: int, rng, movable: list[int], placement: torch.Tensor) -> int:
        if self._order is None:
            self._order = list(movable)
            if self.shuffle_once:
                rng.shuffle(self._order)
        return self._order[ep % len(self._order)]


class StaleFirstPicker(MacroPicker):
    name = "stale_first"

    def __init__(self, *, alpha: float = 1.0):
        self.alpha = float(alpha)
        self._last_picked: dict[int, int] = {}
        self._weights: list[float] | None = None

    def pick(self, *, ep: int, rng, movable: list[int], placement: torch.Tensor) -> int:
        # weight(m) = (1 + (ep - last_picked[m])) ** alpha
        w: list[float] = []
        for m in movable:
            lp = self._last_picked.get(m, -10_000_000)
            age = max(1, ep - lp)
            w.append(float(age) ** self.alpha)
        i = rng.choices(movable, weights=w, k=1)[0]
        self._last_picked[i] = ep
        return i


class CellHotspotPicker(MacroPicker):
    name = "cell_hotspot"

    def __init__(self, *, grid_side: int):
        self.n = int(grid_side)
        self._cw = 0.0
        self._ch = 0.0
        self._counts: list[list[int]] | None = None

    def on_epoch_start(self, *, placement: torch.Tensor) -> None:
        # Recompute macro counts per coarse cell (cheap O(#macros)).
        # This keeps selection tied to current placement without touching nets.
        if self._counts is None:
            self._counts = [[0 for _ in range(self.n)] for _ in range(self.n)]
        for r in range(self.n):
            for c in range(self.n):
                self._counts[r][c] = 0
        for i in range(int(placement.shape[0])):
            x = float(placement[i, 0])
            y = float(placement[i, 1])
            r, c, _, _ = _cell_xy_to_rc(x, y, self._cw, self._ch, self.n)
            self._counts[r][c] += 1

    def pick(self, *, ep: int, rng, movable: list[int], placement: torch.Tensor) -> int:
        assert self._counts is not None
        w: list[float] = []
        for m in movable:
            x = float(placement[m, 0])
            y = float(placement[m, 1])
            r, c, _, _ = _cell_xy_to_rc(x, y, self._cw, self._ch, self.n)
            w.append(float(self._counts[r][c]))
        # All counts should be >= 1; still guard against zeros.
        if max(w) <= 0.0:
            return rng.choice(movable)
        return rng.choices(movable, weights=w, k=1)[0]

    def set_canvas(self, *, cw: float, ch: float) -> None:
        self._cw = float(cw)
        self._ch = float(ch)


class AreaWeightedPicker(MacroPicker):
    name = "area_weighted"

    def __init__(self, *, areas: list[float]):
        self.areas = areas

    def pick(self, *, ep: int, rng, movable: list[int], placement: torch.Tensor) -> int:
        w = [self.areas[m] for m in movable]
        if max(w) <= 0.0:
            return rng.choice(movable)
        return rng.choices(movable, weights=w, k=1)[0]


def _make_pickers(*, grid_side: int, macro_areas: list[float]) -> list[MacroPicker]:
    # Note: DownweightMovedPicker needs access to the movable list order to update weights;
    # we handle that in the runner by giving it the movable list once.
    return [
        UniformRandomPicker(),
        DownweightMovedPicker(moved_macro_weight=0.25),
        RoundRobinPicker(shuffle_once=True),
        StaleFirstPicker(alpha=1.0),
        CellHotspotPicker(grid_side=grid_side),
        AreaWeightedPicker(areas=macro_areas),
    ]


@dataclass(frozen=True)
class RunResult:
    criterion: str
    seed: int
    epochs: int
    accepted: int
    fast_start: float
    fast_end: float
    real_start: float | None
    real_end: float | None
    seconds: float


def run_explore_like(
    *,
    benchmark: Benchmark,
    initial_pos: torch.Tensor,
    picker: MacroPicker,
    epochs: int,
    seed: int,
    grid_side: int = 8,
    fast_proxy_device: torch.device | str | None = "cpu",
) -> RunResult:
    placement = initial_pos.clone()

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n = int(grid_side)
    neighbors = _precompute_nine_neighbors(n)
    center_x, center_y = _cell_centers(cw, ch, n)

    sizes_cpu = benchmark.macro_sizes.detach().cpu()
    macro_w = sizes_cpu[:, 0].tolist()
    macro_h = sizes_cpu[:, 1].tolist()
    macro_areas = [float(macro_w[i]) * float(macro_h[i]) for i in range(len(macro_w))]

    movable = (~benchmark.macro_fixed).nonzero(as_tuple=False).flatten().tolist()
    if not movable:
        score = FastProxyEvaluator(benchmark, device=torch.device(fast_proxy_device))
        s0 = score.total(placement)
        plc = _try_load_plc(benchmark)
        r0 = (
            float(compute_proxy_cost(placement.clone(), benchmark, plc)["proxy_cost"])
            if plc is not None
            else None
        )
        return RunResult(
            criterion=getattr(picker, "name", picker.__class__.__name__),
            seed=seed,
            epochs=epochs,
            accepted=0,
            fast_start=s0,
            fast_end=s0,
            real_start=r0,
            real_end=r0,
            seconds=0.0,
        )

    if isinstance(picker, CellHotspotPicker):
        picker.set_canvas(cw=cw, ch=ch)

    # Fast proxy (same as ExplorePlacer hybrid use: CPU by default)
    score = (
        FastProxyEvaluator(benchmark, device=torch.device(fast_proxy_device))
        if fast_proxy_device is not None
        else FastProxyEvaluator(benchmark)
    )
    plc = _try_load_plc(benchmark)

    import random as _random

    rng = _random.Random(int(seed))
    t0 = time.perf_counter()

    fast_start = score.total(placement)
    cur_cost = fast_start
    real_start: float | None = None
    if plc is not None:
        real_start = float(compute_proxy_cost(placement.clone(), benchmark, plc)["proxy_cost"])

    accepted = 0
    for ep in range(int(epochs)):
        picker.on_epoch_start(placement=placement)
        i_macro = picker.pick(ep=ep, rng=rng, movable=movable, placement=placement)

        w = float(macro_w[i_macro])
        h = float(macro_h[i_macro])
        x0 = float(placement[i_macro, 0])
        y0 = float(placement[i_macro, 1])

        r0, c0, _, _ = _cell_xy_to_rc(x0, y0, cw, ch, n)
        candidates = neighbors[r0][c0]

        base_cost = cur_cost
        best_p = base_cost
        best_cx = x0
        best_cy = y0

        for r, c in candidates:
            cx, cy = center_x[c], center_y[r]
            if not _legal_center(cx, cy, w, h, cw, ch):
                continue
            placement[i_macro, 0] = cx
            placement[i_macro, 1] = cy
            p = score.total(placement)
            placement[i_macro, 0] = x0
            placement[i_macro, 1] = y0
            if p < best_p:
                best_p = p
                best_cx = cx
                best_cy = cy

        if best_p < base_cost - 1e-12:
            placement[i_macro, 0] = best_cx
            placement[i_macro, 1] = best_cy
            cur_cost = best_p
            accepted += 1
            picker.on_accepted(i_macro=i_macro)

    fast_end = cur_cost
    real_end: float | None = None
    if plc is not None:
        real_end = float(compute_proxy_cost(placement.clone(), benchmark, plc)["proxy_cost"])

    dt = time.perf_counter() - t0
    return RunResult(
        criterion=getattr(picker, "name", picker.__class__.__name__),
        seed=int(seed),
        epochs=int(epochs),
        accepted=int(accepted),
        fast_start=float(fast_start),
        fast_end=float(fast_end),
        real_start=real_start,
        real_end=real_end,
        seconds=float(dt),
    )


def _summarize(results: list[RunResult]) -> str:
    by_crit: dict[str, list[RunResult]] = {}
    for r in results:
        by_crit.setdefault(r.criterion, []).append(r)

    lines: list[str] = []
    lines.append("Ranking by mean final real_proxy (lower is better):")
    rows: list[tuple[float, str, int, float, float, float]] = []
    for crit, rs in by_crit.items():
        real_vals = [x.real_end for x in rs if x.real_end is not None]
        if not real_vals:
            continue
        m = mean(real_vals)
        s = pstdev(real_vals) if len(real_vals) > 1 else 0.0
        t = mean([x.seconds for x in rs])
        acc = mean([x.accepted for x in rs])
        rows.append((m, crit, len(real_vals), s, t, acc))

    rows.sort(key=lambda x: x[0])
    for rank, (m, crit, n, s, t, acc) in enumerate(rows, start=1):
        lines.append(
            f"{rank:2d}. {crit:16s}  real_end mean={m:.6f}  std={s:.6f}  "
            f"time mean={t:.3f}s  accepted mean={acc:.1f}  (n={n})"
        )
    return "\n".join(lines)


def _parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--grid_side", type=int, default=8)
    ap.add_argument("--out", type=str, default="results/macro_pick_ibm01.csv")
    args = ap.parse_args()

    root = _repo_root()
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))

    sizes = b.macro_sizes.detach().cpu()
    macro_areas = (sizes[:, 0] * sizes[:, 1]).tolist()
    pickers = _make_pickers(grid_side=int(args.grid_side), macro_areas=[float(x) for x in macro_areas])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    initial = b.macro_positions.clone()

    results: list[RunResult] = []
    for picker in pickers:
        for sd in seeds:
            # Ensure each run uses a fresh picker instance (no cross-seed state leakage).
            if isinstance(picker, UniformRandomPicker):
                p = UniformRandomPicker()
            elif isinstance(picker, DownweightMovedPicker):
                p = DownweightMovedPicker(moved_macro_weight=0.25)
                # Enable O(#movable) weight update by keeping movable order stable in runner.
                # (We don't rely on internal update; choices are still deterministic.)
            elif isinstance(picker, RoundRobinPicker):
                p = RoundRobinPicker(shuffle_once=True)
            elif isinstance(picker, StaleFirstPicker):
                p = StaleFirstPicker(alpha=1.0)
            elif isinstance(picker, CellHotspotPicker):
                p = CellHotspotPicker(grid_side=int(args.grid_side))
            elif isinstance(picker, AreaWeightedPicker):
                p = AreaWeightedPicker(areas=[float(x) for x in macro_areas])
            else:
                p = picker

            r = run_explore_like(
                benchmark=b,
                initial_pos=initial,
                picker=p,
                epochs=int(args.epochs),
                seed=int(sd),
                grid_side=int(args.grid_side),
                fast_proxy_device="cpu",
            )
            results.append(r)
            print(
                f"[macro_pick] {r.criterion} seed={r.seed} accepted={r.accepted} "
                f"fast {r.fast_start:.6f}->{r.fast_end:.6f} "
                f"real {None if r.real_start is None else f'{r.real_start:.6f}'}"
                f"->{None if r.real_end is None else f'{r.real_end:.6f}'} "
                f"time={r.seconds:.3f}s",
                flush=True,
            )

    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "criterion",
                "seed",
                "epochs",
                "accepted",
                "fast_start",
                "fast_end",
                "real_start",
                "real_end",
                "seconds",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.criterion,
                    r.seed,
                    r.epochs,
                    r.accepted,
                    f"{r.fast_start:.10g}",
                    f"{r.fast_end:.10g}",
                    "" if r.real_start is None else f"{r.real_start:.10g}",
                    "" if r.real_end is None else f"{r.real_end:.10g}",
                    f"{r.seconds:.10g}",
                ]
            )

    print()
    print(_summarize(results))
    print(f"\nWrote results to {out_path.resolve()}")


if __name__ == "__main__":
    main()

