"""
Coarse grid resolution sweep for ExplorePlacer-style local search with uniform-random
macro selection only.

Compares uniform_random across grid_side × grid_side coarse partitions (default 8, 12, 16).
Everything else matches macro_pick_experiment / ExplorePlacer:
  - candidate set = 3×3 neighbors of current coarse cell
  - score = FastProxyEvaluator.total(...)
  - accept only strictly improving moves

Winner metric (printed summary): mean final real proxy_cost from compute_proxy_cost.

Usage:
    uv run python submissions/grid_side_uniform_experiment.py
    uv run python submissions/grid_side_uniform_experiment.py --epochs 200 --grid_sides 8,12,16
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, pstdev

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.loader import load_benchmark_from_dir

from submissions.explore import _repo_root
from submissions.macro_pick_experiment import RunResult, UniformRandomPicker, run_explore_like


def _parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("expected at least one integer")
    return out


def _summarize_grid(results: list[tuple[int, RunResult]]) -> str:
    by_gs: dict[int, list[RunResult]] = {}
    for gs, r in results:
        by_gs.setdefault(gs, []).append(r)

    lines: list[str] = []
    lines.append("Ranking by mean final real_proxy (lower is better), uniform_random only:")
    rows: list[tuple[float, int, int, float, float, float]] = []
    for gs, rs in by_gs.items():
        real_vals = [x.real_end for x in rs if x.real_end is not None]
        if not real_vals:
            continue
        m = mean(real_vals)
        s = pstdev(real_vals) if len(real_vals) > 1 else 0.0
        t = mean([x.seconds for x in rs])
        acc = mean([x.accepted for x in rs])
        rows.append((m, gs, len(real_vals), s, t, acc))

    rows.sort(key=lambda x: x[0])
    for rank, (m, gs, n, s, t, acc) in enumerate(rows, start=1):
        lines.append(
            f"{rank:2d}. grid_side={gs:2d}  real_end mean={m:.6f}  std={s:.6f}  "
            f"time mean={t:.3f}s  accepted mean={acc:.1f}  (n={n})"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--grid_sides", type=str, default="8,12,16")
    ap.add_argument("--out", type=str, default="results/grid_side_uniform_ibm01.csv")
    args = ap.parse_args()

    root = _repo_root()
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))

    grid_sides = _parse_int_list(args.grid_sides)
    seeds = _parse_seeds(args.seeds)
    initial = b.macro_positions.clone()

    results: list[tuple[int, RunResult]] = []
    for gs in grid_sides:
        for sd in seeds:
            r = run_explore_like(
                benchmark=b,
                initial_pos=initial,
                picker=UniformRandomPicker(),
                epochs=int(args.epochs),
                seed=int(sd),
                grid_side=int(gs),
                fast_proxy_device="cpu",
            )
            results.append((gs, r))
            print(
                f"[grid_side_uniform] grid_side={gs} seed={r.seed} accepted={r.accepted} "
                f"fast {r.fast_start:.6f}->{r.fast_end:.6f} "
                f"real {None if r.real_start is None else f'{r.real_start:.6f}'}"
                f"->{None if r.real_end is None else f'{r.real_end:.6f}'} "
                f"time={r.seconds:.3f}s",
                flush=True,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "grid_side",
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
        for gs, r in results:
            w.writerow(
                [
                    gs,
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
    print(_summarize_grid(results))
    print(f"\nWrote results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
