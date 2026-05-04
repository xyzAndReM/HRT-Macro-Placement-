"""
Grid snap search with full PlacementCost proxy.

The canvas uses a fixed **64×64** cell grid (not the benchmark routing grid). Every
**movable** macro is visited once (random order). For each macro, its center is tried
at each **nearby** cell center with **Manhattan distance ≤ 1** from the macro’s
current cell (same cell + up to four orthogonal neighbors). For every candidate,
``compute_proxy_cost`` runs the full evaluator. After each macro, if some candidate
beat the design proxy **before** that macro’s trials, the best candidate is kept.

If any macro moves, ``vis/<benchmark>_chosen_pass.png`` shows initial vs final and
arrows for each moved macro.

Usage:
    uv run evaluate submissions/chosen.py -b ibm01
    uv run python submissions/chosen.py
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_plc_for_benchmark(benchmark: Benchmark):
    """Rebuild PlacementCost from the ICCAD04 testcase directory (same as evaluate)."""
    root = _repo_root()
    case_dir = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / benchmark.name
    netlist = case_dir / "netlist.pb.txt"
    if not netlist.is_file():
        raise FileNotFoundError(
            f"Cannot load PlacementCost: missing {netlist}. "
            "Use an ICCAD04 benchmark loaded from that tree."
        )
    _, plc = load_benchmark_from_dir(str(case_dir))
    return plc


def _cell_xy_to_rc(x: float, y: float, cw: float, ch: float, n: int) -> tuple[int, int, float, float]:
    cell_w = cw / float(n)
    cell_h = ch / float(n)
    c = min(n - 1, max(0, int(x / cell_w)))
    r = min(n - 1, max(0, int(y / cell_h)))
    return r, c, cell_w, cell_h


def _cell_center(r: int, c: int, cell_w: float, cell_h: float) -> tuple[float, float]:
    return (c + 0.5) * cell_w, (r + 0.5) * cell_h


def _legal_center(
    cx: float,
    cy: float,
    w: float,
    h: float,
    cw: float,
    ch: float,
) -> bool:
    half_w, half_h = 0.5 * w, 0.5 * h
    return (
        half_w - 1e-9 <= cx <= cw - half_w + 1e-9
        and half_h - 1e-9 <= cy <= ch - half_h + 1e-9
    )


def _neighbor_cells(r0: int, c0: int, n: int, manhattan_max: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in range(-manhattan_max, manhattan_max + 1):
        for dc in range(-manhattan_max, manhattan_max + 1):
            if abs(dr) + abs(dc) > manhattan_max:
                continue
            r, c = r0 + dr, c0 + dc
            if 0 <= r < n and 0 <= c < n:
                out.append((r, c))
    return out


def _moved_macro_indices(pb: torch.Tensor, pa: torch.Tensor, eps: float = 1e-6) -> list[int]:
    d = (pb - pa).abs().max(dim=1).values
    return [i for i in range(pb.shape[0]) if float(d[i].item()) > eps]


def _save_chosen_pass_figure(
    benchmark: Benchmark,
    pos_before: torch.Tensor,
    pos_after: torch.Tensor,
    proxy_before: float,
    proxy_after: float,
    grid_n: int,
    manhattan_max: int,
    n_improved_macros: int,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_macros = int(benchmark.num_macros)
    pb = pos_before.detach().cpu().numpy()
    pa = pos_after.detach().cpu().numpy()
    sizes = benchmark.macro_sizes.detach().cpu().numpy()
    moved = _moved_macro_indices(pos_before, pos_after)

    cmap = plt.cm.tab20
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    ax.set_xlim(0, cw)
    ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.add_patch(Rectangle((0, 0), cw, ch, fill=False, edgecolor="black", linewidth=1.2))

    for g in range(grid_n + 1):
        ax.axvline(g * (cw / grid_n), color="0.88", linewidth=0.22, zorder=0)
        ax.axhline(g * (ch / grid_n), color="0.88", linewidth=0.22, zorder=0)

    for i in range(n_macros):
        x, y = float(pb[i, 0]), float(pb[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        ax.add_patch(
            Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                facecolor="0.90",
                edgecolor="0.60",
                linewidth=0.28,
                alpha=0.45,
                zorder=1,
            )
        )

    for k, i in enumerate(moved):
        col = cmap((k % 20) / 19.0)
        xo, yo = float(pb[i, 0]), float(pb[i, 1])
        xn, yn = float(pa[i, 0]), float(pa[i, 1])
        wm, hm = float(sizes[i, 0]), float(sizes[i, 1])
        ax.add_patch(
            Rectangle(
                (xo - wm / 2, yo - hm / 2),
                wm,
                hm,
                facecolor="none",
                edgecolor=col,
                linewidth=2.0,
                linestyle="--",
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (xn - wm / 2, yn - hm / 2),
                wm,
                hm,
                facecolor=col,
                edgecolor=col,
                linewidth=1.5,
                alpha=0.35,
                zorder=4,
            )
        )
        ax.annotate(
            "",
            xy=(xn, yn),
            xytext=(xo, yo),
            arrowprops=dict(
                arrowstyle="->",
                color=col,
                lw=1.4,
                shrinkA=4,
                shrinkB=4,
            ),
            zorder=5,
        )

    ax.set_title(
        f"{benchmark.name} — ChosenPlacer full pass ({len(moved)} macros moved)\n"
        f"Proxy {proxy_before:.6f} → {proxy_after:.6f}  |  "
        f"{grid_n}×{grid_n} grid, |Δrow|+|Δcol| ≤ {manhattan_max}  |  "
        f"{n_improved_macros} macro positions improved vs local baseline"
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class ChosenPlacer:
    """
    Snap **every** movable macro to the best nearby grid cell center (full proxy).

    Args:
        grid_n: Cells per side (default 64).
        manhattan_max: Max |Δrow|+|Δcol| from current cell (default 1).
        seed: Shuffles order of movable macros (deterministic).
    """

    def __init__(
        self,
        grid_n: int = 64,
        manhattan_max: int = 1,
        seed: int = 0,
    ):
        self.grid_n = int(grid_n)
        self.manhattan_max = int(manhattan_max)
        self.seed = int(seed)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        plc = _load_plc_for_benchmark(benchmark)

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        n = self.grid_n
        rng = random.Random(self.seed)

        movable = (~benchmark.macro_fixed).nonzero(as_tuple=False).flatten().tolist()
        if not movable:
            print("ChosenPlacer: no movable macros; placement unchanged.")
            return placement

        placement_start = placement.clone()
        costs_start = compute_proxy_cost(placement.clone(), benchmark, plc)
        proxy_start = float(costs_start["proxy_cost"])

        order = list(movable)
        rng.shuffle(order)

        n_improved_rounds = 0

        for i_macro in order:
            w = float(benchmark.macro_sizes[i_macro, 0])
            h = float(benchmark.macro_sizes[i_macro, 1])
            x0 = float(placement[i_macro, 0])
            y0 = float(placement[i_macro, 1])

            r0, c0, cell_w, cell_h = _cell_xy_to_rc(x0, y0, cw, ch, n)
            candidates = _neighbor_cells(r0, c0, n, self.manhattan_max)

            base_cost = float(
                compute_proxy_cost(placement.clone(), benchmark, plc)["proxy_cost"]
            )
            best_p = base_cost
            best_trial = placement.clone()

            for r, c in candidates:
                cx, cy = _cell_center(r, c, cell_w, cell_h)
                if not _legal_center(cx, cy, w, h, cw, ch):
                    continue
                trial = placement.clone()
                trial[i_macro, 0] = cx
                trial[i_macro, 1] = cy
                p = float(compute_proxy_cost(trial, benchmark, plc)["proxy_cost"])
                if p < best_p:
                    best_p = p
                    best_trial = trial.clone()

            if best_p < base_cost - 1e-12:
                placement = best_trial
                n_improved_rounds += 1

        costs_end = compute_proxy_cost(placement.clone(), benchmark, plc)
        proxy_end = float(costs_end["proxy_cost"])
        delta_pct = (proxy_end - proxy_start) / max(abs(proxy_start), 1e-30) * 100.0

        moved_idx = _moved_macro_indices(placement_start, placement)
        print(
            f"ChosenPlacer: pass over {len(movable)} movable macros "
            f"({n_improved_rounds} rounds improved proxy vs local trial baseline); "
            f"global proxy {proxy_start:.6f} → {proxy_end:.6f} ({delta_pct:+.3f}%)."
        )

        if moved_idx:
            out_vis = _repo_root() / "vis" / f"{benchmark.name}_chosen_pass.png"
            _save_chosen_pass_figure(
                benchmark,
                placement_start,
                placement,
                proxy_start,
                proxy_end,
                n,
                self.manhattan_max,
                n_improved_rounds,
                out_vis,
            )
            print(f"ChosenPlacer: pass figure saved to {out_vis.resolve()}")

        return placement


def _cli_main() -> None:
    root = _repo_root()
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    ChosenPlacer(seed=1).place(b)


if __name__ == "__main__":
    _cli_main()
