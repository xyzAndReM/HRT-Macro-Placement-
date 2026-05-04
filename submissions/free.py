"""
Free-liberty analysis placer — highlights macros with the most “room to move”.

For each **hard** macro we estimate how far its center can translate along **+x,
−x, +y, −y** before either leaving the canvas or **increasing** the **overlap area**
with any other hard macro (relative to the current placement). **Degree of
liberty** is the **sum** of those four maximal distances (μm).

The **top ``top_frac``** (default 25%) of **movable** hard macros by this score are
marked in the figure; placement is **unchanged** (``place`` returns the input).

Writes ``vis/<benchmark.name>_free_liberty.png``.

Usage:
    uv run evaluate submissions/free.py -b ibm01
    uv run python submissions/free.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark

_GAP_AREA = 1e-6  # tolerate numeric noise on overlap comparisons


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    """overlap[i,j] = area between hard i and j, symmetric, diagonal 0."""
    o = np.zeros((n_hard, n_hard), dtype=np.float64)
    for i in range(n_hard):
        wi, hi = float(sizes[i, 0]), float(sizes[i, 1])
        li, ri, bi, ti = _bbox(float(pos[i, 0]), float(pos[i, 1]), wi, hi)
        for j in range(i + 1, n_hard):
            wj, hj = float(sizes[j, 0]), float(sizes[j, 1])
            lj, rj, bj, tj = _bbox(float(pos[j, 0]), float(pos[j, 1]), wj, hj)
            a = _overlap_area(li, ri, bi, ti, lj, rj, bj, tj)
            o[i, j] = a
            o[j, i] = a
    return o


def _ok_move(
    i: int,
    dcx: float,
    dcy: float,
    pos: np.ndarray,
    sizes: np.ndarray,
    n_hard: int,
    cw: float,
    ch: float,
    baseline: np.ndarray,
) -> bool:
    cx = float(pos[i, 0]) + dcx
    cy = float(pos[i, 1]) + dcy
    w, h = float(sizes[i, 0]), float(sizes[i, 1])
    if cx < w * 0.5 - 1e-9 or cx > cw - w * 0.5 + 1e-9:
        return False
    if cy < h * 0.5 - 1e-9 or cy > ch - h * 0.5 + 1e-9:
        return False
    li, ri, bi, ti = _bbox(cx, cy, w, h)
    for j in range(n_hard):
        if j == i:
            continue
        wj, hj = float(sizes[j, 0]), float(sizes[j, 1])
        lj, rj, bj, tj = _bbox(float(pos[j, 0]), float(pos[j, 1]), wj, hj)
        new_a = _overlap_area(li, ri, bi, ti, lj, rj, bj, tj)
        if new_a > baseline[i, j] + _GAP_AREA:
            return False
    return True


def _max_axis_translation(
    i: int,
    pos: np.ndarray,
    sizes: np.ndarray,
    n_hard: int,
    cw: float,
    ch: float,
    baseline: np.ndarray,
    axis: int,
    sign: int,
    step: float,
) -> float:
    """Greedy positive steps along one axis until constraint breaks."""
    total = 0.0
    if axis == 0:
        if sign > 0:
            w = float(sizes[i, 0])
            t_cap = max(0.0, cw - w * 0.5 - float(pos[i, 0]))
        else:
            w = float(sizes[i, 0])
            t_cap = max(0.0, float(pos[i, 0]) - w * 0.5)
    else:
        if sign > 0:
            h = float(sizes[i, 1])
            t_cap = max(0.0, ch - h * 0.5 - float(pos[i, 1]))
        else:
            h = float(sizes[i, 1])
            t_cap = max(0.0, float(pos[i, 1]) - h * 0.5)

    while total + step <= t_cap + 1e-12:
        if axis == 0:
            trial_dcx = total + step if sign > 0 else -(total + step)
            trial_dcy = 0.0
        else:
            trial_dcx = 0.0
            trial_dcy = total + step if sign > 0 else -(total + step)
        if not _ok_move(i, trial_dcx, trial_dcy, pos, sizes, n_hard, cw, ch, baseline):
            break
        total += step
    return total


def _liberty_score(
    i: int,
    pos: np.ndarray,
    sizes: np.ndarray,
    n_hard: int,
    cw: float,
    ch: float,
    baseline: np.ndarray,
    step: float,
) -> float:
    px = _max_axis_translation(i, pos, sizes, n_hard, cw, ch, baseline, 0, +1, step)
    mx = _max_axis_translation(i, pos, sizes, n_hard, cw, ch, baseline, 0, -1, step)
    py = _max_axis_translation(i, pos, sizes, n_hard, cw, ch, baseline, 1, +1, step)
    my = _max_axis_translation(i, pos, sizes, n_hard, cw, ch, baseline, 1, -1, step)
    return px + mx + py + my


def _save_free_figure(
    placement: torch.Tensor,
    benchmark: Benchmark,
    top_mask: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
    top_frac: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    pos = placement.cpu().numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    nh = int(benchmark.num_hard_macros)

    fig, ax = plt.subplots(1, 1, figsize=(12, 11))
    ax.set_xlim(0, cw)
    ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.add_patch(Rectangle((0, 0), cw, ch, fill=False, edgecolor="black", linewidth=1.5))

    for i in range(nh):
        x, y = float(pos[i, 0]), float(pos[i, 1])
        w, h = float(benchmark.macro_sizes[i, 0]), float(benchmark.macro_sizes[i, 1])
        fixed = bool(benchmark.macro_fixed[i].item())
        if fixed:
            face, ec, lw = "lightgray", "darkred", 2.4
        elif top_mask[i]:
            face, ec, lw = "gold", "darkorange", 2.0
        else:
            face, ec, lw = "steelblue", "black", 0.45
        ax.add_patch(
            Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                facecolor=face,
                edgecolor=ec,
                linewidth=lw,
                alpha=0.72,
            )
        )

    n_top = int(top_mask.sum())
    mv = (
        (~benchmark.macro_fixed[:nh])
        & benchmark.get_hard_macro_mask()[:nh]
    ).numpy()
    stat = ""
    if mv.any():
        sm = scores[mv]
        stat = f" | movable Σ-slacks: min {sm.min():.3f}, med {np.median(sm):.3f}, max {sm.max():.3f} μm"
    ax.set_title(
        f"{benchmark.name} — liberty = sum of max Δx⁺+Δx⁻+Δy⁺+Δy⁻ (μm) without overlap growth vs other hard macros\n"
        f"Gold = top {top_frac:.0%} movable by score ({n_top} macros); red outline = fixed{stat}"
    )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="s", color="w", markerfacecolor="gold", markeredgecolor="darkorange", markersize=11, label="Top liberty (movable)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="steelblue", markeredgecolor="black", markersize=11, label="Other movable"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="lightgray", markeredgecolor="darkred", markersize=11, label="Fixed"),
        ],
        loc="upper left",
        fontsize=9,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class FreePlacer:
    """
    Rank movable hard macros by overlap-non-increasing translation slack; draw vis.

    Args:
        top_frac: Fraction of **movable** hard macros to highlight (default 0.25).
        step_um: Greedy step size (μm) for sweeping each axis (smaller = tighter).
    """

    def __init__(self, top_frac: float = 0.25, step_um: float = 0.025):
        self.top_frac = float(top_frac)
        self.step_um = float(step_um)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        n_hard = int(benchmark.num_hard_macros)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        pos = placement[:n_hard].detach().cpu().numpy().copy()
        sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy()
        fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)
        movable = (~fixed) & benchmark.get_hard_macro_mask()[:n_hard].detach().cpu().numpy().astype(bool)

        baseline = _pair_overlaps_current(pos, sizes, n_hard)
        step = max(self.step_um, min(cw, ch) * 1e-5)

        scores = np.zeros(n_hard, dtype=np.float64)
        for i in range(n_hard):
            if not movable[i]:
                continue
            scores[i] = _liberty_score(i, pos, sizes, n_hard, cw, ch, baseline, step)

        movable_idx = np.nonzero(movable)[0]
        k = max(1, int(math.ceil(self.top_frac * len(movable_idx))))
        if len(movable_idx) > 0:
            order = movable_idx[np.argsort(-scores[movable_idx])]
            top_idx = set(order[:k].tolist())
        else:
            top_idx = set()

        top_mask = np.zeros(n_hard, dtype=bool)
        for i in top_idx:
            top_mask[i] = True

        out = _repo_root() / "vis" / f"{benchmark.name}_free_liberty.png"
        _save_free_figure(placement, benchmark, top_mask, scores, out, self.top_frac)

        return placement


def _cli_main() -> None:
    from macro_place.loader import load_benchmark_from_dir

    root = _repo_root()
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    FreePlacer().place(b)
    print(f"Wrote {root / 'vis' / 'ibm01_free_liberty.png'}")


if __name__ == "__main__":
    _cli_main()
