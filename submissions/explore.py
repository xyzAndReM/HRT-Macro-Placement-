"""
Random local exploration on a coarse grid (``grid_side ** 2`` cells).

Each **epoch**: pick a **non-fixed** macro (hard or soft), try up to nine neighboring **cell
centers**; accept the best if it strictly lowers ``FastProxyEvaluator.total``. Only
targets passing ``_legal_center`` (macro bbox fully inside the canvas) are tried.

Non-fixed macro centers are **clamped** to the canvas before the loop and again
before return (via ``finalize_explore_placement``) so downstream GPU phases never
receive OOB or non-finite coordinates. ``ExploreMultiplePlacer`` replays move logs
with the same finalizer so parent and worker layouts stay consistent.

At the end, prints one summary with fast_proxy change and (when PlacementCost exists)
real proxy before/after.

Writes ``vis/<benchmark>_explore.png``: initial vs final placement (moved macros
highlighted).

Usage:
    uv run evaluate submissions/explore.py -b ibm01
    uv run python submissions/explore.py
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark
from macro_place.fast_proxy import FastProxyEvaluator
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

# Total coarse regions = grid_side ** 2 (default 8×8 = 64).
_GRID_SIDE = 8


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _try_load_plc(benchmark: Benchmark):
    """Return ``PlacementCost`` for ICCAD04 testcase dirs, else ``None``."""
    root = _repo_root()
    case_dir = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / benchmark.name
    if not (case_dir / "netlist.pb.txt").is_file():
        return None
    _, plc = load_benchmark_from_dir(str(case_dir))
    return plc


def _moved_macro_indices(pb: torch.Tensor, pa: torch.Tensor, eps: float = 1e-6) -> list[int]:
    d = (pb - pa).abs().max(dim=1).values
    return [i for i in range(pb.shape[0]) if float(d[i].item()) > eps]


def _save_explore_figure(
    benchmark: Benchmark,
    pos_before: torch.Tensor,
    pos_after: torch.Tensor,
    *,
    grid_side: int,
    epochs: int,
    accepted: int,
    proxy_before: float | None,
    proxy_after: float | None,
    fast_before: float,
    fast_after: float,
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

    gn = int(grid_side)
    for g in range(gn + 1):
        ax.axvline(g * (cw / gn), color="0.88", linewidth=0.22, zorder=0)
        ax.axhline(g * (ch / gn), color="0.88", linewidth=0.22, zorder=0)

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

    proxy_line = ""
    if proxy_before is not None and proxy_after is not None:
        proxy_line = f"Real proxy {proxy_before:.6f} -> {proxy_after:.6f}  |  "
    ax.set_title(
        f"{benchmark.name} — ExplorePlacer ({len(moved)} macros moved)\n"
        f"{proxy_line}"
        f"Fast cost {fast_before:.6f} -> {fast_after:.6f}  |  "
        f"{gn}x{gn} search grid  |  {epochs} epochs, {accepted} accepted moves"
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def _clamp_nonfixed_macro_centers_inplace(
    placement: torch.Tensor,
    benchmark: Benchmark,
    cw: float,
    ch: float,
) -> None:
    """Clamp every non-fixed macro center so its axis-aligned box stays in the canvas.

    Matches ``gradient._clamp_movable`` half-extent bounds. If a macro is wider/taller
    than the canvas, the center is pinned to the canvas midpoint on that axis.
    """
    n = int(benchmark.num_macros)
    fixed = benchmark.macro_fixed
    sizes = benchmark.macro_sizes
    for i in range(n):
        if bool(fixed[i].item()):
            continue
        w = float(sizes[i, 0].item()) * 0.5
        h = float(sizes[i, 1].item()) * 0.5
        if w <= 0.0 or h <= 0.0:
            continue
        lo_x, hi_x = w, cw - w
        lo_y, hi_y = h, ch - h
        x = placement[i, 0]
        y = placement[i, 1]
        if lo_x > hi_x:
            x.fill_(0.5 * cw)
        else:
            x.clamp_(lo_x, hi_x)
        if lo_y > hi_y:
            y.fill_(0.5 * ch)
        else:
            y.clamp_(lo_y, hi_y)


def finalize_explore_placement(
    placement: torch.Tensor,
    benchmark: Benchmark,
    *,
    fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    """Make an explore output safe for downstream GPU / PLC: in-bounds, finite, fixed macros restored.

    If canvas / finiteness checks still fail and ``fallback`` is given, return ``fallback`` unchanged
    (explore is treated as a no-op for that candidate).
    """
    from macro_place.utils import validate_placement

    ref = fallback if fallback is not None else benchmark.macro_positions
    out = placement.detach().clone().to(
        device=benchmark.macro_positions.device,
        dtype=benchmark.macro_positions.dtype,
    )
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    _clamp_nonfixed_macro_centers_inplace(out, benchmark, cw, ch)

    fixed = benchmark.macro_fixed
    out[fixed] = benchmark.macro_positions[fixed]

    if not torch.isfinite(out).all():
        bad = ~torch.isfinite(out).all(dim=1)
        for i in range(int(out.shape[0])):
            if bool(bad[i].item()) and not bool(fixed[i].item()):
                out[i] = ref[i]

    ok, _ = validate_placement(out, benchmark, check_overlaps=False)
    if not ok and fallback is not None:
        return fallback.detach().clone().to(
            device=benchmark.macro_positions.device,
            dtype=benchmark.macro_positions.dtype,
        )
    return out


def _nine_neighbor_cells(r0: int, c0: int, n: int) -> list[tuple[int, int]]:
    """3×3 neighborhood (Chebyshev distance ≤ 1), clipped to the grid."""
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < n and 0 <= c < n:
                out.append((r, c))
    return out


def _precompute_nine_neighbors(n: int) -> list[list[list[tuple[int, int]]]]:
    """``neighbors[r][c]`` = same list as ``_nine_neighbor_cells(r, c, n)``."""
    return [[_nine_neighbor_cells(r, c, n) for c in range(n)] for r in range(n)]


def _movable_macro_indices(
    benchmark: Benchmark,
    *,
    hard_only: bool = False,
    soft_only: bool = False,
) -> list[int]:
    """Non-fixed macro indices; optional restriction to hard or soft macro ranges."""
    if hard_only and soft_only:
        raise ValueError("hard_only and soft_only are mutually exclusive")
    n_hard = int(benchmark.num_hard_macros)
    n_macros = int(benchmark.num_macros)
    if hard_only:
        lo, hi = 0, n_hard
    elif soft_only:
        lo, hi = n_hard, n_macros
    else:
        lo, hi = 0, n_macros
    return [i for i in range(lo, hi) if not bool(benchmark.macro_fixed[i].item())]


class ExplorePlacer:
    """
    Random single-macro moves toward one of nine coarse cell centers per epoch.

    Args:
        grid_side: Cells per axis; total regions = ``grid_side ** 2`` (default 8 → 64).
        epochs: Number of random macro trials.
        seed: RNG seed for macro selection.
        moved_macro_weight: After a macro has been moved once, its selection weight
            becomes this value (default ``1.0`` = uniform). Used by ``ExploreMultiplePlacer``.
        fast_proxy_device: Optional ``torch.device`` for ``FastProxyEvaluator`` (e.g. ``\"cpu\"``).
        hard_macros_only: If ``True``, only non-fixed **hard** macros are candidates.
        soft_macros_only: If ``True``, only non-fixed **soft** macros are candidates.
            Mutually exclusive with ``hard_macros_only``.

    ``place(..., quiet=True)`` skips epoch summaries and real_proxy lines (used by
    ``ExploreMultiplePlacer`` worker processes).
    """

    def __init__(
        self,
        grid_side: int = _GRID_SIDE,
        epochs: int = 2_500,
        seed: int = 0,
        moved_macro_weight: float = 1.0,
        fast_proxy_device: torch.device | str | None = None,
        hard_macros_only: bool = False,
        soft_macros_only: bool = False,
    ):
        self.grid_side = int(grid_side)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.moved_macro_weight = float(moved_macro_weight)
        self.fast_proxy_device = fast_proxy_device
        self.hard_macros_only = bool(hard_macros_only)
        self.soft_macros_only = bool(soft_macros_only)
        if self.hard_macros_only and self.soft_macros_only:
            raise ValueError("hard_macros_only and soft_macros_only are mutually exclusive")

    def place(
        self,
        benchmark: Benchmark,
        *,
        initial_macro_positions: torch.Tensor | None = None,
        save_figure: bool = True,
        accepted_moves: list[tuple[int, float, float]] | None = None,
        quiet: bool = False,
    ) -> torch.Tensor:
        if initial_macro_positions is not None:
            placement = initial_macro_positions.detach().clone().to(
                device=benchmark.macro_positions.device,
                dtype=benchmark.macro_positions.dtype,
            )
        else:
            placement = benchmark.macro_positions.clone()

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        placement = finalize_explore_placement(placement, benchmark, fallback=placement)
        placement_start = placement.clone()

        dev = None if self.fast_proxy_device is None else torch.device(self.fast_proxy_device)
        score = FastProxyEvaluator(benchmark, device=dev)

        plc = _try_load_plc(benchmark)
        if plc is None and not quiet:
            print(
                "ExplorePlacer: no ICCAD04 PlacementCost for this benchmark; "
                "skipping periodic real_proxy logs (fast cost only)."
            )

        n = self.grid_side
        rng = random.Random(self.seed)

        movable = _movable_macro_indices(
            benchmark,
            hard_only=self.hard_macros_only,
            soft_only=self.soft_macros_only,
        )
        if not movable:
            if not quiet:
                print("ExplorePlacer: no movable macros; placement unchanged.")
            return placement

        moved_w = float(self.moved_macro_weight)
        if not (0.0 < moved_w <= 1.0):
            raise ValueError("moved_macro_weight must be in (0, 1].")
        moved_flags = {i: False for i in movable}
        weights = [1.0 for _ in movable]

        sur0 = score.total(placement)
        accepted = 0
        proxy_start: float | None = None
        if plc is not None:
            proxy_start = float(
                compute_proxy_cost(placement.clone(), benchmark, plc)["proxy_cost"]
            )

        for ep in range(self.epochs):
            i_macro = rng.choices(movable, weights=weights, k=1)[0]
            w = float(benchmark.macro_sizes[i_macro, 0])
            h = float(benchmark.macro_sizes[i_macro, 1])
            x0 = float(placement[i_macro, 0])
            y0 = float(placement[i_macro, 1])

            r0, c0, cell_w, cell_h = _cell_xy_to_rc(x0, y0, cw, ch, n)
            candidates = _nine_neighbor_cells(r0, c0, n)

            base_cost = score.total(placement)
            best_trial_p = base_cost
            best_trial = placement.clone()

            for r, c in candidates:
                cx, cy = _cell_center(r, c, cell_w, cell_h)
                if not _legal_center(cx, cy, w, h, cw, ch):
                    continue
                trial = placement.clone()
                trial[i_macro, 0] = cx
                trial[i_macro, 1] = cy
                p = score.total(trial)
                if p < best_trial_p:
                    best_trial_p = p
                    best_trial = trial.clone()

            if best_trial_p < base_cost - 1e-12 and torch.isfinite(best_trial).all():
                placement = best_trial
                accepted += 1
                if accepted_moves is not None:
                    accepted_moves.append(
                        (int(i_macro), float(placement[i_macro, 0]), float(placement[i_macro, 1]))
                    )
                if not moved_flags[i_macro]:
                    moved_flags[i_macro] = True
                    for j, m in enumerate(movable):
                        if moved_flags[m]:
                            weights[j] = moved_w

        placement = finalize_explore_placement(placement, benchmark, fallback=placement_start)

        sur_end = score.total(placement)
        delta_pct = (sur_end - sur0) / max(abs(sur0), 1e-30) * 100.0
        if not quiet:
            print(
                f"ExplorePlacer: {self.epochs} epochs, grid {n}x{n} ({n * n} cells), "
                f"accepted {accepted} improving moves; "
                f"fast_proxy {sur0:.6f} -> {sur_end:.6f} ({delta_pct:+.3f}%)."
            )

        proxy_end = None
        if plc is not None:
            proxy_end = float(compute_proxy_cost(placement.clone(), benchmark, plc)["proxy_cost"])
            assert proxy_start is not None
            dpx = (proxy_end - proxy_start) / max(abs(proxy_start), 1e-30) * 100.0
            if not quiet:
                print(
                    f"ExplorePlacer: real_proxy {proxy_start:.6f} -> {proxy_end:.6f} ({dpx:+.3f}%)."
                )

        if save_figure:
            out_vis = _repo_root() / "vis" / f"{benchmark.name}_explore.png"
            _save_explore_figure(
                benchmark,
                placement_start,
                placement,
                grid_side=n,
                epochs=self.epochs,
                accepted=accepted,
                proxy_before=proxy_start,
                proxy_after=proxy_end,
                fast_before=sur0,
                fast_after=sur_end,
                out_path=out_vis,
            )
            if not quiet:
                print(f"ExplorePlacer: figure saved to {out_vis.resolve()}")

        return placement


def _cli_main() -> None:
    root = _repo_root()
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    ExplorePlacer(seed=1, epochs=500).place(b)


if __name__ == "__main__":
    _cli_main()
