#!/usr/bin/env python3
"""Save before/after hard-macro figures for submissions/snap_spread.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCHMARKS = ("ibm01", "ibm06")


def _load_snap_spread():
    path = ROOT / "submissions" / "snap_spread.py"
    spec = importlib.util.spec_from_file_location("snap_spread_submission", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.SnapSpreadPlacer


def _save_side_by_side(
    before: torch.Tensor,
    after: torch.Tensor,
    benchmark,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, pos, title in zip(
        axes,
        (before, after),
        ("Before (input)", "After SnapSpreadPlacer"),
    ):
        ax.set_xlim(0, benchmark.canvas_width)
        ax.set_xlabel("X (μm)")
        ax.set_ylim(0, benchmark.canvas_height)
        ax.set_ylabel("Y (μm)")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.add_patch(
            Rectangle(
                (0, 0),
                benchmark.canvas_width,
                benchmark.canvas_height,
                fill=False,
                edgecolor="black",
                linewidth=1.5,
            )
        )
        nh = benchmark.num_hard_macros
        for i in range(nh):
            x, y = pos[i].tolist()
            w, h = benchmark.macro_sizes[i].tolist()
            color = "red" if bool(benchmark.macro_fixed[i]) else "steelblue"
            ax.add_patch(
                Rectangle(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.55,
                )
            )
    fig.suptitle(f"{benchmark.name} — snap_spread before / after (hard macros)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _save_nets_pins_side_by_side(
    before: torch.Tensor,
    after: torch.Tensor,
    benchmark,
    plc,
    out_path: Path,
    *,
    wire_alpha: float = 0.14,
) -> None:
    """Before/after with macros, macro pins, I/O ports, and net star wires (like utils panel 1)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    from macro_place.objective import _set_placement
    from macro_place.utils import _draw_canvas

    def draw_panel(ax, pos: torch.Tensor, title: str, show_legend: bool) -> None:
        _set_placement(plc, pos, benchmark)
        _draw_canvas(ax, benchmark)
        num_hard = benchmark.num_hard_macros
        for i in range(benchmark.num_macros):
            x, y = pos[i].tolist()
            w, h = benchmark.macro_sizes[i].tolist()
            is_soft = i >= num_hard
            color = (
                "red"
                if benchmark.macro_fixed[i]
                else "lightsteelblue"
                if is_soft
                else "blue"
            )
            alpha = 0.25 if is_soft else 0.5
            linestyle = "dashed" if is_soft else "solid"
            ax.add_patch(
                Rectangle(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    fill=True,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=0.5,
                    linestyle=linestyle,
                )
            )

        if benchmark.macro_pin_offsets:
            all_pin_x, all_pin_y = [], []
            for i, offsets in enumerate(benchmark.macro_pin_offsets):
                if offsets.shape[0] == 0:
                    continue
                cx, cy = pos[i].tolist()
                all_pin_x.extend((cx + offsets[:, 0]).tolist())
                all_pin_y.extend((cy + offsets[:, 1]).tolist())
            if all_pin_x:
                ax.scatter(all_pin_x, all_pin_y, s=4, c="darkslateblue", zorder=6)

        if benchmark.port_positions.shape[0] > 0:
            ax.scatter(
                benchmark.port_positions[:, 0].tolist(),
                benchmark.port_positions[:, 1].tolist(),
                s=10,
                c="green",
                zorder=5,
                edgecolors="darkgreen",
                linewidths=0.35,
            )

        lines: list[list[tuple[float, float]]] = []
        for driver_name, sink_names in plc.nets.items():
            if driver_name not in plc.mod_name_to_indices:
                continue
            coords = []
            driver_idx = plc.mod_name_to_indices[driver_name]
            dx, dy = plc.modules_w_pins[driver_idx].get_pos()
            coords.append((dx, dy))
            for sink_name in sink_names:
                if sink_name not in plc.mod_name_to_indices:
                    continue
                sink_idx = plc.mod_name_to_indices[sink_name]
                sx, sy = plc.modules_w_pins[sink_idx].get_pos()
                coords.append((sx, sy))
            if len(coords) < 2:
                continue
            avg_x = sum(c[0] for c in coords) / len(coords)
            avg_y = sum(c[1] for c in coords) / len(coords)
            for cx, cy in coords:
                lines.append([(avg_x, avg_y), (cx, cy)])
        if lines:
            ax.add_collection(
                LineCollection(
                    lines,
                    colors="gray",
                    alpha=wire_alpha,
                    linewidths=0.55,
                    zorder=1,
                )
            )

        ax.set_title(title)
        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Y (μm)")
        if show_legend:
            legend_elements = [
                Patch(facecolor="blue", alpha=0.5, edgecolor="black", label="Hard macros"),
                Patch(
                    facecolor="lightsteelblue",
                    alpha=0.25,
                    edgecolor="black",
                    linestyle="dashed",
                    label="Soft macros",
                ),
                Patch(facecolor="red", alpha=0.3, edgecolor="black", label="Fixed macros"),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="darkslateblue",
                    markeredgecolor="darkslateblue",
                    markersize=5,
                    label="Macro pins",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="green",
                    markeredgecolor="darkgreen",
                    markersize=6,
                    label="I/O ports",
                ),
                Line2D([0], [0], color="gray", alpha=0.5, linewidth=2, label="Net spokes"),
            ]
            leg = ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
            leg.set_zorder(10)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    draw_panel(
        axes[0],
        before,
        "Before (input)",
        True,
    )
    draw_panel(
        axes[1],
        after,
        "After SnapSpreadPlacer",
        True,
    )
    fig.suptitle(
        f"{benchmark.name} — placement with pins, I/O ports, and net connectivity (star from net centroid)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _bin_bounds(
    r: int, c: int, cw: float, ch: float, nr: int, nc: int
) -> Tuple[float, float, float, float]:
    cell_w = cw / nc
    cell_h = ch / nr
    return (
        c * cell_w,
        (c + 1) * cell_w,
        r * cell_h,
        (r + 1) * cell_h,
    )


def _macro_indices_overlapping_bin(
    placement: torch.Tensor,
    benchmark,
    r: int,
    c: int,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
) -> List[int]:
    bx0, bx1, by0, by1 = _bin_bounds(r, c, cw, ch, nr, nc)
    out: List[int] = []
    n = benchmark.num_macros
    for i in range(n):
        cx = float(placement[i, 0])
        cy = float(placement[i, 1])
        w = float(benchmark.macro_sizes[i, 0])
        h = float(benchmark.macro_sizes[i, 1])
        mx0, mx1 = cx - w / 2, cx + w / 2
        my0, my1 = cy - h / 2, cy + h / 2
        if mx1 > bx0 and mx0 < bx1 and my1 > by0 and my0 < by1:
            out.append(i)
    return out


def _save_congestion_worst_sources(
    before: torch.Tensor,
    after: torch.Tensor,
    benchmark,
    plc,
    out_path: Path,
) -> None:
    """
    2×2 figure: placements on top; bottom = congestion (max H/V) with worst bin
    outlined and macros intersecting that bin highlighted. Uses PlacementCost
    after syncing each placement.
    """
    from matplotlib.patches import Rectangle

    import matplotlib.pyplot as plt
    from macro_place.objective import _set_placement

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    nr = int(benchmark.grid_rows)
    nc = int(benchmark.grid_cols)
    extent = (0, cw, 0, ch)
    nh = int(benchmark.num_hard_macros)

    def draw_placement(ax, pos: torch.Tensor, title: str) -> None:
        ax.set_xlim(0, cw)
        ax.set_ylim(0, ch)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Y (μm)")
        ax.add_patch(
            Rectangle((0, 0), cw, ch, fill=False, edgecolor="black", linewidth=1.2)
        )
        for i in range(nh):
            x, y = pos[i].tolist()
            w, h = benchmark.macro_sizes[i].tolist()
            color = "red" if bool(benchmark.macro_fixed[i]) else "steelblue"
            ax.add_patch(
                Rectangle(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.35,
                    alpha=0.5,
                )
            )

    def draw_congestion_panel(ax, pos: torch.Tensor, title: str) -> str:
        _set_placement(plc, pos, benchmark)
        plc.get_congestion_cost()
        h_cong = np.asarray(plc.H_routing_cong, dtype=float).reshape(nr, nc)
        v_cong = np.asarray(plc.V_routing_cong, dtype=float).reshape(nr, nc)
        cong = np.maximum(h_cong, v_cong)
        _rpk, _cpk = np.unravel_index(int(np.argmax(cong)), cong.shape)
        r_w, c_w = int(_rpk), int(_cpk)
        peak = float(cong[r_w, c_w])
        pos_flat = cong[cong > 0]
        vmax = float(np.percentile(pos_flat, 99)) if pos_flat.size else 1.0
        vmax = max(vmax, 1e-9)

        im = ax.imshow(
            cong,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="hot",
            alpha=0.65,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="max(H,V) cong.")

        hot = set(_macro_indices_overlapping_bin(pos, benchmark, r_w, c_w, cw, ch, nr, nc))
        for i in range(benchmark.num_macros):
            x, y = pos[i].tolist()
            w, h = benchmark.macro_sizes[i].tolist()
            is_soft = i >= nh
            if is_soft:
                face = "lightgray"
                edgewidth = 0.25
            else:
                face = "red" if bool(benchmark.macro_fixed[i]) else "steelblue"
                edgewidth = 0.35
            edge = "magenta" if i in hot else "black"
            lw = 2.6 if i in hot else edgewidth
            ax.add_patch(
                Rectangle(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=lw,
                    alpha=0.45 if is_soft else 0.5,
                    linestyle="dashed" if is_soft else "solid",
                )
            )

        bx0, bx1, by0, by1 = _bin_bounds(r_w, c_w, cw, ch, nr, nc)
        ax.add_patch(
            Rectangle(
                (bx0, by0),
                bx1 - bx0,
                by1 - by0,
                fill=False,
                edgecolor="lime",
                linewidth=3.0,
                zorder=10,
            )
        )

        names = [benchmark.macro_names[j] for j in sorted(hot)]
        if len(names) > 6:
            name_str = ", ".join(names[:6]) + ", …"
        else:
            name_str = ", ".join(names) if names else "(none — demand is routing-through)"
        ax.set_title(
            f"{title}\n"
            f"Worst bin (row={r_w}, col={c_w}) peak={peak:.4f}\n"
            f"Macros overlapping bin: {name_str}"
        )
        return name_str

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    draw_placement(axes[0, 0], before, "Before — hard macros")
    draw_placement(axes[0, 1], after, "After SnapSpread — hard macros")
    draw_congestion_panel(axes[1, 0], before, "Before — congestion + worst bin")
    draw_congestion_panel(axes[1, 1], after, "After — congestion + worst bin")

    fig.suptitle(
        f"{benchmark.name} — Lime = hottest grid bin; magenta outline = macros intersecting that bin "
        f"(blockers / local area). Congestion also comes from nets crossing tiles.",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _print_cost_deltas(name: str, before: torch.Tensor, after: torch.Tensor, benchmark, plc) -> None:
    from macro_place.objective import compute_proxy_cost

    m0 = compute_proxy_cost(before, benchmark, plc)
    m1 = compute_proxy_cost(after, benchmark, plc)
    keys = (
        "proxy_cost",
        "wirelength_cost",
        "density_cost",
        "congestion_cost",
        "overlap_count",
    )
    print(f"\n=== {name} — PlacementCost metrics (before → after snap_spread) ===")
    print(f"{'metric':<22} {'before':>12} {'after':>12} {'Δ':>12}")
    for k in keys:
        a, b = float(m0[k]), float(m1[k])
        print(f"{k:<22} {a:12.6f} {b:12.6f} {b - a:+12.6f}")


def main() -> None:
    from macro_place.loader import load_benchmark_from_dir

    Placer = _load_snap_spread()

    for name in BENCHMARKS:
        case = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / name
        benchmark, plc = load_benchmark_from_dir(str(case))
        before = benchmark.macro_positions.clone()
        after = Placer().place(benchmark)
        _print_cost_deltas(name, before, after, benchmark, plc)
        out = ROOT / "vis" / f"{name}_snap_spread_before_after.png"
        _save_side_by_side(before, after, benchmark, out)
        out2 = ROOT / "vis" / f"{name}_snap_spread_congestion_worst.png"
        _save_congestion_worst_sources(before, after, benchmark, plc, out2)
        out3 = ROOT / "vis" / f"{name}_snap_spread_nets_pins.png"
        _save_nets_pins_side_by_side(before, after, benchmark, plc, out3)


if __name__ == "__main__":
    main()
