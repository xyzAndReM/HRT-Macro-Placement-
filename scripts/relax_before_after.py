#!/usr/bin/env python3
"""Print relax.py before/after deltas and save placement screenshots (ibm01 by default)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_relax_module():
    path = ROOT / "submissions" / "relax.py"
    spec = importlib.util.spec_from_file_location("relax_submission", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


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
        ("Before (input)", "After RelaxPlacer"),
    ):
        ax.set_xlim(0, benchmark.canvas_width)
        ax.set_ylim(0, benchmark.canvas_height)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Y (μm)")
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
    fig.suptitle(f"{benchmark.name} — relax before / after (hard macros)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    mod = _load_relax_module()
    RelaxPlacer = mod.RelaxPlacer
    _build_area_density_map = mod._build_area_density_map

    case = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    benchmark, plc = load_benchmark_from_dir(str(case))

    before = benchmark.macro_positions.clone()
    pos0 = before.detach().cpu().numpy()

    d0 = _build_area_density_map(pos0, benchmark, include_soft=True)
    stats0 = (
        float(np.mean(d0)),
        float(np.max(d0)),
        float(np.percentile(d0, 99)),
    )

    m0 = compute_proxy_cost(before, benchmark, plc)

    placer = RelaxPlacer()
    after = placer.place(benchmark)
    pos1 = after.detach().cpu().numpy()
    d1 = _build_area_density_map(pos1, benchmark, include_soft=True)
    diff = d1 - d0
    stats1 = (
        float(np.mean(d1)),
        float(np.max(d1)),
        float(np.percentile(d1, 99)),
    )
    l1_change = float(np.sum(np.abs(diff)))
    max_cell = float(np.max(np.abs(diff)))
    m1 = compute_proxy_cost(after, benchmark, plc)

    nh = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed[:nh]
    mask = (~fixed) & benchmark.get_hard_macro_mask()[:nh]
    movable = torch.nonzero(mask, as_tuple=False).flatten().tolist()

    disp = torch.norm(after[:nh, :2] - before[:nh, :2], dim=1)
    moved = [i for i in movable if disp[i].item() > 1e-4]
    max_m = float(disp.max().item()) if nh else 0.0
    max_mov = max((disp[i].item() for i in movable), default=0.0)

    print("=== relax.py doing anything? (ibm01) ===\n")
    print(f"Movable hard macros: {len(movable)}")
    print(f"Macros moved (>1e-4 μm): {len(moved)}  max displacement (any hard): {max_m:.4f} μm")
    print(f"Max displacement (movable only): {max_mov:.4f} μm\n")

    print("--- Area density map (numpy, same as relax Phase 1) ---")
    print(f"  mean:  {stats0[0]:.8f} → {stats1[0]:.8f}   Δ = {stats1[0] - stats0[0]:+.2e}")
    print(f"  max:   {stats0[1]:.6f} → {stats1[1]:.6f}   Δ = {stats1[1] - stats0[1]:+.6f}")
    print(f"  p99:   {stats0[2]:.6f} → {stats1[2]:.6f}   Δ = {stats1[2] - stats0[2]:+.6f}")
    print(f"  Σ|d1-d0| per-cell (L1 map change): {l1_change:.4f}   max |Δcell|: {max_cell:.6f}\n")

    print("--- PlacementCost (evaluator-style) ---")
    for k in (
        "wirelength_cost",
        "density_cost",
        "congestion_cost",
        "proxy_cost",
        "overlap_count",
    ):
        a, b = float(m0[k]), float(m1[k])
        print(f"  {k:22s}  {a:.6f} → {b:.6f}   Δ = {b - a:+.6f}")

    out = ROOT / "vis" / "ibm01_relax_before_after.png"
    _save_side_by_side(before, after, benchmark, out)


if __name__ == "__main__":
    main()
