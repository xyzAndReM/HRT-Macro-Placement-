#!/usr/bin/env python3
"""Run SA placer, print proxy / component deltas, save before/after hard-macro figure.

Usage:
    uv run python scripts/sa_before_after.py
    uv run python scripts/sa_before_after.py ibm06
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_sa_module():
    path = ROOT / "submissions" / "sa.py"
    spec = importlib.util.spec_from_file_location("sa_submission", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _save_side_by_side(
    before: torch.Tensor,
    after: torch.Tensor,
    benchmark,
    out_path: Path,
    *,
    proxy_before: float,
    proxy_after: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dp = proxy_after - proxy_before
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, pos, title in zip(
        axes,
        (before, after),
        ("Before (input)", "After SAPlacer"),
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
    fig.suptitle(
        f"{benchmark.name} — SA before / after (hard macros)  |  "
        f"proxy {proxy_before:.4f} → {proxy_after:.4f}  (Δ {dp:+.4f})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def main() -> None:
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    p = argparse.ArgumentParser(description="SA before/after figure + proxy deltas")
    p.add_argument(
        "benchmark",
        nargs="?",
        default="ibm01",
        help="ICCAD04 case name (default: ibm01)",
    )
    args = p.parse_args()
    name = args.benchmark.strip()

    mod = _load_sa_module()
    SAPlacer = mod.SAPlacer

    case = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / name
    benchmark, plc = load_benchmark_from_dir(str(case))

    before = benchmark.macro_positions.clone()
    m0 = compute_proxy_cost(before, benchmark, plc)

    placer = SAPlacer()
    after = placer.place(benchmark)
    m1 = compute_proxy_cost(after, benchmark, plc)

    nh = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed[:nh]
    mask = (~fixed) & benchmark.get_hard_macro_mask()[:nh]
    movable = torch.nonzero(mask, as_tuple=False).flatten().tolist()

    disp = torch.norm(after[:nh, :2] - before[:nh, :2], dim=1)
    moved = [i for i in movable if disp[i].item() > 1e-4]
    max_m = float(disp.max().item()) if nh else 0.0
    max_mov = max((disp[i].item() for i in movable), default=0.0)

    print(f"=== SA ({name}) — before / after ===\n")
    print(
        f"SA max_iters: {placer.max_iters}  step_um: {placer.delta_um} | "
        "surrogate Δ = WL + 0.5·density + 0.5·RUDY; hard–hard overlap non-increasing"
    )
    print(f"Movable hard macros: {len(movable)}")
    print(f"Macros moved (>1e-4 μm): {len(moved)}  max displacement (any hard): {max_m:.4f} μm")
    print(f"Max displacement (movable only): {max_mov:.4f} μm\n")

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

    out = ROOT / "vis" / f"{name}_sa_before_after.png"
    _save_side_by_side(
        before,
        after,
        benchmark,
        out,
        proxy_before=float(m0["proxy_cost"]),
        proxy_after=float(m1["proxy_cost"]),
    )


if __name__ == "__main__":
    main()
