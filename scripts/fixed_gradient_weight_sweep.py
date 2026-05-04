#!/usr/bin/env python3
"""Run GradientPlacer with fixed (w_wl, w_cong) configs — no proxy-guided adaptation.

Example:
    uv run python scripts/fixed_gradient_weight_sweep.py -b ibm01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from macro_place.evaluate import evaluate_benchmark

from submissions.gradient import GradientPlacer

CONFIGS: list[tuple[float, float]] = [
    (60.0, 20.0),
    (70.0, 15.0),
    (80.0, 10.0),
]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sweep fixed GradientPlacer w_wl / w_cong (500 epochs each, adaptation off).",
    )
    p.add_argument(
        "-b",
        "--benchmark",
        default="ibm01",
        help="ICCAD04 benchmark name (default: ibm01).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Training epochs per configuration (default: 500).",
    )
    args = p.parse_args()

    testcase_root = _REPO_ROOT / "external/MacroPlacement/Testcases/ICCAD04"
    if not testcase_root.is_dir():
        print(f"Missing testcases: {testcase_root}")
        sys.exit(1)

    name = args.benchmark
    print("=" * 72)
    print(f"Fixed-weight sweep · {name} · epochs={args.epochs} · proxy_adaptive_weights=False")
    print("=" * 72)

    for w_wl, w_cong in CONFIGS:
        placer = GradientPlacer(
            epochs=args.epochs,
            w_wl=w_wl,
            w_cong=w_cong,
            proxy_adaptive_weights=False,
            proxy_eval_interval=0,
            proxy_patience=0,
            training_log_csv=None,
            training_log_plot=None,
        )
        r = evaluate_benchmark(placer, name, str(testcase_root))
        dpi, dp = r["proxy_cost_initial"], r["proxy_cost"]
        dlt = (dp - dpi) / dpi * 100.0 if abs(dpi) > 1e-30 else 0.0
        print(
            f"w_wl={w_wl:g}  w_cong={w_cong:g}  "
            f"proxy={dp:.6f}  initial={dpi:.6f}  Δ={dlt:+.2f}%  "
            f"(wl={r['wirelength']:.4f} den={r['density']:.4f} cong={r['congestion']:.4f})  "
            f"overlaps={r['overlaps']}  [{r['runtime']:.2f}s]"
        )

    print()


if __name__ == "__main__":
    main()
