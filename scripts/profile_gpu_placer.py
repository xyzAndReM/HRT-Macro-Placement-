#!/usr/bin/env python3
"""Profile ``GpuPlacer`` surrogate loop via built-in CUDA section timers.

PyTorch's ``torch.profiler`` / autograd profiler often **stalls or becomes impractically
slow** on this codebase's very large backward graphs, so we rely on
``MACRO_PLACE_GPU_PROFILE_SECTIONS=1`` (CUDA Events) implemented in
``submissions/gpu/placer.py``.

Example:

    uv run python scripts/profile_gpu_placer.py -b ibm02 --epochs 20 --warmup 2
    uv run python scripts/profile_gpu_placer.py -b ibm06 --epochs 15 2>&1 | Tee-Object logs/profile_sections.txt

Env forced by this script:
    ``MACRO_PLACE_GPU_PROXY_CHECK_EVERY=0`` — skip PLC proxy on epoch 1 (CPU skew).
    ``MACRO_PLACE_GPU_PROFILE_SECTIONS=1`` — print mean ms/epoch per slice at end.
    ``MACRO_PLACE_GPU_TRAINING_PLOT=0``, ``MACRO_PLACE_GPU_LOG_PATH=nul`` — less noise.

For kernel-level chrome traces on Windows/Linux, use NVIDIA Nsight Systems
(``nsys profile ...``) around the same command instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from macro_place.loader import load_benchmark_from_dir
from submissions.gpu.placer import GpuPlacer


def main() -> None:
    p = argparse.ArgumentParser(description="GpuPlacer surrogate profiling (CUDA section timers).")
    p.add_argument("-b", "--benchmark", default="ibm02", help="ICCAD04 slug under Testcases/ICCAD04.")
    p.add_argument("--epochs", type=int, default=20, help="Timed epochs (after warmup).")
    p.add_argument("--warmup", type=int, default=2, help="Warmup epochs without section report.")
    args = p.parse_args()

    os.environ["MACRO_PLACE_GPU_TRAINING_PLOT"] = "0"
    os.environ["MACRO_PLACE_GPU_PROXY_CHECK_EVERY"] = "0"
    os.environ["MACRO_PLACE_GPU_LOG_PATH"] = os.devnull
    os.environ["MACRO_PLACE_GPU_PROFILE_SECTIONS"] = "1"

    case_dir = _REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / args.benchmark
    if not (case_dir / "netlist.pb.txt").is_file():
        raise SystemExit(f"Missing benchmark: {case_dir}")

    benchmark, _plc = load_benchmark_from_dir(str(case_dir))

    base_kw = dict(
        lr=2e-2,
        log_every=max(args.epochs + args.warmup, 1),
        stagnation_proxy_patience=0,
        stagnation_min_abs_improvement=0.0,
        stagnation_surrogate_patience=0,
        stagnation_surrogate_min_abs=0.0,
        stagnation_surrogate_min_rel_initial=0.0,
        surrogate_stagnation_check_every=0,
        seed=0,
    )

    print(
        f"[profile_gpu_placer] benchmark={args.benchmark} warmup={args.warmup} epochs={args.epochs} "
        f"(MACRO_PLACE_GPU_PROFILE_SECTIONS=1)",
        flush=True,
    )

    if args.warmup > 0:
        os.environ.pop("MACRO_PLACE_GPU_PROFILE_SECTIONS", None)
        GpuPlacer(epochs=args.warmup, **base_kw).place(benchmark)
        os.environ["MACRO_PLACE_GPU_PROFILE_SECTIONS"] = "1"

    GpuPlacer(epochs=args.epochs, **base_kw).place(benchmark)


if __name__ == "__main__":
    main()
