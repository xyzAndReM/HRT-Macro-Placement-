"""
Liquid: alternating **liquid** and **patience** ``GpuPlacer`` passes, then **Abbacus**.

Schedule (default **5** outer cycles, env ``MACRO_PLACE_LIQUID_CYCLES``):

  Repeat:
    1. **Liquid GPU** — fixed **300** epochs, ``w_overlap=0``, ``w_density=0.8``, no PLC stagnation.
    2. **Patience GPU** — Adam, PLC proxy every **200** epochs, min-abs stagnation **0.0005**,
       epoch cap **20000** (or env).

  Then **Abbacus** hard-macro legalization.

Env:
    MACRO_PLACE_LIQUID_CYCLES — liquid→patience pairs (default ``5``)
    MACRO_PLACE_LIQUID_GPU_EPOCHS — liquid phase epoch budget (default ``300``)
    MACRO_PLACE_LIQUID_W_DENSITY — liquid ``w_density`` (default ``0.8``)
    MACRO_PLACE_LIQUID_W_OVERLAP — liquid ``w_overlap`` (default ``0``)
    MACRO_PLACE_LIQUID_PATIENCE_EPOCHS — patience phase epoch cap (default ``20000``)
    MACRO_PLACE_LIQUID_STAGNATION_MIN_ABS — patience min-abs PLC improvement (default ``0.0005``)
    MACRO_PLACE_LIQUID_PROXY_CHECK_EVERY — patience proxy check interval (default ``200``)
    Other ``MACRO_PLACE_GPU_*`` settings apply when not overridden here.

Unless ``MACRO_PLACE_DEVICE`` is unset, GPU phases prefer ``cuda``.

Usage:
    uv run evaluate submissions/liquid.py -b ibm01
"""

from __future__ import annotations

import gc
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.abbacus import AbbacusLegalizer  # noqa: E402
from submissions.gpu.placer import GpuPlacer  # noqa: E402

_DEFAULT_CYCLES = 5
_DEFAULT_LIQUID_EPOCHS = 300
_DEFAULT_LIQUID_W_DENSITY = 0.8
_DEFAULT_LIQUID_W_OVERLAP = 0.0
_DEFAULT_PATIENCE_EPOCH_CAP = 20000
_DEFAULT_STAG_MIN_ABS = 0.0005
_DEFAULT_PROXY_CHECK_EVERY = 200


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    return max(1, int(raw))


def _float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _cuda_reset_between_phases() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


@contextmanager
def _adam_and_proxy_check_env(proxy_check_every: int):
    saved: dict[str, str | None] = {
        "MACRO_PLACE_GPU_OPTIMIZER": os.environ.get("MACRO_PLACE_GPU_OPTIMIZER"),
        "MACRO_PLACE_GPU_PROXY_CHECK_EVERY": os.environ.get("MACRO_PLACE_GPU_PROXY_CHECK_EVERY"),
    }
    os.environ["MACRO_PLACE_GPU_OPTIMIZER"] = "adam"
    os.environ["MACRO_PLACE_GPU_PROXY_CHECK_EVERY"] = str(int(proxy_check_every))
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@contextmanager
def _default_cuda_for_gpu_phase():
    if os.environ.get("MACRO_PLACE_DEVICE") is not None:
        yield
        return
    os.environ["MACRO_PLACE_DEVICE"] = "cuda"
    try:
        yield
    finally:
        os.environ.pop("MACRO_PLACE_DEVICE", None)


class LiquidPlacer:
    """
    (Liquid GpuPlacer → patience GpuPlacer) × N cycles → Abbacus.

    The evaluate loader instantiates this class with no arguments.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        cycles = _int_env("MACRO_PLACE_LIQUID_CYCLES", _DEFAULT_CYCLES)
        liquid_epochs = _int_env("MACRO_PLACE_LIQUID_GPU_EPOCHS", _DEFAULT_LIQUID_EPOCHS)
        liquid_w_den = _float_env("MACRO_PLACE_LIQUID_W_DENSITY", _DEFAULT_LIQUID_W_DENSITY)
        liquid_w_ovl = _float_env("MACRO_PLACE_LIQUID_W_OVERLAP", _DEFAULT_LIQUID_W_OVERLAP)
        patience_epochs = _int_env("MACRO_PLACE_LIQUID_PATIENCE_EPOCHS", _DEFAULT_PATIENCE_EPOCH_CAP)
        min_abs = _float_env("MACRO_PLACE_LIQUID_STAGNATION_MIN_ABS", _DEFAULT_STAG_MIN_ABS)
        every = _int_env("MACRO_PLACE_LIQUID_PROXY_CHECK_EVERY", _DEFAULT_PROXY_CHECK_EVERY)

        pos = benchmark.macro_positions.clone()

        print(
            f"[liquid] cycles={cycles} liquid_epochs={liquid_epochs} "
            f"liquid_w_density={liquid_w_den} liquid_w_overlap={liquid_w_ovl} "
            f"patience_epochs={patience_epochs} stagnation_min_abs={min_abs} "
            f"proxy_check_every={every}",
            flush=True,
        )

        with _default_cuda_for_gpu_phase():
            with _adam_and_proxy_check_env(every):
                for cycle in range(cycles):
                    _cuda_reset_between_phases()
                    gpu_liquid = GpuPlacer(
                        epochs=liquid_epochs,
                        w_overlap=liquid_w_ovl,
                        w_density=liquid_w_den,
                        stagnation_proxy_patience=0,
                        stagnation_min_abs_improvement=0.0,
                        seed=cycle * 10_000,
                    )
                    pos = gpu_liquid.place(benchmark, initial_macro_positions=pos)
                    _cuda_reset_between_phases()

                    gpu_patience = GpuPlacer(
                        epochs=patience_epochs,
                        stagnation_min_abs_improvement=min_abs,
                        stagnation_proxy_patience=0,
                        seed=cycle * 10_000 + 5_000,
                    )
                    pos = gpu_patience.place(benchmark, initial_macro_positions=pos)
                    _cuda_reset_between_phases()

                    print(f"[liquid] finished cycle {cycle + 1}/{cycles}", flush=True)

        return AbbacusLegalizer().place(benchmark, initial_macro_positions=pos)
