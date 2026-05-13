"""
SimpleGpu: **GpuPlacer** with **Adam** + PLC min-abs stagnation, then **Abbacus** legalization.

GPU phase (defaults baked into this placer, overridable via env where noted):

- Optimizer **Adam** (``MACRO_PLACE_GPU_OPTIMIZER=adam`` for the inner ``GpuPlacer`` call).
- PLC proxy is evaluated every **100** epochs (``MACRO_PLACE_GPU_PROXY_CHECK_EVERY=100``).
  After each check, training stops if the global-best proxy did not improve by at least
  **0.001** in absolute terms since the previous check (``stagnation_min_abs_improvement``).
- Relative stagnation patience is **disabled** (``stagnation_proxy_patience=0``).
- Epoch cap **20000** unless ``MACRO_PLACE_SIMPLE_GPU_EPOCHS`` is set (passed as ``GpuPlacer`` epochs).

Final pass: ``submissions/abbacus.py`` (``AbbacusLegalizer``).

Env:
    MACRO_PLACE_SIMPLE_GPU_EPOCHS — max GPU epochs (default ``20000``)
    MACRO_PLACE_SIMPLE_GPU_STAGNATION_MIN_ABS — min absolute PLC proxy improvement
        required between proxy checks to continue (default ``0.001``)
    MACRO_PLACE_SIMPLE_GPU_PROXY_CHECK_EVERY — epochs between PLC proxy checks (default ``100``)
    Other ``MACRO_PLACE_GPU_*`` settings (e.g. learning rate) still apply when not overridden here.

Unless ``MACRO_PLACE_DEVICE`` is already set, this placer sets it to ``cuda`` for the GPU phase.

Usage:
    uv run evaluate submissions/simple_gpu.py -b ibm01
"""

from __future__ import annotations

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

_DEFAULT_EPOCH_CAP = 20000
_DEFAULT_STAG_MIN_ABS = 0.001
_DEFAULT_PROXY_CHECK_EVERY = 100


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


@contextmanager
def _adam_and_proxy_check_env(proxy_check_every: int):
    """Force Adam and PLC proxy check interval for ``GpuPlacer`` for this call."""
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
    """If ``MACRO_PLACE_DEVICE`` is unset, prefer CUDA for ``GpuPlacer``."""
    if os.environ.get("MACRO_PLACE_DEVICE") is not None:
        yield
        return
    os.environ["MACRO_PLACE_DEVICE"] = "cuda"
    try:
        yield
    finally:
        os.environ.pop("MACRO_PLACE_DEVICE", None)


class SimpleGpuPlacer:
    """
    GpuPlacer (Adam, min-abs stagnation defaults) → Abbacus.

    The evaluate loader instantiates this class with no arguments.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        epoch_cap = _int_env("MACRO_PLACE_SIMPLE_GPU_EPOCHS", _DEFAULT_EPOCH_CAP)
        min_abs = _float_env("MACRO_PLACE_SIMPLE_GPU_STAGNATION_MIN_ABS", _DEFAULT_STAG_MIN_ABS)
        every = _int_env("MACRO_PLACE_SIMPLE_GPU_PROXY_CHECK_EVERY", _DEFAULT_PROXY_CHECK_EVERY)

        gpu = GpuPlacer(
            epochs=epoch_cap,
            stagnation_min_abs_improvement=min_abs,
            stagnation_proxy_patience=0,
        )
        with _default_cuda_for_gpu_phase():
            with _adam_and_proxy_check_env(every):
                pos = gpu.place(benchmark)
        return AbbacusLegalizer().place(benchmark, initial_macro_positions=pos)
