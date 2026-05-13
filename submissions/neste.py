"""
Neste: **GpuPlacer** with Nesterov SGD (1000 epochs) + **Abbacus** hard-macro legalization.

- GPU phase uses ``submissions/gpu/placer.py`` (``GpuPlacer``; Nesterov SGD is the default
  optimizer there—``neste.py`` still forces ``MACRO_PLACE_GPU_OPTIMIZER=nesterov`` when
  running this placer so behavior stays explicit).
- Final pass: ``submissions/abbacus.py`` (``AbbacusLegalizer``).

Env:
    MACRO_PLACE_NESTE_EPOCHS — override default ``1000`` GPU epochs
    Other ``MACRO_PLACE_GPU_*`` variables apply to ``GpuPlacer`` (lr, momentum, etc.)

Usage:
    uv run evaluate submissions/neste.py -b ibm01
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

_DEFAULT_EPOCHS = 1000


@contextmanager
def _nesterov_optimizer_env():
    """Temporarily set ``MACRO_PLACE_GPU_OPTIMIZER=nesterov`` for ``GpuPlacer``."""
    old = os.environ.get("MACRO_PLACE_GPU_OPTIMIZER")
    os.environ["MACRO_PLACE_GPU_OPTIMIZER"] = "nesterov"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MACRO_PLACE_GPU_OPTIMIZER", None)
        else:
            os.environ["MACRO_PLACE_GPU_OPTIMIZER"] = old


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


def _neste_epochs() -> int:
    v = os.environ.get("MACRO_PLACE_NESTE_EPOCHS", str(_DEFAULT_EPOCHS))
    return max(1, int(v or str(_DEFAULT_EPOCHS)))


class NestePlacer:
    """
    GpuPlacer (Nesterov, fixed epoch budget) → Abbacus legalizer.

    The evaluate loader instantiates this class with no arguments.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        epochs = _neste_epochs()
        gpu = GpuPlacer(
            epochs=epochs,
            stagnation_proxy_patience=0,
        )
        with _default_cuda_for_gpu_phase():
            with _nesterov_optimizer_env():
                pos = gpu.place(benchmark)
        return AbbacusLegalizer().place(benchmark, initial_macro_positions=pos)
