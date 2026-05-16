"""
Electro: **hybrid** global electrostatic spreading → PLC overlap refinement (single benchmark pass).

GPU optimization is split into two chained ``GpuPlacer`` runs:

**Phase A — epochs ``[0, phase2)``** (default phase2=1000), electrostatic density only:

- ``[0, phase1)``: ``w_density = w_high`` (default **24**) — strong global spreading.
- ``[phase1, phase2)``: linear anneal ``w_high → w_final`` (default **2**).
- Fixed surrogate weights (no affine / PLC calibration); grad clipping on electrostatic.

**Phase B — epochs ``[phase2, total)``** (default total=2500 → **1500** PLC epochs):

- ``MACRO_PLACE_GPU_DENSITY_MODEL=plc`` — PLC overlap grid + proxy density scalar.
- ``w_density=0.5``, ``w_overlap=1.0`` (defaults).
- ``affine_calibrate=True`` with ``affine_calibrate_density=False`` — EMA scale matching for **WL
  and congestion only** (same spirit as the liquid patience phase); density weight stays fixed.

Then **Abbacus** (+ optional QP).

Unless ``plc_proxy_include_epoch_zero`` is enabled inside ``GpuPlacer``, PLC proxy checks align with
multiples of ``MACRO_PLACE_GPU_PROXY_CHECK_EVERY`` (default **500**), yielding surrogate probes near
cumulative epochs **500, 1000, …, total**.

Usage:
    uv run evaluate submissions/electro.py -b ibm01

Env (subset):

    MACRO_PLACE_ELECTRO_TOTAL_EPOCHS — GPU epochs across both runs (default ``2500``).
      Alias: ``MACRO_PLACE_ELECTRO_EPOCHS`` if ``TOTAL`` unset.
    MACRO_PLACE_ELECTRO_PHASE1 / MACRO_PLACE_ELECTRO_PHASE2 — electro schedule breakpoints (``500`` / ``1000``).
    MACRO_PLACE_ELECTRO_W_DEN_HIGH / MACRO_PLACE_ELECTRO_W_DEN_FINAL — electro density weights (``24`` / ``2``).
    MACRO_PLACE_ELECTRO_LR — Adam LR for **phase A** (default ``0.02``).
    MACRO_PLACE_ELECTRO_W_CONG / MACRO_PLACE_ELECTRO_W_OVERLAP — phase A congestion/overlap weights (``0.5``).
    MACRO_PLACE_ELECTRO_PHASE3_LR — Adam LR for **phase B** (defaults to phase A LR).
    MACRO_PLACE_ELECTRO_PHASE3_W_DENSITY / MACRO_PLACE_ELECTRO_PHASE3_W_CONG /
      MACRO_PLACE_ELECTRO_PHASE3_W_OVERLAP — PLC phase (``0.5`` / ``0.5`` / ``1.0``).
    MACRO_PLACE_ELECTRO_PHASE3_STAG_MIN_ABS / MACRO_PLACE_ELECTRO_PHASE3_STAG_PATIENCE —
      PLC proxy stagnation (``0.0001`` / ``4``).
    MACRO_PLACE_GPU_PROXY_CHECK_EVERY — default ``500`` when unset for milestone-aligned proxy logs.
    MACRO_PLACE_ELECTRO_SEED — RNG seed for phase A; phase B uses ``seed + 100_000``.
    MACRO_PLACE_ELECTRO_QP_IF_OVERLAPS — ``1`` → ``QPLegalizer`` after Abbacus if overlaps remain.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.abbacus import AbbacusLegalizer  # noqa: E402
from submissions.gpu.placer import GpuPlacer  # noqa: E402
from submissions.qp import QPLegalizer  # noqa: E402

_DEFAULT_TOTAL_EPOCHS = 2500
_DEFAULT_PHASE1 = 500
_DEFAULT_PHASE2 = 1000
_DEFAULT_W_DEN_HIGH = 24.0
_DEFAULT_W_DEN_FINAL = 2.0
_DEFAULT_LR = 2e-2
_DEFAULT_W_CONG = 0.5
_DEFAULT_W_OVERLAP = 0.5
_DEFAULT_PROXY_CHECK_EVERY = 500
_DEFAULT_PHASE3_STAG_MIN_ABS = 0.0001
_DEFAULT_PHASE3_STAG_PATIENCE = 4


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _electro_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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


@contextmanager
def _density_model_env(mode: str):
    key = "MACRO_PLACE_GPU_DENSITY_MODEL"
    prev = os.environ.get(key)
    os.environ[key] = mode.strip().lower()
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _density_schedule(
    *,
    phase1: int,
    phase2: int,
    w_high: float,
    w_final: float,
) -> Callable[[int], float]:
    span = float(max(phase2 - phase1, 1))

    def sched(epoch: int) -> float:
        if epoch < phase1:
            return float(w_high)
        if epoch < phase2:
            t = float(epoch - phase1) / span
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            return float(w_high + (w_final - w_high) * t)
        return float(w_final)

    return sched


class ElectroPlacer:
    """Electro global (FFT) + PLC local refinement → Abbacus (+ optional QP)."""

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        total_raw = (os.environ.get("MACRO_PLACE_ELECTRO_TOTAL_EPOCHS") or "").strip()
        if total_raw:
            total_epochs = max(1, int(total_raw))
        else:
            total_epochs = max(1, _int_env("MACRO_PLACE_ELECTRO_EPOCHS", _DEFAULT_TOTAL_EPOCHS))

        phase1 = max(0, _int_env("MACRO_PLACE_ELECTRO_PHASE1", _DEFAULT_PHASE1))
        phase2 = max(0, _int_env("MACRO_PLACE_ELECTRO_PHASE2", _DEFAULT_PHASE2))
        if phase1 > phase2:
            phase1, phase2 = phase2, phase1

        electro_epochs = min(phase2, total_epochs)
        plc_epochs = max(0, total_epochs - electro_epochs)

        w_high = _float_env("MACRO_PLACE_ELECTRO_W_DEN_HIGH", _DEFAULT_W_DEN_HIGH)
        w_final = _float_env("MACRO_PLACE_ELECTRO_W_DEN_FINAL", _DEFAULT_W_DEN_FINAL)
        lr_electro = _float_env("MACRO_PLACE_ELECTRO_LR", _DEFAULT_LR)
        w_cong_a = _float_env("MACRO_PLACE_ELECTRO_W_CONG", _DEFAULT_W_CONG)
        w_ovl_a = _float_env("MACRO_PLACE_ELECTRO_W_OVERLAP", _DEFAULT_W_OVERLAP)
        seed = _int_env("MACRO_PLACE_ELECTRO_SEED", 0)
        qp_if_overlaps = _electro_env_bool("MACRO_PLACE_ELECTRO_QP_IF_OVERLAPS")

        lr_plc = _float_env("MACRO_PLACE_ELECTRO_PHASE3_LR", lr_electro)
        w_den_plc = _float_env("MACRO_PLACE_ELECTRO_PHASE3_W_DENSITY", 0.5)
        w_cong_plc = _float_env("MACRO_PLACE_ELECTRO_PHASE3_W_CONG", _DEFAULT_W_CONG)
        w_ovl_plc = _float_env("MACRO_PLACE_ELECTRO_PHASE3_W_OVERLAP", 1.0)
        stag_min_abs = _float_env(
            "MACRO_PLACE_ELECTRO_PHASE3_STAG_MIN_ABS", _DEFAULT_PHASE3_STAG_MIN_ABS
        )
        stag_patience = _int_env(
            "MACRO_PLACE_ELECTRO_PHASE3_STAG_PATIENCE", _DEFAULT_PHASE3_STAG_PATIENCE
        )

        os.environ.setdefault(
            "MACRO_PLACE_GPU_PROXY_CHECK_EVERY", str(_DEFAULT_PROXY_CHECK_EVERY)
        )

        schedule = _density_schedule(
            phase1=min(phase1, electro_epochs),
            phase2=electro_epochs,
            w_high=w_high,
            w_final=w_final,
        )

        p1_eff = min(phase1, electro_epochs)
        print(
            f"[electro] hybrid GpuPlacer total_epochs={total_epochs} | "
            f"phase_A electro [0,{electro_epochs}) schedule "
            f"[0,{p1_eff}) w_den={w_high:g}; [{p1_eff},{electro_epochs}) anneal→{w_final:g} "
            f"| lr={lr_electro:g} w_cong={w_cong_a:g} w_ovl={w_ovl_a:g}",
            flush=True,
        )
        if plc_epochs > 0:
            print(
                f"[electro] phase_B PLC epochs={plc_epochs} "
                f"w_density={w_den_plc:g} w_cong={w_cong_plc:g} w_overlap={w_ovl_plc:g} lr={lr_plc:g} "
                f"affine(WL+cong only) stag_min_abs={stag_min_abs:g} patience={stag_patience}",
                flush=True,
            )
        else:
            print("[electro] phase_B skipped (total_epochs <= electro cutoff)", flush=True)

        pos = benchmark.macro_positions.clone()

        with _default_cuda_for_gpu_phase():
            if electro_epochs > 0:
                with _density_model_env("electrostatic"):
                    gpu_a = GpuPlacer(
                        epochs=electro_epochs,
                        lr=lr_electro,
                        w_wl=1.0,
                        w_density=w_final,
                        w_cong=w_cong_a,
                        w_overlap=w_ovl_a,
                        affine_calibrate=False,
                        stagnation_proxy_patience=0,
                        stagnation_min_abs_improvement=0.0,
                        stagnation_surrogate_patience=0,
                        stagnation_surrogate_min_abs=0.0,
                        stagnation_surrogate_min_rel_initial=0.0,
                        seed=seed,
                        w_density_schedule=schedule,
                    )
                    pos = gpu_a.place(
                        benchmark,
                        initial_macro_positions=pos,
                        epoch_display_base=0,
                        epoch_display_cap=total_epochs,
                        plc_proxy_include_epoch_zero=False,
                    )

            if plc_epochs > 0:
                with _density_model_env("plc"):
                    gpu_b = GpuPlacer(
                        epochs=plc_epochs,
                        lr=lr_plc,
                        w_wl=1.0,
                        w_density=w_den_plc,
                        w_cong=w_cong_plc,
                        w_overlap=w_ovl_plc,
                        affine_calibrate=True,
                        affine_calibrate_density=False,
                        stagnation_proxy_patience=stag_patience,
                        stagnation_min_abs_improvement=stag_min_abs,
                        stagnation_surrogate_patience=0,
                        stagnation_surrogate_min_abs=0.0,
                        stagnation_surrogate_min_rel_initial=0.0,
                        seed=seed + 100_000,
                    )
                    pos = gpu_b.place(
                        benchmark,
                        initial_macro_positions=pos,
                        epoch_display_base=electro_epochs,
                        epoch_display_cap=total_epochs,
                        plc_proxy_include_epoch_zero=False,
                    )

        pos = AbbacusLegalizer().place(benchmark, initial_macro_positions=pos)

        if qp_if_overlaps:
            ov = compute_overlap_metrics(pos, benchmark)
            n_pairs = int(ov["overlap_count"])
            if n_pairs > 0:
                print(
                    f"[electro] Abbacus: {n_pairs} overlapping hard-macro pair(s); "
                    "running QPLegalizer...",
                    flush=True,
                )
                pos = QPLegalizer().place(benchmark, initial_macro_positions=pos)
                ov_qp = compute_overlap_metrics(pos, benchmark)
                print(
                    f"[electro] post-QP: {int(ov_qp['overlap_count'])} overlapping "
                    "hard-macro pair(s)",
                    flush=True,
                )

        return pos
