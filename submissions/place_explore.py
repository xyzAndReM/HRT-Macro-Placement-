"""
Hybrid placer: **GpuPlacer** (DreamPlace-inspired) until PLC proxy stagnates, then
**ExploreMultiplePlacer** (parallel ``ExplorePlacer`` runs, best-of by fast proxy)
for a fixed number of epochs per seed population.

Default schedule (``MACRO_PLACE_PE_OUTER_ROUNDS=1`` — **one** outer cycle, then final GPU):

  Per cycle:
  1. **GPU** until stagnation (PLC min-abs **0.0001** between proxy checks every **100** epochs).
  2. **Explore** — **25** epochs per seed, single wave, **10** seeds, soft movable macros only.
  3. **Liquid GPU** — fixed **50** epochs; same ``w_wl`` / ``w_cong`` as patience; only ``w_density=0.8``, ``w_overlap=0.5`` for exploration.

  After all cycles:
  4. **Final GPU** until stagnation (min-abs **0.0001**, env ``MACRO_PLACE_PE_FINAL_GPU_STAGNATION_MIN_ABS_IMPROVEMENT``).
  5. **Abbacus** hard-macro legalization.

Override ``MACRO_PLACE_PE_OUTER_ROUNDS`` to change how many outer cycles run before the final GPU.

At exit we print how many **hard–hard** overlapping macro pairs remain
(``compute_overlap_metrics``), after legalization.

**Proxy log** (append CSV): set ``MACRO_PLACE_PE_LOG_PATH`` (default
``place_explore_proxy.csv``). Columns include cycle, phase, PLC proxy and sub-costs.

**Devices:** the GPU phase sets ``MACRO_PLACE_DEVICE=cuda`` for ``GpuPlacer`` (falls
back to CPU if CUDA is unavailable). The explore phase uses ``FastProxyEvaluator``
on **CPU** in worker processes regardless of ``MACRO_PLACE_DEVICE``.
``ExploreMultiplePlacer`` logs per-wave metrics and a short summary by default; set
``MACRO_PLACE_PE_EXPLORE_VERBOSE=0`` to silence. Explore pool workers set ``CUDA_VISIBLE_DEVICES=`` (empty) so they do not share the parent's GPU
context (avoids sporadic ``cudaErrorUnknown`` on Windows after multiprocessing).
Set ``MACRO_PLACE_PE_EXPLORE_WORKER_USE_CUDA=1`` only if you intentionally run
CUDA inside explore workers (same GPU as the parent is unsupported).

Between each GPU / explore / liquid phase, ``place_explore`` runs a CUDA teardown
(``synchronize``, ``empty_cache``, ``ipc_collect``, ``gc``) so the next ``GpuPlacer``
does not inherit a poisoned context. A fully poisoned driver state still needs a fresh process.

Usage:
    uv run evaluate submissions/place_explore.py -b ibm01
"""

from __future__ import annotations

import gc
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.abbacus import AbbacusLegalizer  # noqa: E402
from submissions.explore_multiple import ExploreMultiplePlacer  # noqa: E402
from submissions.explore import finalize_explore_placement  # noqa: E402
from submissions.gradient import _try_load_plc_iccad04  # noqa: E402
from submissions.gpu.placer import GpuPlacer  # noqa: E402

# Fixed explore budget for every inner cycle (not env-overridable).
_PLACE_EXPLORE_EPOCHS = 25
# Liquid GPU pass after each explore (fixed epoch budget, no stagnation stop).
_PLACE_LIQUID_GPU_EPOCHS = 50
_PLACE_LIQUID_LR = 1e-2
_PLACE_LIQUID_W_DENSITY = 0.8
_PLACE_LIQUID_W_OVERLAP = 0.5


def _cuda_reset_between_phases() -> None:
    """Best-effort CUDA cleanup between GPU / explore / liquid phases in one process.

    Async errors from a prior ``GpuPlacer`` backward often surface on the *next* kernel
    (e.g. density ``.to(cuda)``). Explore pool workers also run while the parent holds a
    CUDA context — sync + cache flush + ``gc`` reduces illegal-access crashes on Windows.
    """
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


# Backward-compatible alias for timed_place_explore.
_cuda_sync_empty_cache = _cuda_reset_between_phases


@contextmanager
def _forced_macro_place_device(value: str):
    """Temporarily set ``MACRO_PLACE_DEVICE`` for nested code (restore after)."""
    old = os.environ.get("MACRO_PLACE_DEVICE")
    os.environ["MACRO_PLACE_DEVICE"] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MACRO_PLACE_DEVICE", None)
        else:
            os.environ["MACRO_PLACE_DEVICE"] = old


def _log_row(
    fp,
    *,
    cycle: int,
    phase: str,
    step: str,
    benchmark: Benchmark,
    plc,
    pos: torch.Tensor,
) -> float:
    """Write one CSV row; return proxy_cost."""
    if plc is None:
        px = float("nan")
        fp.write(
            f"{cycle},{phase},{step},,,,,,n/a\n",
        )
        fp.flush()
        return px
    costs = compute_proxy_cost(pos.clone(), benchmark, plc)
    px = float(costs["proxy_cost"])
    fp.write(
        f"{cycle},{phase},{step},"
        f"{px:.10g},"
        f"{float(costs['wirelength_cost']):.10g},"
        f"{float(costs['density_cost']):.10g},"
        f"{float(costs['congestion_cost']):.10g},"
        f"{int(costs['overlap_count'])}\n",
    )
    fp.flush()
    return px


def _print_proxy_phase_delta(phase_label: str, px_before: float, px_after: float) -> None:
    """Log PLC proxy change over a phase (lower proxy is better for ICCAD04 evaluator)."""
    if px_before != px_before or px_after != px_after:
        print(
            f"[place_explore] {phase_label}  proxy_delta=n/a (no PLC / PlacementCost)",
            flush=True,
        )
        return
    delta = px_before - px_after
    denom = abs(px_before)
    rel = (delta / denom * 100.0) if denom > 1e-30 else 0.0
    print(
        f"[place_explore] {phase_label}  proxy {px_before:.6g} -> {px_after:.6g}  "
        f"decline={delta:+.6g}  rel_vs_phase_start={rel:+.3f}%",
        flush=True,
    )


def _inner_gpu_stag_min_abs(cycle_index: int) -> float:
    """PLC stagnation min absolute proxy improvement for inner ``GpuPlacer`` (0-based cycle).

    Default **0.0001** (``0.1/1000``) each cycle; override with env if we add per-cycle tuning later.
    ``cycle_index`` is reserved for future tiering.
    """
    _ = cycle_index
    return 0.0001


class PlaceExplorePlacer:
    """
    Alternating GPU placement + coarse exploration with CSV proxy logging.
    Ends with Abbacus hard-macro legalization; overlap pair count is printed at exit.

    Env:
        MACRO_PLACE_PE_OUTER_ROUNDS — default ``1`` (outer cycles: gpu → explore → liquid gpu)
        Explore phase: fixed **25** epochs per seed, single wave, soft movable macros only.
        Liquid GPU per cycle: fixed **50** epochs; ``w_wl``/``w_cong`` match patience; explore via ``w_density=0.8``, ``w_overlap=0.5`` only.
        MACRO_PLACE_PE_EXPLORE_VERBOSE — default **on** (unset or ``1``): print ``ExploreMultiplePlacer``
            per-wave and summary lines. Set ``0`` / ``false`` / ``off`` for quiet parallel explore.
        MACRO_PLACE_PE_EXPLORE_WORKER_USE_CUDA — if ``1``, explore pool workers do **not**
            hide GPUs via ``CUDA_VISIBLE_DEVICES`` (default: workers hide GPUs when fast
            proxy is not ``cuda``; keeps child processes off the parent's CUDA context).
        MACRO_PLACE_PLC_QUIET — set ``1`` to suppress routine ``PlacementCost`` stdout;
            explore pool workers default this on unless already set in the environment.
        MACRO_PLACE_PE_GPU_MAX_EPOCHS — cap per GPU phase (default ``20000``)
        ``MACRO_PLACE_GPU_OPTIMIZER`` — ``GpuPlacer`` optimizer: default ``adam``; set
            ``nesterov`` for SGD+Nesterov. ``MACRO_PLACE_GPU_MOMENTUM`` applies to Nesterov only (default ``0.9``).
        Inner GPU PLC stagnation min-abs between proxy checks: default **0.0001** (``0.1/1000``).
        ``MACRO_PLACE_PE_STAGNATION_MIN_ABS_IMPROVEMENT`` does **not** apply to inner passes
        (value comes from ``_inner_gpu_stag_min_abs``).
        MACRO_PLACE_PE_FINAL_GPU_STAGNATION_MIN_ABS_IMPROVEMENT — min-abs for the **final**
        GPU only (default ``0.0001``, same scale as inner)
        MACRO_PLACE_PE_STAGNATION_PATIENCE — passed to ``GpuPlacer`` (default ``6``); only
            used if a ``GpuPlacer`` is run with min-abs ``0`` (legacy relative stagnation)
        MACRO_PLACE_PE_STAGNATION_REL_DELTA — passed to ``GpuPlacer`` (default ``0.01``);
            same legacy note as patience
        MACRO_PLACE_PE_LOG_PATH — CSV path (default ``place_explore_proxy.csv``)
        MACRO_PLACE_GPU_PROXY_CHECK_EVERY — how often PLC proxy is evaluated
            during GPU phase (default in GpuPlacer: ``100``)
    """

    def __init__(
        self,
        *,
        outer_rounds: int | None = None,
        explore_seed: int = 0,
        gpu_max_epochs: int | None = None,
        stagnation_patience: int | None = None,
        stagnation_rel_delta: float | None = None,
        log_path: Path | str | None = None,
    ) -> None:
        self.outer_rounds = int(
            outer_rounds
            if outer_rounds is not None
            else (os.environ.get("MACRO_PLACE_PE_OUTER_ROUNDS", "1") or "1")
        )
        self.explore_epochs = _PLACE_EXPLORE_EPOCHS
        self.explore_seed = int(explore_seed)
        self.gpu_max_epochs = int(
            gpu_max_epochs
            if gpu_max_epochs is not None
            else (os.environ.get("MACRO_PLACE_PE_GPU_MAX_EPOCHS", "20000") or "20000")
        )
        self.stagnation_patience = int(
            stagnation_patience
            if stagnation_patience is not None
            else (os.environ.get("MACRO_PLACE_PE_STAGNATION_PATIENCE", "6") or "6")
        )
        self.stagnation_rel_delta = float(
            stagnation_rel_delta
            if stagnation_rel_delta is not None
            else (os.environ.get("MACRO_PLACE_PE_STAGNATION_REL_DELTA", "0.01") or "0.01")
        )
        lp = log_path or os.environ.get("MACRO_PLACE_PE_LOG_PATH", "place_explore_proxy.csv")
        self.log_path = Path(lp)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _try_load_plc_iccad04(benchmark)
        pos = benchmark.macro_positions.clone()

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.log_path.exists() or self.log_path.stat().st_size == 0
        log_fp = open(self.log_path, "a", encoding="utf-8")
        if write_header:
            log_fp.write(
                "cycle,phase,step,proxy_cost,wirelength_cost,density_cost,"
                "congestion_cost,overlap_count\n",
            )
            log_fp.flush()

        print(
            f"[place_explore] outer_rounds={self.outer_rounds} explore_epochs={self.explore_epochs} "
            f"liquid_gpu_epochs={_PLACE_LIQUID_GPU_EPOCHS} liquid_lr={_PLACE_LIQUID_LR:g} "
            f"liquid_w_density={_PLACE_LIQUID_W_DENSITY} "
            f"liquid_w_overlap={_PLACE_LIQUID_W_OVERLAP} "
            f"gpu_max_epochs={self.gpu_max_epochs} inner_gpu_plc_min_abs=0.0001 "
            f"stagnation_patience={self.stagnation_patience} "
            f"stagnation_rel_delta={self.stagnation_rel_delta} log={self.log_path.resolve()}",
            flush=True,
        )

        _log_row(
            log_fp,
            cycle=0,
            phase="init",
            step="start",
            benchmark=benchmark,
            plc=plc,
            pos=pos,
        )

        for cycle in range(self.outer_rounds):
            # --- PLC snapshot at start of cycle ---
            px_before_gpu = _log_row(
                log_fp,
                cycle=cycle + 1,
                phase="cycle_start",
                step="before_gpu",
                benchmark=benchmark,
                plc=plc,
                pos=pos,
            )

            # --- GPU phase (stagnation on PLC proxy) ---
            gpu1 = GpuPlacer(
                epochs=self.gpu_max_epochs,
                stagnation_proxy_patience=self.stagnation_patience,
                stagnation_proxy_rel_delta=self.stagnation_rel_delta,
                stagnation_min_abs_improvement=_inner_gpu_stag_min_abs(cycle),
                seed=self.explore_seed + cycle * 10_000,
            )
            with _forced_macro_place_device("cuda"):
                _cuda_reset_between_phases()
                pos_in = pos.clone()
                pos = gpu1.place(benchmark, initial_macro_positions=pos)
            pos = finalize_explore_placement(pos, benchmark, fallback=pos_in)
            _cuda_reset_between_phases()
            px_gpu1 = _log_row(
                log_fp,
                cycle=cycle + 1,
                phase="gpu",
                step="after_stagnation_or_cap",
                benchmark=benchmark,
                plc=plc,
                pos=pos,
            )
            _print_proxy_phase_delta(
                f"cycle {cycle + 1}/{self.outer_rounds} GPU",
                px_before_gpu,
                px_gpu1,
            )

            # --- Explore phase (parallel seeds, return best fast-proxy replay) ---
            seed_base = self.explore_seed + cycle * 1000 + 7
            explorer = ExploreMultiplePlacer(
                epochs=_PLACE_EXPLORE_EPOCHS,
                max_waves=1,
                seeds=tuple(seed_base + i for i in range(10)),
                grid_side=6,
                moved_macro_weight=0.25,
                fast_proxy_device="cpu",
                soft_macros_only=True,
            )
            pos_in = pos.clone()
            pos = explorer.place(benchmark, initial_macro_positions=pos)
            pos = finalize_explore_placement(pos, benchmark, fallback=pos_in)
            _cuda_reset_between_phases()
            ex_stats = getattr(
                explorer, "last_explore_stats", {"waves": 1, "epoch_steps": _PLACE_EXPLORE_EPOCHS}
            )
            px_ex = _log_row(
                log_fp,
                cycle=cycle + 1,
                phase="explore",
                step=(
                    f"after_w{ex_stats.get('waves', 1)}_"
                    f"ep{ex_stats.get('epoch_steps', _PLACE_EXPLORE_EPOCHS)}"
                ),
                benchmark=benchmark,
                plc=plc,
                pos=pos,
            )
            _print_proxy_phase_delta(
                f"cycle {cycle + 1}/{self.outer_rounds} explore",
                px_gpu1,
                px_ex,
            )

            # --- Liquid GPU (fixed epochs, reduced density weight) ---
            gpu_liquid = GpuPlacer(
                epochs=_PLACE_LIQUID_GPU_EPOCHS,
                lr=_PLACE_LIQUID_LR,
                w_density=_PLACE_LIQUID_W_DENSITY,
                w_overlap=_PLACE_LIQUID_W_OVERLAP,
                stagnation_proxy_patience=0,
                stagnation_min_abs_improvement=0.0,
                seed=self.explore_seed + cycle * 10_000 + 5_000,
            )
            with _forced_macro_place_device("cuda"):
                _cuda_reset_between_phases()
                pos_in = pos.clone()
                pos = gpu_liquid.place(benchmark, initial_macro_positions=pos)
            pos = finalize_explore_placement(pos, benchmark, fallback=pos_in)
            _cuda_reset_between_phases()
            px_liq = _log_row(
                log_fp,
                cycle=cycle + 1,
                phase="liquid_gpu",
                step=f"after_{_PLACE_LIQUID_GPU_EPOCHS}_epochs",
                benchmark=benchmark,
                plc=plc,
                pos=pos,
            )
            _print_proxy_phase_delta(
                f"cycle {cycle + 1}/{self.outer_rounds} liquid GPU",
                px_ex,
                px_liq,
            )

        # --- Final GPU phase after all outer cycles ---
        _raw_final_abs = os.environ.get(
            "MACRO_PLACE_PE_FINAL_GPU_STAGNATION_MIN_ABS_IMPROVEMENT", "0.0001"
        )
        final_gpu_stag_min_abs = float(_raw_final_abs.strip() or "0.0001")

        px_before_final_gpu = _log_row(
            log_fp,
            cycle=self.outer_rounds,
            phase="final_gpu_start",
            step="before_gpu",
            benchmark=benchmark,
            plc=plc,
            pos=pos,
        )
        gpu_final = GpuPlacer(
            epochs=self.gpu_max_epochs,
            stagnation_proxy_patience=self.stagnation_patience,
            stagnation_proxy_rel_delta=self.stagnation_rel_delta,
            stagnation_min_abs_improvement=final_gpu_stag_min_abs,
            seed=self.explore_seed + self.outer_rounds * 10_000 + 1,
        )
        with _forced_macro_place_device("cuda"):
            _cuda_reset_between_phases()
            pos_in = pos.clone()
            pos = gpu_final.place(benchmark, initial_macro_positions=pos)
        pos = finalize_explore_placement(pos, benchmark, fallback=pos_in)
        _cuda_reset_between_phases()
        px_gpu_final = _log_row(
            log_fp,
            cycle=self.outer_rounds,
            phase="final_gpu",
            step="after_stagnation_or_cap",
            benchmark=benchmark,
            plc=plc,
            pos=pos,
        )
        _print_proxy_phase_delta("final GPU", px_before_final_gpu, px_gpu_final)

        px_before_abbacus = _log_row(
            log_fp,
            cycle=self.outer_rounds,
            phase="abbacus",
            step="before",
            benchmark=benchmark,
            plc=plc,
            pos=pos,
        )
        pos = AbbacusLegalizer().place(benchmark, initial_macro_positions=pos)
        px_abb = _log_row(
            log_fp,
            cycle=self.outer_rounds,
            phase="abbacus",
            step="after",
            benchmark=benchmark,
            plc=plc,
            pos=pos,
        )
        _print_proxy_phase_delta("Abbacus legalization", px_before_abbacus, px_abb)

        # Closing CSV row (final placement, post-Abbacus).
        _log_row(
            log_fp,
            cycle=self.outer_rounds,
            phase="final",
            step="end",
            benchmark=benchmark,
            plc=plc,
            pos=pos,
        )

        ov = compute_overlap_metrics(pos, benchmark)
        n_pairs = int(ov["overlap_count"])
        print(
            f"[place_explore] overlapping hard-macro pairs: {n_pairs}",
            flush=True,
        )

        log_fp.close()

        return pos


if __name__ == "__main__":
    from macro_place.loader import load_benchmark_from_dir

    root = _ROOT
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    PlaceExplorePlacer().place(b)
