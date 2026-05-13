"""
**Timed** hybrid placer: same pipeline as ``place_explore.py`` (**GpuPlacer** →
**ExploreMultiple** → … → final **GpuPlacer** → **Abbacus**), plus optional
**wall-clock budgeting**.

After **cycle 1**, estimates seconds per GPU epoch (calibrated on the **first** inner
GPU phase) and per explore epoch. **Cycles 2+** cap ``GpuPlacer`` epochs using remaining
budget. Reserved time for the **final** GPU uses **max wall-clock** of completed inner
GPU phases (times a slack factor), **not** ``gpu_max_epochs × sec/epoch`` (which was
wildly pessimistic).

**Baseline / backup:** use ``submissions/place_explore.py`` (no time logic).

Env (timed-only; ``MACRO_PLACE_PE_*`` unchanged from ``place_explore``):

    MACRO_PLACE_TPE_TIME_BUDGET_SEC — total wall budget in seconds (default ``3300``).
        Set to ``0`` to disable adaptive caps (always ``gpu_max_epochs`` per GPU phase).
    MACRO_PLACE_TPE_TAIL_RESERVE_SEC — reserved near the end for Abbacus + slack
        before the **final** GPU epoch cap (default ``150``).
    MACRO_PLACE_TPE_FINAL_WALL_SLACK — multiply ``max(inner_GPU_wall_times)`` when
        reserving for the final GPU phase (default ``1.15``).
    MACRO_PLACE_TPE_LOG_PATH — CSV path (default ``timed_place_explore_proxy.csv``).

Usage:
    uv run evaluate submissions/timed_place_explore.py -b ibm01
"""

from __future__ import annotations

import math
import os
import sys
import time
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
from submissions.gradient import _try_load_plc_iccad04  # noqa: E402
from submissions.gpu.placer import GpuPlacer  # noqa: E402
from submissions.place_explore import _cuda_sync_empty_cache, _inner_gpu_stag_min_abs  # noqa: E402

_DEFAULT_TPE_BUDGET_SEC = 3300.0
_DEFAULT_TPE_TAIL_RESERVE_SEC = 150.0
_DEFAULT_TPE_FINAL_WALL_SLACK = 1.15


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


def _tpe_budget_sec() -> float | None:
    raw = os.environ.get("MACRO_PLACE_TPE_TIME_BUDGET_SEC", str(int(_DEFAULT_TPE_BUDGET_SEC)))
    s = (raw or "").strip()
    if not s or s == "0":
        return None
    v = float(s)
    return v if v > 0.0 else None


def _tpe_tail_reserve_sec() -> float:
    raw = os.environ.get(
        "MACRO_PLACE_TPE_TAIL_RESERVE_SEC", str(int(_DEFAULT_TPE_TAIL_RESERVE_SEC))
    )
    try:
        return max(0.0, float((raw or "0").strip()))
    except ValueError:
        return _DEFAULT_TPE_TAIL_RESERVE_SEC


def _tpe_final_wall_slack() -> float:
    raw = os.environ.get(
        "MACRO_PLACE_TPE_FINAL_WALL_SLACK", str(_DEFAULT_TPE_FINAL_WALL_SLACK)
    )
    try:
        return max(1.0, float((raw or "1").strip()))
    except ValueError:
        return _DEFAULT_TPE_FINAL_WALL_SLACK


def _estimate_reserved_for_final_gpu_wall(
    inner_gpu_walls_sec: list[float],
    *,
    slack: float,
    fallback_sec: float = 300.0,
) -> float:
    """Reserve wall time for the upcoming **final** GPU pass (not epoch-count × worst case)."""
    if not inner_gpu_walls_sec:
        return fallback_sec
    return float(max(inner_gpu_walls_sec)) * slack


def _gpu_epochs_cap(
    *,
    remaining: float,
    avg_gpu: float,
    avg_explore: float,
    explore_epochs: int,
    gpu_cap: int,
) -> int:
    """Epochs for next GPU phase (inner cycle): reserve one explore block after this GPU."""
    num = remaining - float(explore_epochs) * avg_explore
    denom = max(avg_gpu, 1e-9)
    x = math.floor(num / denom)
    return max(1, min(int(x), int(gpu_cap)))


def _gpu_epochs_final(
    *,
    remaining: float,
    avg_gpu: float,
    gpu_cap: int,
) -> int:
    x = math.floor(remaining / max(avg_gpu, 1e-9))
    return max(1, min(int(x), int(gpu_cap)))


class TimedPlaceExplorePlacer:
    """
    Same as ``PlaceExplorePlacer`` plus optional wall-clock budget (``MACRO_PLACE_TPE_*``).

    Final-GPU reservation uses ``MACRO_PLACE_TPE_TAIL_RESERVE_SEC`` plus
    ``MACRO_PLACE_TPE_FINAL_WALL_SLACK`` × max(inner GPU phase wall times).

    Other env vars match ``place_explore`` (``MACRO_PLACE_PE_*``, ``MACRO_PLACE_GPU_*``),
    including PLC stagnation: inner passes use ``_inner_gpu_stag_min_abs`` (default **1e-4**);
    ``MACRO_PLACE_GPU_STAGNATION_MIN_ABS_IMPROVEMENT`` / ``MACRO_PLACE_PE_STAGNATION_MIN_ABS_IMPROVEMENT``
    apply as in ``GpuPlacer`` when used; set min-abs to ``0`` to use patience / relative mode.
    The **final** GPU pass uses ``MACRO_PLACE_PE_FINAL_GPU_STAGNATION_MIN_ABS_IMPROVEMENT``
    (default ``0.0001``) like ``place_explore``.
    """

    def __init__(
        self,
        *,
        outer_rounds: int | None = None,
        explore_epochs: int | None = None,
        explore_seed: int = 0,
        gpu_max_epochs: int | None = None,
        stagnation_patience: int | None = None,
        stagnation_rel_delta: float | None = None,
        log_path: Path | str | None = None,
    ) -> None:
        self.outer_rounds = int(
            outer_rounds
            if outer_rounds is not None
            else (os.environ.get("MACRO_PLACE_PE_OUTER_ROUNDS", "8") or "8")
        )
        self.explore_epochs = int(
            explore_epochs
            if explore_epochs is not None
            else (os.environ.get("MACRO_PLACE_PE_EXPLORE_EPOCHS", "100") or "100")
        )
        _mw = max(1, int(os.environ.get("MACRO_PLACE_PE_EXPLORE_MAX_WAVES", "8") or "8"))
        _we_env = (os.environ.get("MACRO_PLACE_PE_EXPLORE_WAVE_EPOCHS") or "").strip()
        _we = int(_we_env) if _we_env else self.explore_epochs
        self.explore_epoch_wall_budget = int(_mw * max(1, _we))
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
        lp = log_path or os.environ.get(
            "MACRO_PLACE_TPE_LOG_PATH", "timed_place_explore_proxy.csv"
        )
        self.log_path = Path(lp)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        budget = _tpe_budget_sec()
        tail_reserve = _tpe_tail_reserve_sec()
        final_wall_slack = _tpe_final_wall_slack()
        t0 = time.perf_counter()

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

        adaptive_on = budget is not None
        print(
            f"[timed_place_explore] outer_rounds={self.outer_rounds} explore_epochs={self.explore_epochs} "
            f"gpu_max_epochs={self.gpu_max_epochs} stagnation_patience={self.stagnation_patience} "
            f"stagnation_rel_delta={self.stagnation_rel_delta} "
            f"time_budget_sec={budget if adaptive_on else 'off'} tail_reserve_sec={tail_reserve} "
            f"final_wall_slack={final_wall_slack} "
            f"log={self.log_path.resolve()}",
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

        avg_gpu = 1.0
        avg_explore = 1.0
        avg_gpu_calib: float | None = None
        gpu_inner_wall_seconds: list[float] = []

        for cycle in range(self.outer_rounds):
            _log_row(
                log_fp,
                cycle=cycle + 1,
                phase="cycle_start",
                step="before_gpu",
                benchmark=benchmark,
                plc=plc,
                pos=pos,
            )

            if adaptive_on and cycle >= 1:
                elapsed = time.perf_counter() - t0
                est_final_wall = _estimate_reserved_for_final_gpu_wall(
                    gpu_inner_wall_seconds,
                    slack=final_wall_slack,
                )
                reserve_final = tail_reserve + est_final_wall
                remaining = budget - elapsed - reserve_final
                spe_sched = (
                    avg_gpu_calib if avg_gpu_calib is not None else max(avg_gpu, 1e-9)
                )
                gpu_epochs_this = _gpu_epochs_cap(
                    remaining=remaining,
                    avg_gpu=spe_sched,
                    avg_explore=avg_explore,
                    explore_epochs=self.explore_epoch_wall_budget,
                    gpu_cap=self.gpu_max_epochs,
                )
                print(
                    f"[timed_place_explore] adaptive cycle={cycle + 1}/{self.outer_rounds} "
                    f"budget={budget:.1f}s elapsed={elapsed:.1f}s "
                    f"est_final_gpu_wall={est_final_wall:.1f}s reserve_total={reserve_final:.1f}s "
                    f"(tail={tail_reserve:.1f}s) remaining_for_phase={remaining:.1f}s "
                    f"gpu_sec_per_ep_sched={spe_sched:.4g}s/ep avg_explore={avg_explore:.4g}s/ep "
                    f"gpu_epochs={gpu_epochs_this}",
                    flush=True,
                )
            else:
                gpu_epochs_this = self.gpu_max_epochs
                if adaptive_on and cycle == 0:
                    print(
                        "[timed_place_explore] adaptive cycle=1: calibrating avg_gpu / avg_explore "
                        f"(full gpu_max_epochs={self.gpu_max_epochs})",
                        flush=True,
                    )

            telem: dict = {}
            gpu1 = GpuPlacer(
                epochs=gpu_epochs_this,
                stagnation_proxy_patience=self.stagnation_patience,
                stagnation_proxy_rel_delta=self.stagnation_rel_delta,
                stagnation_min_abs_improvement=_inner_gpu_stag_min_abs(cycle),
                seed=self.explore_seed + cycle * 10_000,
            )
            with _forced_macro_place_device("cuda"):
                pos = gpu1.place(benchmark, initial_macro_positions=pos, telemetry=telem)
            ep_done = max(int(telem.get("epochs_completed", 0)), 1)
            wall_gpu = float(telem.get("wall_seconds", 0.0))
            gpu_inner_wall_seconds.append(wall_gpu)
            spe = wall_gpu / ep_done
            if avg_gpu_calib is None:
                avg_gpu_calib = spe
            avg_gpu = spe

            px_gpu1 = _log_row(
                log_fp,
                cycle=cycle + 1,
                phase="gpu",
                step="after_stagnation_or_cap",
                benchmark=benchmark,
                plc=plc,
                pos=pos,
            )
            print(
                f"[timed_place_explore] cycle {cycle + 1}/{self.outer_rounds} GPU phase done "
                f"proxy={px_gpu1:.6g} epochs_done={ep_done} wall_gpu={wall_gpu:.2f}s",
                flush=True,
            )

            t_ex0 = time.perf_counter()
            seed_base = self.explore_seed + cycle * 1000 + 7
            explorer = ExploreMultiplePlacer(
                epochs=self.explore_epochs,
                seeds=tuple(seed_base + i for i in range(10)),
                grid_side=6,
                moved_macro_weight=0.25,
                fast_proxy_device="cpu",
            )
            pos = explorer.place(benchmark, initial_macro_positions=pos)
            _cuda_sync_empty_cache()
            wall_ex = time.perf_counter() - t_ex0
            ex_stats = getattr(explorer, "last_explore_stats", {"waves": 1, "epoch_steps": self.explore_epochs})
            avg_explore = wall_ex / max(int(ex_stats.get("epoch_steps", self.explore_epochs)), 1)

            px_ex = _log_row(
                log_fp,
                cycle=cycle + 1,
                phase="explore",
                step=(
                    f"after_w{ex_stats.get('waves', 1)}_"
                    f"ep{ex_stats.get('epoch_steps', self.explore_epochs)}"
                ),
                benchmark=benchmark,
                plc=plc,
                pos=pos,
            )
            print(
                f"[timed_place_explore] cycle {cycle + 1}/{self.outer_rounds} explore done "
                f"proxy={px_ex:.6g} wall_explore={wall_ex:.2f}s",
                flush=True,
            )

        _log_row(
            log_fp,
            cycle=self.outer_rounds,
            phase="final_gpu_start",
            step="before_gpu",
            benchmark=benchmark,
            plc=plc,
            pos=pos,
        )

        if adaptive_on:
            elapsed = time.perf_counter() - t0
            remaining = budget - elapsed - tail_reserve
            spe_final = avg_gpu_calib if avg_gpu_calib is not None else max(avg_gpu, 1e-9)
            gpu_final_epochs = _gpu_epochs_final(
                remaining=remaining,
                avg_gpu=spe_final,
                gpu_cap=self.gpu_max_epochs,
            )
            print(
                f"[timed_place_explore] adaptive final_gpu budget={budget:.1f}s elapsed={elapsed:.1f}s "
                f"tail_reserve={tail_reserve:.1f}s remaining={remaining:.1f}s "
                f"gpu_sec_per_ep_sched={spe_final:.4g}s/ep gpu_epochs={gpu_final_epochs}",
                flush=True,
            )
        else:
            gpu_final_epochs = self.gpu_max_epochs

        _raw_final_abs = os.environ.get(
            "MACRO_PLACE_PE_FINAL_GPU_STAGNATION_MIN_ABS_IMPROVEMENT", "0.0001"
        )
        final_gpu_stag_min_abs = float(_raw_final_abs.strip() or "0.0001")

        telem_f: dict = {}
        gpu_final = GpuPlacer(
            epochs=gpu_final_epochs,
            stagnation_proxy_patience=self.stagnation_patience,
            stagnation_proxy_rel_delta=self.stagnation_rel_delta,
            stagnation_min_abs_improvement=final_gpu_stag_min_abs,
            seed=self.explore_seed + self.outer_rounds * 10_000 + 1,
        )
        with _forced_macro_place_device("cuda"):
            pos = gpu_final.place(benchmark, initial_macro_positions=pos, telemetry=telem_f)
        px_gpu_final = _log_row(
            log_fp,
            cycle=self.outer_rounds,
            phase="final_gpu",
            step="after_stagnation_or_cap",
            benchmark=benchmark,
            plc=plc,
            pos=pos,
        )
        print(
            f"[timed_place_explore] final GPU phase done proxy={px_gpu_final:.6g} "
            f"epochs_done={telem_f.get('epochs_completed', 0)}",
            flush=True,
        )

        _log_row(
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
        print(
            f"[timed_place_explore] Abbacus legalization done proxy={px_abb:.6g}",
            flush=True,
        )

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
        total_s = time.perf_counter() - t0
        print(
            f"[timed_place_explore] overlapping hard-macro pairs: {n_pairs} "
            f"wall_total={total_s:.1f}s",
            flush=True,
        )

        log_fp.close()

        return pos


if __name__ == "__main__":
    from macro_place.loader import load_benchmark_from_dir

    root = _ROOT
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    TimedPlaceExplorePlacer(explore_epochs=100).place(b)
