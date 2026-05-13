"""
Parallel ensemble ExplorePlacer runs; return the **best** solo result.

Runs ``ExplorePlacer`` (CPU fast proxy path from ``explore.py``; **all non-fixed macros**)
with **distinct seeds** in parallel processes. Each run records accepted moves
``(macro_index, cx, cy)``. Move lists are **replayed** on that worker's **initial**
layout for the wave.

**Multi-wave elitism** (when ``len(seeds) >= 8``): each **wave** runs ``wave_epochs``
(default **100**, same as ``epochs`` unless env ``MACRO_PLACE_PE_EXPLORE_WAVE_EPOCHS``
is set) per seed. Let ``start_best = min_i FastProxy(init_i)`` (pool best at wave
start) and ``end_best = min_i FastProxy(final_i)`` (pool best at wave end). Within-wave
gain is ``delta = start_best - end_best`` (lower fast proxy is better); verbose
logging may print ``delta`` and ``initial_fast_proxy`` (at explore entry) for
diagnostics. Waves run until ``max_waves`` (default **8**, env
``MACRO_PLACE_PE_EXPLORE_MAX_WAVES``) — there is **no** early stop on small ``delta``.
Between waves, the four worst finals are replaced by copies of the four best finals;
other slots carry their own final forward. RNG seeds are offset per wave. If fewer
than **8** seeds, only a **single** wave runs (no elitism).

Explore workers set ``MACRO_PLACE_PLC_QUIET=1`` (unless already set) and run
``ExplorePlacer.place(..., quiet=True)`` so parallel explore does not flood stdout
with PlacementCost init lines or per-seed summaries. Unless
``MACRO_PLACE_PE_EXPLORE_WORKER_USE_CUDA=1`` or fast proxy is ``cuda``, each worker
sets ``CUDA_VISIBLE_DEVICES=`` (empty) before loading the benchmark so child processes
do not initialize CUDA on the same device as a parent ``GpuPlacer`` (reduces
``cudaErrorUnknown`` / poisoned contexts on Windows after multiprocessing). Set env
``MACRO_PLACE_PE_EXPLORE_VERBOSE=0`` to silence per-wave lines and the final ensemble
summary from this placer (default **on**).

**Selection:** minimize ``FastProxyEvaluator.total`` over all waves (tie-break lower
base seed).

After ``place()``, read ``last_explore_stats`` for ``waves`` and ``epoch_steps`` (sum
of ``wave_epochs`` over waves run).

Usage:
    uv run python submissions/explore_multiple.py
"""

from __future__ import annotations

import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.fast_proxy import FastProxyEvaluator
from macro_place.loader import load_benchmark_from_dir

from submissions.explore import ExplorePlacer, _repo_root, finalize_explore_placement

_WAVE_SEED_STEP = 1_000_000


def _iccad04_case_dir(benchmark: Benchmark) -> Path:
    root = _repo_root()
    case_dir = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / benchmark.name
    if not (case_dir / "netlist.pb.txt").is_file():
        raise FileNotFoundError(
            f"Missing ICCAD04 testcase dir for benchmark {benchmark.name}: {case_dir}"
        )
    return case_dir


def _tensor_to_bytes(t: torch.Tensor) -> bytes | None:
    buf = io.BytesIO()
    torch.save(t.detach().cpu(), buf)
    return buf.getvalue()


def _explore_worker(payload: tuple) -> tuple[int, list[tuple[int, float, float]]]:
    """Picklable worker: load benchmark, run ExplorePlacer, return seed and move log."""
    (
        case_dir_str,
        worker_seed,
        grid_side,
        epochs,
        moved_macro_weight,
        fast_proxy_device_str,
        initial_bytes,
        hard_macros_only,
        soft_macros_only,
    ) = payload

    # Keep spawn/forkserver workers off the parent's GPU. Parallel torch + CUDA in
    # child processes while GpuPlacer holds the primary context correlates with
    # cudaErrorUnknown on the *next* parent kernel (often reported at an innocent .to()).
    _allow_cuda = os.environ.get("MACRO_PLACE_PE_EXPLORE_WORKER_USE_CUDA", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not _allow_cuda and fast_proxy_device_str.strip().lower() != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # PlacementCost prints once per worker; suppress routine chatter in explore pool.
    os.environ.setdefault("MACRO_PLACE_PLC_QUIET", "1")

    b, _ = load_benchmark_from_dir(case_dir_str)
    initial = None
    if initial_bytes is not None:
        initial = torch.load(io.BytesIO(initial_bytes), map_location="cpu")

    fpd = None if fast_proxy_device_str == "" else fast_proxy_device_str

    placer = ExplorePlacer(
        grid_side=grid_side,
        epochs=epochs,
        seed=int(worker_seed),
        moved_macro_weight=moved_macro_weight,
        fast_proxy_device=fpd,
        hard_macros_only=bool(hard_macros_only),
        soft_macros_only=bool(soft_macros_only),
    )
    log: list[tuple[int, float, float]] = []
    placer.place(
        b,
        initial_macro_positions=initial,
        save_figure=False,
        accepted_moves=log,
        quiet=True,
    )
    return int(worker_seed), log


class ExploreMultiplePlacer:
    """
    Parallel ``ExplorePlacer`` instances (one per seed); return the best replayed solo.

    Args mirror ``ExplorePlacer`` plus ``seeds`` (default ``0..9``) and
    ``grid_side`` (default **6**). Optional multi-wave controls (see module doc).
    """

    def __init__(
        self,
        *,
        seeds: Sequence[int] | None = None,
        grid_side: int = 6,
        epochs: int = 100,
        wave_epochs: int | None = None,
        max_waves: int | None = None,
        moved_macro_weight: float = 0.25,
        fast_proxy_device: torch.device | str | None = None,
        hard_macros_only: bool = False,
        soft_macros_only: bool = False,
    ):
        self.seeds = tuple(seeds) if seeds is not None else tuple(range(10))
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        self.grid_side = int(grid_side)
        self.epochs = int(epochs)
        _we_env = (os.environ.get("MACRO_PLACE_PE_EXPLORE_WAVE_EPOCHS") or "").strip()
        if wave_epochs is not None:
            self.wave_epochs = int(wave_epochs)
        elif _we_env:
            self.wave_epochs = int(_we_env)
        else:
            self.wave_epochs = self.epochs
        self.max_waves = max(
            1,
            int(
                max_waves
                if max_waves is not None
                else int(os.environ.get("MACRO_PLACE_PE_EXPLORE_MAX_WAVES", "8") or "8")
            ),
        )
        self.moved_macro_weight = float(moved_macro_weight)
        self._fast_proxy_device: torch.device | str | None = fast_proxy_device
        self.hard_macros_only = bool(hard_macros_only)
        self.soft_macros_only = bool(soft_macros_only)
        if self.hard_macros_only and self.soft_macros_only:
            raise ValueError("hard_macros_only and soft_macros_only are mutually exclusive")
        self.last_explore_stats: dict[str, int] = {"waves": 0, "epoch_steps": 0}

    def place(
        self,
        benchmark: Benchmark,
        *,
        initial_macro_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        case_dir = _iccad04_case_dir(benchmark)
        case_dir_str = str(case_dir)

        dev = benchmark.macro_positions.device
        dtype = benchmark.macro_positions.dtype
        if initial_macro_positions is not None:
            solo_base = initial_macro_positions.to(device=dev, dtype=dtype).clone()
            entry_fallback = solo_base.clone()
        else:
            solo_base = benchmark.macro_positions.clone()
            entry_fallback = solo_base.clone()

        solo_base = finalize_explore_placement(solo_base, benchmark, fallback=entry_fallback)

        fpd_str = "" if self._fast_proxy_device is None else str(self._fast_proxy_device)

        if self._fast_proxy_device is not None:
            score = FastProxyEvaluator(benchmark, device=torch.device(self._fast_proxy_device))
        else:
            score = FastProxyEvaluator(benchmark)

        n = len(self.seeds)
        n_waves_cap = 1 if n < 8 else int(self.max_waves)

        inits = [solo_base.clone() for _ in range(n)]

        initial_proxy = float(score.total(solo_base))

        global_best_c = float("inf")
        global_best_pl = solo_base.clone()
        global_best_init = solo_base.clone()
        global_best_base_seed = int(self.seeds[0])
        total_moves = 0
        waves_run = 0
        epochs_explore_total = 0
        _raw_ev = (os.environ.get("MACRO_PLACE_PE_EXPLORE_VERBOSE") or "1").strip().lower()
        explore_verbose = _raw_ev not in ("0", "false", "no", "off")

        for wave in range(n_waves_cap):
            start_costs = [float(score.total(t)) for t in inits]
            start_best = min(start_costs)

            payloads = []
            for slot in range(n):
                ib = _tensor_to_bytes(inits[slot])
                worker_seed = int(self.seeds[slot]) + wave * _WAVE_SEED_STEP
                payloads.append(
                    (
                        case_dir_str,
                        worker_seed,
                        self.grid_side,
                        self.wave_epochs,
                        self.moved_macro_weight,
                        fpd_str,
                        ib,
                        self.hard_macros_only,
                        self.soft_macros_only,
                    )
                )

            max_workers = n
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                results: list[tuple[int, list[tuple[int, float, float]]]] = list(
                    ex.map(_explore_worker, payloads)
                )

            finals: list[torch.Tensor] = []
            end_costs: list[float] = []
            for slot, (ret_seed, move_log) in enumerate(results):
                expected_seed = int(self.seeds[slot]) + wave * _WAVE_SEED_STEP
                if ret_seed != expected_seed:
                    raise RuntimeError(
                        f"worker seed mismatch: slot {slot} expected {expected_seed}, got {ret_seed}"
                    )
                solo = inits[slot].clone()
                for i_macro, cx, cy in move_log:
                    solo[int(i_macro), 0] = float(cx)
                    solo[int(i_macro), 1] = float(cy)
                solo = finalize_explore_placement(solo, benchmark, fallback=inits[slot])
                finals.append(solo.clone())
                gc = float(score.total(solo))
                end_costs.append(gc)
                total_moves += len(move_log)
                if gc < global_best_c:
                    global_best_c = gc
                    global_best_pl = solo.clone()
                    global_best_init = inits[slot].clone()
                    global_best_base_seed = int(self.seeds[slot])

            end_best = min(end_costs)
            delta = start_best - end_best
            if explore_verbose:
                print(
                    f"ExploreMultiplePlacer: wave {wave + 1}/{n_waves_cap} "
                    f"start_best={start_best:.6f} end_best={end_best:.6f} "
                    f"delta={delta:.6f} initial_fast_proxy={initial_proxy:.6f}",
                    flush=True,
                )

            waves_run += 1
            epochs_explore_total += self.wave_epochs

            if wave >= n_waves_cap - 1:
                break

            ranked = sorted(range(n), key=lambda i: (end_costs[i], i))
            best_four = ranked[:4]
            worst_four = ranked[-4:]
            next_inits: list[torch.Tensor | None] = [None] * n
            for k in range(4):
                next_inits[worst_four[k]] = finals[best_four[k]].to(device=dev, dtype=dtype).clone()
            for i in range(n):
                if next_inits[i] is None:
                    next_inits[i] = finals[i].to(device=dev, dtype=dtype).clone()
            inits = next_inits

        self.last_explore_stats = {"waves": int(waves_run), "epoch_steps": int(epochs_explore_total)}

        if explore_verbose:
            print(
                f"ExploreMultiplePlacer: {n} seeds, {waves_run} wave(s), {total_moves} accepted moves; "
                f"winner base_seed {global_best_base_seed} (fast_proxy={global_best_c:.6f}, lowest wins).",
                flush=True,
            )
        global_best_pl = finalize_explore_placement(
            global_best_pl, benchmark, fallback=global_best_init
        )
        return global_best_pl


def _cli_main() -> None:
    root = _repo_root()
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    p = ExploreMultiplePlacer(epochs=50, max_waves=2, wave_epochs=5).place(b)
    print(f"placement shape={tuple(p.shape)} device={p.device}")


if __name__ == "__main__":
    _cli_main()
