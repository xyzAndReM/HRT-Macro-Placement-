"""
Liquid: alternating **liquid** and **patience** ``GpuPlacer`` passes, then **Abbacus**.

Schedule (default **1** cycle, env ``MACRO_PLACE_LIQUID_CYCLES``):

  Before each cycle, record PLC **proxy_cost** and placement snapshot. Then **liquid → patience**:

    1. **Liquid GPU** — surrogate only (L-route congestion, no PLC proxy checks, no spatial/hotspot):
       every **50** epochs, loss must improve by at least **0.001 ×** surrogate total at phase start;
       **1** consecutive sub-threshold check stops liquid (epoch cap **20000**). ``lr=1e-2``;
       ``w_density`` / ``w_overlap`` ``0.8`` / ``0.5``.
    2. **Patience GPU** — Adam, L-route congestion surrogate + **PLC proxy** every **50** epochs,
       scale-only ``w·surrogate`` calibration (WL/density/congestion), min-abs stagnation **0.0001** per
       proxy check vs session **best**;
       **1** consecutive sub-threshold check stops patience (or SGD sub-phase after switch).
       After **patience - 1** sub-threshold checks, switches to plain SGD
       (``late_stagnation_sgd_switch``) for late refinement.
       Epoch cap **20000** (or env).

  After both phases, re-evaluate proxy. If it **increased** (worse; lower is better), restore the
  pre-cycle placement and **stop** further cycles. Otherwise continue until the cycle cap.

  Then **Abbacus** hard-macro legalization.

  Optional: set ``MACRO_PLACE_LIQUID_QP_IF_OVERLAPS=1`` to run **QPLegalizer**
  (``submissions/qp.py``, cvxpy/OSQP) only when hard-macro overlaps remain after Abbacus.

Env:
    MACRO_PLACE_LIQUID_CYCLES — max liquid→patience pairs (default ``1``)
    MACRO_PLACE_LIQUID_GPU_EPOCHS — liquid phase **max** epochs (default ``20000``); set ``0`` to skip.
    MACRO_PLACE_LIQUID_SURROGATE_MIN_REL_INITIAL — min improvement as fraction of initial surrogate
       loss at liquid phase start (default ``0.001``)
    MACRO_PLACE_LIQUID_SURROGATE_STAG_PATIENCE — consecutive sub-threshold surrogate checks (default ``1``)
    MACRO_PLACE_LIQUID_SURROGATE_CHECK_EVERY — liquid surrogate check interval (default ``50``)
    MACRO_PLACE_LIQUID_LR — liquid Adam/SGD learning rate (default ``1e-2``)
    MACRO_PLACE_LIQUID_W_DENSITY — liquid ``w_density`` only (default ``0.8``)
    MACRO_PLACE_LIQUID_PATIENCE_W_DENSITY — patience-phase ``GpuPlacer.w_density`` (default ``0.5``)
    MACRO_PLACE_LIQUID_W_OVERLAP — liquid ``w_overlap`` only (default ``0.5``)
    MACRO_PLACE_LIQUID_PATIENCE_EPOCHS — patience phase epoch cap (default ``20000``)
    MACRO_PLACE_LIQUID_STAGNATION_MIN_ABS — min PLC proxy improvement per proxy check (default ``0.0001``)
    MACRO_PLACE_LIQUID_STAGNATION_PATIENCE — consecutive sub-threshold checks before stop (default ``1``)
    MACRO_PLACE_LIQUID_PROXY_CHECK_EVERY — patience proxy check / log interval (default ``50``)
    MACRO_PLACE_LIQUID_QP_IF_OVERLAPS — if ``1``/``true``, run ``QPLegalizer`` after Abbacus when
       ``compute_overlap_metrics`` reports overlapping hard-macro pairs (default: off)
    MACRO_PLACE_LIQUID_RANDOM_HARD_START — if ``1``/``true``, uniform random centers for movable
       hard macros only (soft macros and fixed macros stay at loader positions)
    MACRO_PLACE_LIQUID_RANDOM_HARD_SEED — optional RNG seed for random hard start
    MACRO_PLACE_LIQUID_SAVE_PLACEMENT — optional: save post-Abbacus placement to ``1`` →
       ``logs/liquid_<benchmark>_placement.pt``, or a file path.
    Liquid **defaults** (no manual env needed): ``MACRO_PLACE_GPU_PLC_NET_ROUTING=0``,
    ``MACRO_PLACE_GPU_SPATIAL_CONG=0`` for the whole placement run (L-route congestion only).

    Other ``MACRO_PLACE_GPU_*`` settings apply when not overridden here.

    If ``MACRO_PLACE_GPU_LOG_PATH`` is unset, proxy-check rows go to
    ``logs/liquid_<benchmark>_proxycheck.csv`` and a short congestion report is written to
    ``logs/liquid_<benchmark>_congestion.md`` after placement (from that CSV).

Unless ``MACRO_PLACE_DEVICE`` is unset, GPU phases prefer ``cuda``.

Usage:
    uv run evaluate submissions/liquid.py -b ibm01
"""

from __future__ import annotations

import csv
import gc
import math
import os
import sys
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.abbacus import AbbacusLegalizer  # noqa: E402
from submissions.gradient import (  # noqa: E402
    _try_load_plc_iccad04,
    randomize_movable_hard_macro_centers,
)
from submissions.gpu.placer import GpuPlacer  # noqa: E402
from submissions.qp import QPLegalizer  # noqa: E402

_DEFAULT_CYCLES = 1
_DEFAULT_LIQUID_EPOCH_CAP = 20000
_DEFAULT_LIQUID_SURROGATE_MIN_REL_INITIAL = 0.001
_DEFAULT_LIQUID_SURR_STAG_PATIENCE = 1
_DEFAULT_LIQUID_SURROGATE_CHECK_EVERY = 50
_DEFAULT_LIQUID_LR = 1e-2
_DEFAULT_LIQUID_W_DENSITY = 0.8
# Matches GpuPlacer default when patience did not pass w_density explicitly.
_DEFAULT_PATIENCE_W_DENSITY = 0.5
_DEFAULT_LIQUID_W_OVERLAP = 0.5
# Patience-aligned surrogate coeffs (GpuPlacer defaults); liquid does not override these.
_PATIENCE_W_WL = 1.0
_PATIENCE_W_CONG = 0.5
_DEFAULT_PATIENCE_EPOCH_CAP = 20000
_DEFAULT_STAG_MIN_ABS = 0.0001
_DEFAULT_STAG_PATIENCE = 1
_DEFAULT_PROXY_CHECK_EVERY = 50
_PLACEMENT_CKPT_VERSION = 1

# Liquid always uses L-route congestion (not PLC net grids / spatial hotspot).
_LIQUID_GPU_PLC_NET_ROUTING = "0"
_LIQUID_GPU_SPATIAL_CONG = "0"
_LIQUID_GPU_CONG_ENV_KEYS = (
    "MACRO_PLACE_GPU_PLC_NET_ROUTING",
    "MACRO_PLACE_GPU_SPATIAL_CONG",
)


def _liquid_verbose() -> bool:
    """Default quiet; set ``MACRO_PLACE_LIQUID_VERBOSE=1`` for phase/cycle console logs."""
    return _liquid_env_bool("MACRO_PLACE_LIQUID_VERBOSE", default=False)


def _liquid_eval_quiet() -> bool:
    return os.environ.get("MACRO_PLACE_EVAL_QUIET", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _liquid_log(msg: str) -> None:
    if _liquid_verbose():
        print(msg, flush=True)


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    return max(1, int(raw))


def _liquid_epoch_cap_env(default: int) -> int:
    """Explore phase epoch cap; ``0`` means skip the liquid phase entirely."""
    raw = (os.environ.get("MACRO_PLACE_LIQUID_GPU_EPOCHS") or "").strip()
    if not raw:
        return default
    return max(0, int(raw))


def _float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _liquid_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _bench_slug(name: str) -> str:
    s = "".join(c if (c.isalnum() or c in "_-") else "_" for c in (name or "").strip())
    return s or "bench"


def _resolve_liquid_save_placement_path(benchmark_name: str) -> Path | None:
    """Path to write checkpoint, or None if saving disabled."""
    raw = (os.environ.get("MACRO_PLACE_LIQUID_SAVE_PLACEMENT") or "").strip()
    if not raw:
        return None
    if raw.lower() in ("1", "true", "yes", "on"):
        slug = _bench_slug(benchmark_name)
        return (_ROOT / "logs" / f"liquid_{slug}_placement.pt").resolve()
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _save_liquid_placement_checkpoint(
    path: Path,
    benchmark: Benchmark,
    placement: torch.Tensor,
    *,
    stage: str,
    plc_proxy: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": _PLACEMENT_CKPT_VERSION,
        "benchmark": benchmark.name,
        "placement": placement.detach().cpu(),
        "stage": stage,
        "placer": "liquid",
    }
    if plc_proxy is not None and math.isfinite(plc_proxy):
        payload["proxy_cost"] = float(plc_proxy)
    torch.save(payload, path)
    print(f"[liquid] saved placement ({stage}) -> {path}", flush=True)


def _setup_proxycheck_log_path(benchmark_name: str) -> tuple[Path, bool]:
    """
    Resolve CSV path for ``GpuPlacer`` proxy-check logging.

    Returns ``(csv_path, using_default)``. When ``using_default`` is True, this run created
    ``MACRO_PLACE_GPU_LOG_PATH`` pointing at ``logs/liquid_<slug>_proxycheck.csv`` and the
    caller should truncate that file at run start so the file reflects only this placement.
    """
    raw = (os.environ.get("MACRO_PLACE_GPU_LOG_PATH") or "").strip()
    logs_dir = _ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    slug = _bench_slug(benchmark_name)
    if raw:
        p = Path(raw)
        p = p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
        return p, False
    p = (logs_dir / f"liquid_{slug}_proxycheck.csv").resolve()
    os.environ["MACRO_PLACE_GPU_LOG_PATH"] = str(p)
    return p, True


def _write_liquid_congestion_md(
    benchmark_name: str,
    proxycheck_csv: Path,
    out_md: Path,
    *,
    cycles_completed: int,
) -> None:
    """Summarize surrogate vs PLC congestion from the patience-phase proxy-check CSV."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    slug = _bench_slug(benchmark_name)
    rel_csv = proxycheck_csv
    try:
        rel_csv = proxycheck_csv.relative_to(_ROOT)
    except ValueError:
        pass
    rel_md = out_md
    try:
        rel_md = out_md.relative_to(_ROOT)
    except ValueError:
        pass

    header = (
        f"# Liquid placement — congestion ({benchmark_name})\n\n"
        f"- Benchmark slug: `{slug}`\n"
        f"- Cycles completed: {cycles_completed}\n"
        f"- Proxy-check CSV: `{rel_csv}`\n\n"
    )

    if not proxycheck_csv.is_file():
        out_md.write_text(
            header
            + "No proxy-check CSV was produced (file missing). "
            "PLC is required during the patience phase with `MACRO_PLACE_GPU_PROXY_CHECK_EVERY` > 0 "
            "for per-checkpoint congestion logging.\n",
            encoding="utf-8",
        )
        return

    rows: list[dict[str, str]] = []
    with proxycheck_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"epoch", "sur_cong", "px_cong", "w_cong", "err_cong"}
        if reader.fieldnames is None or not need.issubset(set(reader.fieldnames)):
            out_md.write_text(
                header
                + "Proxy-check CSV exists but lacks expected columns "
                f"(need {sorted(need)}). Header: {reader.fieldnames!r}\n",
                encoding="utf-8",
            )
            return
        for row in reader:
            rows.append(row)

    if not rows:
        out_md.write_text(
            header
            + "The proxy-check CSV has no data rows (only a header, or empty). "
            "Congestion table is omitted.\n",
            encoding="utf-8",
        )
        return

    def _f(row: dict[str, str], key: str) -> float:
        return float(row[key])

    sur = [_f(r, "sur_cong") for r in rows]
    px = [_f(r, "px_cong") for r in rows]
    wcong = [_f(r, "w_cong") for r in rows]
    err = [_f(r, "err_cong") for r in rows]

    def _stats(xs: list[float]) -> str:
        return (
            f"min={min(xs):.6g}, max={max(xs):.6g}, mean={sum(xs) / len(xs):.6g}"
        )

    lines: list[str] = [
        header,
        "## Surrogate vs PLC congestion (patience proxy checks)\n\n",
        "Per checkpoint: `sur_cong` = GPU surrogate congestion term; `px_cong` = PLC congestion subcost; "
        "`w_cong` = scale on surrogate congestion (EMA calibration); "
        "`err_cong` = `w_cong * sur_cong - px_cong` (CSV column). "
        "`sur_minus_px` = `sur_cong - px_cong`; `pct_sur_vs_px` = `(sur_cong - px_cong) / px_cong * 100` "
        "when `px_cong` is nonzero.\n\n",
        "| epoch | sur_cong | px_cong | w_cong | err_cong | sur_minus_px | pct_sur_vs_px |\n",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for r in rows:
        ep = r["epoch"].strip()
        s = _f(r, "sur_cong")
        p = _f(r, "px_cong")
        w = _f(r, "w_cong")
        e = _f(r, "err_cong")
        d = s - p
        if abs(p) > 1e-30:
            pct = 100.0 * d / p
            pct_s = f"{pct:.4g}"
        else:
            pct_s = "—"
        lines.append(
            f"| {ep} | {s:.6g} | {p:.6g} | {w:.6g} | {e:.6g} | {d:.6g} | {pct_s} |\n"
        )

    lines.extend(
        [
            "\n## Summary\n\n",
            f"- Rows: **{len(rows)}**\n",
            f"- `sur_cong`: {_stats(sur)}\n",
            f"- `px_cong`: {_stats(px)}\n",
            f"- `w_cong`: {_stats(wcong)}\n",
            f"- `err_cong`: {_stats(err)}\n\n",
            f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC._\n",
        ]
    )
    out_md.write_text("".join(lines), encoding="utf-8")


def _plc_proxy_cost(pos: torch.Tensor, benchmark: Benchmark, plc) -> float | None:
    if plc is None:
        return None
    if not torch.isfinite(pos).all():
        return None
    costs = compute_proxy_cost(pos.clone(), benchmark, plc)
    px = float(costs["proxy_cost"])
    return px if math.isfinite(px) else None


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
def _liquid_congestion_defaults():
    """Force L-route congestion for entire ``LiquidPlacer.place`` (overrides user env)."""
    saved = {k: os.environ.get(k) for k in _LIQUID_GPU_CONG_ENV_KEYS}
    os.environ["MACRO_PLACE_GPU_PLC_NET_ROUTING"] = _LIQUID_GPU_PLC_NET_ROUTING
    os.environ["MACRO_PLACE_GPU_SPATIAL_CONG"] = _LIQUID_GPU_SPATIAL_CONG
    os.environ["MACRO_PLACE_GPU_PROXY_LOG"] = "off"
    os.environ["MACRO_PLACE_GPU_TRAINING_PLOT"] = "0"
    os.environ["MACRO_PLACE_GPU_FINAL_SUMMARY"] = "0"
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@contextmanager
def _liquid_gpu_env():
    """Liquid phase: L-route congestion surrogate; no PLC proxy checks."""
    saved = {
        "MACRO_PLACE_GPU_PROXY_CHECK_EVERY": os.environ.get(
            "MACRO_PLACE_GPU_PROXY_CHECK_EVERY"
        ),
        "MACRO_PLACE_GPU_USE_BBOX_RUDY": os.environ.get("MACRO_PLACE_GPU_USE_BBOX_RUDY"),
    }
    os.environ["MACRO_PLACE_GPU_PROXY_CHECK_EVERY"] = "0"
    os.environ["MACRO_PLACE_GPU_USE_BBOX_RUDY"] = "0"
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@contextmanager
def _patience_gpu_env(proxy_check_every: int):
    """Patience phase: L-route cong + PLC proxy checks + affine calibration."""
    saved = {
        "MACRO_PLACE_GPU_OPTIMIZER": os.environ.get("MACRO_PLACE_GPU_OPTIMIZER"),
        "MACRO_PLACE_GPU_PROXY_CHECK_EVERY": os.environ.get(
            "MACRO_PLACE_GPU_PROXY_CHECK_EVERY"
        ),
        "MACRO_PLACE_GPU_USE_BBOX_RUDY": os.environ.get("MACRO_PLACE_GPU_USE_BBOX_RUDY"),
    }
    os.environ["MACRO_PLACE_GPU_OPTIMIZER"] = "adam"
    os.environ["MACRO_PLACE_GPU_PROXY_CHECK_EVERY"] = str(int(proxy_check_every))
    os.environ["MACRO_PLACE_GPU_USE_BBOX_RUDY"] = "0"
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
    Liquid GPU phase → patience GPU phase (× cycles), per-cycle PLC gate → Abbacus.

    Congestion uses **L-route** only: ``use_plc_net_routing=False``, ``use_spatial_cong=False``, and env
    ``MACRO_PLACE_GPU_PLC_NET_ROUTING=0`` / ``SPATIAL_CONG=0``.

    Optional ``MACRO_PLACE_LIQUID_QP_IF_OVERLAPS``: ``QPLegalizer`` after Abbacus if overlaps remain.

    The evaluate loader instantiates this class with no arguments.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t_place0 = time.perf_counter()
        verbose = _liquid_verbose()
        cycles_cap = _int_env("MACRO_PLACE_LIQUID_CYCLES", _DEFAULT_CYCLES)
        liquid_epoch_cap = _liquid_epoch_cap_env(_DEFAULT_LIQUID_EPOCH_CAP)
        liquid_surr_rel = _float_env(
            "MACRO_PLACE_LIQUID_SURROGATE_MIN_REL_INITIAL",
            _DEFAULT_LIQUID_SURROGATE_MIN_REL_INITIAL,
        )
        liquid_surr_patience = _int_env(
            "MACRO_PLACE_LIQUID_SURROGATE_STAG_PATIENCE", _DEFAULT_LIQUID_SURR_STAG_PATIENCE
        )
        liquid_surr_every = _int_env(
            "MACRO_PLACE_LIQUID_SURROGATE_CHECK_EVERY", _DEFAULT_LIQUID_SURROGATE_CHECK_EVERY
        )
        liquid_lr = _float_env("MACRO_PLACE_LIQUID_LR", _DEFAULT_LIQUID_LR)
        liquid_w_den = _float_env("MACRO_PLACE_LIQUID_W_DENSITY", _DEFAULT_LIQUID_W_DENSITY)
        patience_w_den = _float_env(
            "MACRO_PLACE_LIQUID_PATIENCE_W_DENSITY", _DEFAULT_PATIENCE_W_DENSITY
        )
        liquid_w_ovl = _float_env("MACRO_PLACE_LIQUID_W_OVERLAP", _DEFAULT_LIQUID_W_OVERLAP)
        patience_epochs = _int_env("MACRO_PLACE_LIQUID_PATIENCE_EPOCHS", _DEFAULT_PATIENCE_EPOCH_CAP)
        min_abs = _float_env("MACRO_PLACE_LIQUID_STAGNATION_MIN_ABS", _DEFAULT_STAG_MIN_ABS)
        stag_patience = _int_env("MACRO_PLACE_LIQUID_STAGNATION_PATIENCE", _DEFAULT_STAG_PATIENCE)
        every = _int_env("MACRO_PLACE_LIQUID_PROXY_CHECK_EVERY", _DEFAULT_PROXY_CHECK_EVERY)
        qp_if_overlaps = _liquid_env_bool("MACRO_PLACE_LIQUID_QP_IF_OVERLAPS")

        plc = _try_load_plc_iccad04(benchmark)
        if _liquid_env_bool("MACRO_PLACE_LIQUID_RANDOM_HARD_START"):
            seed_raw = (os.environ.get("MACRO_PLACE_LIQUID_RANDOM_HARD_SEED") or "").strip()
            seed = int(seed_raw) if seed_raw else None
            randomize_movable_hard_macro_centers(benchmark, seed=seed)
            n_hard = int(benchmark.num_hard_macros)
            n_mov = sum(
                1
                for i in range(n_hard)
                if not bool(benchmark.macro_fixed[i].item())
            )
            _liquid_log(
                f"[liquid] random hard start: {n_mov} movable hard macro(s) "
                f"(seed={seed})"
            )
        pos = benchmark.macro_positions.clone()

        cycles_env_raw = (os.environ.get("MACRO_PLACE_LIQUID_CYCLES") or "").strip()
        cycles_note = (
            f" [MACRO_PLACE_LIQUID_CYCLES={cycles_env_raw!r} from environment]"
            if cycles_env_raw
            else " [cycles: default 1]"
        )

        if verbose:
            print(
                f"[liquid] cycles_cap={cycles_cap} liquid_epoch_cap={liquid_epoch_cap} "
                f"liquid_surr_min_rel_initial={liquid_surr_rel:g} liquid_surr_stag_patience={liquid_surr_patience} "
                f"liquid_surr_check_every={liquid_surr_every} liquid_lr={liquid_lr:g} "
                f"liquid_w_density={liquid_w_den} patience_w_density={patience_w_den} liquid_w_overlap={liquid_w_ovl} "
                f"(w_wl={_PATIENCE_W_WL} w_cong={_PATIENCE_W_CONG} shared with patience) "
                f"patience_epochs={patience_epochs} plc_stag_min_abs={min_abs} "
                f"stagnation_patience={stag_patience} "
                f"proxy_check_every={every} plc_gate={'on' if plc is not None else 'off'} "
                f"gpu_plc_net_routing={_LIQUID_GPU_PLC_NET_ROUTING} "
                f"gpu_spatial_cong={_LIQUID_GPU_SPATIAL_CONG} "
                f"qp_if_overlaps={'on' if qp_if_overlaps else 'off'}"
                f"{cycles_note}",
                flush=True,
            )

        gpu_log_every = liquid_surr_every if verbose else 0
        proxycheck_csv: Path | None = None
        congestion_md: Path | None = None
        if verbose:
            proxycheck_csv, using_default_log = _setup_proxycheck_log_path(benchmark.name)
            congestion_md = (
                _ROOT / "logs" / f"liquid_{_bench_slug(benchmark.name)}_congestion.md"
            ).resolve()
            if using_default_log:
                proxycheck_csv.unlink(missing_ok=True)

        cycles_done = 0
        with _default_cuda_for_gpu_phase(), _liquid_congestion_defaults():
            for cycle in range(cycles_cap):
                pos_before = pos.clone()
                px_before = _plc_proxy_cost(pos_before, benchmark, plc)
                if verbose and px_before is not None:
                    _liquid_log(
                        f"[liquid] cycle {cycle + 1}/{cycles_cap} start proxy={px_before:.6g}"
                    )

                _cuda_reset_between_phases()
                if liquid_epoch_cap <= 0:
                    _liquid_log(
                        f"[liquid] cycle {cycle + 1}/{cycles_cap} phase=liquid "
                        f"SKIPPED (MACRO_PLACE_LIQUID_GPU_EPOCHS=0)"
                    )
                else:
                    _liquid_log(
                        f"[liquid] cycle {cycle + 1}/{cycles_cap} phase=liquid "
                        f"(surrogate plateau: min_improve={liquid_surr_rel:g}x initial_loss "
                        f"every={liquid_surr_every} patience={liquid_surr_patience}) "
                        f"epoch_cap={liquid_epoch_cap} lr={liquid_lr:g} w_density={liquid_w_den} "
                        f"w_overlap={liquid_w_ovl} (w_wl={_PATIENCE_W_WL} w_cong={_PATIENCE_W_CONG})"
                    )
                    with _liquid_gpu_env():
                        gpu_liquid = GpuPlacer(
                            epochs=liquid_epoch_cap,
                            lr=liquid_lr,
                            w_wl=_PATIENCE_W_WL,
                            w_density=liquid_w_den,
                            w_cong=_PATIENCE_W_CONG,
                            w_overlap=liquid_w_ovl,
                            log_every=gpu_log_every,
                            stagnation_proxy_patience=0,
                            stagnation_min_abs_improvement=0.0,
                            stagnation_surrogate_patience=liquid_surr_patience,
                            stagnation_surrogate_min_abs=0.0,
                            stagnation_surrogate_min_rel_initial=liquid_surr_rel,
                            surrogate_stagnation_check_every=liquid_surr_every,
                            affine_calibrate=False,
                            use_spatial_cong=False,
                            use_plc_net_routing=False,
                            seed=cycle * 10_000,
                        )
                        pos = gpu_liquid.place(benchmark, initial_macro_positions=pos)

                _cuda_reset_between_phases()

                _liquid_log(
                    f"[liquid] cycle {cycle + 1}/{cycles_cap} phase=patience "
                    f"epochs_cap={patience_epochs} w_density={patience_w_den} stagnation_min_abs={min_abs} "
                    f"stagnation_patience={stag_patience} "
                    f"proxy_check_every={every}"
                )
                with _patience_gpu_env(every):
                    gpu_patience = GpuPlacer(
                        epochs=patience_epochs,
                        w_wl=_PATIENCE_W_WL,
                        w_cong=_PATIENCE_W_CONG,
                        stagnation_min_abs_improvement=min_abs,
                        stagnation_proxy_patience=stag_patience,
                        affine_calibrate=True,
                        w_density=patience_w_den,
                        log_every=0,
                        use_spatial_cong=False,
                        use_plc_net_routing=False,
                        late_stagnation_sgd_switch=True,
                        seed=cycle * 10_000 + 5_000,
                    )
                    pos = gpu_patience.place(benchmark, initial_macro_positions=pos)
                _cuda_reset_between_phases()

                cycles_done = cycle + 1
                px_after = _plc_proxy_cost(pos, benchmark, plc)

                if verbose and px_before is not None and px_after is not None:
                    delta = px_after - px_before
                    _liquid_log(
                        f"[liquid] cycle {cycle + 1}/{cycles_cap} end proxy={px_after:.6g} "
                        f"delta_vs_cycle_start={delta:+.6g}"
                    )
                    if px_after > px_before + 1e-22:
                        pos = pos_before
                        _liquid_log(
                            "[liquid] proxy worsened vs cycle start; "
                            "reverting placement and stopping cycles."
                        )
                        break
                elif verbose:
                    _liquid_log(
                        f"[liquid] finished cycle {cycle + 1}/{cycles_cap} "
                        "(no PLC proxy gate; continuing)"
                    )

        _liquid_log(f"[liquid] completed {cycles_done} cycle(s)")

        if verbose and proxycheck_csv is not None and congestion_md is not None:
            _write_liquid_congestion_md(
                benchmark.name,
                proxycheck_csv,
                congestion_md,
                cycles_completed=cycles_done,
            )
            try:
                rel_congestion_md = congestion_md.relative_to(_ROOT)
            except ValueError:
                rel_congestion_md = congestion_md
            _liquid_log(f"[liquid] wrote congestion report {rel_congestion_md}")

        pos = AbbacusLegalizer().place(benchmark, initial_macro_positions=pos)

        save_path = _resolve_liquid_save_placement_path(benchmark.name)
        if save_path is not None:
            px_save = _plc_proxy_cost(pos, benchmark, plc)
            _save_liquid_placement_checkpoint(
                save_path,
                benchmark,
                pos,
                stage="post_abbacus",
                plc_proxy=px_save,
            )

        if qp_if_overlaps:
            ov = compute_overlap_metrics(pos, benchmark)
            n_pairs = int(ov["overlap_count"])
            if n_pairs > 0:
                _liquid_log(
                    f"[liquid] Abbacus: {n_pairs} overlapping hard-macro pair(s); "
                    "running QPLegalizer..."
                )
                pos = QPLegalizer().place(benchmark, initial_macro_positions=pos)
                ov_qp = compute_overlap_metrics(pos, benchmark)
                _liquid_log(
                    f"[liquid] post-QP: {int(ov_qp['overlap_count'])} overlapping "
                    "hard-macro pair(s)"
                )

        px_final = _plc_proxy_cost(pos, benchmark, plc)
        elapsed = time.perf_counter() - t_place0
        if not (_liquid_eval_quiet() and not verbose):
            if px_final is not None:
                print(f"[liquid] proxy={px_final:.6g} elapsed={elapsed:.1f}s", flush=True)
            else:
                print(f"[liquid] elapsed={elapsed:.1f}s", flush=True)

        return pos
