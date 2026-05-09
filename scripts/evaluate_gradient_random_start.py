#!/usr/bin/env python3
"""Gradient placer — optional random legal start or default ``initial.plc`` positions.

By default, movable macro centers are randomized in-bounds (stress-test). Pass
``--no-randomize`` to keep positions from ``initial.plc`` / the loader.

Example:
    uv run python scripts/evaluate_gradient_random_start.py -b ibm01
    uv run python scripts/evaluate_gradient_random_start.py -b ibm01 --epochs 2000 --seed 42
    uv run python scripts/evaluate_gradient_random_start.py -b ibm06 --no-randomize --epochs 100
    uv run python scripts/evaluate_gradient_random_start.py -b ibm01 --save-placement ibm01_state.pt
    uv run python scripts/evaluate_gradient_random_start.py -b ibm01 --load-placement ibm01_state.pt --epochs 500
    uv run python scripts/evaluate_gradient_random_start.py -b ibm01 --load-placement ibm01_state.pt --qp-only
    uv run python scripts/evaluate_gradient_random_start.py -b ibm01 --epochs 2000 --no-randomize --legalize-at-end
    uv run python scripts/evaluate_gradient_random_start.py -b ibm01 --epochs 500 --qp-before-gradient --legalize-at-end --no-randomize

Uses periodic ``compute_proxy_cost`` during training (default every 50 epochs) so the returned
placement is the **best proxy** seen, not the last epoch — ``proxy_eval_interval=0`` disables that
and always returns the final state (misleading Δ on long runs).

Training CSV + ``[surrogate_vs_proxy]`` lines go to ``--training-log-csv`` (default ``logs.txt``).

Placement checkpoints are **only** written when you pass ``--save-placement PATH`` (no default path).
Use ``--qp-only`` with ``--load-placement`` to run ``QPLegalizer`` on a saved tensor and print proxy
before vs after (no gradient training).

``--legalize-at-end`` runs ``QPLegalizer`` after training on the returned placement, prints
Δ vs initial for both gradient and QP endpoints, and appends ``[post_qp]`` to the training log.

``--qp-before-gradient`` runs ``QPLegalizer`` once **before** gradient training so optimization
starts from a legalized layout (use with ``--legalize-at-end`` for QP → gradient → QP).

``--epoch-timing-diagnostic PATH`` writes per-epoch segment timings (seconds) and a footer
summary (mean / median / p95 per column); see ``_EPOCH_DIAG_NUMERIC_COLS`` in ``gradient3.py``.

``gradient3`` (``Gradient3Placer``) uses SGD + Nesterov with separate hard/soft macro LRs; tune
``--lr-pos-hard`` and ``--lr-pos-soft`` when needed (defaults match the placer).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from macro_place.evaluate import evaluate_benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

from submissions.gradient3 import Gradient3Placer, randomize_movable_macro_centers
from submissions.qp import QPLegalizer

_CHECKPOINT_VERSION = 1


def _torch_load_checkpoint(path: Path) -> dict:
    """Load placement checkpoint (dict or legacy bare tensor)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "placement" in obj:
        return obj
    if isinstance(obj, torch.Tensor):
        return {"placement": obj, "benchmark": None, "format_version": 0}
    raise SystemExit(
        f"Unrecognized checkpoint in {path}: expected dict with 'placement' or a tensor."
    )


def _proxy_delta_pct(initial: float, final: float) -> float:
    if abs(initial) < 1e-30:
        return 0.0
    return (final - initial) / initial * 100.0


def _append_post_qp_log(
    log_csv: str,
    *,
    proxy: float,
    wl: float,
    den: float,
    cong: float,
    overlaps: int,
    delta_vs_initial_pct: float,
) -> None:
    path = Path(log_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(
            f"[post_qp] proxy={proxy:.10g} wl={wl:.10g} den={den:.10g} cong={cong:.10g} "
            f"overlaps={overlaps} delta_vs_initial_pct={delta_vs_initial_pct:+.10g}\n"
        )


def _apply_placement_checkpoint(benchmark, ckpt: dict, expect_name: str | None) -> None:
    t = ckpt["placement"]
    if not isinstance(t, torch.Tensor):
        raise SystemExit("Checkpoint 'placement' is not a tensor.")
    want = benchmark.macro_positions.shape
    if t.shape != want:
        raise SystemExit(
            f"Placement shape {tuple(t.shape)} does not match benchmark {tuple(want)}."
        )
    saved_name = ckpt.get("benchmark")
    if expect_name and saved_name and str(saved_name) != str(expect_name):
        print(
            f"Warning: checkpoint benchmark={saved_name!r} differs from -b {expect_name!r}.",
            file=sys.stderr,
        )
    benchmark.macro_positions.copy_(
        t.to(dtype=benchmark.macro_positions.dtype, device=benchmark.macro_positions.device)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate Gradient3Placer from a random start (default) or initial.plc (--no-randomize)."
        ),
    )
    ap.add_argument(
        "-b",
        "--benchmark",
        default="ibm01",
        help="ICCAD04 benchmark name (default: ibm01).",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Training epochs (default: 2000).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed: layout randomization (if used) and Gradient3Placer training (default: 0).",
    )
    ap.add_argument(
        "--no-randomize",
        action="store_true",
        help=(
            "Do not randomize movable macros — use placement from initial.plc / loader as-is."
        ),
    )
    ap.add_argument(
        "--proxy-eval-interval",
        type=int,
        default=50,
        help=(
            "Run PlacementCost proxy every N epochs and keep best placement (default: 50). "
            "Use 0 only for quick runs — training then returns the last epoch, not best proxy."
        ),
    )
    ap.add_argument(
        "--proxy-patience",
        type=int,
        default=0,
        help="Stop after this many proxy checkpoints without improvement (0 = disabled).",
    )
    ap.add_argument(
        "--training-log-csv",
        type=str,
        default="logs.txt",
        help=(
            "Append training CSV and surrogate-vs-proxy lines here "
            "(default: logs.txt in cwd; use empty string to disable)."
        ),
    )
    ap.add_argument(
        "--save-placement",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "After training, write a torch checkpoint (placement tensor + benchmark name) "
            "for continuing with --load-placement. Empty = disabled."
        ),
    )
    ap.add_argument(
        "--load-placement",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Before training, load macro centers from a file saved by --save-placement. "
            "Skips the default random layout (mutually exclusive with a fresh random start)."
        ),
    )
    ap.add_argument(
        "--qp-only",
        action="store_true",
        help=(
            "No training: require --load-placement, run QPLegalizer on that layout, "
            "print PlacementCost proxy before vs after. "
            "Use after saving with --save-placement."
        ),
    )
    ap.add_argument(
        "--legalize-at-end",
        action="store_true",
        help=(
            "After training, run QPLegalizer on the returned placement. "
            "Prints Δ vs initial for gradient result and for QP result; appends [post_qp] to "
            "--training-log-csv when logging is enabled."
        ),
    )
    ap.add_argument(
        "--qp-before-gradient",
        action="store_true",
        help=(
            "Run QPLegalizer once before gradient training; optimization starts from the "
            "QP-legalized placement. Combine with --legalize-at-end for QP → gradient → QP."
        ),
    )
    ap.add_argument(
        "--epoch-timing-diagnostic",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Write per-epoch CUDA-synchronized segment timings (seconds) to this file "
            "(CSV + summary). Empty = disabled."
        ),
    )
    ap.add_argument(
        "--lr-pos-hard",
        type=float,
        default=None,
        metavar="LR",
        help=(
            "Learning rate for movable hard macros (default: Gradient3Placer built-in default)."
        ),
    )
    ap.add_argument(
        "--lr-pos-soft",
        type=float,
        default=None,
        metavar="LR",
        help=(
            "Learning rate for soft macros (default: Gradient3Placer built-in default)."
        ),
    )
    args = ap.parse_args()

    if args.qp_only and not args.load_placement.strip():
        ap.error("--qp-only requires --load-placement PATH")
    if args.qp_only and args.legalize_at_end:
        ap.error("--qp-only and --legalize-at-end are mutually exclusive")
    if args.qp_only and args.qp_before_gradient:
        ap.error("--qp-before-gradient cannot be used with --qp-only")

    testcase_root = _REPO_ROOT / "external/MacroPlacement/Testcases/ICCAD04"
    if not testcase_root.is_dir():
        print(f"Missing testcases: {testcase_root}")
        sys.exit(1)

    name = args.benchmark
    benchmark_dir = str(testcase_root / name)
    benchmark, plc = load_benchmark_from_dir(benchmark_dir)

    load_path = args.load_placement.strip()
    if load_path:
        ckpt = _torch_load_checkpoint(Path(load_path).expanduser().resolve())
        _apply_placement_checkpoint(benchmark, ckpt, name)
        print(f"Loaded placement from {load_path}")
    elif not args.no_randomize:
        randomize_movable_macro_centers(benchmark, seed=args.seed)

    pre_qp_gradient_log_line: str | None = None
    if args.qp_before_gradient:
        print("=" * 72)
        print(f"Pre-gradient QP · {name}")
        print("=" * 72)
        placement_in = benchmark.macro_positions.clone()
        t_qp_pre = time.perf_counter()
        costs_in = compute_proxy_cost(placement_in, benchmark, plc)
        benchmark.macro_positions.copy_(placement_in)
        placement_qp_pre = QPLegalizer().place(benchmark)
        costs_out = compute_proxy_cost(placement_qp_pre, benchmark, plc)
        elapsed_pre = time.perf_counter() - t_qp_pre
        pi, po = costs_in["proxy_cost"], costs_out["proxy_cost"]
        dlt_pre = _proxy_delta_pct(pi, po)
        print(
            f"proxy_before_Qp={pi:.6f}  proxy_after_Qp={po:.6f}  Δ={dlt_pre:+.2f}%  "
            f"overlaps {costs_in['overlap_count']} → {costs_out['overlap_count']}  "
            f"[{elapsed_pre:.2f}s]"
        )
        print(
            f"  before: wl={costs_in['wirelength_cost']:.4f} den={costs_in['density_cost']:.4f} "
            f"cong={costs_in['congestion_cost']:.4f}"
        )
        print(
            f"  after:  wl={costs_out['wirelength_cost']:.4f} den={costs_out['density_cost']:.4f} "
            f"cong={costs_out['congestion_cost']:.4f}"
        )
        print()
        log_pre = args.training_log_csv.strip()
        if log_pre:
            pre_qp_gradient_log_line = (
                f"[pre_qp_gradient] proxy_before={float(pi):.10g} proxy_after={float(po):.10g} "
                f"wl_before={float(costs_in['wirelength_cost']):.10g} "
                f"den_before={float(costs_in['density_cost']):.10g} "
                f"cong_before={float(costs_in['congestion_cost']):.10g} "
                f"overlaps_before={int(costs_in['overlap_count'])} "
                f"wl_after={float(costs_out['wirelength_cost']):.10g} "
                f"den_after={float(costs_out['density_cost']):.10g} "
                f"cong_after={float(costs_out['congestion_cost']):.10g} "
                f"overlaps_after={int(costs_out['overlap_count'])} "
                f"delta_proxy_pct={dlt_pre:+.10g}\n"
            )

    if args.qp_only:
        print("=" * 72)
        print(f"QP legalize only · {name} · checkpoint={load_path}")
        print("=" * 72)
        placement_in = benchmark.macro_positions.clone()
        t0 = time.perf_counter()
        costs_in = compute_proxy_cost(placement_in, benchmark, plc)
        benchmark.macro_positions.copy_(placement_in)
        placement_out = QPLegalizer().place(benchmark)
        costs_out = compute_proxy_cost(placement_out, benchmark, plc)
        elapsed = time.perf_counter() - t0
        pi, po = costs_in["proxy_cost"], costs_out["proxy_cost"]
        dlt = (po - pi) / pi * 100.0 if abs(pi) > 1e-30 else 0.0
        print(
            f"proxy_before_Qp={pi:.6f}  proxy_after_Qp={po:.6f}  Δ={dlt:+.2f}%  "
            f"overlaps {costs_in['overlap_count']} → {costs_out['overlap_count']}  "
            f"[{elapsed:.2f}s]"
        )
        print(
            f"  before: wl={costs_in['wirelength_cost']:.4f} den={costs_in['density_cost']:.4f} "
            f"cong={costs_in['congestion_cost']:.4f}"
        )
        print(
            f"  after:  wl={costs_out['wirelength_cost']:.4f} den={costs_out['density_cost']:.4f} "
            f"cong={costs_out['congestion_cost']:.4f}"
        )
        save_path = args.save_placement.strip()
        if save_path:
            out = Path(save_path).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format_version": _CHECKPOINT_VERSION,
                    "benchmark": name,
                    "placement": placement_out.detach().cpu(),
                    "qp_legalized": True,
                },
                out,
            )
            print(f"Saved QP-legalized placement → {out}")
        print()
        return

    print("=" * 72)
    qp_note = " · pre-QP seed" if args.qp_before_gradient else ""
    if load_path:
        print(f"Resume from file · {name} · epochs={args.epochs} · train_seed={args.seed}{qp_note}")
    elif args.no_randomize:
        print(f"Initial plc · {name} · epochs={args.epochs} · train_seed={args.seed}{qp_note}")
    else:
        print(f"Random start · {name} · epochs={args.epochs} · layout_seed={args.seed}{qp_note}")
    print("=" * 72)

    log_csv = args.training_log_csv.strip() or None
    epoch_diag = args.epoch_timing_diagnostic.strip() or None

    placer_kw: dict = dict(
        epochs=args.epochs,
        proxy_adaptive_weights=False,
        proxy_eval_interval=args.proxy_eval_interval,
        proxy_patience=args.proxy_patience,
        training_log_csv=log_csv,
        training_log_plot=None,
        seed=args.seed,
        epoch_timing_diagnostic=epoch_diag,
    )
    if args.lr_pos_hard is not None:
        placer_kw["lr_pos_hard"] = float(args.lr_pos_hard)
    if args.lr_pos_soft is not None:
        placer_kw["lr_pos_soft"] = float(args.lr_pos_soft)
    placer = Gradient3Placer(**placer_kw)
    r = evaluate_benchmark(
        placer,
        name,
        str(testcase_root),
        benchmark=benchmark,
        plc=plc,
    )
    if pre_qp_gradient_log_line is not None and args.training_log_csv.strip():
        log_path = Path(args.training_log_csv.strip()).expanduser().resolve()
        if log_path.is_file():
            body = log_path.read_text(encoding="utf-8")
            log_path.write_text(pre_qp_gradient_log_line + body, encoding="utf-8")
            print(
                f"Prepended [pre_qp_gradient] line → {log_path}",
                flush=True,
            )

    dpi, dp = r["proxy_cost_initial"], r["proxy_cost"]
    dlt_grad = _proxy_delta_pct(dpi, dp)
    print(
        f"proxy_initial={dpi:.6f}  proxy_after_gradient={dp:.6f}  "
        f"Δ_vs_initial before legalization={dlt_grad:+.2f}%  "
        f"(wl={r['wirelength']:.4f} den={r['density']:.4f} cong={r['congestion']:.4f})  "
        f"overlaps={r['overlaps']}  [{r['runtime']:.2f}s]"
    )

    dp_qp: float | None = None
    if args.legalize_at_end:
        t_qp0 = time.perf_counter()
        pg = r["placement"].detach()
        benchmark.macro_positions.copy_(
            pg.to(dtype=benchmark.macro_positions.dtype, device=benchmark.macro_positions.device)
        )
        placement_qp = QPLegalizer().place(benchmark)
        costs_qp = compute_proxy_cost(placement_qp, benchmark, plc)
        dp_qp = float(costs_qp["proxy_cost"])
        dlt_qp = _proxy_delta_pct(dpi, dp_qp)
        dlt_step = _proxy_delta_pct(dp, dp_qp)
        t_qp = time.perf_counter() - t_qp0
        print(
            f"proxy_after_qp={dp_qp:.6f}  Δ_vs_initial after legalization={dlt_qp:+.2f}%  "
            f"(QP step vs gradient only: {dlt_step:+.2f}%)  "
            f"wl={costs_qp['wirelength_cost']:.4f} den={costs_qp['density_cost']:.4f} "
            f"cong={costs_qp['congestion_cost']:.4f}  overlaps={costs_qp['overlap_count']}  "
            f"[{t_qp:.2f}s]"
        )
        log_csv_path = args.training_log_csv.strip()
        if log_csv_path:
            _append_post_qp_log(
                log_csv_path,
                proxy=dp_qp,
                wl=float(costs_qp["wirelength_cost"]),
                den=float(costs_qp["density_cost"]),
                cong=float(costs_qp["congestion_cost"]),
                overlaps=int(costs_qp["overlap_count"]),
                delta_vs_initial_pct=dlt_qp,
            )
            print(f"Appended [post_qp] line → {Path(log_csv_path).expanduser().resolve()}")

    save_path = args.save_placement.strip()
    if save_path:
        out = Path(save_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        placement_cpu = r["placement"].detach().cpu()
        torch.save(
            {
                "format_version": _CHECKPOINT_VERSION,
                "benchmark": name,
                "placement": placement_cpu,
                "epochs_trained": args.epochs,
                "train_seed": args.seed,
            },
            out,
        )
        print(f"Saved placement checkpoint → {out}")

    print()


if __name__ == "__main__":
    main()
