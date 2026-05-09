"""
Gradient-based macro placement (``gradient2.py`` variant):

Capacity-style **ReLU** gates are replaced with **softplus** where density or RUDY
congestion measures excess above a threshold—``softplus(x - threshold)`` is close to
``relu(x - threshold)`` when far above the limit, but keeps a **continuous nonzero
gradient** below it so bins approaching the limit still receive optimizer signal.

Surrogate losses (differentiable) vs ``compute_proxy_cost`` / PlacementCost:

* **Wirelength surrogate:** weighted HPWL ``(max - min)`` on pins, then **÷ ((canvas_w +
  canvas_h) × num_nets)** to match PlacementCost ``get_cost()`` normalization.
* **Density surrogate:** macro rectangle overlap / bin area on the placement grid, then
  **average of top 10% bin densities × 0.5** (same reduction as ``get_density_cost()``).
* **Congestion surrogate (default):** PLC-style **L-route net demand** + **normalized macro
  blockage**, **smooth net maps** then add macro (matching PlacementCost ``get_routing`` order),
  then **mean of top 5%** over all H/V cells like ``abu(V+H, 0.05)``. Set ``use_plc_routing_cong=False``
  for the legacy LSE RUDY overflow surrogate.

Adam optimization and optional QP legalization on hard macros.

Training can stop early on wall time, repeated lack of evaluator-proxy improvement (patience),
or (after ``loss_flat_min_epoch``) flat surrogate loss; otherwise it runs for ``epochs``.
When evaluator proxy was computed, the returned placement is the best proxy seen.
Default surrogate weights match the evaluator proxy: ``1.0·l_wl + 0.5·l_dh + 0.5·l_cong``
(same coefficients as ``proxy_cost``). Optional **proxy-guided** adjustment trades ``w_wl``
vs ``w_cong`` using proxy deltas between checkpoints; disable with ``proxy_adaptive_weights=False``
to keep the formula coefficients fixed.
Append-only CSV training metrics default to ``logs.txt`` (surrogate ``l_*``, total loss,
proxy when evaluated, weights used that epoch, ``delta_proxy``, ``adapt_step``); optional
``logs_plot.png`` overlays proxy on surrogate curves.

Usage:
    uv run evaluate submissions/gradient2.py -b ibm01
    uv run python submissions/gradient2.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import statistics
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.routing_surrogate import plc_routing_surrogate_scalar


def _load_qp_module():
    qp_path = Path(__file__).resolve().parent / "qp.py"
    spec = importlib.util.spec_from_file_location("_qp_gradient_bind", qp_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_QP = _load_qp_module()
QPLegalizer = _QP.QPLegalizer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_user_path(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else Path.cwd() / path


def _soft_capacity_excess(excess: torch.Tensor) -> torch.Tensor:
    """``softplus(excess)`` as a smooth stand-in for ``relu(excess)`` (``excess = x - threshold``)."""
    return F.softplus(excess)


# Per-epoch diagnostic CSV columns (after ``epoch``); see ``GradientPlacer.epoch_timing_diagnostic``.
_EPOCH_DIAG_NUMERIC_COLS: tuple[str, ...] = (
    "t_zero_grad",
    "t_assemble",
    "t_cat_ports",
    "t_epoch0_audit",
    "t_epoch0_compile_check",
    "t_epoch0_proxy_baseline",
    "t_wl",
    "t_overlap_grid",
    "t_density_scalar",
    "t_cong",
    "t_loss_scalar",
    "t_backward",
    "t_optimizer",
    "t_clamp",
    "t_proxy_checkpoint",
    "t_epoch_wall_s",
)


class _EpochDiagTimer:
    """Context manager: GPU-safe segment timing into ``row[name]`` (seconds)."""

    __slots__ = ("_device", "_name", "_row", "_t0")

    def __init__(self, row: dict[str, float], name: str, device: torch.device) -> None:
        self._row = row
        self._name = name
        self._device = device
        self._t0 = 0.0

    def __enter__(self) -> None:
        _sync(self._device)
        self._t0 = time.perf_counter()

    def __exit__(self, *args: object) -> None:
        _sync(self._device)
        self._row[self._name] = time.perf_counter() - self._t0


def _diag_segment(
    row: dict[str, float] | None, name: str, device: torch.device
):
    if row is None:
        return nullcontext()
    return _EpochDiagTimer(row, name, device)


def _diag_write_footer(
    fp,
    rows: list[dict[str, float]],
    *,
    exit_reason: str,
    configured_epochs: int,
    total_wall_s: float,
) -> None:
    fp.write("\n# summary\n")
    fp.write(f"# exit_reason={exit_reason}  configured_epochs={configured_epochs}\n")
    fp.write(f"# total_training_wall_s={total_wall_s:.6f}\n")
    fp.write(f"# recorded_epochs={len(rows)}\n")
    if not rows:
        return
    for col in _EPOCH_DIAG_NUMERIC_COLS:
        vals = [float(r[col]) for r in rows]
        fp.write(
            f"# {col}  mean_s={statistics.mean(vals):.6g}  "
            f"median_s={statistics.median(vals):.6g}  "
            f"p95_s={_diag_p95(vals):.6g}\n"
        )


def _diag_p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    idx = min(len(s) - 1, max(0, int(math.ceil(0.95 * len(s)) - 1)))
    return s[idx]


@dataclass
class _ProxyWeightAdaptState:
    """Checkpoint-to-checkpoint state for proxy-guided ``w_wl`` / ``w_cong`` coordinate steps."""

    prev_pv: float | None = None
    prev_pw: float | None = None
    prev_pd: float | None = None
    prev_pc: float | None = None
    step: float = 2.0
    sign: int = 1
    weights_before_last_step: tuple[float, float, float, float] | None = None
    applied_step: bool = False


def _subcost_bias_sign_wl_cong(d_pw: float, d_pd: float, d_pc: float) -> int:
    """+1 = next step shifts mass toward wirelength; -1 = toward congestion. Density tie-break."""
    # Largest positive delta = most degraded evaluator component.
    m = max(d_pw, d_pd, d_pc)
    if m <= 0.0:
        return 1
    tol = 1e-12 * (abs(m) + 1.0)
    if abs(m - d_pw) <= tol:
        return 1
    if abs(m - d_pc) <= tol:
        return -1
    return 1


def _proxy_guided_adapt_surrogate_weights(
    w_wl: float,
    w_dh: float,
    w_ds: float,
    w_c: float,
    pv: float,
    pw: float,
    pd: float,
    pc: float,
    state: _ProxyWeightAdaptState,
    *,
    step_min: float,
    use_subcost_bias: bool,
    wl_cong_eps: float = 1e-6,
) -> tuple[float, float, float, float, float | None]:
    """Trade mass between ``w_wl`` and ``w_c`` using consecutive proxy deltas.

    Returns ``(new_w_wl, new_w_dh, new_w_ds, new_w_c, delta_proxy)`` where
    ``delta_proxy`` is ``pv - prev_pv`` for logging (``None`` on the first checkpoint).
    """
    if state.prev_pv is None:
        state.prev_pv = float(pv)
        state.prev_pw = float(pw)
        state.prev_pd = float(pd)
        state.prev_pc = float(pc)
        return w_wl, w_dh, w_ds, w_c, None

    d_pv = float(pv) - float(state.prev_pv)
    d_pw = float(pw) - float(state.prev_pw or 0.0)
    d_pd = float(pd) - float(state.prev_pd or 0.0)
    d_pc = float(pc) - float(state.prev_pc or 0.0)

    ww, wdh, wds, wc = float(w_wl), float(w_dh), float(w_ds), float(w_c)

    if d_pv < 0.0:
        # Proxy improved: take another step in the current direction.
        snap = (ww, wdh, wds, wc)
        state.weights_before_last_step = snap
        delta = float(state.sign) * float(state.step)
        nwl = max(wl_cong_eps, ww + delta)
        nwc = max(wl_cong_eps, wc - delta)
        state.applied_step = True
        ww, wc = nwl, nwc
    else:
        # Worse or flat: revert last step if any, flip, shrink step.
        if state.applied_step and state.weights_before_last_step is not None:
            ww, wdh, wds, wc = state.weights_before_last_step
            state.applied_step = False
            state.weights_before_last_step = None
        state.sign = -int(state.sign)
        nh = 0.5 * float(state.step)
        state.step = max(float(step_min), nh)
        if use_subcost_bias:
            state.sign = _subcost_bias_sign_wl_cong(d_pw, d_pd, d_pc)

    state.prev_pv = float(pv)
    state.prev_pw = float(pw)
    state.prev_pd = float(pd)
    state.prev_pc = float(pc)

    return ww, wdh, wds, wc, float(d_pv)


def _write_gradient_log_plot(
    plot_path: Path,
    epochs: list[int],
    l_wl: list[float],
    l_cong: list[float],
    l_dh: list[float],
    l_ds: list[float],
    loss: list[float],
    proxy_epochs: list[int],
    proxy_vals: list[float],
) -> None:
    """Save surrogate metrics + loss; twin axis for evaluator proxy vs epoch."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not epochs:
        return

    ep = np.asarray(epochs, dtype=np.float64)
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax_top.plot(ep, l_wl, label="l_wl", linewidth=1.0)
    ax_top.plot(ep, l_cong, label="l_cong", linewidth=1.0)
    ax_top.plot(ep, l_dh, label="l_dh", linewidth=1.0)
    ax_top.plot(ep, l_ds, label="l_ds", linewidth=1.0)
    ax_top.set_ylabel("surrogate terms")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper left", fontsize=8)

    if proxy_epochs and proxy_vals:
        ax_tw = ax_top.twinx()
        ax_tw.plot(
            proxy_epochs,
            proxy_vals,
            "ko-",
            markersize=5,
            linewidth=1.2,
            label="evaluator proxy",
        )
        ax_tw.set_ylabel("evaluator proxy")
        ax_tw.legend(loc="upper right", fontsize=8)

    ax_bot.plot(ep, loss, color="tab:red", linewidth=1.0, label="total loss")
    ax_bot.set_xlabel("epoch")
    ax_bot.set_ylabel("total loss")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="upper right", fontsize=8)

    fig.suptitle("GradientPlacer training log")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


def _try_load_plc_iccad04(benchmark: Benchmark):
    """Rebuild ``PlacementCost`` for proxy checks (ICCAD04 layout only)."""
    root = _repo_root()
    case_dir = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / benchmark.name
    netlist = case_dir / "netlist.pb.txt"
    if not netlist.is_file():
        return None
    _, plc = load_benchmark_from_dir(str(case_dir))
    return plc


def _select_device(benchmark: Benchmark) -> torch.device:
    """Use CUDA whenever PyTorch reports it is available (no SM-version gate).

    Override with ``MACRO_PLACE_DEVICE=cpu`` to force CPU, or ``=cuda`` to require GPU
    (falls back to CPU if CUDA is unavailable).
    """
    del benchmark  # kept for API compatibility / future size-based policy
    env = os.environ.get("MACRO_PLACE_DEVICE", "").strip().lower()
    if env == "cpu":
        return torch.device("cpu")
    if env == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _soft_min(v: torch.Tensor, beta: float) -> torch.Tensor:
    return -(1.0 / beta) * torch.logsumexp(-beta * v, dim=0)


def _soft_max(v: torch.Tensor, beta: float) -> torch.Tensor:
    return (1.0 / beta) * torch.logsumexp(beta * v, dim=0)


def _pin_count(benchmark: Benchmark, k: int) -> int:
    if len(benchmark.net_pin_nodes) == benchmark.num_nets:
        return max(int(benchmark.net_pin_nodes[k].shape[0]), 2)
    return max(int(benchmark.net_nodes[k].numel()), 2)


def _build_net_tensors(
    benchmark: Benchmark,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad net node indices to (num_nets, max_pins); indices match ``torch.cat([full, ports], 0)`` rows."""
    num_nets = int(benchmark.num_nets)
    max_pins = 0
    for k in range(num_nets):
        max_pins = max(max_pins, int(benchmark.net_nodes[k].numel()))

    net_idx = torch.zeros((num_nets, max_pins), dtype=torch.long, device=device)
    net_mask = torch.zeros((num_nets, max_pins), dtype=torch.bool, device=device)

    for k in range(num_nets):
        nodes = benchmark.net_nodes[k]
        n = int(nodes.numel())
        if n > 0:
            net_idx[k, :n] = nodes.to(device=device, dtype=torch.long)
            net_mask[k, :n] = True

    w_cpu = benchmark.net_weights
    net_weights = torch.empty((num_nets,), device=device, dtype=dtype)
    for k in range(num_nets):
        pc = _pin_count(benchmark, k)
        wk = float(w_cpu[k].item()) / float(pc - 1)
        net_weights[k] = max(wk, 1e-6)

    net_valid = torch.tensor(
        [int(benchmark.net_nodes[k].numel()) >= 2 for k in range(num_nets)],
        device=device,
        dtype=torch.bool,
    )

    return net_idx, net_mask, net_weights, net_valid


def _assemble_full(
    pos_hard: torch.Tensor,
    pos_soft: torch.Tensor,
    benchmark: Benchmark,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Scatter learned positions into full [N,2] tensor."""
    n_hard = int(benchmark.num_hard_macros)
    n_macros = int(benchmark.num_macros)
    out = torch.zeros(n_macros, 2, device=device, dtype=dtype)
    orig = benchmark.macro_positions.to(device=device, dtype=dtype)
    fixed = benchmark.macro_fixed.to(device=device)

    mh = 0
    for i in range(n_hard):
        if bool(fixed[i].item()):
            out[i] = orig[i]
        else:
            out[i] = pos_hard[mh]
            mh += 1
    assert mh == pos_hard.shape[0]
    if pos_soft.numel() > 0:
        out[n_hard:n_macros] = pos_soft
    return out


def _build_assemble_ctx(
    benchmark: Benchmark, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Per-place precomputation for ``_assemble_full_fast``.

    Returns ``(movable_idx_t, fixed_idx_t, fixed_xy_detached, n_hard, n_macros)``.
    Fixed hard-macro positions are stored detached so they do not enter the autograd graph.
    """
    n_hard = int(benchmark.num_hard_macros)
    n_macros = int(benchmark.num_macros)
    if n_hard > 0:
        fixed = benchmark.macro_fixed[:n_hard].to(device=device).bool()
        movable_idx_t = (
            torch.nonzero(~fixed, as_tuple=False).squeeze(-1).to(torch.long)
        )
        fixed_idx_t = torch.nonzero(fixed, as_tuple=False).squeeze(-1).to(torch.long)
        if fixed_idx_t.numel() > 0:
            idx_cpu = fixed_idx_t.detach().cpu()
            fixed_xy = benchmark.macro_positions[idx_cpu].to(
                device=device, dtype=dtype
            ).detach()
        else:
            fixed_xy = torch.empty(0, 2, device=device, dtype=dtype)
    else:
        movable_idx_t = torch.empty(0, dtype=torch.long, device=device)
        fixed_idx_t = torch.empty(0, dtype=torch.long, device=device)
        fixed_xy = torch.empty(0, 2, device=device, dtype=dtype)

    return movable_idx_t, fixed_idx_t, fixed_xy, n_hard, n_macros


def _assemble_full_fast(
    pos_hard: torch.Tensor,
    pos_soft: torch.Tensor,
    ctx: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int],
) -> torch.Tensor:
    """Hoisted-constant equivalent of ``_assemble_full`` (no Python per-macro loop)."""
    movable_idx_t, fixed_idx_t, fixed_xy, n_hard, n_macros = ctx
    if pos_hard.numel() > 0:
        dev, dt = pos_hard.device, pos_hard.dtype
    elif pos_soft.numel() > 0:
        dev, dt = pos_soft.device, pos_soft.dtype
    else:
        dev, dt = fixed_xy.device, fixed_xy.dtype
    out = torch.empty(n_macros, 2, device=dev, dtype=dt)
    if fixed_idx_t.numel() > 0:
        out[fixed_idx_t] = fixed_xy.to(device=dev, dtype=dt)
    if pos_hard.numel() > 0:
        out[movable_idx_t] = pos_hard
    if pos_soft.numel() > 0:
        out[n_hard:n_macros] = pos_soft
    return out


def _wirelength_loss_loop(
    full_pos: torch.Tensor,
    benchmark: Benchmark,
    beta: float,
) -> torch.Tensor:
    """Reference (Python-loop) HPWL using exact ``max - min``; ``beta`` kept for sig parity."""
    del beta
    device = full_pos.device
    dtype = full_pos.dtype
    n_macros = int(benchmark.num_macros)
    n_ports = int(benchmark.port_positions.shape[0])
    ports = benchmark.port_positions.to(device=device, dtype=dtype)
    weights = benchmark.net_weights.to(device=device, dtype=dtype)
    total = torch.zeros((), device=device, dtype=dtype)

    for k in range(int(benchmark.num_nets)):
        nodes = benchmark.net_nodes[k]
        if nodes.numel() < 2:
            continue
        pc = _pin_count(benchmark, k)
        wk = max(float(weights[k] / float(pc - 1)), 1e-6)

        xs_list: list[torch.Tensor] = []
        ys_list: list[torch.Tensor] = []
        for idx in nodes.tolist():
            idx = int(idx)
            if idx < n_macros:
                xs_list.append(full_pos[idx, 0])
                ys_list.append(full_pos[idx, 1])
            else:
                pi = idx - n_macros
                if 0 <= pi < n_ports:
                    xs_list.append(ports[pi, 0])
                    ys_list.append(ports[pi, 1])
        if len(xs_list) < 2:
            continue
        xv = torch.stack(xs_list)
        yv = torch.stack(ys_list)
        hpwl_x = xv.amax() - xv.amin()
        hpwl_y = yv.amax() - yv.amin()
        total = total + wk * (hpwl_x + hpwl_y)
    return total


def _wirelength_loss_v2(
    combined_pos: torch.Tensor,
    net_idx: torch.Tensor,
    net_mask: torch.Tensor,
    net_weights: torch.Tensor,
    net_valid: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Exact per-net HPWL: ``Σ_k w_k * ((max_x - min_x) + (max_y - min_y))``.

    Uses ``amax``/``amin`` directly — non-smooth but subdifferentiable; PyTorch passes
    gradient through to the argmax/argmin pin (Adam tolerates the resulting sparsity).
    ``beta`` is kept for signature compatibility with the LSE variant but is unused.
    """
    del beta
    all_pins = combined_pos[net_idx]
    x = all_pins[:, :, 0]
    y = all_pins[:, :, 1]

    NEG_INF = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    POS_INF = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)

    x_max = x.masked_fill(~net_mask, NEG_INF).amax(dim=1)
    x_min = x.masked_fill(~net_mask, POS_INF).amin(dim=1)
    y_max = y.masked_fill(~net_mask, NEG_INF).amax(dim=1)
    y_min = y.masked_fill(~net_mask, POS_INF).amin(dim=1)

    hpwl = (x_max - x_min) + (y_max - y_min)
    # ``torch.where`` (instead of multiplying by ``net_valid``) avoids ``0 * inf = nan``
    # for degenerate 0-pin nets where the masked reductions would be ±inf.
    zero = torch.zeros((), device=hpwl.device, dtype=hpwl.dtype)
    hpwl = torch.where(net_valid, hpwl, zero)
    return (net_weights * hpwl).sum()


def _vectorized_gaussian_density_loss(
    full_pos: torch.Tensor,
    benchmark: Benchmark,
    macro_indices: list[int],
    target: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Vectorized Gaussian density penalty (same semantics as loop version)."""
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    nr = max(int(benchmark.grid_rows), 1)
    nc = max(int(benchmark.grid_cols), 1)
    cell_w = cw / nc
    cell_h = ch / nr

    cx_grid = (torch.arange(nc, device=device, dtype=dtype) + 0.5) * cell_w
    cy_grid = (torch.arange(nr, device=device, dtype=dtype) + 0.5) * cell_h
    bx = cx_grid.view(1, nc).expand(nr, nc)
    by = cy_grid.view(nr, 1).expand(nr, nc)

    if not macro_indices:
        return torch.zeros((), device=device, dtype=dtype)

    sizes = benchmark.macro_sizes.to(device=device, dtype=dtype)
    idx_t = torch.tensor(macro_indices, device=device, dtype=torch.long)
    centers = full_pos[idx_t]
    w = sizes[idx_t, 0]
    h = sizes[idx_t, 1]
    sx = w * 0.5
    sy = h * 0.5
    M = centers.shape[0]
    dx = centers[:, 0].view(M, 1, 1) - bx.unsqueeze(0)
    dy = centers[:, 1].view(M, 1, 1) - by.unsqueeze(0)
    sx_b = sx.view(M, 1, 1)
    sy_b = sy.view(M, 1, 1)
    mask = (dx.abs() <= 3.0 * sx_b) & (dy.abs() <= 3.0 * sy_b)
    stabil = 1e-12 if dtype == torch.float32 else 1e-24
    ex = dx * dx / (2.0 * sx_b * sx_b + stabil)
    ey = dy * dy / (2.0 * sy_b * sy_b + stabil)
    g = torch.exp(-ex - ey) * mask.to(dtype)
    rho = g.sum(dim=0)
    overflow = _soft_capacity_excess(rho - target)
    return (overflow * overflow).sum()


def _build_density_ctx(
    benchmark: Benchmark,
    macro_indices: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Per-place precomputation for ``_density_loss_fast``."""
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    nr = max(int(benchmark.grid_rows), 1)
    nc = max(int(benchmark.grid_cols), 1)
    cell_w = cw / nc
    cell_h = ch / nr

    cx_grid = (torch.arange(nc, device=device, dtype=dtype) + 0.5) * cell_w
    cy_grid = (torch.arange(nr, device=device, dtype=dtype) + 0.5) * cell_h
    bx = cx_grid.view(1, nc).expand(nr, nc).contiguous()
    by = cy_grid.view(nr, 1).expand(nr, nc).contiguous()

    if not macro_indices:
        return bx, by, None, None, None, None, None

    sizes = benchmark.macro_sizes.to(device=device, dtype=dtype)
    idx_t = torch.tensor(macro_indices, device=device, dtype=torch.long)
    sx = sizes[idx_t, 0] * 0.5
    sy = sizes[idx_t, 1] * 0.5
    M = idx_t.shape[0]
    sx_b = sx.view(M, 1, 1)
    sy_b = sy.view(M, 1, 1)
    stabil = 1e-12 if dtype == torch.float32 else 1e-24
    inv_2sx2 = 1.0 / (2.0 * sx_b * sx_b + stabil)
    inv_2sy2 = 1.0 / (2.0 * sy_b * sy_b + stabil)
    three_sx = 3.0 * sx_b
    three_sy = 3.0 * sy_b
    return bx, by, idx_t, three_sx, three_sy, inv_2sx2, inv_2sy2


def _density_loss_fast(
    full_pos: torch.Tensor,
    target: float,
    ctx: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ],
) -> torch.Tensor:
    """Equivalent to ``_vectorized_gaussian_density_loss`` with per-place constants hoisted."""
    bx, by, idx_t, three_sx, three_sy, inv_2sx2, inv_2sy2 = ctx
    if idx_t is None:
        return torch.zeros((), device=bx.device, dtype=full_pos.dtype)
    centers = full_pos[idx_t]
    M = centers.shape[0]
    dx = centers[:, 0].view(M, 1, 1) - bx.unsqueeze(0)
    dy = centers[:, 1].view(M, 1, 1) - by.unsqueeze(0)
    mask = (dx.abs() <= three_sx) & (dy.abs() <= three_sy)
    ex = dx * dx * inv_2sx2
    ey = dy * dy * inv_2sy2
    g = torch.exp(-ex - ey) * mask.to(full_pos.dtype)
    rho = g.sum(dim=0)
    overflow = _soft_capacity_excess(rho - target)
    return (overflow * overflow).sum()


def _abu_top_mean(values: torch.Tensor, frac: float) -> torch.Tensor:
    """Mean of the largest ``floor(n * frac)`` entries (PlacementCost ``abu``-style reduction)."""
    flat = values.reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return torch.zeros((), device=flat.device, dtype=flat.dtype)
    k = max(1, int(math.floor(n * frac)))
    k = min(k, n)
    return torch.topk(flat, k, largest=True).values.mean()


def _plc_style_density_from_grid(dens: torch.Tensor) -> torch.Tensor:
    """Top-10% bin-density average × 0.5 — matches ``PlacementCost.get_density_cost`` structure."""
    flat = dens.reshape(-1).clamp(min=0)
    ncells = int(flat.numel())
    if ncells == 0:
        return torch.zeros((), device=flat.device, dtype=flat.dtype)
    density_cnt = max(1, int(math.floor(ncells * 0.1)))
    k = min(density_cnt, ncells)
    top = torch.topk(flat, k, largest=True).values
    return 0.5 * top.sum() / float(density_cnt)


def _plc_macro_overlap_density_grid(
    full_pos: torch.Tensor,
    benchmark: Benchmark,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
) -> torch.Tensor:
    """Per-bin occupancy fraction from macro rectangles (differentiable), like PLC grid density."""
    device, dtype = full_pos.device, full_pos.dtype
    bw = cw / nc
    bh = ch / nr
    rs = torch.arange(nr, device=device, dtype=dtype).view(nr, 1).expand(nr, nc)
    cs = torch.arange(nc, device=device, dtype=dtype).view(1, nc).expand(nr, nc)
    bx0 = cs * bw
    bx1 = bx0 + bw
    by0 = rs * bh
    by1 = by0 + bh
    bin_area = bw * bh
    sizes = benchmark.macro_sizes.to(device=device, dtype=dtype)
    n = int(full_pos.shape[0])
    if n == 0:
        return torch.zeros(nr, nc, device=device, dtype=dtype)

    # Peak memory for full broadcast is O(M * nr * nc); chunk macros when too large.
    max_elems = 256 * 1024 * 1024
    bin_cells = nr * nc
    chunk = n
    if n * bin_cells > max_elems and bin_cells > 0:
        chunk = max(1, max_elems // bin_cells)

    bx0_b = bx0.unsqueeze(0)
    bx1_b = bx1.unsqueeze(0)
    by0_b = by0.unsqueeze(0)
    by1_b = by1.unsqueeze(0)

    chunk_results: list[torch.Tensor] = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        cx = full_pos[start:end, 0].view(-1, 1, 1)
        cy = full_pos[start:end, 1].view(-1, 1, 1)
        w = sizes[start:end, 0].view(-1, 1, 1)
        h = sizes[start:end, 1].view(-1, 1, 1)
        lx = cx - 0.5 * w
        rx = cx + 0.5 * w
        by_m = cy - 0.5 * h
        ty_m = cy + 0.5 * h
        ix0 = torch.relu(torch.minimum(rx, bx1_b) - torch.maximum(lx, bx0_b))
        iy0 = torch.relu(torch.minimum(ty_m, by1_b) - torch.maximum(by_m, by0_b))
        chunk_results.append((ix0 * iy0).sum(dim=0))
    return torch.stack(chunk_results).sum(dim=0) / bin_area


def _rudy_loss_loop(
    full_pos: torch.Tensor,
    benchmark: Benchmark,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reference (Python-loop) RUDY using LSE soft min/max per net (matches ``_rudy_loss_v2``)."""
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    nr = max(int(benchmark.grid_rows), 1)
    nc = max(int(benchmark.grid_cols), 1)
    cell_w = cw / nc
    cell_h = ch / nr

    bx0 = (torch.arange(nc, device=device, dtype=dtype) * cell_w).view(1, nc).expand(nr, nc)
    bx1 = bx0 + cell_w
    by0 = (torch.arange(nr, device=device, dtype=dtype) * cell_h).view(nr, 1).expand(nr, nc)
    by1 = by0 + cell_h

    weights = benchmark.net_weights.to(device=device, dtype=dtype)
    n_macros = int(benchmark.num_macros)
    n_ports = int(benchmark.port_positions.shape[0])
    ports = benchmark.port_positions.to(device=device, dtype=dtype)

    demand = torch.zeros(nr, nc, device=device, dtype=dtype)
    eps_area = torch.tensor(1e-18, device=device, dtype=dtype)

    for k in range(int(benchmark.num_nets)):
        nodes = benchmark.net_nodes[k]
        if nodes.numel() < 2:
            continue
        xs_list: list[torch.Tensor] = []
        ys_list: list[torch.Tensor] = []
        for idx in nodes.tolist():
            idx = int(idx)
            if idx < n_macros:
                xs_list.append(full_pos[idx, 0])
                ys_list.append(full_pos[idx, 1])
            else:
                pi = idx - n_macros
                if 0 <= pi < n_ports:
                    xs_list.append(ports[pi, 0])
                    ys_list.append(ports[pi, 1])
        if len(xs_list) < 2:
            continue
        xv = torch.stack(xs_list)
        yv = torch.stack(ys_list)
        lx = _soft_min(xv, beta)
        rx = _soft_max(xv, beta)
        by_n = _soft_min(yv, beta)
        ty = _soft_max(yv, beta)
        area = torch.relu(rx - lx) * torch.relu(ty - by_n) + eps_area

        ox = torch.relu(torch.minimum(rx, bx1) - torch.maximum(lx, bx0))
        oy = torch.relu(torch.minimum(ty, by1) - torch.maximum(by_n, by0))
        overlap = ox * oy
        demand = demand + weights[k] * overlap / area

    overflow = _soft_capacity_excess(demand - 1.0)
    return (overflow * overflow).sum()


def _rudy_demand_grid(
    combined_pos: torch.Tensor,
    net_idx: torch.Tensor,
    net_mask: torch.Tensor,
    net_weights: torch.Tensor,
    net_valid: torch.Tensor,
    beta: float,
    bin_x0: torch.Tensor,
    bin_x1: torch.Tensor,
    bin_y0: torch.Tensor,
    bin_y1: torch.Tensor,
) -> torch.Tensor:
    """LSE RUDY: per-net demand per routing bin (same routing as ``_rudy_loss_v2``)."""
    num_nets = net_idx.shape[0]
    nr, nc = bin_x0.shape
    estimated_mb = num_nets * nr * nc * combined_pos.element_size() / 1e6
    if estimated_mb > 4000:
        demand = torch.zeros(nr, nc, device=combined_pos.device, dtype=combined_pos.dtype)
        chunk_size = 512
        for start in range(0, num_nets, chunk_size):
            end = min(start + chunk_size, num_nets)
            demand = demand + _rudy_loss_v2_chunk(
                combined_pos,
                net_idx[start:end],
                net_mask[start:end],
                net_weights[start:end],
                net_valid[start:end],
                beta,
                bin_x0,
                bin_x1,
                bin_y0,
                bin_y1,
            )
        return demand

    all_pins = combined_pos[net_idx]
    x = all_pins[:, :, 0]
    y = all_pins[:, :, 1]

    NEG_INF = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    POS_INF = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)

    lx = (-1.0 / beta) * torch.logsumexp(
        -beta * x.masked_fill(~net_mask, POS_INF), dim=1
    )
    rx = (1.0 / beta) * torch.logsumexp(
        beta * x.masked_fill(~net_mask, NEG_INF), dim=1
    )
    by_n = (-1.0 / beta) * torch.logsumexp(
        -beta * y.masked_fill(~net_mask, POS_INF), dim=1
    )
    ty = (1.0 / beta) * torch.logsumexp(
        beta * y.masked_fill(~net_mask, NEG_INF), dim=1
    )

    lx_ = lx.view(-1, 1, 1)
    rx_ = rx.view(-1, 1, 1)
    by_ = by_n.view(-1, 1, 1)
    ty_ = ty.view(-1, 1, 1)
    bx0 = bin_x0.unsqueeze(0)
    bx1 = bin_x1.unsqueeze(0)
    by0 = bin_y0.unsqueeze(0)
    by1 = bin_y1.unsqueeze(0)

    ox = torch.relu(torch.minimum(rx_, bx1) - torch.maximum(lx_, bx0))
    oy = torch.relu(torch.minimum(ty_, by1) - torch.maximum(by_, by0))

    eps = torch.tensor(1e-18, device=combined_pos.device, dtype=combined_pos.dtype)
    area = torch.relu(rx_ - lx_) * torch.relu(ty_ - by_) + eps

    w = (net_weights * net_valid.to(net_weights.dtype)).view(-1, 1, 1)
    return (w * ox * oy / area).sum(dim=0)


def _rudy_loss_v2(
    combined_pos: torch.Tensor,
    net_idx: torch.Tensor,
    net_mask: torch.Tensor,
    net_weights: torch.Tensor,
    net_valid: torch.Tensor,
    beta: float,
    bin_x0: torch.Tensor,
    bin_x1: torch.Tensor,
    bin_y0: torch.Tensor,
    bin_y1: torch.Tensor,
) -> torch.Tensor:
    """Legacy scalar: sum of squared RUDY overflows (reference / sanity checks)."""
    demand = _rudy_demand_grid(
        combined_pos,
        net_idx,
        net_mask,
        net_weights,
        net_valid,
        beta,
        bin_x0,
        bin_x1,
        bin_y0,
        bin_y1,
    )
    overflow = _soft_capacity_excess(demand - 1.0)
    return (overflow * overflow).sum()


try:
    _wirelength_loss_compiled = torch.compile(_wirelength_loss_v2, fullgraph=True)
    _rudy_demand_grid_compiled = torch.compile(_rudy_demand_grid, fullgraph=True)
    _rudy_loss_compiled = torch.compile(_rudy_loss_v2, fullgraph=True)
    _TORCH_COMPILE_AVAILABLE = True
except Exception:
    _wirelength_loss_compiled = _wirelength_loss_v2
    _rudy_demand_grid_compiled = _rudy_demand_grid
    _rudy_loss_compiled = _rudy_loss_v2
    _TORCH_COMPILE_AVAILABLE = False


def _torch_compile_supported(device: torch.device) -> bool:
    """Allow compile warmup whenever stubs exist; ``place()`` catches runtime failures."""
    del device  # CPU and all CUDA SMs may attempt compile; old GPUs may fall back in try/except
    return bool(_TORCH_COMPILE_AVAILABLE)


def _rudy_loss_v2_chunk(
    combined_pos: torch.Tensor,
    net_idx: torch.Tensor,
    net_mask: torch.Tensor,
    net_weights: torch.Tensor,
    net_valid: torch.Tensor,
    beta: float,
    bin_x0: torch.Tensor,
    bin_x1: torch.Tensor,
    bin_y0: torch.Tensor,
    bin_y1: torch.Tensor,
) -> torch.Tensor:
    """Single chunk of RUDY demand accumulation (same LSE math as ``_rudy_loss_v2``)."""
    all_pins = combined_pos[net_idx]
    x = all_pins[:, :, 0]
    y = all_pins[:, :, 1]

    NEG_INF = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    POS_INF = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)

    lx = (-1.0 / beta) * torch.logsumexp(
        -beta * x.masked_fill(~net_mask, POS_INF), dim=1
    )
    rx = (1.0 / beta) * torch.logsumexp(
        beta * x.masked_fill(~net_mask, NEG_INF), dim=1
    )
    by_n = (-1.0 / beta) * torch.logsumexp(
        -beta * y.masked_fill(~net_mask, POS_INF), dim=1
    )
    ty = (1.0 / beta) * torch.logsumexp(
        beta * y.masked_fill(~net_mask, NEG_INF), dim=1
    )

    lx_ = lx.view(-1, 1, 1)
    rx_ = rx.view(-1, 1, 1)
    by_ = by_n.view(-1, 1, 1)
    ty_ = ty.view(-1, 1, 1)
    bx0 = bin_x0.unsqueeze(0)
    bx1 = bin_x1.unsqueeze(0)
    by0 = bin_y0.unsqueeze(0)
    by1 = bin_y1.unsqueeze(0)

    ox = torch.relu(torch.minimum(rx_, bx1) - torch.maximum(lx_, bx0))
    oy = torch.relu(torch.minimum(ty_, by1) - torch.maximum(by_, by0))

    eps = torch.tensor(1e-18, device=combined_pos.device, dtype=combined_pos.dtype)
    area = torch.relu(rx_ - lx_) * torch.relu(ty_ - by_) + eps

    w = (net_weights * net_valid.to(net_weights.dtype)).view(-1, 1, 1)
    return (w * ox * oy / area).sum(dim=0)


def _build_clamp_ctx(
    benchmark: Benchmark,
    movable_idx_t: torch.Tensor,
    n_hard: int,
    cw: float,
    ch: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Per-place precomputation for ``_clamp_movable_fast``: per-row min/max bounds."""
    sizes = benchmark.macro_sizes.to(device=device, dtype=dtype)
    half_hw = 0.5 * sizes
    canvas = torch.tensor([cw, ch], device=device, dtype=dtype).unsqueeze(0)

    if movable_idx_t.numel() > 0:
        half_hw_hard = half_hw[:n_hard][movable_idx_t]
        hard_min = half_hw_hard
        hard_max = canvas - half_hw_hard
    else:
        hard_min = hard_max = None

    n_macros = sizes.shape[0]
    if n_hard < n_macros:
        half_hw_soft = half_hw[n_hard:]
        soft_min = half_hw_soft
        soft_max = canvas - half_hw_soft
    else:
        soft_min = soft_max = None

    return hard_min, hard_max, soft_min, soft_max


def _clamp_movable_fast(
    pos_hard: torch.Tensor,
    pos_soft: torch.Tensor,
    ctx: tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ],
) -> None:
    """Vectorized in-place clamp using precomputed per-row bounds."""
    hard_min, hard_max, soft_min, soft_max = ctx
    if pos_hard.numel() > 0 and hard_min is not None:
        pos_hard.data.clamp_(min=hard_min, max=hard_max)
    if pos_soft.numel() > 0 and soft_min is not None:
        pos_soft.data.clamp_(min=soft_min, max=soft_max)


def _clamp_movable(
    pos_hard: torch.Tensor,
    pos_soft: torch.Tensor,
    benchmark: Benchmark,
    cw: float,
    ch: float,
) -> None:
    """In-place clamp of centers to canvas given per-macro sizes (movable hard + soft)."""
    sizes = benchmark.macro_sizes
    fixed = benchmark.macro_fixed
    n_hard = int(benchmark.num_hard_macros)
    mh = 0
    for i in range(n_hard):
        if bool(fixed[i].item()):
            continue
        w = float(sizes[i, 0].item()) * 0.5
        h = float(sizes[i, 1].item()) * 0.5
        pos_hard.data[mh, 0].clamp_(w, cw - w)
        pos_hard.data[mh, 1].clamp_(h, ch - h)
        mh += 1
    for j in range(pos_soft.shape[0]):
        gi = n_hard + j
        w = float(sizes[gi, 0].item()) * 0.5
        h = float(sizes[gi, 1].item()) * 0.5
        pos_soft.data[j, 0].clamp_(w, cw - w)
        pos_soft.data[j, 1].clamp_(h, ch - h)


def randomize_movable_macro_centers(benchmark: Benchmark, *, seed: int | None = None) -> None:
    """Uniform random center for each non-fixed macro inside its legal canvas box.

    Mutates ``benchmark.macro_positions`` in place (same bounds as training clamp).
    Call **before** ``GradientPlacer.place`` — e.g. when probing recovery from a bad
    initial layout. If you use ``macro_place.evaluate.evaluate_benchmark``, load the
    benchmark, call this, then run evaluation so ``proxy_cost_initial`` matches the
    randomized state.
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    pos = benchmark.macro_positions
    sizes = benchmark.macro_sizes
    fixed = benchmark.macro_fixed
    dev, dt = pos.device, pos.dtype
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n = int(benchmark.num_macros)
    for i in range(n):
        if bool(fixed[i].item()):
            continue
        w = float(sizes[i, 0].item()) * 0.5
        h = float(sizes[i, 1].item()) * 0.5
        span_x = max(cw - 2.0 * w, 1e-9)
        span_y = max(ch - 2.0 * h, 1e-9)
        pos[i, 0] = w + torch.rand((), device=dev, dtype=dt) * span_x
        pos[i, 1] = h + torch.rand((), device=dev, dtype=dt) * span_y


class GradientPlacer:
    """
    Differentiable placement (Adam + smooth losses), then optional ``QPLegalizer``.

    Optimization uses ``float32``; QP / diagnostics use ``float64``. Device is chosen
    heuristically so small benchmarks stay on CPU.

    Stopping: ``epochs`` / optional ``max_wall_seconds``; ICCAD04 benchmarks also support
    periodic ``compute_proxy_cost`` with ``proxy_patience``; optional flat-loss guard
    (``loss_flat_*``). If PLC loads and ``proxy_eval_interval > 0``, the tensor returned is
    the **best proxy** placement seen at checkpoints — not necessarily the last epoch.
    With ``proxy_eval_interval=0``, no checkpoint proxy runs and the **final** placement is
    returned (cheaper but long runs can look worse on proxy even if they improved earlier).

    Optional ``epoch_timing_diagnostic`` path: per-epoch CSV + summary (CUDA-synchronized
    segment times in seconds) for profiling assemble / surrogates / backward / optimizer / proxy.

    Learning rate: by default ``CosineAnnealingWarmRestarts`` on Adam (``T_0=250``, ``T_mult=1``,
    ``eta_min=0``) so LR periodically returns toward the base rate after cosine decay; disable with
    ``use_cosine_restarts=False``.

    Surrogate terms match PlacementCost **structure** (normalized WL, top-10% density,
    top-5% congestion via PLC-aligned routing surrogate by default). Default coefficients match ``compute_proxy_cost``:
    ``w_wl=1.0``, ``w_dens_hard=0.5``, ``w_cong=0.5``. Optional ``proxy_adaptive_weights``
    perturbs only WL vs congestion weights while holding density at ``w_dens_hard``.
    """

    def __init__(
        self,
        epochs: int = 10000,
        lr: float = 0.001,
        beta: float = 1.0,
        seed: int = 0,
        target_hard: float = 0.7,
        target_soft: float = 0.8,
        # Same coefficients as proxy: 1.0*WL + 0.5*density + 0.5*congestion
        w_wl: float = 1.0,
        w_dens_hard: float = 0.5,
        w_dens_soft: float = 0.0,
        w_cong: float = 0.5,
        qp_legalize: bool = False,
        # Stopping: max epochs remains ``epochs``. Optional wall clock, proxy patience,
        # and (secondary) flat surrogate loss after ``loss_flat_min_epoch``.
        max_wall_seconds: float | None = None,
        proxy_eval_interval: int = 50,
        proxy_patience: int = 5,
        proxy_adaptive_weights: bool = False,
        proxy_adapt_step: float = 0.05,
        proxy_adapt_step_min: float = 0.01,
        proxy_adapt_use_subcost_delta: bool = True,
        loss_flat_window: int = 100,
        loss_flat_rel_eps: float = 1e-5,
        loss_flat_min_epoch: int = 500,
        training_log_csv: str | Path | None = "logs.txt",
        training_log_plot: str | Path | None = "logs_plot.png",
        use_plc_routing_cong: bool = True,
        epoch_timing_diagnostic: str | Path | None = None,
        use_cosine_restarts: bool = True,
        lr_cosine_T_0: int = 250,
        lr_cosine_T_mult: int = 1,
        lr_cosine_eta_min: float = 0.0,
    ):
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.beta = float(beta)
        self.seed = int(seed)
        self.target_hard = float(target_hard)
        self.target_soft = float(target_soft)
        self.w_wl = float(w_wl)
        self.w_dens_hard = float(w_dens_hard)
        self.w_dens_soft = float(w_dens_soft)
        self.w_cong = float(w_cong)
        self.qp_legalize = bool(qp_legalize)
        self.max_wall_seconds = (
            None if max_wall_seconds is None else float(max_wall_seconds)
        )
        self.proxy_eval_interval = int(proxy_eval_interval)
        self.proxy_patience = int(proxy_patience)
        self.proxy_adaptive_weights = bool(proxy_adaptive_weights)
        self.proxy_adapt_step = float(proxy_adapt_step)
        self.proxy_adapt_step_min = float(proxy_adapt_step_min)
        self.proxy_adapt_use_subcost_delta = bool(proxy_adapt_use_subcost_delta)
        self.loss_flat_window = int(loss_flat_window)
        self.loss_flat_rel_eps = float(loss_flat_rel_eps)
        self.loss_flat_min_epoch = int(loss_flat_min_epoch)
        self.training_log_csv = training_log_csv
        self.training_log_plot = training_log_plot
        self.use_plc_routing_cong = bool(use_plc_routing_cong)
        self.epoch_timing_diagnostic = epoch_timing_diagnostic
        self.use_cosine_restarts = bool(use_cosine_restarts)
        self.lr_cosine_T_0 = int(lr_cosine_T_0)
        self.lr_cosine_T_mult = int(lr_cosine_T_mult)
        self.lr_cosine_eta_min = float(lr_cosine_eta_min)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        device = _select_device(benchmark)
        grad_dtype = torch.float32
        qp_dtype = torch.float64

        n_hard = int(benchmark.num_hard_macros)
        n_macros = int(benchmark.num_macros)
        n_soft = int(benchmark.num_soft_macros)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        fixed = benchmark.macro_fixed

        movable_hard_idx = [i for i in range(n_hard) if not bool(fixed[i].item())]
        orig = benchmark.macro_positions.to(device=device, dtype=grad_dtype)

        if not movable_hard_idx and n_soft == 0:
            return benchmark.macro_positions.clone()

        if movable_hard_idx:
            pos_hard = nn.Parameter(
                torch.stack([orig[i].clone() for i in movable_hard_idx])
            )
        else:
            pos_hard = nn.Parameter(torch.zeros(0, 2, device=device, dtype=grad_dtype))
        if n_soft > 0:
            pos_soft = nn.Parameter(orig[n_hard:n_macros].clone())
        else:
            pos_soft = nn.Parameter(torch.zeros(0, 2, device=device, dtype=grad_dtype))

        params: list[nn.Parameter] = []
        if pos_hard.numel() > 0:
            params.append(pos_hard)
        if pos_soft.numel() > 0:
            params.append(pos_soft)
        if not params:
            return benchmark.macro_positions.clone()
        opt = torch.optim.Adam(params, lr=self.lr)
        lr_scheduler: CosineAnnealingWarmRestarts | None = None
        if self.use_cosine_restarts and self.lr_cosine_T_0 > 0:
            lr_scheduler = CosineAnnealingWarmRestarts(
                opt,
                T_0=self.lr_cosine_T_0,
                T_mult=self.lr_cosine_T_mult,
                eta_min=self.lr_cosine_eta_min,
            )

        w_wl_dyn = float(self.w_wl)
        w_dh_dyn = float(self.w_dens_hard)
        w_ds_dyn = float(self.w_dens_soft)
        w_c_dyn = float(self.w_cong)

        ports = benchmark.port_positions.to(device=device, dtype=grad_dtype)
        net_idx, net_mask, net_weights_norm, net_valid = _build_net_tensors(
            benchmark, device, grad_dtype
        )
        net_weights_raw = benchmark.net_weights.to(device=device, dtype=grad_dtype)

        nr = max(int(benchmark.grid_rows), 1)
        nc = max(int(benchmark.grid_cols), 1)
        cell_w = cw / nc
        cell_h = ch / nr
        bin_x0 = (
            (torch.arange(nc, device=device, dtype=grad_dtype) * cell_w)
            .view(1, nc)
            .expand(nr, nc)
            .contiguous()
        )
        bin_x1 = bin_x0 + cell_w
        bin_y0 = (
            (torch.arange(nr, device=device, dtype=grad_dtype) * cell_h)
            .view(nr, 1)
            .expand(nr, nc)
            .contiguous()
        )
        bin_y1 = bin_y0 + cell_h

        assemble_ctx = _build_assemble_ctx(benchmark, device, grad_dtype)
        clamp_ctx = _build_clamp_ctx(
            benchmark, assemble_ctx[0], n_hard, cw, ch, device, grad_dtype
        )
        # Match ``PlacementCost.get_cost()``: HPWL sum / ((W+H) * net_cnt). We use the
        # PyTorch net count from ``_build_net_tensors`` (may differ slightly from PLC).
        num_nets_train = max(int(net_idx.shape[0]), 1)
        wl_norm = (cw + ch) * float(num_nets_train)

        using_compiled = False
        if _torch_compile_supported(device):
            try:
                with torch.no_grad():
                    _full_warm = _assemble_full_fast(pos_hard, pos_soft, assemble_ctx)
                    _cp_warm = torch.cat([_full_warm, ports], dim=0)
                    _wirelength_loss_compiled(
                        _cp_warm,
                        net_idx,
                        net_mask,
                        net_weights_norm,
                        net_valid,
                        self.beta,
                    )
                    if not self.use_plc_routing_cong:
                        _rudy_loss_compiled(
                            _cp_warm,
                            net_idx,
                            net_mask,
                            net_weights_raw,
                            net_valid,
                            self.beta,
                            bin_x0,
                            bin_x1,
                            bin_y0,
                            bin_y1,
                        )
                        _rudy_demand_grid_compiled(
                            _cp_warm,
                            net_idx,
                            net_mask,
                            net_weights_raw,
                            net_valid,
                            self.beta,
                            bin_x0,
                            bin_x1,
                            bin_y0,
                            bin_y1,
                        )
                using_compiled = True
            except Exception:
                pass

        if using_compiled:
            _wl_fn = _wirelength_loss_compiled
            _rudy_demand_fn = (
                _rudy_demand_grid_compiled
                if not self.use_plc_routing_cong
                else _rudy_demand_grid
            )
        else:
            _wl_fn = _wirelength_loss_v2
            _rudy_demand_fn = _rudy_demand_grid

        loss_check_done = False

        plc = _try_load_plc_iccad04(benchmark)
        best_proxy_val = float("inf")
        best_proxy_placement: torch.Tensor | None = None
        proxy_stagnation = 0
        t_start = time.perf_counter()
        loss_hist: deque[float] | None = (
            deque(maxlen=self.loss_flat_window) if self.loss_flat_window > 0 else None
        )

        log_csv_path = _resolve_user_path(self.training_log_csv)
        log_plot_path = _resolve_user_path(self.training_log_plot)
        want_training_log = log_csv_path is not None or log_plot_path is not None
        log_epochs: list[int] = []
        log_l_wl: list[float] = []
        log_l_cong: list[float] = []
        log_l_dh: list[float] = []
        log_l_ds: list[float] = []
        log_loss: list[float] = []
        proxy_epochs: list[int] = []
        proxy_vals: list[float] = []
        log_fp = None
        proxy_adapt_state = _ProxyWeightAdaptState(step=self.proxy_adapt_step)
        sur_baseline: tuple[float, float, float] | None = None
        px_baseline: tuple[float, float, float] | None = None

        if log_csv_path is not None:
            log_fp = open(log_csv_path, "w", encoding="utf-8")
            log_fp.write(
                "epoch,l_wl,l_cong,l_dh,l_ds,loss,proxy,w_wl,w_dh,w_ds,w_cong,"
                "delta_proxy,adapt_step\n"
            )

        def _training_row(
            epoch_: int,
            lw_f: float,
            lcong_f: float,
            ldh_f: float,
            lds_f: float,
            lv: float,
            proxy_cell: str,
            ww: float,
            wdh: float,
            wds: float,
            wc: float,
            delta_proxy_cell: str,
            adapt_step_cell: str,
        ) -> None:
            if not want_training_log:
                return
            log_epochs.append(epoch_)
            log_l_wl.append(lw_f)
            log_l_cong.append(lcong_f)
            log_l_dh.append(ldh_f)
            log_l_ds.append(lds_f)
            log_loss.append(lv)
            if log_fp is not None:
                log_fp.write(
                    f"{epoch_},{lw_f:.10g},{lcong_f:.10g},{ldh_f:.10g},{lds_f:.10g},"
                    f"{lv:.10g},{proxy_cell},{ww:.10g},{wdh:.10g},{wds:.10g},{wc:.10g},"
                    f"{delta_proxy_cell},{adapt_step_cell}\n"
                )
                log_fp.flush()

        diag_path = _resolve_user_path(self.epoch_timing_diagnostic)
        diag_fp = None
        diag_rows: list[dict[str, float]] = []
        exit_reason = "completed"
        if diag_path is not None:
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            diag_fp = open(diag_path, "w", encoding="utf-8")
            diag_fp.write("# epoch_timing_diagnostic\n")
            diag_fp.write(f"# created_utc={datetime.now(timezone.utc).isoformat()}\n")
            diag_fp.write(
                f"# device={device} grad_dtype={grad_dtype} "
                f"MACRO_PLACE_DEVICE={os.environ.get('MACRO_PLACE_DEVICE', '')!r}\n"
            )
            diag_fp.write(
                f"# use_plc_routing_cong={self.use_plc_routing_cong} "
                f"using_compiled={using_compiled}\n"
            )
            diag_fp.write(
                f"# canvas_w={cw:.10g} canvas_h={ch:.10g} "
                f"num_hard_macros={n_hard} num_macros={n_macros} num_soft={n_soft}\n"
            )
            diag_fp.write(
                f"# configured_epochs={self.epochs} "
                f"proxy_eval_interval={self.proxy_eval_interval}\n"
            )
            diag_fp.write(
                "# t_zero_grad=opt.zero_grad; t_optimizer=opt.step only; "
                "column names in _EPOCH_DIAG_NUMERIC_COLS (gradient.py)\n"
            )
            diag_fp.write(
                "# all t_* are wall seconds per segment (CUDA sync per segment when enabled)\n"
            )
            diag_fp.write("epoch," + ",".join(_EPOCH_DIAG_NUMERIC_COLS) + "\n")
            diag_fp.flush()

        try:
            _sync(device)
            for epoch in range(self.epochs):
                if diag_fp is not None:
                    epoch_diag_t0 = time.perf_counter()
                    row: dict[str, float] = {
                        k: 0.0 for k in _EPOCH_DIAG_NUMERIC_COLS
                    }
                else:
                    epoch_diag_t0 = 0.0
                    row = None

                with _diag_segment(row, "t_zero_grad", device):
                    opt.zero_grad()
                with _diag_segment(row, "t_assemble", device):
                    full = _assemble_full_fast(pos_hard, pos_soft, assemble_ctx)
                with _diag_segment(row, "t_cat_ports", device):
                    combined_pos = torch.cat([full, ports], dim=0)

                if not loss_check_done and epoch == 0:
                    with _diag_segment(row, "t_epoch0_audit", device):
                        with torch.no_grad():
                            full_ref = _assemble_full(
                                pos_hard, pos_soft, benchmark, device, grad_dtype
                            )
                            if not torch.allclose(full, full_ref, rtol=1e-5, atol=1e-6):
                                raise AssertionError("assemble_full_fast mismatch")
                            old_wl = _wirelength_loss_loop(full, benchmark, self.beta)
                            new_wl = _wirelength_loss_v2(
                                combined_pos,
                                net_idx,
                                net_mask,
                                net_weights_norm,
                                net_valid,
                                self.beta,
                            )
                            if not torch.allclose(old_wl, new_wl, rtol=1e-4, atol=1e-6):
                                raise AssertionError(
                                    f"wirelength v2 mismatch: loop={old_wl.item()} v2={new_wl.item()}"
                                )
                            old_r = _rudy_loss_loop(
                                full, benchmark, self.beta, device, grad_dtype
                            )
                            new_r = _rudy_loss_v2(
                                combined_pos,
                                net_idx,
                                net_mask,
                                net_weights_raw,
                                net_valid,
                                self.beta,
                                bin_x0,
                                bin_x1,
                                bin_y0,
                                bin_y1,
                            )
                            if not torch.allclose(old_r, new_r, rtol=1e-4, atol=1e-6):
                                raise AssertionError(
                                    f"rudy v2 mismatch: loop={old_r.item()} v2={new_r.item()}"
                                )
                    loss_check_done = True

                with _diag_segment(row, "t_wl", device):
                    l_wl_raw = _wl_fn(
                        combined_pos,
                        net_idx,
                        net_mask,
                        net_weights_norm,
                        net_valid,
                        self.beta,
                    )
                    l_wl = l_wl_raw / wl_norm
                with _diag_segment(row, "t_overlap_grid", device):
                    dens_grid = _plc_macro_overlap_density_grid(
                        full, benchmark, cw, ch, nr, nc
                    )
                with _diag_segment(row, "t_density_scalar", device):
                    l_dh = _plc_style_density_from_grid(dens_grid)
                smooth_cong = int(benchmark.congestion_smooth_range)
                if plc is not None:
                    smooth_cong = int(plc.get_congestion_smooth_range())
                with _diag_segment(row, "t_cong", device):
                    if self.use_plc_routing_cong:
                        l_cong = plc_routing_surrogate_scalar(
                            combined_pos,
                            net_idx,
                            net_mask,
                            net_weights_raw,
                            net_valid,
                            full,
                            benchmark,
                            smooth_range=smooth_cong,
                        )
                    else:
                        demand = _rudy_demand_fn(
                            combined_pos,
                            net_idx,
                            net_mask,
                            net_weights_raw,
                            net_valid,
                            self.beta,
                            bin_x0,
                            bin_x1,
                            bin_y0,
                            bin_y1,
                        )
                        overflow = _soft_capacity_excess(demand - 1.0)
                        l_cong = _abu_top_mean(overflow, 0.05)
    
                if epoch == 0 and using_compiled and not self.use_plc_routing_cong:
                    with _diag_segment(row, "t_epoch0_compile_check", device):
                        l_wl_ref = _wirelength_loss_v2(
                            combined_pos,
                            net_idx,
                            net_mask,
                            net_weights_norm,
                            net_valid,
                            self.beta,
                        )
                        rel_wl = abs(l_wl_raw.item() - l_wl_ref.item()) / (
                            abs(l_wl_ref.item()) + 1e-8
                        )
                        assert rel_wl < 1e-3, (
                            f"Compiled wirelength mismatch: {l_wl_raw.item()} vs {l_wl_ref.item()}"
                        )
                        demand_ref = _rudy_demand_grid(
                            combined_pos,
                            net_idx,
                            net_mask,
                            net_weights_raw,
                            net_valid,
                            self.beta,
                            bin_x0,
                            bin_x1,
                            bin_y0,
                            bin_y1,
                        )
                        rel_d = torch.max(
                            torch.abs(demand - demand_ref)
                        ).item() / (torch.max(torch.abs(demand_ref)).item() + 1e-8)
                        assert rel_d < 1e-3, (
                            f"Compiled RUDY demand mismatch: max_rel={rel_d}"
                        )

                lw_f = float(l_wl.item())
                lcong_f = float(l_cong.item())
                ldh_f = float(l_dh.item())
                lds_f = 0.0
                if plc is not None and epoch == 0:
                    with _diag_segment(row, "t_epoch0_proxy_baseline", device):
                        with torch.no_grad():
                            full0 = _assemble_full_fast(pos_hard, pos_soft, assemble_ctx)
                            full_cpu0 = full0.detach().cpu().to(qp_dtype)
                            c0 = compute_proxy_cost(full_cpu0, benchmark, plc)
                    sur_baseline = (lw_f, ldh_f, lcong_f)
                    px_baseline = (
                        float(c0["wirelength_cost"]),
                        float(c0["density_cost"]),
                        float(c0["congestion_cost"]),
                    )
                wl_u = w_wl_dyn
                wdh_u = w_dh_dyn
                wds_u = w_ds_dyn
                wc_u = w_c_dyn
                with _diag_segment(row, "t_loss_scalar", device):
                    loss = wl_u * l_wl + wdh_u * l_dh + wc_u * l_cong
                    lv = float(loss.item())
                with _diag_segment(row, "t_backward", device):
                    loss.backward()
                with _diag_segment(row, "t_optimizer", device):
                    lr_before = opt.param_groups[0]["lr"]
                    opt.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                        lr_after = opt.param_groups[0]["lr"]
                        if lr_after > lr_before + 1e-9:
                            for group in opt.param_groups:
                                for p in group["params"]:
                                    st = opt.state[p]
                                    if "exp_avg" in st:
                                        st["exp_avg"].zero_()
                                    if "exp_avg_sq" in st:
                                        st["exp_avg_sq"].zero_()

                with _diag_segment(row, "t_clamp", device):
                    _clamp_movable_fast(pos_hard, pos_soft, clamp_ctx)
    
                loss_val = lv
                if loss_hist is not None:
                    loss_hist.append(loss_val)
    
                proxy_cell = ""
                delta_proxy_cell = ""
                adapt_step_cell = ""
                stop_proxy_patience = False
                if (
                    plc is not None
                    and self.proxy_eval_interval > 0
                    and (epoch + 1) % self.proxy_eval_interval == 0
                ):
                    with _diag_segment(row, "t_proxy_checkpoint", device):
                        with torch.no_grad():
                            full_ev = _assemble_full_fast(pos_hard, pos_soft, assemble_ctx)
                            full_cpu_ev = full_ev.detach().cpu().to(qp_dtype)
                            combined_ev = torch.cat([full_ev, ports], dim=0)
                            l_wl_raw_ev = _wl_fn(
                                combined_ev,
                                net_idx,
                                net_mask,
                                net_weights_norm,
                                net_valid,
                                self.beta,
                            )
                            l_wl_ev = l_wl_raw_ev / wl_norm
                            dens_grid_ev = _plc_macro_overlap_density_grid(
                                full_ev, benchmark, cw, ch, nr, nc
                            )
                            l_dh_ev = _plc_style_density_from_grid(dens_grid_ev)
                            smooth_ev = int(benchmark.congestion_smooth_range)
                            if plc is not None:
                                smooth_ev = int(plc.get_congestion_smooth_range())
                            if self.use_plc_routing_cong:
                                l_cong_ev = plc_routing_surrogate_scalar(
                                    combined_ev,
                                    net_idx,
                                    net_mask,
                                    net_weights_raw,
                                    net_valid,
                                    full_ev,
                                    benchmark,
                                    smooth_range=smooth_ev,
                                )
                            else:
                                demand_ev = _rudy_demand_fn(
                                    combined_ev,
                                    net_idx,
                                    net_mask,
                                    net_weights_raw,
                                    net_valid,
                                    self.beta,
                                    bin_x0,
                                    bin_x1,
                                    bin_y0,
                                    bin_y1,
                                )
                                overflow_ev = _soft_capacity_excess(demand_ev - 1.0)
                                l_cong_ev = _abu_top_mean(overflow_ev, 0.05)
                            sw = float(l_wl_ev.item())
                            sdh = float(l_dh_ev.item())
                            sc = float(l_cong_ev.item())
                            costs = compute_proxy_cost(full_cpu_ev, benchmark, plc)
                        pv = float(costs["proxy_cost"])
                        pw = float(costs["wirelength_cost"])
                        pd = float(costs["density_cost"])
                        pc = float(costs["congestion_cost"])
                        if sur_baseline is not None and px_baseline is not None:
                            eps = 1e-30
                            n_sw = sw / max(sur_baseline[0], eps)
                            n_sdh = sdh / max(sur_baseline[1], eps)
                            n_sc = sc / max(sur_baseline[2], eps)
                            n_pw = pw / max(px_baseline[0], eps)
                            n_pd = pd / max(px_baseline[1], eps)
                            n_pc = pc / max(px_baseline[2], eps)
                            gap_wl = n_sw - n_pw
                            gap_dh = n_sdh - n_pd
                            gap_c = n_sc - n_pc
                            abs_gaps = (abs(gap_wl), abs(gap_dh), abs(gap_c))
                            max_idx = max(range(3), key=lambda i: abs_gaps[i])
                            max_names = ("wl", "density", "congestion")
                            _svp = (
                                f"[surrogate_vs_proxy] epoch={epoch} "
                                f"sur_raw(wl,dh,cong)=({sw:.6g},{sdh:.6g},{sc:.6g}) "
                                f"px_raw(wl,den,cong)=({pw:.6g},{pd:.6g},{pc:.6g}) "
                                f"pv={pv:.6g} "
                                f"sur_n={n_sw:.4f}/{n_sdh:.4f}/{n_sc:.4f} "
                                f"px_n={n_pw:.4f}/{n_pd:.4f}/{n_pc:.4f} "
                                f"gap_n(sur-px)={gap_wl:+.4f}/{gap_dh:+.4f}/{gap_c:+.4f} "
                                f"max_abs_gap={max_names[max_idx]}({abs_gaps[max_idx]:.4f})"
                            )
                            print(_svp, flush=True)
                            if log_fp is not None:
                                log_fp.write(_svp + "\n")
                                log_fp.flush()
                        else:
                            _svp = (
                                f"[surrogate_vs_proxy] epoch={epoch} "
                                f"sur_raw(wl,dh,cong)=({sw:.6g},{sdh:.6g},{sc:.6g}) "
                                f"px_raw(wl,den,cong)=({pw:.6g},{pd:.6g},{pc:.6g}) "
                                f"pv={pv:.6g} (no epoch-0 baseline; skipped norm/gap)"
                            )
                            print(_svp, flush=True)
                            if log_fp is not None:
                                log_fp.write(_svp + "\n")
                                log_fp.flush()
                        if self.proxy_adaptive_weights:
                            w_wl_dyn, w_dh_dyn, w_ds_dyn, w_c_dyn, d_pv_log = (
                                _proxy_guided_adapt_surrogate_weights(
                                    w_wl_dyn,
                                    w_dh_dyn,
                                    w_ds_dyn,
                                    w_c_dyn,
                                    pv,
                                    pw,
                                    pd,
                                    pc,
                                    proxy_adapt_state,
                                    step_min=self.proxy_adapt_step_min,
                                    use_subcost_bias=self.proxy_adapt_use_subcost_delta,
                                )
                            )
                            adapt_step_cell = f"{proxy_adapt_state.step:.10g}"
                            if d_pv_log is not None:
                                delta_proxy_cell = f"{d_pv_log:.10g}"
                        proxy_cell = f"{pv:.10g}"
                        proxy_epochs.append(epoch)
                        proxy_vals.append(pv)
                        if pv < best_proxy_val:
                            best_proxy_val = pv
                            best_proxy_placement = full_cpu_ev.clone()
                            proxy_stagnation = 0
                        elif self.proxy_patience > 0:
                            proxy_stagnation += 1
                            if proxy_stagnation >= self.proxy_patience:
                                stop_proxy_patience = True

                _training_row(
                    epoch,
                    lw_f,
                    lcong_f,
                    ldh_f,
                    lds_f,
                    lv,
                    proxy_cell,
                    wl_u,
                    wdh_u,
                    wds_u,
                    wc_u,
                    delta_proxy_cell,
                    adapt_step_cell,
                )

                if diag_fp is not None and row is not None:
                    row["t_epoch_wall_s"] = time.perf_counter() - epoch_diag_t0
                    diag_fp.write(
                        f"{epoch},"
                        + ",".join(f"{row[k]:.10g}" for k in _EPOCH_DIAG_NUMERIC_COLS)
                        + "\n"
                    )
                    diag_fp.flush()
                    diag_rows.append(dict(row))

                if stop_proxy_patience:
                    exit_reason = "proxy_patience"
                    break

                if self.max_wall_seconds is not None:
                    if time.perf_counter() - t_start >= self.max_wall_seconds:
                        exit_reason = "wall_time"
                        break

                if (
                    loss_hist is not None
                    and epoch >= self.loss_flat_min_epoch
                    and len(loss_hist) == self.loss_flat_window
                ):
                    lo = min(loss_hist)
                    hi = max(loss_hist)
                    mean_abs = abs(sum(loss_hist) / len(loss_hist))
                    denom = max(mean_abs, 1e-12)
                    if (hi - lo) / denom < self.loss_flat_rel_eps:
                        exit_reason = "flat_loss"
                        break
    
        finally:
            if diag_fp is not None:
                _diag_write_footer(
                    diag_fp,
                    diag_rows,
                    exit_reason=exit_reason,
                    configured_epochs=self.epochs,
                    total_wall_s=time.perf_counter() - t_start,
                )
                diag_fp.close()

        _sync(device)

        if log_fp is not None:
            log_fp.close()

        if log_plot_path is not None and log_epochs:
            _write_gradient_log_plot(
                log_plot_path,
                log_epochs,
                log_l_wl,
                log_l_cong,
                log_l_dh,
                log_l_ds,
                log_loss,
                proxy_epochs,
                proxy_vals,
            )

        full_final = _assemble_full_fast(pos_hard, pos_soft, assemble_ctx)
        if best_proxy_placement is not None:
            optimized_cpu = best_proxy_placement
        else:
            optimized_cpu = full_final.detach().cpu().to(qp_dtype)

        if self.qp_legalize:
            saved = benchmark.macro_positions.clone()
            try:
                benchmark.macro_positions.copy_(optimized_cpu.float())
                placement_out = QPLegalizer().place(benchmark)
            finally:
                benchmark.macro_positions.copy_(saved)
        else:
            placement_out = optimized_cpu

        return placement_out


def _cli_main() -> None:
    from macro_place.loader import load_benchmark_from_dir

    root = _repo_root()
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    GradientPlacer().place(b)


if __name__ == "__main__":
    _cli_main()
