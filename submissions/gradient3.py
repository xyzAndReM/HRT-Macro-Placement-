"""
**gradient3** — local-refinement placer: SGD+Nesterov, β/softplus scheduling, top-percentile
hotspots, optional narrow Gaussian density, QP lookahead, surrogate–proxy correlation plot.

See module source and ``Gradient3Placer`` for parameters.

Usage:
    uv run python submissions/gradient3.py
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


def _soft_capacity_excess(excess: torch.Tensor, sharpness: float = 1.0) -> torch.Tensor:
    """Scaled softplus: ``softplus(s·x)/s`` — larger ``sharpness`` approaches a hard hinge."""
    s = float(sharpness)
    if s <= 0.0:
        s = 1.0
    return F.softplus(s * excess) / s


def _sched_exponential_beta(epoch: int, epochs: int, beta0: float, end_ratio: float) -> float:
    """β grows exponentially from ``beta0`` to ``beta0 * end_ratio`` over training."""
    if epochs <= 1:
        return float(beta0)
    t = float(epoch) / float(epochs - 1)
    return float(beta0) * (float(end_ratio) ** t)


def _sched_sigma_local(epoch: int, epochs: int, sigma0: float, sigma1: float) -> float:
    """Linear schedule for Gaussian σ multipliers (local narrowing over time)."""
    if epochs <= 1:
        return float(sigma0)
    t = float(epoch) / float(epochs - 1)
    return float(sigma0) + t * (float(sigma1) - float(sigma0))


# Float32-safe L-route area denominator inside RUDY (avoid NaNs without biasing WL).
_RUDY_AREA_EPS = 1e-6


def _macro_connectivity_row_mult(
    benchmark: Benchmark,
    macro_indices: list[int],
    device: torch.device,
    dtype: torch.dtype,
    boost: float,
) -> torch.Tensor | None:
    """Per-macro σ multipliers ``1 + boost * norm_degree`` for Gaussian density kernels."""
    if boost <= 0.0 or not macro_indices:
        return None
    M = len(macro_indices)
    g2l = {int(g): i for i, g in enumerate(macro_indices)}
    counts = [0] * M
    for k in range(int(benchmark.num_nets)):
        nodes = benchmark.net_nodes[k]
        for t in range(int(nodes.numel())):
            gi = int(nodes[t].item())
            j = g2l.get(gi)
            if j is not None:
                counts[j] += 1
    c = torch.tensor(counts, device=device, dtype=dtype)
    cmax = float(c.max().item()) if M > 0 else 1.0
    nrm = c / (float(cmax) + 1e-6)
    return 1.0 + float(boost) * nrm


def _rudy_nets_active_mask(
    combined_pos: torch.Tensor,
    net_idx: torch.Tensor,
    net_mask: torch.Tensor,
    dens_grid: torch.Tensor,
    hotspot_frac: float,
    cw: float,
    ch: float,
) -> torch.Tensor:
    """Nets with at least one pin in a top-``hotspot_frac`` density bin (cheap congestion focus)."""
    nr, nc = dens_grid.shape
    device = combined_pos.device
    flat = dens_grid.reshape(-1)
    n = int(flat.numel())
    num_nets = int(net_idx.shape[0])
    if n == 0:
        return torch.ones(num_nets, device=device, dtype=torch.bool)
    hf = min(max(float(hotspot_frac), 1e-6), 1.0)
    k = max(1, int(math.floor(n * hf)))
    thr = torch.topk(flat, min(k, n), largest=True).values.min()
    hot = dens_grid >= thr
    cell_w = cw / float(nc)
    cell_h = ch / float(nr)
    pi = net_idx.clamp(min=0)
    cx = combined_pos[pi, 0]
    cy = combined_pos[pi, 1]
    ci = (cx / cell_w).long().clamp(0, nc - 1)
    ri = (cy / cell_h).long().clamp(0, nr - 1)
    touch = hot[ri, ci] & net_mask
    active = touch.any(dim=1)
    if not bool(active.any()):
        return torch.ones(num_nets, device=device, dtype=torch.bool)
    return active


def _build_pin_gather_tensors(
    benchmark: Benchmark,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Indices into ``combined_pos`` and macro-local offsets for every pin (see ``Benchmark.net_pin_nodes``)."""
    n_m = int(benchmark.num_macros)
    n_ports = int(benchmark.port_positions.shape[0])
    owners: list[int] = []
    offs: list[list[float]] = []

    if len(benchmark.net_pin_nodes) == int(benchmark.num_nets):
        for k in range(int(benchmark.num_nets)):
            pn = benchmark.net_pin_nodes[k]
            for r in range(int(pn.shape[0])):
                o = int(pn[r, 0].item())
                s = int(pn[r, 1].item())
                owners.append(o)
                if o < n_m:
                    mo_list = benchmark.macro_pin_offsets
                    if (
                        o < len(mo_list)
                        and mo_list[o].numel() > 0
                        and s < mo_list[o].shape[0]
                    ):
                        off = mo_list[o][s].to(dtype=torch.float32)
                        offs.append([float(off[0].item()), float(off[1].item())])
                    else:
                        offs.append([0.0, 0.0])
                else:
                    offs.append([0.0, 0.0])
        if not owners:
            return None
        pin_owner = torch.tensor(owners, device=device, dtype=torch.long)
        pin_off = torch.tensor(offs, device=device, dtype=dtype)
        return pin_owner, pin_off

    owners = list(range(n_m)) + [n_m + j for j in range(n_ports)]
    pin_owner = torch.tensor(owners, device=device, dtype=torch.long)
    pin_off = torch.zeros(len(owners), 2, device=device, dtype=dtype)
    return pin_owner, pin_off


def _compute_pin_demand_grid(
    combined_pos: torch.Tensor,
    pin_owner: torch.Tensor,
    pin_off: torch.Tensor,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
    sigma_pin: float,
    max_elems: int = 256 * 1024 * 1024,
) -> torch.Tensor:
    """Gaussian pin-density map on an ``nr``×``nc`` canvas (same orientation as RUDY bins)."""
    device = combined_pos.device
    dtype = combined_pos.dtype
    rs = torch.arange(nr, device=device, dtype=dtype).view(nr, 1).expand(nr, nc)
    cs = torch.arange(nc, device=device, dtype=dtype).view(1, nc).expand(nr, nc)
    cx_bin = (cs + 0.5) * (cw / float(nc))
    cy_bin = (rs + 0.5) * (ch / float(nr))
    pin_pos = combined_pos[pin_owner] + pin_off
    pcount = int(pin_pos.shape[0])
    s2 = 2.0 * (float(sigma_pin) ** 2)
    if s2 <= 1e-18:
        s2 = 1e-6
    px = pin_pos[:, 0]
    py = pin_pos[:, 1]
    bin_cells = nr * nc
    chunk = pcount
    if pcount * bin_cells > max_elems and bin_cells > 0:
        chunk = max(1, max_elems // bin_cells)
    acc = torch.zeros(nr, nc, device=device, dtype=dtype)
    for start in range(0, pcount, chunk):
        end = min(start + chunk, pcount)
        dx = px[start:end].view(-1, 1, 1) - cx_bin.unsqueeze(0)
        dy = py[start:end].view(-1, 1, 1) - cy_bin.unsqueeze(0)
        acc = acc + torch.exp(-(dx * dx + dy * dy) / s2).sum(dim=0)
    return acc


def _refinement_rudy_pin_cong_loss(
    rudy_demand: torch.Tensor,
    pin_demand: torch.Tensor,
    *,
    pin_weight: float,
    util_threshold: float,
    overflow_sharpness: float,
    top_k_frac: float,
) -> torch.Tensor:
    """RUDY + normalized pin pressure, soft overflow above ``util_threshold``, ABU top-mean."""
    pin_n = pin_demand / (pin_demand.amax() + 1e-8)
    total = rudy_demand + float(pin_weight) * pin_n
    overflow = _soft_capacity_excess(
        total - float(util_threshold), sharpness=float(overflow_sharpness)
    )
    return _abu_top_mean(overflow, float(top_k_frac))


def _nesterov_momentum_restart(optimizer: torch.optim.Optimizer) -> None:
    """O'Donoghue–Candes style: zero momentum when it disagrees with the new gradient."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            st = optimizer.state.get(p)
            if st is None or "momentum_buffer" not in st:
                continue
            buf = st["momentum_buffer"]
            g = p.grad.detach()
            if torch.dot(buf.flatten(), g.flatten()) < 0:
                buf.zero_()


def _zero_sgd_momentum(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p)
            if st is None or "momentum_buffer" not in st:
                continue
            st["momentum_buffer"].zero_()


def _scatter_legal_placement_into_params(
    placement_cpu: torch.Tensor,
    pos_hard: nn.Parameter,
    pos_soft: nn.Parameter,
    movable_hard_idx: list[int],
    n_hard: int,
    n_macros: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Load CPU placement tensor into ``pos_hard`` / ``pos_soft`` (movable rows only)."""
    ph = placement_cpu.to(device=device, dtype=dtype)
    with torch.no_grad():
        for r, gi in enumerate(movable_hard_idx):
            pos_hard[r].copy_(ph[gi])
        if pos_soft.numel() > 0:
            pos_soft.copy_(ph[n_hard:n_macros])


def _write_surrogate_proxy_corr_plot(
    plot_path: Path,
    surrogate_vals: list[float],
    proxy_vals: list[float],
) -> None:
    """Scatter surrogate total loss vs evaluator proxy at checkpoints."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if len(surrogate_vals) < 2 or len(proxy_vals) < 2:
        return
    if len(surrogate_vals) != len(proxy_vals):
        return
    s = np.asarray(surrogate_vals, dtype=np.float64)
    p = np.asarray(proxy_vals, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(s, p, s=14, alpha=0.75)
    ax.set_xlabel("weighted surrogate at checkpoint")
    ax.set_ylabel("evaluator proxy")
    ax.set_title("Surrogate vs proxy (checkpoints)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


# Per-epoch diagnostic CSV columns (after ``epoch``); see ``Gradient3Placer.epoch_timing_diagnostic``.
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
    "t_qp_lookahead",
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

    fig.suptitle("Gradient3Placer training log")
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
    sigma_scale: float = 1.0,
    connectivity_row_mult: torch.Tensor | None = None,
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
    sig = float(sigma_scale)
    if sig <= 0.0:
        sig = 1.0
    sx = sizes[idx_t, 0] * 0.5 * sig
    sy = sizes[idx_t, 1] * 0.5 * sig
    if connectivity_row_mult is not None:
        m = connectivity_row_mult.to(device=device, dtype=dtype).view(-1)
        if m.shape[0] == sx.shape[0]:
            sx = sx * m
            sy = sy * m
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
    sharpness: float = 1.0,
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
    overflow = _soft_capacity_excess(rho - target, sharpness=sharpness)
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


def _plc_style_density_from_grid(
    dens: torch.Tensor, top_frac: float = 0.1
) -> torch.Tensor:
    """Top-``top_frac`` bin-density average × 0.5 — ``PlacementCost.get_density_cost``-like."""
    flat = dens.reshape(-1).clamp(min=0)
    ncells = int(flat.numel())
    if ncells == 0:
        return torch.zeros((), device=flat.device, dtype=flat.dtype)
    tf = min(max(float(top_frac), 1e-6), 1.0)
    density_cnt = max(1, int(math.floor(ncells * tf)))
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
    eps_area = torch.tensor(_RUDY_AREA_EPS, device=device, dtype=dtype)

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

    eps = torch.tensor(
        _RUDY_AREA_EPS, device=combined_pos.device, dtype=combined_pos.dtype
    )
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

    eps = torch.tensor(
        _RUDY_AREA_EPS, device=combined_pos.device, dtype=combined_pos.dtype
    )
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
    Call **before** a gradient placer's ``place`` — e.g. when probing recovery from a bad
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


class Gradient3Placer:
    """
    Local-refinement placement: **SGD + Nesterov** (optional momentum restart), exponential
    ``β`` schedule for LSE / softplus surrogates, tighter **top-percentile** reductions,
    optional **narrow Gaussian** density kernels, **RUDY + pin-demand fusion** (non-PLC) on
    a matching or finer grid, periodic **QP legalization lookahead**,
    optional **routing-grid doubling**, surrogate/proxy correlation plot. Wirelength and RUDY
    use ``torch.compile`` subgraphs when supported (PLC routing path is not compiled; pin
    fusion disables RUDY compile).

    Surrogate coefficients default like ``compute_proxy_cost``: ``w_wl=1.0``, ``w_dens_hard=0.5``,
    ``w_cong=0.5``. Returns **best proxy** placement when PLC checkpoints run
    (``proxy_eval_interval > 0``).
    """

    def __init__(
        self,
        epochs: int = 10000,
        lr_pos_hard: float = 0.04,
        lr_pos_soft: float = 0.012,
        momentum: float = 0.9,
        nesterov: bool = True,
        momentum_restart: bool = True,
        beta0: float = 1.0,
        beta_end_factor: float = 10.0,
        sigma0: float = 1.0,
        sigma1: float = 0.25,
        sigma_floor: float = 0.25,
        connectivity_density_boost: float = 0.0,
        use_local_gaussian_density: bool = False,
        top_frac_density: float = 0.1,
        top_frac_cong: float = 0.05,
        seed: int = 0,
        target_hard: float = 0.7,
        target_soft: float = 0.8,
        w_wl: float = 1.0,
        w_dens_hard: float = 0.5,
        w_dens_soft: float = 0.0,
        w_cong: float = 0.5,
        qp_legalize: bool = False,
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
        surrogate_proxy_plot: str | Path | None = "surrogate_proxy_corr.png",
        use_plc_routing_cong: bool = True,
        epoch_timing_diagnostic: str | Path | None = None,
        qp_lookahead_interval: int = 0,
        grid_doubling: bool = False,
        proxy_anchor_interval: int = 0,
        proxy_anchor_revert_ratio: float = 1.05,
        rudy_hotspot_net_frac: float | None = 0.12,
        surrogate_beta_boost_on_divergence: float = 1.15,
        surrogate_beta_boost_max: float = 25.0,
        pin_rudy_fusion: bool = True,
        pin_demand_sigma_um: float = 3.0,
        pin_demand_weight: float = 0.5,
        pin_util_threshold: float = 0.7,
        pin_overflow_sharpness: float = 20.0,
        pin_grid_rows: int | None = None,
        pin_grid_cols: int | None = None,
        pin_sigma_max_um: float = 80.0,
        proxy_anchor_tune_pins: bool = True,
        congestion_proxy_anchor_interval: int = 20,
    ):
        self.epochs = int(epochs)
        self.lr_pos_hard = float(lr_pos_hard)
        self.lr_pos_soft = float(lr_pos_soft)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.momentum_restart = bool(momentum_restart)
        self.beta0 = float(beta0)
        self.beta_end_factor = float(beta_end_factor)
        self.sigma0 = float(sigma0)
        self.sigma1 = float(sigma1)
        self.sigma_floor = float(sigma_floor)
        self.connectivity_density_boost = float(connectivity_density_boost)
        self.use_local_gaussian_density = bool(use_local_gaussian_density)
        tf_d = float(top_frac_density)
        self.top_frac_density = min(max(tf_d, 1e-6), 1.0)
        tf_c = float(top_frac_cong)
        self.top_frac_cong = min(max(tf_c, 1e-6), 1.0)
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
        self.surrogate_proxy_plot = surrogate_proxy_plot
        self.use_plc_routing_cong = bool(use_plc_routing_cong)
        self.epoch_timing_diagnostic = epoch_timing_diagnostic
        self.qp_lookahead_interval = int(qp_lookahead_interval)
        self.grid_doubling = bool(grid_doubling)
        self.proxy_anchor_interval = int(proxy_anchor_interval)
        self.proxy_anchor_revert_ratio = float(proxy_anchor_revert_ratio)
        self.rudy_hotspot_net_frac = rudy_hotspot_net_frac
        self.surrogate_beta_boost_on_divergence = float(
            surrogate_beta_boost_on_divergence
        )
        self.surrogate_beta_boost_max = float(surrogate_beta_boost_max)
        self.pin_rudy_fusion = bool(pin_rudy_fusion)
        self.pin_demand_sigma_um = float(pin_demand_sigma_um)
        self.pin_demand_weight = float(pin_demand_weight)
        self.pin_util_threshold = float(pin_util_threshold)
        self.pin_overflow_sharpness = float(pin_overflow_sharpness)
        self.pin_grid_rows = pin_grid_rows
        self.pin_grid_cols = pin_grid_cols
        self.pin_sigma_max_um = float(pin_sigma_max_um)
        self.proxy_anchor_tune_pins = bool(proxy_anchor_tune_pins)
        self.congestion_proxy_anchor_interval = int(congestion_proxy_anchor_interval)

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

        param_groups: list[dict] = []
        if pos_hard.numel() > 0:
            param_groups.append(
                {
                    "params": [pos_hard],
                    "lr": self.lr_pos_hard,
                    "momentum": self.momentum,
                    "nesterov": self.nesterov,
                }
            )
        if pos_soft.numel() > 0:
            param_groups.append(
                {
                    "params": [pos_soft],
                    "lr": self.lr_pos_soft,
                    "momentum": self.momentum,
                    "nesterov": self.nesterov,
                }
            )
        opt = torch.optim.SGD(param_groups)

        nr_base = max(int(benchmark.grid_rows), 1)
        nc_base = max(int(benchmark.grid_cols), 1)
        nr_sur = nr_base
        nc_sur = nc_base
        grid_doubled_yet = False
        density_macro_idx = list(range(n_macros))

        connectivity_row_mult = _macro_connectivity_row_mult(
            benchmark,
            density_macro_idx,
            device,
            grad_dtype,
            self.connectivity_density_boost,
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
        pin_gather = _build_pin_gather_tensors(benchmark, device, grad_dtype)

        def _rebuild_rudy_bins(
            nrr: int, ncc: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            cell_w_ = cw / float(ncc)
            cell_h_ = ch / float(nrr)
            bx0 = (
                (torch.arange(ncc, device=device, dtype=grad_dtype) * cell_w_)
                .view(1, ncc)
                .expand(nrr, ncc)
                .contiguous()
            )
            bx1 = bx0 + cell_w_
            by0 = (
                (torch.arange(nrr, device=device, dtype=grad_dtype) * cell_h_)
                .view(nrr, 1)
                .expand(nrr, ncc)
                .contiguous()
            )
            by1 = by0 + cell_h_
            return bx0, bx1, by0, by1

        bin_x0, bin_x1, bin_y0, bin_y1 = _rebuild_rudy_bins(nr_sur, nc_sur)

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
                        self.beta0,
                    )
                    if not self.use_plc_routing_cong:
                        _rudy_loss_compiled(
                            _cp_warm,
                            net_idx,
                            net_mask,
                            net_weights_raw,
                            net_valid,
                            self.beta0,
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
                            self.beta0,
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

        if (
            not self.use_plc_routing_cong
            and self.rudy_hotspot_net_frac is not None
        ):
            using_compiled = False
            _rudy_demand_fn = _rudy_demand_grid

        if (
            not self.use_plc_routing_cong
            and self.pin_rudy_fusion
            and pin_gather is not None
        ):
            using_compiled = False
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
        corr_sur: list[float] = []
        corr_px: list[float] = []
        log_fp = None
        proxy_adapt_state = _ProxyWeightAdaptState(step=self.proxy_adapt_step)
        sur_baseline: tuple[float, float, float] | None = None
        px_baseline: tuple[float, float, float] | None = None

        beta_end_dynamic = 1.0
        prev_ck_sur: float | None = None
        prev_ck_px: float | None = None

        anchor_best_proxy = float("inf")
        anchor_pos_hard: torch.Tensor | None = None
        anchor_pos_soft: torch.Tensor | None = None

        sigma_pin_um = float(self.pin_demand_sigma_um)
        top_frac_cong_dyn = float(self.top_frac_cong)
        anchor_prev_proxy_tune: float | None = None
        anchor_prev_l_cong_tune: float | None = None

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
                "column names in _EPOCH_DIAG_NUMERIC_COLS (gradient3.py)\n"
            )
            diag_fp.write(
                "# all t_* are wall seconds per segment (CUDA sync per segment when enabled)\n"
            )
            diag_fp.write("epoch," + ",".join(_EPOCH_DIAG_NUMERIC_COLS) + "\n")
            diag_fp.flush()

        try:
            _sync(device)
            congestion_px_anchor_bias = 0.0
            for epoch in range(self.epochs):
                if diag_fp is not None:
                    epoch_diag_t0 = time.perf_counter()
                    row: dict[str, float] = {
                        k: 0.0 for k in _EPOCH_DIAG_NUMERIC_COLS
                    }
                else:
                    epoch_diag_t0 = 0.0
                    row = None

                beta_eff = _sched_exponential_beta(
                    epoch,
                    self.epochs,
                    self.beta0,
                    self.beta_end_factor * beta_end_dynamic,
                )
                sigma_scale = max(
                    _sched_sigma_local(
                        epoch, self.epochs, self.sigma0, self.sigma1
                    ),
                    self.sigma_floor,
                )
                den_ctx = (
                    _build_density_ctx(
                        benchmark,
                        density_macro_idx,
                        device,
                        grad_dtype,
                        sigma_scale,
                        connectivity_row_mult
                        if self.use_local_gaussian_density
                        else None,
                    )
                    if self.use_local_gaussian_density
                    else None
                )

                if (
                    self.grid_doubling
                    and (not grid_doubled_yet)
                    and self.epochs > 1
                    and epoch == max(1, self.epochs // 2)
                ):
                    nr_sur = nr_base * 2
                    nc_sur = nc_base * 2
                    bin_x0, bin_x1, bin_y0, bin_y1 = _rebuild_rudy_bins(nr_sur, nc_sur)
                    using_compiled = False
                    _wl_fn = _wirelength_loss_v2
                    _rudy_demand_fn = _rudy_demand_grid
                    grid_doubled_yet = True

                rout_nr = nr_sur if self.grid_doubling and grid_doubled_yet else None
                rout_nc = nc_sur if self.grid_doubling and grid_doubled_yet else None

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
                            old_wl = _wirelength_loss_loop(full, benchmark, beta_eff)
                            new_wl = _wirelength_loss_v2(
                                combined_pos,
                                net_idx,
                                net_mask,
                                net_weights_norm,
                                net_valid,
                                beta_eff,
                            )
                            if not torch.allclose(old_wl, new_wl, rtol=1e-4, atol=1e-6):
                                raise AssertionError(
                                    f"wirelength v2 mismatch: loop={old_wl.item()} v2={new_wl.item()}"
                                )
                            old_r = _rudy_loss_loop(
                                full, benchmark, beta_eff, device, grad_dtype
                            )
                            new_r = _rudy_loss_v2(
                                combined_pos,
                                net_idx,
                                net_mask,
                                net_weights_raw,
                                net_valid,
                                beta_eff,
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
                        beta_eff,
                    )
                    l_wl = l_wl_raw / wl_norm
                with _diag_segment(row, "t_overlap_grid", device):
                    if self.use_local_gaussian_density:
                        dens_grid = None
                        if den_ctx is None or den_ctx[2] is None:
                            l_dh = torch.zeros((), device=device, dtype=grad_dtype)
                        else:
                            l_dh = _density_loss_fast(
                                full,
                                self.target_hard,
                                den_ctx,
                                sharpness=beta_eff,
                            )
                    else:
                        dens_grid = _plc_macro_overlap_density_grid(
                            full, benchmark, cw, ch, nr_sur, nc_sur
                        )
                with _diag_segment(row, "t_density_scalar", device):
                    if not self.use_local_gaussian_density:
                        l_dh = _plc_style_density_from_grid(
                            dens_grid,
                            top_frac=self.top_frac_density,
                        )
                dens_for_rudy_filter: torch.Tensor | None = None
                if not self.use_local_gaussian_density:
                    dens_for_rudy_filter = dens_grid
                elif (
                    not self.use_plc_routing_cong
                    and self.rudy_hotspot_net_frac is not None
                ):
                    with _diag_segment(row, "t_rudy_filter_grid", device):
                        dens_for_rudy_filter = _plc_macro_overlap_density_grid(
                            full, benchmark, cw, ch, nr_sur, nc_sur
                        )

                net_idx_r = net_idx
                net_mask_r = net_mask
                net_weights_raw_r = net_weights_raw
                net_weights_norm_r = net_weights_norm
                net_valid_r = net_valid
                if (
                    not self.use_plc_routing_cong
                    and self.rudy_hotspot_net_frac is not None
                    and dens_for_rudy_filter is not None
                ):
                    am = _rudy_nets_active_mask(
                        combined_pos,
                        net_idx,
                        net_mask,
                        dens_for_rudy_filter,
                        self.rudy_hotspot_net_frac,
                        cw,
                        ch,
                    )
                    net_idx_r = net_idx[am]
                    net_mask_r = net_mask[am]
                    net_weights_raw_r = net_weights_raw[am]
                    net_weights_norm_r = net_weights_norm[am]
                    net_valid_r = net_valid[am]

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
                            abu_frac=top_frac_cong_dyn,
                            nr=rout_nr,
                            nc=rout_nc,
                        )
                    else:
                        demand = _rudy_demand_fn(
                            combined_pos,
                            net_idx_r,
                            net_mask_r,
                            net_weights_raw_r,
                            net_valid_r,
                            beta_eff,
                            bin_x0,
                            bin_x1,
                            bin_y0,
                            bin_y1,
                        )
                        if (
                            self.pin_rudy_fusion
                            and pin_gather is not None
                        ):
                            nrp = self.pin_grid_rows
                            ncp = self.pin_grid_cols
                            if nrp is None:
                                nrp = nr_sur
                            if ncp is None:
                                ncp = nc_sur
                            nrp = max(int(nrp), 1)
                            ncp = max(int(ncp), 1)
                            with _diag_segment(row, "t_pin_demand", device):
                                pin_d = _compute_pin_demand_grid(
                                    combined_pos,
                                    pin_gather[0],
                                    pin_gather[1],
                                    nrp,
                                    ncp,
                                    cw,
                                    ch,
                                    sigma_pin_um,
                                )
                                if nrp != nr_sur or ncp != nc_sur:
                                    pin_d = F.interpolate(
                                        pin_d.unsqueeze(0).unsqueeze(0),
                                        size=(nr_sur, nc_sur),
                                        mode="bilinear",
                                        align_corners=False,
                                    ).squeeze(0).squeeze(0)
                            l_cong = _refinement_rudy_pin_cong_loss(
                                demand,
                                pin_d,
                                pin_weight=self.pin_demand_weight,
                                util_threshold=self.pin_util_threshold,
                                overflow_sharpness=self.pin_overflow_sharpness,
                                top_k_frac=top_frac_cong_dyn,
                            )
                        else:
                            overflow = _soft_capacity_excess(
                                demand - 1.0, sharpness=beta_eff
                            )
                            l_cong = _abu_top_mean(
                                overflow, top_frac_cong_dyn
                            )
    
                if (
                    epoch == 0
                    and using_compiled
                    and not self.use_plc_routing_cong
                    and not (
                        self.pin_rudy_fusion and pin_gather is not None
                    )
                ):
                    with _diag_segment(row, "t_epoch0_compile_check", device):
                        l_wl_ref = _wirelength_loss_v2(
                            combined_pos,
                            net_idx,
                            net_mask,
                            net_weights_norm,
                            net_valid,
                            beta_eff,
                        )
                        rel_wl = abs(l_wl_raw.item() - l_wl_ref.item()) / (
                            abs(l_wl_ref.item()) + 1e-8
                        )
                        assert rel_wl < 1e-3, (
                            f"Compiled wirelength mismatch: {l_wl_raw.item()} vs {l_wl_ref.item()}"
                        )
                        demand_ref = _rudy_demand_grid(
                            combined_pos,
                            net_idx_r,
                            net_mask_r,
                            net_weights_raw_r,
                            net_valid_r,
                            beta_eff,
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
                l_cong_for_loss = l_cong
                if self.congestion_proxy_anchor_interval > 0:
                    l_cong_for_loss = l_cong + congestion_px_anchor_bias
                with _diag_segment(row, "t_loss_scalar", device):
                    loss = (
                        wl_u * l_wl + wdh_u * l_dh + wc_u * l_cong_for_loss
                    )
                    lv = float(loss.item())
                with _diag_segment(row, "t_backward", device):
                    loss.backward()
                with _diag_segment(row, "t_optimizer", device):
                    if self.momentum_restart:
                        _nesterov_momentum_restart(opt)
                    opt.step()

                with _diag_segment(row, "t_clamp", device):
                    _clamp_movable_fast(pos_hard, pos_soft, clamp_ctx)

                if (
                    self.qp_lookahead_interval > 0
                    and plc is not None
                    and n_hard > 0
                    and movable_hard_idx
                    and epoch > 0
                    and epoch % self.qp_lookahead_interval == 0
                ):
                    pv_la = float("inf")
                    legal_la: torch.Tensor | None = None
                    with _diag_segment(row, "t_qp_lookahead", device):
                        with torch.no_grad():
                            full_la = _assemble_full_fast(
                                pos_hard, pos_soft, assemble_ctx
                            )
                            full_cpu_la = full_la.detach().cpu().to(qp_dtype)
                            saved_bp = benchmark.macro_positions.clone()
                            try:
                                benchmark.macro_positions.copy_(full_cpu_la.float())
                                legal_la = QPLegalizer().place(benchmark)
                                c_la = compute_proxy_cost(legal_la, benchmark, plc)
                                pv_la = float(c_la["proxy_cost"])
                            finally:
                                benchmark.macro_positions.copy_(saved_bp)
                    if legal_la is not None and pv_la < best_proxy_val:
                        best_proxy_val = pv_la
                        best_proxy_placement = legal_la.clone()
                        _scatter_legal_placement_into_params(
                            legal_la,
                            pos_hard,
                            pos_soft,
                            movable_hard_idx,
                            n_hard,
                            n_macros,
                            device,
                            grad_dtype,
                        )
                        _zero_sgd_momentum(opt)

                if (
                    self.proxy_anchor_interval > 0
                    and plc is not None
                    and epoch > 0
                    and epoch % self.proxy_anchor_interval == 0
                ):
                    with torch.no_grad():
                        full_an = _assemble_full_fast(
                            pos_hard, pos_soft, assemble_ctx
                        )
                        full_cpu_an = full_an.detach().cpu().to(qp_dtype)
                        c_an = compute_proxy_cost(full_cpu_an, benchmark, plc)
                        pv_an = float(c_an["proxy_cost"])
                    if pv_an < anchor_best_proxy:
                        anchor_best_proxy = pv_an
                        if pos_hard.numel() > 0:
                            anchor_pos_hard = pos_hard.detach().clone()
                        if pos_soft.numel() > 0:
                            anchor_pos_soft = pos_soft.detach().clone()
                    elif (
                        anchor_best_proxy < float("inf")
                        and pv_an
                        > anchor_best_proxy * self.proxy_anchor_revert_ratio
                    ):
                        if pos_hard.numel() > 0 and anchor_pos_hard is not None:
                            pos_hard.data.copy_(anchor_pos_hard)
                        if pos_soft.numel() > 0 and anchor_pos_soft is not None:
                            pos_soft.data.copy_(anchor_pos_soft)
                        opt.zero_grad()
                        _zero_sgd_momentum(opt)
                    if (
                        self.proxy_anchor_tune_pins
                        and self.pin_rudy_fusion
                        and pin_gather is not None
                    ):
                        if (
                            anchor_prev_proxy_tune is not None
                            and anchor_prev_l_cong_tune is not None
                        ):
                            if (
                                pv_an > anchor_prev_proxy_tune * 1.05
                                and lcong_f < anchor_prev_l_cong_tune * 0.99
                            ):
                                sigma_pin_um = min(
                                    sigma_pin_um * 1.08,
                                    self.pin_sigma_max_um,
                                )
                                top_frac_cong_dyn = max(
                                    top_frac_cong_dyn * 0.92, 0.01
                                )
                        anchor_prev_proxy_tune = pv_an
                        anchor_prev_l_cong_tune = lcong_f

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
                                beta_eff,
                            )
                            l_wl_ev = l_wl_raw_ev / wl_norm
                            if self.use_local_gaussian_density:
                                if den_ctx is None or den_ctx[2] is None:
                                    l_dh_ev = torch.zeros(
                                        (), device=device, dtype=grad_dtype
                                    )
                                else:
                                    l_dh_ev = _density_loss_fast(
                                        full_ev,
                                        self.target_hard,
                                        den_ctx,
                                        sharpness=beta_eff,
                                    )
                            else:
                                dens_grid_ev = _plc_macro_overlap_density_grid(
                                    full_ev, benchmark, cw, ch, nr_sur, nc_sur
                                )
                                l_dh_ev = _plc_style_density_from_grid(
                                    dens_grid_ev,
                                    top_frac=self.top_frac_density,
                                )
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
                                    abu_frac=top_frac_cong_dyn,
                                    nr=rout_nr,
                                    nc=rout_nc,
                                )
                            else:
                                dens_fe: torch.Tensor | None = None
                                if not self.use_local_gaussian_density:
                                    dens_fe = dens_grid_ev
                                elif (
                                    self.rudy_hotspot_net_frac is not None
                                ):
                                    dens_fe = _plc_macro_overlap_density_grid(
                                        full_ev,
                                        benchmark,
                                        cw,
                                        ch,
                                        nr_sur,
                                        nc_sur,
                                    )
                                nix = net_idx
                                nmk = net_mask
                                nwr = net_weights_raw
                                nvl = net_valid
                                if (
                                    self.rudy_hotspot_net_frac is not None
                                    and dens_fe is not None
                                ):
                                    am_ev = _rudy_nets_active_mask(
                                        combined_ev,
                                        net_idx,
                                        net_mask,
                                        dens_fe,
                                        self.rudy_hotspot_net_frac,
                                        cw,
                                        ch,
                                    )
                                    nix = net_idx[am_ev]
                                    nmk = net_mask[am_ev]
                                    nwr = net_weights_raw[am_ev]
                                    nvl = net_valid[am_ev]
                                demand_ev = _rudy_demand_fn(
                                    combined_ev,
                                    nix,
                                    nmk,
                                    nwr,
                                    nvl,
                                    beta_eff,
                                    bin_x0,
                                    bin_x1,
                                    bin_y0,
                                    bin_y1,
                                )
                                if (
                                    self.pin_rudy_fusion
                                    and pin_gather is not None
                                ):
                                    nrp = self.pin_grid_rows
                                    ncp = self.pin_grid_cols
                                    if nrp is None:
                                        nrp = nr_sur
                                    if ncp is None:
                                        ncp = nc_sur
                                    nrp = max(int(nrp), 1)
                                    ncp = max(int(ncp), 1)
                                    pin_de = _compute_pin_demand_grid(
                                        combined_ev,
                                        pin_gather[0],
                                        pin_gather[1],
                                        nrp,
                                        ncp,
                                        cw,
                                        ch,
                                        sigma_pin_um,
                                    )
                                    if nrp != nr_sur or ncp != nc_sur:
                                        pin_de = F.interpolate(
                                            pin_de.unsqueeze(0).unsqueeze(0),
                                            size=(nr_sur, nc_sur),
                                            mode="bilinear",
                                            align_corners=False,
                                        ).squeeze(0).squeeze(0)
                                    l_cong_ev = _refinement_rudy_pin_cong_loss(
                                        demand_ev,
                                        pin_de,
                                        pin_weight=self.pin_demand_weight,
                                        util_threshold=self.pin_util_threshold,
                                        overflow_sharpness=self.pin_overflow_sharpness,
                                        top_k_frac=top_frac_cong_dyn,
                                    )
                                else:
                                    overflow_ev = _soft_capacity_excess(
                                        demand_ev - 1.0,
                                        sharpness=beta_eff,
                                    )
                                    l_cong_ev = _abu_top_mean(
                                        overflow_ev,
                                        top_frac_cong_dyn,
                                    )
                            sw = float(l_wl_ev.item())
                            sdh = float(l_dh_ev.item())
                            sc = float(l_cong_ev.item())
                            costs = compute_proxy_cost(full_cpu_ev, benchmark, plc)
                            pv = float(costs["proxy_cost"])
                            pw = float(costs["wirelength_cost"])
                            pd = float(costs["density_cost"])
                            pc = float(costs["congestion_cost"])
                            if (
                                self.congestion_proxy_anchor_interval > 0
                                and (epoch + 1)
                                % self.congestion_proxy_anchor_interval
                                == 0
                            ):
                                congestion_px_anchor_bias = float(pc - sc)
                                _cpa = (
                                    f"[congestion_proxy_anchor] epoch={epoch} "
                                    f"bias=px_raw_cong-sur_raw_cong={congestion_px_anchor_bias:.6g}"
                                )
                                print(_cpa, flush=True)
                                if log_fp is not None:
                                    log_fp.write(_cpa + "\n")
                                    log_fp.flush()
                            corr_sur.append(
                                w_wl_dyn * sw + w_dh_dyn * sdh + w_c_dyn * sc
                            )
                            corr_px.append(pv)
                            sur_now = (
                                w_wl_dyn * sw + w_dh_dyn * sdh + w_c_dyn * sc
                            )
                            if prev_ck_sur is not None and prev_ck_px is not None:
                                ds = (sur_now - prev_ck_sur) / (
                                    abs(prev_ck_sur) + 1e-30
                                )
                                dp = (pv - prev_ck_px) / (abs(prev_ck_px) + 1e-30)
                                if ds < -0.10 and dp > 0.05:
                                    beta_end_dynamic = min(
                                        beta_end_dynamic
                                        * self.surrogate_beta_boost_on_divergence,
                                        self.surrogate_beta_boost_max,
                                    )
                            prev_ck_sur = sur_now
                            prev_ck_px = pv
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

        corr_plot_path = _resolve_user_path(self.surrogate_proxy_plot)
        if corr_plot_path is not None and len(corr_sur) >= 2:
            _write_surrogate_proxy_corr_plot(corr_plot_path, corr_sur, corr_px)

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
    Gradient3Placer().place(b)


if __name__ == "__main__":
    _cli_main()
