"""
GPU Placer (DreamPlace-inspired, legality-agnostic).

Goal: run the entire optimization loop on GPU (when available) and optimize a
smooth surrogate objective. Congestion defaults to **GPU pin-aware L-routes** when pin tables
exist; macro-center L-routes or bbox RUDY are opt-in via env (see module doc).
This placer does **not** attempt to legalize or
remove overlaps; it relies on a soft density penalty (optional) and box clamps.

Usage:
    uv run evaluate submissions/gpu/placer.py -b ibm01
    MACRO_PLACE_DEVICE=cuda uv run evaluate submissions/gpu/placer.py -b ibm01

Training curves under ``vis/``:

- ``<benchmark>_gpu_proxy_vs_epoch.png`` — PLC ``proxy_cost`` (needs ICCAD04 ``PlacementCost``),
  sampled every ``MACRO_PLACE_GPU_PROXY_CHECK_EVERY`` (default **50**).
- ``<benchmark>_gpu_surrogate_loss_vs_epoch.png`` — surrogate total loss at ``log_every``
  (default 50), saved whenever those samples exist (in addition to the proxy curve when PLC loads).

Set ``MACRO_PLACE_GPU_TRAINING_PLOT=0`` to skip all figures.

Proxy-check console logging:
    ``MACRO_PLACE_GPU_PROXY_LOG`` — default ``off``; ``compact`` / ``legacy`` for console reports.
    ``MACRO_PLACE_GPU_VERBOSE=1`` — epoch logs, proxy reports, plot messages (default quiet).
    ``MACRO_PLACE_GPU_TRAINING_PLOT`` — default ``0`` (set ``1`` for PNG curves).

Optimizer (movable macro positions):
    ``MACRO_PLACE_GPU_OPTIMIZER`` — ``adam`` (default) for ``torch.optim.Adam``, or
    ``nesterov`` for ``torch.optim.SGD(..., nesterov=True)``. With Nesterov,
    ``MACRO_PLACE_GPU_MOMENTUM`` defaults to ``0.9``.

Surrogate early-stop (no PLC required):
    ``MACRO_PLACE_GPU_SURROGATE_STAGNATION_MIN_ABS`` / ``_PATIENCE`` / ``_CHECK_EVERY`` — min-abs
    streak on GPU **total surrogate loss**. ``MACRO_PLACE_GPU_SURROGATE_STAGNATION_MIN_REL_INITIAL``
    sets the threshold to ``rel × loss`` at the first check of the phase (overrides fixed min-abs).

PLC proxy early-stop (when ``PlacementCost`` loads):
    ``late_stagnation_sgd_switch`` (constructor, default off) — when enabled with abs stagnation,
    at ``patience - 1`` consecutive sub-threshold checks switch Adam to plain SGD
    (``MACRO_PLACE_GPU_SGD_LR_RATIO``, default ``0.2`` × Adam lr)
    and reset the streak. ``liquid.py`` patience phase sets this on.
    ``MACRO_PLACE_GPU_STAGNATION_MIN_ABS_IMPROVEMENT`` — if **> 0** (default ``0.01``),
    after each PLC proxy check (every ``MACRO_PLACE_GPU_PROXY_CHECK_EVERY`` epochs,
    default ``100``), stop only after ``MACRO_PLACE_GPU_STAGNATION_PATIENCE`` consecutive
    checks (default ``1`` when patience is 0) where proxy improved by less than this
    **absolute** amount vs the session **best** proxy so far. When **<= 0**, falls back to
    ``MACRO_PLACE_GPU_STAGNATION_PATIENCE`` / ``MACRO_PLACE_GPU_STAGNATION_REL_DELTA``
    if patience **> 0``.     ``MACRO_PLACE_PE_STAGNATION_MIN_ABS_IMPROVEMENT`` overrides
    the GPU env when set (same semantics).

Density / CUDA stability:
    ``MACRO_PLACE_GPU_DENSITY_MODEL`` — surrogate for the density term in the GPU loop:
    ``plc`` (default): PLC rectangle overlap grid + proxy scalar (unchanged);
    ``electrostatic``: smooth triangular charge spreading + FFT Poisson + electrostatic energy;
    ``both``: ``0.5`` PLC loss + ``0.5`` electrostatic loss (transition / tuning).
    With ``electrostatic``, affine proxy calibration does **not** adjust ``w_den_run`` from PLC density.
    Without ``w_density_schedule``, ``w_den_run`` stays ``self.w_density``. With ``w_density_schedule``,
    ``w_den_run`` is overwritten each epoch by the schedule (still no PLC density calibration).
    WL/congestion affine calibration is unchanged when enabled.
    Gradient norm clipping (``max_norm=1.0``) runs after ``backward`` as a safety net.

    Optional constructor ``w_density_schedule(epoch0) -> float``: when ``density_model`` is
    ``electrostatic``, called each epoch before the forward pass (ePlace-style schedules).

    ``MACRO_PLACE_GPU_ELECTRO_ENERGY_SCALE`` — multiplier on the electrostatic scalar
    (after ``/(nr*nc)``); default ``100000`` so logged ``den`` is comparable in magnitude to the
    PLC proxy density scalar for typical ICCAD04 canvases (set ``1`` for the raw normalized energy).
    ``MACRO_PLACE_GPU_DENSITY_CHECKPOINT`` — default ``0`` (off). Set ``1`` to
    trade VRAM for activation checkpointing in the PLC overlap density path
    (checkpointing can trigger rare illegal-access errors on some CUDA stacks).
    ``MACRO_PLACE_GPU_OVERLAP_TRITON`` — default ``0``. On CUDA **float32**, ``1`` uses a
    Triton kernel for the **forward** pairwise overlap reduction (backward still
    PyTorch). Requires Triton (Linux dependency stack or e.g. ``triton-windows``).
    ``MACRO_PLACE_GPU_PROFILE_SECTIONS`` — ``1`` prints **mean CUDA ms/epoch** per surrogate
    slice (wirelength / density / congestion / overlap forward, backward, optimizer, clamp)
    using CUDA events — avoids ``torch.profiler`` overhead stalls on very large graphs.
    ``MACRO_PLACE_GPU_CUDA_FALLBACK`` — default ``1`` (on): **only** clear GPU OOM
    (message contains ``out of memory``) retries on CPU. Other CUDA failures are not
    retried: the device context is often unusable and copying parameters to CPU can
    fail with the same error. Set ``0`` to never fall back to CPU.

Congestion surrogate:
    Default (``MACRO_PLACE_GPU_USE_RUDY_CONG=1``): GPU **pin-aware star L-routes** when
    ``net_pin_nodes`` is complete (hard-macro pin offsets + PLC route weights), else macro-center
    L-routes. ``MACRO_PLACE_GPU_USE_BBOX_RUDY=1`` — legacy uniform bbox RUDY.
    ``MACRO_PLACE_GPU_PIN_CONG=0`` — macro-center L-routes even when pin tables exist.
    ``MACRO_PLACE_GPU_PLC_NET_ROUTING=1`` (default): after each proxy check, use PLC net routing
    grids (total minus macro blockage) plus differentiable ``_macro_blockage_raw`` instead of
    L-route net demand until the next check; L-route fallback before the first check.

Surrogate form (per term): fit ``w * surrogate ≈ proxy_subcost`` (no offset ``k``). Proxy weights
``1.0 / 0.5 / 0.5`` apply when summing into ``loss``. Patience ``affine_calibrate=True`` sets
``w`` from PLC checkpoints for WL, density, **and congestion** by default. WL and congestion use
level-only EMA (``_fit_scale_wl``: ``px_wl/sur_wl``, ``alpha=0.20``, clamp **0.5–3.0**;
``_fit_scale_cong``: ``px_cong/sur_cong``, ``alpha=0.15``, clamp **0.5–1.5**).
Density uses ``_fit_scale_surrogate_to_proxy`` unless ``affine_calibrate_density=False``, which keeps
``w_den_run`` fixed while still fitting WL/congestion when ``affine_calibrate=True``.

Spatial congestion hotspot loss (patience / ``affine_calibrate=True`` only):
    ``MACRO_PLACE_GPU_SPATIAL_CONG`` — default ``1``: weight surrogate congestion in PLC
    overloaded bins (EMA hotspot map from ``get_*_routing_congestion`` after proxy checks).
    ``MACRO_PLACE_GPU_HOTSPOT_ALPHA`` — EMA blend for hotspot map (default ``0.3``).
    ``MACRO_PLACE_GPU_HOTSPOT_SCALE`` — multiplier on spatial term vs scalar ABU cong (default ``0.1``).
    ``MACRO_PLACE_GPU_HOTSPOT_MIN_EPOCH`` — first displayed epoch for hotspot map/loss (default ``200``).
    ``MACRO_PLACE_GPU_HOTSPOT_H_WEIGHT`` / ``_V_WEIGHT`` — H vs V excess in PLC hotspot map (default ``2.0`` / ``1.0``).
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from macro_place.benchmark import Benchmark
from macro_place.routing_surrogate import (
    grid_routing_capacities,
    plc_routing_surrogate_hv_totals,
    plc_routing_surrogate_hv_totals_pins,
    plc_routing_surrogate_scalar,
    plc_routing_surrogate_scalar_pins,
    smooth_routing_cong_plc,
    _macro_blockage_raw,
)
from macro_place.objective import compute_proxy_cost

# Reuse proven, vectorized building blocks from the gradient baseline.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from submissions.gradient import (
    _assemble_full_fast,
    _build_assemble_ctx,
    _build_clamp_ctx,
    _build_net_tensors,
    _clamp_movable_fast,
    _abu_top_mean,
    _rudy_demand_grid,
    _select_device,
    _try_load_plc_iccad04,
    _wirelength_loss_v2,
)
from submissions.gpu.pairwise_overlap import (
    build_overlap_pair_indices,
    pairwise_overlap_sum_normalized,
    want_triton_overlap,
)


def _plc_overlap_density_chunk(
    centers_xy: torch.Tensor,
    wh: torch.Tensor,
    bx0: torch.Tensor,
    bx1: torch.Tensor,
    by0: torch.Tensor,
    by1: torch.Tensor,
    bin_area: torch.Tensor,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
) -> torch.Tensor:
    """Add overlap density contribution for a chunk of macros.

    ``centers_xy``: [C,2], ``wh``: [C,2]
    ``cw`` / ``ch`` are **full canvas** width and height (same as PlacementCost.width/height).
    Bin bounds are broadcast grids with shape [nr, nc].
    Returns a density increment with shape [nr, nc] summing over the C macros.
    """
    # Non-finite centers from upstream optimization can produce undefined long
    # indices and rare illegal CUDA access; snap bad rows to canvas center.
    finite = torch.isfinite(centers_xy).all(dim=-1, keepdim=True)
    safe_c = centers_xy.new_tensor([0.5 * float(cw), 0.5 * float(ch)]).view(1, 2)
    centers_xy = torch.where(finite, centers_xy, safe_c.expand_as(centers_xy))
    wh_safe = torch.nan_to_num(wh, nan=1.0, posinf=max(float(cw), float(ch)), neginf=1e-9)
    w = wh_safe[:, 0].clamp(min=1e-9, max=float(cw) + 1.0)
    h = wh_safe[:, 1].clamp(min=1e-9, max=float(ch) + 1.0)
    cx = centers_xy[:, 0]
    cy = centers_xy[:, 1]

    # Match PlacementCost.__get_grid_cell_location: floor(pos / grid_height), etc.
    bin_w = float(cw) / float(max(nc, 1))
    bin_h = float(ch) / float(max(nr, 1))

    # Mirror PlacementCost.__add_module_to_grid_cells row/col windowing (clamp to grid).
    ur_x = cx + 0.5 * w
    ur_y = cy + 0.5 * h
    bl_x = cx - 0.5 * w
    bl_y = cy - 0.5 * h

    ur_row = torch.floor((ur_y / bin_h).clamp(min=-1e9)).to(dtype=torch.long)
    ur_col = torch.floor((ur_x / bin_w).clamp(min=-1e9)).to(dtype=torch.long)
    bl_row = torch.floor((bl_y / bin_h).clamp(min=-1e9)).to(dtype=torch.long)
    bl_col = torch.floor((bl_x / bin_w).clamp(min=-1e9)).to(dtype=torch.long)

    # PLC: if upper-right cell is OOB (row or col < 0), skip module entirely.
    oob_skip = (ur_row < 0) | (ur_col < 0)
    # Else clamp negative bottom-left to 0.
    bl_row = torch.where(~oob_skip & (bl_row < 0), torch.zeros_like(bl_row), bl_row)
    bl_col = torch.where(~oob_skip & (bl_col < 0), torch.zeros_like(bl_col), bl_col)
    # PLC: if bottom-left still invalid, skip.
    oob_skip = oob_skip | (bl_row < 0) | (bl_col < 0)

    ur_row = torch.where(~oob_skip & (ur_row > nr - 1), torch.full_like(ur_row, nr - 1), ur_row)
    ur_col = torch.where(~oob_skip & (ur_col > nc - 1), torch.full_like(ur_col, nc - 1), ur_col)

    # Convert to float masks on the grid by comparing row/col indices to arange tensors.
    rs = torch.arange(nr, device=centers_xy.device, dtype=torch.long).view(1, nr, 1)
    cs = torch.arange(nc, device=centers_xy.device, dtype=torch.long).view(1, 1, nc)

    bl_r = bl_row.view(-1, 1, 1)
    ur_r = ur_row.view(-1, 1, 1)
    bl_c = bl_col.view(-1, 1, 1)
    ur_c = ur_col.view(-1, 1, 1)

    row_ok = (rs >= bl_r) & (rs <= ur_r)
    col_ok = (cs >= bl_c) & (cs <= ur_c)
    win = row_ok & col_ok  # [C, nr, nc]

    m = (~oob_skip).view(-1, 1, 1).to(dtype=win.dtype)
    win = win & m

    cxg = cx.view(-1, 1, 1)
    cyg = cy.view(-1, 1, 1)
    wg = w.view(-1, 1, 1)
    hg = h.view(-1, 1, 1)
    lx = cxg - 0.5 * wg
    rx = cxg + 0.5 * wg
    by_m = cyg - 0.5 * hg
    ty_m = cyg + 0.5 * hg

    ix0 = torch.relu(torch.minimum(rx, bx1) - torch.maximum(lx, bx0))
    iy0 = torch.relu(torch.minimum(ty_m, by1) - torch.maximum(by_m, by0))
    contrib = (ix0 * iy0 / bin_area) * win.to(ix0.dtype)
    return contrib.sum(dim=0)


def _plc_proxy_density_cost(
    dens: torch.Tensor,
) -> torch.Tensor:
    """Mirror ``PlacementCost.get_density_cost`` reduction on a dense grid.

    This follows the structure in ``plc_client_os.PlacementCost.get_density_cost``:
    - If total grid cells < 10: average of *occupied* (>0) cells, ×0.5
    - Else: let ``density_cnt = floor(N * 0.1)`` (>=1). Sort occupied cells
      descending and accumulate up to ``density_cnt`` terms (or exhaust occupied),
      then divide by ``density_cnt`` (not by the number of accumulated terms),
      and multiply by 0.5.
    """
    flat = dens.reshape(-1)
    ncells = int(flat.numel())
    if ncells == 0:
        return torch.zeros((), device=flat.device, dtype=flat.dtype)

    occ = flat[flat > 0]
    if int(occ.numel()) == 0:
        return torch.zeros((), device=flat.device, dtype=flat.dtype)

    if ncells < 10:
        return 0.5 * occ.mean()

    density_cnt = max(1, int(math.floor(ncells * 0.1)))
    occ_sorted, _ = torch.sort(occ, descending=True)
    k = min(density_cnt, int(occ_sorted.numel()))
    top = occ_sorted[:k]
    return 0.5 * (top.sum() / float(density_cnt))


def _plc_macro_overlap_density_grid_checkpointed(
    full_pos: torch.Tensor,
    benchmark: Benchmark,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
    *,
    chunk: int,
    use_checkpoint: bool,
) -> torch.Tensor:
    """PLC-style per-bin occupancy fraction, accumulated in macro chunks.

    Optionally uses activation checkpointing on CUDA to save VRAM (see
    ``MACRO_PLACE_GPU_DENSITY_CHECKPOINT``); when disabled, the full macro loop
    still runs in chunks to limit peak graph size.
    """
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
    dens = torch.zeros(nr, nc, device=device, dtype=dtype)

    ck = max(int(chunk), 1)
    for s in range(0, n, ck):
        e = min(s + ck, n)
        centers = full_pos[s:e]
        wh = sizes[s:e]
        if use_checkpoint:
            dens = dens + torch.utils.checkpoint.checkpoint(
                _plc_overlap_density_chunk,
                centers,
                wh,
                bx0,
                bx1,
                by0,
                by1,
                torch.as_tensor(bin_area, device=device, dtype=dtype),
                cw,
                ch,
                nr,
                nc,
                use_reentrant=False,
            )
        else:
            dens = dens + _plc_overlap_density_chunk(
                centers,
                wh,
                bx0,
                bx1,
                by0,
                by1,
                torch.as_tensor(bin_area, device=device, dtype=dtype),
                cw,
                ch,
                nr,
                nc,
            )
    return dens


def _spread_charge_to_grid(
    full_pos: torch.Tensor,
    sizes: torch.Tensor,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
) -> torch.Tensor:
    """Smooth charge density ρ on the PLC grid (triangular kernel), differentiable.

    Each macro contributes mass ``area / canvas_area``. Kernel weights are separable
    triangles with radius ``w/2 + bin_w`` (x) and ``h/2 + bin_h`` (y), normalized
    per macro so ``sum_bins ρ_ij · bin_area = mass`` for that macro.
    """
    device = full_pos.device
    dtype = full_pos.dtype
    bw = cw / nc
    bh = ch / nr
    bin_area = bw * bh
    canvas_area = cw * ch

    cols = torch.arange(nc, device=device, dtype=dtype)
    rows = torch.arange(nr, device=device, dtype=dtype)
    bin_cx = (cols + 0.5) * bw
    bin_cy = (rows + 0.5) * bh

    cx = full_pos[:, 0].reshape(-1, 1, 1)
    cy = full_pos[:, 1].reshape(-1, 1, 1)
    w = sizes[:, 0].reshape(-1, 1, 1)
    h = sizes[:, 1].reshape(-1, 1, 1)

    rx = 0.5 * w + bw
    ry = 0.5 * h + bh

    bin_cx_e = bin_cx.reshape(1, 1, nc)
    bin_cy_e = bin_cy.reshape(1, nr, 1)

    tx = torch.relu(1.0 - torch.abs(cx - bin_cx_e) / rx)
    ty = torch.relu(1.0 - torch.abs(cy - bin_cy_e) / ry)
    kernel = tx * ty

    mass = (w.squeeze(-1).squeeze(-1) * h.squeeze(-1).squeeze(-1)) / canvas_area
    mass = mass.reshape(-1, 1, 1)

    denom = (kernel * bin_area).sum(dim=(1, 2), keepdim=True).clamp(
        min=torch.finfo(dtype).tiny
    )
    contrib = mass * kernel / denom
    return contrib.sum(dim=0)


def _solve_poisson_fft(charge_grid: torch.Tensor) -> torch.Tensor:
    """Solve ∇²φ = −ρ on a periodic torus via real FFT (ePlace-style spectral Laplacian)."""
    nr, nc = int(charge_grid.shape[0]), int(charge_grid.shape[1])
    dt = charge_grid.dtype
    dev = charge_grid.device

    rho_hat = torch.fft.rfft2(charge_grid)
    kx = torch.fft.fftfreq(nr, device=dev, dtype=dt) * (2.0 * math.pi)
    ky = torch.fft.rfftfreq(nc, device=dev, dtype=dt) * (2.0 * math.pi)
    kx2 = kx[:, None] ** 2
    ky2 = ky[None, :] ** 2
    k2 = kx2 + ky2
    k2_safe = k2.clone()
    k2_safe[0, 0] = 1.0
    phi_hat = rho_hat / k2_safe
    phi_hat = phi_hat.clone()
    phi_hat[0, 0] = 0.0
    return torch.fft.irfft2(phi_hat, s=(nr, nc))


def _electrostatic_density_loss(
    full_pos: torch.Tensor,
    benchmark: Benchmark,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
    target_density: float = 1.0,
) -> torch.Tensor:
    """Electrostatic energy ∑ ρ φ with ∇²φ = −ρ; gradient flows via spreading."""
    del target_density  # reserved for future uniform-background subtraction / tuning
    device, dtype = full_pos.device, full_pos.dtype
    sizes = benchmark.macro_sizes.to(device=device, dtype=dtype)
    charge = _spread_charge_to_grid(full_pos, sizes, nr, nc, cw, ch)
    phi = _solve_poisson_fft(charge)
    energy = (charge * phi).sum()
    norm = float(nr * nc)
    scale_raw = os.environ.get("MACRO_PLACE_GPU_ELECTRO_ENERGY_SCALE", "100000")
    try:
        scale = float((scale_raw or "100000").strip())
    except ValueError:
        scale = 100_000.0
    return (energy / norm) * scale


def _build_pin_net_tensors(
    benchmark: Benchmark,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pin-level net tensors for HPWL (matches PlacementCost pin positions)."""
    num_nets = int(benchmark.num_nets)

    def _normalize_net_pins(pins: torch.Tensor) -> torch.Tensor:
        """Return ``pins`` as a ``[num_pins, 2]`` long tensor on ``device``."""
        p = pins.to(device=device, dtype=torch.long)
        if p.numel() == 0:
            return p.reshape(0, 2)
        if p.dim() == 1:
            if int(p.numel()) % 2 != 0:
                raise ValueError(f"odd-length pin encoding: shape={tuple(pins.shape)}")
            return p.reshape(-1, 2)
        if p.dim() == 2:
            if int(p.shape[-1]) == 2:
                return p.reshape(-1, 2)
            # Rare malformed shapes like [2,2] should be interpreted as 2 pins.
            if int(p.numel()) % 2 == 0:
                return p.reshape(-1, 2)
        raise ValueError(f"unexpected net_pin_nodes shape: {tuple(pins.shape)}")

    pin_rows: list[torch.Tensor] = []
    max_pins = 0
    for k in range(num_nets):
        pr = _normalize_net_pins(benchmark.net_pin_nodes[k])
        pin_rows.append(pr)
        max_pins = max(max_pins, int(pr.shape[0]))

    net_idx = torch.zeros((num_nets, max_pins, 2), dtype=torch.long, device=device)
    net_mask = torch.zeros((num_nets, max_pins), dtype=torch.bool, device=device)

    for k in range(num_nets):
        pr = pin_rows[k]
        n = int(pr.shape[0])
        if n > 0:
            net_idx[k, :n] = pr
            net_mask[k, :n] = True

    w_cpu = benchmark.net_weights
    net_weights = torch.empty((num_nets,), device=device, dtype=dtype)
    wl_w_mode = os.environ.get("MACRO_PLACE_WL_WEIGHT_MODE", "plc").strip().lower()
    for k in range(num_nets):
        pn = int(pin_rows[k].shape[0])
        pc = max(pn, 2)
        base_w = float(w_cpu[k].item())
        if wl_w_mode in ("legacy", "divide", "pin_norm"):
            wk = base_w / float(pc - 1)
        else:
            # Match ``PlacementCost.get_wirelength()``: per-net scale is the driver pin weight,
            # not divided by number of pins.
            wk = base_w
        net_weights[k] = max(wk, 1e-6)

    net_valid = torch.tensor(
        [int(pin_rows[k].shape[0]) >= 2 for k in range(num_nets)],
        device=device,
        dtype=torch.bool,
    )
    return net_idx, net_mask, net_weights, net_valid


def _pin_positions_for_hpwl(
    owner_pos: torch.Tensor,
    pin_owner: torch.Tensor,
    pin_slot: torch.Tensor,
    *,
    n_hard: int,
    macro_pin_offsets: list[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Absolute pin coordinates from owner centers + hard-macro pin offsets."""
    n_nodes = int(owner_pos.shape[0])
    # Guard OOB owner indices (padding or bad data); avoids illegal CUDA access on gather.
    pin_owner = pin_owner.long().clamp(min=0, max=max(n_nodes - 1, 0))
    pin_slot = pin_slot.long()

    base = owner_pos[pin_owner]
    off = torch.zeros_like(base)

    if n_hard <= 0:
        return base

    hard_owner = pin_owner < int(n_hard)
    if not torch.any(hard_owner):
        return base

    ho = pin_owner[hard_owner].contiguous()
    hs = pin_slot[hard_owner].contiguous()
    ox = torch.zeros((ho.shape[0],), device=owner_pos.device, dtype=dtype)
    oy = torch.zeros((ho.shape[0],), device=owner_pos.device, dtype=dtype)

    # Typically a small loop over hard macros (tens–low hundreds), keeps offsets exact.
    for mi in range(int(n_hard)):
        m = ho == mi
        if not torch.any(m):
            continue
        offs = macro_pin_offsets[mi].to(device=owner_pos.device, dtype=dtype)
        if offs.numel() == 0:
            continue
        slots = hs[m].clamp(min=0, max=max(int(offs.shape[0]) - 1, 0)).long()
        sel = offs[slots]
        ox[m] = sel[:, 0]
        oy[m] = sel[:, 1]

    off[hard_owner, 0] = ox
    off[hard_owner, 1] = oy
    return base + off


def _training_plots_enabled() -> bool:
    return os.environ.get("MACRO_PLACE_GPU_TRAINING_PLOT", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _save_proxy_vs_epoch_plot(name: str, epochs: list[int], proxies: list[float]) -> bool:
    """Save PLC proxy vs epoch when matplotlib is available. Returns True if a file was written."""
    if not epochs:
        return False
    if not _training_plots_enabled():
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _gpu_log("[gpu_placer] matplotlib missing; skip proxy vs epoch plot", flush=True)
        return False

    vis = _ROOT / "vis"
    vis.mkdir(parents=True, exist_ok=True)
    out = vis / f"{name}_gpu_proxy_vs_epoch.png"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, proxies, color="tab:blue", linewidth=1.2, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PLC proxy_cost")
    ax.set_title(f"{name} — GpuPlacer PLC proxy vs epoch")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _gpu_log(f"[gpu_placer] proxy vs epoch plot saved to {out.resolve()}", flush=True)
    return True


def _save_surrogate_loss_vs_epoch_plot(name: str, epochs: list[int], losses: list[float]) -> bool:
    """Save surrogate total loss vs epoch (GPU-side objective) when matplotlib is available."""
    if not epochs:
        return False
    if not _training_plots_enabled():
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _gpu_log("[gpu_placer] matplotlib missing; skip surrogate loss plot", flush=True)
        return False

    vis = _ROOT / "vis"
    vis.mkdir(parents=True, exist_ok=True)
    out = vis / f"{name}_gpu_surrogate_loss_vs_epoch.png"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, losses, color="tab:orange", linewidth=1.2, marker="o", markersize=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Surrogate loss (GpuPlacer)")
    ax.set_title(f"{name} — surrogate total loss vs epoch")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _gpu_log(f"[gpu_placer] surrogate loss plot saved to {out.resolve()}", flush=True)
    return True


def _report_training_plots(
    *,
    plc_loaded: bool,
    proxy_check_every: int,
    log_every: int,
    saved_proxy: bool,
    saved_surr: bool,
) -> None:
    if not _gpu_placer_verbose():
        return
    if saved_proxy or saved_surr:
        return
    parts: list[str] = []
    if not plc_loaded:
        parts.append("PlacementCost (ICCAD04) did not load")
    elif proxy_check_every <= 0:
        parts.append("MACRO_PLACE_GPU_PROXY_CHECK_EVERY is 0")
    if log_every <= 0:
        parts.append("log_every is 0 (no surrogate samples)")
    if not parts:
        parts.append("matplotlib missing, MACRO_PLACE_GPU_TRAINING_PLOT=0, or no data")
    _gpu_log(
        f"[gpu_placer] no training plot saved ({'; '.join(parts)}). "
        "Install matplotlib or fix ICC paths; surrogate PNG uses log_every>0."
    )


def _make_gpu_optimizer(params: list[nn.Parameter], lr: float) -> torch.optim.Optimizer:
    """Adam (default) or SGD+Nesterov via ``MACRO_PLACE_GPU_OPTIMIZER``."""
    kind = os.environ.get("MACRO_PLACE_GPU_OPTIMIZER", "adam").strip().lower()
    if kind in ("nesterov", "sgd_nesterov", "nesterov_sgd"):
        mom = float(os.environ.get("MACRO_PLACE_GPU_MOMENTUM", "0.9") or "0.9")
        opt = torch.optim.SGD(
            params,
            lr=lr,
            momentum=mom,
            dampening=0,
            weight_decay=0.0,
            nesterov=True,
        )
        _gpu_log(f"[gpu_placer] optimizer=SGD+Nesterov lr={lr:g} momentum={mom:g}")
        return opt
    if kind not in ("adam", ""):
        _gpu_log(f"[gpu_placer] unknown MACRO_PLACE_GPU_OPTIMIZER={kind!r}; using Adam")
    _gpu_log(f"[gpu_placer] optimizer=Adam lr={lr:g}")
    return torch.optim.Adam(params, lr=lr)


def _switch_to_sgd(
    params: list[nn.Parameter],
    lr_sgd: float,
) -> torch.optim.Optimizer:
    """Replace Adam with pure SGD (no momentum) for late-stage refinement."""
    _gpu_log(
        f"[gpu_placer] optimizer switch: Adam -> SGD lr={lr_sgd:g} "
        "(late-stage refinement, no momentum)"
    )
    return torch.optim.SGD(params, lr=lr_sgd, momentum=0.0, nesterov=False)


def _cuda_teardown_after_training() -> None:
    """Drain async CUDA work and release cached allocations before CPU copy / matplotlib."""
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


def _fit_scale_surrogate_to_proxy(
    l1: float,
    px1: float,
    l2: float,
    px2: float,
    w_default: float,
    *,
    w_min_ratio: float = 0.1,
    w_max_ratio: float = 10.0,
) -> float:
    """Scale ``w`` so ``w*surrogate ≈ proxy_subcost`` (no intercept)."""
    l2f, px2f = float(l2), float(px2)
    dl = l2f - float(l1)
    dpx = px2f - float(px1)
    l_scale = max(abs(l2f), abs(float(l1)), 1e-9)
    w_level = px2f / max(l2f, 1e-9)
    if abs(dl) < max(1e-9, 1e-4 * l_scale):
        w = w_level
    else:
        w_slope = dpx / dl
        w = w_slope if w_slope > 0.0 else w_level
    w_lo = float(w_default) * float(w_min_ratio)
    w_hi = float(w_default) * float(w_max_ratio)
    return max(w_lo, min(w_hi, w))


def _fit_scale_cong(
    sur_cong: float,
    px_cong: float,
    ema_w: float,
    *,
    alpha: float = 0.15,
    w_min: float = 0.5,
    w_max: float = 1.5,
) -> float:
    """EMA-smoothed level-only scale for the congestion surrogate.

    Uses only the instantaneous level ``px_cong / sur_cong`` (ratio is stable
    on many benchmarks); no slope between checkpoints, which was too noisy.

    Args:
        sur_cong: surrogate congestion at the current checkpoint.
        px_cong: PLC congestion subcost at the current checkpoint.
        ema_w: previous EMA value; pass return value back on the next call.
        alpha: EMA blend toward ``level`` each proxy check (default 0.15).
        w_min / w_max: clamp on the returned weight (default 0.5–1.5).

    Returns:
        Updated EMA weight (use as ``w_cong_run`` and pass back as ``ema_w``).
    """
    sur = max(float(sur_cong), 1e-9)
    level = float(px_cong) / sur
    new_ema = (1.0 - alpha) * float(ema_w) + alpha * level
    return max(float(w_min), min(float(w_max), new_ema))


def _fit_scale_wl(
    sur_wl: float,
    px_wl: float,
    ema_w: float,
    *,
    alpha: float = 0.20,
    w_min: float = 0.5,
    w_max: float = 3.0,
) -> float:
    """EMA-smoothed level-only scale fit for the WL surrogate.

    The ``px_wl / sur_wl`` ratio is very stable on many benchmarks; the slope
    estimator in ``_fit_scale_surrogate_to_proxy`` can make ``w_wl`` oscillate
    checkpoint-to-checkpoint when ``dl`` is noisy.

    Args:
        sur_wl: surrogate WL at the current checkpoint.
        px_wl: PLC proxy WL at the current checkpoint.
        ema_w: previous EMA value; pass return value back on the next call.
        alpha: EMA blend toward ``level`` each proxy check (default 0.20).
        w_min / w_max: clamp on the returned weight (default 0.5–3.0).

    Returns:
        Updated EMA weight (use as ``w_wl_run`` and pass back as ``ema_w_wl``).
    """
    sur = max(float(sur_wl), 1e-9)
    level = float(px_wl) / sur
    new_ema = (1.0 - alpha) * float(ema_w) + alpha * level
    return max(float(w_min), min(float(w_max), new_ema))


_PROXY_DEN_COEF = 0.5
_PROXY_CONG_COEF = 0.5


def _gpu_placer_verbose() -> bool:
    """Default quiet; set ``MACRO_PLACE_GPU_VERBOSE=1`` for training/proxy console logs."""
    return os.environ.get("MACRO_PLACE_GPU_VERBOSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _gpu_log(msg: str) -> None:
    if _gpu_placer_verbose():
        print(msg, flush=True)


def _gpu_final_summary_enabled() -> bool:
    if _gpu_placer_verbose():
        return False
    if os.environ.get("MACRO_PLACE_EVAL_QUIET", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    raw = (os.environ.get("MACRO_PLACE_GPU_FINAL_SUMMARY") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _gpu_print_final_summary(
    *,
    benchmark: Benchmark,
    full: torch.Tensor,
    plc,
    elapsed: float,
    epochs: int,
) -> None:
    if not _gpu_final_summary_enabled():
        return
    if plc is not None:
        with torch.no_grad():
            costs = compute_proxy_cost(
                full.detach().to(device="cpu", dtype=torch.float32),
                benchmark,
                plc,
            )
        px = float(costs["proxy_cost"])
        print(
            f"[gpu_placer] proxy={px:.6g} elapsed={elapsed:.1f}s epochs={epochs}",
            flush=True,
        )
    else:
        print(f"[gpu_placer] elapsed={elapsed:.1f}s epochs={epochs}", flush=True)


def _proxy_log_mode() -> str:
    raw = (os.environ.get("MACRO_PLACE_GPU_PROXY_LOG") or "off").strip().lower()
    if raw in ("compact", "legacy", "off"):
        return raw
    return "compact"


def _print_proxy_check_report(
    *,
    epoch: int,
    px_proxy: float,
    px_proxy_start: float | None,
    px_proxy_prev: float | None,
    px_proxy_best: float | None,
    sur_wl: float,
    sur_den: float,
    sur_cong: float,
    px_wl: float,
    px_den: float,
    px_cong: float,
    w_wl: float,
    w_den: float,
    w_cong: float,
    al_wl: float,
    al_den: float,
    al_cong: float,
    al_sum: float,
    mode: str,
) -> None:
    err_wl = al_wl - px_wl
    err_den = al_den - px_den
    err_cong = al_cong - px_cong
    loss_minus_proxy = al_sum - px_proxy

    if mode == "off":
        return

    if mode == "legacy":
        print(
            f"[gpu_placer] scale_calib epoch={epoch} "
            f"wl sur={sur_wl:.4g} px={px_wl:.4g} w={w_wl:.4g} "
            f"scaled={al_wl:.4g} | "
            f"den sur={sur_den:.4g} px={px_den:.4g} w={w_den:.4g} "
            f"scaled={al_den:.4g} | "
            f"cong sur={sur_cong:.4g} px={px_cong:.4g} w={w_cong:.4g} "
            f"scaled={al_cong:.4g}",
            flush=True,
        )
        print(
            f"[gpu_placer] surrogate_vs_proxy epoch={epoch} "
            f"sur(wl,den,cong)=({sur_wl:.4g},{sur_den:.4g},{sur_cong:.4g}) "
            f"plc(wl,den,cong)=({px_wl:.4g},{px_den:.4g},{px_cong:.4g}) "
            f"scaled(w*sur)=({al_wl:.4g},{al_den:.4g},{al_cong:.4g}) "
            f"scale_err(w*sur - plc)=({err_wl:+.4g},{err_den:+.4g},{err_cong:+.4g}) "
            f"pv={px_proxy:.4g} loss_terms_sum={al_sum:.4g} pv-sum={loss_minus_proxy:+.4g}",
            flush=True,
        )
        return

    d_last = (px_proxy - px_proxy_prev) if px_proxy_prev is not None else None
    d_start = (px_proxy - px_proxy_start) if px_proxy_start is not None else None
    d_last_s = f"{d_last:+.4g}" if d_last is not None else "n/a"
    d_start_s = f"{d_start:+.4g}" if d_start is not None else "n/a"
    start_s = f"{px_proxy_start:.4g}" if px_proxy_start is not None else "n/a"
    best_s = f"{px_proxy_best:.4g}" if px_proxy_best is not None else "n/a"

    _gpu_log(f"[gpu_placer] proxy_check epoch={epoch}", flush=True)
    print(
        f"  proxy: {px_proxy:.4g}  (start {start_s}, last {d_last_s}, "
        f"total {d_start_s}, best {best_s})",
        flush=True,
    )
    print(
        f"  surrogate_total: {al_sum:.4g}  vs_proxy: {loss_minus_proxy:+.4g}  "
        f"({_PROXY_DEN_COEF}*wl + {_PROXY_DEN_COEF}*den + {_PROXY_CONG_COEF}*cong after w scaling)",
        flush=True,
    )
    print("  term        sur      plc    err      w     note", flush=True)

    terms = [
        ("wl", sur_wl, px_wl, err_wl, w_wl),
        ("den", sur_den, px_den, err_den, w_den),
        ("cong", sur_cong, px_cong, err_cong, w_cong),
    ]
    worst = max(terms, key=lambda t: abs(t[3]))
    for name, sur, plc, err, w in terms:
        note = " <- largest gap" if name == worst[0] else ""
        print(
            f"  {name:<8s}{sur:9.4g} {plc:9.4g} {err:+8.4g} {w:6.4g}{note}",
            flush=True,
        )


class _CudaSectionTimer:
    """CUDA event timing for training-loop slices (avoids torch.profiler stalls)."""

    __slots__ = ("enabled", "totals")

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = bool(enabled and device.type == "cuda")
        self.totals: dict[str, float] = defaultdict(float)

    @contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        try:
            yield
        finally:
            ev1.record()
            torch.cuda.synchronize()
            self.totals[name] += float(ev0.elapsed_time(ev1))

    def report(self, epochs_completed: int) -> None:
        if not _gpu_placer_verbose() or not self.enabled or not self.totals:
            return
        n = max(1, int(epochs_completed))
        print(
            "[gpu_placer] CUDA section timing - mean ms/epoch "
            "(wl/den/cong/ovl forwards, backward, optim, clamp); excludes PLC proxy CPU work:",
            flush=True,
        )
        order = ("wl_fwd", "den_fwd", "cong_fwd", "ovl_fwd", "backward", "optim", "clamp")
        for k in order:
            if k in self.totals:
                print(f"  {k}: {self.totals[k] / n:.4f}", flush=True)
        for k in sorted(self.totals):
            if k not in order:
                print(f"  {k}: {self.totals[k] / n:.4f}", flush=True)


class GpuPlacer:
    """
    GPU-first differentiable placer.

    - Optimizes smooth-HPWL + soft density.
    - Keeps macros within the canvas via clamping.
    - Does not enforce non-overlap.
    - Optional ``affine_calibrate_density`` (default ``True``): set ``False`` with ``affine_calibrate=True``
      to run WL/congestion PLC calibration while holding ``w_den_run`` at ``self.w_density``.
    - Optional PLC-proxy early-stop: ``stagnation_min_abs_improvement`` (see module
      doc / ``MACRO_PLACE_GPU_STAGNATION_MIN_ABS_IMPROVEMENT``).
    - Training curves under ``vis/`` are written by default (disable with env
      ``MACRO_PLACE_GPU_TRAINING_PLOT=0``).
    """

    def __init__(
        self,
        *,
        epochs: int | None = None,
        lr: float = 2e-2,
        beta: float = 1.0,
        w_wl: float = 1.0,
        w_density: float = 0.5,
        w_cong: float = 0.5,
        w_overlap: float = 1.0,
        affine_calibrate: bool | None = None,
        affine_calibrate_density: bool | None = None,
        target_density: float = 0.7,
        seed: int = 0,
        log_every: int = 50,
        stagnation_proxy_patience: int | None = None,
        stagnation_proxy_rel_delta: float | None = None,
        stagnation_min_abs_improvement: float | None = None,
        stagnation_surrogate_patience: int | None = None,
        stagnation_surrogate_min_abs: float | None = None,
        stagnation_surrogate_min_rel_initial: float | None = None,
        surrogate_stagnation_check_every: int | None = None,
        w_density_schedule: Callable[[int], float] | None = None,
        restore_best_proxy_placement: bool | None = None,
        late_stagnation_sgd_switch: bool = False,
        use_spatial_cong: bool | None = None,
        use_plc_net_routing: bool | None = None,
    ) -> None:
        self.epochs = int(epochs) if epochs is not None else int(
            os.environ.get("MACRO_PLACE_GPU_EPOCHS", "5000")
        )
        if stagnation_proxy_patience is None:
            stagnation_proxy_patience = int(
                os.environ.get("MACRO_PLACE_GPU_STAGNATION_PATIENCE", "0") or "0"
            )
        if stagnation_proxy_rel_delta is None:
            stagnation_proxy_rel_delta = float(
                os.environ.get("MACRO_PLACE_GPU_STAGNATION_REL_DELTA", "0.01") or "0.01"
            )
        if stagnation_min_abs_improvement is None:
            if "MACRO_PLACE_PE_STAGNATION_MIN_ABS_IMPROVEMENT" in os.environ:
                raw_pe = os.environ.get("MACRO_PLACE_PE_STAGNATION_MIN_ABS_IMPROVEMENT", "") or ""
                stagnation_min_abs_improvement = float(raw_pe.strip() or "0")
            else:
                stagnation_min_abs_improvement = float(
                    os.environ.get("MACRO_PLACE_GPU_STAGNATION_MIN_ABS_IMPROVEMENT", "0.01")
                    or "0.01"
                )
        self.stagnation_proxy_patience = max(0, int(stagnation_proxy_patience))
        self.stagnation_proxy_rel_delta = float(stagnation_proxy_rel_delta)
        self.stagnation_min_abs_improvement = float(stagnation_min_abs_improvement)
        if stagnation_surrogate_patience is None:
            stagnation_surrogate_patience = int(
                os.environ.get("MACRO_PLACE_GPU_SURROGATE_STAGNATION_PATIENCE", "0") or "0"
            )
        if stagnation_surrogate_min_abs is None:
            stagnation_surrogate_min_abs = float(
                os.environ.get("MACRO_PLACE_GPU_SURROGATE_STAGNATION_MIN_ABS", "0") or "0"
            )
        self.stagnation_surrogate_patience = max(0, int(stagnation_surrogate_patience))
        self.stagnation_surrogate_min_abs = float(stagnation_surrogate_min_abs)
        if stagnation_surrogate_min_rel_initial is None:
            stagnation_surrogate_min_rel_initial = float(
                os.environ.get("MACRO_PLACE_GPU_SURROGATE_STAGNATION_MIN_REL_INITIAL", "0") or "0"
            )
        self.stagnation_surrogate_min_rel_initial = float(stagnation_surrogate_min_rel_initial)
        if surrogate_stagnation_check_every is None:
            surrogate_stagnation_check_every = int(
                os.environ.get("MACRO_PLACE_GPU_SURROGATE_STAGNATION_CHECK_EVERY", "0") or "0"
            )
        self.surrogate_stagnation_check_every = max(0, int(surrogate_stagnation_check_every))
        self.lr = float(lr)
        self.beta = float(beta)
        self.w_wl = float(w_wl)
        self.w_density = float(w_density)
        self.w_cong = float(w_cong)
        self.w_overlap = float(w_overlap)
        if affine_calibrate is None:
            affine_calibrate = (os.environ.get("MACRO_PLACE_GPU_AFFINE_CALIB", "0") or "0").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        self.affine_calibrate = bool(affine_calibrate)
        if affine_calibrate_density is None:
            affine_calibrate_density = True
        self.affine_calibrate_density = bool(affine_calibrate_density)
        self._w_wl_default = float(w_wl)
        self._w_density_default = float(w_density)
        self._w_cong_default = float(w_cong)
        self.target_density = float(target_density)
        self.seed = int(seed)
        self.log_every = int(log_every)
        self._w_density_schedule = w_density_schedule
        if restore_best_proxy_placement is None:
            restore_best_proxy_placement = self.stagnation_min_abs_improvement > 0.0
        self.restore_best_proxy_placement = bool(restore_best_proxy_placement)
        self.late_stagnation_sgd_switch = bool(late_stagnation_sgd_switch)
        if use_spatial_cong is None:
            use_spatial_cong = (
                os.environ.get("MACRO_PLACE_GPU_SPATIAL_CONG", "1") or "1"
            ).strip().lower() not in ("0", "false", "no", "off")
        self.use_spatial_cong = bool(use_spatial_cong)
        if use_plc_net_routing is None:
            use_plc_net_routing = (
                os.environ.get("MACRO_PLACE_GPU_PLC_NET_ROUTING", "1") or "1"
            ).strip().lower() not in ("0", "false", "no", "off")
        self.use_plc_net_routing = bool(use_plc_net_routing)
        self._hotspot_map_alpha = float(
            os.environ.get("MACRO_PLACE_GPU_HOTSPOT_ALPHA", "0.3") or "0.3"
        )
        self._hotspot_scale = float(
            os.environ.get("MACRO_PLACE_GPU_HOTSPOT_SCALE", "0.1") or "0.1"
        )
        self._hotspot_min_epoch = int(
            os.environ.get("MACRO_PLACE_GPU_HOTSPOT_MIN_EPOCH", "200") or "200"
        )

    def place(
        self,
        benchmark: Benchmark,
        *,
        initial_macro_positions: torch.Tensor | None = None,
        telemetry: dict | None = None,
        epoch_display_base: int = 0,
        epoch_display_cap: int | None = None,
        plc_proxy_include_epoch_zero: bool = True,
        on_proxy_check: Callable[..., None] | None = None,
    ) -> torch.Tensor:
        """If ``telemetry`` is a dict, it is filled with ``epochs_completed`` (int) and
        ``wall_seconds`` (float) for the differentiable training loop only (``_run_on_device``,
        including OOM CPU fallback re-run), excluding matplotlib plot I/O at the end.

        ``epoch_display_base`` / ``epoch_display_cap`` shift printed / logged epoch indices (e.g. two
        consecutive ``GpuPlacer`` phases with a shared cumulative cap). ``plc_proxy_include_epoch_zero``
        mirrors the legacy ``epoch==0`` PLC proxy probe; set ``False`` to only check on the usual
        ``MACRO_PLACE_GPU_PROXY_CHECK_EVERY`` grid.
        """
        torch.manual_seed(self.seed)

        # Prefer CUDA if available; allow env override via existing helper.
        device = _select_device(benchmark)
        # Frozen reference for CUDA->CPU fallback: ``_run_on_device`` must detect when
        # ``run_device`` differs from the device used to allocate ``pos_*`` / constants.
        # Do not compare to ``device`` after the except block may reassign it to CPU.
        param_device = device
        if device.type != "cuda" and torch.cuda.is_available():
            # If user didn't force cpu, _select_device will choose cuda. If we end
            # up here, they likely forced cpu; keep it as-is.
            pass
        if device.type == "cuda":
            # Reduce the chance of catastrophic driver crashes by failing fast on
            # async CUDA errors and by limiting the size of soft-density graphs.
            # (These are safe no-ops on CPU.)
            os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

        dtype = torch.float32
        n_hard = int(benchmark.num_hard_macros)
        n_macros = int(benchmark.num_macros)
        n_soft = int(benchmark.num_soft_macros)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        fixed = benchmark.macro_fixed

        movable_hard_idx = [i for i in range(n_hard) if not bool(fixed[i].item())]
        if initial_macro_positions is not None:
            orig = initial_macro_positions.to(device=device, dtype=dtype)
        else:
            orig = benchmark.macro_positions.to(device=device, dtype=dtype)

        if not movable_hard_idx and n_soft == 0:
            if telemetry is not None:
                telemetry["epochs_completed"] = 0
                telemetry["wall_seconds"] = 0.0
            return benchmark.macro_positions.clone()

        if movable_hard_idx:
            pos_hard = nn.Parameter(torch.stack([orig[i].clone() for i in movable_hard_idx]))
        else:
            pos_hard = nn.Parameter(torch.zeros(0, 2, device=device, dtype=dtype))
        if n_soft > 0:
            pos_soft = nn.Parameter(orig[n_hard:n_macros].clone())
        else:
            pos_soft = nn.Parameter(torch.zeros(0, 2, device=device, dtype=dtype))

        params: list[nn.Parameter] = []
        if pos_hard.numel() > 0:
            params.append(pos_hard)
        if pos_soft.numel() > 0:
            params.append(pos_soft)

        opt = _make_gpu_optimizer(params, self.lr)

        def _rebuild_optimizer() -> None:
            nonlocal opt
            ps: list[nn.Parameter] = []
            if pos_hard.numel() > 0:
                ps.append(pos_hard)
            if pos_soft.numel() > 0:
                ps.append(pos_soft)
            opt = _make_gpu_optimizer(ps, self.lr)

        # PLC is used for proxy logging and for matching PlacementCost WL normalization:
        # get_cost() == get_wirelength() / ((W+H) * plc.net_cnt)
        proxy_check_every = int(os.environ.get("MACRO_PLACE_GPU_PROXY_CHECK_EVERY", "50") or "50")
        plc = _try_load_plc_iccad04(benchmark)

        net_cnt = float(benchmark.num_nets)
        if plc is not None:
            nc = float(getattr(plc, "net_cnt", 0.0) or 0.0)
            if nc > 0.0:
                net_cnt = nc
        wl_norm = (cw + ch) * max(net_cnt, 1e-9)

        use_pin_wl = len(benchmark.net_pin_nodes) == int(benchmark.num_nets)
        use_rudy_cong = (os.environ.get("MACRO_PLACE_GPU_USE_RUDY_CONG", "1") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        use_bbox_rudy = (os.environ.get("MACRO_PLACE_GPU_USE_BBOX_RUDY", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        pin_cong_off = (os.environ.get("MACRO_PLACE_GPU_PIN_CONG", "1") or "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        )
        use_pin_cong = (
            use_pin_wl
            and use_rudy_cong
            and not use_bbox_rudy
            and not pin_cong_off
        )

        net_weights_raw = benchmark.net_weights.to(device=device, dtype=dtype)

        # Build constant tensors once on-device.
        ports = benchmark.port_positions.to(device=device, dtype=dtype)
        net_idx_macro, net_mask_macro, net_weights_macro, net_valid_macro = _build_net_tensors(
            benchmark, device, dtype
        )
        net_idx_pin_cong = net_mask_pin_cong = net_valid_pin_cong = None
        if use_pin_wl or use_pin_cong:
            net_idx_pin_cong, net_mask_pin_cong, _, net_valid_pin_cong = _build_pin_net_tensors(
                benchmark, device, dtype
            )
        if use_pin_wl:
            net_idx_wl, net_mask_wl, net_weights_wl, net_valid_wl = _build_pin_net_tensors(
                benchmark, device, dtype
            )
        else:
            net_idx_wl, net_mask_wl, net_weights_wl, net_valid_wl = (
                net_idx_macro,
                net_mask_macro,
                net_weights_macro,
                net_valid_macro,
            )

        nr = max(int(benchmark.grid_rows), 1)
        nc = max(int(benchmark.grid_cols), 1)

        assemble_ctx = _build_assemble_ctx(benchmark, device, dtype)
        clamp_ctx = _build_clamp_ctx(
            benchmark, assemble_ctx[1], n_hard, cw, ch, device, dtype
        )

        # Density: PLC-style bin overlap occupancy + proxy reduction (see PlacementCost density).
        dens_chunk = int(os.environ.get("MACRO_PLACE_GPU_DENSITY_CHUNK", "16") or "16")
        dens_ckpt_env = os.environ.get("MACRO_PLACE_GPU_DENSITY_CHECKPOINT", "").strip().lower()
        if dens_ckpt_env in ("1", "true", "yes", "on"):
            dens_use_ckpt = True
        elif dens_ckpt_env in ("0", "false", "no", "off"):
            dens_use_ckpt = False
        else:
            # Default off: checkpointed backward on CUDA has been a common source
            # of illegal-memory-access reports; enable explicitly with =1 if VRAM
            # requires it (often with a smaller MACRO_PLACE_GPU_DENSITY_CHUNK).
            dens_use_ckpt = False

        _dm_raw = (os.environ.get("MACRO_PLACE_GPU_DENSITY_MODEL") or "plc").strip().lower()
        density_model = _dm_raw if _dm_raw in ("plc", "electrostatic", "both") else "plc"

        # Congestion: PLC L-route surrogate (pin-aware by default) or legacy RUDY.
        smooth_cong = int(benchmark.congestion_smooth_range)

        if use_rudy_cong:
            cell_w = cw / nc
            cell_h = ch / nr
            bin_x0 = (torch.arange(nc, device=device, dtype=dtype) * cell_w).view(1, nc).expand(nr, nc)
            bin_x1 = bin_x0 + cell_w
            bin_y0 = (torch.arange(nr, device=device, dtype=dtype) * cell_h).view(nr, 1).expand(nr, nc)
            bin_y1 = bin_y0 + cell_h
            grid_h_routes, grid_v_routes = grid_routing_capacities(benchmark)
            gh = torch.tensor(max(float(grid_h_routes), 1e-30), device=device, dtype=dtype)
            gv = torch.tensor(max(float(grid_v_routes), 1e-30), device=device, dtype=dtype)
            h_ma = float(benchmark.hrouting_alloc)
            v_ma = float(benchmark.vrouting_alloc)
        else:
            bin_x0 = bin_x1 = bin_y0 = bin_y1 = None
            gh = gv = None
            h_ma = v_ma = 0.0

        sizes = benchmark.macro_sizes.to(device=device, dtype=dtype)
        fixed_mask = benchmark.macro_fixed.to(device=device)
        eps_canvas = torch.tensor(max(cw * ch, 1e-12), device=device, dtype=dtype)

        log_path = Path(os.environ.get("MACRO_PLACE_GPU_LOG_PATH", "gpu_proxycheck.csv"))
        log_fp = None
        proxy_log_mode = _proxy_log_mode()
        proxy_csv = os.environ.get("MACRO_PLACE_GPU_PROXY_CSV", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if (
            plc is not None
            and proxy_check_every > 0
            and (proxy_csv or _gpu_placer_verbose())
        ):
            log_fp = open(log_path, "a", encoding="utf-8")
            if log_fp.tell() == 0:
                log_fp.write(
                    "epoch,"
                    "sur_wl,sur_den,sur_cong,sur_ovl,sur_loss,"
                    "px_proxy,px_wl,px_den,px_cong,px_ovl_pairs,"
                    "d_wl,d_den,d_cong,d_proxy,"
                    "px_proxy_start,d_px_last,d_px_start,px_proxy_best,"
                    "loss_terms_sum,loss_minus_proxy,"
                    "err_wl,err_den,err_cong,"
                    "w_wl,w_den,w_cong\n"
                )
                log_fp.flush()

        proxy_epochs: list[int] = []
        proxy_values: list[float] = []
        surrogate_epochs: list[int] = []
        surrogate_losses: list[float] = []
        run_meta: dict[str, int] = {"epochs_completed": 0}

        _ed_base = int(epoch_display_base)
        _ed_cap = int(epoch_display_cap) if epoch_display_cap is not None else None
        _plc_proxy_e0 = bool(plc_proxy_include_epoch_zero)

        def _run_on_device(run_device: torch.device) -> torch.Tensor:
            nonlocal opt
            nonlocal ports
            nonlocal net_idx_macro, net_mask_macro, net_weights_macro, net_valid_macro
            nonlocal net_idx_pin_cong, net_mask_pin_cong, net_valid_pin_cong
            nonlocal net_weights_raw
            nonlocal net_idx_wl, net_mask_wl, net_weights_wl, net_valid_wl
            nonlocal assemble_ctx, clamp_ctx
            nonlocal bin_x0, bin_x1, bin_y0, bin_y1
            nonlocal sizes, fixed_mask, eps_canvas
            nonlocal gh, gv, smooth_cong, h_ma, v_ma
            if run_device != param_device:
                # Rebuild all constant tensors on the fallback device.
                ports = benchmark.port_positions.to(device=run_device, dtype=dtype)
                net_idx_macro, net_mask_macro, net_weights_macro, net_valid_macro = _build_net_tensors(
                    benchmark, run_device, dtype
                )
                net_weights_raw = benchmark.net_weights.to(device=run_device, dtype=dtype)
                if use_pin_wl or use_pin_cong:
                    net_idx_pin_cong, net_mask_pin_cong, _, net_valid_pin_cong = _build_pin_net_tensors(
                        benchmark, run_device, dtype
                    )
                if use_pin_wl:
                    net_idx_wl, net_mask_wl, net_weights_wl, net_valid_wl = _build_pin_net_tensors(
                        benchmark, run_device, dtype
                    )
                else:
                    net_idx_wl, net_mask_wl, net_weights_wl, net_valid_wl = (
                        net_idx_macro,
                        net_mask_macro,
                        net_weights_macro,
                        net_valid_macro,
                    )
                assemble_ctx = _build_assemble_ctx(benchmark, run_device, dtype)
                clamp_ctx = _build_clamp_ctx(
                    benchmark, assemble_ctx[1], n_hard, cw, ch, run_device, dtype
                )
                if use_rudy_cong:
                    cell_w_fb = cw / nc
                    cell_h_fb = ch / nr
                    bin_x0 = (
                        torch.arange(nc, device=run_device, dtype=dtype) * cell_w_fb
                    ).view(1, nc).expand(nr, nc)
                    bin_x1 = bin_x0 + cell_w_fb
                    bin_y0 = (
                        torch.arange(nr, device=run_device, dtype=dtype) * cell_h_fb
                    ).view(nr, 1).expand(nr, nc)
                    bin_y1 = bin_y0 + cell_h_fb
                    grid_h_routes, grid_v_routes = grid_routing_capacities(benchmark)
                    gh = torch.tensor(max(float(grid_h_routes), 1e-30), device=run_device, dtype=dtype)
                    gv = torch.tensor(max(float(grid_v_routes), 1e-30), device=run_device, dtype=dtype)
                sizes = benchmark.macro_sizes.to(device=run_device, dtype=dtype)
                fixed_mask = benchmark.macro_fixed.to(device=run_device)
                eps_canvas = torch.tensor(max(cw * ch, 1e-12), device=run_device, dtype=dtype)
                if pos_hard.numel() > 0:
                    pos_hard.data = pos_hard.data.to(run_device)
                if pos_soft.numel() > 0:
                    pos_soft.data = pos_soft.data.to(run_device)
                # Adam state must live on the same device as parameters after migration.
                _rebuild_optimizer()

            stag_best: float | None = None
            stag_streak = 0
            switched_to_sgd = False
            _stag_patience_cap = (
                max(1, int(self.stagnation_proxy_patience))
                if self.stagnation_proxy_patience > 0
                else 1
            )
            stagnation_switch_threshold = max(1, _stag_patience_cap - 1)
            surr_stag_best: float | None = None
            surr_stag_streak = 0
            surr_stag_min_abs: float | None = None
            surr_check_every = int(self.surrogate_stagnation_check_every)
            surr_stag_enabled = (
                self.stagnation_surrogate_min_abs > 0.0
                or self.stagnation_surrogate_min_rel_initial > 0.0
            )
            if surr_check_every <= 0 and surr_stag_enabled:
                surr_check_every = max(1, int(self.log_every))
            px_proxy_start: float | None = None
            px_proxy_prev: float | None = None
            px_proxy_best: float | None = None
            best_proxy_restore: float | None = None
            best_hard_restore: torch.Tensor | None = None
            best_soft_restore: torch.Tensor | None = None
            _restore_best = bool(self.restore_best_proxy_placement)

            w_wl_run = float(self.w_wl)
            w_den_run = float(self.w_density)
            w_cong_run = 1.0
            # EMA state for congestion scale (affine_calibrate); first proxy check pulls toward px/sur.
            ema_w_cong = 1.0
            # EMA state for WL scale; init near constructor w_wl (often close to px/sur).
            ema_w_wl = float(self.w_wl)
            w_ovl_run = float(self.w_overlap)
            affine_prev: dict[str, float] | None = None

            _spatial_cong_env = (
                os.environ.get("MACRO_PLACE_GPU_SPATIAL_CONG", "1") or "1"
            ).strip().lower()
            _use_spatial_cong = self.use_spatial_cong and _spatial_cong_env not in (
                "0",
                "false",
                "no",
                "off",
            )
            _hotspot_h_weight = float(
                os.environ.get("MACRO_PLACE_GPU_HOTSPOT_H_WEIGHT", "2.0") or "2.0"
            )
            _hotspot_v_weight = float(
                os.environ.get("MACRO_PLACE_GPU_HOTSPOT_V_WEIGHT", "1.0") or "1.0"
            )
            _hotspot_map: torch.Tensor | None = None
            _hotspot_min_epoch = int(self._hotspot_min_epoch)

            plc_net_h: torch.Tensor | None = None
            plc_net_v: torch.Tensor | None = None
            _plc_net_env = (
                os.environ.get("MACRO_PLACE_GPU_PLC_NET_ROUTING", "1") or "1"
            ).strip().lower() not in ("0", "false", "no", "off")
            _use_plc_net_routing = (
                self.use_plc_net_routing
                and _plc_net_env
                and use_rudy_cong
                and not use_bbox_rudy
            )

            proxy_epochs.clear()
            proxy_values.clear()
            surrogate_epochs.clear()
            surrogate_losses.clear()

            last_good_hard: torch.Tensor | None = (
                pos_hard.detach().clone() if pos_hard.numel() > 0 else None
            )
            last_good_soft: torch.Tensor | None = (
                pos_soft.detach().clone() if pos_soft.numel() > 0 else None
            )

            def _restore_last_good_params() -> None:
                if last_good_hard is not None and pos_hard.numel() > 0:
                    pos_hard.data.copy_(last_good_hard.to(pos_hard.device))
                if last_good_soft is not None and pos_soft.numel() > 0:
                    pos_soft.data.copy_(last_good_soft.to(pos_soft.device))

            def _snapshot_good_params() -> None:
                nonlocal last_good_hard, last_good_soft
                if pos_hard.numel() > 0 and torch.isfinite(pos_hard).all():
                    last_good_hard = pos_hard.detach().clone()
                if pos_soft.numel() > 0 and torch.isfinite(pos_soft).all():
                    last_good_soft = pos_soft.detach().clone()

            ovl_use_triton = self.w_overlap != 0.0 and want_triton_overlap(run_device, dtype)
            if self.w_overlap != 0.0 and ovl_use_triton:
                ovl_pair_i, ovl_pair_j = build_overlap_pair_indices(
                    fixed_mask, device=run_device
                )
            else:
                z0 = torch.zeros(0, dtype=torch.int32, device=run_device)
                ovl_pair_i = ovl_pair_j = z0

            def _overlap_loss(full_pos: torch.Tensor) -> torch.Tensor:
                # Soft overlap penalty; optional Triton forward (backward stays PyTorch).
                if full_pos.shape[0] <= 1:
                    return torch.zeros((), device=full_pos.device, dtype=full_pos.dtype)
                return pairwise_overlap_sum_normalized(
                    full_pos,
                    sizes,
                    fixed_mask,
                    eps_canvas,
                    ovl_pair_i,
                    ovl_pair_j,
                    use_triton=ovl_use_triton,
                )

            cuda_sect = (os.environ.get("MACRO_PLACE_GPU_PROFILE_SECTIONS") or "").strip().lower()
            cuda_timer = _CudaSectionTimer(
                cuda_sect in ("1", "true", "yes", "on"),
                run_device,
            )

            _epoch_cap_disp = _ed_cap if _ed_cap is not None else _ed_base + self.epochs

            _use_den_schedule = (
                density_model == "electrostatic" and self._w_density_schedule is not None
            )

            l_ovl = torch.zeros((), device=run_device, dtype=dtype)

            for epoch in range(self.epochs):
                ge = _ed_base + epoch + 1
                opt.zero_grad(set_to_none=True)
                if _use_den_schedule:
                    w_den_run = float(self._w_density_schedule(epoch))

                with cuda_timer.span("wl_fwd"):
                    full = _assemble_full_fast(pos_hard, pos_soft, assemble_ctx)
                    combined_pos = torch.cat([full, ports], dim=0)

                    if use_pin_wl:
                        pin_owner = net_idx_wl[:, :, 0]
                        pin_slot = net_idx_wl[:, :, 1]
                        pin_flat = _pin_positions_for_hpwl(
                            combined_pos,
                            pin_owner,
                            pin_slot,
                            n_hard=n_hard,
                            macro_pin_offsets=benchmark.macro_pin_offsets,
                            dtype=dtype,
                        )
                        num_nets_wl = int(net_idx_wl.shape[0])
                        max_pins_wl = int(net_idx_wl.shape[1])
                        wl_pos = pin_flat.reshape(num_nets_wl * max_pins_wl, 2)
                        wl_idx = (
                            torch.arange(num_nets_wl, device=run_device, dtype=torch.long).view(-1, 1)
                            * max_pins_wl
                            + torch.arange(max_pins_wl, device=run_device, dtype=torch.long).view(1, -1)
                        )
                    else:
                        wl_pos = combined_pos
                        wl_idx = net_idx_wl

                    l_wl_raw = _wirelength_loss_v2(
                        wl_pos,
                        wl_idx,
                        net_mask_wl,
                        net_weights_wl,
                        net_valid_wl,
                        self.beta,
                    )
                    l_wl = l_wl_raw / wl_norm

                # Density: PLC overlap grid + proxy-style top-10% scalar (×0.5).
                with cuda_timer.span("den_fwd"):
                    _compute_den = self.w_density != 0.0 or _use_den_schedule
                    if _compute_den:
                        if density_model == "electrostatic":
                            l_den = _electrostatic_density_loss(
                                full, benchmark, nr, nc, cw, ch
                            )
                        elif density_model == "both":
                            dens_grid = _plc_macro_overlap_density_grid_checkpointed(
                                full,
                                benchmark,
                                cw,
                                ch,
                                nr,
                                nc,
                                chunk=dens_chunk,
                                use_checkpoint=(
                                    dens_use_ckpt and run_device.type == "cuda"
                                ),
                            )
                            l_den_plc = _plc_proxy_density_cost(dens_grid)
                            l_den_es = _electrostatic_density_loss(
                                full, benchmark, nr, nc, cw, ch
                            )
                            l_den = 0.5 * l_den_plc + 0.5 * l_den_es
                        else:
                            dens_grid = _plc_macro_overlap_density_grid_checkpointed(
                                full,
                                benchmark,
                                cw,
                                ch,
                                nr,
                                nc,
                                chunk=dens_chunk,
                                use_checkpoint=(
                                    dens_use_ckpt and run_device.type == "cuda"
                                ),
                            )
                            l_den = _plc_proxy_density_cost(dens_grid)
                    else:
                        l_den = torch.zeros((), device=run_device, dtype=dtype)

                # Congestion: bbox RUDY, pin L-routes (default when pin tables exist), or macro L-routes.
                with cuda_timer.span("cong_fwd"):
                    _spatial_cong_active = (
                        _use_spatial_cong
                        and self.affine_calibrate
                        and _hotspot_map is not None
                        and ge >= _hotspot_min_epoch
                        and self.w_cong != 0.0
                        and use_rudy_cong
                        and not use_bbox_rudy
                    )
                    if self.w_cong != 0.0:
                        if use_rudy_cong and use_bbox_rudy:
                            demand = _rudy_demand_grid(
                                combined_pos,
                                net_idx_macro,
                                net_mask_macro,
                                net_weights_macro,
                                net_valid_macro,
                                self.beta,
                                bin_x0,
                                bin_x1,
                                bin_y0,
                                bin_y1,
                            )
                            H_net = demand / gh
                            V_net = demand / gv
                            H_net, V_net = smooth_routing_cong_plc(H_net, V_net, smooth_cong)
                            h_blk, v_blk = _macro_blockage_raw(
                                full,
                                benchmark,
                                nr,
                                nc,
                                cw,
                                ch,
                                n_hard,
                                h_ma,
                                v_ma,
                            )
                            H_tot = H_net + (h_blk / gh)
                            V_tot = V_net + (v_blk / gv)
                            both = torch.cat([V_tot.reshape(-1), H_tot.reshape(-1)])
                            l_cong = _abu_top_mean(both, 0.05)
                        elif (
                            _use_plc_net_routing
                            and plc_net_h is not None
                            and plc_net_v is not None
                        ):
                            h_blk, v_blk = _macro_blockage_raw(
                                full,
                                benchmark,
                                nr,
                                nc,
                                cw,
                                ch,
                                n_hard,
                                h_ma,
                                v_ma,
                            )
                            h_blk_norm = h_blk / gh.clamp(min=1e-9)
                            v_blk_norm = v_blk / gv.clamp(min=1e-9)
                            h_total = h_blk_norm + plc_net_h.detach()
                            v_total = v_blk_norm + plc_net_v.detach()
                            both = torch.cat(
                                [v_total.reshape(-1), h_total.reshape(-1)]
                            )
                            l_cong = _abu_top_mean(both, 0.05)
                            if _spatial_cong_active:
                                sur_cong_grid = h_total + v_total
                                l_cong = l_cong + self._hotspot_scale * (
                                    (sur_cong_grid * _hotspot_map.detach()).sum()
                                    / float(nr * nc)
                                )
                        elif use_pin_cong:
                            if _spatial_cong_active:
                                H_tot, V_tot = plc_routing_surrogate_hv_totals_pins(
                                    combined_pos,
                                    net_idx_pin_cong,
                                    net_mask_pin_cong,
                                    net_valid_pin_cong,
                                    full,
                                    benchmark,
                                    smooth_range=smooth_cong,
                                    nr=nr,
                                    nc=nc,
                                )
                                both = torch.cat(
                                    [V_tot.reshape(-1), H_tot.reshape(-1)]
                                )
                                l_cong = _abu_top_mean(both, 0.05)
                                sur_cong_grid = H_tot + V_tot
                                l_cong = l_cong + self._hotspot_scale * (
                                    (sur_cong_grid * _hotspot_map.detach()).sum()
                                    / float(nr * nc)
                                )
                            else:
                                l_cong = plc_routing_surrogate_scalar_pins(
                                    combined_pos,
                                    net_idx_pin_cong,
                                    net_mask_pin_cong,
                                    net_valid_pin_cong,
                                    full,
                                    benchmark,
                                    smooth_range=smooth_cong,
                                )
                        elif use_rudy_cong:
                            if _spatial_cong_active:
                                H_tot, V_tot = plc_routing_surrogate_hv_totals(
                                    combined_pos,
                                    net_idx_macro,
                                    net_mask_macro,
                                    net_weights_macro,
                                    net_valid_macro,
                                    full,
                                    benchmark,
                                    smooth_range=smooth_cong,
                                    nr=nr,
                                    nc=nc,
                                )
                                both = torch.cat(
                                    [V_tot.reshape(-1), H_tot.reshape(-1)]
                                )
                                l_cong = _abu_top_mean(both, 0.05)
                                sur_cong_grid = H_tot + V_tot
                                l_cong = l_cong + self._hotspot_scale * (
                                    (sur_cong_grid * _hotspot_map.detach()).sum()
                                    / float(nr * nc)
                                )
                            else:
                                l_cong = plc_routing_surrogate_scalar(
                                    combined_pos,
                                    net_idx_macro,
                                    net_mask_macro,
                                    net_weights_macro,
                                    net_valid_macro,
                                    full,
                                    benchmark,
                                    smooth_range=smooth_cong,
                                )
                        else:
                            l_cong = torch.zeros((), device=run_device, dtype=dtype)
                    else:
                        l_cong = torch.zeros((), device=run_device, dtype=dtype)

                # Overlap avoidance: penalize pairwise overlap area.
                with cuda_timer.span("ovl_fwd"):
                    _compute_ovl = self.w_overlap != 0.0 and (
                        epoch < 200 or float(l_ovl.detach()) > 0.005
                    )
                    if _compute_ovl:
                        l_ovl = _overlap_loss(full)
                    else:
                        l_ovl = torch.zeros((), device=run_device, dtype=dtype)

                loss = (
                    w_wl_run * l_wl
                    + _PROXY_DEN_COEF * w_den_run * l_den
                    + _PROXY_CONG_COEF * w_cong_run * l_cong
                    + w_ovl_run * l_ovl
                )
                if not torch.isfinite(loss) or not torch.isfinite(full).all():
                    bad_terms = [
                        name
                        for name, term in (
                            ("wl", l_wl),
                            ("den", l_den),
                            ("cong", l_cong),
                            ("ovl", l_ovl),
                        )
                        if not torch.isfinite(term)
                    ]
                    print(
                        "[gpu_placer] stopped: non-finite surrogate or placement "
                        f"(epoch {ge}); reverting to last good params. "
                        f"non_finite_terms={bad_terms or ['placement']}",
                        flush=True,
                    )
                    _restore_last_good_params()
                    break

                with cuda_timer.span("backward"):
                    loss.backward()
                if density_model == "electrostatic" and params:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                with cuda_timer.span("optim"):
                    opt.step()

                # Keep centers in-bounds (still allows overlaps).
                with cuda_timer.span("clamp"):
                    _clamp_movable_fast(pos_hard, pos_soft, clamp_ctx)

                if not (
                    (pos_hard.numel() == 0 or torch.isfinite(pos_hard).all())
                    and (pos_soft.numel() == 0 or torch.isfinite(pos_soft).all())
                ):
                    print(
                        "[gpu_placer] stopped: non-finite movable parameters after optimizer step "
                        f"(epoch {ge}); reverting to last good params.",
                        flush=True,
                    )
                    _restore_last_good_params()
                    break

                _snapshot_good_params()

                run_meta["epochs_completed"] = epoch + 1

                if (
                    _gpu_placer_verbose()
                    and self.log_every > 0
                    and (epoch == 0 or (epoch + 1) % self.log_every == 0)
                ):
                    lw = float(l_wl.detach().cpu().item())
                    ld = float(l_den.detach().cpu().item())
                    lc = float(l_cong.detach().cpu().item())
                    lo = float(l_ovl.detach().cpu().item())
                    ls = float(loss.detach().cpu().item())
                    surrogate_epochs.append(ge)
                    surrogate_losses.append(ls)
                    wd_note = f" w_den={w_den_run:g}" if _use_den_schedule else ""
                    _gpu_log(
                        f"[gpu_placer] epoch={ge}/{_epoch_cap_disp} "
                        f"loss={ls:.6g} wl={lw:.6g} den={ld:.6g} cong={lc:.6g} ovl={lo:.6g}{wd_note} "
                        f"device={run_device.type} bins={nr}x{nc}"
                    )

                if surr_stag_enabled and surr_check_every > 0:
                    if (epoch + 1) % surr_check_every == 0 or epoch == 0:
                        ls_check = float(loss.detach().cpu().item())
                        surr_patience = (
                            max(1, int(self.stagnation_surrogate_patience))
                            if self.stagnation_surrogate_patience > 0
                            else 1
                        )
                        if surr_stag_best is None:
                            surr_stag_best = ls_check
                            surr_stag_streak = 0
                            rel0 = float(self.stagnation_surrogate_min_rel_initial)
                            if rel0 > 0.0:
                                surr_stag_min_abs = max(1e-12, rel0 * ls_check)
                                _gpu_log(
                                    "[gpu_placer] surrogate stagnation threshold: "
                                    f"{surr_stag_min_abs:.6g} = {rel0:g} × initial_loss {ls_check:.6g}"
                                )
                        else:
                            surr_min_abs = (
                                float(surr_stag_min_abs)
                                if surr_stag_min_abs is not None
                                else float(self.stagnation_surrogate_min_abs)
                            )
                            if surr_min_abs <= 0.0:
                                surr_stag_best = min(float(surr_stag_best), ls_check)
                                continue
                            best_before = float(surr_stag_best)
                            new_best = min(best_before, ls_check)
                            delta = best_before - new_best
                            surr_stag_best = new_best
                            if delta >= surr_min_abs:
                                surr_stag_streak = 0
                            else:
                                surr_stag_streak += 1
                                _gpu_log(
                                    "[gpu_placer] surrogate stagnation: total loss improved only "
                                    f"{delta:.6g} vs best over last {surr_check_every} epochs "
                                    f"(min required {surr_min_abs:g}); "
                                    f"consecutive {surr_stag_streak}/{surr_patience} "
                                    f"best={surr_stag_best:.6g} current={ls_check:.6g}."
                                )
                                if surr_stag_streak >= surr_patience:
                                    break

                # Evaluator proxy check (CPU) for surrogate accuracy diagnostics.
                if (
                    plc is not None
                    and proxy_check_every > 0
                    and (
                        ((epoch + 1) % proxy_check_every == 0)
                        or (_plc_proxy_e0 and epoch == 0)
                    )
                ):
                    with torch.no_grad():
                        if not torch.isfinite(full).all():
                            print(
                                "[gpu_placer] skipping PLC proxy check: non-finite full placement "
                                f"(epoch {ge}).",
                                flush=True,
                            )
                            break
                        full_cpu = full.detach().to(device="cpu", dtype=torch.float32)
                        costs = compute_proxy_cost(full_cpu, benchmark, plc)

                        _need_cong_grids = _use_plc_net_routing or (
                            _use_spatial_cong
                            and self.affine_calibrate
                            and ge >= _hotspot_min_epoch
                        )
                        if _need_cong_grids:
                            try:
                                import numpy as np

                                h_raw_np = np.array(
                                    plc.get_horizontal_routing_congestion(),
                                    dtype=np.float32,
                                ).reshape(nr, nc)
                                v_raw_np = np.array(
                                    plc.get_vertical_routing_congestion(),
                                    dtype=np.float32,
                                ).reshape(nr, nc)
                                h_macro_np = np.array(
                                    plc.H_macro_routing_cong, dtype=np.float32
                                ).reshape(nr, nc)
                                v_macro_np = np.array(
                                    plc.V_macro_routing_cong, dtype=np.float32
                                ).reshape(nr, nc)

                                if _use_plc_net_routing:
                                    h_net_only = np.maximum(
                                        h_raw_np - h_macro_np, 0.0
                                    )
                                    v_net_only = np.maximum(
                                        v_raw_np - v_macro_np, 0.0
                                    )
                                    plc_net_h = torch.from_numpy(h_net_only).to(
                                        device=run_device, dtype=dtype
                                    )
                                    plc_net_v = torch.from_numpy(v_net_only).to(
                                        device=run_device, dtype=dtype
                                    )
                                    _gpu_log(
                                        f"[gpu_placer] plc_net_routing updated: "
                                        f"H_net mean={h_net_only.mean():.3f} "
                                        f"max={h_net_only.max():.3f} "
                                        f"V_net mean={v_net_only.mean():.3f} "
                                        f"max={v_net_only.max():.3f}"
                                    )

                                if (
                                    _use_spatial_cong
                                    and self.affine_calibrate
                                    and ge >= _hotspot_min_epoch
                                ):
                                    h_t = torch.from_numpy(h_raw_np).to(
                                        device=run_device, dtype=dtype
                                    )
                                    v_t = torch.from_numpy(v_raw_np).to(
                                        device=run_device, dtype=dtype
                                    )
                                    h_excess = torch.relu(h_t - 1.0)
                                    v_excess = torch.relu(v_t - 1.0)
                                    new_hotspot = (
                                        h_excess * _hotspot_h_weight
                                        + v_excess * _hotspot_v_weight
                                    )
                                    alpha = float(self._hotspot_map_alpha)
                                    if _hotspot_map is None:
                                        _hotspot_map = new_hotspot
                                    else:
                                        _hotspot_map = (
                                            (1.0 - alpha) * _hotspot_map
                                            + alpha * new_hotspot
                                        )
                                    n_hot_h = int((h_t > 1.0).sum().item())
                                    n_hot_v = int((v_t > 1.0).sum().item())
                                    _gpu_log(
                                        f"[gpu_placer] hotspot_map updated: "
                                        f"H_overloaded={n_hot_h} V_overloaded={n_hot_v} "
                                        f"H_max={h_t.max().item():.3f} "
                                        f"V_max={v_t.max().item():.3f}"
                                    )
                            except Exception as e:
                                _gpu_log(
                                    f"[gpu_placer] plc congestion grids update failed: {e}"
                                )
                                if _use_plc_net_routing:
                                    plc_net_h = None
                                    plc_net_v = None

                        px_proxy = float(costs["proxy_cost"])
                        px_wl = float(costs["wirelength_cost"])
                        px_den = float(costs["density_cost"])
                        px_cong = float(costs["congestion_cost"])
                        px_ovl = int(costs["overlap_count"])

                        sur_wl = float(l_wl.detach().cpu().item())
                        sur_den = float(l_den.detach().cpu().item())
                        sur_cong = float(l_cong.detach().cpu().item())
                        sur_ovl = float(l_ovl.detach().cpu().item())

                        if self.affine_calibrate:
                            p_den = affine_prev["sur_den"] if affine_prev is not None else sur_den
                            p_px_den = affine_prev["px_den"] if affine_prev is not None else px_den
                            ema_w_wl = _fit_scale_wl(sur_wl, px_wl, ema_w_wl)
                            w_wl_run = ema_w_wl
                            if density_model != "electrostatic" and self.affine_calibrate_density:
                                w_den_run = _fit_scale_surrogate_to_proxy(
                                    p_den, p_px_den, sur_den, px_den, 1.0
                                )
                            ema_w_cong = _fit_scale_cong(sur_cong, px_cong, ema_w_cong)
                            w_cong_run = ema_w_cong
                            affine_prev = {"sur_den": sur_den, "px_den": px_den}

                        al_wl = w_wl_run * sur_wl
                        al_den = w_den_run * sur_den
                        al_cong = w_cong_run * sur_cong
                        al_sum = (
                            al_wl
                            + _PROXY_DEN_COEF * al_den
                            + _PROXY_CONG_COEF * al_cong
                        )

                        if px_proxy_start is None:
                            px_proxy_start = float(px_proxy)
                            px_proxy_best = float(px_proxy)
                        else:
                            px_proxy_best = min(float(px_proxy_best), float(px_proxy))
                        d_px_last = (
                            float(px_proxy) - float(px_proxy_prev)
                            if px_proxy_prev is not None
                            else 0.0
                        )
                        d_px_start = float(px_proxy) - float(px_proxy_start)

                        proxy_epochs.append(ge)
                        proxy_values.append(px_proxy)

                        d_wl = px_wl - sur_wl
                        d_den = px_den - sur_den
                        d_cong = px_cong - sur_cong
                        d_proxy = px_proxy - al_sum
                        loss_minus_proxy = al_sum - px_proxy
                        err_wl = al_wl - px_wl
                        err_den = al_den - px_den
                        err_cong = al_cong - px_cong

                        if _restore_best and (
                            best_proxy_restore is None or px_proxy < best_proxy_restore
                        ):
                            best_proxy_restore = float(px_proxy)
                            if pos_hard.numel() > 0:
                                best_hard_restore = pos_hard.detach().clone()
                            if pos_soft.numel() > 0:
                                best_soft_restore = pos_soft.detach().clone()

                        if on_proxy_check is not None:
                            on_proxy_check(
                                epoch=ge,
                                px_proxy=px_proxy,
                                best_proxy=float(
                                    best_proxy_restore
                                    if best_proxy_restore is not None
                                    else px_proxy
                                ),
                                ema_w_wl=float(ema_w_wl),
                                ema_w_cong=float(ema_w_cong),
                                err_wl=float(err_wl),
                                err_cong=float(err_cong),
                                al_wl=float(al_wl),
                                al_cong=float(al_cong),
                                px_wl=float(px_wl),
                                px_cong=float(px_cong),
                            )

                        if proxy_log_mode != "off":
                            _print_proxy_check_report(
                                epoch=ge,
                                px_proxy=px_proxy,
                                px_proxy_start=px_proxy_start,
                                px_proxy_prev=px_proxy_prev,
                                px_proxy_best=px_proxy_best,
                                sur_wl=sur_wl,
                                sur_den=sur_den,
                                sur_cong=sur_cong,
                                px_wl=px_wl,
                                px_den=px_den,
                                px_cong=px_cong,
                                w_wl=w_wl_run,
                                w_den=w_den_run,
                                w_cong=w_cong_run,
                                al_wl=al_wl,
                                al_den=al_den,
                                al_cong=al_cong,
                                al_sum=al_sum,
                                mode=proxy_log_mode,
                            )

                        if _gpu_placer_verbose() and _hotspot_map is not None:
                            hot_bins = int((_hotspot_map > 0.01).sum().item())
                            hot_max = float(_hotspot_map.max().item())
                            _gpu_log(
                                f"[gpu_placer] hotspot: {hot_bins} active bins, "
                                f"max_weight={hot_max:.3f}"
                            )

                        if (
                            _gpu_placer_verbose()
                            and _use_plc_net_routing
                            and plc_net_h is not None
                            and plc_net_v is not None
                        ):
                            h_blk_log, v_blk_log = _macro_blockage_raw(
                                full,
                                benchmark,
                                nr,
                                nc,
                                cw,
                                ch,
                                n_hard,
                                h_ma,
                                v_ma,
                            )
                            h_tot_log = h_blk_log / gh.clamp(min=1e-9) + plc_net_h
                            v_tot_log = v_blk_log / gv.clamp(min=1e-9) + plc_net_v
                            both_log = torch.cat(
                                [v_tot_log.reshape(-1), h_tot_log.reshape(-1)]
                            )
                            sur_cong_plc = float(
                                _abu_top_mean(both_log, 0.05).detach().cpu().item()
                            )
                            err_plc = sur_cong_plc - px_cong
                            err_lroute = sur_cong - px_cong
                            if use_pin_cong:
                                sur_lroute = float(
                                    plc_routing_surrogate_scalar_pins(
                                        combined_pos,
                                        net_idx_pin_cong,
                                        net_mask_pin_cong,
                                        net_valid_pin_cong,
                                        full,
                                        benchmark,
                                        smooth_range=smooth_cong,
                                    )
                                    .detach()
                                    .cpu()
                                    .item()
                                )
                            elif use_rudy_cong:
                                sur_lroute = float(
                                    plc_routing_surrogate_scalar(
                                        combined_pos,
                                        net_idx_macro,
                                        net_mask_macro,
                                        net_weights_macro,
                                        net_valid_macro,
                                        full,
                                        benchmark,
                                        smooth_range=smooth_cong,
                                    )
                                    .detach()
                                    .cpu()
                                    .item()
                                )
                            else:
                                sur_lroute = sur_cong
                            _gpu_log(
                                f"[gpu_placer] plc_net_cong: surrogate={sur_cong_plc:.4f} "
                                f"px_cong={px_cong:.4f} err={err_plc:+.4f} "
                                f"(was {err_lroute:+.4f} with L-route, "
                                f"L-route sur={sur_lroute:.4f})"
                            )

                        px_proxy_prev = float(px_proxy)

                        if log_fp is not None:
                            log_fp.write(
                                f"{ge},"
                                f"{sur_wl:.10g},"
                                f"{sur_den:.10g},"
                                f"{sur_cong:.10g},"
                                f"{float(l_ovl.detach().cpu().item()):.10g},"
                                f"{float(loss.detach().cpu().item()):.10g},"
                                f"{px_proxy:.10g},{px_wl:.10g},{px_den:.10g},{px_cong:.10g},{px_ovl},"
                                f"{d_wl:.10g},{d_den:.10g},{d_cong:.10g},{d_proxy:.10g},"
                                f"{px_proxy_start:.10g},{d_px_last:.10g},{d_px_start:.10g},{px_proxy_best:.10g},"
                                f"{al_sum:.10g},{loss_minus_proxy:.10g},"
                                f"{err_wl:.10g},{err_den:.10g},{err_cong:.10g},"
                                f"{w_wl_run:.10g},{w_den_run:.10g},{w_cong_run:.10g}\n"
                            )
                            log_fp.flush()

                        min_abs = float(self.stagnation_min_abs_improvement)
                        if min_abs > 0.0:
                            min_abs_patience = (
                                max(1, int(self.stagnation_proxy_patience))
                                if self.stagnation_proxy_patience > 0
                                else 1
                            )
                            if stag_best is None:
                                stag_best = float(px_proxy)
                                stag_streak = 0
                            else:
                                delta = float(stag_best) - float(px_proxy)
                                if delta >= min_abs:
                                    stag_best = float(px_proxy)
                                    stag_streak = 0
                                    switched_to_sgd = False
                                else:
                                    stag_streak += 1
                                    _gpu_log(
                                        "[gpu_placer] stagnation: PLC proxy improved only "
                                        f"{delta:.6g} vs best over last "
                                        f"{proxy_check_every} epochs "
                                        f"(min required {min_abs:g}); "
                                        f"consecutive {stag_streak}/{min_abs_patience} "
                                        f"best={stag_best:.6g} current={px_proxy:.6g}."
                                    )
                                    if (
                                        self.late_stagnation_sgd_switch
                                        and not switched_to_sgd
                                        and stag_streak >= stagnation_switch_threshold
                                        and min_abs_patience > 1
                                    ):
                                        ratio_raw = os.environ.get(
                                            "MACRO_PLACE_GPU_SGD_LR_RATIO", "0.2"
                                        )
                                        try:
                                            ratio = float((ratio_raw or "0.2").strip())
                                        except ValueError:
                                            ratio = 0.2
                                        lr_sgd = self.lr * ratio
                                        opt = _switch_to_sgd(params, lr_sgd)
                                        stag_streak = 0
                                        switched_to_sgd = True
                                        _gpu_log(
                                            "[gpu_placer] stagnation counter reset after "
                                            "optimizer switch."
                                        )
                                    elif stag_streak >= min_abs_patience:
                                        break
                        elif self.stagnation_proxy_patience > 0:
                            thr = max(
                                1e-12,
                                abs(float(px_proxy)) * self.stagnation_proxy_rel_delta,
                            )
                            if stag_best is None:
                                stag_best = float(px_proxy)
                                stag_streak = 0
                            elif float(px_proxy) < float(stag_best) - thr:
                                stag_best = float(px_proxy)
                                stag_streak = 0
                            else:
                                stag_streak += 1
                                if stag_streak >= self.stagnation_proxy_patience:
                                    _gpu_log(
                                        "[gpu_placer] stagnation: PLC proxy did not improve by "
                                        f"rel>{self.stagnation_proxy_rel_delta} for "
                                        f"{self.stagnation_proxy_patience} consecutive checks "
                                        f"(best={stag_best:.6g}, current={px_proxy:.6g})."
                                    )
                                    break
            if _restore_best:
                if best_hard_restore is not None and pos_hard.numel() > 0:
                    pos_hard.data.copy_(best_hard_restore.to(pos_hard.device))
                if best_soft_restore is not None and pos_soft.numel() > 0:
                    pos_soft.data.copy_(best_soft_restore.to(pos_soft.device))
                full = _assemble_full_fast(pos_hard, pos_soft, assemble_ctx)
                if best_proxy_restore is not None:
                    _gpu_log(
                        "[gpu_placer] restored best-proxy placement "
                        f"(proxy={best_proxy_restore:.6g}) before exit."
                    )
            cuda_timer.report(int(run_meta.get("epochs_completed", 0)))
            return full

        cuda_fb_env = os.environ.get("MACRO_PLACE_GPU_CUDA_FALLBACK", "1").strip().lower()
        cuda_cpu_fallback = cuda_fb_env not in ("0", "false", "no", "off")

        def _cuda_message_is_oom(err: Exception) -> bool:
            if "out of memory" in str(err).lower():
                return True
            c = getattr(err, "__cause__", None)
            return bool(c and "out of memory" in str(c).lower())

        t_train0 = time.perf_counter()
        try:
            full = _run_on_device(device)
        except RuntimeError as e:
            is_oom = _cuda_message_is_oom(e)
            if device.type == "cuda" and cuda_cpu_fallback and is_oom:
                print(
                    "[gpu_placer] CUDA OOM; falling back to CPU for stability.",
                    flush=True,
                )
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                device = torch.device("cpu")
                full = _run_on_device(device)
            elif device.type == "cuda" and cuda_cpu_fallback and not is_oom and (
                "cuda" in str(e).lower() or "accelerator" in type(e).__name__.lower()
            ):
                raise RuntimeError(
                    "[gpu_placer] CUDA failure that is not a clear out-of-memory condition; "
                    "not falling back to CPU (the GPU context is often poisoned, and copying "
                    "parameters to CPU usually hits the same error). Start a fresh process, "
                    "try MACRO_PLACE_GPU_DENSITY_CHUNK smaller, CUDA_LAUNCH_BLOCKING=1 once to "
                    "localize, or set MACRO_PLACE_GPU_CUDA_FALLBACK=0 to skip this message path. "
                    f"Original: {e}"
                ) from e
            else:
                if device.type == "cuda" and not cuda_cpu_fallback and (
                    "cuda" in str(e).lower() or "accelerator" in type(e).__name__.lower()
                ):
                    raise RuntimeError(
                        "[gpu_placer] CUDA error with MACRO_PLACE_GPU_CUDA_FALLBACK=0 "
                        "(CPU fallback disabled). Mitigations: ensure density checkpoint stays off "
                        "(default), set MACRO_PLACE_GPU_DENSITY_CHECKPOINT=0 explicitly, reduce "
                        "MACRO_PLACE_GPU_DENSITY_CHUNK or batch size, run a fresh process after "
                        "VRAM pressure, or set CUDA_LAUNCH_BLOCKING=1 once to localize the op."
                    ) from e
                raise
        t_train1 = time.perf_counter()
        if device.type == "cuda":
            _cuda_teardown_after_training()
        if telemetry is not None:
            telemetry["epochs_completed"] = int(run_meta["epochs_completed"])
            telemetry["wall_seconds"] = float(t_train1 - t_train0)

        plc_loaded = plc is not None
        saved_proxy = _save_proxy_vs_epoch_plot(benchmark.name, proxy_epochs, proxy_values)
        saved_surr = _save_surrogate_loss_vs_epoch_plot(
            benchmark.name, surrogate_epochs, surrogate_losses
        )
        _report_training_plots(
            plc_loaded=plc_loaded,
            proxy_check_every=proxy_check_every,
            log_every=self.log_every,
            saved_proxy=saved_proxy,
            saved_surr=saved_surr,
        )

        # Return full placement on CPU in the expected dtype.
        out = benchmark.macro_positions.clone()
        out[:n_hard] = full[:n_hard].detach().to(device=out.device, dtype=out.dtype)
        if n_soft > 0:
            out[n_hard:n_macros] = full[n_hard:n_macros].detach().to(
                device=out.device, dtype=out.dtype
            )
        # Preserve fixed macros exactly.
        out[fixed] = benchmark.macro_positions[fixed]
        _gpu_print_final_summary(
            benchmark=benchmark,
            full=out,
            plc=plc,
            elapsed=float(t_train1 - t_train0),
            epochs=int(run_meta.get("epochs_completed", 0)),
        )
        return out


if __name__ == "__main__":
    # Synthetic gradcheck: triangular spreading + FFT Poisson + energy (float64).
    def _electro_chain_for_gradcheck(pos: torch.Tensor) -> torch.Tensor:
        nr_i, nc_i = 8, 8
        cw_i, ch_i = 100.0, 100.0
        sz = torch.tensor([[10.0, 12.0], [8.0, 8.0], [15.0, 10.0]], dtype=torch.float64)
        c = _spread_charge_to_grid(pos, sz, nr_i, nc_i, cw_i, ch_i)
        p = _solve_poisson_fft(c)
        return (c * p).sum() / float(nr_i * nc_i)

    torch.manual_seed(0)
    pos64 = torch.rand(3, 2, dtype=torch.float64) * 80.0 + 10.0
    pos64.requires_grad_(True)
    torch.autograd.gradcheck(_electro_chain_for_gradcheck, (pos64,), eps=1e-3, atol=1e-2)
    print("[gpu/placer] electrostatic chain gradcheck OK")
