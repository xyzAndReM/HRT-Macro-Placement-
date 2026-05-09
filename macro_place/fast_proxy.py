"""
Fast approximation to ``compute_proxy_cost`` for inner optimization loops.

Uses the same decomposition and weights as ``macro_place.objective.compute_proxy_cost``:

    1.0 * wirelength + 0.5 * density + 0.5 * congestion

* **Wirelength** — when ``net_pin_nodes`` is populated, vectorized **pin** HPWL with
  hard-macro pin offsets (matches ``PlacementCost`` geometry much better than
  deduped macro centers). Weights are ``benchmark.net_weights`` (driver weight per net).
  Normalizer is ``(W+H) * wl_normalize_weight_sum`` (PlacementCost ``net_cnt`` from
  the loader). Falls back to macro-cluster HPWL if pin data is missing.
* **Density** — macro rectangle overlap / bin area on the placement grid, top-10%
  mean × 0.5 (same structure as ``_plc_style_density_from_grid`` in gradient).
* **Congestion** — ``plc_routing_surrogate_discrete_pins`` when pin lists exist:
  same pin grid cells, 2-/3-pin rules, and per-branch weights as ``PlacementCost``
  ``get_routing``; then identical smooth / macro / ABU as the evaluator. Falls
  back to the soft differentiable surrogate if pin data is missing.

This avoids mutating ``PlacementCost`` / C++ routing on every trial; it is much
closer to the evaluator than the RUDY-based heuristic in ``sa.py``.
"""

from __future__ import annotations

import math
import os

import torch

from macro_place.benchmark import Benchmark
from macro_place.routing_surrogate import (
    plc_routing_surrogate_discrete_pins,
    plc_routing_surrogate_scalar,
)


def _select_device() -> torch.device:
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


def _pin_count(benchmark: Benchmark, k: int) -> int:
    if len(benchmark.net_pin_nodes) == benchmark.num_nets:
        return max(int(benchmark.net_pin_nodes[k].shape[0]), 2)
    return max(int(benchmark.net_nodes[k].numel()), 2)


def _build_net_tensors(
    benchmark: Benchmark,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _weighted_hpwl_sum(
    combined_pos: torch.Tensor,
    net_idx: torch.Tensor,
    net_mask: torch.Tensor,
    net_weights: torch.Tensor,
    net_valid: torch.Tensor,
) -> torch.Tensor:
    """Σ_k w_k * HPWL(net k); ``combined_pos`` = macros + ports (see gradient ``_wirelength_loss_v2``)."""
    all_pins = combined_pos[net_idx]
    x = all_pins[:, :, 0]
    y = all_pins[:, :, 1]

    ninf = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    pinf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)

    x_max = x.masked_fill(~net_mask, ninf).amax(dim=1)
    x_min = x.masked_fill(~net_mask, pinf).amin(dim=1)
    y_max = y.masked_fill(~net_mask, ninf).amax(dim=1)
    y_min = y.masked_fill(~net_mask, pinf).amin(dim=1)

    hpwl = (x_max - x_min) + (y_max - y_min)
    zero = torch.zeros((), device=hpwl.device, dtype=hpwl.dtype)
    hpwl = torch.where(net_valid, hpwl, zero)
    return (net_weights * hpwl).sum()


def _build_pin_wl_tensors(
    benchmark: Benchmark,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Packed pin tables for PLC-style HPWL; ``None`` if pin lists are incomplete."""
    num_nets = int(benchmark.num_nets)
    if len(benchmark.net_pin_nodes) != num_nets:
        return None

    n_hard = int(benchmark.num_hard_macros)
    max_p = 0
    for k in range(num_nets):
        max_p = max(max_p, int(benchmark.net_pin_nodes[k].shape[0]))
    if max_p < 2:
        return None

    owners = torch.zeros(num_nets, max_p, dtype=torch.long, device=device)
    mask = torch.zeros(num_nets, max_p, dtype=torch.bool, device=device)
    delta = torch.zeros(num_nets, max_p, 2, device=device, dtype=dtype)
    net_valid = torch.zeros(num_nets, dtype=torch.bool, device=device)

    for k in range(num_nets):
        pins = benchmark.net_pin_nodes[k]
        np_ = int(pins.shape[0])
        if np_ < 2:
            continue
        net_valid[k] = True
        mask[k, :np_] = True
        owners[k, :np_] = pins[:, 0].to(device=device, dtype=torch.long)
        slots = pins[:, 1].tolist()
        for j in range(np_):
            owner = int(pins[j, 0].item())
            slot = int(slots[j])
            if owner < n_hard and 0 <= slot < len(benchmark.macro_pin_offsets[owner]):
                off = benchmark.macro_pin_offsets[owner][slot].to(device=device, dtype=dtype)
                delta[k, j, 0] = off[0]
                delta[k, j, 1] = off[1]

    w_src = (
        benchmark.net_driver_weights
        if benchmark.net_driver_weights is not None
        else benchmark.net_weights
    )
    driver_w = w_src.to(device=device, dtype=dtype)
    return owners, mask, delta, net_valid, driver_w


def _weighted_hpwl_sum_pins(
    combined_pos: torch.Tensor,
    pin_owner: torch.Tensor,
    pin_mask: torch.Tensor,
    pin_delta: torch.Tensor,
    driver_w: torch.Tensor,
    net_valid: torch.Tensor,
) -> torch.Tensor:
    """Σ_k driver_w[k] * HPWL over physical pins (macro center + offset)."""
    base = combined_pos[pin_owner]
    xy = base + pin_delta
    x = xy[:, :, 0]
    y = xy[:, :, 1]

    ninf = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    pinf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)

    x_max = x.masked_fill(~pin_mask, ninf).amax(dim=1)
    x_min = x.masked_fill(~pin_mask, pinf).amin(dim=1)
    y_max = y.masked_fill(~pin_mask, ninf).amax(dim=1)
    y_min = y.masked_fill(~pin_mask, pinf).amin(dim=1)

    hpwl = (x_max - x_min) + (y_max - y_min)
    zero = torch.zeros((), device=hpwl.device, dtype=hpwl.dtype)
    hpwl = torch.where(net_valid, hpwl, zero)
    return (driver_w * hpwl).sum()


def _plc_macro_overlap_density_grid(
    full_pos: torch.Tensor,
    benchmark: Benchmark,
    cw: float,
    ch: float,
    nr: int,
    nc: int,
) -> torch.Tensor:
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

    dens = torch.zeros(nr, nc, device=device, dtype=dtype)
    max_elems = 256 * 1024 * 1024
    bin_cells = nr * nc
    chunk = n
    if n * bin_cells > max_elems and bin_cells > 0:
        chunk = max(1, max_elems // bin_cells)

    bx0_b = bx0.unsqueeze(0)
    bx1_b = bx1.unsqueeze(0)
    by0_b = by0.unsqueeze(0)
    by1_b = by1.unsqueeze(0)

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
        dens = dens + (ix0 * iy0).sum(dim=0) / bin_area
    return dens


def _plc_style_density_from_grid(dens: torch.Tensor) -> torch.Tensor:
    flat = dens.reshape(-1).clamp(min=0)
    ncells = int(flat.numel())
    if ncells == 0:
        return torch.zeros((), device=flat.device, dtype=flat.dtype)
    density_cnt = max(1, int(math.floor(ncells * 0.1)))
    k = min(density_cnt, ncells)
    top = torch.topk(flat, k, largest=True).values
    return 0.5 * top.sum() / float(density_cnt)


class FastProxyEvaluator:
    """
    Callable ``float = eval(placement)`` with the same weighted sum as ``proxy_cost``.

    ``placement`` may live on CPU or GPU; work runs on ``device`` (CUDA if available).
    """

    __slots__ = (
        "benchmark",
        "device",
        "dtype",
        "cw",
        "ch",
        "nr",
        "nc",
        "n_hard",
        "net_idx",
        "net_mask",
        "net_weights",
        "net_valid",
        "wl_norm",
        "ports",
        "_pin_wl",
        "_use_discrete_cong",
    )

    def __init__(self, benchmark: Benchmark, *, device: torch.device | None = None):
        self.benchmark = benchmark
        self.device = device if device is not None else _select_device()
        self.dtype = torch.float32
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.nr = max(int(benchmark.grid_rows), 1)
        self.nc = max(int(benchmark.grid_cols), 1)
        self.n_hard = int(benchmark.num_hard_macros)

        self.net_idx, self.net_mask, self.net_weights, self.net_valid = _build_net_tensors(
            benchmark, self.device, self.dtype
        )
        wsum = max(float(benchmark.wl_normalize_weight_sum), 1.0)
        self.wl_norm = (self.cw + self.ch) * wsum
        self.ports = benchmark.port_positions.to(device=self.device, dtype=self.dtype)
        self._pin_wl = _build_pin_wl_tensors(benchmark, self.device, self.dtype)
        self._use_discrete_cong = len(benchmark.net_pin_nodes) == int(benchmark.num_nets)

    @torch.inference_mode()
    def total(self, placement: torch.Tensor) -> float:
        full = placement.to(device=self.device, dtype=self.dtype)
        combined = torch.cat([full, self.ports], dim=0)

        if self._pin_wl is not None:
            po, pm, pd, pvalid, dw = self._pin_wl
            wl_raw = _weighted_hpwl_sum_pins(combined, po, pm, pd, dw, pvalid)
        else:
            wl_raw = _weighted_hpwl_sum(
                combined, self.net_idx, self.net_mask, self.net_weights, self.net_valid
            )
        wl = wl_raw / self.wl_norm

        dens_grid = _plc_macro_overlap_density_grid(
            full, self.benchmark, self.cw, self.ch, self.nr, self.nc
        )
        dens = _plc_style_density_from_grid(dens_grid)

        if self._use_discrete_cong:
            cong = plc_routing_surrogate_discrete_pins(
                combined,
                full[: self.n_hard],
                self.benchmark,
            )
        else:
            cong = plc_routing_surrogate_scalar(
                combined,
                self.net_idx,
                self.net_mask,
                self.net_weights,
                self.net_valid,
                full[: self.n_hard],
                self.benchmark,
            )

        return float((wl + 0.5 * dens + 0.5 * cong).item())

    def __call__(self, placement: torch.Tensor) -> float:
        return self.total(placement)
