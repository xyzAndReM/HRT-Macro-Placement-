"""
PLC-aligned differentiable routing congestion surrogate.

Mirrors PlacementCost ``get_routing`` / ``get_congestion_cost`` structure:
net L-shaped segment demand, normalize by per-cell route capacities, apply
``__smooth_routing_cong``-style spreading on net maps only, then add normalized
macro blockage (evaluator order: smooth nets, then add macro).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from macro_place.benchmark import Benchmark


def grid_routing_capacities(benchmark: "Benchmark") -> Tuple[float, float]:
    """
    Per-cell route capacities matching ``PlacementCost.get_routing``:

    ``grid_v_routes = grid_width * vroutes_per_micron``
    ``grid_h_routes = grid_height * hroutes_per_micron``
    """
    nr = max(int(benchmark.grid_rows), 1)
    nc = max(int(benchmark.grid_cols), 1)
    gw = float(benchmark.canvas_width) / nc
    gh = float(benchmark.canvas_height) / nr
    grid_v_routes = gw * float(benchmark.vroutes_per_micron)
    grid_h_routes = gh * float(benchmark.hroutes_per_micron)
    return grid_h_routes, grid_v_routes


def plc_congestion_smooth_range(plc) -> int:
    """``PlacementCost.get_congestion_smooth_range()`` wrapper."""
    return int(plc.get_congestion_smooth_range())


def _abu_top_mean(values: torch.Tensor, frac: float) -> torch.Tensor:
    flat = values.reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return torch.zeros((), device=flat.device, dtype=flat.dtype)
    k = max(1, int(math.floor(n * frac)))
    k = min(k, n)
    return torch.topk(flat, k, largest=True).values.mean()


def smooth_routing_cong_plc(
    H: torch.Tensor, V: torch.Tensor, smooth_range: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replica of ``PlacementCost.__smooth_routing_cong``: first spread each V cell
    along its row, then spread each H cell along its column.

    Vectorized: each source cell distributes uniformly to a sliding window; range
    adds use ``scatter_add`` + ``cumsum`` (``O(nr * nc)``), matching the reference loop.
    """
    nr, nc = H.shape
    sr = int(smooth_range)
    if sr <= 0:
        return H, V

    device, dtype = V.device, V.dtype

    # --- V pass (spread along columns within each row) ---
    cols = torch.arange(nc, device=device, dtype=torch.long)
    lp_c = (cols - sr).clamp(0, nc - 1)
    rp_c = (cols + sr).clamp(0, nc - 1)
    cnt_c = (rp_c - lp_c + 1).to(dtype).unsqueeze(0).expand(nr, -1)
    vals_v = V / cnt_c
    diff_v = torch.zeros(nr, nc + 1, device=device, dtype=dtype)
    lp_e = lp_c.unsqueeze(0).expand(nr, -1)
    rp1_e = (rp_c + 1).clamp(max=nc).unsqueeze(0).expand(nr, -1)
    diff_v.scatter_add_(1, lp_e, vals_v)
    diff_v.scatter_add_(1, rp1_e, -vals_v)
    V = diff_v.cumsum(dim=1)[:, :nc]

    # --- H pass (spread along rows within each column) ---
    rows = torch.arange(nr, device=device, dtype=torch.long)
    lp_r = (rows - sr).clamp(0, nr - 1)
    rp_r = (rows + sr).clamp(0, nr - 1)
    cnt_r = (rp_r - lp_r + 1).to(dtype).unsqueeze(0).expand(nc, -1)
    vals_h = H.T / cnt_r
    diff_h = torch.zeros(nc, nr + 1, device=device, dtype=dtype)
    lp_e = lp_r.unsqueeze(0).expand(nc, -1)
    rp1_e = (rp_r + 1).clamp(max=nr).unsqueeze(0).expand(nc, -1)
    diff_h.scatter_add_(1, lp_e, vals_h)
    diff_h.scatter_add_(1, rp1_e, -vals_h)
    H = diff_h.cumsum(dim=1)[:, :nr].T

    return H, V


def _macro_blockage_raw(
    full_pos: torch.Tensor,
    benchmark: "Benchmark",
    nr: int,
    nc: int,
    cw: float,
    ch: float,
    num_hard: int,
    hrouting_alloc: float,
    vrouting_alloc: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Raw H_macro / V_macro before division by grid_*_routes (matches plc sums)."""
    device, dtype = full_pos.device, full_pos.dtype
    h_macro = torch.zeros(nr, nc, device=device, dtype=dtype)
    v_macro = torch.zeros(nr, nc, device=device, dtype=dtype)
    if num_hard <= 0:
        return h_macro, v_macro

    bw = cw / nc
    bh = ch / nr
    sizes = benchmark.macro_sizes.to(device=device, dtype=dtype)
    rs = torch.arange(nr, device=device, dtype=dtype).view(nr, 1).expand(nr, nc)
    cs = torch.arange(nc, device=device, dtype=dtype).view(1, nc).expand(nr, nc)
    bx0 = cs * bw
    bx1 = bx0 + bw
    by0 = rs * bh
    by1 = by0 + bh

    fp = full_pos[:num_hard]
    cx = fp[:, 0].view(num_hard, 1, 1)
    cy = fp[:, 1].view(num_hard, 1, 1)
    w_m = sizes[:num_hard, 0].view(num_hard, 1, 1)
    h_m = sizes[:num_hard, 1].view(num_hard, 1, 1)
    lx = cx - 0.5 * w_m
    rx = cx + 0.5 * w_m
    by_m = cy - 0.5 * h_m
    ty_m = cy + 0.5 * h_m
    ix = torch.relu(torch.minimum(rx, bx1) - torch.maximum(lx, bx0))
    iy = torch.relu(torch.minimum(ty_m, by1) - torch.maximum(by_m, by0))
    mask = ((ix > 0) & (iy > 0)).to(dtype)
    v_macro = (ix * vrouting_alloc * mask).sum(dim=0)
    h_macro = (iy * hrouting_alloc * mask).sum(dim=0)

    return h_macro, v_macro


def _collect_l_route_segments(
    combined_pos: torch.Tensor,
    net_idx: torch.Tensor,
    net_mask: torch.Tensor,
    net_weights: torch.Tensor,
    net_valid: torch.Tensor,
    benchmark: "Benchmark",
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build stacked segment tensors ``sx,sy,tx,ty,w`` with shape ``[S]`` for all star
    pairs (pin0 -> pin j). Empty batch returns zero-length tensors on ``device``.
    """
    num_nets = int(net_idx.shape[0])
    n_macros = int(benchmark.num_macros)
    n_ports = int(benchmark.port_positions.shape[0])

    sx_list: List[torch.Tensor] = []
    sy_list: List[torch.Tensor] = []
    tx_list: List[torch.Tensor] = []
    ty_list: List[torch.Tensor] = []
    w_list: List[torch.Tensor] = []

    for k in range(num_nets):
        if not net_valid[k]:
            continue
        m = net_mask[k]
        idx = net_idx[k][m]
        pn = int(idx.numel())
        if pn < 2:
            continue
        w = net_weights[k]

        pins_x = torch.empty((pn,), device=device, dtype=dtype)
        pins_y = torch.empty((pn,), device=device, dtype=dtype)
        for j in range(pn):
            node = int(idx[j].item())
            if node < n_macros:
                pins_x[j] = combined_pos[node, 0]
                pins_y[j] = combined_pos[node, 1]
            else:
                pi = node - n_macros
                if 0 <= pi < n_ports:
                    pins_x[j] = combined_pos[n_macros + pi, 0]
                    pins_y[j] = combined_pos[n_macros + pi, 1]
                else:
                    pins_x[j] = torch.zeros((), device=device, dtype=dtype)
                    pins_y[j] = torch.zeros((), device=device, dtype=dtype)

        sx0 = pins_x[0]
        sy0 = pins_y[0]
        for j in range(1, pn):
            sx_list.append(sx0)
            sy_list.append(sy0)
            tx_list.append(pins_x[j])
            ty_list.append(pins_y[j])
            w_list.append(w.expand(()))

    if not sx_list:
        z = torch.zeros(0, device=device, dtype=dtype)
        return z, z, z, z, z

    sx = torch.stack(sx_list)
    sy = torch.stack(sy_list)
    tx = torch.stack(tx_list)
    ty = torch.stack(ty_list)
    w_seg = torch.stack(w_list)
    return sx, sy, tx, ty, w_seg


def _apply_l_route_segments_vectorized(
    H_net: torch.Tensor,
    V_net: torch.Tensor,
    sx: torch.Tensor,
    sy: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    w_seg: torch.Tensor,
    nr: int,
    nc: int,
    cell_w: float,
    cell_h: float,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Mutates ``H_net`` and ``V_net`` in place."""
    if sx.numel() == 0:
        return

    S = sx.shape[0]
    cols = torch.arange(nc, device=device, dtype=dtype).unsqueeze(0).expand(S, -1)

    cs = torch.floor(sx / cell_w).clamp(0, nc - 1)
    ct = torch.floor(tx / cell_w).clamp(0, nc - 1)
    col_lo = torch.minimum(cs, ct).unsqueeze(1)
    col_hi = torch.maximum(cs, ct).unsqueeze(1)
    h_mask = (cols >= col_lo) & (cols < col_hi)

    row_cont = sy / cell_h
    r0 = torch.floor(row_cont).long().clamp(0, nr - 1)
    r1 = torch.minimum(
        r0 + 1,
        torch.tensor(nr - 1, device=device, dtype=torch.long),
    )
    alpha = (row_cont - r0.float()).clamp(0.0, 1.0)

    contrib0 = w_seg.unsqueeze(1) * (1.0 - alpha).unsqueeze(1) * h_mask.to(dtype)
    contrib1 = w_seg.unsqueeze(1) * alpha.unsqueeze(1) * h_mask.to(dtype)
    H_net.index_add_(0, r0, contrib0)
    H_net.index_add_(0, r1, contrib1)

    rs_f = torch.floor(sy / cell_h).clamp(0, nr - 1)
    rt_f = torch.floor(ty / cell_h).clamp(0, nr - 1)
    row_lo = torch.minimum(rs_f, rt_f).float().unsqueeze(1)
    row_hi = torch.maximum(rs_f, rt_f).float().unsqueeze(1)
    rows = torch.arange(nr, device=device, dtype=dtype).unsqueeze(0).expand(S, -1)
    v_mask = (rows >= row_lo) & (rows < row_hi)

    col_cont = tx / cell_w
    c0 = torch.floor(col_cont).long().clamp(0, nc - 1)
    c1 = torch.minimum(
        c0 + 1,
        torch.tensor(nc - 1, device=device, dtype=torch.long),
    )
    beta = (col_cont - c0.float()).clamp(0.0, 1.0)

    contrib_v0 = (w_seg * (1.0 - beta)).unsqueeze(1) * v_mask.to(dtype)
    contrib_v1 = (w_seg * beta).unsqueeze(1) * v_mask.to(dtype)
    V_net.index_add_(1, c0, contrib_v0.T)
    V_net.index_add_(1, c1, contrib_v1.T)


def plc_routing_surrogate_scalar(
    combined_pos: torch.Tensor,
    net_idx: torch.Tensor,
    net_mask: torch.Tensor,
    net_weights: torch.Tensor,
    net_valid: torch.Tensor,
    full_macro_pos: torch.Tensor,
    benchmark: "Benchmark",
    *,
    smooth_range: Optional[int] = None,
) -> torch.Tensor:
    """
    Scalar congestion matching ``get_congestion_cost`` reduction: mean of top 5%% of
    all smoothed-then-macro-added normalized H and V cell utilizations (concatenated).

    Gradients flow via soft row/column interpolation on L-route deposits.
    """
    device = combined_pos.device
    dtype = combined_pos.dtype
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    nr = max(int(benchmark.grid_rows), 1)
    nc = max(int(benchmark.grid_cols), 1)
    cell_w = cw / nc
    cell_h = ch / nr

    sr = smooth_range
    if sr is None:
        sr = int(benchmark.congestion_smooth_range)
    sr = int(sr)

    grid_h_routes, grid_v_routes = grid_routing_capacities(benchmark)
    gh = torch.tensor(max(grid_h_routes, 1e-30), device=device, dtype=dtype)
    gv = torch.tensor(max(grid_v_routes, 1e-30), device=device, dtype=dtype)

    h_ma = float(benchmark.hrouting_alloc)
    v_ma = float(benchmark.vrouting_alloc)

    H_net = torch.zeros(nr, nc, device=device, dtype=dtype)
    V_net = torch.zeros(nr, nc, device=device, dtype=dtype)

    sx, sy, tx, ty, w_seg = _collect_l_route_segments(
        combined_pos,
        net_idx,
        net_mask,
        net_weights,
        net_valid,
        benchmark,
        device,
        dtype,
    )
    _apply_l_route_segments_vectorized(
        H_net,
        V_net,
        sx,
        sy,
        tx,
        ty,
        w_seg,
        nr,
        nc,
        cell_w,
        cell_h,
        device,
        dtype,
    )

    H_net = H_net / gh
    V_net = V_net / gv

    H_net, V_net = smooth_routing_cong_plc(H_net, V_net, sr)

    num_hard = int(benchmark.num_hard_macros)
    h_blk, v_blk = _macro_blockage_raw(
        full_macro_pos,
        benchmark,
        nr,
        nc,
        cw,
        ch,
        num_hard,
        h_ma,
        v_ma,
    )
    h_blk = h_blk / gh
    v_blk = v_blk / gv

    H_tot = H_net + h_blk
    V_tot = V_net + v_blk

    both = torch.cat([V_tot.reshape(-1), H_tot.reshape(-1)])
    return _abu_top_mean(both, 0.05)
