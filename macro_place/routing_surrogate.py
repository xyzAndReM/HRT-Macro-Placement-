"""
PLC-aligned routing congestion.

* ``plc_routing_surrogate_scalar`` — soft, differentiable star-L routes on macro
  centers (for gradient training).
* ``plc_routing_surrogate_discrete_pins`` — **evaluator-quality** congestion on
  ICCAD-style benchmarks: pin grid cells, same 2-/3-pin/decompose rules and
  routing weights as ``plc_client_os.get_routing``, PLC macro overlap with
  partial-edge corrections, then smooth + ABU(0.05) like ``get_congestion_cost``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from macro_place.benchmark import Benchmark


def grid_routing_capacities(
    benchmark: "Benchmark",
    *,
    nr: Optional[int] = None,
    nc: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Per-cell route capacities matching ``PlacementCost.get_routing``:

    ``grid_v_routes = grid_width * vroutes_per_micron``
    ``grid_h_routes = grid_height * hroutes_per_micron``

    Optional ``nr`` / ``nc`` override the benchmark grid (e.g. refinement with finer cells).
    """
    nr = max(int(benchmark.grid_rows), 1) if nr is None else max(int(nr), 1)
    nc = max(int(benchmark.grid_cols), 1) if nc is None else max(int(nc), 1)
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


# ── Discrete pin-grid routing (matches ``plc_client_os.get_routing`` net loops) ──


def _np_two_pin_routing(
    H: np.ndarray,
    V: np.ndarray,
    source_gcell: Tuple[int, int],
    sink_gcell: Tuple[int, int],
    weight: float,
) -> None:
    """Same cell indexing as ``PlacementCost.__two_pin_net_routing`` (row, col)."""
    sr, sc = source_gcell
    tr, tc = sink_gcell
    row_min = min(sr, tr)
    row_max = max(sr, tr)
    col_min = min(sc, tc)
    col_max = max(sc, tc)
    for col_idx in range(col_min, col_max):
        H[sr, col_idx] += weight
    for row_idx in range(row_min, row_max):
        V[row_idx, tc] += weight


def _np_l_routing_three(H: np.ndarray, V: np.ndarray, nodes: List[Tuple[int, int]], weight: float) -> None:
    nodes = sorted(nodes, key=lambda x: (x[1], x[0]))
    y1, x1 = nodes[0]
    y2, x2 = nodes[1]
    y3, x3 = nodes[2]
    for col in range(x1, x2):
        row = y1
        H[row, col] += weight
    for col in range(x2, x3):
        row = y2
        H[row, col] += weight
    for row in range(min(y1, y2), max(y1, y2)):
        col = x2
        V[row, col] += weight
    for row in range(min(y2, y3), max(y2, y3)):
        col = x3
        V[row, col] += weight


def _np_t_routing_three(H: np.ndarray, V: np.ndarray, nodes: List[Tuple[int, int]], weight: float) -> None:
    nodes = sorted(nodes)
    y1, x1 = nodes[0]
    y2, x2 = nodes[1]
    y3, x3 = nodes[2]
    xmin = min(x1, x2, x3)
    xmax = max(x1, x2, x3)
    for col in range(xmin, xmax):
        row = y2
        H[row, col] += weight
    for row in range(min(y1, y2), max(y1, y2)):
        col = x1
        V[row, col] += weight
    for row in range(min(y2, y3), max(y2, y3)):
        col = x3
        V[row, col] += weight


def _np_three_pin_routing(
    H: np.ndarray,
    V: np.ndarray,
    node_gcells: Set[Tuple[int, int]],
    weight: float,
) -> None:
    """Mirror ``PlacementCost.__three_pin_net_routing`` (``plc_client_os.py``)."""
    temp_gcell = list(node_gcells)
    temp_gcell.sort(key=lambda x: (x[1], x[0]))
    y1, x1 = temp_gcell[0]
    y2, x2 = temp_gcell[1]
    y3, x3 = temp_gcell[2]

    if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
        _np_l_routing_three(H, V, temp_gcell, weight)
    elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
        for col_idx in range(x1, x2):
            row = y1
            H[row, col_idx] += weight
        for row_idx in range(y1, max(y2, y3)):
            col = x2
            V[row_idx, col] += weight
    elif y2 == y3:
        for col in range(x1, x2):
            row = y1
            H[row, col] += weight
        for col in range(x2, x3):
            row = y2
            H[row, col] += weight
        for row in range(min(y2, y1), max(y2, y1)):
            col = x2
            V[row, col] += weight
    else:
        _np_t_routing_three(H, V, temp_gcell, weight)


def _np_pin_rc(
    combined_xy: np.ndarray,
    benchmark: "Benchmark",
    owner: int,
    slot: int,
    n_hard: int,
    cell_w: float,
    cell_h: float,
    nr: int,
    nc: int,
) -> Tuple[int, int]:
    ox = float(combined_xy[owner, 0])
    oy = float(combined_xy[owner, 1])
    if owner < n_hard and 0 <= slot < len(benchmark.macro_pin_offsets[owner]):
        off = benchmark.macro_pin_offsets[owner][slot].detach().cpu().numpy()
        ox += float(off[0])
        oy += float(off[1])
    row = int(math.floor(oy / cell_h))
    col = int(math.floor(ox / cell_w))
    return max(0, min(row, nr - 1)), max(0, min(col, nc - 1))


def _discrete_net_routing_grids(
    benchmark: "Benchmark",
    combined_xy: np.ndarray,
    nr: int,
    nc: int,
    cw: float,
    ch: float,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Raw H/V demand on the placement grid (same topology/weights as PLC), or ``None``
    if ``net_pin_nodes`` is incomplete.
    """
    num_nets = int(benchmark.num_nets)
    if len(benchmark.net_pin_nodes) != num_nets:
        return None

    cell_w = cw / nc
    cell_h = ch / nr
    n_hard = int(benchmark.num_hard_macros)
    H = np.zeros((nr, nc), dtype=np.float64)
    V = np.zeros((nr, nc), dtype=np.float64)
    w_tensor = (
        benchmark.net_driver_weights
        if benchmark.net_driver_weights is not None
        else benchmark.net_weights
    )
    w_cpu = w_tensor.detach().cpu().numpy()
    n_macros_b = int(benchmark.num_macros)

    for k in range(num_nets):
        pins = benchmark.net_pin_nodes[k]
        npin = int(pins.shape[0])
        if npin < 2:
            continue
        gw = float(w_cpu[k])
        if gw <= 0.0:
            continue
        # ``plc_client_os.get_routing``: PORT nets use weight 1; MACRO_PIN uses 1 unless get_weight()>1.
        first_owner = int(pins[0, 0].item())
        if first_owner >= n_macros_b:
            weight = 1.0
        else:
            weight = 1.0 if gw <= 1.0 else gw

        gcells_ordered: List[Tuple[int, int]] = []
        for j in range(npin):
            owner = int(pins[j, 0].item())
            slot = int(pins[j, 1].item())
            gcells_ordered.append(
                _np_pin_rc(combined_xy, benchmark, owner, slot, n_hard, cell_w, cell_h, nr, nc)
            )

        source_gcell = gcells_ordered[0]
        node_gcells: Set[Tuple[int, int]] = set(gcells_ordered)
        nuniq = len(node_gcells)

        if nuniq == 2:
            other = next(c for c in node_gcells if c != source_gcell)
            _np_two_pin_routing(H, V, source_gcell, other, weight)
        elif nuniq == 3:
            _np_three_pin_routing(H, V, node_gcells, weight)
        elif nuniq > 3:
            for gcell in node_gcells:
                if gcell != source_gcell:
                    _np_two_pin_routing(H, V, source_gcell, gcell, weight)

    return H, V


class _PLCBlock:
    __slots__ = ("x_min", "x_max", "y_min", "y_max")

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


def _plc_overlap_dist(bi: _PLCBlock, bj: _PLCBlock) -> Tuple[float, float]:
    x_diff = min(bi.x_max, bj.x_max) - max(bi.x_min, bj.x_min)
    y_diff = min(bi.y_max, bj.y_max) - max(bi.y_min, bj.y_min)
    if x_diff > 0 and y_diff > 0:
        return float(x_diff), float(y_diff)
    return 0.0, 0.0


def _macro_blockage_plc_numpy(
    full_xy: np.ndarray,
    benchmark: "Benchmark",
    nr: int,
    nc: int,
    cw: float,
    ch: float,
    num_hard: int,
    v_alloc: float,
    h_alloc: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror ``PlacementCost.__macro_route_over_grid_cell`` (partial-overlap fixes included)."""
    grid_w = cw / nc
    grid_h = ch / nr
    V = np.zeros((nr, nc), dtype=np.float64)
    H = np.zeros((nr, nc), dtype=np.float64)
    if num_hard <= 0:
        return H, V

    sizes = benchmark.macro_sizes[:num_hard].detach().cpu().numpy()

    def clamp_cell(row: int, col: int) -> Tuple[int, int]:
        return max(0, min(row, nr - 1)), max(0, min(col, nc - 1))

    for i in range(num_hard):
        mod_x = float(full_xy[i, 0])
        mod_y = float(full_xy[i, 1])
        mod_w = float(sizes[i, 0])
        mod_h = float(sizes[i, 1])

        ur = (mod_x + (mod_w / 2), mod_y + (mod_h / 2))
        bl = (mod_x - (mod_w / 2), mod_y - (mod_h / 2))
        module_block = _PLCBlock(bl[0], ur[0], bl[1], ur[1])

        ur_row, ur_col = clamp_cell(int(math.floor(ur[1] / grid_h)), int(math.floor(ur[0] / grid_w)))
        bl_row, bl_col = clamp_cell(int(math.floor(bl[1] / grid_h)), int(math.floor(bl[0] / grid_w)))

        if ur_row >= 0 and ur_col >= 0:
            if bl_row < 0:
                bl_row = 0
            if bl_col < 0:
                bl_col = 0
        else:
            continue

        if bl_row >= 0 and bl_col >= 0:
            if ur_row > nr - 1:
                ur_row = nr - 1
            if ur_col > nc - 1:
                ur_col = nc - 1
        else:
            continue

        partial_v = False
        partial_h = False

        for r_i in range(bl_row, ur_row + 1):
            for c_i in range(bl_col, ur_col + 1):
                grid_cell_block = _PLCBlock(
                    c_i * grid_w,
                    (c_i + 1) * grid_w,
                    r_i * grid_h,
                    (r_i + 1) * grid_h,
                )
                x_dist, y_dist = _plc_overlap_dist(module_block, grid_cell_block)

                if ur_row != bl_row:
                    if (r_i == bl_row and abs(y_dist - grid_h) > 1e-5) or (
                        r_i == ur_row and abs(y_dist - grid_h) > 1e-5
                    ):
                        partial_v = True

                if ur_col != bl_col:
                    if (c_i == bl_col and abs(x_dist - grid_w) > 1e-5) or (
                        c_i == ur_col and abs(x_dist - grid_w) > 1e-5
                    ):
                        partial_h = True

                V[r_i, c_i] += x_dist * v_alloc
                H[r_i, c_i] += y_dist * h_alloc

        if partial_v:
            r_i = ur_row
            for c_i in range(bl_col, ur_col + 1):
                grid_cell_block = _PLCBlock(
                    c_i * grid_w,
                    (c_i + 1) * grid_w,
                    r_i * grid_h,
                    (r_i + 1) * grid_h,
                )
                x_dist, y_dist = _plc_overlap_dist(module_block, grid_cell_block)
                V[r_i, c_i] -= x_dist * v_alloc

        if partial_h:
            c_i = ur_col
            for r_i in range(bl_row, ur_row + 1):
                grid_cell_block = _PLCBlock(
                    c_i * grid_w,
                    (c_i + 1) * grid_w,
                    r_i * grid_h,
                    (r_i + 1) * grid_h,
                )
                x_dist, y_dist = _plc_overlap_dist(module_block, grid_cell_block)
                H[r_i, c_i] -= y_dist * h_alloc

    return H, V


def plc_routing_surrogate_discrete_pins(
    combined_pos: torch.Tensor,
    full_macro_pos: torch.Tensor,
    benchmark: "Benchmark",
    *,
    smooth_range: Optional[int] = None,
    abu_frac: float = 0.05,
    nr: Optional[int] = None,
    nc: Optional[int] = None,
) -> torch.Tensor:
    """
    Congestion scalar matching ``get_congestion_cost`` **much** more closely than
    ``plc_routing_surrogate_scalar``: net demand uses **pin** grid cells, PLC
    **two-/three-pin** rules, and **full driver weight** on every two-pin branch
    (same as ``__split_net``). Not differentiable.

    Normalization, smoothing, macro blockage, and ABU(0.05) match the differentiable path.
    """
    device = combined_pos.device
    dtype = combined_pos.dtype
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    nr = max(int(benchmark.grid_rows), 1) if nr is None else max(int(nr), 1)
    nc = max(int(benchmark.grid_cols), 1) if nc is None else max(int(nc), 1)

    sr = int(benchmark.congestion_smooth_range) if smooth_range is None else int(smooth_range)

    grid_h_routes, grid_v_routes = grid_routing_capacities(benchmark, nr=nr, nc=nc)
    gh = torch.tensor(max(grid_h_routes, 1e-30), device=device, dtype=dtype)
    gv = torch.tensor(max(grid_v_routes, 1e-30), device=device, dtype=dtype)

    h_ma = float(benchmark.hrouting_alloc)
    v_ma = float(benchmark.vrouting_alloc)

    combined_xy = combined_pos.detach().cpu().numpy().astype(np.float64)
    raw = _discrete_net_routing_grids(benchmark, combined_xy, nr, nc, cw, ch)
    if raw is None:
        raise ValueError(
            "plc_routing_surrogate_discrete_pins requires complete benchmark.net_pin_nodes"
        )
    H_np, V_np = raw

    H_net = torch.from_numpy(H_np).to(device=device, dtype=dtype)
    V_net = torch.from_numpy(V_np).to(device=device, dtype=dtype)

    H_net = H_net / gh
    V_net = V_net / gv

    H_net, V_net = smooth_routing_cong_plc(H_net, V_net, sr)

    num_hard = int(benchmark.num_hard_macros)
    full_xy = full_macro_pos.detach().cpu().numpy().astype(np.float64)
    h_blk_np, v_blk_np = _macro_blockage_plc_numpy(
        full_xy, benchmark, nr, nc, cw, ch, num_hard, v_ma, h_ma
    )
    h_blk = torch.from_numpy(h_blk_np).to(device=device, dtype=dtype)
    v_blk = torch.from_numpy(v_blk_np).to(device=device, dtype=dtype)
    h_blk = h_blk / gh
    v_blk = v_blk / gv

    H_tot = H_net + h_blk
    V_tot = V_net + v_blk

    both = torch.cat([V_tot.reshape(-1), H_tot.reshape(-1)])
    return _abu_top_mean(both, float(abu_frac))


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
    abu_frac: float = 0.05,
    nr: Optional[int] = None,
    nc: Optional[int] = None,
) -> torch.Tensor:
    """
    Scalar congestion matching ``get_congestion_cost`` reduction: mean of top ``abu_frac``
    fraction of all smoothed-then-macro-added normalized H and V cell utilizations (concatenated).

    Gradients flow via soft row/column interpolation on L-route deposits.
    """
    device = combined_pos.device
    dtype = combined_pos.dtype
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    nr = max(int(benchmark.grid_rows), 1) if nr is None else max(int(nr), 1)
    nc = max(int(benchmark.grid_cols), 1) if nc is None else max(int(nc), 1)
    cell_w = cw / nc
    cell_h = ch / nr

    sr = smooth_range
    if sr is None:
        sr = int(benchmark.congestion_smooth_range)
    sr = int(sr)

    grid_h_routes, grid_v_routes = grid_routing_capacities(
        benchmark, nr=nr, nc=nc
    )
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
    return _abu_top_mean(both, float(abu_frac))
