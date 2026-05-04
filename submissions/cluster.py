"""
Net-based macro clustering for rigid-body style placement moves.

Builds a weighted graph on **movable hard macros**: hyperedges (nets) are converted to
pairwise edges using the standard pin-dilution factor::

    Δw(i, j) += net_weight[k] / (pin_count(k) - 1)

for every net k that contains both macros i and j. Multi-pin nets contribute less per
pair than 2-pin nets.

Greedy agglomeration merges the cluster pair with highest::

    merge_score = net_weight(i, j) / max(distance(i, j), eps)

subject to a **minimum net edge weight** (connectivity threshold), **maximum macros per
cluster**, and **maximum cluster area** (area balance vs average expected cluster size).

Macros that never form strong enough pairwise edges remain singletons (“free” for
independent moves).

Writes ``vis/<benchmark.name>_macro_clusters.png`` (cluster hulls, colored macros, net edges)
and prints that absolute path to stdout. Placement is unchanged (analysis).

Usage:
    uv run evaluate submissions/cluster.py -b ibm01
    uv run python submissions/cluster.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _movable_hard_local_indices(benchmark: Benchmark) -> tuple[list[int], np.ndarray]:
    """Global macro indices of movable hard macros and their areas (μm²)."""
    nh = int(benchmark.num_hard_macros)
    fixed = benchmark.macro_fixed[:nh].detach().cpu().numpy().astype(bool)
    movable = (~fixed) & benchmark.get_hard_macro_mask()[:nh].detach().cpu().numpy().astype(
        bool
    )
    sizes = benchmark.macro_sizes[:nh].detach().cpu().numpy()
    idx = np.nonzero(movable)[0].tolist()
    areas = np.array(
        [float(sizes[g, 0]) * float(sizes[g, 1]) for g in idx], dtype=np.float64
    )
    return idx, areas


def _pin_counts(benchmark: Benchmark) -> np.ndarray:
    """Pin count per net (for 1/(p-1) dilution). Falls back to dedup macro count."""
    n_nets = int(benchmark.num_nets)
    out = np.zeros(n_nets, dtype=np.int32)
    if len(benchmark.net_pin_nodes) == n_nets:
        for k in range(n_nets):
            out[k] = int(benchmark.net_pin_nodes[k].shape[0])
    else:
        for k in range(n_nets):
            out[k] = int(benchmark.net_nodes[k].shape[0])
    return out


def build_pairwise_net_weights(
    benchmark: Benchmark,
    global_indices: list[int],
) -> np.ndarray:
    """
    Full pairwise net-derived weights W[a,b] for local indices a,b.

    For each net k with pin count p >= 2, for each unordered pair of distinct macros
    (i, j) on that net with both i,j in ``global_indices``, add
    net_weights[k] / (p - 1) to W[loc(i), loc(j)].
    """
    m = len(global_indices)
    if m == 0:
        return np.zeros((0, 0), dtype=np.float64)

    g2l = {g: a for a, g in enumerate(global_indices)}
    W = np.zeros((m, m), dtype=np.float64)
    nw = benchmark.net_weights.detach().cpu().numpy()
    pin_counts = _pin_counts(benchmark)

    for k in range(int(benchmark.num_nets)):
        p = int(pin_counts[k])
        if p < 2:
            continue
        contrib = float(nw[k]) / float(p - 1)
        nodes = benchmark.net_nodes[k].detach().cpu().numpy().astype(np.int64)
        locals_: list[int] = []
        for g in nodes.tolist():
            g = int(g)
            if g in g2l:
                locals_.append(g2l[g])
        if len(locals_) < 2:
            continue
        locals_.sort()
        for ii in range(len(locals_)):
            a = locals_[ii]
            for jj in range(ii + 1, len(locals_)):
                b = locals_[jj]
                W[a, b] += contrib
                W[b, a] += contrib
    return W


def apply_edge_threshold(W: np.ndarray, min_edge_weight: float) -> np.ndarray:
    """Zero out entries below ``min_edge_weight`` (symmetric)."""
    if W.size == 0:
        return W
    out = W.copy()
    mask = out < min_edge_weight
    out[mask] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def merge_scores_vs_distance(
    W: np.ndarray,
    positions: np.ndarray,
    eps_um: float,
) -> np.ndarray:
    """S[i,j] = W[i,j] / max(dist(i,j), eps). Diagonal 0."""
    m = W.shape[0]
    if m == 0:
        return W
    S = np.zeros((m, m), dtype=np.float64)
    eps = max(float(eps_um), 1e-9)
    for a in range(m):
        for b in range(a + 1, m):
            w = W[a, b]
            if w <= 0.0:
                continue
            dx = float(positions[a, 0]) - float(positions[b, 0])
            dy = float(positions[a, 1]) - float(positions[b, 1])
            d = math.hypot(dx, dy)
            s = w / max(d, eps)
            S[a, b] = s
            S[b, a] = s
    return S


def _cluster_areas(cluster_sets: list[set[int]], areas: np.ndarray) -> list[float]:
    return [float(sum(areas[i] for i in s)) for s in cluster_sets]


def greedy_merge_clusters(
    W: np.ndarray,
    S: np.ndarray,
    areas: np.ndarray,
    max_macros_per_cluster: int,
    max_cluster_area_um2: float | None,
) -> list[set[int]]:
    """
    Start with one macro per cluster; repeatedly merge two clusters that maximize
    max_{i in A, j in B} S[i,j] subject to size and area caps.
    """
    m = W.shape[0]
    if m == 0:
        return []
    if m == 1:
        return [{0}]

    clusters: list[set[int]] = [{i} for i in range(m)]
    alive = list(range(m))

    while len(alive) >= 2:
        best_score = -1.0
        best_pair: tuple[int, int] | None = None

        for ii in range(len(alive)):
            ci = alive[ii]
            set_i = clusters[ci]
            ni = len(set_i)
            ai = sum(areas[j] for j in set_i)
            for jj in range(ii + 1, len(alive)):
                cj = alive[jj]
                set_j = clusters[cj]
                nj = len(set_j)
                aj = sum(areas[k] for k in set_j)
                if ni + nj > max_macros_per_cluster:
                    continue
                if max_cluster_area_um2 is not None and ai + aj > max_cluster_area_um2 + 1e-12:
                    continue
                # Best pairwise merge score between the two clusters
                score = 0.0
                for i in set_i:
                    row = S[i]
                    for j in set_j:
                        if i == j:
                            continue
                        v = float(row[j])
                        if v > score:
                            score = v
                if score > best_score:
                    best_score = score
                    best_pair = (ci, cj)

        if best_pair is None or best_score <= 0.0:
            break

        ca, cb = best_pair
        clusters[ca] = clusters[ca] | clusters[cb]
        clusters[cb] = set()
        alive = [c for c in alive if c != cb]

    out = [clusters[c] for c in alive]
    return out


def cluster_movable_hard_macros(
    benchmark: Benchmark,
    *,
    min_edge_weight: float = 1e-4,
    max_macros_per_cluster: int = 8,
    area_balance_rel_tol: float = 0.35,
    eps_um: float = 0.5,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """
    Cluster movable hard macros.

    Returns:
        clusters_global: list of clusters; each cluster is a list of **global** macro indices.
        local_to_global: mapping local row -> global macro index.
        W_thresholded: symmetric net-weight matrix (local indices), after threshold.
    """
    global_indices, areas = _movable_hard_local_indices(benchmark)
    m = len(global_indices)
    if m == 0:
        return [], np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float64)

    pos = benchmark.macro_positions.detach().cpu().numpy()
    local_pos = np.stack([pos[g] for g in global_indices], axis=0)

    W = build_pairwise_net_weights(benchmark, global_indices)
    Wt = apply_edge_threshold(W, min_edge_weight)
    S = merge_scores_vs_distance(Wt, local_pos, eps_um)

    total_area = float(areas.sum())
    # Expected ~ceil(m / max_macros_per_cluster) clusters if we filled to max size
    k_expect = max(1, math.ceil(m / float(max_macros_per_cluster)))
    avg_area = total_area / float(k_expect)
    max_cluster_area = avg_area * (1.0 + float(area_balance_rel_tol))

    raw = greedy_merge_clusters(Wt, S, areas, max_macros_per_cluster, max_cluster_area)

    clusters_global: list[list[int]] = [
        sorted([global_indices[i] for i in cl]) for cl in raw
    ]
    local_to_global = np.array(global_indices, dtype=np.int64)
    return clusters_global, local_to_global, Wt


def _cluster_color(cid: int):
    """Distinct fill color per cluster (tab20, cycling)."""
    import matplotlib.pyplot as plt

    return plt.colormaps["tab20"]((cid % 20) / 19.0)


def _save_cluster_figure(
    benchmark: Benchmark,
    placement: torch.Tensor,
    clusters_global: list[list[int]],
    W_thresholded: np.ndarray,
    local_to_global: np.ndarray,
    out_path: Path,
) -> None:
    from scipy.spatial import ConvexHull, QhullError

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Rectangle

    nh = int(benchmark.num_hard_macros)
    pos = placement[:nh].detach().cpu().numpy()
    sizes = benchmark.macro_sizes[:nh].detach().cpu().numpy()
    fixed = benchmark.macro_fixed[:nh].detach().cpu().numpy().astype(bool)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_cl = len(clusters_global)

    macro_cluster = np.full(nh, -1, dtype=np.int32)
    for cid, cl in enumerate(clusters_global):
        for g in cl:
            macro_cluster[int(g)] = cid

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(0, cw)
    ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")
    ax.add_patch(Rectangle((0, 0), cw, ch, fill=False, edgecolor="black", linewidth=1.5))

    # Net connectivity (weak edges), behind hulls
    if W_thresholded.size > 0:
        mloc = W_thresholded.shape[0]
        for a in range(mloc):
            for b in range(a + 1, mloc):
                if W_thresholded[a, b] <= 0.0:
                    continue
                ga = int(local_to_global[a])
                gb = int(local_to_global[b])
                ax.plot(
                    [pos[ga, 0], pos[gb, 0]],
                    [pos[ga, 1], pos[gb, 1]],
                    color="0.55",
                    linewidth=0.4,
                    alpha=0.35,
                    zorder=1,
                )

    # Cluster envelopes: hull / segment / point marker
    for cid, cl in enumerate(clusters_global):
        col = _cluster_color(cid)
        pts = np.array([[float(pos[g, 0]), float(pos[g, 1])] for g in cl], dtype=np.float64)
        if len(cl) >= 3:
            try:
                hull = ConvexHull(pts)
                vidx = hull.vertices
                center = pts.mean(axis=0)
                ang = np.arctan2(pts[vidx, 1] - center[1], pts[vidx, 0] - center[0])
                ordered = vidx[np.argsort(ang)]
                poly = Polygon(
                    pts[ordered],
                    closed=True,
                    facecolor=col,
                    edgecolor=col,
                    linewidth=2.2,
                    alpha=0.22,
                    zorder=2,
                )
                ax.add_patch(poly)
            except QhullError:
                pass
        elif len(cl) == 2:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=col,
                linewidth=3.5,
                alpha=0.55,
                solid_capstyle="round",
                zorder=2,
            )
        elif len(cl) == 1:
            ax.scatter(
                pts[0, 0],
                pts[0, 1],
                s=120,
                c=[col],
                edgecolors="k",
                linewidths=0.8,
                zorder=2,
                alpha=0.85,
            )

        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        ax.annotate(
            f"C{cid}\n({len(cl)})",
            (cx, cy),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="0.15",
            zorder=4,
        )

    for i in range(nh):
        x, y = float(pos[i, 0]), float(pos[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        if fixed[i]:
            face, ec, lw = "lightgray", "darkred", 2.2
        elif macro_cluster[i] >= 0:
            face = _cluster_color(int(macro_cluster[i]))
            ec = "0.1"
            lw = 1.25
        else:
            face = "white"
            ec = "black"
            lw = 0.6
        ax.add_patch(
            Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                facecolor=face,
                edgecolor=ec,
                linewidth=lw,
                alpha=0.82,
                zorder=3,
            )
        )

    sizes_hist = [len(c) for c in clusters_global]
    stat = ""
    if sizes_hist:
        stat = f" | cluster sizes: min {min(sizes_hist)}, max {max(sizes_hist)}, "
        stat += f"mean {float(np.mean(sizes_hist)):.2f}"
    ax.set_title(
        f"{benchmark.name} — macro clusters (net weight ÷ distance, greedy merge)\n"
        f"{n_cl} clusters, {len(local_to_global)} movable hard macros{stat}"
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=_cluster_color(i),
            markeredgecolor="k",
            markersize=8,
            linestyle="None",
            label=f"Cluster {i} ({len(clusters_global[i])} macros)",
        )
        for i in range(min(n_cl, 16))
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="lightgray",
            markeredgecolor="darkred",
            markersize=10,
            label="Fixed macro",
        )
    )
    if n_cl > 16:
        legend_handles.insert(
            0,
            Line2D([0], [0], linestyle="None", label=f"… +{n_cl - 16} more clusters"),
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.92)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class ClusterPlacer:
    """
    Compute net-based macro clusters and write a visualization; placement unchanged.

    Args:
        min_edge_weight: Drop pairwise net weights below this before merging / drawing edges.
        max_macros_per_cluster: Hard cap on macros per cluster.
        area_balance_rel_tol: Max cluster area ≤ (total/k_expect)×(1+tol) with k_expect
            from ceil(n/max_macros).
        eps_um: Minimum distance (μm) in merge_score denominator.
    """

    def __init__(
        self,
        min_edge_weight: float = 1e-4,
        max_macros_per_cluster: int = 8,
        area_balance_rel_tol: float = 0.35,
        eps_um: float = 0.5,
    ):
        self.min_edge_weight = float(min_edge_weight)
        self.max_macros_per_cluster = int(max_macros_per_cluster)
        self.area_balance_rel_tol = float(area_balance_rel_tol)
        self.eps_um = float(eps_um)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        clusters, local_to_global, Wt = cluster_movable_hard_macros(
            benchmark,
            min_edge_weight=self.min_edge_weight,
            max_macros_per_cluster=self.max_macros_per_cluster,
            area_balance_rel_tol=self.area_balance_rel_tol,
            eps_um=self.eps_um,
        )
        out = _repo_root() / "vis" / f"{benchmark.name}_macro_clusters.png"
        _save_cluster_figure(
            benchmark,
            placement,
            clusters,
            Wt,
            local_to_global,
            out,
        )
        print(f"Cluster visualization saved to {out.resolve()}")
        return placement


def _cli_main() -> None:
    from macro_place.loader import load_benchmark_from_dir

    root = _repo_root()
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    ClusterPlacer().place(b)
    print(f"Wrote {root / 'vis' / 'ibm01_macro_clusters.png'}")


if __name__ == "__main__":
    _cli_main()
