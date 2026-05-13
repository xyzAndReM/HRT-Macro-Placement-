"""
CUDA coarse-grid exploration (same loop as ``ExplorePlacer`` in ``explore.py``).

Moves **all non-fixed macros** (same candidate set as ``ExplorePlacer``).

**Fast surrogate**: ``FastProxyEvaluator`` with ``use_discrete_congestion=False`` so
congestion uses ``plc_routing_surrogate_scalar`` only (pure Torch on GPU—no NumPy
routing grids). Wirelength/density match ``explore.py``; congestion differs from the
default discrete-pin path used there when pin tables are complete.

Keep ``placement`` on ``cuda`` float32 end-to-end; return value stays on CUDA for
pipelines into ``submissions/gpu/placer.py``.

Usage:
    MACRO_PLACE_DEVICE=cuda uv run python submissions/explore_gpu.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.fast_proxy import FastProxyEvaluator
from macro_place.loader import load_benchmark_from_dir

from submissions.explore import (
    _GRID_SIDE,
    _legal_center,
    _movable_macro_indices,
    _precompute_nine_neighbors,
    _repo_root,
    _save_explore_figure,
    _cell_xy_to_rc,
)


class ExploreGpuPlacer:
    """
    Same exploration semantics as ``ExplorePlacer``, CUDA placement, scalar congestion.

    ``fast_proxy_device`` is accepted for API symmetry with ``ExplorePlacer`` but ignored
    (device is always ``cuda``).
    """

    def __init__(
        self,
        grid_side: int = _GRID_SIDE,
        epochs: int = 50,
        seed: int = 0,
        moved_macro_weight: float = 0.25,
        fast_proxy_device: torch.device | str | None = None,
    ):
        self.grid_side = int(grid_side)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.moved_macro_weight = float(moved_macro_weight)
        _ = fast_proxy_device  # API parity with ExplorePlacer; always CUDA here.

    def place(
        self,
        benchmark: Benchmark,
        *,
        initial_macro_positions: torch.Tensor | None = None,
        save_figure: bool = False,
    ) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("ExploreGpuPlacer requires CUDA (no GPU detected).")

        device = torch.device("cuda")
        if initial_macro_positions is not None:
            placement = initial_macro_positions.to(device=device, dtype=torch.float32).clone()
        else:
            placement = benchmark.macro_positions.to(device=device, dtype=torch.float32).clone()

        placement_start = placement.clone()
        score = FastProxyEvaluator(
            benchmark,
            device=device,
            use_discrete_congestion=False,
        )

        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        n = self.grid_side
        rng = random.Random(self.seed)

        neighbors = _precompute_nine_neighbors(n)
        cell_w = cw / float(n)
        cell_h = ch / float(n)
        center_x = [(c + 0.5) * cell_w for c in range(n)]
        center_y = [(r + 0.5) * cell_h for r in range(n)]

        movable = _movable_macro_indices(benchmark)
        if not movable:
            print("ExploreGpuPlacer: no movable macros; placement unchanged.")
            return placement

        sizes_cpu = benchmark.macro_sizes.detach().cpu()
        macro_w = sizes_cpu[:, 0].tolist()
        macro_h = sizes_cpu[:, 1].tolist()

        moved_w = float(self.moved_macro_weight)
        if not (0.0 < moved_w <= 1.0):
            raise ValueError("moved_macro_weight must be in (0, 1].")
        moved_flags = {i: False for i in movable}
        weights = [1.0 for _ in movable]

        sur0 = score.total(placement)
        cur_cost = sur0
        accepted = 0

        for ep in range(self.epochs):
            i_macro = rng.choices(movable, weights=weights, k=1)[0]
            w = float(macro_w[i_macro])
            h = float(macro_h[i_macro])
            x0 = float(placement[i_macro, 0].item())
            y0 = float(placement[i_macro, 1].item())

            r0, c0, _, _ = _cell_xy_to_rc(x0, y0, cw, ch, n)
            candidates = neighbors[r0][c0]

            base_cost = cur_cost
            best_p = base_cost
            best_cx = x0
            best_cy = y0

            for r, c in candidates:
                cx, cy = center_x[c], center_y[r]
                if not _legal_center(cx, cy, w, h, cw, ch):
                    continue
                placement[i_macro, 0] = cx
                placement[i_macro, 1] = cy
                p = score.total(placement)
                placement[i_macro, 0] = x0
                placement[i_macro, 1] = y0
                if p < best_p:
                    best_p = p
                    best_cx = cx
                    best_cy = cy

            if best_p < base_cost - 1e-12:
                placement[i_macro, 0] = best_cx
                placement[i_macro, 1] = best_cy
                cur_cost = best_p
                accepted += 1
                if not moved_flags[i_macro]:
                    moved_flags[i_macro] = True
                    for j, m in enumerate(movable):
                        if moved_flags[m]:
                            weights[j] = moved_w

        sur_end = cur_cost
        delta_pct = (sur_end - sur0) / max(abs(sur0), 1e-30) * 100.0
        print(
            f"ExploreGpuPlacer: {self.epochs} epochs, grid {n}×{n} ({n * n} cells), "
            f"accepted {accepted} improving moves; "
            f"fast_proxy {sur0:.6f} -> {sur_end:.6f} ({delta_pct:+.3f}%)."
        )

        if save_figure:
            out_vis = _repo_root() / "vis" / f"{benchmark.name}_explore_gpu.png"
            _save_explore_figure(
                benchmark,
                placement_start.detach().cpu(),
                placement.detach().cpu(),
                grid_side=n,
                epochs=self.epochs,
                accepted=accepted,
                proxy_before=None,
                proxy_after=None,
                fast_before=sur0,
                fast_after=sur_end,
                out_path=out_vis,
                explorer_label="ExploreGpuPlacer",
            )
            print(f"ExploreGpuPlacer: figure saved to {out_vis.resolve()}")

        return placement


def _cli_main() -> None:
    root = _repo_root()
    case = root / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / "ibm01"
    b, _ = load_benchmark_from_dir(str(case))
    ExploreGpuPlacer(seed=1, epochs=50).place(b)


if __name__ == "__main__":
    _cli_main()
