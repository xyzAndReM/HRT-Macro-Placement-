"""
Hybrid pipeline (configurable QP):

1. Optional **QPLegalizer** on the testcase layout (``MACRO_PLACE_LEGAL_QP_INITIAL``).
2. **GpuPlacer** (``submissions/gpu/placer.py``) for **2000** epochs.
3. Optional **QPLegalizer** on the GPU output (``MACRO_PLACE_LEGAL_QP_FINAL``).

**Why QP can hurt quality:** QP minimizes squared movement subject to **overlap**
constraints — not PLC proxy / WL / congestion. The **final** QP pass especially
often **undoes** GPU progress on surrogate objectives while fixing overlaps.

Defaults: **initial QP on**, **final QP off** (return GPU placement unless you opt in).

Env:
    MACRO_PLACE_LEGAL_QP_INITIAL — ``1`` (default) or ``0``
    MACRO_PLACE_LEGAL_QP_FINAL — ``0`` (default) or ``1``

Writes ``vis/<benchmark.name>_legal.png`` at the end (three-panel figure).

Unless ``MACRO_PLACE_DEVICE`` is already set, this placer sets it to ``cuda`` for the
GPU phase. Use ``MACRO_PLACE_DEVICE=cpu`` to force CPU.

Usage:
    uv run evaluate submissions/legal.py -b ibm01
    set MACRO_PLACE_LEGAL_QP_FINAL=1   # re-enable post-GPU QP if you need legality
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import visualize_placement

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from submissions.gpu.placer import GpuPlacer  # noqa: E402
from submissions.qp import QPLegalizer  # noqa: E402

_GPU_EPOCHS = 2000


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _iccad04_case_dir(benchmark: Benchmark) -> Path | None:
    d = _ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / benchmark.name
    return d if (d / "netlist.pb.txt").is_file() else None


@contextmanager
def _default_cuda_for_gpu_phase():
    """If unset, prefer CUDA for ``GpuPlacer`` (see ``_select_device`` in gradient.py)."""
    if os.environ.get("MACRO_PLACE_DEVICE") is not None:
        yield
        return
    os.environ["MACRO_PLACE_DEVICE"] = "cuda"
    try:
        yield
    finally:
        os.environ.pop("MACRO_PLACE_DEVICE", None)


class LegalPlacer:
    """
    Optional QP → ``GpuPlacer(epochs=2000)`` → optional QP.

    Saves ``vis/<name>_legal.png`` when ICCAD04 collateral exists.

    The evaluate loader instantiates this class with no arguments.
    """

    def place(self, benchmark: Benchmark):
        qp0 = _env_bool("MACRO_PLACE_LEGAL_QP_INITIAL", True)
        qp1 = _env_bool("MACRO_PLACE_LEGAL_QP_FINAL", False)

        if qp0:
            pos = QPLegalizer().place(benchmark)
        else:
            pos = benchmark.macro_positions.clone()

        gpu = GpuPlacer(
            epochs=_GPU_EPOCHS,
            stagnation_proxy_patience=0,
        )
        with _default_cuda_for_gpu_phase():
            pos = gpu.place(benchmark, initial_macro_positions=pos)

        if qp1:
            final = QPLegalizer().place(benchmark, initial_macro_positions=pos)
        else:
            final = pos

        pos_cpu = final.detach().cpu()
        case_dir = _iccad04_case_dir(benchmark)
        plc = None
        if case_dir is not None:
            _, plc = load_benchmark_from_dir(str(case_dir))
            compute_proxy_cost(pos_cpu.clone(), benchmark, plc)

        vis_dir = _ROOT / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        out_png = vis_dir / f"{benchmark.name}_legal.png"
        visualize_placement(pos_cpu, benchmark, save_path=str(out_png.resolve()), plc=plc)

        return final
