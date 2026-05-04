#!/usr/bin/env python3
"""Plot evaluator proxy vs epoch from GradientPlacer training log (CSV + optional comment lines)."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_proxy_epochs_csv(path: Path) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    proxies: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            if line.startswith("epoch,"):
                continue
            parts = next(csv.reader([line]))
            if len(parts) < 7:
                continue
            try:
                ep = int(parts[0])
            except ValueError:
                continue
            proxy_cell = parts[6].strip()
            if not proxy_cell:
                continue
            try:
                pv = float(proxy_cell)
            except ValueError:
                continue
            epochs.append(ep)
            proxies.append(pv)
    return epochs, proxies


def parse_proxy_from_surrogate_lines(path: Path) -> tuple[list[int], list[float]]:
    """Fallback: pv= from [surrogate_vs_proxy] lines (same checkpoints as CSV proxy)."""
    pat = re.compile(r"epoch=(\d+).*?pv=([\d.]+)")
    epochs: list[int] = []
    proxies: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if "[surrogate_vs_proxy]" not in line:
                continue
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                proxies.append(float(m.group(2)))
    return epochs, proxies


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=Path("logs.txt"),
        help="Training log path (default: ./logs.txt).",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <log_stem>_proxy_vs_epoch.png next to log).",
    )
    args = ap.parse_args()
    log_path = args.log.resolve()
    if not log_path.is_file():
        raise SystemExit(f"Not found: {log_path}")

    epochs, proxies = parse_proxy_epochs_csv(log_path)
    if not epochs:
        epochs, proxies = parse_proxy_from_surrogate_lines(log_path)
    if not epochs:
        raise SystemExit(f"No proxy values found in {log_path}")

    out = args.output
    if out is None:
        out = log_path.parent / f"{log_path.stem}_proxy_vs_epoch.png"

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.plot(epochs, proxies, "o-", markersize=3, linewidth=1, color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("proxy cost (PlacementCost)")
    ax.set_title(f"Proxy vs epoch — {log_path.name}")
    ax.grid(True, alpha=0.35)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out} ({len(epochs)} points)")


if __name__ == "__main__":
    main()
