#!/usr/bin/env python3
"""Plot |sur_raw_cong - px_raw_cong| from [surrogate_vs_proxy] lines in a training log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_log(text: str) -> tuple[list[int], list[float]]:
    pat = re.compile(
        r"\[surrogate_vs_proxy\] epoch=(\d+) "
        r"sur_raw\(wl,dh,cong\)=\(([^,]+),([^,]+),([^)]+)\) "
        r"px_raw\(wl,den,cong\)=\(([^,]+),([^,]+),([^)]+)\)"
    )
    epochs: list[int] = []
    gaps: list[float] = []
    for m in pat.finditer(text):
        ep = int(m.group(1))
        sur_c = float(m.group(4).strip())
        px_c = float(m.group(7).strip())
        epochs.append(ep)
        gaps.append(abs(sur_c - px_c))
    return epochs, gaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "log_path",
        type=Path,
        help="Training log containing [surrogate_vs_proxy] lines",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: next to log, *_cong_sur_px_abs_delta.png)",
    )
    args = ap.parse_args()

    text = args.log_path.read_text(encoding="utf-8")
    epochs, gaps = parse_log(text)
    if not epochs:
        raise SystemExit("No [surrogate_vs_proxy] lines found.")

    out = args.output
    if out is None:
        out = args.log_path.with_name(
            args.log_path.stem + "_cong_sur_px_abs_delta.png"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(epochs, gaps, color="tab:red", linewidth=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("|sur_raw_cong − px_raw_cong|")
    ax.set_title(f"{args.log_path.name} — absolute congestion surrogate − proxy gap")
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out.resolve()} ({len(epochs)} points)")


if __name__ == "__main__":
    main()
