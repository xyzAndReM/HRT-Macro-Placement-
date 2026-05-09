# Session record — macro-place-challenge-2026

Date: 2026-05-04 (local). Purpose: handoff after sleep + performance direction.

---

## What we did today (concise)

### Gradient runner / evaluation script

- **`scripts/evaluate_gradient_random_start.py`**
  - **`--save-placement PATH`**: after training, torch-save dict (`placement`, `benchmark`, metadata) for resume or QP-only workflows.
  - **`--load-placement PATH`**: seed `benchmark.macro_positions` before training; skips random initial layout.
  - **`--qp-only`**: no training — load checkpoint, run **`QPLegalizer`**, print proxy before/after; optional **`--save-placement`** to write legalized tensor. Mutually exclusive with **`--legalize-at-end`**.
  - **`--legalize-at-end`**: after training, QP-legalize the **returned** placement; print **Δ vs initial** before and after legalization; append **`[post_qp]`** line to **`--training-log-csv`** when logging is enabled.
  - **`--epoch-timing-diagnostic PATH`**: per-epoch segment timings (see below).

### Plotting

- **`scripts/plot_training_log_proxy.py`**: read training CSV from `GradientPlacer` logs, plot **proxy vs epoch**; supports `-o outfile.png`. Default stem-based name if `-o` omitted.

### Surrogate / proxy interpretation (discussion)

- Clarified: with **`proxy_eval_interval=20`**, total **`compute_proxy_cost`** time is often **not** the dominant wall time vs thousands of **per-epoch** surrogate + **backward** steps.
- Benchmarked (one environment): **`compute_proxy_cost`** on **ibm01** ~**1.8–1.9 s/call** (PLC path); 50 calls ≈ ~100 s vs multi-hour runs is plausible **minor** fraction — **per-epoch backward + congestion surrogate** tend to dominate.

### Epoch timing diagnostic (implemented)

- **`submissions/gradient.py` — `GradientPlacer`**
  - New kwarg: **`epoch_timing_diagnostic: str | Path | None = None`**.
  - When set: writes **`diagnostic.txt`** (or chosen path) with:
    - `#` header (device, flags, canvas, macro counts, epochs, proxy interval).
    - CSV columns: **`epoch`**, then **`_EPOCH_DIAG_NUMERIC_COLS`**:  
      `t_zero_grad`, `t_assemble`, `t_cat_ports`, `t_epoch0_audit`, `t_epoch0_compile_check`, `t_epoch0_proxy_baseline`, `t_wl`, `t_overlap_grid`, `t_density_scalar`, `t_cong`, `t_loss_scalar`, `t_backward`, `t_optimizer`, `t_clamp`, `t_proxy_checkpoint`, **`t_epoch_wall_s`**.
    - **`try` / `finally`**: footer **`# summary`** with **mean / median / p95** per column, **`exit_reason`**, **`total_training_wall_s`** (from existing `t_start` in `place()`).
  - CUDA: extra **`_sync(device)`** only when diagnostic enabled (avoids slowing default runs).

### Smoke observation (ibm01, 2 epochs, CPU, `proxy_eval_interval=0`)

- **`t_backward`** ~**2 s** — largest **steady** bucket (autograd through full loss).
- **`t_cong`** (PLC routing surrogate) ~**1 s+** — second-largest recurring cost.
- **`t_overlap_grid`** noticeable; **`t_epoch0_audit`** / **`t_epoch0_proxy_baseline`** are **epoch-0-only** (means in a 2-row summary look “medium” but are one-off spikes).

---

## What to do next (suggested)

### Speed — highest leverage ideas

1. **Hardware / device**
   - Train on **GPU** when available (`MACRO_PLACE_DEVICE` unset or `cuda`); keep in mind **`compute_proxy_cost`** stays **CPU/PLC** — you still pay proxy time at checkpoints, but **backward / surrogates** usually shrink a lot.

2. **Surrogate cost**
   - A/B **`use_plc_routing_cong=False`** (RUDY overflow path): often **faster** per epoch; may hurt proxy alignment — measure proxy delta, not only speed.
   - **`torch.compile`**: already used for some WL/Rudy paths when supported; extending or warming compile for the **congestion** path (if safe) is an experiment.

3. **Fewer / cheaper epochs**
   - **`proxy_eval_interval`**, **`proxy_patience`**, **`max_wall_seconds`**, flat-loss early stop — less work without changing math much.

4. **Backward / memory–time tradeoffs**
   - **`torch.utils.checkpoint`** on the **congestion** subgraph (if refactor-friendly): **slower forward, less memory**, sometimes net win on GPU; on CPU often **not** a win — profile first.

5. **Profiling focus**
   - Use **`--epoch-timing-diagnostic`** on a **representative** run (same `epochs`, `proxy_eval_interval`, device as production).
   - Optional: PyTorch **`profiler`** (`schedule`, `tensorboard` export) on **one** epoch to see **which ops** inside backward dominate (likely congestion-related ops).

6. **PlacementCost proxy**
   - Cannot “compile away” without changing **`PlacementCost`**; reduce **call count** or accept cost. **`_set_placement`** Python loop is minor vs **`get_*`** internals.

### Product / workflow

- **Git**: push to your remote when ready (`git remote set-url`, `git add`, `commit`, `push`); avoid committing huge **`logs.txt` / `*.png`** unless intentional — consider **`.gitignore`** for local artifacts.
- **Windows + RTX 5070**: native **Python + `uv`** usually easier than GPU VM; match **PyTorch CUDA wheel** to driver if **`cu126`** wheels misbehave.

### Possible code follow-ups (not done)

- Expose **`lr`**, **`qp_legalize`**, **`proxy_adaptive_weights`** on **`evaluate_gradient_random_start`** CLI for faster sweeps without editing **`gradient.py`**.
- Single-command “train + save **legalized** placement” (today: train **`--save-placement`**, then **`--qp-only --save-placement`**).
- Optional **`--timing-every-k`** if diagnostic CSVs get huge.

---

## Files touched / added (reference)

| Path | Role |
|------|------|
| `scripts/evaluate_gradient_random_start.py` | CLI: placement save/load, QP-only, legalize-at-end, epoch diagnostic |
| `scripts/plot_training_log_proxy.py` | Proxy vs epoch plot |
| `submissions/gradient.py` | `epoch_timing_diagnostic`, timing helpers, `GradientPlacer` loop instrumentation |

---

## One-line “why backward is slow”

**`loss.backward()`** runs reverse autodiff through **wirelength + density + (especially) PLC congestion surrogate** in one graph; that graph is large on **ibm01**, so the **adjoint pass** dominates wall time on CPU.

---

*End of RECORD — add dated sections below for future sessions.*
