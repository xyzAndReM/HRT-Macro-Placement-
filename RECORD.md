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

---

## 2026-05-15 — `GpuPlacer`: online congestion scale (`_fit_scale_cong`)

**Problem:** With pin L-route surrogate, PLC **`px_cong`** is often below GPU **`sur_cong`**; WL/density calibrate each proxy check, but congestion was long fixed at **`w_cong_run = 1`**, overweighting congestion vs true proxy.

### v1 (same day, superseded)

- **`_fit_scale_cong`**: blend slope + level, **30/70** momentum vs previous weight, clamp **`[0.2, 5.0]`**.
- **Observation (ibm01 logs):** **`w_cong`** oscillated (~**0.61 ↔ 0.97**) every few checkpoints — slope across 50-epoch gaps too noisy.

### v2 (current)

- **`_fit_scale_cong`**: **EMA only** on the level ratio **`px_cong / sur_cong`**:  
  `new = (1-α)·ema + α·level` with **`α = 0.15`**, then clamp **`[0.5, 1.5]`**.
- Separate state **`ema_w_cong`** (init **1.0**) per `place()` run; **`w_cong_run = ema_w_cong`** after each proxy check when **`affine_calibrate`**.
- Rationale: **`px/sur`** ~**0.81–0.83** stable on ibm01-like runs; no slope term; tighter clamp matches “~18% overestimate” band and avoids zeroing congestion.

**Unchanged:** Liquid phase has **`MACRO_PLACE_GPU_PROXY_CHECK_EVERY=0`** — no congestion calib there.

---

## 2026-05-13 — `LiquidPlacer` / `GpuPlacer` congestion surrogate & proxy calibration

**Focus:** ibm02 liquid→patience cycles; align GPU congestion surrogate with PLC proxy without killing GPU throughput.

### Congestion surrogate path (evolution)

1. **Started** with pin L-route Python loop (`plc_routing_surrogate_scalar_pins`) — accurate-ish but **~1 min/epoch** on ibm02 (per-net `.item()` + Python loop). **Reverted default to RUDY.**
2. **BBox RUDY** (`_rudy_demand_grid`) — fast but **~2× low** vs PLC (~1.2 vs ~2.4 on ibm02); poor proxy tracking.
3. **GPU-batched macro-center L-routes** (`_collect_l_route_segments_batched` + `plc_routing_surrogate_scalar`) — same PLC-style H/V deposit + smooth + blockage + top-5% mean; still macro centers.
4. **GPU-batched pin-aware L-routes** (final default when `net_pin_nodes` complete):
   - `_pin_xy_from_pin_net_idx`, `_routing_weights_pin_batched`, `_collect_l_route_segments_pin_batched`
   - `plc_routing_surrogate_scalar_pins` rewritten to use batched path (no per-net Python loop)
   - Raw surrogate scale **~2.1–2.6** vs PLC **~1.97–2.33** (≈8–15% gap vs old 2× gap)
   - Env: `MACRO_PLACE_GPU_PIN_CONG=0` → macro L-routes; `MACRO_PLACE_GPU_USE_BBOX_RUDY=1` → legacy RUDY

### Proxy logging & patience (`liquid.py` + `gpu/placer.py`)

- **`surrogate_vs_proxy`** + **`scale_calib`** every **`MACRO_PLACE_LIQUID_PROXY_CHECK_EVERY`** (default **50**)
- **Stagnation:** require **5 consecutive** sub-threshold proxy checks (`MACRO_PLACE_LIQUID_STAGNATION_PATIENCE=5`) before stop (not 1)
- Liquid phase: no proxy checks; patience: `affine_calibrate=True`

### Scale calibration (`w` only, no `k`)

- Removed intercept **`k`** — loss uses **`w * surrogate`** only; proxy coeffs **1.0 / 0.5 / 0.5** applied in loss sum
- **`_fit_scale_surrogate_to_proxy`**: fit WL/density `w`; target `w*sur ≈ px` (raw PLC subcost)
- **2026-05-14:** congestion scale was fixed at **`w_cong = 1`** (pin surrogate near PLC scale; two-point `w` was unstable).
- **2026-05-15 (patience `affine_calibrate`):** congestion **`_fit_scale_cong`** — **v2:** EMA on **`px/sur`** (`α=0.15`, clamp **0.5–1.5**); **v1** (blend+slope, clamp 0.2–5) dropped after oscillation in logs. See section **2026-05-15** above.

### Observations (ibm02)

- Pin L-route + `w=1` on cong: surrogate and PLC congestion **trend down together**; proxy improved ~1.51 → ~1.33 over long patience runs
- WL/density track PLC well; congestion raw gap much smaller than RUDY era
- Remaining gap: surrogate sometimes **over**-estimates PLC slightly; shape mismatch when PLC drops faster than surrogate

### Files touched

| Path | Changes |
|------|---------|
| `submissions/gpu/placer.py` | Congestion modes; scale calib; **`_fit_scale_cong`** for `w_cong_run` when `affine_calibrate`; logging; stagnation |
| `submissions/liquid.py` | Cycle schedule docs; stagnation patience 5; proxy every 50 |
| `macro_place/routing_surrogate.py` | Batched L-route segment collectors; vectorized pin congestion |

### Env quick reference

| Variable | Default | Meaning |
|----------|---------|---------|
| `MACRO_PLACE_GPU_USE_RUDY_CONG` | `1` | GPU congestion on (not PLC CPU checks) |
| `MACRO_PLACE_GPU_PIN_CONG` | `1` | Pin L-routes when pin tables exist |
| `MACRO_PLACE_GPU_USE_BBOX_RUDY` | `0` | `1` = legacy bbox RUDY |
| `MACRO_PLACE_LIQUID_PROXY_CHECK_EVERY` | `50` | Proxy log + calib interval |
| `MACRO_PLACE_LIQUID_STAGNATION_PATIENCE` | `5` | Consecutive bad proxy checks before stop |

### Possible next steps (not done)

- Soft differentiable ABU instead of hard `topk` mean
- Clamp WL-only calib or single-point `w=px/sur` with narrow band for WL
- Vectorized pin offsets already in; further PLC alignment may need route-weight / 3-pin rules

