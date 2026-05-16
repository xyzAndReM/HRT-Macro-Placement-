# Session notes — gradient + GPU proxy calibration

Working files: `submissions/gradient.py`, `submissions/gpu/placer.py` (patience / liquid).
Eval: `uv run evaluate submissions/gradient.py -b ibm01` or `uv run evaluate submissions/liquid.py -b ibm01`.

## 2026-05-15 — `GpuPlacer` affine scale: WL EMA (after congestion EMA v2)

**Problem:** After congestion moved to `_fit_scale_cong` (level EMA, `α=0.15`), `w_wl` became
the noisiest calibrated term. Slope-based `_fit_scale_surrogate_to_proxy` made `w_wl` oscillate
(e.g. 1.079 → 1.399 → 1.519 → 1.079) while the true `px_wl / sur_wl` ratio drifts slowly
(~1.071→1.080 over 1000 epochs on ibm01).

**Change (`submissions/gpu/placer.py`):**
- Added **`_fit_scale_wl`**: EMA on **`px_wl / sur_wl`**, `alpha=0.20`, clamp **`[0.5, 3.0]`**
  (faster than congestion because WL ratio is cleaner).
- State **`ema_w_wl`**, initialized to constructor **`w_wl`**.
- Patience **`affine_calibrate`** block: WL uses EMA; congestion unchanged; **density** still
  uses `_fit_scale_surrogate_to_proxy` (harmless — `w_den` stays ~1.0 when surrogate matches PLC).

**Expect:** `w_wl` converges in ~5 proxy checks without spikes; `err_wl` should stay small
(<~0.002) once settled. Congestion EMA unchanged (`_fit_scale_cong`, `α=0.15`, clamp 0.5–1.5).

---

## 2026-05-15 — `LiquidPlacer` / ibm02 baseline log (explore `w_density=0.8`, `w_overlap=0.5`)

Annotated harness excerpt (`evaluate submissions/liquid.py -b ibm02`), **2 cycles**, explore density/overlap as noted (defaults before explore weights moved to 0.5 / 0.5):

| Field | Value |
| --- | --- |
| Final proxy | **1.1983** (harness line; stagnation stop saw **1.19769** at last PLC check) |
| Initial proxy | **1.5658** |
| Δ vs initial | **-23.47%** |
| Subcosts (reported) | wl **0.108**, den **0.501**, cong **1.680** |
| Legal | **VALID** (post-Abbacus) |
| Wall-clock | **~2698 s** (~45 min) |
| Cycle 2 PLC gate | `delta_vs_cycle_start` **-0.0494** (improved vs cycle start; gate OK) |
| Patience exit | PLC stagnation **4/4** consecutive checks with improvement **&lt; 0.0001** vs best (`best≈1.19371`, `current≈1.19769` — slight drift above best at stop) |
| Artifacts | `vis/ibm02_gpu_proxy_vs_epoch.png`, `vis/ibm02_gpu_surrogate_loss_vs_epoch.png`, `logs/liquid_ibm02_congestion.md`, `vis/ibm02.png` |

**Takeaway:** Large proxy gain vs initialization with legal placement; patience stopped on tight **1e-4** stagnation while proxy briefly sat above session best (~**0.3%** relative overshoot).

---

## 2026-05-15 — `LiquidPlacer` current defaults vs previous ibm02 run (reference)

**Previous logged run** (ibm02, `evaluate submissions/liquid.py -b ibm02`, ~2698 s, VALID, proxy **1.198** from **1.566**):

| Knob | Previous execution | **Current** (`submissions/liquid.py` defaults) |
| --- | --- | --- |
| Cycles | **2** | **2** |
| Explore `w_density` | **0.8** | **0.8** |
| Explore `w_overlap` | **0.5** | **0.5** |
| Explore `lr` | **1e-2** | **1e-2** |
| Patience `lr` | **0.02** (GpuPlacer default) | **0.02** (GpuPlacer default) |
| Explore stop | surrogate plateau `0.001×` initial, patience **3**, every **50** | same |
| Patience stop | PLC `0.0001` min improve, patience **4**, every **50** | same |
| Patience `w_density` / `w_overlap` | **0.5** / **1.0** (GpuPlacer defaults) | unchanged |
| `w_wl` / `w_cong` (both phases) | **1.0** / **0.5** | unchanged |

**Also in codebase (not liquid schedule):** Abbacus fallback default `max(5000, 25×n_hard)`, `alpha=0.3`, post-fallback cooperative 50/50 GS sweep; WL/cong EMA calibration in patience (`_fit_scale_wl`, `_fit_scale_cong`).

Use this table when comparing the next ibm02 (or `--all`) run to the annotated baseline above.

**2026-05-15 update:** `liquid.py` defaults reverted to match the **Previous execution** column (2 cycles, explore cap **20000**, explore `0.8`/`0.5`/`1e-2`, patience stagnation **4** vs **best**, `lr` = GpuPlacer default `0.02`).

---

## `submissions/gradient.py` (earlier session)

## What changed today

### 1. WL surrogate: LSE → exact HPWL (subgradients) ✅ landed
- `_wirelength_loss_v2`: replaced 4× `logsumexp` with `amax`/`amin` along `dim=1`.
  Mask via `masked_fill(~net_mask, ±inf)`, then `torch.where(net_valid, hpwl, 0)`
  to avoid `0 * inf = nan` on degenerate 0-pin nets.
  `beta` argument is kept in the signature (compiled-binding compat) but unused.
- `_wirelength_loss_loop`: same swap so the `epoch == 0` `torch.allclose` parity
  check still passes against the reference.
- Removed `_lse_beta`, `_soft_hpwl_1d` (now dead).

**Why**: proxy WL *is* HPWL (`plc.get_cost()`), so the surrogate now matches the
metric exactly. Subgradients are sparse but Adam handles it well.

### 2. RUDY: tried exact `amin`/`amax` bbox → REGRESSED → reverted ❌
- Swapped, measured, reverted. Recorded the result so we don't repeat the experiment.
- Result on ibm01 (`w_wl=1`): exact-bbox RUDY proxy=1.0399 vs LSE-RUDY proxy=1.0309 (**+0.87% worse**).
- Reason: `plc.get_congestion_cost()` is *not* a per-net bbox metric, so neither
  variant is "exact" w.r.t. the proxy. LSE-RUDY's slightly bloated bbox gives:
  - dense gradients (every pin contributes, not just bbox extremes),
  - smoother per-bin demand maps,
  which empirically beat the sparse exact-bbox version.
- File comment in `_rudy_loss_v2` records this so it isn't tried again.

### 3. `w_wl` sweep on ibm01 (post WL swap) ✅ default updated to 20
With LSE-RUDY + exact HPWL, sweep over `w_wl` (everything else default):

| w_wl    | proxy   | Δ vs init |
| ------- | ------- | --------- |
|  1.0    | 1.0309  | -0.73%    |
|  5.0    | 1.0309  | -0.73%    |
| 10.0    | 1.0308  | -0.74%    |
| 12.0    | 1.0301  | -0.80%    |
| 15.0    | 1.0285  | -0.96%    |
| 18.0    | 1.0287  | -0.95%    |
| **20.0**| **1.0281** | **-1.00%** |
| 22.0    | 1.0288  | -0.94%    |
| 25.0    | 1.0295  | -0.87%    |
| 30.0    | 1.0330  | -0.53%    |
| 50.0    | 1.0398  | +0.12%    |

Broad optimum 15–22, hard rolloff past 30.
**Default in `GradientPlacer.__init__` is now `w_wl=20.0`** (was 1.0).

## Current state on ibm01 (defaults, no QP)

```
proxy = 1.0281    initial = 1.0385    Δ = -1.00%
   wl = 0.065     den = 0.778         cong = 1.148
   116 hard-hard overlaps  (QP disabled)
   ~19s wall-clock
```

`L_wl` still creeps up slightly per epoch (14530 → 14563) — optimizer is trading
WL for density/congestion. That's healthy now (gradient is honest), but suggests
there's still room.

## Defaults in `GradientPlacer.__init__`

```python
epochs=100, lr=0.001, beta=1.0, seed=0,
target_hard=0.7, target_soft=0.8,
w_wl=20.0, w_dens_hard=100.0, w_dens_soft=50.0, w_cong=100.0,
anneal=1.0, qp_legalize=False,
```

`beta` is now only used by RUDY (LSE).

## ⚠ Open issue: loss keeps dropping but proxy keeps getting worse past ~100 epochs

Confirmed on ibm01 with current defaults: training loss is monotonically decreasing
but the *real proxy* gets worse the longer we train. Running 500 epochs returns a
worse placement than 100 epochs.

**Diagnosis** — the surrogate's relative weights don't match the proxy's:

| Term       | Proxy weight | Surrogate weight | Ratio (surrogate / proxy) |
| ---------- | -----------: | ---------------: | ------------------------: |
| WL         |          1.0 |             20.0 |                       20× |
| Density H  |          0.5 |            100.0 |                      200× |
| Density S  |          0.5 |             50.0 |                      100× |
| Congestion |          0.5 |            100.0 |                      200× |

So the surrogate weights density/cong **10× harder** than the proxy does (relative
to WL). Around epoch ~100 we're near the proxy optimum; past that, Adam keeps
trading WL away to chase further density/cong gains because that's what the loss
asks for. The 100-epoch trace already shows it: `L_wl` 14530 → 14563 (going UP)
while `L_dh` and `L_cong` keep dropping.

**Likely fixes (cheap → expensive)**:

1. **Best-snapshot return** — track best proxy seen during training, return that.
   Removes sensitivity to `epochs` entirely. One proxy eval per N epochs is cheap
   (proxy uses `plc.get_*_cost()`, fast on CPU). **Top priority — implement first.**
2. **Re-balance weights to proxy ratios** — try
   `w_wl=20, w_dens_hard=20, w_dens_soft=10, w_cong=20` (or similar) so
   surrogate ratio ≈ proxy ratio (1 : 0.5 : 0.5 with overall scale 20).
   Will need its own sweep. Removes the structural bias.
3. **LR decay** — `CosineAnnealingLR(T_max=epochs)`. Helps Adam settle near the
   minimum instead of overshooting it.
4. **Reverse anneal** — keep current high density/cong weights for first ~30
   epochs (break overlaps), then *decay* them toward proxy-matched ratios so WL
   takes over for the final 70 epochs.

## Next steps (in priority order)

1. **Best-snapshot return** (see Open Issue above) — biggest single fix.
2. **Re-enable QP legalization** (`qp_legalize=True`) and check post-QP proxy.
   With the honest WL the trajectory should be cleaner, so QP's job is easier.
   116 overlaps currently → DISQUALIFIED until legalized.
3. **Re-balance loss weights** so surrogate ratio matches proxy 1:0.5:0.5.
4. **Validate `w_wl=20` (and any new defaults) across `--all`** — the sweep
   was ibm01-only. Run `uv run evaluate submissions/gradient.py --all`.
5. **Soft-macro density is still terrible**: max bin Gaussian occupancy ≈ 3.5
   vs target 0.8. Soft macros are stacking. Either bump `w_dens_soft` (currently
   50) or revisit the soft density loss.
6. **β-annealing for RUDY only**: now that WL doesn't need β, we could schedule
   β_rudy from low (smooth) to high (closer to true bbox) over training.
7. **`w_wl` warmup**: start at 1.0, ramp to 20.0 — give density/congestion
   a head-start on un-tangling, then pull WL tight.

## Things NOT to retry (recorded so we don't burn time again)

- Exact `amin`/`amax` for RUDY — regresses by ~0.9%. Stick with LSE.
- Pushing `w_wl` past ~25 — over-stretches, proxy regresses sharply.
- `anneal=1.01` — runs away over 100 epochs, weights blow up. Keep `anneal=1.0`
  unless you also cap or reset weights.
- GPU on Pascal (sm_61) for this workload — `_select_device` already routes to
  CPU for it; FP64 + sm_61 + small problems = slower than CPU.

## Files touched

- `submissions/gradient.py` (only file modified this session).
- No other source files changed.
