## Macro-selection criteria shootout — notes / status

### What changed already

#### `submissions/place_explore.py`
- **Schedule**: for each outer round do **`GpuPlacer -> ExplorePlacer`**, and after all rounds do **one final `GpuPlacer`**.
- **Default env fallbacks restored**:
  - `MACRO_PLACE_PE_OUTER_ROUNDS`: fallback **1**
  - `MACRO_PLACE_PE_EXPLORE_EPOCHS`: fallback **200**

#### `submissions/explore.py`
- **Speedups (same accept logic)**:
  - Carry `cur_cost` so we avoid recomputing `score.total(placement)` as the base cost each epoch.
  - No per-candidate cloning: in-place trial move + restore.
  - Precompute 3×3 neighbor lists and coarse cell centers.
  - Cache macro sizes into Python lists.
- **Bias against re-moving the same macros**:
  - Added `moved_macro_weight` (default `0.25`) so macros that already moved once are less likely to be picked again.

### Macro-selection criteria experiment

#### Implemented script
- File: `submissions/macro_pick_experiment.py`
- Benchmark: ICCAD04 **`ibm01`**
- Epochs: default **200**
- Seeds: default **0..9**
- Output: CSV written to `results/macro_pick_ibm01.csv`
- Winner metric: **final real `compute_proxy_cost(...)[\"proxy_cost\"]`**
- Criteria compared:
  - `uniform_random`
  - `downweight_moved`
  - `round_robin`
  - `stale_first`
  - `cell_hotspot`
  - `area_weighted`

#### How it runs
- Uses ExplorePlacer-like loop:
  - Pick macro via criterion
  - Evaluate 3×3 neighbor cell centers
  - Accept only strictly improving `FastProxyEvaluator.total(...)`
  - At end compute real proxy via `compute_proxy_cost`

### What blocked us running it
- `python submissions/macro_pick_experiment.py ...` failed because that interpreter didn’t have `torch`.
- `uv run python submissions/macro_pick_experiment.py ...` initially failed because `submissions` wasn’t on `sys.path`.
  - Fixed by adding a `_ROOT`/`sys.path.insert` bootstrap (same pattern as other submission files).
- After the fix, `uv run ...` started but produced **no output for a long time** (had not finished diagnosing whether it was just slow startup or stuck before first print).

### Next steps (tomorrow)
- Run a tiny sanity check first (1 seed, 1 criterion) to confirm output appears quickly.
- If it’s still slow/hanging, add early debug prints (start-of-script, after benchmark load, after first run) and/or add flags like `--criteria uniform_random` to run a subset.
- Once it runs end-to-end, inspect `results/macro_pick_ibm01.csv` and report ranking by mean final real proxy.

