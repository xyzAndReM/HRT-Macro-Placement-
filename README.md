# Liquid (`submissions/liquid.py`)

Alternates two GPU placement phases (liquid → patience), then Abbacus legalization. Optional QP if overlaps remain.

### Pipeline (per cycle, default 1)

1. **Liquid phase** — surrogate loss only (wirelength, density, overlap, L-route congestion). No PLC proxy. Stops on surrogate stagnation (default: &lt; `0.001 ×` initial loss for one check every 50 epochs) or epoch cap 20,000.
2. **Patience phase** — surrogate + PLC proxy every 50 epochs, weight calibration. Stops on proxy stagnation (default: &lt; `0.0001` vs. best for one check) or epoch cap 20,000.
3. **Cycle gate** — if proxy worsened vs. pre-cycle, restore snapshot and stop cycles.
4. **Abbacus** — hard-macro legalization.
5. **Optional QP** — if `MACRO_PLACE_LIQUID_QP_IF_OVERLAPS=1` and overlaps remain.

### Run

```bash
uv run evaluate submissions/liquid.py -b ibm01
uv run evaluate submissions/liquid.py --all
```

```powershell
$env:PYTHONPATH = "."
uv run evaluate submissions/liquid.py -b ibm01
```

Quiet default: `proxy=0.8312 elapsed=1686.4s VALID`. Verbose: `MACRO_PLACE_LIQUID_VERBOSE=1`.

### Results

From `logs/ibm*_run_liquid.txt`. Lower proxy is better.

| Benchmark | Proxy | Initial | Δ% | Time (s) | Time (min) | Status |
|-----------|------:|--------:|---:|---------:|-----------:|--------|
| ibm01 | 0.8312 | 1.0383 | −19.9 | 1686.4 | 28.1 | VALID |
| ibm02 | 1.2239 | 1.5658 | −21.8 | 2717.2 | 45.3 | VALID |
| ibm03 | 1.0228 | 1.3255 | −22.8 | 2192.0 | 36.5 | VALID |
| ibm04 | 1.0748 | 1.3133 | −18.2 | 1230.8 | 20.5 | VALID |
| ibm06 | 1.2433 | 1.6577 | −25.0 | 2536.5 | 42.3 | VALID |
| ibm07 | 1.1437 | 1.4758 | −22.5 | 2667.0 | 44.5 | VALID |
| ibm08 | 1.1543 | 1.4664 | −21.3 | 2410.1 | 40.2 | VALID |
| ibm09 | 0.8811 | 1.1126 | −20.8 | 1943.8 | 32.4 | VALID |
| ibm10 | 1.1432 | 1.3397 | −14.7 | 4412.5 | 73.5 | VALID |
| ibm11 | 0.9885 | 1.2141 | −18.6 | 2665.7 | 44.4 | VALID |
| ibm12 | 1.2495 | 1.6251 | −23.1 | 5748.3 | 95.8 | VALID |
| ibm13 | 0.9700 | 1.3854 | −30.0 | 4568.8 | 76.1 | VALID |
| ibm14 | 1.3001 | 1.5938 | −18.4 | 5053.4 | 84.2 | VALID |
| ibm15 | 1.2878 | 1.6033 | −19.7 | 2542.5 | 42.4 | VALID |
| ibm16 | 1.1198 | 1.4911 | −24.9 | 4061.8 | 67.7 | VALID |
| ibm17 | 1.3951 | 1.7391 | −19.8 | 3722.0 | 62.0 | VALID |
| ibm18 | 1.4273 | 1.7899 | −20.3 | 2571.5 | 42.9 | VALID |
| **AVG** | **1.1445** | **1.4551** | **−21.3** | **3101.8** | **51.7** | **0 overlaps** |

Total: 52 730 s (~14.6 h) over 17 benchmarks.

### `MACRO_PLACE_LIQUID_*` environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `MACRO_PLACE_LIQUID_CYCLES` | `1` | Max liquid → patience pairs |
| `MACRO_PLACE_LIQUID_GPU_EPOCHS` | `20000` | Liquid epoch cap (`0` skips liquid) |
| `MACRO_PLACE_LIQUID_PATIENCE_EPOCHS` | `20000` | Patience epoch cap |
| `MACRO_PLACE_LIQUID_LR` | `1e-2` | Learning rate |
| `MACRO_PLACE_LIQUID_W_DENSITY` | `0.8` | Liquid density weight |
| `MACRO_PLACE_LIQUID_PATIENCE_W_DENSITY` | `0.5` | Patience density weight |
| `MACRO_PLACE_LIQUID_W_OVERLAP` | `0.5` | Liquid overlap weight |
| `MACRO_PLACE_LIQUID_PROXY_CHECK_EVERY` | `50` | Patience proxy check interval |
| `MACRO_PLACE_LIQUID_STAGNATION_MIN_ABS` | `0.0001` | Min proxy improvement per check |
| `MACRO_PLACE_LIQUID_STAGNATION_PATIENCE` | `1` | Sub-threshold proxy checks before stop |
| `MACRO_PLACE_LIQUID_SURROGATE_CHECK_EVERY` | `50` | Liquid surrogate check interval |
| `MACRO_PLACE_LIQUID_SURROGATE_STAG_PATIENCE` | `1` | Sub-threshold surrogate checks before stop |
| `MACRO_PLACE_LIQUID_QP_IF_OVERLAPS` | off | QP after Abbacus if overlaps remain |
| `MACRO_PLACE_LIQUID_RANDOM_HARD_START` | off | Random movable hard-macro centers |
| `MACRO_PLACE_LIQUID_SAVE_PLACEMENT` | off | Save `logs/liquid_<bench>_placement.pt` |
| `MACRO_PLACE_LIQUID_VERBOSE` | off | Phase logs; writes `logs/liquid_<bench>_proxycheck.csv` and `logs/liquid_<bench>_congestion.md` |
