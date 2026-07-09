# Batch ↔ Gillespie mode-switching logic

This documents the logic that decides, at each step of the CRN simulator, whether to advance with
the **batching** engine or with **Gillespie** (rebop), why the original rule mis-fired on stiff
oscillators like the Oregonator, the wall-clock-aware rule that replaced it, and the open question
about a cheaper predictive rule. All code is in [`src/simulator_crn.rs`](src/simulator_crn.rs);
the switching state lives in the `SwitchState` struct.

## Background: two engines, one loop

`BatchSimulator::run` alternates between:

- **batch mode** (`batch_step`): the Berenbrink-style batching algorithm (see arXiv:2508.04079).
  One "batch" advances ~√n reactions at once via collision sampling. Each batch pays a fixed cost
  dominated by **f128 high-precision `ln_gamma` collision sampling** (`sample_collision_fast_f128`)
  plus hypergeometric/multinomial sampling and, when the K-count drifts, a `construct_transition_arrays`
  rebuild. Its advantage is asymptotic: cost per unit simulated time falls like ~1/√n, so it wins at
  large n **when a healthy fraction of reactions are active**.
- **Gillespie mode** (`gillespie_steps`): delegates to rebop's exact SSA. Runs ~√n reactions per
  call, cheap per reaction, no per-batch machinery. Entering it rebuilds the rebop object once
  (`initialize_gillespie_config`); leaving it syncs the urn (`finalize_gillespie`).

The two engines are both exact; switching only changes speed, never the distribution.

## The original rule (still available as `HEURISTIC_PROXY`)

Switch to Gillespie when the **expected number of active reactions in the next batch** is below
the reaction count:

```
rough = active_probability * sqrt(n_including_extra_species)   // expected active rxns
use_gillespie = rough < num_reactions                              // num_reactions = proxy_threshold
```

Intuition: if a whole batch would produce fewer real reactions than there are reactions in the CRN,
the batch machinery isn't earning its keep, so fall back to Gillespie.

## What went wrong (the Oregonator)

The 2026-07-03 fix (`CHANGES_2026-07-03.md`) repaired a hang where the simulator got stuck in
Gillespie mode with a frozen urn. Fixing that made mode-switching actually work — and revealed that
the proxy sends the **Oregonator** into batch mode during its spike phases, where batch mode is
**~30–90× slower than rebop** at n ≤ 1e6. Measured: at n=1e5, t=2, batch mode took 4.55s of
wall-clock vs 0.10s for the Gillespie portions (which run at rebop speed). The proxy uses
*expected reaction count* as a stand-in for batch's cost, but that cost is dominated by the f128
collision sampling, which the count ignores.

(Aside: before the 2026-07-03 fix the Oregonator was accidentally *fast*, because the urn-sync bug
pinned it in Gillespie mode — real speed, but with corrupted intermediate config/history. The
wall-clock rule below makes that fast Gillespie choice happen correctly and deliberately.)

## The wall-clock rule (`HEURISTIC_WALLCLOCK`, default)

Decide by **measured wall-clock per unit continuous time** (call it `w/dt`) and pick the cheaper
engine, using the proxy only as the bootstrap default and as a tie-breaker:

1. Time every `batch_step` / `gillespie_steps` call and fold `(wall, dt)` into per-mode EMAs.
2. A mode's cost estimate is `wall_ema / dt_ema`.
3. **Default to the proxy.** Once both modes have been measured, **override** to the other mode only
   if it is measured ≥ `WDT_OVERRIDE_FACTOR` (=4×) cheaper.
4. Measure the other mode with brief **probes**: frequently while bootstrapping
   (`WDT_PROBE_INTERVAL` = 256 iters), then rarely once both are known
   (`WDT_PROBE_INTERVAL_COMMITTED` = 8192) — just enough to catch a slow regime change.

### The critical correctness bug: dt-weighting

The first implementation averaged `elapsed/dt` **per call**. That is wrong. Gillespie's `dt` per
call is `√n / propensity`, which swings by **orders of magnitude** as the propensity changes, so a
per-call average mis-weights tiny-dt calls. It measured Gillespie as ~4× *cheaper* than batch for
Dimerization (firing ~1900 spurious "switch to Gillespie" overrides) when the truth is ~2×
*more expensive*. **Fix:** keep separate `wall_ema` and `dt_ema` and use the ratio `Σwall/Σdt`
(a dt-weighted average) — the objective-aligned figure of merit, since the goal is to minimize
wall-clock to reach `t_max`. This one change was the difference between right and wrong.

### Why probing is scheduled carefully

A Gillespie call is **coarse-grained** (large dt per call at low propensity), so a Gillespie probe
"commits" a chunk of continuous time to the costlier engine. If Dimerization re-probed Gillespie
every 256 iters forever, those probes alone slowed it ~15–30%. Hence: probe often only until both
modes are measured, then back off to a rare interval.

## Results

Min over 3 trials. Oregonator via `oregonator_spec()` (working rates, on-cycle IC);
Dimerization vs the cached pre-change pure-batch data.

| CRN | n | batss (wall-clock) | reference | ratio |
|---|---|---|---|---|
| Oregonator (t=2) | 1e4 | 0.025s | rebop 0.016s | 1.6× |
| | 1e5 | 0.203s | rebop 0.156s | 1.3× |
| | 1e6 | 1.975s | rebop 1.566s | 1.3× |
| Dimerization (t=0.5) | 1e7 | 0.062s | cached batch 0.065s | 0.96× |
| | 1e8 | 0.199s | cached batch 0.210s | 0.95× |
| | 1e9 | 1.065s | cached batch 1.188s | 0.90× |

Oregonator: **~30–90× → 1.3–1.6×**. Dimerization: **no regression** (a hair faster; it even exploits
its own within-run crossover — Gillespie during the brief high-propensity start, batch at
equilibrium). All 8 tests in `tests/batss_tests.py` pass; the Oregonator trajectory still oscillates.

The batch↔Gillespie crossover for Dimerization is **n-dependent** (~1e7–1e8), the expected batss
behavior: pure-batch vs pure-Gillespie(rebop) is 0.065 vs 0.039s at n=1e7 (Gillespie wins) but
0.21 vs 0.37s at n=1e8 (batch wins). The wall-clock rule tracks this automatically.

## The open question: a cheaper predictive rule

Wall-clock measurement is **bug-prone** (the dt-weighting bug above) and adds per-call overhead. The
goal we actually want is to **predict, before a batch, which engine will be faster** — i.e. fix the
proxy rather than measure. `HEURISTIC_PROXY` with a settable `proxy_threshold` exists to explore this.

Empirical evidence that a tuned proxy can work — *but not with a single constant*:

Oregonator n=1e5, t=2, `HEURISTIC_PROXY` at various `proxy_threshold` (default = num_reactions = 5):

| proxy_threshold | time | gillespie_frac |
|---|---|---|
| 5 (original) | 7.7s | 0.30 |
| 50 | 2.3s | 0.88 |
| **500** | **0.38s** | **1.00** |
| 5000 | 0.38s | 1.00 |
| (wall-clock rule) | 0.44s | 1.00 |

So threshold ≈ 500 (~100× the reaction count) makes the *proxy alone* beat the wall-clock rule for
the Oregonator — consistent with "batch's fixed per-batch overhead is worth ~100 reactions, not
~num_reactions." **But** the same high threshold breaks Dimerization:

| Dimerization n=1e8 | proxy_threshold=2 | proxy_threshold=2000 |
|---|---|---|
| time | 0.31s (batch, correct) | 0.59s (Gillespie, 2.8× too slow) |

No single fixed `proxy_threshold` separates "Oregonator wants Gillespie" from "Dimerization wants
batch" — because `rough = active_probability·√n` alone doesn't encode batch's per-batch cost,
which depends on n and CRN structure. The likely next step is a **predictive formula** for the
threshold (or for batch's per-batch overhead in units of reactions) from measurable quantities
(n, reaction order/generativity, the f128 collision-sampling cost model), so the proxy predicts the
crossover without timing anything. The wall-clock rule is the robust fallback that sidesteps needing
that formula.

## Where things live / how to test

- **Code:** `src/simulator_crn.rs`. `SwitchState` holds all switching state (config + EMAs + probe
  schedule + observability). The decision is in `run()`. Tuning constants near the `WDT_*`
  definitions: `WDT_EMA_ALPHA` (0.3), `WDT_OVERRIDE_FACTOR` (4.0), `WDT_PROBE_INTERVAL` (256),
  `WDT_PROBE_INTERVAL_COMMITTED` (8192).
- **Python control:** `sim.simulator.heuristic = HEURISTIC_PROXY | HEURISTIC_WALLCLOCK`;
  `sim.simulator.proxy_threshold = <float>`. Constants in `examples/dimerization_benchmarks.py`.
- **Observability:** `sim.simulator.switch` is a read-only snapshot with per-mode
  `*_wallclock_seconds`, `*_continuous_time`, `*_calls`, plus `mode_switches` and
  `switch_overhead_seconds`. `mode_split(sim)` in `examples/dimerization_benchmarks.py` bundles them.
- **Specs:** `oregonator_spec()` and `dimerization_spec()` in `examples/dimerization_benchmarks.py`.
- **Build after editing Rust:** `CARGO_TARGET_DIR="$LOCALAPPDATA/batss-cargo-target" maturin develop --release`
  (the `CARGO_TARGET_DIR` outside Dropbox avoids a Windows file-lock; and close any Jupyter kernel
  that has `import batss` loaded, or the `.pyd` can't be overwritten).
- **Tests:** `python -m unittest tests.batss_tests` (includes `test_gillespie_switch_reaches_target_time`).
