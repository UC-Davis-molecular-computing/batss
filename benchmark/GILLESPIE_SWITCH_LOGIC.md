# Batch ↔ Gillespie mode-switching logic

This documents the logic that decides, at each step of the CRN simulator, whether to advance with
the **batching** engine or with **Gillespie** (rebop), why the original rule mis-fired on stiff
oscillators like the Oregonator, the wall-clock-aware rule that replaced it, and the open question
about a cheaper predictive rule. All code is in [`src/simulator_crn.rs`](../src/simulator_crn.rs);
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

## Experimental deterministic prospective rule (`HEURISTIC_PROSPECTIVE`)

Wall-clock measurement is **bug-prone** (the dt-weighting bug above), adds overhead, and makes the
mode sequence depend on machine timing. The experimental selector `HEURISTIC_PROSPECTIVE` (= 2)
instead estimates how much useful work the next batch would accomplish:

> [!IMPORTANT]
> **Definition of the threshold `T`:** `T` is the cutoff applied to the prospective batch score,
> which estimates the number of real (non-passive) reactions that one collision-free batch would
> accomplish. The simulator uses Gillespie when `score < T` and batching when `score >= T`.
> Therefore, increasing `T` favors Gillespie and decreasing `T` favors batching. In the API,
> `T` is stored in the historically named `sim.simulator.proxy_threshold` field. A policy may use
> one fixed `T`, as in the experiment below, or calculate `T` from CRN/state features.

```text
k = k_reset_target()
N = n + k
p_real = calculate_total_propensity(false)
p_total = max_adjusted_rate_constant_at(k) * C(N, o)
E[L] = sqrt(pi / (2 * o * (o + g))) * sqrt(N)
score = (p_real / p_total) * E[L]
use Gillespie when score < T
```

Evaluating at `k_reset_target()` is deliberate. K is frozen during Gillespie, but leaving Gillespie
resets it before the next batch; the current K can therefore describe a batch that would never run.
The adjusted-rate helper is pure and allocation-free and does not rebuild transition arrays. For
order-2 benchmark CRNs the K target is exact; for order >= 3 it inherits `k_reset_target`'s documented
approximation (see `K0_SELECTION.md`).

The default remains `HEURISTIC_WALLCLOCK`. Selector 2 is an opt-in experiment, uses the existing
`proxy_threshold`, and has no hysteresis yet. Every simulation still executes one initial batch
before making its first mode decision.

### First bounded results (2026-07-11)

Anchor timings below are the median of two fresh seed-1 runs. Threshold 0 is the batch anchor and
`1e12` is Gillespie after the mandatory first batch. Candidate thresholds 200 and 300 were then
repeated with seeds 1, 2, and 3; their mode fractions were stable across seeds.

| CRN | n | threshold | median time | Gillespie continuous-time fraction |
|---|---:|---:|---:|---:|
| Dimerization (t=0.5) | 1e8 | 0 | 0.135s | 0.000000 |
| | | 200 | 0.136s | 0.000000 |
| | | 300 | 0.131s | 0.000000 |
| | | 500 | 0.138s | 0.000000 |
| | | 1e12 | 0.700s | 0.999971 |
| Oregonator (t=1) | 1e5 | 0 | 3.602s | 0.000000 |
| | | 200 | 0.263s | 0.999999 |
| | | 300 | 0.122s | 0.999999 |
| | | 500 | 0.229s | 0.999999 |
| | | 1e12 | 0.238s | 0.999999 |
| Rössler-Willamowski (t=1) | 1e5 | 0 | 2.294s | 0.000000 |
| | | 200 | 2.501s | 0.066754 |
| | | 300 | 1.893s | 0.156681 |
| | | 500 | 2.591s | 0.563908 |
| | | 1e12 | 4.452s | 0.999977 |

The initial prospective scores were 1392.6 for Dimerization, 1.18 for Oregonator, 79.35 for
Rössler, and 4522.5 for Lotka-Volterra at n=1e8. The focused multi-seed results were:

- Dimerization n=1e8: thresholds 200 and 300 stayed entirely in batch mode for all three seeds.
- Oregonator n=1e5: both spent more than 0.999994 of continuous time in Gillespie for all seeds.
- Rössler n=1e5: threshold 200 used 0.0662-0.0668 Gillespie; threshold 300 used 0.1541-0.1583.
- Lotka-Volterra: both were effectively all Gillespie at n=1e5 and all batch at n=1e6, 1e7,
  and 1e8 for all three seeds, following the population-dependent engine crossover.

Threshold 50 is too low globally: it keeps Lotka-Volterra n=1e5 in batch mode, about three times
slower than its Gillespie anchor here. Threshold 500 catches the small Lotka n=1e6 Gillespie win,
but sends Rössler through Gillespie for about 56% of its time. On this matrix, **200 is the safer
fixed candidate and 300 is a plausible nearby alternative**. This is evidence for a useful constant,
not yet a production-default decision.

Across the quick matrix and all focused sweeps, all **105/105** fresh-repeat pairs matched their
trajectory, visible-final, hidden-full-state, and aggregate mode-signature hashes; there were no
errors or timeouts. Two independent Python processes also matched eight-checkpoint trajectory,
final, full-state, and mode-signature hashes. Unit tests cover a batch -> Gillespie -> batch replay,
a stale-K score evaluation, and deterministic reaction registration.

Timing still has only two repeats per seed and the benchmark replay hash records configured
checkpoints rather than every decision. Before changing the default or adding hysteresis, run the
full matrix with more timing repeats and evaluate regret against both forced-mode anchors. If that
shows a structural outlier, benchmark both engines from the same frozen states and use those paired
costs as the fitting oracle; the wall-clock policy is an adaptive baseline, not an oracle.

Run the bounded matrix with:

```powershell
python -B benchmark/threshold_sweep.py --preset quick --heuristic prospective --overwrite
```

The earlier claim that no constant could separate Oregonator and Dimerization was not established:
it compared Oregonator at threshold 500 with Dimerization at threshold 2000. The matched-threshold
experiment above is the missing comparison.

## CRN-structured threshold experiment (2026-07-11)

The fixed sweep above was a preliminary baseline. The structural experiment tested this proposed
CRN-dependent steady-throughput threshold:

```text
T = theta0 + theta1 log2(N) + theta2 q^o + theta3 R + theta4 B
    + theta5 o log2(N) [g > 0]
```

> [!CAUTION]
> This six-term expression is a **mechanistic hypothesis that was tested and rejected as a fitted
> recommendation**, not the threshold currently used by the simulator. Its structural coefficients
> did not generalize reliably across CRNs. The definitions and rationale below explain why the terms
> were considered; they do not imply that the terms earned inclusion in a production model.

The experimental Rust method `benchmark_engine_call` starts both engines from the same canonical
frozen state. The fitted target is the break-even prospective score

```text
T* = C_batch / C_Gillespie-per-real-reaction.
```

Thus a larger batch cost raises `T*`, while a larger Gillespie cost lowers it. Ordinary batch
postprocessing and amortized Gillespie-to-urn synchronization are included. K preparation and
Gillespie-object construction are excluded because they are switching costs; they belong in a later
hysteresis/amortization model. The benchmark amortizes synchronization over its 5,000-event
Gillespie block; a production predictor must instead use its expected Gillespie residence horizon.

### What `q^o`, `R`, and `B` count

After uniformization, `q` is the number of species including the filler species K and W, `o` is the
common number of reactants per reaction, and `N = n + k_reset_target()` is the population that a
prospective batch would use.

The clearest way to define `R` is to imagine putting the simulator's internal directed reaction
channels into buckets. Reversible reactions have first been expanded into two directed channels,
and every channel has been uniformized/padded. A bucket label is that channel's left-hand-side
reactant multiset:

```text
R = number of nonempty buckets of internal channels
    having the same padded input multiset.
```

Thus `R` **is a count of reactant multisets**, but only the input multisets actually present in the original untransformed
CRN, not every multiset that could be formed from its species. "Reactant group" below means exactly
one such bucket. It is not another combinatorial object, and `R` is a static CRN property: it is not
recomputed when species counts change.

> [!NOTE]
> Conceptually the bucket label is a multiset. Rust currently hashes the exact reactant vector and
> then fills its permutations, so this interpretation assumes reactants were written in a consistent
> canonical order. Canonicalizing before grouping would make mixed `A+B` and `B+A` spellings safer.

A small example makes both `R` and `B` explicit:

| internal directed channel | input bucket | products |
|---|---|---|
| `A + B -> A + C` | `{A,B}` | `A + C` |
| `A + B -> B + D` | `{A,B}` | `B + D` |
| `2A -> A + B` | `{A,A}` | `A + B` |

There are two nonempty input buckets, `{A,B}` and `{A,A}`, so `R = 2`. Let `b_r` be the
number of channels in bucket `r`. Here `b_{AB}=2` and `b_{AA}=1`, so

```text
B = sum_r b_r = 3.
```

Duplicate channels count separately. The two `A+B` channels share one input bucket but remain two
product/rate alternatives. All three displayed channels already have order two on both sides, so
the example is also a valid uniformized internal channel list.

The related quantities are:

- `C(q + o - 1, o)` is the mathematical count of unordered size-`o` multisets if all `q`
  species were permitted as inputs. It equals `C(q + o - 1, q - 1)`, not in general
  `C(q + o - 1, q)`. Batss forbids waste species `W` as a reactant, so with the usual one `W`
  species the implementation-valid maximum is instead `C(q + o - 2, o)`. Static `R` reaches
  that maximum only if the CRN explicitly declares every permitted input bucket.
- If only `p_react` reactant-capable species have positive count,
  `C(p_react + o - 1, o)` is only an upper bound on multiplicity-feasible inputs using those
  species. Define the actual state-dependent count as

  ```text
  R_feasible(x) = number of the R declared input buckets
                  whose required multiplicities are available in x.
  ```

  Hence `R_feasible(x) <= min(R, C(p_react + o - 1, o))`. In the example, `A=10,B=10`
  makes both buckets feasible, while `A=1,B=0` makes neither feasible; static `R` remains two
  in both states. A feasible bucket can still contain only zero-rate channels, and the CRN may
  omit most mathematically possible inputs.
- `q^o` is the number of **ordered input tuples**. The sorted urn does save some work inside
  one hypergeometric draw: it stops once the requested sample has been allocated among the
  largest-count species. The current recursive sampler nevertheless ignores the returned
  last-nonzero prefix, calls `sample_vector` at every dense tree node, and scans all `q^o`
  terminal lanes. Therefore `q^o` is a real implementation cost today even when most species
  are absent or tiny. A future sorted-prefix implementation would replace much of this cost by
  state-dependent support/occupancy work.
- `B` is the total number of internal directed channels. When an occupied batch lane belongs
  to bucket `r`, batching copies its `b_r` channel probabilities, may append one passive-remainder
  category, samples the alternatives, and applies their products. Transition-array construction
  also visits every channel. In the current dense rebop engine, every Gillespie event evaluates
  all `B` channel propensities and scans all `B` cumulative rates. Thus `B` can slow **both**
  engines, and where its channels are placed among occupied buckets matters to batching. Raw `B`
  has no predetermined signed additive effect on `T* = C_batch / C_Gillespie`.

### Why each term was proposed

| term | intended connection to `T* = C_batch / C_Gillespie` | important limitation |
|---|---|---|
| `theta0` | Fixed per-engine-call overhead. | It mixes unrelated constant costs into one coefficient. |
| `theta1 log2(N)` | Collision-length inversion uses a search whose iteration count grows roughly with the number of bits in `N`. | Both engines also have smaller population-dependent effects. |
| `theta2 q^o` | Dense multidimensional sampling and the full transition-lane scan are batch costs paid even for passive lanes. | Only one measured CRN is order 3/high-`q^o`, so the coefficient cannot extrapolate safely. |
| `theta3 R` | Declared input buckets can add transition-construction, adjusted-rate, and occupied-lane work beyond the passive baseline. | The frozen-state target excludes most construction cost; static `R` does not say which lanes are occupied, and Gillespie's cost is governed more directly by total channel count `B`. |
| `theta4 B` | Multiple channels add multinomial/output-update work in batching. | Gillespie also pays for every channel; `B` can increase both numerator and denominator. |
| `theta5 o log2(N) [g > 0]` | Generative collision searches perform repeated order-dependent gamma/multifactorial work. | This is coarse: it ignores the magnitude of `g` and does not isolate every collision-sampler operation. |

The terms are therefore not simple additive laws of runtime. In particular, `q^o` represents the
dense baseline while `R` and `B` were intended as incremental active-lane work, so they are not
literal duplicates. But `R` and `B` are almost collinear in the available data (correlation about
0.98), which makes a direct unconstrained linear fit for their difference especially unstable.
`B` directly affects both engines; `R` directly affects batching construction and is correlated
with channel structure in these data, but rebop receives the `B` individual channels rather than
the batch input-bucket grouping.

### What the requested dynamic CRNs contribute

Nine paired timing repeats were collected at eight frozen states per CRN. In the three-CRN
`quick` experiment:

- Oregonator spanned prospective scores 0.41--36.3, but Gillespie was optimal at all eight
  sampled states on this machine.
- Rössler-Willamowski contained both regimes (four batch, four Gillespie).
- Shrinking crossed once (one batch state followed by seven Gillespie states).

These are useful switching trajectories, but the design matrix is only rank 5/6. Repeating many
states within one CRN varies `N` and composition; it does not create new information about
`q^o`, `R`, or `B`, which are constant for that CRN. An unconstrained leave-one-CRN-out
pseudoinverse is therefore not an identified six-parameter model (its diagnostic mean/worst
regret was 83x/1062x).

Adding Dimerization, Lotka-Volterra, and the order-3 Brusselator makes the 48-row `full` matrix
rank 6/6, but it remains structurally near-saturated: raw condition number 903 (standardized
13.6), Oregonator is the only `B != R` CRN, Brusselator is the only order-3/high-`q^o` CRN, and
Rössler is the only high-`R` CRN. Holding out Oregonator drops the training matrix back to rank
5/6. The all-data OLS classified its own states perfectly, but the rank-deficient/near-saturated
leave-one-CRN-out pseudoinverse had diagnostic mean/worst regret 179x/2096x.

A second complete run gave similar empirical threshold ranges and the same regime pattern, but
very different static coefficients. For example, `(theta3, theta4)` moved from approximately
`(-194, 122)` to `(-64, 8)`. That instability is expected when `R` and `B` are almost perfectly
correlated; these numbers must not be interpreted as a learned portable cost model.

### Matched Shrinking topology experiment (2026-07-12)

Removing `B -> None` from Shrinking gives post-uniform features
`(q^o, R, B, o[g>0]) = (16, 2, 2, 0)`. At initial `n=1e8`, its prospective score
fell from 2903 to about 0.20 by `t=0.1`, while real population retained about 40.4% of its
initial count. A dense timing bracket put the local crossover between `t=0.014` (score 954,
`T*=872`, batch) and `t=0.015` (score 874, `T*=898`, Gillespie), when more than 53% of
the initial population remained. This isolates falling active fraction much more cleanly than
a trajectory whose total population itself spans many decades.

Adding this fourth topology makes the all-data matrix rank 6/6, but every leave-one-CRN-out
training fold still has only three structures and rank 5/6. A fifth, dynamics-preserving topology
fixes that formal problem: replace original Shrinking's `B -> None` rate-100 reaction with two
identical rate-50 channels. Duplicate channels are retained as separate channels, so the CTMC is
unchanged while the features become `(16, 3, 4, 0)` instead of `(16, 3, 3, 0)`.

With Oregonator, Rössler, original Shrinking, no-B-decay Shrinking, and split-B-decay
Shrinking, every grouped training fold was rank 6/6. Two scale-matched collections (eight states
and nine paired timing repeats per family) gave:

| predictor | seed 1 mean / worst regret | seed 2 mean / worst regret |
|---|---:|---:|
| six-feature structural OLS | 1.061 / 1.984 | 1.042 / 1.839 |
| training-selected constant | 1.012 / 1.234 | 1.025 / 1.624 |
| `1 + log2(N)` OLS | 1.000 / 1.000 | 1.002 / 1.085 |

The variants therefore solve rank, but not predictive value: the smaller population model still
wins, and the structural coefficients remain unstable. In the integrated seed-1 and seed-2 runs,
`(theta_R, theta_B)` moved from about `(-19, 21)` to `(-17, 80)`; another repeated seed-1
collection produced about `(-114, 126)`. The standardized all-data condition number was about
34, with the Rössler-held-out fold above 340. More matched topology contrasts and longer timing blocks are
needed before interpreting the structural coefficients, even though the new `matched` preset now
provides a formally identifiable grouped smoke test.

### Baseline comparison and revised recommendation

On the first full dataset, fixed threshold 500 classified 47/48 sampled states with geometric
mean regret 1.004 and worst regret 1.22; it classified all 48 states in the independent seed-2
run. A constant selected in-sample was 511 for seed 1 and 441 for seed 2 and separated every
sampled state in each run. When that constant was selected inside each leave-one-CRN-out training
fold, mean/worst regret was 1.012/1.22 for seed 1 and 1.021/1.62 for seed 2.

The smaller population-only model `T = beta0 + beta1 log2(N)` was identifiable in every grouped
fold and performed best: grouped mean/worst regret 1.000/1.000 for seed 1 and 1.002/1.085 for
seed 2. Its all-data coefficients were also much more stable:

```text
seed 1: T = 460.2 + 8.35 log2(N)
seed 2: T = 506.0 + 7.56 log2(N)
```

This makes the affine population model

```text
T_logN(N) = beta0 + beta1 log2(N)
```

a useful **parsimonious baseline**, not the only structural model worth investigating. It won the
small grouped regression because that dataset did not independently vary enough implementation
costs; the result does not establish that all CRNs with the same `N` have the same break-even cost.

Also, `T` is only the right-hand cost threshold. The full rule compares

```text
prospective score S(x) = p_active(x) E[L]
against
cost threshold T(x).
```

Rates and composition already enter `S(x)` through `p_active`. Therefore two CRNs at the same `N`
can make different decisions even if they use exactly the same `T(N)`.

### Batch-cost profiling (2026-07-12)

The Flame profile was changed before collecting these data: per-lane spans were removed, only coarse
phase boundaries remain, reports retain microsecond precision and call counts, and the recorder can
be cleared after simulator construction. Component attribution used two independent passes of 1000
fresh frozen-state batches, one forward and one reverse through the case matrix. True engine costs
used ordinary non-Flame builds, two seeds, 301 warmed and globally interleaved repeats per case, and
5000 Gillespie reactions per timing block. Values below are medians across the two passes; raw
per-seed values and ranges are in the CSV files.

#### Collision-length sampling depends on `N`, `o`, and the magnitude of `g`

| `o` | `g` | collision-length cost at real `n=1e4` | at real `n=1e8` |
|---:|---:|---:|---:|
| 2 | 0 | 2.16 us | 3.70 us |
| 2 | 1 | 5.23 us | 7.01 us |
| 2 | 2 | 5.44 us | 7.76 us |
| 3 | 0 | 2.44 us | 4.02 us |
| 3 | 1 | 6.54 us | 8.31 us |
| 3 | 2 | 6.67 us | 10.10 us |

The sampler is usually the largest phase for the order-2 CRNs, but it is not a universal dominant
cost. Its code performs one high-precision gamma calculation unconditionally; `g>0` adds `o`
static high-precision gamma terms and `o` more gamma/multifactorial terms during each search
iteration. The observed `g=1` versus `g=2` difference shows that the former indicator-only
`o log2(N) [g>0]` term is too coarse.

#### Sorting does not currently remove the dense `q^o` cost

A controlled family held real `n=2e6`, prospective `N=6e6`, `o=2`, `g=0`, and the prospective
score fixed. Only `A` and `K` had positive counts; extra species were zero-count spectators. Thus
this is the situation where sorted sampling should help most.

| `q` | `q^o` | recursive multidimensional sample | dense terminal-lane processing | profiled batch core |
|---:|---:|---:|---:|---:|
| 4 | 16 | 0.96 us | 1.06 us | 5.52 us |
| 6 | 36 | 1.96 us | 3.00 us | 9.41 us |
| 8 | 64 | 2.36 us | 4.10 us | 10.49 us |
| 12 | 144 | 3.10 us | 6.62 us | 13.35 us |

The spectator channels were zero-rate and their input lanes received no sampled mass, so they did
not execute channel-alternative work. The increase comes from dense recursion, clearing, and
terminal scanning.
The current implementation calls `sample_vector` at `1 + q + ... + q^(o-1)` nodes, clears `q`
counters at each node, and then processes all `q^o` terminal lanes. If the sorted-prefix TODO is
implemented, this term should be re-profiled and replaced by support/occupancy features.

#### Static `R` does not describe state-dependent benefit

In no-B-decay Shrinking, `N,q,o,g,R,B` were fixed while the initial A fraction changed:

| A fraction | `R` | `B` | prospective score | active reactions per sampled batch | dense processing |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 2 | 2 | 410.56 | 388 | 1.33 us |
| 0.10 | 2 | 2 | 68.26 | 62 | 1.33 us |
| 0.01 | 2 | 2 | 6.82 | 6 | 1.40 us |

The score correctly changes by about 60x while static `R` and the dense processing cost remain
essentially unchanged. That is expected: current batching pays the full dense scan regardless of
how many declared inputs are useful. `R` may help construct a future occupied-lane feature, but it
is not itself the missing state variable in `T`.

#### `B` matters, but in both numerator and denominator

The controlled channel family split the same rate-100 `B -> None` reaction into 1, 2, 4, or 8
identical channels whose rates still sum to 100. The CTMC generator, `N,q,o,g,R`, composition,
and prospective score were unchanged.

| channels in split B decay | total `B` | batch cost | Gillespie cost per reaction | measured `T*` |
|---:|---:|---:|---:|---:|
| 1 | 3 | 10.20 us | 14.31 ns | 713 |
| 2 | 4 | 10.75 us | 16.76 ns | 641 |
| 4 | 6 | 11.50 us | 20.42 ns | 563 |
| 8 | 10 | 13.30 us | 28.63 ns | 465 |

Increasing `B` from 3 to 10 raised batch cost by about 30%, but Gillespie cost by about 100%, so
the ratio `T*` fell by about 35%. This falsifies a raw positive `theta_B B` correction to `T`.
The batch placement of channels and the Gillespie representation both matter.

#### Real-CRN phase check

| CRN | `(q,o,g)` | `q^o` | collision-length | recursive sample | dense processing | profiled batch core |
|---|---:|---:|---:|---:|---:|---:|
| Oregonator | (5,2,2) | 25 | 5.59 us | 1.59 us | 2.16 us | 10.14 us |
| Rössler-Willamowski | (5,2,1) | 25 | 5.14 us | 2.03 us | 3.56 us | 11.58 us |
| Dimerization | (4,2,1) | 16 | 4.99 us | 0.93 us | 1.24 us | 8.03 us |
| Lotka-Volterra | (4,2,1) | 16 | 5.41 us | 1.33 us | 1.71 us | 9.46 us |
| Order-3 Brusselator | (6,3,1) | 216 | 6.08 us | 4.14 us | 11.32 us | 22.35 us |

Collision sampling is about 55% of the Oregonator batch core and 62% of Dimerization, but only 27%
of the Brusselator core; the Brusselator's dense lane processing is about 51%. An affine function
of `log2(N)` alone cannot represent these implementation costs outside the narrow regression sample.

### Revised research hypothesis: model the two engine costs separately

Retain `T_logN` and a selected constant as baselines. The next structural model to test is

```text
use batch when S(x) > T_hat(x)

T_hat(x) = C_batch_hat(x) / C_G_hat_per_real_reaction(x)

C_batch_hat(x) =
    C_collision_hat(N, o, g)
  + a_sample sum_{d=1}^o q^d
  + a_scan q^o
  + a_branch W_lanes(x, E[L], {b_r})
  + C_finish_hat(q, I_batch_sort)

C_G_hat_per_real_reaction(x) =
    c0
  + c_rate B (q - 2)
  + c_order sum_over_channels(real reactant order)
  + c_select B
  + c_jump (q - 2)
  + C_sync_hat(q, I_sync_sort) / H
```

The recursive-sampling term is written as `q + ... + q^o` because the implementation does roughly
`q` counter/draw work at each of `1 + q + ... + q^(o-1)` dense tree nodes; the separate `q^o`
term represents scanning the terminal lanes. They describe two distinct costs even though both
contain a highest-order `q^o` contribution.

`C_collision_hat` should allow both `o` and the magnitude of `g` to interact with `log2(N)`,
or use a small profiled lookup over the few supported orders/generativities. `W_lanes` should
summarize expected occupied ordered lanes and weight an occupied bucket by its `b_r` alternatives;
for example, with prospective lane probability `pi_l` and expected length `E[L]`, its occupancy is
approximately `1 - exp(-E[L] pi_l)`. Raw `R` is not a substitute for that quantity.

`C_finish_hat` covers applying the sampled net change and restoring urn order. Adding the update
vector is `O(q)`, but the current insertion sort is only near-linear when counts move little; more
generally it is `O(q + I_sort)`, where `I_sort` is the number of inversions created by the update.
A raw `a_commit q` term would silently assume the urn always remains nearly sorted.

The Gillespie form reflects the implementation actually in use. Batss constructs rebop in dense
mode. On every event rebop evaluates all `B` rates across the `q-2` real species, builds and scans
all `B` cumulative rates, and applies one dense jump. `C_sync/H` represents copying the Gillespie
state back into the urn and re-sorting it once per expected `H`-reaction Gillespie residence or
probe block. There is no dependency-update optimization to model in this version. A future sparse
rebop configuration would require a different cost model. Including `C_finish` and `C_sync/H`
aligns the displayed predictors with the postprocessing included in the measured `T*`; setup and K
rebuild remain separate switching costs for the hysteresis/amortization layer.

Fit the two costs with nonnegative coefficients or a positive link, validate them by held-out CRN
families, and take their ratio only afterward. This prevents one unstable signed `R` or `B`
coefficient from pretending to be the difference of two real costs. The earlier permutation counts

```text
P = sum_r o! / product_s(m_rs!)
E = sum_r (o! / product_s(m_rs!)) (b_r - 1)
```

did not improve the old grouped regression consistently because they still omitted state occupancy
and the Gillespie denominator.

The three matched Shrinking variants illustrate why both sides of the switch rule matter at the
same prospective `N=6e6`:

| topology | `R` | `B` | score `S` | local `T*` |
|---|---:|---:|---:|---:|
| original | 3 | 3 | 683 | 694 |
| no B decay | 2 | 2 | 411 | 769 |
| split B decay | 3 | 4 | 683 | 640 |

Most of the no-B separation is already on the left: removing `B -> None` turns the `B+K` draws
passive, so the score falls even though `N` is unchanged. Original versus split-B is the cleaner
threshold-cost contrast because their scores and generators agree while `B` differs.

The frozen-state result around 500 and the earlier end-to-end preference for 200--300 measure
different objectives: the oracle estimates local steady crossover with setup excluded, whereas a
whole trajectory includes mode duration, setup/rebuild costs, truncation, and switching. Every
candidate still needs forced-policy/end-to-end regret guardrails before becoming a default.

Further validation should add several independent order-3 and order-4 holdouts, cross `q` with
support and composition rather than adding only zero spectators, count collision-search and occupied
lane operations in a separate low-overhead pass, and repeat on another CPU. If dense prefix pruning
or sparse Gillespie is implemented, the cost profile is version-specific and must be regenerated.

Run the structural regression with:

```powershell
python benchmark/threshold_model.py run --preset quick
python benchmark/threshold_model.py run --preset matched
python benchmark/threshold_model.py run --preset full
```

Run the profiler with:

```powershell
$env:CARGO_TARGET_DIR = Join-Path $env:LOCALAPPDATA 'batss-cargo-target'
maturin develop --release --features flm
python benchmark/profile_batch_costs.py components --repeats 1000 --seed 1

maturin develop --release
python benchmark/profile_batch_costs.py timings --repeats 301 --seed 1 --gillespie-reactions 5000
```

The saved profiling outputs are `batch_cost_profile_components*.csv` and
`batch_cost_profile_timings*.csv` in this directory; the `*_summary.csv` files combine the two
seed/order passes used in the tables above.
The CSVs retain two historical column names: `reactant_sets_R` is the declared-bucket count `R`,
and `output_branches_B` is the total directed reaction-channel count `B` defined above.

## End-to-end held-out threshold comparison (2026-07-12)

The frozen-state experiment above measures local engine crossovers. This end-to-end experiment
instead times one uninterrupted simulator run through every regime reached by each CRN. The pilot
swept thresholds `0, 20, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 1e12`
on seeds 1--3 with two fresh timing repeats per seed. Thresholds 0 and `1e12` were forced-engine
anchors, and the wall-clock policy was recorded but excluded from threshold selection. The
equal-family, tail-regret objective produced a 5% plateau at 400--500 and selected `T = 500`.

The held-out comparison used unseen seeds 101--106 and two repeats per seed. Its fixed controls
were two lower thresholds (`T_l = 100, 250`), `T = 500`, two higher thresholds
(`T_h = 1000, 2000`), and the timing-based policy. Each cell below is
`median seconds (runtime / timing-policy runtime)`, so timing is always `1.00x`. The median is
taken first across repeats for each seed and then across seeds. Construction is excluded; the
timer covers one raw
`BatchSimulator.run` call to the scenario end time.

| CRN (initial n) | timing | `T=500` | `T_l=100` | `T_l=250` | `T_h=1000` | `T_h=2000` |
|---|---:|---:|---:|---:|---:|---:|
| Oregonator (100000) | 0.1703 (1.00x) | 0.1753 (1.03x) | 0.1793 (1.05x) | 0.1664 (0.98x) | 0.1621 (0.95x) | 0.1669 (0.98x) |
| Rössler-Willamowski (100000) | 2.403 (1.00x) | 2.597 (1.08x) | 2.558 (1.06x) | 2.397 (1.00x) | 3.761 (1.56x) | 4.099 (1.71x) |
| Shrinking (2000000) | 0.04562 (1.00x) | 0.04167 (0.91x) | 0.04165 (0.91x) | 0.03999 (0.88x) | 0.05510 (1.21x) | 0.05499 (1.21x) |
| Shrinking no B decay (2000000) | 0.05439 (1.00x) | 0.02643 (0.49x) | 0.04086 (0.75x) | 0.02682 (0.49x) | 0.02679 (0.49x) | 0.02465 (0.45x) |
| Shrinking split B decay (2000000) | 0.04630 (1.00x) | 0.04254 (0.92x) | 0.04307 (0.93x) | 0.03843 (0.83x) | 0.05921 (1.28x) | 0.05880 (1.27x) |
| Dimerization (100000000) | 0.2008 (1.00x) | 0.2015 (1.00x) | 0.1880 (0.94x) | 0.1921 (0.96x) | 0.2036 (1.01x) | 0.6957 (3.47x) |
| Lotka-Volterra (100000) | 0.007668 (1.00x) | 0.004362 (0.57x) | 0.009903 (1.29x) | 0.003738 (0.49x) | 0.003873 (0.51x) | 0.003686 (0.48x) |
| Lotka-Volterra (1000000) | 0.03231 (1.00x) | 0.03020 (0.93x) | 0.03310 (1.02x) | 0.03290 (1.02x) | 0.02797 (0.87x) | 0.03134 (0.97x) |
| Lotka-Volterra (10000000) | 0.1118 (1.00x) | 0.1082 (0.97x) | 0.1136 (1.02x) | 0.1119 (1.00x) | 0.1130 (1.01x) | 0.3095 (2.77x) |
| Order-3 Brusselator (20000) | 0.1110 (1.00x) | 0.1066 (0.96x) | 0.1107 (1.00x) | 0.1034 (0.93x) | 0.1102 (0.99x) | 0.1101 (0.99x) |

For a compact central summary, the next table takes the geometric mean within the related
Shrinking and Lotka-Volterra scenario families, then gives all six CRN families equal weight.
Every column remains normalized to timing, and the final column is the largest single-scenario
ratio to timing.

| policy | role | central equal-family ratio | conservative family-worst ratio | worst scenario ratio |
|---|---|---:|---:|---:|
| timing | measured wall-clock | 1.000x | 1.000x | 1.000x |
| `T=500` | pilot-selected | 0.928x | 0.992x | 1.080x |
| `T_l=100` | far low | 0.999x | 1.039x | 1.291x |
| `T_l=250` | near low | 0.888x | 0.958x | 1.018x |
| `T_h=1000` | near high | 1.007x | 1.117x | 1.565x |
| `T_h=2000` | far high | 1.330x | 1.650x | 3.465x |

### Why some lower thresholds appear faster

A threshold is a decision boundary, not a continuous speed control. If two thresholds remain on
the same side of every score encountered by a trajectory, they execute the same mode sequence and
consume exactly the same random stream. Any measured runtime difference between them is then timing
noise, not evidence that one threshold is better.

This matters for the two most visually striking rows:

- Lotka-Volterra at `n = 1e5` takes only about 3--5 ms under a deterministic policy. Thresholds
  250, 500, 1000, and 2000 all execute exactly one initial batch and then about 412 Gillespie calls,
  with identical mode and final-state hashes for each seed. Their different table entries cannot be
  caused by the threshold.
- For Shrinking without B decay, thresholds 500, 1000, and 2000 are also identical policies: one
  batch followed by about 1038 Gillespie calls. Threshold 250 is genuinely different—it spends
  about 6.6% of simulated time batching—and was slower than 500 in five of six paired seed blocks,
  with a paired median slowdown of about 12%.

Thresholds 250 and 500 also produced identical recorded mode signatures and final-state hashes for
Oregonator, Dimerization, the order-3 Brusselator, and Lotka-Volterra at `n = 1e7`. Where their
recorded mode behavior differs, the tradeoff is mixed: 250 is
better for Rössler and split-B-decay Shrinking in these runs, but worse for Lotka-Volterra at
`n = 1e6` and no-B-decay Shrinking. Thus `T_l` being close to `1x` in many rows mostly indicates a
broad mode-sequence plateau or two engines near crossover; it does not identify new structural
coefficients.

The timing policy is also an adaptive baseline, not an oracle. It bootstraps from the old proxy,
probes periodically, and requires a 4x measured advantage before overriding. On a short run it can
spend much of the trajectory learning a choice that a fixed threshold makes immediately. The paired
frozen-state engine timings—not the adaptive trajectory time—are the fitting oracle for `T*`.

The end-to-end result therefore supports a **broad useful threshold region**, not a unique empirical
optimum. `T = 500` is close to timing, while the separately aggregated table makes `T = 250` look
4.3% faster centrally; policy-equivalent rows and opposing paired effects mean that difference is
not sufficient to replace 500 or fit CRN-static terms. Moving far from the useful region is still
demonstrably harmful: `T_l = 100` reaches 1.29x timing on small Lotka, `T_h = 1000` reaches 1.56x
timing on Rössler, and `T_h = 2000` reaches 3.47x timing on Dimerization.

All 720 requested held-out comparison trials completed. Every prospective-policy same-seed repeat
had identical final-configuration and mode-signature hashes; the wall-clock policy was, as expected,
not exactly replayable on several CRNs.

The forced-batch order-3 Brusselator anchor exceeded the 30-second cap in every pilot repeat and
in its one held-out anchor run; this is recorded as a timeout, never as a timing observation. The
final comparison therefore repeated the five finite thresholds and timing policy on all six
held-out seeds, but collected the forced anchors only once. Several sub-50-ms cases remain noisy
despite seed blocking, so small differences inside the plateau should not be over-interpreted.

The reproducible inputs and outputs are the
[`switching_policy_comparison.py`](switching_policy_comparison.py) runner,
[`switching_policy_comparison_runs.csv`](switching_policy_comparison_runs.csv) raw trials,
[`switching_policy_comparison_selection.json`](switching_policy_comparison_selection.json)
pilot decision, per-CRN
[`pilot`](switching_policy_comparison_pilot_case_summary.csv) and
[`held-out`](switching_policy_comparison_final_case_summary.csv) summaries, aggregate
[`pilot`](switching_policy_comparison_pilot_policy_summary.csv) and
[`held-out`](switching_policy_comparison_final_policy_summary.csv) policy summaries, and
[`switching_policy_comparison_metadata.json`](switching_policy_comparison_metadata.json)
environment/source fingerprint.

## Separate-cost threshold model (2026-08-01)

This tests the "Revised research hypothesis" above: instead of regressing `T` directly, fit
`C_batch` and `C_Gillespie` **separately** with nonnegative coefficients and take the ratio only
afterwards. The result is that the decomposition works, and that the earlier preference for the
population-only model `T_logN` was an artifact of an evaluation metric that could not discriminate.

### The measurement gap that had to be closed first

`profile_batch_costs.py` set `measure_gillespie=True` for only the `matched_shrinking` and
`branch_count_control` families, so 30 of 37 profile cases had **no denominator measured**. Those 30
are exactly the cases that vary `q`, `o`, `g`, and `N`, so `C_Gillespie`'s dependence on them was
unidentifiable. The new opt-in flag `--measure-gillespie-all` times both engines for every case;
without it the script reproduces the original CSVs byte-for-byte.

### Data

37 paired states, 6 families, seeds 1 and 2, 301 interleaved repeats, 5,000-reaction Gillespie
blocks — the original timing protocol, now applied to every case:

```powershell
python -B benchmark/profile_batch_costs.py timings --repeats 301 --seed 1 `
    --gillespie-reactions 5000 --measure-gillespie-all `
    --output benchmark/batch_cost_profile_timings_allg.csv
# ...and again with --seed 2 --output benchmark/batch_cost_profile_timings_allg_seed2.csv
python benchmark/threshold_cost_model.py `
    --timings benchmark/batch_cost_profile_timings_allg.csv `
    --timings benchmark/batch_cost_profile_timings_allg_seed2.csv `
    --json-out benchmark/results/threshold_cost_model_fit.json
```

> [!IMPORTANT]
> This collection ran on a **different CPU and a newer nightly** than the 2026-07-12 profile, and is
> uniformly 60-75% slower in absolute terms. That is the "repeat on another CPU" validation the
> earlier section asked for: the *ordering and structure* of `T*` replicated (for example `T*` still
> falls monotonically from about 622 to 342 as split-`B` channels go 3 -> 10), but absolute
> microsecond costs did not. **Do not pool the `_allg` CSVs with the 2026-07-12 CSVs.** Coefficients
> below are machine- and build-specific; the *form* is the portable claim, not the numbers.

### Fitted costs

Both fits minimize relative error (rows scaled by `1/target`) under nonnegativity, so no term can
claim negative work. NNLS dropped the terms marked below, which is itself informative.

`C_batch` (us per batch call), R^2 = 0.954, median relative error 3.6%:

| term | coefficient | note |
|---|---:|---|
| `const` | 4.267 | fixed per-call overhead |
| `o` | 1.883 | |
| `o log2(N)` | 0.0644 | order-dependent collision-search work |
| `[g >= 1]` | 4.157 | the large generative jump |
| `g log2(N)` | 0.0636 | magnitude of `g` matters, confirming the old indicator was too coarse |
| `sum_{d=1}^{o} q^d` | 0.0731 | dense recursive sampling nodes |
| `score` | 0.00231 | active-lane work |
| `B` | 0.642 | channel alternatives |
| `log2(N)`, `g`, `q^o`, `W_lanes`, `q` | 0 | dropped |

`C_Gillespie` (us per exact reaction), R^2 = 0.994, median relative error 2.6%:

| term | coefficient | note |
|---|---:|---|
| `const` | 0.01444 | |
| `B (q - 2)` | 0.000917 | dense rate evaluation |
| `B o` | 0.000957 | per-channel reactant work |
| `(q - 2)` | 0.000622 | dense jump |
| `B`, `C_sync/H` | 0 | dropped; at `H = 5000` the sync term is negligible |

`q^o` was dropped only because `sum_{d=1}^{o} q^d` is nearly collinear with it and absorbed the
effect; this is a reparameterization, not evidence that terminal-lane scanning is free.

### The ratio predicts `T*`; the population-only model does not

| predictor of `T*` | median rel. error | max rel. error | correlation with `T*` |
|---|---:|---:|---:|
| separate-cost ratio, in-sample | 0.036 | 0.163 | **0.986** |
| separate-cost ratio, leave-one-family-out | 0.189 | 0.652 | **0.944** |
| `T_logN` refit on this CPU (`713.8 + 4.40 log2 N`) | 0.276 | 2.644 | 0.059 |
| `T_logN` seed-1 coefficients (`460.2 + 8.35 log2 N`) | 0.321 | 1.906 | 0.059 |
| constant `T = 500` | 0.404 | 1.242 | n/a (constant) |

Measured `T*` spans 223 to 1524 (6.8x) across these states. An affine function of `log2(N)` has
**essentially no correlation with it** — it cannot represent the structural variation at all.

### Why the earlier evaluation preferred `T_logN`

The regret metric is **degenerate on frozen-state data**. Only 8 of 37 states lie within 2x of their
own break-even and only 6 prefer batch; the rest sit 10x-800x away from the boundary, where every
candidate threshold in a wide band makes the same decision. Consequently the separate-cost ratio,
`T_logN`, `T = 500`, **and the oracle `T*` itself** all score mean/worst regret 1.0000/1.0000 on
these states. Regret cannot rank predictors here; threshold-prediction accuracy can. The earlier
grouped-regret comparison that made `T_logN` look best (1.000/1.000) was measuring the same
degeneracy, not predictive quality.

### Controlled structural contrasts

Both families hold `N`, `o`, `g`, and the prospective score fixed, so every change in `T*` is
structural. `T_hat` is the leave-one-family-out prediction, i.e. the family was absent from training.

| contrast | varied | `T*` | `T_hat` (held out) |
|---|---|---:|---:|
| `dense_support` | `q` = 4, 6, 8, 12 | 675 -> 463 -> 333 -> 223 | 700 -> 571 -> 477 -> 368 |
| `branch_count` | `B` = 3, 4, 6, 10 | 622 -> 545 -> 475 -> 342 | 603 -> 557 -> 491 -> 415 |

Direction and ordering are recovered in both, with the slope under-predicted when the family
supplying that feature's only variation is held out — expected, and the clearest statement of what
the design still lacks.

### Near-boundary held-out decision test

The section above establishes that the ratio predicts `T*` well but *cannot* be ranked by regret,
because the profile states are far from their boundaries. This experiment removes that excuse by
constructing states that sit on the boundary, and it is the test that could have falsified the model.

`near_boundary_states.py` exploits an exact scaling law. `inits_from_n` scales every initial count
proportionally and `_make_sim` uses `volume = initial_n`, so composition -- and therefore the active
probability -- is invariant under scaling `n`; since `E[L]` is proportional to `sqrt(N)`,

```text
score(n) = score(n_ref) * sqrt(n / n_ref)     (verified exact to 4 significant figures)
```

so solving `score(n) = ratio * T*` needs a single Newton step, `n = n_ref (ratio T* / score_ref)^2`,
with only `T*` re-measured because it drifts logarithmically. For 18 base CRNs it targets
`score/T*` of 0.6, 1.0, and 1.6, bracketing each boundary from both sides.

| state set | states | within 2x of break-even | prefer batch |
|---|---:|---:|---:|
| profile matrix (training) | 37 | 8 | 6 |
| near-boundary (held out) | 46 | **44** | 14 |

The held-out set spans `score/T*` of 0.327 to 1.67 and `T*` of 223 to **3397** -- note the top of
that range is more than double the training maximum of 1524, so the ratio model is **extrapolating**,
not interpolating. Fitting uses the 37 profile states only; every predictor below is then applied
unchanged, including the constant and `T_logN` baselines, which are selected/fit on the same training
states so the comparison is fair.

| predictor | mean regret | worst regret | misclassified | median rel. error on `T*` |
|---|---:|---:|---:|---:|
| **separate-cost ratio** | **1.0094** | **1.309** | **3 / 46** | **0.087** |
| constant selected on train (`T = 604`) | 1.1124 | 2.259 | 13 / 46 | 0.293 |
| constant `T = 500` | 1.1029 | 2.259 | 14 / 46 | 0.381 |
| `T_logN` fit on train (`714 + 4.40 log2 N`) | 1.1153 | 2.259 | 10 / 46 | 0.421 |
| oracle `T*` (unattainable) | 1.0000 | 1.000 | 0 / 46 | 0.000 |

Regret is now discriminating -- the oracle scores 1.0000 while every baseline sits near 1.11 -- and
the ratio model captures most of the available gain: **mean regret falls from about 11% to 0.9%,
worst case from 2.26x to 1.31x, and misclassifications from 10-14 down to 3.**

Where each predictor fails is the informative part:

- The ratio model's three errors are all at `score/T* ~ 1`, where the two engines are genuinely
  near-tied and being wrong is nearly free (regrets 1.309, 1.118, 1.006). Its single worst state,
  `collision_o3_g0` at `T* = 2541`, is an `o = 3` extrapolation past the training range where it
  under-predicts (`T_hat = 1361`).
- Every baseline fails on the same **structural** extremes, in both directions: `collision_o3_g2`
  (`T* = 3397`, constant says 604 -- batches when it should not) and `dense_support_q8`/`q12`
  (`T* = 349`/`229`, constant says 604 -- uses Gillespie when it should batch). No constant can be
  simultaneously right about a CRN needing 3397 and one needing 229.
- `T_logN` misclassifies slightly fewer states than the constant (10 vs 13) but has both higher mean
  regret and by far the worst threshold error (0.42). It is not tracking structure; it is
  effectively a constant with a small population tilt, and its count advantage is luck of placement.

### End-to-end check: does the CRN dependence actually change wall-clock? (2026-08-01)

The frozen-state results say the break-even rate varies 15x, which predicts that no constant can be
right everywhere. This experiment asks the practical follow-up: **does that matter for real
run times?** `end_to_end_threshold_check.py` runs the ten comparison scenarios under the wall-clock
timing policy, a constant `T = 670`, a constant `T = 500`, and a per-CRN `T_hat` from the cost model
evaluated at each scenario's initial state, on six fresh seeds with three repeats each (720 runs).

> [!CAUTION]
> **Seven of the ten scenarios are policy-equivalent.** `T = 670`, `T = 500`, and the cost model
> execute the *identical* mode sequence there -- same batch/Gillespie/switch counts, hence the same
> random stream and the same work. Their timing differences on those rows are pure noise, and the
> measured noise floor is **1.7% to 10.8%**. Any aggregate over all ten scenarios is therefore
> mostly averaging noise, and differences below about 10% between deterministic policies are not
> measurable on this suite.

Only `rossler_n1e5`, `shrinking_n2e6`, and `shrinking_split_b_decay_n2e6` are genuinely different
policies. Restricting to those, with per-seed medians paired against the timing policy:

| scenario | timing | `T = 670` | `T = 500` | cost model |
|---|---:|---:|---:|---:|
| `rossler_n1e5` | **3.488 s** | 4.481 (1.288x) | 3.982 (1.147x) | 3.786 (1.098x) |
| `shrinking_n2e6` | 0.0655 | 0.0727 (1.101x) | **0.0614 (0.937x)** | 0.0657 (1.045x) |
| `shrinking_split_b_decay_n2e6` | 0.0677 | 0.0804 (1.235x) | **0.0636 (0.929x)** | 0.0683 (1.009x) |
| **central (these 3 only)** | 1.000x | 1.191x | **0.991x** | 1.028x |
| **worst (these 3 only)** | 1.000x | 1.287x | 1.126x | **1.076x** |

The paired seed counts are consistent, not marginal: on Rössler the ordering
`cost model < 500 < 670` holds in all six seeds and *none* of the three beats timing in any seed;
`T = 500` beats timing in 5/6 seeds on Shrinking and 4/6 on split-B Shrinking.

Three conclusions, none of which is what the frozen-state work alone would have predicted:

1. **`T = 670` is a poor threshold.** It is the worst policy on every scenario that can tell the
   difference (1.19x central, 1.29x worst), because it pushes Rössler and Shrinking through far too
   much Gillespie. A rate measured on one CRN must not be applied to others -- which is the whole
   point, but it is worth stating that the specific number is bad, not merely unjustified.
2. **`T = 500` remains an excellent constant**, essentially tying the timing policy centrally
   (0.991x) on the discriminating rows.
3. **The cost model buys worst-case robustness, not average speed.** It has the best worst case
   (1.076x vs 1.126x and 1.287x) but is slightly behind `T = 500` centrally. That is exactly the
   frozen-state story carried through: the model helps where structure is extreme, and these ten
   scenarios contain no extreme structure -- the per-CRN `T_hat` values span only 461 to 1009.

**This suite cannot settle the question.** Three discriminating scenarios, two of which are the same
CRN family, against a noise floor of up to 10.8%. Demonstrating that structure-aware thresholds pay
off end-to-end requires scenarios whose *trajectories actually pass near their own boundary* and
whose structures differ sharply -- the `dense_support`-style high-`q` and order-3 CRNs, run as full
trajectories rather than frozen states. Until then, `T = 500` is the defensible practical choice and
the cost model is the better-understood one.

### Where the scale factor comes from (2026-08-02)

The separate-cost model predicts `T*` well but has to be multiplied by a fitted constant well below
1 before it wins end to end. That constant had no established mechanism. Four were proposed and
tested; three are refuted and one is confirmed but only partially sufficient. Recording all four,
because the refutations are as useful as the confirmation.

#### The constant is a constant, not a missing term

`optimal_threshold_fit.py` measures the end-to-end optimum directly rather than fitting to `T*`.
For each scenario it sweeps multipliers of that scenario's own `T_hat`, groups them into
equivalence classes by mode signature (thresholds that never straddle an encountered score execute
the identical run, so timing differences between them are noise), times one representative per
class, and treats every class within 5% of the fastest as tied. The optimum is therefore a plateau,
and `alpha = T_opt / T_hat` is the geometric centre of the winning plateau.

| | |
|---|---|
| `alpha` | **0.368**, 95% CI **[0.304, 0.445]** -- so the shipped 0.5 was too high |
| free slope of `log T_opt` on `log T_hat` | 1.209 +/- 0.310, **t = +0.68 versus slope 1** |
| does a free slope fit better? | no -- residual sd 0.4866 free versus **0.4805** with `alpha` forced constant |
| structural correlates of `log alpha` | all weak: `g` +0.29, `q` -0.12, `B` -0.10, `log2 N` -0.12 |
| family means | branch_count [0.29, 0.47], dense_support [0.24, 0.46], shrinking [0.21, 0.61] -- overlapping |

A single constant is statistically adequate and no missing structural term is detectable. The
apparent 5.9x spread in `alpha` is not structure: per-scenario optima are poorly determined, with
the runner-up within 5% in 11 of 23 scenarios, so the argmin flips between distant multipliers on
noise. Observed sd of `log alpha` is 0.481 against 0.097 explainable by grid spacing.

> [!NOTE]
> This 0.368 is the mean of per-scenario optima, weighting every scenario equally. The earlier
> held-out sweep found 0.5-0.6 best because it optimised a different thing -- a single global
> multiplier against the equal-family aggregate, which is dominated by scenarios where being wrong
> is expensive. For choosing a shipped constant the second is the relevant objective. The two
> numbers are not in conflict; they answer different questions.

Caveats: 23 of 38 scenarios survived, and **every survivor has `o = 2`**, so this says nothing about
reaction-order dependence.

#### Refuted: the training data was sampled from the wrong states

`profile_batch_costs.py` times every case from its CRN's **initial** configuration, but a running
policy spends its whole life at intermediate states. Fitting on the wrong state distribution would
bias the costs, and `alpha` would be absorbing that bias. `threshold_model.py` already samples the
right distribution -- `capture_frozen_states` walks a trajectory and freezes states along it -- so
the fit was repeated there (`--trajectory-states`).

| fit -> tested on | bias (geo-mean `T_hat`/`T*`) | median rel. err | corr |
|---|---:|---:|---:|
| initial -> initial (in sample) | 0.998 | 0.036 | 0.986 |
| **initial -> trajectory** | **1.092** | 0.155 | 0.736 |
| trajectory -> trajectory (in sample) | 1.035 | 0.109 | 0.820 |

The initial-configuration model over-predicts `T*` at trajectory states by only 9%, implying
`alpha ~ 0.92`; explaining 0.368 would need a bias near 2.7x. Per-CRN biases scatter 0.88-1.35 with
no systematic direction. **Refuted.** Two by-products: the model handles Brusselator's order-3
trajectory states with bias 0.920, partly closing the `o = 3` gap; and a trajectory-only *fit* gives
wildly unstable coefficients, because 96 states across only 6 CRNs give the structural features just
six distinct values.

#### Refuted: switching overhead, and Gillespie block coarseness

Both were tested earlier and are recorded above: switch overhead is 0.018% of run time under the
timing policy and 0.003% under a fixed threshold, three orders of magnitude too small; and budgeting
Gillespie blocks by reaction count rather than duration changed wall-clock by 0.2% even though it
corrected a genuine 2x block-sizing error.

#### Confirmed: the fitted costs were measured cold, and a real run is warm

`_benchmark_once` builds a **fresh simulator for every timed call**, so every fitted cost is a
*first* engine call on cold data structures -- cold caches, untrained branch predictor, first-touch
page faults on the urn and transition arrays. A real run executes thousands of warm calls. This is
not symmetric between engines: batch touches the transition arrays, the urn and all `q^o` terminal
lanes, while Gillespie touches a small dense rate vector.

`profile_batch_costs.py --warm` keeps one simulator alive per case and restores its configuration
with `reset` before each timed call, so memory stays warm as in a real run while the state is held
fixed -- separating warmth from trajectory drift. Across all 37 cases, 301 repeats, two seeds:

| | geometric mean warm/cold |
|---|---:|
| batch cost | **0.617** |
| Gillespie cost | **0.942** |
| `T*` | **0.656** (range 0.526 .. 1.067) |

The asymmetry is exactly as predicted, and consistent across every case. Refitting on warm costs
gives the strongest single piece of evidence that this is a real mechanism rather than a
coincidence: in the cold fit `const` was the **largest** coefficient (4.267) and simultaneously the
**least identifiable** (nonzero in only 73% of bootstrap resamples, confidence interval spanning
zero). Fit on warm costs it drops to **exactly zero**. A fixed per-call overhead that vanishes once
you stop measuring first calls is what a cold-start artifact looks like.

**But it is not sufficient.** 0.656 accounts for about 42% of the log-gap to 0.368, leaving a
residual near 0.56 still unexplained. Warm measurement is a large, real contributor; something else
remains.

#### Reading the coefficients

A coefficient's magnitude alone means nothing here, because the features differ in scale by orders
of magnitude. `active_reactions` has the smallest cold coefficient (0.0023) and the largest feature
range (1.18 to 3590), contributing up to **26%** of `C_batch`; a 400-sample bootstrap finds it
nonzero in **98%** of resamples. The intercept, with the largest coefficient, is nonzero in only
73%. Judge terms by contribution share and bootstrap stability, not by coefficient size.

### The cost model as a real selector: `HEURISTIC_COST_MODEL` (2026-08-02)

Every comparison before this one fed the simulator a *fixed* threshold computed once in Python from
the initial state. Selector 3 recomputes the threshold from the live configuration at every
decision, which is the production form:

```rust
let threshold = if self.switch.heuristic == HEURISTIC_COST_MODEL {
    self.cost_model_threshold(score)      // state-dependent, per iteration
} else {
    self.switch.proxy_threshold
};
self.set_mode(score < threshold);
```

`cost_model_threshold` is allocation-free and costs one `log2` plus a handful of multiplies, which
is negligible beside the `calculate_total_propensity` already running every iteration. It was
verified against the Python fit on all 37 profile cases and agrees to **3.2e-16**; a silent
divergence between the fitted model and its implementation would have invalidated every comparison
built on it.

Design notes:

- **Coefficients are settable from Python, not compiled in.** They are machine- and build-specific:
  the functional form transfers between CPUs, the numbers do not. This also allows a refit to be
  tested without a rebuild.
- **`cost_model_scale` defaults to 1.0**, so a model fitted on warm costs needs no correction.
- With no coefficients supplied the selector falls back to `proxy_threshold`, degrading to a fixed
  threshold rather than to nonsense.
- The feature order must match `BATCH_FEATURES` / `GILLESPIE_FEATURES` in
  `threshold_cost_model.py`. This is checked only by length, which is the weakest part of the
  design; a reordering on the Python side would silently produce wrong thresholds.

### Head-to-head with the real selector (2026-08-02)

The first comparison in which the cost model runs as it would ship: `HEURISTIC_COST_MODEL`
recomputing the threshold from the live configuration at every decision, rather than a threshold
fixed for the whole run. The measured quantity is wall-clock seconds for one `BatchSimulator.run`
call to reach `t_max`, construction excluded. 43 scenarios, 6 policies, seeds 901-904, two repeats.

| policy | equal-family | family-worst | worst scenario | vs best tested | worst regret |
|---|---:|---:|---:|---:|---:|
| `timing` | 1.000x | 1.000x | 1.000x | 1.180x | **3.065x** |
| `constant_250` | **0.885x** | 1.001x | 1.201x | **1.044x** | 1.211x |
| `constant_500` | 0.931x | 1.133x | 1.755x | 1.098x | 1.771x |
| **`cost_model_cold` x0.368** | 0.888x | **0.996x** | **1.061x** | 1.048x | **1.177x** |
| **`cost_model_warm` x1.0** | 0.891x | 1.023x | 1.206x | 1.051x | 1.247x |
| `cost_model_warm` x0.56 | 0.898x | 1.018x | 1.070x | 1.060x | 1.202x |

Four conclusions, in decreasing order of how much the data supports them:

1. **Every deterministic policy beats the adaptive timing policy**, by about 10% centrally. Timing
   is 1.180x off the best-tested envelope and has by far the worst tail, 3.065x on its worst
   scenario. It bootstraps from the old proxy, probes rarely, and needs a 4x measured advantage
   before overriding, so on a short run it spends much of the trajectory still learning. **Timing is
   not the target to beat; it was passed some time ago.**
2. **`T = 500` is clearly worse than `T = 250`** on every metric, so the earlier pilot's selection
   of 500 does not survive a wider scenario matrix.
3. **The cost model's advantage is the tail, not the average.** Centrally, `constant_250` (0.885)
   and the three cost-model variants (0.888-0.898) are within about 1.5% of each other, which is
   well inside the noise floor and should not be read as an ordering. What separates them is the
   worst case: `cost_model_cold` is the only policy whose family-worst is below 1.0 (0.996) and its
   worst scenario is 1.061x against 1.201x for `constant_250`. That is the same pattern the
   frozen-state work found, and the expected one: a constant must be wrong somewhere once the true
   break-even spans 15x, while a structural model need not be.
4. **A warm-fitted model with no correction at all is competitive.** `cost_model_warm` at scale 1.0
   scores 0.891 centrally against 0.888 for the cold model with its fitted 0.368 -- indistinguishable
   -- at the cost of a worse tail (1.206x versus 1.061x). This matters for portability: a model that
   needs no empirical constant can move to a new machine with a recalibration of coefficients alone.

Note also that applying the residual 0.56 to the *warm* model does not help centrally (0.898 versus
0.891 unscaled), which argues the residual is not a real further correction so much as the breadth
of the plateau. The earlier finding stands: the optimum is a wide region, the runner-up is within 5%
in about half of scenarios, and differences of a few percent between policies inside that region are
not meaningful.

### Held-out replication, and the one CRN where adaptive timing still wins (2026-08-02)

The head-to-head above was repeated on fresh seeds 921-925, with two changes: scenarios that hit the
time cap are screened out once instead of burning cap-seconds on every run, and the CRNs are also
raced at their **natural** populations. That second change matters because boundary placement drops
every order-3 CRN -- Brusselator cannot reach its break-even below the population cap and the
collision `o = 3` family times out there -- so the natural runs are the only end-to-end order-3
coverage available, at the cost of not sitting near their own boundary.

Read as a single aggregate the result looks like a failure to replicate: `cost_model_cold`'s worst
case goes from 1.061x to 1.253x and it no longer beats `constant_250`. Split by scenario type, the
picture is the opposite.

| subset | `constant_250` | `cost_model_cold` | `cost_model_warm` | `constant_500` |
|---|---|---|---|---|
| boundary-placed, n=40 (geo / worst) | 0.851 / 1.221 | 0.858 / **1.052** | 0.856 / 1.162 | 0.912 / 1.777 |
| `dense_support` only, n=7 (geo / worst) | 1.006 / 1.221 | **0.949 / 1.052** | 1.045 / 1.162 | 1.169 / 1.777 |
| natural populations, n=2 (geo) | 1.089 | 1.113 | 1.093 | 1.042 |
| all 42 (geo / worst) | 0.861 / 1.229 | 0.868 / 1.253 | 0.866 / 1.279 | 0.917 / 1.777 |

1. **The tail advantage replicates on the comparable subset.** On boundary-placed scenarios
   `cost_model_cold`'s worst case is 1.052x against `constant_250`'s 1.221x, closely matching the
   first run's 1.061x against 1.201x.
2. **It is concentrated exactly where predicted.** On `dense_support` -- the only family varying `q`,
   and the one whose break-even sits furthest from typical -- the cost model wins both centrally
   (0.949 vs 1.006) and on the tail (1.052 vs 1.221). That is the structural advantage doing the
   thing it was built to do.
3. **The aggregate moved because of two added scenarios, not because the effect vanished.** All
   deterministic policies lose to timing at natural populations, and the worst single scenario for
   every one of them is `oregonator_n1e5`.

#### Oregonator appeared to be where deterministic switching loses -- it is not

| policy | ratio to timing on `oregonator_n1e5` |
|---|---:|
| timing | 1.000x |
| `constant_500` | 1.204x |
| `constant_250` | 1.229x |
| `cost_model_cold` | 1.253x |
| `cost_model_warm` | 1.279x |

> [!WARNING]
> **This table is retracted.** All four deterministic policies execute the identical mode sequence
> on this CRN, so it reports one run measured four times, and the differences are noise. A dedicated
> sweep (below) measures a 17% spread across policy-equivalent thresholds and finds the adaptive
> policy itself swinging 34% between runs. It is kept here because the reasoning that followed from
> it shaped several later decisions.

A "hybrid" that keeps the wall-clock override is **not** an acceptable answer here, and it is worth
saying so explicitly because it is a tempting one. The entire point of a deterministic rule is that
the mode sequence does not depend on machine timing: runs replay exactly, results do not shift when
the CPU is busy, and the same input gives the same output everywhere. Reintroducing a measured-time
override to cover the cases the model misses gives all of that back up. Whatever closes this gap has
to be deterministic.

#### What the Oregonator gap actually is

Mode splits on `oregonator_n1e5`, median over seeds:

| policy | batch calls | Gillespie calls | mode switches | seconds |
|---|---:|---:|---:|---:|
| timing | **14** | **14,405** | 7 | **0.1191** |
| every deterministic policy tested | **1** | 18,459 | 1 | 0.1433-0.1502 |

The deterministic policies never batch at all -- the single call is the mandatory initial batch --
while the adaptive policy batches 14 times and needs 22% fewer Gillespie calls.

The frozen-state oracle says the deterministic policies are *right* to refuse. Across 16 states
sampled along the Oregonator trajectory, batching is locally optimal at **none** of them:

| | |
|---|---|
| peak score along the trajectory | 39.79 |
| `T*` range along the trajectory | 393 .. 529 |
| peak `score / T*` | **0.077** |
| states where `score > T*` | **0 / 16** |

For contrast the same collection finds 8/16 batch-optimal states for Rössler and 6/16 for Shrinking,
so this is a property of the stiff oscillator, not of the measurement.

That also explains why no threshold tested so far changes anything on this CRN: the score never
reaches 250, so every candidate from 250 upward executes the identical run. Making a deterministic
rule batch during an Oregonator spike would need a threshold below about 40, an order of magnitude
under anything that works globally.

#### Resolved: there is no Oregonator gap, and the oracle was right

`oregonator_threshold_probe.py` swept thresholds from 0.5 to 500 on the three CRNs where
deterministic switching appeared to lose. The result retracts the gap.

| `oregonator_n1e5` | seconds | vs timing | batch | Gillespie | mode sequence |
|---|---:|---:|---:|---:|---|
| timing | 0.16002 | 1.000x | 18 | 15,805 | |
| `T = 0.5` | 4.72976 | **29.6x** | 411,250 | 0 | batches everything |
| `T = 8` | 1.65198 | 10.3x | 111,964 | 2,974 | |
| `T = 25` | 0.70152 | 4.38x | 59,712 | 7,572 | |
| `T = 40` | 0.14299 | 0.894x | 1 | 17,636 | identical from here down |
| **`T = 60`** | **0.12263** | **0.766x** | 1 | 17,636 | identical |
| `T = 250` | 0.12297 | 0.768x | 1 | 17,636 | identical |
| `T = 500` | 0.12878 | 0.805x | 1 | 17,636 | identical |

Three things follow, and the first two are corrections.

1. **The gap was noise.** Every threshold from 40 upward executes the *identical* mode sequence --
   one batch call and 17,636 Gillespie calls -- so those rows are the same run measured five times.
   Their spread is 0.1226 to 0.1430, i.e. **17%**, which is the noise floor for a run this short.
   The adaptive policy is separately non-reproducible by construction and measured 0.1191s in the
   held-out head-to-head against 0.1600s here, a **34% swing**. The claimed 20-28% deficit sits
   entirely inside those two spreads, and in this run the deterministic policies *beat* timing by
   23%. The earlier conclusion is withdrawn.
2. **The frozen-state oracle was right, emphatically.** It said batching is never locally optimal
   for the Oregonator, and forcing batches confirms it: `T = 0.5` is **29.6x** slower, and on
   Brusselator `T = 1` is **156x** slower. The hypothesis that the local break-even is "the wrong
   criterion for stiff oscillators" is refuted -- it is exactly right, and refusing to batch is the
   correct behaviour.
3. **Brusselator behaves the same way.** Every threshold from 4 upward gives one batch and 5,476
   Gillespie calls; their measured times span 0.0683 to 0.0885, a **30%** spread on identical runs.
   The best deterministic time is 0.737x the adaptive policy.

Rössler appeared to be the one CRN where the adaptive policy genuinely leads, at 1.04-1.15x. It does
not either; see below. Note that on Rössler thresholds of 60 and below all force *complete* batching
(161,441 batch calls, zero Gillespie), which is another policy-equivalent block spanning 11%.

#### Rössler, checked properly: no adaptive advantage anywhere

`paired_policy_check.py` compares policies **per seed**, so each comparison is between runs of the
same trajectory rather than between pooled medians, and applies a sign test over seeds. Eight seeds,
three repeats, `rossler_n1e5`:

| policy | paired ratio to timing, by seed | median | beats timing |
|---|---|---:|---:|
| `T = 8` | 1.020 1.038 0.981 1.013 0.958 0.965 1.025 0.920 | 0.997x | 4/8 |
| `T = 100` | 1.065 1.058 0.996 1.066 0.946 1.018 1.000 0.954 | 1.009x | 3/8 |
| `T = 250` | 1.009 1.042 0.946 1.035 0.907 0.992 0.955 0.901 | 0.973x | 5/8 |
| `T = 500` | 1.059 1.079 0.991 1.133 0.879 0.983 0.997 0.977 | 0.994x | 5/8 |
| `cost_model_cold` | 0.944 0.525 1.000 1.086 0.955 0.982 1.034 0.903 | 0.969x | 5/8 |
| **`cost_model_warm`** | 0.944 0.832 0.901 1.021 0.854 0.967 0.975 0.908 | **0.926x** | **7/8** |

Measured within-policy noise floor: 1.040x median spread across repeats of the same policy and seed.

Every deterministic policy sits at or below 1.0. The adaptive policy's apparent lead was, once more,
an artifact of comparing pooled medians rather than paired runs. `cost_model_warm` beating it in 7 of
8 seeds is the strongest single result in this document -- under a sign test that is p = 0.035
one-sided -- though with a paired median of 0.926x it is a modest effect, and the harness's own
"decisive" label is too strict in demanding a clean sweep.

**Taken with the Oregonator and Brusselator retractions above, there is now no CRN in this suite
where adaptive wall-clock switching measurably beats a deterministic rule.** All three apparent
cases dissolved under a policy-equivalence or paired-comparison check.

#### The structural case, on paired data

The same paired treatment applied to two CRNs whose optimal constants are **opposite**. Paired
medians against the adaptive policy, eight seeds each; noise floors 1.087x and 1.048x.

| policy | `shrinking_n2e6` | `dense_support_q12_r2` | worst of the two |
|---|---:|---:|---:|
| `T = 500` | 0.828x (8/8) | **1.673x** (1/8) | 1.673x |
| `T = 250` | **0.802x** (7/8) | 1.118x (1/8) | 1.118x |
| `T = 100` | 0.890x (7/8) | **0.931x** (6/8) | **0.931x** |
| `cost_model_cold` | 0.830x (7/8) | 0.954x (6/8) | 0.954x |
| `cost_model_warm` | 0.813x (8/8) | 1.065x (1/8) | 1.065x |

`T = 250` is the best policy on Shrinking and 12% *worse* than adaptive on `dense_support`;
`T = 500` is fine on one and catastrophic on the other. The cost model lands near-best on both
without being told which CRN it is running, which is the entire structural claim, now supported by
paired per-seed data rather than by pooled medians.

Two qualifications, both against the version of this claim made earlier in this document:

- **`T = 100` matches or beats `cost_model_cold` on both of these scenarios.** On this pair the
  model is second, not first. It is only on the wider matrix that low constants fall away
  (`constant_125` 0.981 central and `constant_60` 1.127, against 0.878 for the scaled cost model),
  so the defensible statement is that no constant is good everywhere and the model is consistently
  near-best everywhere -- not that the model is the best policy on any given CRN.
- **`cost_model_warm` is clearly worse here** (1.065x) than `cost_model_cold` (0.954x). Warm at
  scale 1.0 is an effectively ~1.8x higher threshold than cold at 0.368, and this CRN wants more
  batching, not less. The warm model's advantage is portability, not accuracy.

#### Determinism, quantified

The same run makes the case for determinism concretely. Across 24 runs (8 seeds x 3 repeats):

| policy | distinct mode signatures |
|---|---|
| timing | **24** -- a different engine schedule on every run |
| every deterministic policy | **8** -- exactly one per seed |

The adaptive policy does not reproduce even when given the same seed, because its decisions depend
on measured wall-clock. Every deterministic policy reproduces exactly. That is the property being
bought, and it is bought at no measured cost in speed.

> [!CAUTION]
> This was a self-inflicted error worth recording. The document already warns that two thresholds
> which never straddle an encountered score execute the identical run, so timing differences between
> them are noise -- and the head-to-head recorded `batch_calls` and `gillespie_calls` all along. The
> mode signatures were not checked before declaring a gap. **Check policy-equivalence before
> attributing any difference to a policy**, especially on short runs, where the noise floor reaches
> 30%.

> [!NOTE]
> Deterministic policies beat timing by about 15% on boundary-placed scenarios (0.851-0.858) and
> lose by about 10% on natural ones. Which number is "the" answer depends entirely on the scenario
> mix, and neither mix is a random sample of real use. Quote the split, not the aggregate.

## Report: the deterministic switching rule, measured against the best we can build (2026-08-10)

All numbers here were collected on **AC power** with the cost model refit on AC-measured costs.
Timings from earlier sessions were taken on battery and are not comparable; batch and Gillespie do
not slow by the same factor when conditions change (measured 1.365x versus 1.221x), so `T*` itself
moves about 12%.

### What the simulator is deciding

At every iteration `run()` chooses between two engines:

* **batch** -- one expensive call (~10-50 us) that resolves a clump of about `E[L]` *reaction
  attempts* at once, many of which are passive and produce nothing.
* **Gillespie** -- one exact reaction at a time, cheap per reaction (~20-100 ns), delegated to rebop.

So the decision is a bulk-buy: batching is worth it only when enough of the clump turns into real
reactions.

### The three quantities

**`score(x)`** -- the existing prospective estimate of how many *real* reactions the next batch would
deliver, computed in Rust at the live configuration, at the K the next batch would actually use:

```text
k = k_reset_target();  N = n + k
E[L]  = sqrt(pi / (2 o (o+g))) sqrt(N)                 expected collision-free batch length
score = (p_real / (kmax(k) C(N,o))) E[L]               = P(a draw is a real reaction) x E[L]
```

**`T*`** -- the *measured* break-even: `C_batch / C_Gillespie-per-reaction`. It reads as "how many
Gillespie reactions this one batch costs the same as", so batching pays exactly when
`score > T*`. Measured across the profile matrix on AC it spans **176 to 1114, a 6.3x range**. That
range is the whole problem: no single constant can be right across it.

**`T_hat(x)`** -- the model's prediction of `T*`, computed from the live state. The rule is
`batch if score >= T_hat`, and `HEURISTIC_COST_MODEL` (selector 3) evaluates it every iteration.

### The model that is fitted

`C_batch` and `C_Gillespie` are fitted **separately**, each against its own directly measured target,
and divided only afterwards. Fitting `T*` directly does not work -- it is a quotient, and regressing
it as a sum gave coefficients whose signs flipped between datasets. Every coefficient is constrained
**nonnegative**, because each term is an amount of real work and none can take negative time. The
fit minimises *relative* error, so a 45 us case does not drown out a 12 us one.

`C_batch`, microseconds per batch call (AC, warm-measured), R^2 = 0.933, median relative error 3.7%:

| term | coefficient | what it counts |
|---|---:|---|
| `o` | 1.6954 | order-dependent fixed work |
| `o log2(N)` | 0.044486 | collision-search iterations, weighted by order |
| `[g >= 1]` | 3.3725 | the generative jump: extra high-precision gamma terms |
| `g` | 0.39704 | magnitude of generativity, not just its presence |
| `g log2(N)` | 0.033379 | generative work per search iteration |
| `sum_{d=1..o} q^d` | 0.062209 | dense recursive sampling nodes |
| `score` | 0.0012967 | occupied-lane work |
| `B` | 0.33955 | channel alternatives |

`C_Gillespie`, microseconds per exact reaction, R^2 = 0.979, median relative error 3.1%:

| term | coefficient | what it counts |
|---|---:|---|
| `1` | 0.017392 | fixed per-reaction cost |
| `B (q-2)` | 0.00065225 | dense rate evaluation across real species |
| `B o` | 0.00070836 | per-channel reactant work |

Note the batch intercept is **absent**: fitted on warm costs it goes to zero, having previously been
the largest coefficient. It was measuring cold-start overhead, not real work.

Prediction quality: correlation with measured `T*` is **0.968**, and leave-one-CRN-family-out regret
is 1.000/1.000 on the frozen states.

### How it performs end to end

Wall-clock seconds for one `BatchSimulator.run` to reach `t_max`, 43 scenarios, 8 seeds x 2 repeats,
paired per seed, execution order rotated. **Regret is against the best policy achieved on that same
scenario and seed**, so 1.000 would mean always being the best available choice. Restricted to the
38 scenarios whose mode signatures show the constants actually executing different runs:

| policy | equal-family regret | worst scenario | vs shipped adaptive |
|---|---:|---:|---:|
| `timing` (shipped, override 4.0) | 1.283x | 2.304x | 1.000x |
| `timing` override 1.0 (greedy) | 1.210x | 1.696x | 0.980x |
| `timing` override 2.0 | 1.209x | 1.748x | 0.972x |
| `timing` override 1.0 + eager probing | 1.218x | 1.825x | 0.981x |
| `T = 100` | 1.363x | 1.941x | 0.994x |
| `T = 250` | 1.193x | **1.407x** | 0.876x |
| `T = 400` | 1.150x | 1.616x | 0.901x |
| `cost_model_cold` (x0.368) | 1.191x | 1.571x | 0.882x |
| **`cost_model_warm` (no correction)** | **1.112x** | **1.331x** | **0.855x** |

**The cost model wins outright**: best regret, best worst case, and fastest against the incumbent --
and it does so at scale 1.0, with no fitted correction, because the cold-measurement artifact that
required one has been removed.

### Why the adaptive policy underperforms

Its decision is, in full:

```rust
let override_proxy =
    self.switch.wdt(non_proxy) * WDT_OVERRIDE_FACTOR < self.switch.wdt(proxy_gillespie);
self.set_mode(if override_proxy { non_proxy } else { proxy_gillespie });
```

with `WDT_OVERRIDE_FACTOR = 4.0`. **It is not a "measure both engines and pick the cheaper one"
policy.** It runs whatever the *old proxy rule* chooses -- the rule this issue exists to replace --
and overrides only when measurement shows the alternative more than **4x** cheaper. When the better
engine is 1.5-3x better, which is the common case, the gate never opens.

The evidence matches exactly. Every scenario where it loses worst is an `_r05` case, placed at half
its break-even and therefore Gillespie-favoured, and it sits in batch mode nearly throughout:

| scenario | timing/best | switches | batch calls | Gillespie calls | best policy |
|---|---:|---:|---:|---:|---|
| `shrinking_r05` | 2.056x | 3 | 2,164 | 10 | `T = 400` |
| `shrinking_b_decay_channels_8_r05` | 2.007x | 2 | 1,348 | 6 | `T = 600` |
| `shrinking_split_b_decay_r05` | 1.955x | 4 | 1,983 | 15 | `T = 600` |

And `corr(log(1 + mode_switches), log(timing/best)) = -0.366`: it loses where it switches **least**.
It is stuck, not thrashing.

Setting the override factor to 1.0 confirms the diagnosis and quantifies it: regret improves from
1.283x to 1.210x and the worst case from 2.304x to 1.696x. So the gate really is the defect. But
even fully greedy the adaptive policy remains **worse than a plain constant** (1.210x versus 1.193x
for `T = 250`) and clearly worse than the model (1.112x). Eager probing does not help (1.218x), so
stale estimates are not the limiter either.

Two structural reasons it cannot catch up, both unfixable by tuning: it must *pay* for the
information, since every probe runs the mode it suspects is worse; and it learns only after being
wrong, whereas the model knows the answer before the first call from CRN structure alone.

### Should a fixed constant be used instead?

The case against, on paired data. `T = 250` is the best constant overall, but its quality depends
entirely on which CRN it meets:

| policy | `shrinking_n2e6` | `dense_support_q12_r2` |
|---|---:|---:|
| `T = 100` | 0.924x (6/8 seeds) | **0.975x** (6/8) |
| `T = 250` | **0.799x** (8/8) | 1.210x (0/8) |
| `T = 500` | 0.855x (7/8) | **1.776x** (0/8) |
| `cost_model_warm` | **0.763x** (7/8) | 1.146x (1/8) |

`T = 250` wins 8/8 on one CRN and loses 0/8 on the other; `T = 500` is 1.776x on `dense_support`.
The optimal constant **inverts** between two CRNs in the same suite, which is what a 6.3x spread in
`T*` implies. A constant must be wrong somewhere; the model need not be, and measurably is not --
worst case 1.331x against 1.407x for the best constant, while also being better on average.

The honest caveat: on any *single* CRN a constant tuned for that CRN beats the model. The model's
claim is that it is near-best everywhere without being told which CRN it is running.

### What this costs and what it needs

Per decision: one `log2`, a handful of multiplies, no allocation -- negligible beside the
`calculate_total_propensity` already running every iteration. The Rust implementation reproduces the
Python fit to 3.2e-16.

The real cost is **calibration**: the coefficients are machine- and build-specific, and `T*` moved
12% merely between battery and AC on one laptop. The functional form transfers; the numbers do not.

> [!IMPORTANT]
> **Corrected 2026-08-10.** This section previously said shipping means "either a calibration step
> or accepting per-machine error". That framing is wrong: **runtime calibration is not an available
> option**, because it reintroduces exactly the nondeterminism issue #14 exists to remove. Only the
> second branch is real. See below.

### Why per-machine calibration is not an option

The goal of issue #14 is that the switching decision be a function of the simulation state alone, so
that a given seed produces a given trajectory. Calibration violates this at two levels:

1. **Across machines.** Coefficients fitted on machine A differ from those fitted on machine B, so
   the two machines switch at different points, consume the random stream differently, and produce
   different trajectories from the same seed. Results stop being reproducible or comparable --
   which is the property the whole exercise is trying to buy.
2. **Within one machine.** Calibration is itself a *timing measurement*. Its output moves with
   thermal state, power source, and background load -- the 12% battery/AC shift is exactly this.
   So an auto-calibrating build is not even reproducible against itself between runs. That is the
   wall-clock nondeterminism of `HEURISTIC_WALLCLOCK`, moved from the inner loop to startup.

The distinction that matters is not "fixed vs. calibrated" but **whether the coefficients are part
of the specification or measured at runtime**. Determinism is preserved if and only if they are part
of the specification. So:

- **Ship fixed default coefficients.** Compiled-in constants, versioned like any other constant.
- **An explicit user override is fine** -- a coefficient vector passed in as configuration is still
  part of the specification, reproducible by anyone given the same vector. What is not fine is the
  library measuring timings and choosing coefficients for itself.
- **Auto-calibration is ruled out** at any point in the lifecycle, startup included.

### What this changes

A fixed shipped vector is still fully deterministic. What varies across machines is not the
*trajectory* but only *how close to optimal the switch points are* -- a performance property, not a
correctness or reproducibility one. That is an acceptable cost; auto-calibration is not.

Two consequences for the open questions:

1. **Measuring the loss from one fixed vector is now mandatory, not optional.** It was previously
   framed as one of two branches. It is the only branch, so the size of that loss determines
   whether the cost model is shippable at all. Still unmeasured. This is now the highest-priority
   open question, ahead of the residual `alpha` and the `o = 3` gap.
2. **It levels the comparison with a fixed constant `T`.** Both a fixed `T` and a fixed coefficient
   vector are machine-independent constants; neither adapts to the machine. So the fixed-constant
   question is not "constant vs. calibrated" but "does evaluating a formula over the *CRN and
   state* beat a single number, when both are frozen at build time?" The cost model's advantage was
   never machine-adaptivity -- it is that `T*` spans 176-1114 (6.3x) *across CRNs and states on one
   machine*. That variation is what a single constant cannot track, and it is unaffected by this
   correction.

## Simulator panic: the collision sampler's precision guard (2026-08-10)

A sweep died mid-run on a panic from the batching engine, not from any switching logic. It is
recorded here because it aborted a multi-hour experiment and shapes how the harnesses are written,
but it is a **simulator bug, independent of issue #14**. Filed as issue #15.

Verbatim, from `global_constant_sweep.py` on scenario `lotka_volterra_r2`, seeds 1501-1512:

```
thread '<unnamed>' (34748) panicked at src\simulator_crn.rs:2412:13:
lhs + ln(u) should always be less than rhs, except in the last iteration.
lhs + ln(u) and rhs: "1646416784.15542727737136986040335493771664261664858935896305",
                     "1646416784.1554272174835205078125".
Potential error: 6.092965495782715e-7. Diff: "0.00000098670417765187809270428866485725619833724309".
t_mid: 2. n = 23394833.
This may indicate a floating point precision bug.
```

> [!NOTE]
> The commit message of `1c2e705` cites `n = 23,394,809`. That is a transcription error; the
> logged value is `23394833`, as above. `n` here is `self.n`, the **real** population -- not
> `n_including_extra_species`.

### Diagnosis

`sample_collision_fast_f128` runs its binary search in f64 and escalates to f128 only when the
comparison looks too close to call. The escalation guard is `simulator_crn.rs:2481`:

```rust
let potential_error = (last_lngamma_value * 2.5 * self.crn.o as f64) * f64::EPSILON;
```

`last_lngamma_value` is assigned in exactly one place -- line 2420, the **first** `ln_gamma` term,
which seeds `rhs`. The loop at line 2424 then adds `o` more `ln_gamma` results of comparable
magnitude to `rhs` without ever updating `last_lngamma_value`. So the bound is scaled by one term
while the quantity actually compared has magnitude roughly `(o+1)` times larger, and floating-point
error scales with the magnitude of the accumulated sum, not with one summand.

The logged numbers confirm this exactly. Back out the implied term from the printed bound
(`o = 2` for Lotka-Volterra):

| quantity | value |
| --- | --- |
| implied `last_lngamma_value` | 5.488055e+08 |
| magnitude actually compared (`lhs + ln u`) | 1.646417e+09 |
| ratio | **3.0000** = `o + 1` |
| actual discrepancy | 9.8670e-07 |
| guard bound as written | 6.0930e-07 -- **does not fire** |
| bound scaled by the compared magnitude | 1.8279e-06 -- **fires** |

The ratio is 3.0000 to five significant figures. Had the bound been scaled by `lhs.abs().max(rhs.abs())`
instead of `last_lngamma_value`, it would have escalated to f128 and the assertion would not have
fired.

This also explains the comment on line 2480 -- *"gonna throw in a 2.5 to be safer, as 1.5 still
encountered the bug"* (`0dad502`, 2026-04-21). Enlarging the fudge factor is compensating for using
the wrong magnitude, so it reduces the failure rate without removing the failure: `2.5` needed to be
at least `3 x 1.62 ~ 4.9` for this particular sample. A correctly scaled bound needs no fudge factor
of that kind.

### Reproduction: not achieved

The panic has **not** been reduced to a deterministic repro. What was tried, all without a hit:

| attempt | coverage | result |
| --- | --- | --- |
| Forced batch (`proxy_threshold = 0`), Lotka-Volterra | 5 populations x 12 seeds = 60 runs | none |
| Direct `sample_collision(r, u, ...)` sweep at padded N = 46.8M | 600k `(r, u)` pairs | none |
| Direct sweep at N = 23,394,830 / 832 / 834 | 480k `(r, u)` pairs | none |
| Mixed mode, thresholds 50-800, n = 11.6-12.0M | 96 runs | none |
| Mixed mode, thresholds 0-1200, n = 23,394,809 / 23,394,833 | 96 runs | none |

Two things make this hard, and both are worth knowing before trying again:

1. **The scenario's population is not reproducible.** `lotka_volterra_r2` places `n` from a
   *measured* break-even, which moves run to run, so re-running the sweep does not revisit the same
   state.
2. **It is a rare coincidence, not a state.** The assertion fires only when `lhs + ln u` lands
   within ~1e-6 of `rhs` on the wrong side. At these magnitudes that is a narrow target in `u`,
   so most trajectories through the same population never hit it.

The productive route is almost certainly not a search over runs but a **direct unit test of the
guard**: construct the `lhs`/`rhs` pair from the logged inputs and assert the escalation condition
fires. That tests the fix rather than the coincidence.

### What was done instead

The harnesses catch `PanicException`, record the run as failed, exclude it from timings, and report
it explicitly rather than silently shrinking the sample. **The sampler bug itself is not fixed** --
nothing in this document's numbers depends on the fix, but a long unattended sweep can still lose a
scenario to it.

## What to run next, and how (2026-08-06)

Two methodological problems were found after the results above were collected. Both are fixed in the
harnesses but **not** in the recorded numbers, so the experiments below need running before any of
the policy comparisons should be trusted further.

### Preconditions -- read before starting

1. **Power state.** Every timing number in this document was collected **on battery**, at 1400 MHz
   base clock on an Intel Core Ultra 7 155H. Plugged in, the processor clocks much higher, so an AC
   run cannot be compared against any of it.
   - Staying on battery keeps the new numbers comparable with the existing ones.
   - Plugging in means **everything must be recollected**, including the cold/warm cost profiles.
   - `python benchmark/measurement_conditions.py` prints the current state. Every harness now stamps
     its JSON output with it, and `measurement_conditions.comparable(a, b)` refuses a cross-power
     comparison. Older result files predate this and carry no stamp, which is why it is not possible
     to verify after the fact that the cold and warm collections were taken at similar battery
     levels -- and their ratio, 0.617x, is what the cold-measurement finding rests on.
2. **The machine must be otherwise idle**, and nothing else may run concurrently. Parallel work
   invalidates the measurement; the noise floor is already 4-30% depending on scenario.
3. **Close any Jupyter kernel holding `import batss`**, or the `.pyd` cannot be replaced on rebuild.
4. Battery drains across a long session and throttles progressively. Paired within-run comparisons
   survive that; comparisons between the start and end of a run do not.

### The ordering bias, and why it matters

`timing` was executed **first** for every scenario and seed in all three comparison harnesses, with
policies in fixed order. Any monotonic drift -- thermal, or battery drain -- therefore accrued to
whichever policy ran first, which was always the adaptive one. `global_constant_sweep.py` now
rotates the execution order and records `execution_position` per run, so a residual position effect
is measurable rather than assumed absent. The other harnesses have **not** been fixed; their results
should be treated as provisional on this point.

### Experiment 1 -- which fixed threshold is actually best (highest priority)

`T = 500` came from an early pilot and `T = 250` from a later one, but both selections rest on
comparisons since retracted for pooling medians across policy-equivalent runs. Paired checks then
found `T = 100` matching or beating both on two CRNs. **The constant the cost model is measured
against is therefore not established**, which undermines every "the model beats a constant" claim.

```powershell
python benchmark/global_constant_sweep.py --seeds 6 --seed-base 1301 --repeats 2 --cap-seconds 8
```

Races seven constants (60 to 1000) plus the adaptive policy and both cost models across the full
matrix. Roughly 1-2 hours. It is paired per seed, measures regret against the per-run best rather
than against the non-reproducible adaptive policy, rotates execution order, and reports results
twice: over all scenarios, and over only those whose mode signatures show the constants actually
executing different runs.

**Read the discriminating-scenarios line first.** If most scenarios cannot distinguish the
constants, the all-scenario averages are mostly noise -- that is exactly what produced three
retractions in this document.

### Experiment 2 -- re-run the paired checks with rotation

`paired_policy_check.py` still has the fixed-order bias. Re-run on the scenarios whose results are
being relied on:

```powershell
python benchmark/paired_policy_check.py --scenario shrinking_n2e6      --seeds 8 --repeats 3
python benchmark/paired_policy_check.py --scenario dense_support_q12_r2 --seeds 8 --repeats 3
python benchmark/paired_policy_check.py --scenario rossler_n1e5         --seeds 8 --repeats 3
```

About 20 minutes each. These are the results the structural case rests on, so they should be clean.

### Experiment 3 -- close the order-3 gap

No order-3 CRN survives boundary placement: Brusselator cannot reach its break-even below the
population cap, and the `collision_o3_*` family times out there. So no end-to-end evidence
constrains behaviour at `o = 3`, while `o` carries one of the more reliable cost-model coefficients.
The cost model does predict Brusselator's *trajectory* states well (bias 0.920), so the gap is in
the policy evidence, not the cost model. Needs either an order-3 CRN whose boundary is reachable, or
a raised population cap with a longer time budget.

### Experiment 4 -- the unexplained residual

After correcting for cold measurement, the warm model still needs `alpha = 0.574` [0.513, 0.648].
This is **not** a cost-model error: the model predicts `T*` at trajectory states well, including its
`q` dependence (`corr(q, log predicted/measured T*) = +0.118`). The gap is between "which engine is
cheaper right now" and "which fixed threshold minimises a whole trajectory", so it is a property of
the policy problem. Candidates not yet tested, all deterministic: hysteresis (a deadband between
entering and leaving batch mode, which the selector still lacks), and a threshold that anticipates
the drift in `score` along the trajectory rather than reading it pointwise.

### Honest limitations

- **Held-out slopes are shallow where a family is the sole source of a feature.** Holding out
  `dense_support_control` removes all `q` variation (median error 33%, worst 65%); holding out
  `collision_grid` removes all `o = 3` and all `N` variation (median 25%). Fixing this needs a
  second, independent family varying each of `q` and `o`, not more states inside existing families.
- **The profile matrix is deliberately adversarial to a population-only model**, since two families
  hold `N` fixed while varying structure. It is the right test for structure sensitivity but is not
  a random sample of states a real trajectory visits. The near-boundary test above addresses this
  directly: its states are placed by scaling `n` on real and controlled CRNs alike, and the
  structural advantage survives.
- **The near-boundary states are still frozen states, not trajectories.** They measure the local
  steady crossover with setup excluded. A whole run also pays mode duration, rebuild and switching
  costs, and truncation, which is why `T = 500` can look fine end-to-end while being a poor estimate
  of any individual `T*`. Nothing here supersedes the end-to-end comparison.
- **Two CRNs could not be brought to their boundary at all.** Brusselator needs `N` far beyond the
  1e10 cap (score reaches only 363 against `T* ~ 3000`), and Oregonator only reaches `score/T* =
  0.33` at `n = 1e10`. Both are excluded and reported by the harness. This is a real property of
  stiff, low-active-fraction CRNs, and it means the order-3 corner of the test set is thin.
- 37 training states, 6 families, one CPU, one build. Coefficients are not portable; the functional
  form is the claim.
- **Not yet wired into the simulator.** `T_hat` is a Python-side predictor. Making it a selector
  requires implementing both cost forms in `SwitchState` and re-running the end-to-end held-out
  comparison; nothing here changes the default, which remains `HEURISTIC_WALLCLOCK`.

### What to do next

1. ~~Build a near-boundary frozen-state set so decision regret becomes discriminating.~~ **Done**
   (`near_boundary_states.py`); the ratio model passed, cutting mean regret from ~11% to 0.9%.
2. Add one independent `q`-varying and one `o`-varying family so no feature depends on a single
   family, then re-check the held-out slopes. The `o = 3` under-prediction at `T* > 2000` is the
   clearest remaining gap, and the only order-3 CRNs available are the ones that cannot reach their
   own boundary.
3. Implement `T_hat` as `HEURISTIC_COST_MODEL` in Rust. Both cost forms are cheap: no `f128`, no
   allocation, and every input (`N`, `q`, `o`, `g`, `B`, `score`) is already computed by the
   prospective selector. ~~The per-machine coefficients are the open design question — either
   calibrate once at build time or ship a profiled default and re-fit on demand.~~ **Superseded**
   (2026-08-10): re-fitting on demand is ruled out — it reintroduces the nondeterminism issue #14
   exists to remove. Ship fixed compiled-in coefficients; allow an explicit user-supplied vector,
   never an auto-measured one. See "Why per-machine calibration is not an option".
4. Then run `switching_policy_comparison.py` end-to-end against the timing policy and fixed
   `T = 500` on held-out seeds. Frozen-state regret does not by itself justify changing the default.
5. **Highest priority, and newly mandatory:** measure how much a *single* fixed coefficient vector
   loses when applied to a machine other than the one it was fitted on. Because calibration is off
   the table, this number decides whether the cost model is shippable at all. Fit on machine A,
   evaluate decision regret and end-to-end wall-clock on machine B, against a vector fitted on B.
   If the loss is small the model ships; if it is comparable to the model's advantage over a fixed
   `T`, the extra complexity buys nothing.

## Where things live / how to test

- **Code:** `src/simulator_crn.rs`. `SwitchState` holds all switching state (config + EMAs + probe
  schedule + observability). The decision is in `run()`. Tuning constants near the `WDT_*`
  definitions: `WDT_EMA_ALPHA` (0.3), `WDT_OVERRIDE_FACTOR` (4.0), `WDT_PROBE_INTERVAL` (256),
  `WDT_PROBE_INTERVAL_COMMITTED` (8192).
- **Python control:** `sim.simulator.heuristic_gillespie_switching = 0 | 1 | 2` for wall-clock,
  proxy, or prospective respectively; `sim.simulator.proxy_threshold = <float>` sets the score
  threshold used by selectors 1 and 2.
- **Threshold harness:** `benchmark/threshold_sweep.py`; its quick/full presets use the authoritative
  CRNs from `benchmark/generate_gallery_figures.py` and checkpoint local JSONL under ignored
  `benchmark/results/`.
- **Structural-model harness:** `benchmark/threshold_model.py`; `quick` collects Oregonator,
  Rössler, and Shrinking, while `full` adds three structural-coverage CRNs. It writes paired raw
  trials, aggregate frozen states, predictions, and the fit/identifiability report under the
  ignored `benchmark/results/` directory.
- **Profiling harness:** `benchmark/profile_batch_costs.py`; `components` attributes coarse Flame
  phases, while `timings` measures uninstrumented batch-call and Gillespie-per-reaction costs with
  warmup, paired seeds, alternating engine order, and globally interleaved cases. Its six saved
  `batch_cost_profile_*.csv` files contain both seed passes and their summaries. Pass
  `--measure-gillespie-all` to time Gillespie for every case, not just the two paired Shrinking
  families; the `batch_cost_profile_timings_allg*.csv` files were collected that way.
- **Optimal-threshold harness:** `benchmark/optimal_threshold_fit.py`; measures the end-to-end
  optimal threshold per scenario by sweeping multipliers of that scenario's own `T_hat`, grouping
  them into mode-signature equivalence classes, and treating classes within a tolerance of the
  fastest as tied. Reports `alpha = T_opt / T_hat` and whether it tracks any structural feature.
- **Warm/cold probe:** `benchmark/warm_vs_cold_engine_cost.py`; compares one-call-per-fresh-simulator
  against repeated calls on a live simulator, to show how much of the fitted cost is cold-start.
  Note its warm series advances the trajectory, so small-`n` cases are drift-contaminated; the
  controlled version is `profile_batch_costs.py --warm`, which resets the configuration each call.
- **Head-to-head harness:** `benchmark/cost_model_head_to_head.py`; races `HEURISTIC_COST_MODEL`
  with cold and warm coefficient sets against the timing policy and fixed thresholds, measuring the
  quantity that actually matters -- wall-clock seconds for one `run` call to reach `t_max`.
- **Separate-cost model harness:** `benchmark/threshold_cost_model.py`; fits `C_batch` and
  `C_Gillespie` independently with nonnegative least squares, forms `T_hat` as their ratio, and
  reports leave-one-family-out threshold accuracy plus the constant and `T_logN` baselines. Note it
  reports **both** threshold-prediction accuracy and decision regret, because regret alone is
  degenerate on frozen-state data (see the 2026-08-01 section). Pass `--evaluate-timings` to fit on
  one CSV set and score an untouched held-out set.
- **End-to-end threshold check:** `benchmark/end_to_end_threshold_check.py`; times whole runs under
  the timing policy, fixed thresholds, and a per-CRN cost-model threshold, and reports both the
  ratio to timing and the regret against the best policy actually achieved. Always check its mode
  signatures before believing a difference: most scenarios are policy-equivalent.
- **Near-boundary harness:** `benchmark/near_boundary_states.py`; solves for the population that
  puts each CRN's prospective score at a requested multiple of its measured break-even, then times
  both engines there. Use it to generate decision test sets where regret can actually rank
  predictors. It runs the population search once and reuses it across `--timing-seeds` so the seed
  passes stay poolable, and it reports CRNs whose boundary is unreachable below `n = 1e10`.
- **End-to-end comparison harness:** `benchmark/switching_policy_comparison.py`; it records raw
  timing/prospective-policy runs, selects `T` only from pilot seeds, and evaluates the frozen
  timing policy, selected `T`, and deliberately high/low controls on held-out seeds.
- **Observability:** `sim.simulator.switch` is a read-only snapshot with per-mode
  `*_wallclock_seconds`, `*_continuous_time`, `*_calls`, plus `mode_switches` and
  `switch_overhead_seconds`. `mode_split(sim)` in `benchmark/dimerization_benchmarks.py` bundles them.
- **Specs:** the four-CRN experiment uses `gallery_specs()` in
  `benchmark/generate_gallery_figures.py`.
- **Build after editing Rust:** `CARGO_TARGET_DIR="$LOCALAPPDATA/batss-cargo-target" maturin develop --release`
  (the `CARGO_TARGET_DIR` outside Dropbox avoids a Windows file-lock; and close any Jupyter kernel
  that has `import batss` loaded, or the `.pyd` can't be overwritten).
- **Tests:** `python -m unittest tests.batss_tests` includes target-time, exact replay, stale-K, and
  deterministic reaction-order regressions.
