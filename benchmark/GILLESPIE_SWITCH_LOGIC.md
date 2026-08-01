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
  `batch_cost_profile_*.csv` files contain both seed passes and their summaries.
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
