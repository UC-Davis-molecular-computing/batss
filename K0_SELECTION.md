# Choosing the filler count k0

How `batss` sizes the filler/catalyst species **K** that pads every reaction up to the CRN's order.
This supersedes the old "reset K to n" heuristic. The claims here are all confirmed empirically
(see [Empirical validation](#empirical-validation)).

## Notation

| symbol | meaning |
|---|---|
| n | number of *real* molecules (sum of the original-species counts) |
| k0 | number of filler molecules **K** (the knob we choose) |
| N = n + k0 | total population the sampler draws from |
| o | order of the CRN = largest reactant count of any reaction |
| a | original order of one reaction (before padding); δ₀ = o − a fillers are added to it |
| v | volume (rate constants are given in the deterministic/macroscopic convention; an a-reactant reaction's rate is divided by v^(a−1)) |
| p | probability a sampled o-molecule set fires a *real* (non-passive) reaction |
| E[ℓ] | expected collision-free run length: reactions per batch before a collision |
| kmax | largest adjusted rate constant over all reactions (`continuous_time_correction_factor`) |

## The objective: maximize E[ℓ]·p

Each batch costs one (expensive) collision-length sample and accomplishes `E[ℓ]·p` real reactions,
so to simulate a fixed number `R` of real reactions,

```
#batches = R / (E[ℓ]·p),     R independent of k0,
```

and minimizing the batch count is exactly **maximizing E[ℓ]·p**. Two facts make this tractable:

- **E[ℓ] = c·√N**, with `c = √(π / (2·o·(o+g)))` a constant fixed by the CRN (g is its
  "generativity"; g=1 for LV/Rössler, g=2 for Oregonator). c does not depend on k0.
- **p = P_real / (kmax · C(N,o))**, and **P_real is independent of k0**: Definition 2.7's rate
  correction (dividing a padded reaction's constant by the k0-falling-factorial of its K-multiplicity)
  exactly cancels the extra ways a padded reaction can draw its K's from the k0 available, so the real
  dynamics are filler-invariant.

Dropping the k0-independent factors,

```
maximize E[ℓ]·p   ⟺   minimize  kmax(k0) · N^(o − 1/2).
```

(Contrast the paper's "simulation slowdown factor" `S = 1/p` (Def. 2.8): minimizing S maximizes p
alone, i.e. it minimizes `kmax·N^o`, dropping the `√N = E[ℓ]` factor. That targets `min(n, crossover)`
instead of `min(2n, crossover)` — the same answer only when `n ≥ crossover`.)

## kmax has two branches; their meeting point is the crossover

kmax is the largest adjusted rate constant, and reactions split into two kinds:

- **padded reactions** (a < o, carrying δ₀ ≥ 1 fillers): adjusted rate falls as `C_padded / k0^δ₀`;
- **genuine order-o reactions** (a = o, no filler): adjusted rate is a constant `C_flat`.

So `kmax(k0) = max(C_padded / k0^δ₀, C_flat)` — it rides the falling padded branch, then flattens
onto `C_flat`. The **crossover** is where they meet. For o = 2 (all padded reactions are order 1, so
δ₀ = 1):

```
crossover = C_padded / C_flat.
```

Because order-1 reactions get no volume division while order-2 reactions are divided by v, this is

```
crossover = v · (max unimolecular rate) / (max bimolecular rate · symmetry),
```

a **config-independent** quantity (rate constants and volume only, not n). The rate-constant ratio is
the CRN-specific multiple of v:

| CRN | max unimol. | max bimol.·sym | crossover |
|---|---|---|---|
| Lotka–Volterra | 1 | 1 | **v** |
| Oregonator | 520 | 1000 | **0.52 v** |
| Rössler–Willamowski | 30 | 1 | **30 v** |

(Computed in `UniformCRN::crossover_k_count`.)

## The optimum: k0* = min(2n, crossover)

Feed each branch of kmax into `minimize kmax·N^(o−1/2)`:

- **below the crossover** (`kmax = C_padded/k0`): objective ∝ `N^(o−1/2) / k0`, which *decreases* with
  k0 up to the interior optimum `k0 = n/(o − 3/2)` (= **2n** for o = 2);
- **above the crossover** (`kmax = C_flat`): objective ∝ `N^(o−1/2)`, which only *increases*.

So the batch count is minimized at

```
k0* = min( n/(o − 3/2),  crossover )      =      min(2n, crossover)   for o = 2.
```

Two regimes:

- **crossover ≤ 2n** (LV, Oregonator): the crossover binds. k0* is **config-independent** (∝ v) — it
  can be computed once from the reaction table and never revisited.
- **crossover > 2n** (Rössler, whose fast autocatalysis makes crossover = 30v): the `2n` term binds,
  so k0* tracks the population. As Rössler's n explodes past `crossover/2`, k0* rises to the crossover
  and the cap takes over.

## General CRNs (any order o)

The order-2 result is a special case. In general each reaction *j* has adjusted rate
`A_j / ff(k0, δ_j)`, where `A_j = (volume-adjusted rate constant)·(symmetry degree)` is the
k0-independent numerator, `δ_j = o − order(j)` is the number of fillers reaction *j* carries, and
`ff(k0, δ) = k0·(k0−1)···(k0−δ+1)` is the K-falling-factorial (`ff(k0,0)=1`; `≈ k0^δ` for `k0 ≫ δ`).
Then

```
kmax(k0) = max_j  A_j / ff(k0, δ_j),        k0* = argmin_{k0 ≥ 1}  kmax(k0) · (n + k0)^(o − 1/2).
```

Because kmax is a max of terms each ∝ k0^(−δ_j), the objective is piecewise-smooth and its minimizer
is the smallest-objective candidate among the points below.

**Structure of `kmax`.** In log–log each reaction is a straight line `ln A_j − δ_j·ln k0` of slope
`−δ_j`, and `kmax(k0)` is their upper envelope. As k0 grows the active (topmost) line has **strictly
decreasing** `δ`: the steepest padded reaction (largest δ) wins at small k0, successively flatter ones
take over, and the flat floor `C_flat = max_{δ_j = 0} A_j` wins at large k0. Each adjacent pair of
active lines meets at **one crossover**, so in general there are *several* crossovers, not one.

**Candidates for the minimizer** of `f(k0) = kmax(k0)·(n + k0)^{o − ½}`:

- **the interior optimum of each active branch j**, `k0 = δ_j·n / (o − ½ − δ_j)` — but only when it
  lands inside that branch's own k0-interval (between its two crossovers) and `o − ½ − δ_j > 0`;
  otherwise `f` is monotonic across that branch and it contributes no interior point;
- **each crossover kink**, since `f` is continuous but bends there and its minimum can sit exactly on a
  kink (left branch still falling, right branch already rising).

Evaluate `f` at every valid candidate and take the smallest.

**So it is *not* `min(n/(o−3/2), crossover₁, crossover₂, …)`.** The interior optimum is not a fixed
`n/(o−3/2)` shared by all branches — it is `δ_j·n/(o−½−δ_j)`, which changes with the branch. For
example at o=3 it is `2n/3` on the δ=1 branch but `4n` on the δ=2 branch. And only the interiors that
fall in-range enter the `min`, next to the (several) crossovers. The clean single-`n/(o−3/2)` form is
exactly the special case `δ ∈ {0,1}`.

### Where the two exponents come from, and why the interior optimum is δ·n/(o−½−δ)

The objective's exponent `o − ½` and the interior optimum's denominator `o − ½ − δ_j` look
suspiciously alike; they should, because the second is the first minus `δ_j`, and both come from one
place.

**The objective exponent `o − ½`.** To cover a fixed span of simulated time, `#batches = T / Δt_batch`,
and one batch advances `Δt_batch = E[ℓ] / P_total = E[ℓ] / (kmax · C(N,o))`. With `E[ℓ] = c√N` and
`C(N,o) ≈ N^o / o!`,

```
#batches  ∝  kmax · C(N,o) / E[ℓ]  ∝  kmax · N^o / √N  =  kmax · N^(o − 1/2).
```

So the exponent splits cleanly: the **o** is the reactant-set combinatorics `C(N,o) ~ N^o` (the clock),
and the **−½** is the collision length `E[ℓ] ~ √N` sitting in the denominator. Neither half has anything
to do with a particular reaction.

**The filler-dilution exponent δ_j.** On the branch where reaction *j* sets kmax,
`kmax = A_j / ff(k0, δ_j) ≈ A_j / k0^{δ_j}` — reaction *j* carries `δ_j` fillers, and its adjusted rate
is diluted by the K-falling-factorial. So on that branch the objective is

```
f(k0)  =  A_j · (n + k0)^{o − 1/2} / k0^{δ_j}.
```

**The optimum.** This is an instance of the elementary fact

```
minimize  (n + k)^p / k^q   over k > 0    ⟹    k* = q·n / (p − q)      (for p > q > 0),
```

whose first-order condition just equates the numerator's logarithmic growth rate `p/(n+k)` with the
denominator's `q/k`. With `p = o − ½` (the batch-cost exponent) and `q = δ_j` (the filler dilution),

```
k0*  =  δ_j · n / ( (o − 1/2) − δ_j ).
```

So the `o − ½ − δ_j` is exactly `p − q`: the *net* power of k0 in the objective, after the numerator's
growth and the denominator's dilution are combined. It must be positive (`o − ½ > δ_j`) for a finite
optimum to exist; otherwise dilution always wins, `f` falls monotonically, and you would add filler
without bound (the degenerate case we avoid by only padding up to order o and taking o ≥ 2).

**Order-2 collapse.** The only padded reactions have δ=1, so `kmax = max(C_padded/k0, C_flat)` with
`C_padded = max_{order 1} A_j`; the single interior optimum is `k0 = 1·n/(2−½−1) = n/½ = 2n`
(equivalently `2n/(2o−3)` at o=2) and the single crossover is `C_padded/C_flat`, giving
`k0* = min(2n, crossover)`.

`k_reset_target` implements the δ=1 case: `min(n/(o−3/2), crossover)`, where `crossover` is the
δ=1-to-flat handoff `A_padded/C_flat` (`crossover_k_count`). That is **exact for o = 2** — there
`δ ∈ {0,1}` is all there is, one padded branch, one crossover, interior `2n`. For o ≥ 3 it is an
approximation: a δ ≥ 2 branch could set `kmax` near the optimum, and getting it exactly right would need
the full candidate search above (both the per-branch interiors `δ_j·n/(o−½−δ_j)` and all the
crossovers). No benchmark CRN is affected — all are order 2.

## Comparison to the old policy (K = n)

The old heuristic reset K to n whenever K drifted out of `[n/2, 2n]`. Relative to k0* = min(2n,
crossover):

- when `n ≫ crossover` (LV at its population peaks), `K = n` *overshoots* the optimum, inflating N and
  wasting batches — this is where the new policy wins;
- it also makes the non-passive fraction wobble and jump (it depends on the drifting, periodically
  reset K), which is the artifact behind the "the fraction differs at near-identical configurations"
  observation. Pinning K at the constant crossover removes the drift and the jumps.

## Empirical validation

All measured with the actual sampled E[ℓ] (`sample_collision`) and exact p; at a frozen config,
`argmax_K E[ℓ]·p` is the batch-optimal K.

- **Batch counts (pure batch, n = 1e5):** LV **1.4–1.5× fewer batches** under min(2n, crossover) vs
  K = n, with a flat non-passive fraction ≈ 0.62 (vs old's 0.28–0.63 with reset jumps). Oregonator
  ≈ 1.03× (its crossover ≈ n, so K = n was already near-optimal).
- **Volume scaling (n fixed at 1e5, v varied):** the optimum doubles when v doubles and halves when v
  halves — confirming k0* ∝ v, not n. crossover = 1.00·v (LV) and 0.52·v (Oregonator) across
  v ∈ {n/2, n, 2n}.
- **Rössler, both regimes:** argmax E[ℓ]·p = 186k ≈ 2n at the IC (n = 1e5, so 2n < crossover = 3e6),
  and = 3.13e6 ≈ crossover at an evolved config (n = 6.1e6, so crossover < 2n) — the formula picks the
  right branch in each.

## Implementation notes

- Target computed in `BatchSimulator::k_reset_target` (`src/simulator_crn.rs`); `reset_k_count` moves K
  toward it. It is recomputed and checked before **every** batch (cheap — the crossover is
  config-independent and cached at construction in `crossover_k0`). The band rule in `run()` rebuilds
  the transition arrays only when K has drifted more than a factor `K_RESET_BAND_FACTOR` (= 1.1) from
  the target, so K tracks the `2n` branch to within ~10% without rebuilding on every reaction, and once
  it reaches the config-independent crossover it stops firing.
- **Always dynamic.** Whether a run's population moves enough to need re-tuning K is undecidable in
  general (CRNs can simulate Turing machines), so there is no separate "static vs dynamic" mode — the
  target is always recomputed and the band always applies. For crossover-binding CRNs (LV, Oregonator)
  the target is constant, so the band fires once and never again; for CRNs whose population moves it
  tracks `2n` (Rössler upward past the crossover; the Shrinking CRN `A→∅, B→∅, 2A→2B` downward, where a
  frozen crossover-sized K would be ~7.5× slower).
- **The non-passive-fraction "jumps"** seen when plotting are these rebuilds: between resets K is fixed
  while n moves, so the fraction drifts, then snaps back to the optimum at each reset. A smaller
  `K_RESET_BAND_FACTOR` makes the jumps smaller and more frequent; removing them entirely would require
  rebuilding every batch.
- `k0_manual_multiplier > 0` overrides the target with `round(mult·n)` for K sweeps (test-only).
