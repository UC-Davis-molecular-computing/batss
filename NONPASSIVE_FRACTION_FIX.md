# Non-passive reaction fraction: the drifting-K bug and its fix

This documents a bug in the **non-passive reaction fraction** that `Simulation` records for plotting
(`Simulation.non_passive_fractions`), why it made the plotted fraction disagree with itself at equal
configurations, and the fix. All simulator code is in
[`src/simulator_crn.rs`](src/simulator_crn.rs); the recording happens in
[`python/batss/simulation.py`](python/batss/simulation.py).

## What the quantity is supposed to be

When `plot_trajectory` draws a trajectory it overlays, on a second y-axis, the *fraction of reactions
that are non-passive* — the fraction that actually change the configuration of the original CRN, as
opposed to the passive self-loops the batching algorithm's modified CRN introduces. It is meant to be
a **function of the configuration alone**: at a given vector of species counts there is one right
answer, so two snapshots with the same counts must show the same fraction. That is exactly what the
field docstring promised ("Because it depends only on the current configuration…").

## The symptom

On Lotka–Volterra (`R + F → 2F`, `R → 2R`, `F → ∅`, all rate 1, n = 10⁵) the populations trace a
clean limit cycle, so `(R, F)` returns to (nearly) the same value once per period. The recorded
fraction, however, did **not** repeat: it jumped between periods and differed markedly at times when
`R` and `F` were nearly identical. Concretely, two snapshots from the same run:

| snapshot | R | F | recorded fraction | filler count K |
|---:|---:|---:|---:|---:|
| 15 | 70,244 | 38,857 | **0.612** | 102,033 |
| 183 | 70,060 | 38,750 | **0.140** | 331,404 |

Same populations, a 4× difference in the "fraction." The last column is the tell.

## Root cause: the denominator includes the drifting filler species K

The fraction is computed by `non_passive_reaction_probability`
([`simulator_crn.rs:996`](src/simulator_crn.rs#L996)) as a ratio of two propensities:

```
fraction = total_propensity(non-passive) / total_propensity(including passive)
```

- **Numerator** — `calculate_total_propensity(false)`
  ([`simulator_crn.rs:1636`](src/simulator_crn.rs#L1636)) sums the real reactions' propensities from
  the real species counts in `self.urn`. It skips the filler species K, so it is genuinely a function
  of the configuration. **This part is correct.**
- **Denominator** — `calculate_total_propensity(true)` returns
  `continuous_time_correction_factor * binomial(n_including_extra_species, o)`, i.e. the total event
  rate of the *modified* CRN, where `o` is the maximum reaction order. Here
  `n_including_extra_species = (real species) + K + W`, and it **includes the filler species K**.

K is not a property of the configuration — it is the batching algorithm's tuning knob (the number of
"clock"/filler agents that pad the population so lower-order reactions are rated correctly). It
**drifts** over a run:

- `reset_k_count` ([`simulator_crn.rs:1672`](src/simulator_crn.rs#L1672)) targets `K = n` (the real
  population), but `run` only calls it when K strays past `K_COUNT_RATIO_THRESHOLD = 0.5`
  ([`simulator_crn.rs:1057`](src/simulator_crn.rs#L1057)) — so between resets K wanders freely.
- During a Gillespie phase K is held fixed (`sync_urn_from_gillespie` preserves it) while the real
  counts move, so the ratio `K / n` can leave the batch-mode band entirely.

In the LV run above, `K / n_real` ranged from **0.37 to 3.48**. Because the denominator's
`binomial(real + K, o)` grows with K, the same configuration divides by a different number depending
on the run's recent history, and the reported fraction changes. W is 0 at each recorded snapshot
(`recycle_waste` runs after every batch), so K is the whole story.

Note that this K-dependence is **correct** for the other caller of the method: `run` uses it in the
batch/Gillespie switching heuristic ([`simulator_crn.rs:718`](src/simulator_crn.rs#L718)), where the
question is how efficient the *actual* next batch will be — and that genuinely depends on the current
K. The bug was only in reusing this actual-K value as if it were configuration-only for plotting.

## The fix: evaluate at the canonical filler count K = n

Add a sibling method `non_passive_reaction_probability_canonical`
([`simulator_crn.rs:1017`](src/simulator_crn.rs#L1017)) that keeps the (already correct) numerator but
fixes the denominator's population at the **canonical** value the algorithm targets, `K = n`, giving a
total population of `2n`:

```rust
pub fn non_passive_reaction_probability_canonical(&self) -> f64 {
    let real_propensity = self.calculate_total_propensity(false);   // config-only numerator
    let total_including_passive = self.get_exponential_rate(2 * self.n);  // canonical denominator
    if total_including_passive == 0.0 { return 0.0; }
    real_propensity / total_including_passive
}
```

`get_exponential_rate(pop_size)` ([`simulator_crn.rs:1949`](src/simulator_crn.rs#L1949)) is exactly
`continuous_time_correction_factor * binomial(pop_size, o)` — the same denominator formula
`calculate_total_propensity(true)` uses, just evaluated at the configuration-determined population
`2n` instead of the drifting `n_including_extra_species`. Both numerator and denominator are now
functions of the configuration alone, so equal configurations give equal fractions.

`Simulation.add_config` ([`simulation.py:865`](python/batss/simulation.py#L865)) now records this
canonical value; the switching heuristic in `run` is untouched and keeps using the actual-K
`non_passive_reaction_probability`.

### Why K = n rather than K = 0

`K = 0` (sample o agents from just the real population) is also configuration-only, but it describes a
different, filler-free CRN than the one the batching algorithm actually runs, and it would shift every
plotted value upward. `K = n` is the count `reset_k_count` restores, so it is the modified CRN the
algorithm is designed to operate in; it also **agrees with the previously recorded values at the
snapshots where K had not yet drifted**, so the fix removes the artifact without changing the meaning
of the curve.

## Verification

Re-running the LV trajectory (n = 10⁵, t = 20, seed 1) after the fix:

- The paired discrepancy is gone: snapshots 15 and 183 now report **0.573** and **0.574** (vs. 0.612
  and 0.140 before).
- The worst spread in the recorded fraction among snapshots that share a configuration dropped from
  **~0.4** (before) to **~0.002** (after) — the residual being genuine small differences in `(R, F)`
  within a bin, not K drift.
- The overlaid fraction is now smooth and periodic, matching the periodicity of `R` and `F` (see
  [`examples/data/nonpassive_fix_before_after.png`](examples/data/nonpassive_fix_before_after.png):
  left = before, stepped and non-repeating; right = after, smooth and repeating).

## Expected vs. actual, and why Lotka–Volterra swings more than the paper's Figure 4

The recorded fraction is an **expected** value — a propensity ratio evaluated at the snapshot. The
batss paper (arXiv:2508.04079, Figure 4, top) instead plots the **actual** fraction, *counting* the
non-passive reactions the batches materialized, and reports it staying "around half" for
Lotka–Volterra. That is a different measurement, so it is worth checking the expected value against a
direct count and understanding why the current benchmark swings from ~0.27 to ~0.63 rather than
hovering near 0.5.

**The expected calculation is faithful to what the algorithm actually samples.** Temporarily counting
non-passive vs. total reactions inside `batch_step` (with Gillespie switching disabled, so it is pure
batch mode), the counted fraction tracks the propensity-based `non_passive_reaction_probability`
(evaluated at the *actual* current K) to an RMS of **0.015** over the LV run — they lie on top of each
other (see [`examples/data/actual_vs_expected.png`](examples/data/actual_vs_expected.png)). So the
fraction is not being miscomputed; the propensity ratio is exactly the sampling probability. That
plot also shows the actual/actual-K curve carries **step discontinuities** at each K reset — the very
artifact the canonical fix removes — while the canonical curve is its smooth, configuration-only
envelope.

**The size of the swing is set by the trajectory, not the measurement.** The non-passive fraction is
essentially a function of the population `n_real = R + F`: with the canonical denominator
`binomial(2·n_real, 2)` growing faster than the real propensity, the fraction falls monotonically as
`n_real` grows (from ~0.63 at `n_real = n` to ~0.27 at `n_real ≈ 3.5n`). Plotting the fraction against
`n_real` for two different LV orbits, both lie on the same curve; a large orbit just sweeps more of it
(see [`examples/data/amplitude_test.png`](examples/data/amplitude_test.png)). The current benchmark
starts at `R = F = n/2`, far from the LV fixed point at `R = F = n` (with `volume = n`), so it traces a
**large** orbit — `n_real` sweeps ~3.5× (matching the paper's own Figure 1, where LV counts span
0–2n) — and the fraction swings the full 0.27–0.63. The paper's Figure 4 LV panel instead has a counts
axis of only 50000–100000, i.e. a much **narrower** orbit whose `n_real` barely moves, so its counted
fraction stays flat near ½. Same algorithm, same measurement; a smaller-amplitude trajectory.

**The K policy matches the paper.** Section 6.3.2 states "we set #K = n … when we reset #K, since this
gives … a Ω(1) constant fraction of sampled reactions [that] are non-passive" — exactly what
`reset_k_count` does. So the swing is not a K-policy regression; it is the honest fraction for a
wide-amplitude oscillator.

(The `batch_step` counters used for this check were temporary and have been reverted; the hot loop is
back to allocation-free. They can be re-added as an opt-in diagnostic if a permanent actual-count
measurement is wanted.)

## Files changed

- [`src/simulator_crn.rs`](src/simulator_crn.rs): added
  `non_passive_reaction_probability_canonical`; expanded the doc comment on
  `non_passive_reaction_probability` to spell out that its denominator is actual-K (correct for the
  heuristic, wrong for plotting). Rebuild the extension after editing
  (`CARGO_TARGET_DIR="$LOCALAPPDATA/batss-cargo-target" maturin develop --release`).
- [`python/batss/simulation.py`](python/batss/simulation.py): `add_config` records the canonical
  value; updated the `non_passive_fractions` field docstring.
- [`python/batss/batss_rust/batss_rust.pyi`](python/batss/batss_rust/batss_rust.pyi): declared the new
  method and corrected the description of the old one.
