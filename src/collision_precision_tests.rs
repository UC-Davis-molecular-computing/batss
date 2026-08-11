//! Regression tests for the collision sampler's f64 -> f128 escalation guard (issue #15).
//!
//! The guard decides whether the binary search in `sample_collision_fast_f128` can trust its f64
//! arithmetic. If its error bound is too small, the search proceeds on rounding noise, which shows
//! up either as a spurious assertion failure (the reported panic) or, more often, as a silently
//! wrong batch size. These tests pin the bound against the error actually committed by
//! `collision_rhs` -- the function whose f64 `ln_gamma` calls produced the panic.
//!
//! Run with: `cargo test --lib --release`

use super::*;

/// The escalation bound as the code computed it before issue #15: scaled by the *first* `ln_gamma`
/// term only, with a per-call budget of 2.5 ULP.
fn old_bound(n: u64, r: u64, t_mid: u64, o: usize) -> f64 {
    let first_term = ln_gamma((n - r - (t_mid * o as u64)) as f64 + 1.0);
    first_term * 2.5 * o as f64 * f64::EPSILON
}

/// The escalation bound as the code computes it now: scaled by the summed magnitude of every
/// f64-computed term, with a per-call budget of `COLLISION_F64_ULP_BUDGET`.
fn new_bound(term_magnitude_sum: f64) -> f64 {
    term_magnitude_sum * COLLISION_F64_ULP_BUDGET * f64::EPSILON
}

/// The error `collision_rhs` actually commits in f64, measured against its own f128 result,
/// together with the magnitude sum it reports.
fn measure(n: u64, r: u64, t_mid: u64, o: usize, g: usize) -> (f64, f64) {
    let ln_g = if g > 0 { ln_f128(g as f128) } else { f128::NAN };
    let (rhs_f64, magnitude_sum, _) = collision_rhs(n, r, t_mid, o, g, ln_g, false);
    let (rhs_hp, hp_sum, _) = collision_rhs(n, r, t_mid, o, g, ln_g, true);
    // Both paths sum the same terms, so their reported magnitudes must agree closely; the bound is
    // only meaningful if the f128 path reports one too (it is what makes the loud
    // out-of-precision check possible).
    assert!(
        (hp_sum - magnitude_sum).abs() <= 1e-6 * magnitude_sum.max(1.0),
        "magnitude sums disagree between precisions: f64 {magnitude_sum:e} vs f128 {hp_sum:e}"
    );
    let error = (rhs_f64 - rhs_hp).abs() as f64;
    (error, magnitude_sum)
}

/// A grid spanning the shapes the benchmark CRNs actually use: (o, g) = (2,1) for Dimerization,
/// Lotka-Volterra and Rossler; (2,2) Oregonator; (3,1) the order-3 Brusselator; (2,0) and (3,0)
/// the Shrinking family. Populations bracket the one the panic was logged at.
fn grid() -> Vec<(u64, u64, u64, usize, usize)> {
    let mut cases = Vec::new();
    for &(o, g) in &[(2usize, 1usize), (2, 2), (3, 1), (3, 2), (2, 0), (3, 0)] {
        for &n in &[
            1_000_003u64,
            9_999_991,
            23_394_833,
            33_606_715,
            33_606_716,
            47_000_011,
            101_000_003,
        ] {
            for &t_mid in &[2u64, 3, 5, 17, 1021, 5797] {
                if n > t_mid * o as u64 + 8 {
                    cases.push((n, 0, t_mid, o, g));
                }
            }
        }
    }
    cases
}

/// THE REGRESSION TEST. The guard is sound only if its bound is at least the error actually
/// committed. Any case where it is not is a state in which the binary search can be steered by
/// rounding noise, with no escalation and no panic.
#[test]
fn escalation_bound_covers_the_f64_error() {
    let mut worst = (0.0f64, (0u64, 0u64, 0u64, 0usize, 0usize));
    for case in grid() {
        let (n, r, t_mid, o, g) = case;
        let (error, magnitude_sum) = measure(n, r, t_mid, o, g);
        let bound = new_bound(magnitude_sum);
        let ratio = error / bound;
        if ratio > worst.0 {
            worst = (ratio, case);
        }
        assert!(
            error <= bound,
            "f64 error exceeds the escalation bound at n={n}, r={r}, t_mid={t_mid}, o={o}, g={g}: \
             error {error:e} > bound {bound:e} (magnitude sum {magnitude_sum:e}). \
             The binary search would proceed on rounding noise here."
        );
    }
    println!("worst error/bound ratio on the grid: {:.4} at {:?}", worst.0, worst.1);
}

/// Records what the mis-scaling does and does not cost at `r = 0`, which is the only value the
/// engine itself ever passes.
///
/// The pre-fix bound is under-scaled by `(1 + o/g) / o` -- a factor of 1.5 at (o, g) = (2, 1) -- but
/// on this grid that headroom is never actually consumed: the measured error stays below even the
/// old bound. So the mis-scaling alone does not explain the panic recorded in issue #15, and this
/// test exists to keep that honest. What did explain it is the guard/assertion mismatch, pinned by
/// `assertion_quantity_is_below_the_noise_floor_at_small_t`.
#[test]
fn pre_fix_bound_held_at_r_zero_despite_being_under_scaled() {
    let mut worst_old = 0.0f64;
    for (n, r, t_mid, o, g) in grid() {
        let (error, magnitude_sum) = measure(n, r, t_mid, o, g);
        worst_old = worst_old.max(error / old_bound(n, r, t_mid, o));
        // The under-scaling itself is structural and always present when g > 0.
        if g > 0 {
            let first_term = ln_gamma((n - r - (t_mid * o as u64)) as f64 + 1.0);
            let scaling_shortfall = (magnitude_sum / first_term) / o as f64;
            assert!(
                scaling_shortfall > 0.4,
                "unexpected scaling ratio {scaling_shortfall} at n={n}, t_mid={t_mid}, o={o}, g={g}"
            );
        }
    }
    assert!(
        worst_old < 1.0,
        "expected the pre-fix bound to hold at r = 0 on this grid; worst error/old_bound = {worst_old}"
    );
    println!("worst error/old_bound at r = 0: {worst_old:.4} (under 1.0, so the panic came from elsewhere)");
}

/// The magnitude sum must count every f64 term, not just the first. For g = 1 all o+1 gamma
/// arguments are ~n, so the sum is (1 + o/g) = o+1 times the first term; scaling by the first term
/// alone -- the pre-fix behaviour -- under-counts by exactly that factor.
#[test]
fn magnitude_sum_counts_every_f64_term() {
    for &(o, g, expected_ratio) in &[(2usize, 1usize, 3.0f64), (3, 1, 4.0), (2, 2, 1.96), (3, 2, 2.44)] {
        let n = 33_606_715u64;
        let (_, magnitude_sum) = measure(n, 0, 2, o, g);
        let first_term = ln_gamma((n - (2 * o as u64)) as f64 + 1.0);
        let ratio = magnitude_sum / first_term;
        assert!(
            (ratio - expected_ratio).abs() < 0.05,
            "o={o}, g={g}: magnitude sum / first term = {ratio}, expected ~{expected_ratio}"
        );
    }
}

/// The pre-fix scaling degenerates completely as r approaches n: the first term is
/// ln_gamma(n - r - t*o + 1), which tends to ln_gamma(1) = 0, so the old bound collapses toward
/// zero while the remaining terms stay of order n ln n and keep their error.
///
/// This is latent rather than live. The engine always calls with r = 0, and `sample_collision`'s
/// r > 0 path is separately broken for reasons that have nothing to do with precision (it panics
/// with "Binary search should never return t_lo = 0" at essentially every u, both before and after
/// this fix). It matters for the r_i/u_i lookup-table path the function's doc comment anticipates:
/// that path would call with r > 0, and would need a bound that does not collapse.
#[test]
fn guard_holds_when_r_approaches_n() {
    let n = 33_606_715u64;
    let mut collapses = 0;
    for &(o, g) in &[(2usize, 1usize), (3, 1), (2, 2)] {
        for &r in &[n / 2, n - 1_000_000, n - 1000, n - 100] {
            let t_mid = 2u64;
            if r + t_mid * o as u64 + 8 >= n {
                continue;
            }
            let (error, magnitude_sum) = measure(n, r, t_mid, o, g);
            let bound = new_bound(magnitude_sum);
            assert!(
                error <= bound,
                "f64 error exceeds the bound at r={r} (n={n}, o={o}, g={g}): \
                 error {error:e} > bound {bound:e}"
            );
            if old_bound(n, r, t_mid, o) < error {
                collapses += 1;
                println!(
                    "old bound collapsed at r={r}, o={o}, g={g}: {:e} < error {error:e}",
                    old_bound(n, r, t_mid, o)
                );
            }
        }
    }
    assert!(
        collapses > 0,
        "expected the pre-fix bound to collapse below the real error for some r near n"
    );
}

/// The exact state the logged panic came from, recovered by inverting the bound it printed:
/// `last_lngamma_value = 6.092965495782715e-7 / (2.5 * 2 * EPSILON) = 548805542.7277665`, and
/// `ln_gamma(33_606_712)` equals that, so `n_including_extra_species = 33_606_715` with r = 0,
/// t_mid = 2, o = 2, g = 1. (The panic message's `n = 23394833` is `self.n`, the *unpadded*
/// population, which is why searching at that value never reproduced anything.)
#[test]
fn logged_panic_state_is_reconstructed_and_guarded() {
    let (n, r, t_mid, o, g) = (33_606_715u64, 0u64, 2u64, 2usize, 1usize);
    let first_term = ln_gamma((n - r - (t_mid * o as u64)) as f64 + 1.0);
    assert!(
        (first_term - 548_805_542.727_766_5).abs() < 1e-6,
        "state reconstruction is wrong: first ln_gamma term is {first_term}, expected the \
         548805542.7277665 implied by the logged bound"
    );
    let logged_bound = 6.092_965_495_782_715e-7;
    assert!(
        (old_bound(n, r, t_mid, o) - logged_bound).abs() < 1e-13,
        "the pre-fix formula should reproduce the bound printed in the panic, got {}",
        old_bound(n, r, t_mid, o)
    );

    let (error, magnitude_sum) = measure(n, r, t_mid, o, g);
    let bound = new_bound(magnitude_sum);
    assert!(
        error <= bound,
        "the logged state is still unguarded: error {error:e} > bound {bound:e}"
    );
    println!(
        "logged state: first term {first_term:e}, magnitude sum {magnitude_sum:e} (ratio {:.4}), \
         f64 error {error:e}, old bound {logged_bound:e}, new bound {bound:e}",
        magnitude_sum / first_term
    );
}

/// The escalation must also cover the quantity the *assertion* tests. The guard protects
/// `|lhs - rhs|` (which steers the search), but the assertion checks `lhs + ln_u <= rhs`, i.e.
/// `|lhs + ln_u - rhs|` = the computed `-ln P(l >= t_mid)`. These coincide only when `ln_u` is near
/// zero. At a large population and small t_mid, P(l >= t_mid) is itself below the f64 noise floor,
/// so the assertion's quantity is unresolvable while the guard's is comfortably large.
#[test]
fn assertion_quantity_is_below_the_noise_floor_at_small_t() {
    let (n, o, g) = (33_606_715u64, 2usize, 1usize);
    let ln_g = ln_f128(g as f128);
    // -ln P(l >= t) computed exactly, via the high-precision path, relative to t = 1.
    let (base, _, _) = collision_rhs(n, 0, 1, o, g, ln_g, true);
    let (_, magnitude_sum) = measure(n, 0, 2, o, g);
    let bound = new_bound(magnitude_sum);
    for t_mid in 2u64..=6 {
        let (rhs_hp, _, _) = collision_rhs(n, 0, t_mid, o, g, ln_g, true);
        let neg_ln_p = (base - rhs_hp).abs() as f64;
        if t_mid <= 4 {
            assert!(
                neg_ln_p < bound,
                "expected -ln P(l >= {t_mid}) = {neg_ln_p:e} to be below the f64 noise floor \
                 {bound:e}; if it is not, the assertion is resolvable here and this test's premise \
                 no longer holds"
            );
        }
        println!("t_mid={t_mid}: -ln P(l >= t) = {neg_ln_p:e}, noise floor {bound:e}");
    }
}
