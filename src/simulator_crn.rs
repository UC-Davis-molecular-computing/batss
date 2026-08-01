// use rug::ops::CompleteRound;
// use rug::Float;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::time::{Duration, Instant};
use std::vec;

use crate::flame;

use compensated_summation::KahanBabuskaNeumaier;
use numpy::PyArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
// use ndarray::prelude::*;
use ndarray::{ArrayD, Axis};
use pyo3::types::PyNone;

use num_integer::{binomial, Roots};
use numpy::PyReadonlyArray1;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::{Rng, RngCore};
use rand_distr::{Distribution, Exp, Gamma, StandardUniform};

use rebop::gillespie::{Gillespie, Rate};

use itertools::Itertools;

use crate::simulator_abstract::Simulator;

use crate::urn::Urn;
use crate::util::{
    binomial_as_f64, f128_to_decimal, ln_f128, ln_factorial, ln_gamma, ln_gamma_manual_high_precision,
    ln_gamma_small_rational, multinomial_sample,
};

type State = usize;
type RateConstant = f64;
type StateList = Vec<State>;
type Reaction = (StateList, StateList, RateConstant);
// A map from each state that appears to how many times that state appears in this set of reactants.

/// Remembers the CRN we started with, because we may need to recompute reaction
/// probabilities between batches if we change the count of K. This should also hopefully
/// make it easier to interface with the Simulator class from both python and rust.
/// This struct is named to emphasize that it stores the CRN *after* the uniform transformation
/// is applied (so all reactions should have equal order and equal generativity). Rate constants
/// are stored here *before* adjusting for the count of K.
pub struct UniformCRN {
    /// Reaction order.
    pub o: usize,
    /// Generativity.
    pub g: usize,
    /// Number of species, including K and W.
    pub q: usize,
    /// The specific State representing the special species K.
    pub k: State,
    /// The specific State representing the special species W.
    pub w: State,
    /// The CRN's reactions. If multiple reactions share the same reactants, they are stored in
    /// the same Reaction object, for ease of iterating over reactions.
    /// Reactions include K and W, but rate constants are as in the original CRN,
    /// not yet adjusted based on the count of K in the configuration being simulated.
    pub reactions: Vec<CombinedReactions>,
    /// The correction factor for running reactions in continuous time. The whole CRN is treated
    /// as having a total propensity equal to (n choose o) * continuous_time_correction_factor.
    pub continuous_time_correction_factor: f64,
}

/// A struct combining all reactions with the same reactants into a single place.
#[derive(Debug)]
pub struct CombinedReactions {
    pub reactants: StateList,
    pub products_and_rate_constants: Vec<(StateList, RateConstant)>,
}

impl UniformCRN {
    // Make sure that a set of reactions is uniform and the reactions valid, and combine reactions
    // that share the same set of reactants for easier iteration.
    fn verify_and_combine_reactions(reactions: Vec<Reaction>, k: State, w: State) -> UniformCRN {
        assert!(reactions.len() > 0, "Cannot run CRN with no reactions.");
        let first_reaction = &reactions[0];
        let o = first_reaction.0.len();
        let g = first_reaction.1.len() - o;
        let mut all_species_seen: HashSet<State> = HashSet::from([k, w]);
        let mut highest_species_seen = k.max(w);
        let mut collected_reactions: HashMap<StateList, Vec<(StateList, RateConstant)>> = HashMap::new();
        for reaction in reactions {
            assert!(
                reaction.0.len() == o,
                "All reactions must have the same number of inputs"
            );
            assert!(
                reaction.1.len() - reaction.0.len() == g,
                "All reactions must have the same number of outputs"
            );
            for reactant in &reaction.0 {
                all_species_seen.insert(*reactant);
                highest_species_seen = highest_species_seen.max(*reactant);
            }
            for product in &reaction.1 {
                all_species_seen.insert(*product);
                highest_species_seen = highest_species_seen.max(*product);
            }
            collected_reactions
                .entry(reaction.0)
                .or_default()
                .push((reaction.1, reaction.2));
        }
        let q = highest_species_seen + 1;
        assert!(
            q == all_species_seen.len(),
            "Species must be indexed using contiguous integers starting from 0"
        );
        let mut reactions_out: Vec<CombinedReactions> = collected_reactions
            .keys()
            .map(|x| CombinedReactions {
                reactants: x.to_vec(),
                products_and_rate_constants: collected_reactions[x].clone(),
            })
            .collect();
        // HashMap iteration order is randomized independently for every map. Keeping that order
        // here made otherwise-identical simulators register reactions with rebop in different
        // orders, so a fixed Gillespie seed could select a different reaction.
        reactions_out.sort_unstable_by(|left, right| left.reactants.cmp(&right.reactants));
        return UniformCRN {
            o,
            g,
            q,
            k,
            w,
            reactions: reactions_out,
            continuous_time_correction_factor: 1.0,
        };
    }
    /// Build or rebuild random_transitions, random_outputs, and transition_probabilities
    /// for a @BatchSimulator.
    /// We need to rebuild these tables because reaction propensities depend on the count of K,
    /// which we may want to change throughout the execution.
    /// Returns a tuple of these three objects in that order.
    fn construct_transition_arrays(&mut self, k_count: u64) -> (ArrayD<usize>, Vec<StateList>, Vec<f64>) {
        flame::start("construct_transition_arrays");
        // Iterate through reactions, adjusting rate constants to account for how many K
        // are being added, and for symmetry that results from the scheduler having different
        // orders it can pick, so that the adjusted CRN keeps the original dynamics.
        flame::start("construct_transition_arrays: first reaction loop");
        let max_total_adjusted_rate_constant = self.max_adjusted_rate_constant_at(k_count);
        self.continuous_time_correction_factor = max_total_adjusted_rate_constant;
        flame::end("construct_transition_arrays: first reaction loop");
        // random_transitions has o+1 dimensions, the first o of which have length q,
        // and the last of which has length 2.
        let mut shape_vec = vec![self.q; self.o];
        shape_vec.push(2);
        let mut random_transitions = ArrayD::<usize>::zeros(shape_vec);
        let mut random_outputs: Vec<Vec<State>> = Vec::new();
        let mut random_probabilities: Vec<f64> = Vec::new();
        // Add any active reactions. Passive reactions don't need any special handling.
        let mut cur_output_index = 0;
        flame::start("construct_transition_arrays: second reaction loop");
        for reaction in &self.reactions {
            // Add info from this reaction to all possible permutations of reactants.
            let reactants = &reaction.reactants;
            let symmetry_degree = Self::symmetry_degree(reactants);
            let k_count_correction_factor = self.k_count_correction_factor(reactants, k_count);
            let artificial_speedup_factor = k_count_correction_factor as f64 / symmetry_degree as f64;
            for output in &reaction.products_and_rate_constants {
                let probability = (output.1 / artificial_speedup_factor) / max_total_adjusted_rate_constant;
                random_outputs.push(output.0.clone());
                random_probabilities.push(probability);
                for permutation in reaction.reactants.iter().permutations(self.o).unique() {
                    let mut view = random_transitions.view_mut();
                    // This loop indexes into random_transitions.
                    for dimension in 0..self.o {
                        view = view.index_axis_move(Axis(0), *permutation[dimension]);
                    }
                    // Make sure that this is one-dimensional.
                    let mut inner_view = view.into_dimensionality::<ndarray::Ix1>().unwrap();
                    // Increment the number of possible outputs for these reactants.
                    inner_view[0] += 1;
                    inner_view[1] = cur_output_index;
                }
            }
            cur_output_index += reaction.products_and_rate_constants.len();
        }
        flame::end("construct_transition_arrays: second reaction loop");
        assert_eq!(
            random_outputs.len(),
            random_probabilities.len(),
            "random_outputs and transition_probabilities length mismatch"
        );
        flame::end("construct_transition_arrays");
        return (random_transitions, random_outputs, random_probabilities);
    }

    /// The largest total adjusted rate for any combined reactant set at an arbitrary K count.
    ///
    /// This is the scalar calculation performed before `construct_transition_arrays` fills its
    /// tables, factored out so the prospective switching heuristic can evaluate a future batch's K
    /// without rebuilding or allocating those tables. Rates for outputs sharing a reactant set are
    /// summed because that combined set owns one transition-table lane.
    fn max_adjusted_rate_constant_at(&self, k_count: u64) -> f64 {
        let mut max_total_adjusted_rate_constant: f64 = 0.0;
        for reaction in &self.reactions {
            let symmetry_degree = Self::symmetry_degree(&reaction.reactants);
            let k_count_correction_factor = self.k_count_correction_factor(&reaction.reactants, k_count);
            let total_rate_constant: f64 = reaction.products_and_rate_constants.iter().map(|output| output.1).sum();
            let artificial_speedup_factor = k_count_correction_factor as f64 / symmetry_degree as f64;
            let total_adjusted_rate_constant = total_rate_constant / artificial_speedup_factor;
            max_total_adjusted_rate_constant = max_total_adjusted_rate_constant.max(total_adjusted_rate_constant);
        }
        max_total_adjusted_rate_constant
    }

    fn k_count_correction_factor(&self, reactants: &Vec<State>, k_count: u64) -> u64 {
        let mut correction_factor = 1;
        let k_multiplicity = reactants.iter().filter(|&s| *s == self.k).count();
        // We've artificially sped up this reaction by |K| * (|K| - 1) * ... * (|K| - k_multiplicity + 1).
        // This loop undoes that artificial speedup.
        for i in 0..k_multiplicity {
            correction_factor *= k_count - i as u64;
        }
        return correction_factor;
    }

    /// The count of K at which the two branches of kmax(k0) meet. As K grows, a padded reaction's
    /// adjusted rate (`rate * symmetry / k0`, for a reaction carrying one K) falls, while a genuine
    /// order-o reaction's adjusted rate (`rate * symmetry`) is constant. `kmax` rides the falling
    /// branch, then flattens onto that constant floor; the handoff is the crossover, returned here as
    /// `A_pad / C_flat` where
    ///   C_flat = max over reactions with no K of (rate * symmetry), and
    ///   A_pad  = max over reactions with exactly one K of (rate * symmetry).
    /// It depends only on the (volume-adjusted) rate constants and symmetry, not on the configuration.
    /// Returns +inf when there is no padded reaction or no order-o reaction, so the crossover does not
    /// bind. Assumes the padded branch is governed by order-(o-1) reactions (one K, delta0 = 1), which
    /// holds for o = 2 (all our benchmark CRNs); reactions carrying >= 2 K flatten at smaller K and do
    /// not govern the final handoff, so they are skipped.
    fn crossover_k_count(&self) -> f64 {
        let mut c_flat = 0.0_f64;
        let mut a_pad = 0.0_f64;
        for reaction in &self.reactions {
            let reactants = &reaction.reactants;
            let k_multiplicity = reactants.iter().filter(|&s| *s == self.k).count();
            let symmetry_degree = Self::symmetry_degree(reactants) as f64;
            let mut total_rate_constant = 0.0;
            for output in &reaction.products_and_rate_constants {
                total_rate_constant += output.1;
            }
            let numerator = total_rate_constant * symmetry_degree;
            match k_multiplicity {
                0 => c_flat = c_flat.max(numerator),
                1 => a_pad = a_pad.max(numerator),
                _ => {}
            }
        }
        if a_pad <= 0.0 || c_flat <= 0.0 {
            return f64::INFINITY;
        }
        a_pad / c_flat
    }

    /// Determine the degree of symmetry of a reaction, i.e., for any given ordering of its reactants,
    /// the number of reorderings that are redundant. Obtained as the product of the factorial of
    /// the count of each reactant.
    fn symmetry_degree(reactants: &Vec<State>) -> u64 {
        let mut factor = 1;
        // Reaction orders are small, so an O(o^2) scan is cheaper than allocating a HashMap here.
        // This matters because the prospective score calls the adjusted-rate helper in the hot loop.
        for (index, reactant) in reactants.iter().enumerate() {
            if reactants[..index].contains(reactant) {
                continue;
            }
            let frequency = reactants.iter().filter(|candidate| **candidate == *reactant).count() as u64;
            for i in 2..frequency + 1 {
                factor *= i;
            }
        }
        return factor;
    }
}

/// Which heuristic decides, before each iteration, whether the next reactions run faster in the
/// batching engine or in Gillespie mode. Settable from Python via `BatchSimulator.heuristic`.
pub const HEURISTIC_WALLCLOCK: u8 = 0;
pub const HEURISTIC_PROXY: u8 = 1;
pub const HEURISTIC_PROSPECTIVE: u8 = 2;

/// All state for the batch/Gillespie mode-switching heuristic, grouped into one struct so the
/// simulator doesn't carry a dozen more top-level fields. Holds the heuristic selector and its
/// tuning, the measured-throughput EMAs, the probe schedule, and read-only observability counters.
///
/// Exposed to Python as a read-only snapshot via `BatchSimulator.switch`; the two config
/// knobs are also get/set via `BatchSimulator.heuristic` and `.proxy_threshold`.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct SwitchState {
    // --- configuration (selects and tunes the heuristic) ---
    /// HEURISTIC_WALLCLOCK (0, default): switch by measured wall-clock per unit continuous time.
    /// HEURISTIC_PROXY (1): use only the original reaction-count rule (no timing, no probing).
    /// HEURISTIC_PROSPECTIVE (2): use the experimental deterministic prospective-batch score.
    #[pyo3(get)]
    pub heuristic: u8,
    /// Score threshold for HEURISTIC_PROXY and HEURISTIC_PROSPECTIVE: prefer Gillespie when the
    /// selected estimate of active reactions in the next batch is below this. Defaults to the
    /// reaction count (the original rule). Raise it to model batch mode's fixed per-batch overhead
    /// -- i.e. require more expected active reactions before batching is worth it.
    #[pyo3(get)]
    pub proxy_threshold: f64,

    // --- measured throughput, per mode ---
    // Exponentially-weighted moving averages of wall-clock seconds and continuous time advanced per
    // engine call. A mode's cost is their RATIO (wall_ema / dt_ema): a dt-weighted average, which is
    // the right figure of merit since we minimize wall-clock per unit continuous time. Averaging
    // elapsed/dt per call instead would wrongly weight tiny-dt calls equally with large-dt ones --
    // Gillespie's dt per call swings by orders of magnitude with the propensity, so that bias is
    // severe (it was the original measurement bug; see GILLESPIE_SWITCH_LOGIC.md).
    batch_wall_ema: f64,
    batch_dt_ema: f64,
    gillespie_wall_ema: f64,
    gillespie_dt_ema: f64,
    has_batch_est: bool,
    has_gillespie_est: bool,

    // --- probe schedule ---
    /// Run-loop iterations since we last probed the non-current mode.
    iters_since_probe: u64,
    /// Remaining iterations in an in-flight probe of the non-current mode (0 = not probing).
    probe_remaining: u64,

    // --- read-only observability (no effect on the decision) ---
    #[pyo3(get)]
    pub batch_wallclock_seconds: f64,
    #[pyo3(get)]
    pub gillespie_wallclock_seconds: f64,
    #[pyo3(get)]
    pub batch_continuous_time: f64,
    #[pyo3(get)]
    pub gillespie_continuous_time: f64,
    #[pyo3(get)]
    pub batch_calls: u64,
    #[pyo3(get)]
    pub gillespie_calls: u64,
    #[pyo3(get)]
    pub mode_switches: u64,
    #[pyo3(get)]
    pub switch_overhead_seconds: f64,
}

impl SwitchState {
    fn new(proxy_threshold: f64) -> Self {
        SwitchState {
            heuristic: HEURISTIC_WALLCLOCK,
            proxy_threshold,
            batch_wall_ema: 0.0,
            batch_dt_ema: 0.0,
            gillespie_wall_ema: 0.0,
            gillespie_dt_ema: 0.0,
            has_batch_est: false,
            has_gillespie_est: false,
            iters_since_probe: 0,
            probe_remaining: 0,
            batch_wallclock_seconds: 0.0,
            gillespie_wallclock_seconds: 0.0,
            batch_continuous_time: 0.0,
            gillespie_continuous_time: 0.0,
            batch_calls: 0,
            gillespie_calls: 0,
            mode_switches: 0,
            switch_overhead_seconds: 0.0,
        }
    }

    /// Clear all measured/probe/observability state back to a freshly-constructed value, keeping
    /// only the user-set configuration (`heuristic`, `proxy_threshold`). Used by
    /// `SimulatorCRN::reset` so a reused simulator makes the same switching decisions as a fresh one.
    fn reset(&mut self) {
        *self = SwitchState {
            heuristic: self.heuristic,
            proxy_threshold: self.proxy_threshold,
            ..SwitchState::new(self.proxy_threshold)
        };
    }

    /// A mode's estimated cost as wall-clock per unit continuous time: wall EMA / dt EMA (a
    /// dt-weighted average). `true` = Gillespie, `false` = batch. Meaningful only when `has_est`.
    fn wdt(&self, gillespie: bool) -> f64 {
        if gillespie {
            self.gillespie_wall_ema / self.gillespie_dt_ema
        } else {
            self.batch_wall_ema / self.batch_dt_ema
        }
    }

    /// Whether a mode's EMAs hold a real measurement yet (`true` = Gillespie).
    fn has_est(&self, gillespie: bool) -> bool {
        if gillespie {
            self.has_gillespie_est
        } else {
            self.has_batch_est
        }
    }

    /// Fold one engine call's wall-clock (`wall`) and continuous-time-advanced (`dt`, > 0) into a
    /// mode's EMAs (`true` = Gillespie).
    fn record(&mut self, gillespie: bool, wall: f64, dt: f64) {
        if gillespie {
            if self.has_gillespie_est {
                self.gillespie_wall_ema = WDT_EMA_ALPHA * wall + (1.0 - WDT_EMA_ALPHA) * self.gillespie_wall_ema;
                self.gillespie_dt_ema = WDT_EMA_ALPHA * dt + (1.0 - WDT_EMA_ALPHA) * self.gillespie_dt_ema;
            } else {
                self.gillespie_wall_ema = wall;
                self.gillespie_dt_ema = dt;
                self.has_gillespie_est = true;
            }
        } else if self.has_batch_est {
            self.batch_wall_ema = WDT_EMA_ALPHA * wall + (1.0 - WDT_EMA_ALPHA) * self.batch_wall_ema;
            self.batch_dt_ema = WDT_EMA_ALPHA * dt + (1.0 - WDT_EMA_ALPHA) * self.batch_dt_ema;
        } else {
            self.batch_wall_ema = wall;
            self.batch_dt_ema = dt;
            self.has_batch_est = true;
        }
    }
}

/// Timings and work counts from one experimental, frozen-state engine call.
///
/// Setup and state canonicalization are reported separately from steady engine work so an offline
/// model can fit the batch/Gillespie crossover without accidentally learning switch overhead.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct EngineCallBenchmark {
    #[pyo3(get)]
    pub gillespie: bool,
    #[pyo3(get)]
    pub preparation_seconds: f64,
    #[pyo3(get)]
    pub setup_seconds: f64,
    #[pyo3(get)]
    pub engine_seconds: f64,
    #[pyo3(get)]
    pub postprocess_seconds: f64,
    #[pyo3(get)]
    pub continuous_time_advanced: f64,
    #[pyo3(get)]
    pub total_reactions: u64,
    #[pyo3(get)]
    pub active_reactions: u64,
    #[pyo3(get)]
    pub k_rebuilt: bool,
}

#[pyclass(extends = Simulator)]
pub struct BatchSimulator {
    /// The CRN with a list of reactions, so we can recompute probabilities when the
    /// count of K is updated between batches.
    pub crn: UniformCRN,

    /// The population size (sum of values in urn.config).
    #[pyo3(get, set)]
    pub n_including_extra_species: u64,

    /// The population size of all species except k and w.
    #[pyo3(get, set)]
    pub n: u64,

    /// The amount of continuous time that has been simulated so far.
    #[pyo3(get, set)]
    pub continuous_time: f64,

    /// The total number of states (length of urn.config).
    /// Includes the auxiliary species K and W.
    pub q: usize,

    /// An (o + 1)-dimensional array. The first o dimensions represent reactants. After indexing through
    /// the first o dimensions, the last dimension always has size two, with elements (`num_outputs`, `first_idx`).
    /// `num_outputs` is the number of possible outputs if transition i,j --> ... is active,
    /// otherwise it is 0. `first_idx` gives the starting index to find
    /// the outputs in the array `self.random_outputs` if it is active.
    /// TODO: it would be much, much more readable if this was o-dimensional of pairs,
    /// rather than (o+1)-dimensional.
    /// #[pyo3(get, set)] // XXX: for testing
    pub random_transitions: ArrayD<usize>,

    /// A 1D array of tuples containing all outputs of random transitions,
    /// whose indexing information is contained in random_transitions.
    /// For example, if there are random transitions
    /// 3,4 --> 5,6,7 and 3,4 --> 7,7,8 and 3,4 --> 3,2,1, then
    /// `random_transitions[3][4] = (3, first_idx)` for some `first_idx`, and
    /// `random_outputs[first_idx]   = (5,6,7)`,
    /// `random_outputs[first_idx+1] = (7,7,8)`, and
    /// `random_outputs[first_idx+2] = (3,2,1)`.
    /// TODO: combine this with transition_probabilities into a single structure, since
    /// they should only ever be iterated through together.
    #[pyo3(get, set)] // XXX: for testing
    pub random_outputs: Vec<StateList>,

    /// An array containing all random transition probabilities,
    /// whose indexing matches random_outputs.
    /// May add up to less than 1 for a given reaction, in which case the remainder is assumed passive.
    #[pyo3(get, set)] // XXX: for testing
    pub transition_probabilities: Vec<f64>,

    /// The maximum number of random outputs from any random transition.
    pub random_depth: usize,

    /// A pseudorandom number generator.
    rng: SmallRng,

    /// An :any:`Urn` object that stores the configuration (as urn.config) and has methods for sampling.
    /// This is the equivalent of C in the pseudocode for the batching algorithm in the
    /// original Berenbrink et al. paper.
    urn: Urn,

    /// An additional :any:`Urn` where agents are stored that have been
    /// updated during a batch. Called `C'` in the pseudocode for the batching algorithm.
    #[allow(dead_code)]
    updated_counts: Urn,

    /// Struct which stores the result of hypergeometric sampling.
    array_sums: NDBatchResult,

    /// Vector holding multinomial samples when doing randomized transitions.
    m: Vec<u64>,

    /// A boolean determining if we are currently doing Gillespie steps.
    #[pyo3(get, set)]
    pub do_gillespie: bool,

    /// A boolean tracking whether we just started doing Gillespie steps.
    pub just_started_gillespie: bool,

    /// A boolean tracking whether we just stopped doing Gillespie steps.
    pub just_finished_gillespie: bool,

    /// We use rebop to run the Gillespie algorithm.
    gillespie: Option<Gillespie>,

    /// A boolean determining if the configuration is silent (all interactions are passive).
    #[pyo3(get, set)]
    pub silent: bool,

    /// All batch/Gillespie mode-switching state (heuristic selector + tuning, measurement EMAs,
    /// probe schedule, and observability counters), grouped into one struct. See `SwitchState`.
    switch: SwitchState,

    /// A module containing code for calling python-implemented collision sampling.
    pub python_module: Py<PyModule>,

    /// The crossover K count `C_padded / C_flat`, cached at construction: it is config-independent
    /// (rate constants and volume only), so `k_reset_target` reads it instead of recomputing (which
    /// would allocate) on every batch. See `UniformCRN::crossover_k_count`.
    pub crossover_k0: f64,

    /// Experiment override for sweeping K: if > 0, `reset_k_count` targets `round(k0_manual_multiplier
    /// * n)` instead of the optimal `min(2n, crossover)`. Used to empirically locate the batch-count
    /// optimum over K. 0 (default) disables the override.
    #[pyo3(get, set)]
    pub k0_manual_multiplier: f64,

    /// Number of K resets performed in run()'s batch branch (observability; used to compare policies).
    #[pyo3(get)]
    pub k_resets: u64,

    /// How a Gillespie block decides when to stop (issue #13 experiment).
    ///
    /// `false` (default) is the historical behaviour: convert the target reaction count into a
    /// duration using the total propensity *measured on entry*, then run rebop until that time.
    /// That conversion assumes the propensity is constant across the block, so it overshoots the
    /// intended count whenever the propensity rises and undershoots when it falls.
    ///
    /// `true` budgets the block directly in reactions via
    /// `Gillespie::advance_until_or_reactions`, which stops after exactly `sqrt(n)` reactions or at
    /// `t_max`, whichever comes first.
    #[pyo3(get, set)]
    pub gillespie_block_by_reactions: bool,

    /// Total reactions executed inside `gillespie_steps`, and the count those blocks aimed for.
    /// Their ratio is the realized over/undershoot of the block-sizing rule (observability only).
    #[pyo3(get)]
    pub gillespie_reactions_executed: u64,
    #[pyo3(get)]
    pub gillespie_reactions_targeted: u64,
}

#[pymethods]
impl BatchSimulator {
    /// Initializes the main data structures for BatchSimulator.
    /// We take numpy arrays as input because that's how n-dimensional arrays are represented in python.
    /// We convert those numpy arrays into rust ndarrray::ArrayD for storage.
    ///
    /// Args:
    ///     init_array: A length-q integer array of counts representing the initial configuration.
    ///     delta: A 2D q x q x 2 array representing the transition function. TODO remove if I can, might not be able to for consistency with other simulators.
    ///         Delta[i, j] gives contains the two output states.
    ///     random_transitions: A q^o x 2 array. That is, it has o+1 dimensions, all but the last have length q,
    ///         and the last dimension always has length two.
    ///         Entry [r, 0] is the number of possible outputs if transition on reactant set r is active,
    ///         otherwise it is 0. Entry [r, 1] gives the starting index to find the outputs in the array random_outputs if it is active.
    ///     random_outputs: A ? x (o + g) array containing all outputs of random transitions,
    ///         whose indexing information is contained in random_transitions.
    ///     transition_probabilities: A 1D length-? array containing all random transition probabilities,
    ///         whose indexing matches random_outputs.
    ///     seed (optional): An integer seed for the pseudorandom number generator.
    #[new]
    #[pyo3(signature = (init_config, _delta, _random_transitions, _random_outputs, _transition_probabilities, _transition_order, _gillespie, seed, reactions, k, w))]
    pub fn new(
        init_config: PyReadonlyArray1<u64>,
        _delta: Py<PyNone>,
        _random_transitions: Py<PyNone>,
        _random_outputs: Py<PyNone>,
        _transition_probabilities: Py<PyNone>,
        _transition_order: Py<PyNone>,
        _gillespie: Py<PyNone>,
        seed: Option<u64>,
        reactions: Vec<Reaction>,
        k: State,
        w: State,
    ) -> (Self, Simulator) {
        let crn = UniformCRN::verify_and_combine_reactions(reactions, k, w);
        let config = init_config.to_vec().unwrap();

        let n = config.iter().sum();
        let n_including_extra_species = n;
        let q = config.len() as State;

        // random_depth is the maximum number of outputs for any randomized transition
        let random_depth = crn
            .reactions
            .iter()
            .map(|x| x.products_and_rate_constants.len())
            .fold(0, |acc, x| acc.max(x));

        let continuous_time = 0.0;
        let rng = if let Some(s) = seed {
            SmallRng::seed_from_u64(s)
        } else {
            SmallRng::from_os_rng()
        };

        let updated_counts = Urn::new(vec![0; q], seed);
        let urn = Urn::new(config.clone(), seed);
        let array_sums = make_batch_result(crn.o, q);
        // The +1 here is to sample how many reactions are passive.
        let m = vec![0; random_depth + 1];
        let silent = false;
        let gillespie = None;
        let do_gillespie = false; // this changes during run
        let just_started_gillespie = false; // this changes during run
        let just_finished_gillespie = false; // this changes during run

        // next three fields are only used with Gillespie steps;
        // they will be set accordingly if we switch to Gillespie
        // let propensities = vec![0.0; reactions.len()];
        // let enabled_reactions = vec![0; reactions.len()];
        // let num_enabled_reactions = 0;

        // below here we give meaningless default values to the other fields and rely on
        // set_n_parameters and get_enabled_reactions to set them to the correct values
        // let gillespie_threshold = 0.0;
        // let coll_table = vec![vec![0; 1]; 1];
        // let coll_table_r_values = vec![0; 1];
        // let coll_table_u_values = vec![0.0; 1];

        // The following will be initialized during reset_k_count() below.
        let random_transitions = ArrayD::<usize>::zeros(Vec::new());
        let random_outputs = Vec::new();
        let transition_probabilities = Vec::new();
        let python_code = c_str!(include_str!("sample_coll.py"));
        let python_module = Python::attach(|py| {
            let module: Py<PyModule> =
                PyModule::from_code(py, python_code, c_str!("sample.py"), c_str!("sample_module"))
                    .unwrap()
                    .into();

            module
        });

        // The proxy heuristic's default threshold is the reaction count (the original rule).
        let switch = SwitchState::new(crn.reactions.len() as f64);

        let crossover_k0 = crn.crossover_k_count();
        let mut simulator = BatchSimulator {
            crn,
            n,
            n_including_extra_species,
            continuous_time,
            q,
            random_transitions,
            random_outputs,
            transition_probabilities,
            random_depth,
            rng,
            urn,
            updated_counts,
            array_sums,
            m,
            do_gillespie,
            gillespie,
            just_started_gillespie,
            just_finished_gillespie,
            silent,
            switch,
            python_module,
            crossover_k0,
            k0_manual_multiplier: 0.0,
            k_resets: 0,
            gillespie_block_by_reactions: false,
            gillespie_reactions_executed: 0,
            gillespie_reactions_targeted: 0,
            // gillespie_threshold,
            // coll_table,
            // coll_table_r_values,
            // coll_table_u_values,
        };
        simulator.reset_k_count();
        (simulator, Simulator::default())
    }

    #[getter]
    pub fn config(&self) -> Vec<u64> {
        self.urn.config.clone()
    }

    /// The batch/Gillespie switching heuristic in use: 0 = wall-clock (default), 1 = the simpler
    /// reaction-count proxy, 2 = the experimental deterministic prospective score. Settable from
    /// Python to A/B-test heuristics empirically.
    #[getter]
    pub fn heuristic_gillespie_switching(&self) -> u8 {
        self.switch.heuristic
    }

    #[setter]
    pub fn set_heuristic_gillespie_switching(&mut self, value: u8) {
        self.switch.heuristic = value;
    }

    /// Threshold for the proxy or prospective score (prefer Gillespie when the selected estimate
    /// of active reactions in the next batch falls below it). See :class:`SwitchState`.
    #[getter]
    pub fn proxy_threshold(&self) -> f64 {
        self.switch.proxy_threshold
    }

    #[setter]
    pub fn set_proxy_threshold(&mut self, value: f64) {
        self.switch.proxy_threshold = value;
    }

    /// Experimental deterministic estimate of active reactions in a batch prepared at the K value
    /// returned by `k_reset_target`. This is pure and does not rebuild transition arrays.
    pub fn prospective_batch_score(&self) -> f64 {
        let real_propensity = self.calculate_total_propensity(false);
        self.prospective_batch_score_from_real_propensity(real_propensity)
    }

    /// Benchmark one engine call from the current real-species configuration.
    ///
    /// This experimental API canonicalizes K before timing, then reports preparation, Gillespie
    /// construction, steady engine work, and postprocessing separately. It mutates the simulator;
    /// callers collecting a frozen-state sample should use a fresh or reset simulator per trial.
    #[pyo3(signature = (gillespie, gillespie_reactions=None))]
    pub fn benchmark_engine_call(
        &mut self,
        gillespie: bool,
        gillespie_reactions: Option<u64>,
    ) -> PyResult<EngineCallBenchmark> {
        if !(self.calculate_total_propensity(false) > 0.0) {
            return Err(PyValueError::new_err(
                "Cannot benchmark an engine call from a silent configuration.",
            ));
        }

        // Put both trials at the same prospective batch state. This work is reported but excluded
        // from steady engine time, because a fitted threshold should not absorb switch overhead.
        let preparation_start = Instant::now();
        self.recycle_waste();
        let prospective_k = self.k_reset_target();
        let mut k_rebuilt = false;
        if self.urn.config[self.crn.k] != prospective_k {
            self.reset_k_count();
            k_rebuilt = true;
        }
        let preparation_seconds = preparation_start.elapsed().as_secs_f64();
        let continuous_time_before = self.continuous_time;

        if gillespie {
            let setup_start = Instant::now();
            self.initialize_gillespie_config();
            let setup_seconds = setup_start.elapsed().as_secs_f64();

            let requested_reactions = gillespie_reactions.unwrap_or_else(|| self.n.sqrt().max(1)).max(1);
            let engine_start = Instant::now();
            let mut actual_reactions = 0;
            let gillespie_time;
            {
                let gillespie_engine = self.gillespie.as_mut().unwrap();
                gillespie_engine.set_time(continuous_time_before);
                let mut rates = vec![f64::NAN; gillespie_engine.nb_reactions()];
                for _ in 0..requested_reactions {
                    let time_before_reaction = gillespie_engine.get_time();
                    gillespie_engine._advance_one_reaction(&mut rates);
                    if !gillespie_engine.get_time().is_finite() {
                        // rebop uses +infinity to report that no reaction remains enabled.
                        gillespie_engine.set_time(time_before_reaction);
                        break;
                    }
                    actual_reactions += 1;
                }
                gillespie_time = gillespie_engine.get_time();
            }
            let engine_seconds = engine_start.elapsed().as_secs_f64();
            self.continuous_time = gillespie_time;

            let postprocess_start = Instant::now();
            self.sync_urn_from_gillespie();
            let postprocess_seconds = postprocess_start.elapsed().as_secs_f64();

            Ok(EngineCallBenchmark {
                gillespie,
                preparation_seconds,
                setup_seconds,
                engine_seconds,
                postprocess_seconds,
                continuous_time_advanced: self.continuous_time - continuous_time_before,
                total_reactions: actual_reactions,
                active_reactions: actual_reactions,
                k_rebuilt,
            })
        } else {
            let engine_start = Instant::now();
            let (total_reactions, active_reactions) = self.batch_step(f64::INFINITY, true);
            let engine_seconds = engine_start.elapsed().as_secs_f64();

            let postprocess_start = Instant::now();
            k_rebuilt |= self.finish_batch_step();
            let postprocess_seconds = postprocess_start.elapsed().as_secs_f64();

            Ok(EngineCallBenchmark {
                gillespie,
                preparation_seconds,
                setup_seconds: 0.0,
                engine_seconds,
                postprocess_seconds,
                continuous_time_advanced: self.continuous_time - continuous_time_before,
                total_reactions,
                active_reactions,
                k_rebuilt,
            })
        }
    }
    /// A read-only snapshot of the mode-switching state (config + measurement EMAs + observability
    /// counters). See :class:`SwitchState`.
    #[getter]
    pub fn switch(&self) -> SwitchState {
        self.switch.clone()
    }

    /// Run the simulation for a specified number of steps or until max time is reached
    #[pyo3(signature = (t_max, max_wallclock_time=3600.0))]
    pub fn run(&mut self, t_max: f64, max_wallclock_time: f64) -> PyResult<()> {
        if self.silent {
            return Err(PyValueError::new_err("Simulation is silent; cannot run."));
        }
        let max_wallclock_milliseconds = (max_wallclock_time * 1_000.0).ceil() as u64;
        let duration = Duration::from_millis(max_wallclock_milliseconds);
        let start_time = Instant::now();
        // Evaluate silence from the current configuration before the first engine call: a
        // simulation can start with no enabled reaction at all (e.g. an all-zero initial
        // condition, where K is reset to 1 and the total count is below the reaction order),
        // and batch_step cannot sample from such a configuration. The matching check inside
        // the loop only runs after an engine call.
        if !(self.active_reaction_probability() > 0.0) {
            self.silent = true;
        }
        while self.continuous_time < t_max && start_time.elapsed() < duration {
            if self.silent {
                // TODO: there should be some more robust behavior here,
                // in case the user expects the simulator to tell them when it became silent.
                self.continuous_time = t_max;
                return Ok(());
            }
            // The current iteration's engine call is "first after a switch" if we are about to
            // pay a rebuild / warm-up cost (initialize_gillespie_config or finalize_gillespie).
            // Such a call is not representative of steady-state throughput and is excluded from
            // the w/dt estimate below.
            let first_after_switch = (self.do_gillespie && self.just_started_gillespie)
                || (!self.do_gillespie && self.just_finished_gillespie);
            let ct_before = self.continuous_time;

            if self.do_gillespie {
                if self.just_started_gillespie {
                    let s = Instant::now();
                    self.initialize_gillespie_config();
                    self.switch.switch_overhead_seconds += s.elapsed().as_secs_f64();
                    self.switch.mode_switches += 1;
                }
                let s = Instant::now();
                self.gillespie_steps(t_max);
                let elapsed = s.elapsed().as_secs_f64();
                let dt = self.continuous_time - ct_before;
                self.switch.gillespie_wallclock_seconds += elapsed;
                self.switch.gillespie_continuous_time += dt;
                self.switch.gillespie_calls += 1;
                // Gillespie's fixed per-call cost is tiny, so even a call truncated at t_max
                // still gives a usable w/dt; only exclude the unrepresentative first call after
                // a switch.
                if !first_after_switch && dt > 0.0 {
                    self.switch.record(true, elapsed, dt);
                }
            } else {
                if self.just_finished_gillespie {
                    let s = Instant::now();
                    self.finalize_gillespie();
                    self.switch.switch_overhead_seconds += s.elapsed().as_secs_f64();
                    self.switch.mode_switches += 1;
                }
                let s = Instant::now();
                let _ = self.batch_step(t_max, false);
                self.finish_batch_step();
                let elapsed = s.elapsed().as_secs_f64();
                let dt = self.continuous_time - ct_before;
                self.switch.batch_wallclock_seconds += elapsed;
                self.switch.batch_continuous_time += dt;
                self.switch.batch_calls += 1;
                // A batch truncated at t_max did fewer than a full batch's reactions but paid the
                // full fixed cost, so its w/dt is inflated -- exclude it (unlike Gillespie).
                if !first_after_switch && dt > 0.0 && self.continuous_time < t_max {
                    self.switch.record(false, elapsed, dt);
                }
            }

            // Decide the mode for the next iteration. HEURISTIC_PROXY uses only the original
            // reaction-count rule; HEURISTIC_WALLCLOCK (default) starts from that rule but lets a
            // measured wall-clock-per-continuous-time (w/dt) comparison override it once the other
            // mode is measured decisively cheaper, probing the other mode occasionally to measure it.
            // HEURISTIC_PROSPECTIVE is deterministic and evaluates the batch at k_reset_target().
            // !(p > 0.0) rather than p == 0.0 so that a NaN from any future degenerate
            // propensity ratio is also treated as silent instead of bypassing the check.
            if self.switch.heuristic == HEURISTIC_PROSPECTIVE {
                let real_propensity = self.calculate_total_propensity(false);
                if !(real_propensity > 0.0) {
                    self.silent = true;
                }
                let score = self.prospective_batch_score_from_real_propensity(real_propensity);
                self.set_mode(score < self.switch.proxy_threshold);
                continue;
            }
            let active_probability = self.active_reaction_probability();
            if !(active_probability > 0.0) {
                self.silent = true;
            }
            let rough_expected_active_reactions_next_batch =
                active_probability * (self.n_including_extra_species as f64).sqrt();
            let proxy_gillespie = rough_expected_active_reactions_next_batch < self.switch.proxy_threshold;

            if self.switch.heuristic == HEURISTIC_PROXY {
                // Simpler heuristic: the reaction-count rule alone, no timing-based override.
                self.set_mode(proxy_gillespie);
            } else if self.switch.probe_remaining > 0 {
                // In-flight probe: keep running the (non-current) probed mode so we get one
                // representative measurement; the normal decision below then acts on it.
                self.switch.probe_remaining -= 1;
                let probed = self.do_gillespie;
                self.set_mode(probed);
            } else {
                self.switch.iters_since_probe += 1;
                let cur = self.do_gillespie;
                let both_measured = self.switch.has_est(true) && self.switch.has_est(false);
                let interval = if both_measured {
                    WDT_PROBE_INTERVAL_COMMITTED
                } else {
                    WDT_PROBE_INTERVAL
                };
                if self.switch.has_est(cur) && self.switch.iters_since_probe >= interval {
                    // Probe the other mode (one switch iteration, excluded from the EMA, then one
                    // representative measurement) so its estimate is available/fresh.
                    self.switch.iters_since_probe = 0;
                    self.switch.probe_remaining = 1;
                    self.set_mode(!cur);
                } else if both_measured {
                    // Both modes measured: follow the proxy unless the other is decisively cheaper.
                    let non_proxy = !proxy_gillespie;
                    let override_proxy =
                        self.switch.wdt(non_proxy) * WDT_OVERRIDE_FACTOR < self.switch.wdt(proxy_gillespie);
                    self.set_mode(if override_proxy { non_proxy } else { proxy_gillespie });
                } else {
                    // Bootstrap: follow the proxy (also measures whichever mode it runs).
                    self.set_mode(proxy_gillespie);
                }
            }
        }
        Ok(())
    }

    /// Called when switching from Gillespie to batch mode; populates the batch config with Gillespie's current
    /// configuration.
    fn finalize_gillespie(&mut self) {
        self.sync_urn_from_gillespie();
        self.reset_k_count();
    }

    /// Initialize the Gillespie simulator; load batch config and reactions into the Gillespie object.
    /// Doing Gillespie, we may as well operate in the original CRN,
    /// because it is faithfully simulated.
    fn initialize_gillespie_config(&mut self) {
        let mut gillespie_config: Vec<isize> = vec![0; self.q - 2];
        let mut species_index = 0;
        // Keep track of how species correspond since we need to ignore K and W.
        let mut batching_idx_to_gillespie_idx: HashMap<usize, usize> = HashMap::new();
        // Iterate through species skipping k and w so that they appear in
        // the same order in the rebop Gillespie object.
        for i in 0..self.q {
            if i == self.crn.k || i == self.crn.w {
                continue;
            }

            batching_idx_to_gillespie_idx.insert(i, species_index);
            gillespie_config[species_index] = self.urn.config[i] as isize;
            species_index += 1;
        }
        //TODO: the current method name only describes the code above, because the code below we intend to move
        // to the SimulatorCRN constructor as in issue https://github.com/UC-Davis-molecular-computing/batss/issues/11
        // above will have to be rewritten to call get_species on the existing self.gillespie object.

        // The "false" here means we aren't optimizing for the CRN to be "sparse."
        // See https://github.com/Armavica/rebop/pull/35 for a discussion.
        // I think the kinds of CRNs that this system is good at simulating,
        // will typically not be sparse, but that might not be true.
        let mut gillespie = Gillespie::new_with_seed(gillespie_config, false, self.rng.next_u64());
        // Put the reactions into the Gillespie object.
        for reaction in &self.crn.reactions {
            let mut rebop_reactant_stoichs = vec![0; self.q - 2];
            let mut rebop_reactant_stoichs_negated = vec![0; self.q - 2];
            for reactant in &reaction.reactants {
                assert!(*reactant != self.crn.w, "W should never be a reactant.");
                if *reactant == self.crn.k {
                    continue;
                }
                let reactant_gillespie_idx = batching_idx_to_gillespie_idx[&reactant];
                rebop_reactant_stoichs[reactant_gillespie_idx] += 1;
                rebop_reactant_stoichs_negated[reactant_gillespie_idx] -= 1;
            }
            // iterate over all reactions that have this set of reactants
            for (products, rate_constant) in &reaction.products_and_rate_constants {
                let mut rebop_reaction_net_productions = rebop_reactant_stoichs_negated.clone();
                for product in products {
                    if *product == self.crn.k || *product == self.crn.w {
                        continue;
                    }
                    let product_gillespie_idx = batching_idx_to_gillespie_idx[&product];
                    rebop_reaction_net_productions[product_gillespie_idx] += 1;
                }
                gillespie.add_reaction(
                    Rate::lma(*rate_constant, &rebop_reactant_stoichs),
                    &rebop_reaction_net_productions,
                );
            }
        }
        self.gillespie = Some(gillespie);
    }

    /// Run the simulation until it is silent, i.e., no reactions are applicable.
    #[pyo3()]
    pub fn run_until_silent(&mut self) {
        unimplemented!();
        // while !self.silent {
        //     if self.do_gillespie {
        //         self.gillespie_step(0);
        //     } else {
        //         self.batch_step(0);
        //     }
        // }
    }

    /// Reset the simulation with a new configuration
    /// Sets all parameters necessary to change the configuration.
    /// Args:
    ///     config: The configuration array to reset to.
    ///     t: The new value of :any:`t`. Defaults to 0.
    #[pyo3(signature = (config, t=0.0))]
    pub fn reset(&mut self, config: PyReadonlyArray1<u64>, t: f64) -> PyResult<()> {
        let config = config.to_vec().unwrap();
        let old_k_count = self.urn.config[self.crn.k];
        self.urn.reset_config(&config);
        self.n_including_extra_species = self.urn.size;
        self.n = self.n_including_extra_species - (self.urn.config[self.crn.k] + self.urn.config[self.crn.w]);
        if old_k_count != config[self.crn.k] {
            // If the count of k changed during the simulation, we need to do the expensive operation
            // of recomputing transition arrays.
            // Otherwise, k should already be set correctly, as it should be in the input config.
            self.reset_k_count();
        }
        self.n_including_extra_species = self.n + self.urn.config[self.crn.k];
        self.continuous_time = t;
        self.silent = self.n == 0;
        // Return to the batch-mode starting state a fresh simulator has. Without this, a reset
        // called mid-Gillespie would leave `do_gillespie` true with a stale `self.gillespie`
        // (holding the pre-reset population), and the next run() would assert on the n mismatch.
        // Clearing the switch measurements too makes the reused simulator's mode decisions match
        // a freshly-constructed one's (important when reset is used to warm up a benchmark run).
        self.do_gillespie = false;
        self.just_started_gillespie = false;
        self.just_finished_gillespie = false;
        self.switch.reset();
        Ok(())
    }

    #[pyo3(signature = (filename=None))]
    pub fn write_profile(&self, filename: Option<String>) -> PyResult<()> {
        let spans = flame::spans();
        if spans.is_empty() {
            println!("No profiling data available since flame_profiling feature disabled.");
            return Ok(());
        }

        let mut content = String::new();
        content.push_str("Flame Profile Report\n");
        content.push_str("====================\n");

        // Process the span tree recursively
        let mut span_data_map: HashMap<String, SpanData> = HashMap::new();
        for span in &spans {
            process_span(&mut span_data_map, span);
        }

        write_span_data(&mut content, &span_data_map, 0);

        // content.push_str(&format!("\nTotal time: {}ms\n", total_time_ms));

        if filename.is_none() {
            println!("{}", content);
        } else {
            let filename = filename.unwrap();
            let mut file = std::fs::File::create(filename)?;
            file.write_all(content.as_bytes())?;
        }

        Ok(())
    }

    /// Clear all Flame spans accumulated by the current thread.
    ///
    /// This lets a benchmark discard simulator-construction spans before profiling one frozen
    /// state. It is a no-op in ordinary builds where the flm feature is disabled.
    pub fn clear_profile(&self) {
        flame::clear();
    }

    #[pyo3(signature = (r, u, has_bounds=false))]
    pub fn sample_collision(&self, r: u64, u: f64, has_bounds: bool) -> u64 {
        return self.sample_collision_fast_f128(r, u, has_bounds);
    }

    /// Sample from collision distribution using python's mpmath library, which allows us to
    /// do arbitrary-precision floating point arithmetic without rolling our own loggamma function.
    pub fn sample_collision_fast_python(&self, r: u64, u: f64, _has_bounds: bool) -> u64 {
        let args = (self.n_including_extra_species, r, self.crn.o, self.crn.g, u);
        let result = Python::attach(|py| -> PyResult<u64> {
            let result = self
                .python_module
                .getattr(py, "sample_coll")?
                .call1(py, args)?
                .extract(py);
            result
        })
        .unwrap();
        return result;
    }

    /// Sample from birthday-like distribution "directly". This is the number of times
    /// we can do the following before seeing something painted red: sample o objects without replacement,
    /// remove them and add o + g red objects, given that r objects out of n are initially red.
    #[pyo3()]
    pub fn sample_collision_directly(&mut self, n: u64, r: u64) -> u64 {
        let mut idx = 0u64;
        assert!(r < n, "r must be less than n");
        assert!(n < u64::MAX, "n must be less than u64::MAX");
        let mut num_seen = r;
        let mut pop_size = n;
        loop {
            for _ in 0..self.crn.o {
                let sample = self.rng.random_range(0..pop_size);
                if sample < num_seen {
                    return idx;
                }
                pop_size -= 1;
            }
            idx += 1;
            pop_size += (self.crn.o + self.crn.g) as u64;
            num_seen += (self.crn.o + self.crn.g) as u64;
        }
    }

    /// Sample from the length of how long this batch would take to run, in continuous time.
    /// Population size is included as a parameter for checkpoint rejection sampling, so that
    /// this function can be called on population sizes other than n without updating n.
    pub fn sample_batch_time(&mut self, initial_n: u64, batch_size: u64) -> f64 {
        assert!(batch_size > 0);
        // For now, we're just gonna do the geometric mean thing.
        let first_term = 1.0 / self.get_exponential_rate(initial_n);
        let last_term = 1.0 / self.get_exponential_rate(initial_n + self.crn.g as u64 * (batch_size - 1));
        let prod = first_term * last_term;
        assert!(prod > 0.0);
        let geom_mean = prod.sqrt();
        let estimated_mean = batch_size as f64 * geom_mean;
        // Also copying the simplest variance estimation method
        let mut estimated_variance = 0.0;
        let last_term = 1.0
            / self
                .get_exponential_rate(initial_n + (batch_size - 1) * self.crn.g as u64)
                .powi(2);
        if last_term == 0.0 {
            println!("Last term is 0!");
        } else {
            let first_term = 1.0 / self.get_exponential_rate(initial_n).powi(2);
            let _relative_error_first_last_term = relative_error(first_term, last_term);
            let var_prod = first_term * last_term;
            estimated_variance = batch_size as f64 * var_prod.sqrt();
        }
        let shape = estimated_mean.powi(2) / estimated_variance;
        let scale = estimated_variance / estimated_mean;
        let gamma = Gamma::new(shape, scale).unwrap();
        let val = self.rng.sample(gamma);
        return val;
    }

    pub fn sample_hypo_directly(&mut self, initial_n: u64, batch_size: u64) -> f64 {
        let mut answer = 0.0;
        for i in 0..batch_size {
            answer += self.sample_exponential(self.get_exponential_rate(initial_n + i * self.crn.g as u64));
        }
        return answer;
    }
    /// The probability that the next sampled reaction is active (actually changes the original
    /// CRN's configuration): `P_real / P_total`, where `P_total = calculate_total_propensity(true)` uses
    /// `n_including_extra_species`, the current population including the filler species K. It is a
    /// function of the current configuration -- the full urn state including the count of K, not the
    /// original species counts alone. Since K drifts over the run (restored to n only when K/n leaves
    /// [0.5, 2], and frozen during Gillespie phases), two snapshots with the same original-species
    /// counts but different K report different values. Used to record `Simulation.active_fractions`
    /// and by the switching heuristic in `run`.
    pub fn active_reaction_probability(&self) -> f64 {
        let total_propensity_not_including_passive_reactions = self.calculate_total_propensity(false);
        let total_propensity_including_passive_reactions = self.calculate_total_propensity(true);
        assert!(
            total_propensity_not_including_passive_reactions <= total_propensity_including_passive_reactions,
            "Total propensity should not be lower when including passive reactions: {:?} with vs {:?} without",
            total_propensity_including_passive_reactions,
            total_propensity_not_including_passive_reactions
        );
        if total_propensity_including_passive_reactions == 0.0 {
            // No reaction, active or passive, is enabled: n_including_extra_species < o makes
            // C(N, o) = 0, e.g. after a batch-mode extinction resets K to 1. The assert above
            // guarantees the numerator is also 0; return 0 rather than the 0/0 NaN, so callers
            // (in particular the silence check in `run`) see a real probability.
            return 0.0;
        }
        return total_propensity_not_including_passive_reactions / total_propensity_including_passive_reactions;
    }

    // Observability for the optimal-K analysis: expose the propensity components, CRN constants, the
    // config-independent crossover, and the current reset target.
    pub fn debug_p_active(&self) -> f64 {
        self.calculate_total_propensity(false)
    }
    pub fn debug_p_total(&self) -> f64 {
        self.calculate_total_propensity(true)
    }
    pub fn debug_kmax(&self) -> f64 {
        self.crn.continuous_time_correction_factor
    }
    pub fn debug_o(&self) -> usize {
        self.crn.o
    }
    pub fn debug_g(&self) -> usize {
        self.crn.g
    }
    pub fn debug_crossover(&self) -> f64 {
        self.crossover_k0
    }
    pub fn debug_k_reset_target(&self) -> u64 {
        self.k_reset_target()
    }
    pub fn debug_prospective_n(&self) -> u64 {
        self.n.saturating_add(self.k_reset_target())
    }
    pub fn debug_q(&self) -> usize {
        self.crn.q
    }
    pub fn debug_reactant_sets(&self) -> usize {
        self.crn.reactions.len()
    }
    pub fn debug_output_branches(&self) -> usize {
        self.crn
            .reactions
            .iter()
            .map(|reaction| reaction.products_and_rate_constants.len())
            .sum()
    }
}

// --- Wall-clock-aware batch/Gillespie switching ---------------------------------
// EMA smoothing factor for each mode's measured wall-clock-per-continuous-time (w/dt).
const WDT_EMA_ALPHA: f64 = 0.3;
// The reaction-count proxy is the default mode choice; deviate from it only when the other
// mode's measured w/dt is at least this many times smaller. A decisive margin fixes the cases
// the proxy gets badly wrong (the Oregonator, where Gillespie is ~60x cheaper) while leaving the
// cases it gets right undisturbed (Dimerization, where the two modes stay within ~2x, so we keep
// the proxy's batch choice and its published performance).
const WDT_OVERRIDE_FACTOR: f64 = 4.0;
// While bootstrapping (a mode not yet measured), probe the other mode this often so that it gets
// measured at all -- the proxy may never choose it (e.g. the Oregonator at large n, where the
// proxy always says "batch").
const WDT_PROBE_INTERVAL: u64 = 256;
// Once both modes are measured, re-probe only this often -- rarely, just to catch a slow regime
// change. Frequent probing of the coarse-grained Gillespie engine (whose dt per call is large)
// would waste continuous time in the costlier mode; this is what would otherwise slow Dimerization,
// whose batch engine wins decisively at large n.
const WDT_PROBE_INTERVAL_COMMITTED: u64 = 8192;

// Rebuild K only once it has drifted more than this multiplicative factor from its target
// min(2n, crossover). Smaller = K tracks the optimum more tightly (smaller jumps in the active
// fraction each time it resets) but rebuilds more often; larger = fewer rebuilds, bigger jumps.
// (This is why changing it visibly changes the active-fraction trace: each reset is a jump back
// to the optimal K, and this factor sets how far K drifts -- and how big the jump is -- between resets.)
const K_RESET_BAND_FACTOR: f64 = 1.1;

fn relative_error(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        return b.abs();
    }
    if b == 0.0 {
        return a.abs();
    }
    return (a - b).abs() / a.abs().min(b.abs());
}

fn write_span_data(content: &mut String, span_data_map: &HashMap<String, SpanData>, depth: usize) {
    let indent = "  ".repeat(depth);
    let mut span_datas: Vec<&SpanData> = span_data_map.values().collect();
    span_datas.sort_by_key(|span_data| span_data.ns);
    span_datas.reverse();
    let mut name_length = 0;
    for span_data in &span_datas {
        name_length = name_length.max(span_data.name.len());
    }
    for span_data in span_datas {
        content.push_str(&format!(
            "{}{:name_length$}: {:.3} ms ({} calls, {:.3} us/call)\n",
            indent,
            span_data.name,
            span_data.ns as f64 / 1_000_000.0,
            span_data.count,
            span_data.ns as f64 / span_data.count as f64 / 1_000.0,
        ));
        write_span_data(content, &span_data.children, depth + 1);
    }
}

struct SpanData {
    name: String,
    ns: u64,
    count: u64,
    children: HashMap<String, SpanData>,
}

impl SpanData {
    fn new(name: String) -> Self {
        SpanData {
            name,
            ns: 0,
            count: 0,
            children: HashMap::new(),
        }
    }
}

// Helper function to process spans recursively
fn process_span(span_data_map: &mut HashMap<String, SpanData>, span: &flame::Span) {
    let span_name = span.name.to_string();
    if !span_data_map.contains_key(&span_name) {
        span_data_map.insert(span_name.clone(), SpanData::new(span_name.clone()));
    }

    let span_data = span_data_map.get_mut(&span_name).unwrap();
    span_data.ns += span.delta;
    span_data.count += 1;

    // Process children recursively
    for child in &span.children {
        process_span(&mut span_data.children, child);
    }
}

// A struct for holding the q^o results of multidimensional hypergeometric sampling from an urn.
// Recursive because of unknown nesting depth. To efficiently sample the k reactions of a batch,
// we first need to sample all the codimension-1 sums, that is, how many reactions have each species
// as its first reactant. Then, for each of those, we need to sample the codimension-2 sums, e.g.,
// how many reactions with A as their first reactant have each reactant as their second. And so on,
// recursively down to o dimensions.
// When sampling, the each struct will store its codimension-1 sums in values. Then, for each
// of those values, it will recursively sample that many subreactions into its subresults.
// This could be implemented in a more memory-efficient way where the subresult is just a single
// NDBatchResult instead of a Vec, and only one result is stored at a time during the recursion.
// I'm not sure if it's a better implementation; it's definitely better if we expect q^o to
// potentially be large enough that we couldn't store it all at once, though at that point
// it's unlikely that this is the right algorithm for simulation.
struct NDBatchResult {
    // Tells you what level of recursion you're on.
    // For the top level, dimensions = o (number of reactants).
    // For the bottom level, dimensions = 1.
    dimensions: usize,
    q: usize,
    o: usize,
    // For iterating over results
    curr_species: State,
    // Initialized to 0, then sampled into via urn.sample_vector().
    pub counts: Vec<u64>,
    // If dimensions > 1, this is a vector of subresults. If dimensions = 1, it is empty.
    pub subresults: Option<Vec<NDBatchResult>>,
}

impl NDBatchResult {
    /// Recursive function used to generate new NDBatchResult.
    /// Creates and initializes all recursive substructures.
    fn populate_empty(&mut self) {
        if self.dimensions > 1 {
            for _ in 0..self.q {
                let mut subresult = NDBatchResult {
                    dimensions: self.dimensions - 1,
                    q: self.q,
                    o: self.o,
                    curr_species: 0,
                    counts: vec![0; self.q],
                    subresults: {
                        if self.dimensions == 2 {
                            None
                        } else {
                            Some(Vec::with_capacity(self.q))
                        }
                    },
                };
                subresult.populate_empty();
                self.subresults.as_mut().unwrap().push(subresult);
            }
        }
    }
    /// Recursive function to sample how many of each possible reaction vector
    /// happen within some batch, using hypergeometric sampling via sample_vector.
    fn sample_batch_result(&mut self, num_reactions: u64, urn: &mut Urn) {
        urn.sample_vector(num_reactions, &mut self.counts).unwrap();
        self.curr_species = 0;
        if self.dimensions > 1 {
            for i in 0..self.q {
                let subresults = self.subresults.as_mut().unwrap();
                subresults[i].sample_batch_result(self.counts[i], urn);
            }
        }
    }
    /// Method used for recursively iterating through NDBatchResult.
    /// Returns triple (reactants, count, done).
    /// reactants: which reactant vector this entry represents
    /// count: how many times that reactant vector has been sampled in this batch
    /// done: true iff this is the last entry in the NDBatchResult.
    /// TODO this should probably implement an iterable trait
    fn get_next(&mut self) -> (Vec<State>, u64, bool) {
        assert!(self.curr_species < self.q, "NDBatchResult iterated past final species");
        let mut done = false;
        if self.dimensions == 1 {
            // flame::start("base case processing");
            let mut curr_reaction = vec![0; self.o];
            curr_reaction[self.o - self.dimensions] = self.curr_species;
            self.curr_species += 1;
            // flame::end("base case processing");
            return (
                curr_reaction,
                self.counts[self.curr_species - 1],
                self.curr_species == self.q,
            );
        } else {
            // flame::start("recursive case processing");
            let curr_subresult = &mut self.subresults.as_mut().unwrap()[self.curr_species];
            // flame::end("recursive case processing");
            let (mut curr_reaction, count, subresult_done) = curr_subresult.get_next();
            // flame::start("recursive case processing");
            curr_reaction[self.o - self.dimensions] = self.curr_species;
            if subresult_done {
                self.curr_species += 1;
                if self.curr_species == self.q {
                    done = true;
                }
            }
            // flame::end("recursive case processing");
            return (curr_reaction, count, done);
        }
    }
}

fn make_batch_result(dimensions: usize, length: usize) -> NDBatchResult {
    let mut result = NDBatchResult {
        dimensions: dimensions,
        q: length,
        o: dimensions,
        curr_species: 0,
        counts: vec![0; length],
        subresults: Some(Vec::with_capacity(length)),
    };
    result.populate_empty();
    result
}

impl BatchSimulator {
    /// Compute the prospective score from a real-reaction propensity already evaluated for the
    /// current original-species configuration. Keeping the propensity as an argument avoids doing
    /// that work twice in run's hot loop.
    fn prospective_batch_score_from_real_propensity(&self, real_propensity: f64) -> f64 {
        if !(real_propensity > 0.0) {
            return 0.0;
        }

        let prospective_k = self.k_reset_target();
        let prospective_n = self.n.saturating_add(prospective_k);
        let kmax = self.crn.max_adjusted_rate_constant_at(prospective_k);
        let total_propensity = kmax * binomial_as_f64(prospective_n, self.crn.o as u64);
        if !(total_propensity > 0.0) {
            return 0.0;
        }

        let o = self.crn.o as f64;
        let g = self.crn.g as f64;
        let collision_denominator = 2.0 * o * (o + g);
        if !(collision_denominator > 0.0) {
            return 0.0;
        }
        let expected_batch_length =
            (std::f64::consts::PI / collision_denominator).sqrt() * (prospective_n as f64).sqrt();
        (real_propensity / total_propensity) * expected_batch_length
    }

    /// Choose the engine for the next iteration (`true` = Gillespie), maintaining the
    /// `just_started_gillespie` / `just_finished_gillespie` transition flags exactly as the
    /// original switching code did, so the one-time rebuild still fires on a real transition.
    fn set_mode(&mut self, want_gillespie: bool) {
        if want_gillespie {
            if self.do_gillespie {
                self.just_started_gillespie = false;
            } else {
                self.do_gillespie = true;
                self.just_started_gillespie = true;
            }
        } else if self.do_gillespie {
            self.do_gillespie = false;
            self.just_finished_gillespie = true;
        } else {
            self.just_finished_gillespie = false;
        }
    }

    /// Apply the normal between-batch K-band check and recycle W.
    ///
    /// Returning whether K was rebuilt lets the offline oracle keep rare transition-array rebuilds
    /// separate from steady batch work while production run keeps the same timing/accounting.
    fn finish_batch_step(&mut self) -> bool {
        flame::start("finish batch step");
        let current_k_count = self.urn.config[self.crn.k];
        let target_k_count = self.k_reset_target();
        // Rebuild K only when it has drifted more than the multiplicative band from its target.
        let lo = current_k_count.min(target_k_count).max(1) as f64;
        let hi = current_k_count.max(target_k_count) as f64;
        let k_rebuilt = hi > K_RESET_BAND_FACTOR * lo;
        if k_rebuilt {
            self.reset_k_count();
            self.k_resets += 1;
        }
        self.recycle_waste();
        flame::end("finish batch step");
        k_rebuilt
    }

    /// Run one batch of reactions, on average O(sqrt(n)) of them, some of which will typically be passive.
    /// Returns after simulating one batch, and does not necessarily run until `t_max`.
    /// Updates the urn and any relevant variables; the `BatchSimulator` should be in a valid state afterward.
    fn batch_step(&mut self, t_max: f64, track_active_reactions: bool) -> (u64, u64) {
        flame::start("batch step");
        self.updated_counts.reset();
        assert_eq!(
            self.n_including_extra_species, self.urn.size,
            "Self.n_including_extra_species should match self.urn.size."
        );
        assert_eq!(
            self.n,
            self.urn.size - (self.urn.config[self.crn.k] + self.urn.config[self.crn.w]),
            "Self.n should match self.urn.size minus counts of K and W."
        );
        let initial_k_count: u64 = self.urn.config[self.crn.k];

        let u: f64 = self.rng.sample(StandardUniform);

        let has_bounds = false;
        flame::start("sample_coll");
        let l = self.sample_collision(0, u, has_bounds);
        flame::end("sample_coll");

        let mut rxns_before_coll = l;
        assert!(l > 0, "sample_coll must return at least 1 for batching");
        flame::start("sample batch clock");
        let batch_time = self.sample_batch_time(self.n_including_extra_species, l);
        let mut do_collision = true;
        if self.continuous_time + batch_time <= t_max {
            self.continuous_time += batch_time;
            // It's possible that all of the reactions *except* the collision happen before t_max,
            // but then the collision happens after t_max.
            let collision_time = self
                .sample_exponential(self.get_exponential_rate(self.n_including_extra_species + self.crn.g as u64 * l));
            if self.continuous_time + collision_time < t_max {
                self.continuous_time += collision_time;
            } else {
                do_collision = false;
                self.continuous_time = t_max;
            }
        } else {
            // The next collision happens after t_max. In order to stop at t_max, we need to
            // figure out how many reactions happen before t_max. We do this by rejection sampling:
            // we check how long each individual reaction would take to run until the total time
            // exceeds t_max. If the total time fails to exceed t_max after running l reactions,
            // we reject and start over.
            // There is some probability p of having the next collision after t_max; we will have
            // to do this rejection sampling with probability p, and it will succeed with
            // probability p, meaning we will have to run it about 1/p times, so about once per
            // checkpoint that the user wants.
            do_collision = false;
            flame::start("checkpoint rejection sampling");
            rxns_before_coll = self.checkpoint_rejection_sampling(l, t_max);

            self.continuous_time = t_max;
            flame::end("checkpoint rejection sampling");
        }
        flame::end("sample batch clock");

        // The idea here is to iterate through random_transitions and array_sums together; they should
        // both be indexed by q^o-tuples when iterated through this way, and the iteration order should
        // be lexicographic for both of them.
        flame::start("sample batch");
        self.array_sums.sample_batch_result(rxns_before_coll, &mut self.urn);
        flame::end("sample batch");

        flame::start("process batch");
        let mut done = false;
        let mut active_reactions_this_batch = 0;
        let reactions_iter = self.random_transitions.lanes(Axis(self.crn.o)).into_iter();
        // TODO: we might be able to reintroduce the optimzation around keeping the urn sorted
        // and taking advantage of sample_vector returning the highest state returned. Probably
        // in the current implementation, this would live in NDBatchResult and its iteration,
        // and we'd iterate over it instead of self.random_transitions.
        for random_transition in reactions_iter {
            assert!(
                !done,
                "self.array_sums finished iterating before self.random_transitions"
            );
            let next_array_sum = self.array_sums.get_next();
            // TODO maybe add an assert check that the two structures are iterated through
            // in the same order, i.e. reactants match
            let (reactants, quantity) = (next_array_sum.0, next_array_sum.1);
            done = next_array_sum.2;
            if quantity == 0 {
                continue;
            }
            let initial_updated_counts_size = self.updated_counts.size;
            let (num_outputs, first_idx) = (random_transition[0], random_transition[1]);
            // TODO and WARNING: this code is more or less copy-paste with the collision sampling code.
            // They do the same thing. But it's apparently very annoying to refactor this into a
            // helper method in rust because of the immutable borrow of self above.
            if num_outputs == 0 {
                // Passive reaction. Move the reactants from self.urn to self.updated_counts (for collision sampling), and add W.
                for reactant in reactants {
                    self.updated_counts.add_to_entry(reactant, quantity as i64);
                }
                self.updated_counts
                    .add_to_entry(self.crn.w, (quantity * self.crn.g as u64) as i64);
            } else {
                let mut probabilities = self.transition_probabilities[first_idx..first_idx + num_outputs].to_vec();
                let active_probability_sum: f64 = probabilities.iter().sum();
                if active_probability_sum < 1.0 {
                    probabilities.push(1.0 - active_probability_sum);
                }
                multinomial_sample(
                    quantity,
                    &probabilities,
                    &mut self.m[0..probabilities.len()],
                    &mut self.rng,
                );
                assert_eq!(
                    self.m[0..probabilities.len()].iter().sum::<u64>(),
                    quantity,
                    "sample sum mismatch"
                );
                if track_active_reactions {
                    active_reactions_this_batch += self.m[0..num_outputs].iter().sum::<u64>();
                }
                for offset in 0..num_outputs {
                    let idx = first_idx + offset;
                    let outputs = &self.random_outputs[idx];
                    for output in outputs {
                        self.updated_counts.add_to_entry(*output, self.m[offset] as i64);
                    }
                }
                // Add any W produced by passive reactions, and add those reactants to updated_counts.
                if active_probability_sum < 1.0 {
                    let passive_count = self.m[num_outputs];
                    self.updated_counts
                        .add_to_entry(self.crn.w, (passive_count * self.crn.g as u64) as i64);
                    for reactant in reactants {
                        self.updated_counts.add_to_entry(reactant, passive_count as i64);
                    }
                }
            }
            assert_eq!(
                quantity * (self.crn.o + self.crn.g) as u64,
                self.updated_counts.size - initial_updated_counts_size,
                "Mismatch between how many elements were added to updated_counts."
            )
        }
        assert!(
            done,
            "self.random_transitions finished iterating before self.array_sums"
        );

        assert_eq!(
            (self.crn.g + self.crn.o) as u64 * rxns_before_coll,
            self.updated_counts.size,
            "Total number of molecules added is not consistent"
        );

        flame::end("process batch");
        flame::start("sample collision");
        // We need to sample a collision. It could involve as few as 1 already-used molecule,
        // or as many as o. So we need to decide how many are involved.
        // TODO: I'm going to use u128 here because I'm pretty worried about fitting things
        // into anything smaller. In fact I'm a little worried about u128; on population size
        // 10^12 which is about 2^40, if we have 4 reactants, then the denominator for the
        // relevant probability distribution will be too large to store.
        let mut num_resampled = 0;
        if do_collision {
            let mut collision_count_num_ways: Vec<u128> = Vec::with_capacity(self.crn.o);
            let num_new_molecules = self.updated_counts.size;
            let num_old_molecules = self.urn.size;
            // Count the number of ways that the collision reaction could have had exactly
            // 1 reactant that has already been touched, or exactly 2, up to exactly o.
            for num_updated_reactants_in_collision in 1..self.crn.o + 1 {
                collision_count_num_ways.push(
                    (num_old_molecules as u128)
                        .pow((self.crn.o - num_updated_reactants_in_collision).try_into().unwrap())
                        * (num_new_molecules as u128).pow(num_updated_reactants_in_collision.try_into().unwrap())
                        * binomial(self.crn.o as u64, num_updated_reactants_in_collision as u64) as u128,
                );
                //XXX: note that binomial is from the num_integer crate and returns a u64:
                // https://docs.rs/num-integer/latest/num_integer/fn.binomial.html
                // use this only for small inputs, and use util::binomial_as_f64 for large inputs
                // where the result might not fit into a u64.
            }
            // TODO: there should be some standard way to sample from this discrete probability distribution.
            // Should be rand::WeightedIndex. Also urn::sample_one.
            let total_ways_with_at_least_one_collision: u128 = collision_count_num_ways.iter().sum();
            let u2: f64 = self.rng.sample(StandardUniform);
            let mut num_colliding_molecules = 0;
            let mut total_ways_so_far = 0;
            for i in 0..self.crn.o {
                total_ways_so_far += collision_count_num_ways[i];
                if u2 < (total_ways_so_far as f64) / (total_ways_with_at_least_one_collision as f64) {
                    num_colliding_molecules = i + 1;
                    break;
                }
            }
            assert!(num_colliding_molecules > 0, "Failed to sample collision size");
            let mut collision: Vec<State> = Vec::with_capacity(self.crn.o);
            for _ in 0..num_colliding_molecules {
                collision.push(self.updated_counts.sample_one().unwrap());
                num_resampled += 1;
            }
            for _ in num_colliding_molecules..self.crn.o {
                collision.push(self.urn.sample_one().unwrap());
            }
            // Index into random_probabilities to sample what the collision will do.
            let mut view = self.random_transitions.view();
            for dimension in 0..self.crn.o {
                view = view.index_axis_move(Axis(0), collision[dimension]);
            }
            // Verify that the view is now a 1-dimensional subarray of random_probabilities,
            // which should just have two elements in it (number of random outputs and starting index)
            assert_eq!(
                view.ndim(),
                1,
                "Was not left with 1-dimensional vector after indexing collision"
            );
            assert_eq!(view.len(), 2, "Indexing collision did not leave two-element subarray");

            let (num_outputs, first_idx) = (view[0], view[1]);
            // TODO: this code is heavily copy-pasted. See other TODO comment above.
            if num_outputs == 0 {
                // Passive reaction.
                for reactant in collision {
                    self.updated_counts.add_to_entry(reactant, 1);
                }
                self.updated_counts.add_to_entry(self.crn.w, (self.crn.g) as i64);
            } else {
                let mut probabilities = self.transition_probabilities[first_idx..first_idx + num_outputs].to_vec();
                let active_probability_sum: f64 = probabilities.iter().sum();
                if active_probability_sum < 1.0 {
                    probabilities.push(1.0 - active_probability_sum);
                }
                multinomial_sample(1, &probabilities, &mut self.m[0..probabilities.len()], &mut self.rng);
                assert_eq!(
                    self.m[0..probabilities.len()].iter().sum::<u64>(),
                    1,
                    "sample sum mismatch"
                );
                if track_active_reactions {
                    active_reactions_this_batch += self.m[0..num_outputs].iter().sum::<u64>();
                }
                for c in 0..num_outputs {
                    let idx = first_idx + c;
                    let outputs = &self.random_outputs[idx];
                    for offset in outputs {
                        self.updated_counts.add_to_entry(*offset, self.m[c] as i64);
                    }
                }
                // Add W if the collision was a probabilistic passive reaction.
                if active_probability_sum < 1.0 {
                    let passive_count = self.m[num_outputs];
                    self.updated_counts
                        .add_to_entry(self.crn.w, (passive_count * self.crn.g as u64) as i64);
                    for reactant in collision {
                        self.updated_counts.add_to_entry(reactant, passive_count as i64);
                    }
                }
            }
            assert_eq!(
                self.updated_counts.size - num_new_molecules,
                (self.crn.o + self.crn.g) as u64 - num_resampled,
                "Collision failed to add exactly g things to updated_counts"
            );
        }
        flame::end("sample collision");

        // Total reactions simulated in this batch: the reactions before the collision, plus the
        // collision reaction itself when one occurred.
        let reactions_this_batch = rxns_before_coll + if do_collision { 1 } else { 0 };

        flame::start("urn commit and sort");
        self.urn.add_vector(&self.updated_counts.config);
        self.urn.sort();
        flame::end("urn commit and sort");
        // Check that we added the right number of things to the urn.
        assert_eq!(
            self.urn.size - self.n_including_extra_species,
            reactions_this_batch * self.crn.g as u64,
            "Inconsistency between number of reactions simulated and population size change."
        );
        assert_eq!(
            initial_k_count, self.urn.config[self.crn.k],
            "Count of K should never change within running a batch."
        );
        self.n_including_extra_species = self.urn.size;
        self.n = self.n_including_extra_species - self.urn.config[self.crn.k] - self.urn.config[self.crn.w];

        flame::end("batch step");
        (reactions_this_batch, active_reactions_this_batch)
    }

    /// Perform some Gillespie steps.
    /// Note that this does not in general run until `t_max`; `t_max` is just a maximum time after which to quit.
    /// The exact number of steps performed is not deterministic;
    /// we run Gillespie for roughly sqrt(n) steps, and then re-check whether batching
    /// is likely to be faster again.
    fn gillespie_steps(&mut self, t_max: f64) -> () {
        let original_gillespie_n = self.gillespie_total_population_count();
        assert!(
            original_gillespie_n == self.n,
            "self.n ({:?}) does not match gillespie value of n ({:?})",
            self.n,
            original_gillespie_n
        );
        assert!(
            self.continuous_time < t_max,
            "gillespie_steps should not be called when already past t_max"
        );

        let total_propensity = self.calculate_total_propensity(false);
        // Borrow mutably for as long as we need it, for convenience.
        let gillespie = self.gillespie.as_mut().unwrap();
        gillespie.set_time(self.continuous_time);

        // For now, we're going to assume that we will need to do, say,
        // at least O(sqrt n) reactions until it's worth turning on batching again.
        let num_rxns_to_execute = self.n.sqrt();
        let block_by_reactions = self.gillespie_block_by_reactions;
        let executed = if block_by_reactions {
            // Budget the block in reactions directly. rebop stops at whichever comes first, the
            // reaction count or t_max, and the t_max test still happens between sampling a
            // reaction time and applying the reaction, so exactness at t_max is unchanged.
            gillespie.advance_until_or_reactions(t_max, Some(num_rxns_to_execute))
        } else {
            // Historical path: turn the target count into a duration using the propensity measured
            // on entry. This is exact only if the propensity is constant over the block.
            let ave_time_per_rxn = 1.0 / total_propensity;
            let time_to_run_gillespie = ave_time_per_rxn * num_rxns_to_execute as f64;
            let time_to_run_gillespie_until =
                (self.continuous_time + time_to_run_gillespie).min(t_max);
            gillespie.advance_until_or_reactions(time_to_run_gillespie_until, None)
        };
        self.continuous_time = gillespie.get_time();
        self.gillespie_reactions_executed += executed;
        self.gillespie_reactions_targeted += num_rxns_to_execute;

        // Sync the counts rebop just changed back into self.urn. Without this, the propensity
        // calculated at the top of this function (and by active_reaction_probability in the
        // switching heuristic in run) is based on the stale configuration from when we entered
        // Gillespie mode. If the propensity has dropped a lot since then (e.g. a fast species
        // was consumed), the stale propensity makes time_to_run_gillespie so small that the
        // simulation appears to hang, advancing continuous_time by a tiny amount per call.
        self.sync_urn_from_gillespie();
    }

    /// Copy the current species counts in the rebop Gillespie object into self.urn,
    /// keeping the current count of K and setting W to 0 (W is always 0 while doing
    /// Gillespie steps, since recycle_waste is called after every batch step).
    /// Also updates self.n and self.n_including_extra_species to match.
    /// While doing Gillespie steps the true configuration lives in self.gillespie, so this
    /// must be called after each round of Gillespie steps to keep propensity calculations,
    /// the batch/Gillespie switching heuristic, and the config exposed to Python correct.
    fn sync_urn_from_gillespie(&mut self) {
        let mut gillespie_config: Vec<u64> = vec![0; self.q];
        let mut species_index = 0;
        for i in 0..self.q {
            if i == self.crn.w {
                continue;
            }
            if i == self.crn.k {
                gillespie_config[i] = self.urn.config[i];
                continue;
            }
            gillespie_config[i] = self.gillespie.as_ref().unwrap().get_species(species_index) as u64;
            species_index += 1;
        }
        self.urn.reset_config(&gillespie_config);
        self.n_including_extra_species = self.urn.size;
        self.n = self.n_including_extra_species - self.urn.config[self.crn.k] - self.urn.config[self.crn.w];
    }

    /// Helper to get the total current population in the Gillespie simulation.
    fn gillespie_total_population_count(&self) -> u64 {
        let mut total = 0;
        let gillespie = self.gillespie.as_ref().unwrap();
        for i in 0..self.q - 2 {
            total += gillespie.get_species(i) as u64;
        }
        total
    }

    /// Helper to calculate the total propensity of the CRN given the current configuration in `self.urn`.
    /// Note that this value will be the same whether we want the total propensity of the original CRN in the
    /// current configuration, or the total propensity of active reactions in the modified CRN;
    /// this must be true because the algorithm is exact, so the expected time until any active reaction must
    /// be the same as the expected time until any reaction in the original CRN.
    /// We do this calculation in the original CRN, as it is more direct.
    /// If include_passive_reactions is true, calculates the total propensity in the new CRN including these.
    fn calculate_total_propensity(&self, include_passive_reactions: bool) -> f64 {
        if include_passive_reactions {
            // The propensity of the CRN as a whole is treated as having this value.
            // This matches the value in `get_exponential_rate` on the whole population size.
            return self.crn.continuous_time_correction_factor
                * binomial_as_f64(self.n_including_extra_species, self.crn.o as u64);
        }
        let mut total_propensity = 0.0;
        for reaction in &self.crn.reactions {
            let reactants = &reaction.reactants;
            let mut num_times_reactant_seen: HashMap<State, u64> = HashMap::new();
            let mut propensity_factor_from_stoichiometry = 1.0;
            for reactant in reactants {
                assert!(*reactant != self.crn.w, "W should never be a reactant");
                // We're skipping over k, because we want to work in the original, unmodified CRN.
                if *reactant == self.crn.k {
                    continue;
                }
                *num_times_reactant_seen.entry(*reactant).or_default() += 1;
                let num_copies_of_next_reactant = self.urn.config[*reactant] - num_times_reactant_seen[&reactant] + 1;
                propensity_factor_from_stoichiometry *= num_copies_of_next_reactant as f64;
                propensity_factor_from_stoichiometry /= num_times_reactant_seen[&reactant] as f64;
            }

            let mut total_rate_constant = 0.0;
            for (_, rate_constant) in &reaction.products_and_rate_constants {
                total_rate_constant += rate_constant;
            }
            total_propensity += total_rate_constant * propensity_factor_from_stoichiometry;
        }
        total_propensity
    }

    /// Update the count of K in preparation for the next batch.
    /// We will try to choose a value for the count of K that maximizes the expected amount
    /// of progress we make in simulating the original CRN.
    fn reset_k_count(&mut self) {
        // construct_transition_arrays is the bottleneck here, so run()'s band rule calls this only
        // when the count of K has drifted significantly from its target (or on first construction).
        let current_k_count = self.urn.config[self.crn.k];
        let target_k_count = self.k_reset_target();
        let delta_k = target_k_count as i64 - current_k_count as i64;
        assert!(self.n_including_extra_species as i64 + delta_k >= 0);
        self.n_including_extra_species = (self.n_including_extra_species as i64 + delta_k) as u64;
        self.urn.add_to_entry(self.crn.k, delta_k);
        (
            self.random_transitions,
            self.random_outputs,
            self.transition_probabilities,
        ) = self.crn.construct_transition_arrays(target_k_count);
    }

    /// The count of K that `reset_k_count` aims for: the throughput-optimal `min(2n, crossover)`
    /// (generally `min(n/(o-1.5), crossover)`). This maximizes E[l]*p, the expected active
    /// reactions accomplished per (costly) batch: E[l] ~ c*sqrt(N) and p = P_real / (kmax * (N choose
    /// o)) with P_real independent of K, so minimizing the batch count means minimizing
    /// kmax(k0)*N^(o-1/2). kmax falls as 1/k0 (a padded reaction is the bottleneck) until it flattens
    /// onto the config-independent `crossover`; the objective decreases up to the interior optimum
    /// `n/(o-1.5)` (= 2n when o=2), then increases -- so the optimum is that interior value, capped at
    /// the crossover. `k0_manual_multiplier > 0` overrides this with `round(mult * n)` for K sweeps.
    fn k_reset_target(&self) -> u64 {
        if self.k0_manual_multiplier > 0.0 {
            return ((self.n as f64) * self.k0_manual_multiplier).round().max(1.0) as u64;
        }
        let o = self.crn.o as f64;
        if o >= 2.0 {
            let interior = (self.n as f64) / (o - 1.5);
            let target = interior.min(self.crossover_k0);
            if target.is_finite() && target >= 1.0 {
                return target.round() as u64;
            }
        }
        self.n.max(1) // degenerate o < 2: no order-o reaction to pad against, so no crossover
    }

    /// Get rid of W from self.urn.
    /// It is recycled to a better place.
    fn recycle_waste(&mut self) {
        let delta_w = -(self.urn.config[self.crn.w] as i64);
        assert!(self.n_including_extra_species as i64 + delta_w >= 0);
        self.n_including_extra_species = (self.n_including_extra_species as i64 + delta_w) as u64;
        self.urn.add_to_entry(self.crn.w, delta_w);
    }

    /// Sample a collision event from the urn
    /// Returns a sample l ~ coll(n, r, o, g) from the collision length distribution.
    /// See https://arxiv.org/abs/2508.04079
    /// The distribution gives the number of reactions that will occur before a collision.
    /// Inversion sampling with binary search is used, based on the formula
    ///     P(l >= t) = (n-r)! / (n-r-t*o)! * prod_{j=0}^{o-1} [(n-g-j)!(g) / (n+g(t-1)-j)!(g)].
    /// !(g) denotes a multifactorial: n!(g) = n * (n - g) * (n - 2g) * ..., until these terms become nonpositive.
    /// This is the formula when g > 0; when g = 0 or g < 0, the formulas are slightly different
    /// (see the full formula for coll(n,r,o,g) in the paper), but the method is the same:
    /// We sample a uniform random variable u, and find the value t such that
    ///     P(l >= t) < U < P(l >= t - 1).
    /// Taking logarithms and using the ln_gamma function, this required formula becomes
    ///     P(l >= t) < U
    ///       <-->
    ///     ln_gamma(n-r+1) - ln_gamma(n-r-t*o+1) + sum_{j=0}^{o-1} [log((n-g-j)!(g)) - log((n+g(t-1)-j)!(g))] < log(U).
    /// which can be rewritten by using the fact that gamma(x) = (x - 1) * gamma(x-1) even for non-integer x,
    /// by factoring out a factor of g from every term in the multifactorial.
    /// To this end, if we let a and b denote the number of terms in these multifactorial products,
    /// that is, let a = ceil((n-g-j)/g) and b = ceil((n+g(t-1)-j)/g),
    ///     ln_gamma(n-r+1) - ln_gamma(n-r-t*o+1) + sum_{j=0}^{o-1} [log(g^a * gamma((n-j)/g) / gamma((n-ag-j)/g)) - log(g^b * gamma((n+gt-j)/g) / gamma((n+g(t-b)-j)/g))] < log(U).
    ///     ln_gamma(n-r+1) - ln_gamma(n-r-t*o+1) + sum_{j=0}^{o-1} [a*log(g) + ln_gamma((n-j)/g) - ln_gamma((n-ag-j)/g) - b*log(g) - ln_gamma((n+gt-j)/g) + ln_gamma((n+g(t-b)-j)/g)] < log(U).
    /// We will do binary search with bounds t_lo, t_hi that maintain the invariant
    ///     P(l > t_hi) < U and P(l > t_lo) >= U.
    /// Once we get t_lo = t_hi - 1, we can then return t = t_hi as the output.
    ///
    /// A value of fixed outputs for u, r will be precomputed, which gives a lookup table for starting values
    /// of t_lo, t_hi. This function will first get called to give coll(n, r_i, u_i) for a fixed range of values
    /// r_i, u_i. Then actual samples of coll(n, r, u) will find values r_i <= r < r_{i+1} and u_j <= u < u_{j+1}.
    /// By monotonicity in u, r, we can then set t_lo = coll(n, r_{i+i}, u_{j+1}) and t_hi = coll(n, r_i, u_j).
    ///
    /// Args:
    ///     r: The number of agents which have already been chosen.
    ///     u: A uniform random variable.
    ///     has_bounds: Has the table for precomputed values of r, u already been computed?
    ///         (This will be false while the function is being called to populate the table.)
    /// Returns:
    ///     The number of interactions that will happen before the next collision.
    ///     Note that this is a different convention from :any:`SimulatorMultiBatch`, which
    ///     returns the index at which an agent collision occurs.
    pub fn sample_collision_fast_f128(&self, r: u64, u: f64, _has_bounds: bool) -> u64 {
        // If every agent counts as a collision, the next reaction is a collision.
        assert!(r <= self.n_including_extra_species);
        if r == self.n_including_extra_species {
            return 0;
        }
        let mut t_lo: u64;
        let mut t_hi: u64;

        let mut lhs: f128 = 0.0;

        // We take ln(u) before converting. This is fine, because we don't need high precision
        // for ln(u) itself; its lowest-order bits aren't affecting the calculation.
        // This allows the hand-rolled ln_f128 to assume its input is at least 1, since this
        // is the only call to ln on something smaller, since small inputs to ln_gamma
        // are handled by rational special casing.
        let ln_u = u.ln() as f128;
        // We *do* need precision for ln(g), because it is being multiplied by large values.
        // It's only used if g > 0.
        let ln_g = if self.crn.g > 0 {
            ln_f128(self.crn.g as f128)
        } else {
            f128::NAN
        };
        let ln_gamma_diff = ln_gamma_manual_high_precision((self.n_including_extra_species + 1 - r) as f128);
        // lhs tracks all of the terms that don't include t, i.e., those that we don't need to
        // update each iteration of binary search.

        lhs += ln_gamma_diff;
        lhs -= ln_u;

        if self.crn.g > 0 {
            for j in 0..self.crn.o {
                // Calculates a = ceil((n-g-j)/g). This is the number of terms in the expansion of
                // a multifactorial. For example, 11!^(3) (the third multifactorial of 11) is
                // 11 * 8 * 5 * 2 so there are 4 terms in it. The way we calculate the log of a
                // multifactorial is to "factor out" the amount each term decreases by (in this example 3,
                // in general it will always equal g for the multifactorials we care about) from
                // every term (whether or not they're divisible by it), then rewrite it using gamma.
                // In this example, 11 * 8 * 5 * 2 = 3^4 * (11 / 3) * (8 / 3) * (5 / 3) * (2 / 3).
                // So, log(11!^(3)) = 4*log(3) + log((11 / 3) * (8 / 3) * (5 / 3) * (2 / 3))
                // = 4*log(3) * log(Gamma(14/3) / Gamma(2/3)) [because Gamma(x) = (x-1)*Gamma(x-1)]
                // = 4*log(3) + lgamma(14/3) - lgamma(2/3).
                // These three terms are the three terms that are added and subtracted from lhs, to
                // account for the term log((n-g-j)!(g)).
                let num_static_terms: u64 = (((self.n_including_extra_species - j as u64) as f64 - self.crn.g as f64)
                    / self.crn.g as f64)
                    .ceil() as u64;
                lhs += num_static_terms as f128 * ln_g;
                lhs += ln_gamma_manual_high_precision(
                    ((self.n_including_extra_species - j as u64) as f128) / (self.crn.g as f128),
                );
                lhs -= ln_gamma_small_rational(
                    (self.n_including_extra_species - (num_static_terms * self.crn.g as u64) - j as u64) as usize,
                    self.crn.g,
                );
            }
        } else {
            // Nothing to do here. There are no other static terms in the g = 0 case.
        }

        // TODO: it might be worth adding some code to jump-start the search with precomputed values,
        // as can be done in the population protocols case.
        // For now, we start with bounds that always hold.

        t_lo = 0;
        t_hi = 1 + ((self.n_including_extra_species - r) / self.crn.o as u64);

        // Calling high-precision ln_gamma in this loop is extremely expensive at high molecular count.
        // We can get away with low-precision calls until LHS and RHS are so close that we need
        // the higher precision offered by it.
        let mut use_high_precision_in_loop: bool = false;
        // We maintain the invariant that P(l >= t_lo) >= u and P(l >= t_hi) < u.
        // It would be good to jump start this search since the first many iterations will
        // "always" go one direction, because we're going to land at O(sqrt(t_hi)) on average.
        // TODO: do this, but it's not *as* important as other things right now because
        // the loop iterations we'd manage to skip are going to be ones where we don't need
        // high precision arithmetic, so they're not the bottleneck at high pop size anyway.
        while t_lo < t_hi - 1 {
            // We know the correct value of t to search for is typically Theta(sqrt n),
            // so it's wasteful to binary search with an upper limit that is on the order of n.
            // To that end, we start by linearly searching forward in increments of sqrt(n).
            // This should quickly establish a better upper bound.
            let mut t_mid: u64;
            if t_hi > self.n_including_extra_species / (5 * self.crn.o as u64) {
                t_mid = t_lo + self.n_including_extra_species.sqrt();
                if t_mid >= t_hi {
                    t_mid = (t_lo + t_hi) / 2;
                }
            } else {
                t_mid = (t_lo + t_hi) / 2;
            }
            // rhs tracks all of the terms that include t, i.e., those that we need to
            // update each iteration of binary search.
            let mut rhs;
            // This tracks a value that we calculated from the faster (f64-based) lngamma,
            // so we can check later if we need to switch to high-precision.
            let mut last_lngamma_value: f64 = 0.0;
            if use_high_precision_in_loop {
                rhs = ln_gamma_manual_high_precision(
                    (self.n_including_extra_species - r - (t_mid * self.crn.o as u64)) as f128 + 1.0,
                );
            } else {
                last_lngamma_value =
                    ln_gamma((self.n_including_extra_species - r - (t_mid * self.crn.o as u64)) as f64 + 1.0);
                rhs = last_lngamma_value as f128;
            }
            if self.crn.g > 0 {
                for j in 0..self.crn.o {
                    // Calculates b = ceil((n+g(t-1)-j)/g).
                    // See the comment in the loop above where num_static_terms is defined for an explanation.
                    // This is the same thing, for the term log((n+g(t-1)-j)!(g)).
                    let num_dynamic_terms = (((self.n_including_extra_species + (self.crn.g as u64 * (t_mid - 1))
                        - j as u64) as f64)
                        / self.crn.g as f64)
                        .ceil() as u64;
                    rhs += (num_dynamic_terms as f128) * ln_g;
                    if use_high_precision_in_loop {
                        rhs += ln_gamma_manual_high_precision(
                            (self.n_including_extra_species + (self.crn.g as u64 * t_mid) - j as u64) as f128
                                / self.crn.g as f128,
                        );
                    } else {
                        rhs += ln_gamma(
                            (self.n_including_extra_species + (self.crn.g as u64 * t_mid) - j as u64) as f64
                                / self.crn.g as f64,
                        ) as f128;
                    }

                    rhs -= ln_gamma_small_rational(
                        (self.n_including_extra_species as isize
                            + (self.crn.g as isize * (t_mid as isize - num_dynamic_terms as isize))
                            - j as isize) as usize,
                        self.crn.g,
                    );
                }
            } else {
                // g = 0 case is much simpler; there's no multifactorial, as it's analogous
                // to the population protocols case.
                for j in 0..self.crn.o {
                    if use_high_precision_in_loop {
                        rhs += (t_mid as f128) * ln_f128((self.n_including_extra_species - j as u64) as f128);
                    } else {
                        rhs += (t_mid as f128) * ((self.n_including_extra_species - j as u64) as f64).ln() as f128;
                    }
                }
            }

            // There's a nasty floating-point precision bug here. If u (the sampled 0-1 uniform
            // value) is equal to 1.0, then lhs and rhs will fundamentally contain the same terms
            // added together in a different order. This is unavoidable, but means they might not
            // be equal as floating point numbers.
            // Of course, u will never equal 1.0, but on large enough population sizes, the values
            // of lhs and rhs have high enough magnitudes that for values of u very close to 1.0,
            // ln(u) might be smaller than the lowest-precision part of lhs and rhs. For example
            // if u = 1 - 10^-7, then ln(u) is around 10^-7, but at population size 10^9 the
            // order of magnitude of floating point error in lhs and rhs is greater than this.
            assert!(!lhs.is_nan() && !rhs.is_nan());
            // If the calculation of whether rhs or lhs might depend on f128-level precision
            // for the ln_gamma calculation, we need to start using it (including in the iteration
            // we just tried to compute).
            // There are self.crn.o calls to lngamma in each loop iteration, and each of them
            // might be wrong by around the magnitude of the computed value times epsilon.
            // Also gonna throw in a 2.5 to be safer, as 1.5 still encountered the bug.
            let potential_error = (last_lngamma_value * 2.5 * self.crn.o as f64) * f64::EPSILON;
            if !use_high_precision_in_loop && (lhs - rhs).abs() < potential_error as f128 {
                use_high_precision_in_loop = true;
                continue;
            }
            // If lhs + ln_u <= rhs, this implies there's *no* value of u that could have changed
            // the outcome of this binary search step.
            // There's only one time this makes sense: if we're in the *last step* of the search,
            // where t_mid = 1, and we are sampling a batch of size exactly 1 reaction.

            // In all cases, the expression rhs - (lhs + ln_u) gives the probability that
            // a batch would have size smaller than t_mid. So whenever t_mid > 1, this
            // value should be positive.
            assert!(
                lhs + ln_u <= rhs || (t_mid == 1),
                "lhs + ln(u) should always be less than rhs, except in the last iteration.
                lhs + ln(u) and rhs: {:?}, {:?}. Potential error: {:?}. Diff: {:?}.
                t_mid: {:?}. n = {:?}.
                This may indicate a floating point precision bug.",
                f128_to_decimal(lhs + ln_u),
                f128_to_decimal(rhs),
                potential_error,
                f128_to_decimal((lhs - rhs).abs()),
                t_mid,
                self.n,
            );

            if lhs < rhs {
                t_hi = t_mid;
            } else {
                t_lo = t_mid;
            }
        }

        // Return t_lo instead of t_hi (which simulator_pp_multibatch returns) because the CDF here
        // is written in terms of p(l >= t) instead of p(l > t).
        assert!(
            t_lo > 0,
            "Binary search should never return t_lo = 0.
            This may indicate a floating-point precision bug."
        );

        t_lo
    }

    /// Helper function to get the rate of the exponential representing the time to the next reaction
    /// at some particular population size.
    pub fn get_exponential_rate(&self, pop_size: u64) -> f64 {
        return self.crn.continuous_time_correction_factor * binomial_as_f64(pop_size, self.crn.o as u64);
    }
    /// Sample from an exponential distribution.
    pub fn sample_exponential(&mut self, rate: f64) -> f64 {
        let exp = Exp::new(rate).unwrap();
        return exp.sample(&mut self.rng);
    }

    pub fn sample_collision_fast_legacy(&self, r: u64, u: f64, _has_bounds: bool) -> u64 {
        // If every agent counts as a collision, the next reaction is a collision.
        assert!(r <= self.n_including_extra_species);
        if r == self.n_including_extra_species {
            return 0;
        }
        let mut t_lo: u64;
        let mut t_hi: u64;

        // We use a compensated summation algorithm to minimze floating point issues.
        let mut lhs = KahanBabuskaNeumaier::new();

        let logu = u.ln();
        let diff = self.n_including_extra_species + 1 - r;
        let ln_gamma_diff = ln_factorial(diff - 1);

        // lhs tracks all of the terms that don't include t, i.e., those that we don't need to
        // update each iteration of binary search.
        lhs += ln_gamma_diff;
        lhs -= logu;

        if self.crn.g > 0 {
            for j in 0..self.crn.o {
                // Calculates a = ceil((n-g-j)/g). This is the number of terms in the expansion of
                // a multifactorial. For example, 11!^(3) (the third multifactorial of 11) is
                // 11 * 8 * 5 * 2 so there are 4 terms in it. The way we calculate the log of a
                // multifactorial is to "factor out" the amount each term decreases by (in this example 3,
                // in general it will always equal g for the multifactorials we care about) from
                // every term (whether or not they're divisible by it), then rewrite it using gamma.
                // In this example, 11 * 8 * 5 * 2 = 3^4 * (11 / 3) * (8 / 3) * (5 / 3) * (2 / 3).
                // So, log(11!^(3)) = 4*log(3) + log((11 / 3) * (8 / 3) * (5 / 3) * (2 / 3))
                // = 4*log(3) * log(Gamma(14/3) / Gamma(2/3)) [because Gamma(x) = (x-1)*Gamma(x-1)]
                // = 4*log(3) + lgamma(14/3) - lgamma(2/3).
                // These three terms are the three terms that are added and subtracted from lhs, to
                // account for the term log((n-g-j)!(g)).
                let num_static_terms: f64 = (((self.n_including_extra_species - j as u64) as f64 - self.crn.g as f64)
                    / self.crn.g as f64)
                    .ceil();
                lhs += num_static_terms * (self.crn.g as f64).ln();
                lhs += ln_gamma((self.n_including_extra_species - j as u64) as f64 / self.crn.g as f64);
                lhs -= ln_gamma(
                    (self.n_including_extra_species as f64 - (num_static_terms * self.crn.g as f64) - j as f64)
                        / self.crn.g as f64,
                );
            }
        } else {
            // Nothing to do here. There are no other static terms in the g = 0 case.
        }

        // TODO: it might be worth adding some code to jump-start the search with precomputed values,
        // as can be done in the population protocols case.
        // For now, we start with bounds that always hold.

        t_lo = 0;
        t_hi = 1 + ((self.n_including_extra_species - r) / self.crn.o as u64);

        // We maintain the invariant that P(l >= t_lo) >= u and P(l >= t_hi) < u
        while t_lo < t_hi - 1 {
            let t_mid = (t_lo + t_hi) / 2;
            // rhs tracks all of the terms that include t, i.e., those that we need to
            // update each iteration of binary search.
            let mut rhs = KahanBabuskaNeumaier::new();
            rhs += ln_gamma((self.n_including_extra_species - r - (t_mid * self.crn.o as u64)) as f64 + 1.0);
            if self.crn.g > 0 {
                for j in 0..self.crn.o {
                    // Calculates b = ceil((n+g(t-1)-j)/g).
                    // See the comment in the loop above where num_static_terms is defined for an explanation.
                    // This is the same thing, for the term log((n+g(t-1)-j)!(g)).
                    let num_dynamic_terms = (((self.n_including_extra_species + (self.crn.g as u64 * (t_mid - 1))
                        - j as u64) as f64)
                        / self.crn.g as f64)
                        .ceil();
                    rhs += num_dynamic_terms * (self.crn.g as f64).ln();
                    rhs += ln_gamma(
                        (self.n_including_extra_species + (self.crn.g as u64 * t_mid) - j as u64) as f64
                            / self.crn.g as f64,
                    );
                    rhs -= ln_gamma(
                        (self.n_including_extra_species as f64
                            + (self.crn.g as f64 * (t_mid as f64 - num_dynamic_terms))
                            - j as f64)
                            / self.crn.g as f64,
                    );
                }
            } else {
                // g = 0 case is much simpler; there's no multifactorial, as it's analogous
                // to the population protocols case.
                for j in 0..self.crn.o {
                    rhs += t_mid as f64 * ((self.n_including_extra_species - j as u64) as f64).ln();
                }
            }

            // There's a nasty floating-point precision bug here. If u (the sampled 0-1 uniform
            // value) is equal to 1.0, then lhs and rhs will fundamentally contain the same terms
            // added together in a different order. This is unavoidable, but means they might not
            // be equal as floating point numbers.
            // Of course, u will never equal 1.0, but on large enough population sizes, the values
            // of lhs and rhs have high enough magnitudes that for values of u very close to 1.0,
            // ln(u) might be smaller than the lowest-precision part of lhs and rhs. For example
            // if u = 1 - 10^-7, then ln(u) is around 10^-7, but at population size 10^9 the
            // order of magnitude of floating point error in lhs and rhs is greater than this.
            if lhs.total() < rhs.total() {
                t_hi = t_mid;
            } else {
                t_lo = t_mid;
            }
        }

        // Return t_lo instead of t_hi (which simulator_pp_multibatch returns) because the CDF here
        // is written in terms of p(l >= t) instead of p(l > t).

        // TODO: this is a duct tape fix for the returning 0 bug
        if t_lo == 0 {
            return 1;
        }

        t_lo
    }

    /// Rejection sampling to ensure that we sample at exactly the right time if a batch
    /// runs past t_max. We do this by binary searching: if the simulation went past t_max
    /// in some run of l reactions, we can sample how long the first half of them took,
    /// and binary search in this manner to figure out the first index of a reaction that
    /// goes past t_max. The actual rejection sampling conditions are complicated, due to
    /// the method assuming that t must exceed t_max by the time l reactions have occurred.
    ///
    /// Args:
    ///     l: number of reactions to search within
    ///     t_max: time which is known to be exceeded within l reactions
    ///
    /// Returns:
    ///     Index of the reaction at which the total time taken first exceeds t_max.
    fn checkpoint_rejection_sampling(&mut self, l: u64, t_max: f64) -> u64 {
        // We assume (by preconditioning) initially that the CRN goes past t_max in l reactions.
        // We binary search for the index of the interaction, indexing from 0, at which it goes over.
        // Equivalently, the number of reactions that happen before it goes over.
        // We may need to change this preconditioning at some point, making a new "checkpoint".
        let mut latest_possible_collision_index = l;
        let mut time_at_checkpoint = self.continuous_time;
        let mut done_reactions_at_checkpoint = 0;
        let mut pop_size_at_checkpoint = self.n_including_extra_species;
        let mut ran_over_end_time: bool = false;
        // Special case: if l = 1, we know the answer is 0 but the below loop won't realize that,
        // as it never even gets a chance to enter the loop that can set ran_over_end_time to true.
        if l == 1 {
            return 0;
        }
        while !ran_over_end_time {
            // We're either starting for the first time, or just rejected a sample.
            let mut current_simulated_time = time_at_checkpoint;
            let mut current_simulated_reactions = done_reactions_at_checkpoint;
            let mut current_simulated_population_size = pop_size_at_checkpoint;
            while current_simulated_reactions < latest_possible_collision_index {
                let halfway_point = (latest_possible_collision_index - current_simulated_reactions + 1) / 2;
                // Check how long it would take to get halfway through the part of the batch
                // that we have yet to figure out what to do with.
                let time_to_halfway_point = self.sample_batch_time(current_simulated_population_size, halfway_point);
                // Did it run past the end in that many reactions?
                if current_simulated_time + time_to_halfway_point > t_max {
                    // If so, then we know that it ran past t_max at some point in this range.
                    // In fact, there's no way anymore for this sample to be a failure based on
                    // our initial preconditioning. So now we need to change the failure
                    // condition for the rejection sampling, essentially "locking in" *everything*
                    // we have done so far.
                    done_reactions_at_checkpoint = current_simulated_reactions;
                    latest_possible_collision_index = current_simulated_reactions + halfway_point;
                    time_at_checkpoint = current_simulated_time;
                    pop_size_at_checkpoint = current_simulated_population_size;
                    // If we ran over in the last step of binary searching, we've found the
                    // exact collision index.
                    if halfway_point == 1 {
                        ran_over_end_time = true;
                        break;
                    }
                } else {
                    // If not, then the first half of the reactions happen before a collision.
                    // We just sampled how long they'll take, so we update accordingly.
                    current_simulated_time += time_to_halfway_point;
                    // We're now sampling the remaining reaction times from a larger population size.
                    current_simulated_reactions += halfway_point;
                    current_simulated_population_size += self.crn.g as u64 * halfway_point;
                }
            }
            // If we get here and ran_over_end_time is false, that means we're rejecting a sample.
        }
        return done_reactions_at_checkpoint;
    }
}
