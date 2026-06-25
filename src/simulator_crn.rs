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
    binomial_as_f64, f128_to_decimal, ln_f128, ln_factorial, ln_gamma,
    ln_gamma_manual_high_precision, ln_gamma_small_rational, multinomial_sample,
};

type State = usize;
type RateConstant = f64;
type StateList = Vec<State>;
type Reaction = (StateList, StateList, RateConstant);
type ProductsAndRateConstant = (StateList, RateConstant);
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
    pub outputs: Vec<ProductsAndRateConstant>,
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
        let mut collected_reactions: HashMap<StateList, Vec<ProductsAndRateConstant>> =
            HashMap::new();
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
        let reactions_out: Vec<CombinedReactions> = collected_reactions
            .keys()
            .map(|x| CombinedReactions {
                reactants: x.to_vec(),
                outputs: collected_reactions[x].clone(),
            })
            .collect();
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
    /// for a @SimulatorCRNMultiBatch.
    /// We need to rebuild these tables because reaction propensities depend on the count of K,
    /// which we may want to change throughout the execution.
    /// Returns a tuple of these three objects in that order.
    fn construct_transition_arrays(
        &mut self,
        k_count: u64,
    ) -> (ArrayD<usize>, Vec<StateList>, Vec<f64>) {
        flame::start("construct_transition_arrays");
        let mut max_total_adjusted_rate_constant: f64 = 0.0;
        // Iterate through reactions, adjusting rate constants to account for how many K
        // are being added, and for symmetry that results from the scheduler having different
        // orders it can pick, so that the adjusted CRN keeps the original dynamics.
        flame::start("construct_transition_arrays: first reaction loop");
        for reaction in &self.reactions {
            let reactants = &reaction.reactants;
            let symmetry_degree = Self::symmetry_degree(reactants);
            let k_count_correction_factor = self.k_count_correction_factor(reactants, k_count);
            // By adding K as a reactant, we artificially speed up any reaction with K in it.
            // We also artificially speed up any reaction that can have its reactants written in
            // many different orders (i.e., with many distinct reactants) because the batching
            // algorithm chooses reactants one at a time, so e.g. there are twice as many ways
            // to choose an 'A' and a 'B' compared to an 'A' and an 'A', as either could be first.
            // We correct for those factors here.
            let artificial_speedup_factor: f64 =
                k_count_correction_factor as f64 / symmetry_degree as f64;
            let mut total_rate_constant = 0.0;
            for output in &reaction.outputs {
                total_rate_constant += output.1;
            }
            let total_adjusted_rate_constant = total_rate_constant / artificial_speedup_factor;
            // The batching algorithm needs to know how much continuous time one step corresponds to.
            // This function is doing the calculation that can determine this.
            // When batching, the reactant set with the highest total rate constant always causes a
            // non-passive reaction when chosen, and the number of ways to choose it is equal to
            // its propensity times its symmetry degree.
            // The missing factor to convert between the expected time to this reaction in the
            // original CRN and the batching CRN is the total rate constant.
            // TODO I'm not sure if I'm handling symmetry factor correction right here.
            // Make sure to test it on, say, a CRN with A + A -> blah and A + B -> blah,
            // in both the case where the first and the second reaction are much faster.
            if total_adjusted_rate_constant > max_total_adjusted_rate_constant {
                max_total_adjusted_rate_constant = total_adjusted_rate_constant;
                self.continuous_time_correction_factor = total_adjusted_rate_constant;
            }
        }
        flame::end("construct_transition_arrays: first reaction loop");
        // random_transitions has o+1 dimensions, the first o of which have length q,
        // and the last of which has length 2.
        let mut shape_vec = vec![self.q; self.o];
        shape_vec.push(2);
        let mut random_transitions = ArrayD::<usize>::zeros(shape_vec);
        let mut random_outputs: Vec<Vec<State>> = Vec::new();
        let mut random_probabilities: Vec<f64> = Vec::new();
        // Add any non-passive reactions. Passive reactions don't need any special handling.
        let mut cur_output_index = 0;
        flame::start("construct_transition_arrays: second reaction loop");
        for reaction in &self.reactions {
            // Add info from this reaction to all possible permutations of reactants.
            let reactants = &reaction.reactants;
            let symmetry_degree = Self::symmetry_degree(reactants);
            let k_count_correction_factor = self.k_count_correction_factor(reactants, k_count);
            let artificial_speedup_factor =
                k_count_correction_factor as f64 / symmetry_degree as f64;
            for output in &reaction.outputs {
                let probability =
                    (output.1 / artificial_speedup_factor) / max_total_adjusted_rate_constant;
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
            cur_output_index += reaction.outputs.len();
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

    /// Determine the degree of symmetry of a reaction, i.e., for any given ordering of its reactants,
    /// the number of reorderings that are redundant. Obtained as the product of the factorial of
    /// the count of each reactant.
    fn symmetry_degree(reactants: &Vec<State>) -> u64 {
        let mut factor = 1;
        let mut frequencies: HashMap<State, u64> = HashMap::new();
        for reactant in reactants {
            *frequencies.entry(*reactant).or_default() += 1;
        }
        for frequency in frequencies.values() {
            for i in 2..frequency + 1 {
                factor *= i;
            }
        }
        return factor;
    }
}
#[pyclass(extends = Simulator)]
pub struct SimulatorCRNMultiBatch {
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

    /// The current number of elapsed interaction steps that actually simulated something in
    /// the original CRN, rather than being a passive reaction,
    /// since the Simulator was created.
    /// WARNING: does not count steps from reactions simulated using Gillespie.
    #[pyo3(get, set)]
    pub discrete_batched_steps_no_passives: u64,

    /// The current number of elapsed interaction steps in this CRN, including passive reactions,
    /// since the Simulator was created.
    /// WARNING: does not count steps from reactions simulated using Gillespie.
    #[pyo3(get, set)]
    pub discrete_batched_steps_total: u64,

    /// The total number of states (length of urn.config),
    /// in the most recent call to run.
    /// WARNING: does not count steps from reactions simulated using Gillespie.
    #[pyo3(get, set)]
    pub discrete_batched_steps_no_passives_last_run: u64,

    /// The number of elapsed interaction steps in this CRN, including passive reactions,
    /// in the most recent call to run.
    /// WARNING: does not count steps from reactions simulated using Gillespie.
    #[pyo3(get, set)]
    pub discrete_batched_steps_total_last_run: u64,

    /// The total number of states (length of urn.config).
    /// Includes the auxiliary species K and W.
    pub q: usize,

    /// An (o + 1)-dimensional array. The first o dimensions represent reactants. After indexing through
    /// the first o dimensions, the last dimension always has size two, with elements (`num_outputs`, `first_idx`).
    /// `num_outputs` is the number of possible outputs if transition i,j --> ... is non-passive,
    /// otherwise it is 0. `first_idx` gives the starting index to find
    /// the outputs in the array `self.random_outputs` if it is non-passive.
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

    /// A module containing code for calling python-implemented collision sampling.
    pub python_module: Py<PyModule>,
}

#[pymethods]
impl SimulatorCRNMultiBatch {
    /// Initializes the main data structures for SimulatorCRNMultiBatch.
    /// We take numpy arrays as input because that's how n-dimensional arrays are represented in python.
    /// We convert those numpy arrays into rust ndarrray::ArrayD for storage.
    ///
    /// Args:
    ///     init_array: A length-q integer array of counts representing the initial configuration.
    ///     delta: A 2D q x q x 2 array representing the transition function. TODO remove if I can, might not be able to for consistency with other simulators.
    ///         Delta[i, j] gives contains the two output states.
    ///     random_transitions: A q^o x 2 array. That is, it has o+1 dimensions, all but the last have length q,
    ///         and the last dimension always has length two.
    ///         Entry [r, 0] is the number of possible outputs if transition on reactant set r is non-passive,
    ///         otherwise it is 0. Entry [r, 1] gives the starting index to find the outputs in the array random_outputs if it is non-passive.
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
            .map(|x| x.outputs.len())
            .fold(0, |acc, x| acc.max(x));

        let continuous_time = 0.0;
        let discrete_batched_steps_no_passives = 0;
        let discrete_batched_steps_total = 0;
        let discrete_batched_steps_no_passives_last_run = 0;
        let discrete_batched_steps_total_last_run = 0;
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
        let python_module = Python::with_gil(|py| {
            let module: Py<PyModule> = PyModule::from_code(
                py,
                python_code,
                c_str!("sample.py"),
                c_str!("sample_module"),
            )
            .unwrap()
            .into();

            module
        });

        let mut simulator = SimulatorCRNMultiBatch {
            crn,
            n,
            n_including_extra_species,
            continuous_time,
            discrete_batched_steps_no_passives,
            discrete_batched_steps_total,
            discrete_batched_steps_no_passives_last_run,
            discrete_batched_steps_total_last_run,
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
            python_module,
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

    /// Run the simulation for a specified number of steps or until max time is reached
    #[pyo3(signature = (t_max, max_wallclock_time=3600.0))]
    pub fn run(&mut self, t_max: f64, max_wallclock_time: f64) -> PyResult<()> {
        self.discrete_batched_steps_no_passives_last_run = 0;
        self.discrete_batched_steps_total_last_run = 0;
        if self.silent {
            return Err(PyValueError::new_err("Simulation is silent; cannot run."));
        }
        let max_wallclock_milliseconds = (max_wallclock_time * 1_000.0).ceil() as u64;
        let duration = Duration::from_millis(max_wallclock_milliseconds);
        let start_time = Instant::now();
        while self.continuous_time < t_max && start_time.elapsed() < duration {
            if self.silent {
                return Ok(());
            } else {
                if self.do_gillespie {
                    if self.just_started_gillespie {
                        // Initialize the Gillespie simulator.
                        // Doing Gillespie, we may as well operate in the original CRN,
                        // because it is faithfully simulated.
                        let mut gillespie_config: Vec<isize> = vec![0; self.q - 2];
                        let mut species_index = 0;
                        // Keep track of how species correspond since we need to ignore K and W.
                        let mut batching_index_to_gillespie_index: HashMap<usize, usize> =
                            HashMap::new();
                        // Iterate through species skipping k and w so that they appear in
                        // the same order in the rebop Gillespie object.
                        for i in 0..self.q {
                            if i == self.crn.k || i == self.crn.w {
                                continue;
                            }
                            batching_index_to_gillespie_index.insert(i, species_index);
                            gillespie_config[species_index] = self.urn.config[i] as isize;
                            species_index += 1;
                        }
                        // The "false" here means we aren't optimizing for the CRN to be "sparse."
                        // See https://github.com/Armavica/rebop/pull/35 for a discussion.
                        // I think the kinds of CRNs that this system is good at simulating,
                        // will typically not be sparse, but that might not be true.
                        let mut gillespie =
                            Gillespie::new_with_seed(gillespie_config, false, self.rng.next_u64());
                        // Put the reactions into the Gillespie object.
                        for reaction in &self.crn.reactions {
                            let mut rebop_reaction_inputs = vec![0; self.q - 2];
                            let mut rebop_reaction_base_deltas = vec![0; self.q - 2];
                            for reactant in &reaction.reactants {
                                if *reactant == self.crn.k {
                                    continue;
                                }
                                rebop_reaction_inputs
                                    [batching_index_to_gillespie_index[&reactant]] += 1;
                                rebop_reaction_base_deltas
                                    [batching_index_to_gillespie_index[&reactant]] -= 1;
                            }
                            for possible_output in &reaction.outputs {
                                let mut rebop_reaction_deltas = rebop_reaction_base_deltas.clone();
                                for output_species in &possible_output.0 {
                                    if *output_species == self.crn.k
                                        || *output_species == self.crn.w
                                    {
                                        continue;
                                    }
                                    rebop_reaction_deltas
                                        [batching_index_to_gillespie_index[&output_species]] += 1;
                                }
                                gillespie.add_reaction(
                                    Rate::lma(possible_output.1, &rebop_reaction_inputs),
                                    &rebop_reaction_deltas,
                                );
                            }
                        }
                        self.gillespie = Some(gillespie);
                    }
                    self.gillespie_steps(t_max);
                } else {
                    if self.just_finished_gillespie {
                        // We need to load the configuration from gillespie into the urn.
                        let mut gillespie_config: Vec<u64> = vec![0; self.q];
                        let mut species_index = 0;
                        for i in 0..self.q {
                            if i == self.crn.w {
                                continue;
                            }
                            if i == self.crn.k {
                                gillespie_config[i] = self.urn.config[i];
                            }
                            gillespie_config[i] =
                                self.gillespie.as_ref().unwrap().get_species(species_index) as u64;
                            species_index += 1;
                        }
                        self.urn.reset_config(&gillespie_config);
                        self.reset_k_count();
                    }
                    self.batch_step(t_max);
                    let current_k_count = self.urn.config[self.crn.k];
                    if (current_k_count.min(self.n) as f64) / (current_k_count.max(self.n) as f64)
                        < K_COUNT_RATIO_THRESHOLD
                    {
                        self.reset_k_count();
                    }
                    self.recycle_waste();
                }
            }
            // As a rough guideline, we will switch to Gillespie if the next batch is
            // expected to do less than some constant number of reactions.
            // The main loop in batch_step currently loops over all possible reactant vectors,
            // and certainly needs to loop over all non-passive reactions at least,
            // so we'll use the number of reactions as a conservative baseline
            // (that is to say, this will probably be too reticent to use Gillespie)
            let non_passive_probability = self.non_passive_reaction_probability();
            let rough_expected_non_passive_reactions_next_batch =
                non_passive_probability * (self.n_including_extra_species as f64).sqrt();
            if rough_expected_non_passive_reactions_next_batch < self.crn.reactions.len() as f64 {
                if self.do_gillespie {
                    self.just_started_gillespie = false;
                } else {
                    self.do_gillespie = true;
                    self.just_started_gillespie = true;
                }
            } else {
                if self.do_gillespie {
                    self.do_gillespie = false;
                    self.just_finished_gillespie = true;
                } else {
                    self.just_finished_gillespie = false;
                }
            }
            if non_passive_probability == 0.0 {
                self.silent = true;
            }
        }
        Ok(())
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
        self.n = self.n_including_extra_species
            - (self.urn.config[self.crn.k] + self.urn.config[self.crn.w]);
        if old_k_count != config[self.crn.k] {
            // If the count of k changed during the simulation, we need to do the expensive operation
            // of recomputing transition arrays.
            // Otherwise, k should already be set correctly, as it should be in the input config.
            self.reset_k_count();
        }
        self.n_including_extra_species = self.n + self.urn.config[self.crn.k];
        self.continuous_time = t;
        self.discrete_batched_steps_no_passives = 0;
        self.discrete_batched_steps_total = 0;
        self.discrete_batched_steps_no_passives_last_run = 0;
        self.discrete_batched_steps_total_last_run = 0;
        self.silent = self.n == 0;
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

    #[pyo3(signature = (r, u, has_bounds=false))]
    pub fn sample_collision(&self, r: u64, u: f64, has_bounds: bool) -> u64 {
        return self.sample_collision_fast_f128(r, u, has_bounds);
    }

    /// Sample from collision distribution using python's mpmath library, which allows us to
    /// do arbitrary-precision floating point arithmetic without rolling our own loggamma function.
    pub fn sample_collision_fast_python(&self, r: u64, u: f64, _has_bounds: bool) -> u64 {
        let args = (self.n_including_extra_species, r, self.crn.o, self.crn.g, u);
        let result = Python::with_gil(|py| -> PyResult<u64> {
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
        let last_term =
            1.0 / self.get_exponential_rate(initial_n + self.crn.g as u64 * (batch_size - 1));
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
            answer += self
                .sample_exponential(self.get_exponential_rate(initial_n + i * self.crn.g as u64));
        }
        return answer;
    }
    /// Calculate the probability that a reaction in the next batch will be non-passive.
    /// Done by manually adding up all of the possible vectors that might be sampled,
    /// essentially executing the same loop as in batch_step.
    /// There is some code duplication from this, which is tough to avoid.
    /// Mutable because it resets self.array_sums in order to easily iterate through reactions.
    pub fn non_passive_reaction_probability(&mut self) -> f64 {
        // As a hack to avoid making a new NDBatchResult, we just sample 0 reactions into this one.
        // This function should not be called in the middle of batch sampling,
        // so this step should be safe to do.
        self.array_sums.reset_contents();
        // These aren't probabilities, they're parts of some total space that will be normalized later.
        let mut total_passive_probability_mass = 0.0;
        let mut total_non_passive_probability_mass = 0.0;
        let mut done = false;
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
            let reactants = next_array_sum.0;
            // We weight by the probability that this reaction will actually happen next,
            // with these reactants in this order,
            // non-passively. This is similar to calculating its propensity, as it's
            // the number of ways to choose these reactants (in this order)
            // divided by the total number of ways to choose reactants.
            // Since this division is by a constant, we omit it.
            let mut num_times_seen: HashMap<State, u64> = HashMap::new();
            let mut num_ways_to_choose_these_reactants_in_order = 1.0;
            for reactant in reactants {
                *num_times_seen.entry(reactant).or_default() += 1;
                let num_copies_of_next_reactant =
                    self.urn.config[reactant] - num_times_seen[&reactant] - 1;
                num_ways_to_choose_these_reactants_in_order *= num_copies_of_next_reactant as f64;
                num_ways_to_choose_these_reactants_in_order /= num_times_seen[&reactant] as f64;
            }
            done = next_array_sum.2;
            let (num_outputs, first_idx) = (random_transition[0], random_transition[1]);
            // TODO and WARNING: this code is more or less copy-paste with the collision sampling code.
            // They do the same thing. But it's apparently very annoying to refactor this into a
            // helper method in rust because of the immutable borrow of self above.
            if num_outputs == 0 {
                // Passive reaction.
                total_passive_probability_mass += num_ways_to_choose_these_reactants_in_order;
            } else {
                let probabilities =
                    self.transition_probabilities[first_idx..first_idx + num_outputs].to_vec();
                let non_passive_probability_sum: f64 = probabilities.iter().sum();
                total_passive_probability_mass +=
                    non_passive_probability_sum * num_ways_to_choose_these_reactants_in_order;
                total_non_passive_probability_mass += (1.0 - non_passive_probability_sum)
                    * num_ways_to_choose_these_reactants_in_order;
            }
        }
        assert!(
            done,
            "self.random_transitions finished iterating before self.array_sums"
        );
        let total_probability_mass =
            total_passive_probability_mass + total_non_passive_probability_mass;
        total_non_passive_probability_mass / total_probability_mass
    }
}

// TODO: if we're using this, figure out a sensible value for it? It will only matter really
// for CRNs whose population size fluctuates a lot.
const K_COUNT_RATIO_THRESHOLD: f64 = 0.5;
// DD: I'm confused about a phenomenon; if this is 0.5, the fraction of passive reactions with LV
// (with rate constant 1.5 for the F+R-->2F reaction) starts around 0.4 and drops, then increases
// hovering closer to 0.5. However, if we set this to 0.33, then it starts around 0.4 and *increases*;
// This is just the way we decide when to reset K, so I don't understand why setting it differently
// would change the behavior of the simulation *before* the first switch, i.e., why the fraction of
// passive reactions goes up in one case and down in the other.

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
            "{}{:name_length$}: {} ms\n",
            indent,
            span_data.name,
            span_data.ns / 1_000_000
        ));
        write_span_data(content, &span_data.children, depth + 1);
    }
}

struct SpanData {
    name: String,
    ns: u64,
    children: HashMap<String, SpanData>,
}

impl SpanData {
    fn new(name: String) -> Self {
        SpanData {
            name,
            ns: 0,
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
    /// Reset the contents of this batch result, as though it sampled a batch of size 0.
    fn reset_contents(&mut self) {
        self.counts = vec![0; self.q];
        self.curr_species = 0;
        if self.dimensions > 1 {
            for i in 0..self.q {
                let subresults = self.subresults.as_mut().unwrap();
                subresults[i].reset_contents();
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
        assert!(
            self.curr_species < self.q,
            "NDBatchResult iterated past final species"
        );
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

impl SimulatorCRNMultiBatch {
    fn batch_step(&mut self, t_max: f64) -> () {
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
        let batch_time = self.sample_batch_time(self.n_including_extra_species, l);
        let mut do_collision = true;
        if self.continuous_time + batch_time <= t_max {
            self.continuous_time += batch_time;
            // It's possible that all of the reactions *except* the collision happen before t_max,
            // but then the collision happens after t_max.
            let collision_time = self.sample_exponential(
                self.get_exponential_rate(self.n_including_extra_species + self.crn.g as u64 * l),
            );
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

            // let mut time_exceeded = false;
            // while !time_exceeded {
            //     let mut partial_batch_time = 0.0;
            //     for i in 0..l {
            //         let rate =
            //             self.get_exponential_rate(self.n_including_extra_species + i * self.crn.g);
            //         partial_batch_time += self.sample_exponential(rate);
            //         if partial_batch_time + self.continuous_time > t_max {
            //             time_exceeded = true;
            //             rxns_before_coll = i;
            //             break;
            //         }
            //     }
            // }
            self.continuous_time = t_max;
            flame::end("checkpoint rejection sampling");
        }

        // The idea here is to iterate through random_transitions and array_sums together; they should
        // both be indexed by q^o-tuples when iterated through this way, and the iteration order should
        // be lexicographic for both of them.
        flame::start("sample batch");
        self.array_sums
            .sample_batch_result(rxns_before_coll, &mut self.urn);
        flame::end("sample batch");

        let initial_t_including_passives = self.discrete_batched_steps_total;
        flame::start("process batch");
        let mut done = false;
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
            flame::start("pre-reaction-checking");
            let next_array_sum = self.array_sums.get_next();
            // TODO maybe add an assert check that the two structures are iterated through
            // in the same order, i.e. reactants match
            let (reactants, quantity) = (next_array_sum.0, next_array_sum.1);
            done = next_array_sum.2;
            if quantity == 0 {
                flame::end("pre-reaction-checking");
                continue;
            }
            let initial_updated_counts_size = self.updated_counts.size;
            let (num_outputs, first_idx) = (random_transition[0], random_transition[1]);
            // TODO and WARNING: this code is more or less copy-paste with the collision sampling code.
            // They do the same thing. But it's apparently very annoying to refactor this into a
            // helper method in rust because of the immutable borrow of self above.
            flame::end("pre-reaction-checking");
            if num_outputs == 0 {
                flame::start("passive reaction");
                // Passive reaction. Move the reactants from self.urn to self.updated_counts (for collision sampling), and add W.
                for reactant in reactants {
                    self.updated_counts.add_to_entry(reactant, quantity as i64);
                }
                self.updated_counts
                    .add_to_entry(self.crn.w, (quantity * self.crn.g as u64) as i64);
                flame::end("passive reaction");
            } else {
                flame::start("non-passive reaction");
                // We'll add this for now, then subtract off the probabilistically passive reactions later.
                self.discrete_batched_steps_no_passives += quantity;
                self.discrete_batched_steps_no_passives_last_run += quantity;
                let mut probabilities =
                    self.transition_probabilities[first_idx..first_idx + num_outputs].to_vec();
                let non_passive_probability_sum: f64 = probabilities.iter().sum();
                if non_passive_probability_sum < 1.0 {
                    probabilities.push(1.0 - non_passive_probability_sum);
                }
                flame::start("multinomial sample");
                multinomial_sample(
                    quantity,
                    &probabilities,
                    &mut self.m[0..probabilities.len()],
                    &mut self.rng,
                );
                flame::end("multinomial sample");
                assert_eq!(
                    self.m[0..probabilities.len()].iter().sum::<u64>(),
                    quantity,
                    "sample sum mismatch"
                );
                for offset in 0..num_outputs {
                    let idx = first_idx + offset;
                    let outputs = &self.random_outputs[idx];
                    for output in outputs {
                        self.updated_counts
                            .add_to_entry(*output, self.m[offset] as i64);
                    }
                }
                // Add any W produced by passive reactions, and add those reactants to updated_counts.
                if non_passive_probability_sum < 1.0 {
                    let passive_count = self.m[num_outputs];
                    self.updated_counts
                        .add_to_entry(self.crn.w, (passive_count * self.crn.g as u64) as i64);
                    for reactant in reactants {
                        self.updated_counts
                            .add_to_entry(reactant, passive_count as i64);
                    }
                    self.discrete_batched_steps_no_passives -= passive_count;
                    self.discrete_batched_steps_no_passives_last_run -= passive_count;
                }
                flame::end("non-passive reaction");
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
                    (num_old_molecules as u128).pow(
                        (self.crn.o - num_updated_reactants_in_collision)
                            .try_into()
                            .unwrap(),
                    ) * (num_new_molecules as u128)
                        .pow(num_updated_reactants_in_collision.try_into().unwrap())
                        * binomial(self.crn.o as u64, num_updated_reactants_in_collision as u64)
                            as u128,
                );
                //XXX: note that binomial is from the num_integer crate and returns a u64:
                // https://docs.rs/num-integer/latest/num_integer/fn.binomial.html
                // use this only for small inputs, and use util::binomial_as_f64 for large inputs
                // where the result might not fit into a u64.
            }
            // TODO: there should be some standard way to sample from this discrete probability distribution.
            // Should be rand::WeightedIndex. Also urn::sample_one.
            let total_ways_with_at_least_one_collision: u128 =
                collision_count_num_ways.iter().sum();
            let u2: f64 = self.rng.sample(StandardUniform);
            let mut num_colliding_molecules = 0;
            let mut total_ways_so_far = 0;
            for i in 0..self.crn.o {
                total_ways_so_far += collision_count_num_ways[i];
                if u2 < (total_ways_so_far as f64) / (total_ways_with_at_least_one_collision as f64)
                {
                    num_colliding_molecules = i + 1;
                    break;
                }
            }
            assert!(
                num_colliding_molecules > 0,
                "Failed to sample collision size"
            );
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
            assert_eq!(
                view.len(),
                2,
                "Indexing collision did not leave two-element subarray"
            );

            let (num_outputs, first_idx) = (view[0], view[1]);
            // TODO: this code is heavily copy-pasted. See other TODO comment above.
            if num_outputs == 0 {
                // Passive reaction.
                for reactant in collision {
                    self.updated_counts.add_to_entry(reactant, 1);
                }
                self.updated_counts
                    .add_to_entry(self.crn.w, (self.crn.g) as i64);
            } else {
                self.discrete_batched_steps_no_passives += 1;
                self.discrete_batched_steps_no_passives_last_run += 1;
                let mut probabilities =
                    self.transition_probabilities[first_idx..first_idx + num_outputs].to_vec();
                let non_passive_probability_sum: f64 = probabilities.iter().sum();
                if non_passive_probability_sum < 1.0 {
                    probabilities.push(1.0 - non_passive_probability_sum);
                }
                flame::start("multinomial sample");
                multinomial_sample(
                    1,
                    &probabilities,
                    &mut self.m[0..probabilities.len()],
                    &mut self.rng,
                );
                flame::end("multinomial sample");
                assert_eq!(
                    self.m[0..probabilities.len()].iter().sum::<u64>(),
                    1,
                    "sample sum mismatch"
                );
                for c in 0..num_outputs {
                    let idx = first_idx + c;
                    let outputs = &self.random_outputs[idx];
                    for offset in outputs {
                        self.updated_counts.add_to_entry(*offset, self.m[c] as i64);
                    }
                }
                // Add W if the collision was a probabilistic passive reaction.
                if non_passive_probability_sum < 1.0 {
                    let passive_count = self.m[num_outputs];
                    self.updated_counts
                        .add_to_entry(self.crn.w, (passive_count * self.crn.g as u64) as i64);
                    for reactant in collision {
                        self.updated_counts
                            .add_to_entry(reactant, passive_count as i64);
                    }
                    self.discrete_batched_steps_no_passives -= passive_count;
                    self.discrete_batched_steps_no_passives_last_run -= passive_count;
                }
            }
            assert_eq!(
                self.updated_counts.size - num_new_molecules,
                (self.crn.o + self.crn.g) as u64 - num_resampled,
                "Collision failed to add exactly g things to updated_counts"
            );
            self.discrete_batched_steps_total += 1;
            self.discrete_batched_steps_total_last_run += 1;
        }
        flame::end("sample collision");

        // TODO move this line to a more sensible place
        self.discrete_batched_steps_total += rxns_before_coll;
        self.discrete_batched_steps_total_last_run += rxns_before_coll;

        self.urn.add_vector(&self.updated_counts.config);
        self.urn.sort();
        // Check that we added the right number of things to the urn.
        assert_eq!(
            self.urn.size - self.n_including_extra_species,
            (self.discrete_batched_steps_total - initial_t_including_passives) * self.crn.g as u64,
            "Inconsistency between number of reactions simulated and population size change."
        );
        assert_eq!(
            initial_k_count, self.urn.config[self.crn.k],
            "Count of K should never change within running a batch."
        );
        self.n_including_extra_species = self.urn.size;
        self.n = self.n_including_extra_species
            - self.urn.config[self.crn.k]
            - self.urn.config[self.crn.w];

        //self.update_enabled_reactions();
    }

    /// Perform some Gillespie steps.
    /// The exact number of steps performed is not deterministic;
    /// we run Gillespie for roughly sqrt(n) steps, and then re-check whether batching
    /// is likely to be faster again.
    fn gillespie_steps(&mut self, t_max: f64) -> () {
        let original_gillespie_population = self.total_gillespie_population();
        assert!(
            original_gillespie_population == self.n,
            "self.n ({:?}) does not match gillespie value of n ({:?})",
            self.n,
            original_gillespie_population
        );
        const NUM_CONST_GILLESPIE_STEPS: usize = 30;
        {
            // Borrow mutably for as long as we need it, for convenience.
            let gillespie = self.gillespie.as_mut().unwrap();
            gillespie.set_time(self.continuous_time);
            // We'll just run a small constant number of steps, see how much (CRN) time that took,
            // and use that as an estimate for how long we need to run to get the desired
            // number of steps. Doing this because I'm not confident whether calling
            // advance_one_reaction in a loop will be slower than advance_until,
            // and curerntly the rebop API doesn't have a function to advance a fixed number of steps.
            let mut last_configuration = vec![0; self.q - 2];
            for _ in 0..NUM_CONST_GILLESPIE_STEPS {
                // This is pretty grossly inefficient but, for correctness,
                // we want to make sure we don't accidentally simulate a reaction past t_max,
                // so we make sure we can undo if needed.
                // This loop shouldn't be a big part of runtime anyway.
                for species in 0..self.q - 2 {
                    last_configuration[species] = gillespie.get_species(species);
                }
                gillespie.advance_one_reaction();
                if gillespie.get_time() > t_max {
                    self.continuous_time = t_max;
                    gillespie.set_species(last_configuration);
                    break;
                }
            }
        }
        // Unless we're already over t_max after that small number of steps,
        // run enough to simulate around sqrt(n) additional reactions.
        if self.continuous_time < t_max {
            let cur_gillespie_time = self.gillespie.as_ref().unwrap().get_time();
            let elapsed_time = cur_gillespie_time - self.continuous_time;
            let time_per_step = elapsed_time / NUM_CONST_GILLESPIE_STEPS as f64;
            // For now, we're going to assume that we will need to do, say,
            // at least O(sqrt n) reactions until it's worth turning on batching again.
            let num_steps_to_take = self.n.sqrt();
            let time_to_run_gillespie = time_per_step * num_steps_to_take as f64;
            let time_to_run_gillespie_until =
                (cur_gillespie_time + time_to_run_gillespie).min(t_max);
            self.gillespie
                .as_mut()
                .unwrap()
                .advance_until(time_to_run_gillespie_until);
            self.continuous_time = self.gillespie.as_ref().unwrap().get_time();
        }

        // Update fields that have changed.
        let new_gillespie_population = self.total_gillespie_population();
        let delta_n = new_gillespie_population - original_gillespie_population;
        self.n += delta_n;
        self.n_including_extra_species += delta_n;
        // We'd like to update self.discrete_batched_steps_no_passives and
        // self.discrete_batched_steps_total, but rebop does not keep track of how many reactions
        // it simulates, so at least for now those variables only count batched steps.
    }

    /// Helper to get the total current population in the Gillespie simulation.
    fn total_gillespie_population(&self) -> u64 {
        let mut total = 0;
        for i in 0..self.q - 2 {
            total += self.gillespie.as_ref().unwrap().get_species(i) as u64;
        }
        total
    }

    /// Update the count of K in preparation for the next batch.
    /// We will try to choose a value for the count of K that maximizes the expected amount
    /// of progress we make in simulating the original CRN.
    fn reset_k_count(&mut self) {
        // TODO: maybe do something more complicated than this to actually be efficient.
        // For now, it looks like construct_transition_arrays is a bottleneck, so we're only going
        // to change the count of K if the population size has significantly changed, or
        // we're constructing them for the first time or resetting
        let current_k_count = self.urn.config[self.crn.k];
        let target_k_count = self.n;
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
        let ln_gamma_diff =
            ln_gamma_manual_high_precision((self.n_including_extra_species + 1 - r) as f128);
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
                let num_static_terms: u64 = (((self.n_including_extra_species - j as u64) as f64
                    - self.crn.g as f64)
                    / self.crn.g as f64)
                    .ceil() as u64;
                lhs += num_static_terms as f128 * ln_g;
                lhs += ln_gamma_manual_high_precision(
                    ((self.n_including_extra_species - j as u64) as f128) / (self.crn.g as f128),
                );
                lhs -= ln_gamma_small_rational(
                    (self.n_including_extra_species
                        - (num_static_terms * self.crn.g as u64)
                        - j as u64) as usize,
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
            let t_mid = (t_lo + t_hi) / 2;
            // rhs tracks all of the terms that include t, i.e., those that we need to
            // update each iteration of binary search.
            let mut rhs;
            // This tracks a value that we calculated from the faster (f64-based) lngamma,
            // so we can check later if we need to switch to high-precision.
            let mut last_lngamma_value: f64 = 0.0;
            if use_high_precision_in_loop {
                rhs = ln_gamma_manual_high_precision(
                    (self.n_including_extra_species - r - (t_mid * self.crn.o as u64)) as f128
                        + 1.0,
                );
            } else {
                last_lngamma_value = ln_gamma(
                    (self.n_including_extra_species - r - (t_mid * self.crn.o as u64)) as f64 + 1.0,
                );
                rhs = last_lngamma_value as f128;
            }
            if self.crn.g > 0 {
                for j in 0..self.crn.o {
                    // Calculates b = ceil((n+g(t-1)-j)/g).
                    // See the comment in the loop above where num_static_terms is defined for an explanation.
                    // This is the same thing, for the term log((n+g(t-1)-j)!(g)).
                    let num_dynamic_terms = (((self.n_including_extra_species
                        + (self.crn.g as u64 * (t_mid - 1))
                        - j as u64) as f64)
                        / self.crn.g as f64)
                        .ceil() as u64;
                    rhs += (num_dynamic_terms as f128) * ln_g;
                    if use_high_precision_in_loop {
                        rhs += ln_gamma_manual_high_precision(
                            (self.n_including_extra_species + (self.crn.g as u64 * t_mid)
                                - j as u64) as f128
                                / self.crn.g as f128,
                        );
                    } else {
                        rhs += ln_gamma(
                            (self.n_including_extra_species + (self.crn.g as u64 * t_mid)
                                - j as u64) as f64
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
                        rhs += (t_mid as f128)
                            * ln_f128((self.n_including_extra_species - j as u64) as f128);
                    } else {
                        rhs += (t_mid as f128)
                            * ((self.n_including_extra_species - j as u64) as f64).ln() as f128;
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
        return self.crn.continuous_time_correction_factor
            * binomial_as_f64(pop_size, self.crn.o as u64);
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
                let num_static_terms: f64 = (((self.n_including_extra_species - j as u64) as f64
                    - self.crn.g as f64)
                    / self.crn.g as f64)
                    .ceil();
                lhs += num_static_terms * (self.crn.g as f64).ln();
                lhs += ln_gamma(
                    (self.n_including_extra_species - j as u64) as f64 / self.crn.g as f64,
                );
                lhs -= ln_gamma(
                    (self.n_including_extra_species as f64
                        - (num_static_terms * self.crn.g as f64)
                        - j as f64)
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
            rhs += ln_gamma(
                (self.n_including_extra_species - r - (t_mid * self.crn.o as u64)) as f64 + 1.0,
            );
            if self.crn.g > 0 {
                for j in 0..self.crn.o {
                    // Calculates b = ceil((n+g(t-1)-j)/g).
                    // See the comment in the loop above where num_static_terms is defined for an explanation.
                    // This is the same thing, for the term log((n+g(t-1)-j)!(g)).
                    let num_dynamic_terms = (((self.n_including_extra_species
                        + (self.crn.g as u64 * (t_mid - 1))
                        - j as u64) as f64)
                        / self.crn.g as f64)
                        .ceil();
                    rhs += num_dynamic_terms * (self.crn.g as f64).ln();
                    rhs += ln_gamma(
                        (self.n_including_extra_species + (self.crn.g as u64 * t_mid) - j as u64)
                            as f64
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
                let halfway_point =
                    (latest_possible_collision_index - current_simulated_reactions + 1) / 2;
                // Check how long it would take to get halfway through the part of the batch
                // that we have yet to figure out what to do with.
                let time_to_halfway_point =
                    self.sample_batch_time(current_simulated_population_size, halfway_point);
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
