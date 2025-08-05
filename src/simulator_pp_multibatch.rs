use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::time::{Duration, Instant};

use crate::flame;

use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyNone;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Geometric, StandardUniform, Uniform};

use crate::simulator_abstract::Simulator;

use crate::urn::Urn;
#[allow(unused_imports)]
use crate::util::{ln_factorial, ln_gamma, multinomial_sample};

type State = usize;

//TODO: consider using ndarrays instead of multi-dimensional vectors
// I think native Rust arrays won't work because their size needs to be known at compile time.
#[pyclass(extends = Simulator)]
pub struct SimulatorMultiBatch {
    /// The population size (sum of values in urn.config).
    #[pyo3(get, set)]
    pub n: usize,
    /// The current number of elapsed interaction steps.
    #[pyo3(get, set)]
    pub t: usize,
    /// The total number of states (length of urn.config).
    pub q: usize,
    /// A q x q array of pairs (c,d) representing the transition function.
    /// delta[a][b] gives contains the two output states for a
    /// deterministic transition a,b --> c,d.
    #[pyo3(get, set)]
    pub delta: Vec<Vec<(State, State)>>,
    /// A q x q boolean array where null_transitions[i][j] says if these states have a null interaction.
    #[pyo3(get, set)]
    pub null_transitions: Vec<Vec<bool>>,
    /// A boolean that is true if there are any random transitions.
    pub is_random: bool,
    /// A q x q array of pairs random_transitions[i][j] = (`num_outputs`, `first_idx`).
    /// `num_outputs` is the number of possible outputs if transition i,j --> ... is random,
    /// otherwise it is 0. `first_idx` gives the starting index to find
    /// the outputs in the array `self.random_outputs` if it is random.
    #[pyo3(get, set)] // XXX: for testing
    pub random_transitions: Vec<Vec<(usize, usize)>>,
    /// A 1D array of pairs containing all (out1,out2) outputs of random transitions,
    /// whose indexing information is contained in random_transitions.
    /// For example, if there are random transitions
    /// 3,4 --> 5,6 and 3,4 --> 7,8 and 3,4 --> 3,2, then
    /// `random_transitions[3][4] = (3, first_idx)` for some `first_idx`, and
    /// `random_outputs[first_idx]   = (5,6)`,
    /// `random_outputs[first_idx+1] = (7,8)`, and
    /// `random_outputs[first_idx+2] = (3,2)`.
    #[pyo3(get, set)] // XXX: for testing
    pub random_outputs: Vec<(State, State)>,
    /// An array containing all random transition probabilities,
    /// whose indexing matches random_outputs.
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
    updated_counts: Urn,
    /// Precomputed log(n).
    logn: f64,
    /// Minimum number of interactions that must be simulated in each
    /// batch. Collisions will be repeatedly sampled up until batch_threshold
    /// interaction steps, then all non-colliding pairs of 'delayed agents' are
    /// processed in parallel.
    batch_threshold: usize,
    /// Array which stores sampled counts of initiator agents
    /// (row sums of the 'D' matrix from the paper).
    row_sums: Vec<usize>,
    /// Array which stores the counts of responder agents for each type of
    /// initiator agent (one row of the 'D' matrix from the paper).
    row: Vec<usize>,
    /// Vector holding multinomial samples when doing randomized transitions.
    m: Vec<usize>,
    /// A boolean determining if we are currently doing Gillespie steps.
    #[pyo3(get, set)]
    pub do_gillespie: bool,
    /// A boolean determining if the configuration is silent (all interactions are null).
    #[pyo3(get, set)]
    pub silent: bool,
    /// A list of reactions, as (input, input, output, output).
    #[pyo3(get, set)]
    pub reactions: Vec<(State, State, State, State)>,
    /// An array holding indices into `self.reactions` of all currently enabled
    /// (i.e., applicable; positive counts of reactants) reactions.
    #[pyo3(get, set)]
    pub enabled_reactions: Vec<usize>,
    /// The number of meaningful indices in `self.enabled_reactions`.
    #[pyo3(get, set)]
    pub num_enabled_reactions: usize,
    /// An array of length `self.reactions.len()` holding the propensities of each reaction.
    propensities: Vec<f64>, // these are used only when doing Gillespie steps and are all 0 otherwise
    /// The probability of each reaction.
    #[pyo3(get, set)]
    pub reaction_probabilities: Vec<f64>,
    /// The probability of a non-null interaction must be below this
    /// threshold to keep doing Gillespie steps.
    gillespie_threshold: f64,
    /// Precomputed values to speed up the function sample_coll(r, u).
    /// This is a 2D array of size (`coll_table_r_values.len()`, `coll_table_u_values.len()`).
    coll_table: Vec<Vec<usize>>,
    /// Values of r, giving one axis of coll_table.
    coll_table_r_values: Vec<usize>,
    /// Values of u, giving the other axis of coll_table.
    coll_table_u_values: Vec<f64>,
    /// Used to populate coll_table_r_values.
    r_constant: usize,
    /// If true, unconditionally use the Gillespie algorithm.
    gillespie_always: bool,
}

#[pymethods]
impl SimulatorMultiBatch {
    /// Initializes the main data structures for SimulatorMultiBatch.
    /// We take numpy arrays as input because that's how the original Cython implementation
    /// worked and I wanted to change as little as possible. But this is awkward because
    /// we don't work the the numpy arrays; we convert them to Rust vectors.
    /// It would be more straightforward to accept native Python types, but
    /// for now let's keep it this way to avoid re-writing the code in simulation.py.
    ///
    /// Args:
    ///     init_array: A 2D length-q integer array of counts representing the initial configuration.
    ///     delta: A 2D q x q x 2 array representing the transition function.
    ///         Delta[i, j] gives contains the two output states.
    ///     null_transitions: A 2D q x q boolean array where entry [i, j] says if these states have a null interaction.
    ///     random_transitions: A 2D q x q x 2 array. Entry [i, j, 0] is the number of possible outputs if
    ///         transition [i, j] is random, otherwise it is 0. Entry [i, j, 1] gives the starting index to find
    ///         the outputs in the array random_outputs if it is random.
    ///     random_outputs: A ? x 2 array containing all (out1,out2) outputs of random transitions,
    ///         whose indexing information is contained in random_transitions.
    ///     transition_probabilities: A 1D length-? array containing all random transition probabilities,
    ///         whose indexing matches random_outputs.
    ///     seed (optional): An integer seed for the pseudorandom number generator.
    #[new]
    #[pyo3(signature = (init_config, delta, random_transitions, random_outputs, transition_probabilities, transition_order, gillespie, seed, _reactions, _k, _w))]
    pub fn new(
        init_config: PyReadonlyArray1<State>,
        delta: PyReadonlyArray3<State>,
        random_transitions: PyReadonlyArray3<usize>,
        random_outputs: PyReadonlyArray2<State>,
        transition_probabilities: PyReadonlyArray1<f64>,
        transition_order: String,
        gillespie: bool,
        seed: Option<u64>,
        _reactions: Py<PyNone>,
        _k: Py<PyNone>,
        _w: Py<PyNone>,
    ) -> (Self, Simulator) {
        let init_config = init_config.to_vec().unwrap();
        let q: usize = init_config.len() as State;

        assert_eq!(delta.shape()[0], q, "delta shape mismatch");
        assert_eq!(delta.shape()[1], q, "delta shape mismatch");
        assert_eq!(delta.shape()[2], 2 as usize, "delta shape mismatch");
        let mut delta_vec = Vec::with_capacity(q);
        for i in 0..q {
            let mut delta_inner_vec = Vec::with_capacity(q);
            for j in 0..q {
                let out1 = *delta.get([i, j, 0]).unwrap();
                let out2 = *delta.get([i, j, 1]).unwrap();
                delta_inner_vec.push((out1, out2));
            }
            delta_vec.push(delta_inner_vec);
        }
        let delta = delta_vec;

        let mut random_transitions_vec = Vec::with_capacity(q);
        for i in 0..q {
            let mut random_inner_vec = Vec::with_capacity(q);
            for j in 0..q {
                let num = *random_transitions.get([i, j, 0]).unwrap();
                let idx = *random_transitions.get([i, j, 1]).unwrap();
                random_inner_vec.push((num, idx));
            }
            random_transitions_vec.push(random_inner_vec);
        }
        let random_transitions = random_transitions_vec;

        let random_outputs_length = random_outputs.shape()[0];
        assert_eq!(
            random_outputs.shape()[1],
            2 as usize,
            "random_outputs shape mismatch"
        );
        let mut random_outputs_vec = Vec::with_capacity(random_outputs_length);
        for i in 0..random_outputs_length {
            let out1 = *random_outputs.get([i, 0]).unwrap();
            let out2 = *random_outputs.get([i, 1]).unwrap();
            random_outputs_vec.push((out1, out2));
        }
        let random_outputs = random_outputs_vec;

        let transition_probabilities = transition_probabilities.to_vec().unwrap();
        assert_eq!(
            random_outputs.len(),
            transition_probabilities.len(),
            "random_outputs and transition_probabilities length mismatch"
        );

        (
            SimulatorMultiBatch::from_delta_random(
                delta,
                init_config,
                random_transitions,
                random_outputs,
                transition_probabilities,
                transition_order,
                gillespie,
                seed,
            ),
            Simulator::default(),
        )
    }

    #[getter]
    pub fn config(&self) -> Vec<State> {
        self.urn.config.clone()
    }

    /// Run the simulation for a specified number of steps or until max time is reached
    #[pyo3(signature = (t_max, max_wallclock_time=3600.0))]
    pub fn run(&mut self, t_max: usize, max_wallclock_time: f64) -> PyResult<()> {
        if self.silent {
            return Err(PyValueError::new_err("Simulation is silent; cannot run."));
        }
        let max_wallclock_milliseconds = (max_wallclock_time * 1_000.0).ceil() as u64;
        let duration = Duration::from_millis(max_wallclock_milliseconds);
        let start_time = Instant::now();
        while self.t < t_max && start_time.elapsed() < duration {
            // println!("self.gillespie_always = {}", self.gillespie_always);
            if self.gillespie_always {
                self.do_gillespie = true;
            }
            if self.silent {
                return Ok(());
            } else if self.do_gillespie {
                flame::start("gillespie step");
                self.gillespie_step(t_max);
                flame::end("gillespie step");
            } else {
                self.multibatch_step(t_max);
            }
        }
        Ok(())
    }

    /// Run the simulation until it is silent, i.e., no reactions are applicable.
    #[pyo3()]
    pub fn run_until_silent(&mut self) {
        while !self.silent {
            if self.gillespie_always {
                self.do_gillespie = true;
            }
            if self.do_gillespie {
                self.gillespie_step(0);
            } else {
                self.multibatch_step(0);
            }
        }
    }

    /// Reset the simulation with a new configuration
    /// Sets all parameters necessary to change the configuration.
    /// Args:
    ///     config: The configuration array to reset to.
    ///     t: The new value of :any:`t`. Defaults to 0.
    #[pyo3(signature = (config, t=0))]
    pub fn reset(&mut self, config: PyReadonlyArray1<State>, t: usize) -> PyResult<()> {
        let config = config.to_vec().unwrap();
        self.urn.reset_config(&config);
        let n: usize = config.iter().sum();
        if n != self.n {
            self.n = n;
            self.set_n_parameters();
        }
        self.t = t;
        self.update_enabled_reactions();
        self.do_gillespie = self.gillespie_always;
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

    #[pyo3(signature = (r, u, has_bounds=false, pp=false))]
    pub fn sample_collision(&self, r: usize, u: f64, has_bounds: bool, pp: bool) -> usize {
        self.sample_coll(r, u, has_bounds, pp)
    }

    /// Sample from birthday distribution "directly". This is the number of times
    /// we can sample with replacement from a set of size n, given that r objects
    /// have already been sampled. If `pp` is true, we use the slight variant of
    /// this where the sample events are numbered 1,2, 3,4, 5,6, ..., and we
    /// automatically rule out collisions of the form i,i+1 for odd i. This corresponds
    /// to sampling for a population protocol, where the initiator is chosen on
    /// step i (for odd i), and the responder is chosen on step i+1, but guaranteed
    /// to be different from the initiator.
    #[pyo3()]
    pub fn sample_collision_directly(&mut self, n: usize, r: usize, pp: bool) -> usize {
        let mut idx = 0usize;
        let mut seen = HashSet::new();
        assert!(r < n, "r must be less than n");
        assert!(n < usize::MAX, "n must be less than usize::MAX");
        let mut prev_sample = usize::MAX;
        loop {
            idx += 1;
            let sample = self.rng.random_range(0..n);
            if sample < r {
                return idx;
            }
            if seen.contains(&sample) {
                if !pp || idx % 2 == 1 || sample != prev_sample {
                    return idx;
                }
            }
            seen.insert(sample);
            prev_sample = sample;
        }
    }
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

const CAP_BATCH_THRESHOLD: bool = true;

impl SimulatorMultiBatch {
    fn multibatch_step(&mut self, t_max: usize) -> () {
        let max_batch_threshold = self.n / 4;
        if CAP_BATCH_THRESHOLD && self.batch_threshold > max_batch_threshold {
            self.batch_threshold = max_batch_threshold;
        }
        self.updated_counts.reset();

        for i in 0..self.urn.order.len() {
            self.updated_counts.order[i] = self.urn.order[i];
        }

        // start with count 2 of delayed agents (guaranteed for the next interaction)
        let mut num_delayed: usize = 2;

        // let now = Instant::now();
        // let t1 = now.elapsed().as_secs_f64();

        // batch will go for at least batch_threshold interactions, unless passing t_max
        let mut end_step = self.t + self.batch_threshold;
        if t_max > 0 {
            end_step = end_step.min(t_max);
        }

        flame::start("process collisions");

        while self.t + num_delayed / 2 < end_step {
            let mut u: f64 = self.rng.sample(StandardUniform);

            let pp = true;
            let has_bounds = false;
            flame::start("sample_coll");
            let l = self.sample_coll(num_delayed + self.updated_counts.size, u, has_bounds, pp);
            flame::end("sample_coll");

            assert!(l > 0, "sample_coll must return at least 1");

            // add (l-1) // 2 pairs of delayed agents, the lth agent a was already picked, so has a collision
            num_delayed += 2 * ((l - 1) / 2);

            // If the sampled collision happens after t_max, then include delayed agents up until t_max
            //   and do not perform the collision.
            if t_max > 0 && self.t + num_delayed / 2 >= t_max {
                assert!(t_max > self.t);
                num_delayed = (t_max - self.t) * 2;
                break;
            }

            /*
            Definitions from paper https://arxiv.org/abs/2005.03584

            - *untouched* agents did not interact in the current epoch (multibatch step).
              Hence, all agents are labeled untouched at the beginning of an epoch.

            - *updated* agents took part in at least one interaction that was already evaluated.
              Thus, updated agents are already assigned their most recent state.

            - *delayed* agents took part in exactly one interaction that was not yet evaluated.
              Thus, delayed agents are still in the same state they had at the beginning of the
              epoch, but are scheduled to interact at a later point in time. We additionally
              require that their interaction partner is also labeled delayed.
             */

            let mut initiator: State; // initiator, called a in Cython implementation
            let mut responder: State; // responder, called b in Cython implementation

            flame::start("process collision");

            // sample if initiator was delayed or updated
            u = self.rng.sample(StandardUniform);
            // initiator is delayed with probability num_delayed / (num_delayed + num_updated)
            let initiator_delayed =
                u * ((num_delayed + self.updated_counts.size) as f64) < num_delayed as f64;
            if initiator_delayed {
                // if initiator is delayed, need to first update it with its first interaction before the collision
                // c is the delayed partner that initiator interacted with, so add this interaction
                initiator = self.urn.sample_one().unwrap();
                let mut c = self.urn.sample_one().unwrap();
                (initiator, c) = self.unordered_delta(initiator, c);
                self.t += 1;
                // c is moved from delayed to updated, initiator is currently uncounted;
                // we've updated initiator state, but don't put it in updated_counts because
                // we'd just need to take it back out to do the initiator/responder interaction
                self.updated_counts.add_to_entry(c, 1);
                num_delayed -= 2;
            } else {
                // if initiator is updated, we simply sample initiator and remove it from updated_counts
                initiator = self.updated_counts.sample_one().unwrap();
            }

            // sample responder
            if l % 2 == 0 {
                // when l is even, the collision must with a formerly untouched agent
                responder = self.urn.sample_one().unwrap();
            } else {
                // when l is odd, the collision is with the next agent, either untouched, delayed, or updated
                u = self.rng.sample(StandardUniform);
                if (u * ((self.n - 1) as f64)) < self.updated_counts.size as f64 {
                    // responder is an updated agent, simply remove it
                    responder = self.updated_counts.sample_one().unwrap();
                } else {
                    // responder is untouched or delayed; we remove responder from self.urn in either case
                    responder = self.urn.sample_one().unwrap();
                    // if responder is delayed, we also have to do the past interaction
                    if (u * (self.n - 1) as f64) < (self.updated_counts.size + num_delayed) as f64 {
                        let mut c = self.urn.sample_one().unwrap();
                        (responder, c) = self.unordered_delta(responder, c);
                        self.t += 1;
                        self.updated_counts.add_to_entry(c, 1);
                        num_delayed -= 2;
                    }
                }
            }

            (initiator, responder) = self.unordered_delta(initiator, responder);
            self.t += 1;
            self.updated_counts.add_to_entry(initiator, 1);
            self.updated_counts.add_to_entry(responder, 1);

            flame::end("process collision");
        }

        flame::end("process collisions");

        flame::start("process batch");

        self.do_gillespie = true; // if entire batch are null reactions, stays true and switches to gillspie

        let i_max = self
            .urn
            .sample_vector(num_delayed / 2, &mut self.row_sums)
            .unwrap();

        for i in 0..i_max {
            let o_i = self.urn.order[i];
            let j_max = self
                .urn
                .sample_vector(self.row_sums[o_i], &mut self.row)
                .unwrap();

            for j in 0..j_max {
                let o_j = self.urn.order[j];
                if self.is_random && self.random_transitions[o_i][o_j].0 > 0 {
                    // don't switch to gillespie because we did a random transition
                    // TODO: this might not switch to gillespie soon enough in certain cases
                    // better to test if the random transition is null or not
                    self.do_gillespie = false;
                    let (num_outputs, first_idx) = self.random_transitions[o_i][o_j];
                    // updates the first num_outputs entries of sample to hold a multinomial,
                    // giving the number of times for each random transition
                    let probabilities =
                        self.transition_probabilities[first_idx..first_idx + num_outputs].to_vec();
                    flame::start("multinomial sample");
                    multinomial_sample(
                        self.row[o_j],
                        &probabilities,
                        &mut self.m[0..num_outputs],
                        &mut self.rng,
                    );
                    flame::end("multinomial sample");
                    assert_eq!(
                        self.m.iter().sum::<usize>(),
                        self.row[o_j],
                        "sample sum mismatch"
                    );
                    for c in 0..num_outputs {
                        let idx = first_idx + c;
                        let (out1, out2) = self.random_outputs[idx];
                        self.updated_counts.add_to_entry(out1, self.m[c] as i64);
                        self.updated_counts.add_to_entry(out2, self.m[c] as i64);
                    }
                } else {
                    if self.do_gillespie {
                        // if transition is non-null, we will set do_gillespie = False
                        self.do_gillespie = self.null_transitions[o_i][o_j];
                    }
                    // We are directly adding to updated_counts.config rather than using the function
                    //   updated_counts.add_to_entry for speed. None of the other urn features of updated_counts will
                    //   be used until it is reset in the next loop, so this is fine.
                    self.updated_counts.config[self.delta[o_i][o_j].0] += self.row[o_j];
                    self.updated_counts.config[self.delta[o_i][o_j].1] += self.row[o_j];
                }
            }
        }

        self.t += num_delayed / 2;
        // TODO: this is the only part scaling when the number of states (but not reached states) blows up
        self.urn.add_vector(&self.updated_counts.config);

        flame::end("process batch");

        self.urn.sort();
        self.update_enabled_reactions();
    }

    /// Chooses sender/receiver, then applies delta to input states a, b.
    fn unordered_delta(&mut self, a: State, b: State) -> (State, State) {
        let heads = self.rng.random_bool(0.5); // fair coin flip
        let mut i1 = a;
        let mut i2 = b;
        // swap roles of a, b and swap return order if heads is true
        if heads {
            (i2, i1) = (i1, i2);
        }
        let o1: State;
        let o2: State;
        if self.is_random && self.random_transitions[i1][i2].0 > 0 {
            // find the appropriate random output by linear search
            let mut k = self.random_transitions[i1][i2].1;
            let mut u =
                self.rng.sample::<f64, _>(StandardUniform) - self.transition_probabilities[k];
            while u > 0.0 {
                k += 1;
                u -= self.transition_probabilities[k];
            }
            (o1, o2) = self.random_outputs[k];
        } else {
            (o1, o2) = self.delta[i1][i2];
        }
        // swap outputs if heads is true
        if heads {
            (o2, o1)
        } else {
            (o1, o2)
        }
    }

    /// Perform a Gillespie step.
    /// Samples the time until the next non-null interaction and updates.
    /// Args:
    /// num_steps:
    ///     If positive, the maximum value of :any:`t` that will be reached.
    ///     If the sampled time is greater than num_steps, then it will instead
    ///     be set to num_steps and no reaction will be performed.
    ///     (Because of the memoryless property of the geometric, this gives a
    ///     faithful simulation up to step num_steps).
    fn gillespie_step(&mut self, t_max: usize) -> () {
        // println!("gillespie_step at interaction {}", self.t);
        let total_propensity = self.get_total_propensity();
        if total_propensity == 0.0 {
            self.silent = true;
            return;
        }
        let n_choose_2 = (self.n * (self.n - 1) / 2) as f64;
        let success_probability = total_propensity / n_choose_2;

        if success_probability > self.gillespie_threshold {
            self.do_gillespie = false;
        }
        let geometric = Geometric::new(success_probability).unwrap();
        let uniform = Uniform::new(0.0, total_propensity).unwrap();
        // add a geometric number of steps, based on success probability
        let steps: u64 = self.rng.sample(geometric);
        self.t += steps as usize;
        if t_max > 0 && self.t > t_max {
            self.t = t_max;
            return;
        }
        // sample the successful reaction r, currently just using linear search
        let mut x: f64 = self.rng.sample(uniform);
        let mut i = 0;
        while x > 0.0 {
            x -= self.propensities[self.enabled_reactions[i]];
            i += 1;
        }
        let (r1, r2, p1, p2) = self.reactions[self.enabled_reactions[i - 1]];

        // update with the successful reaction r1+r2 --> p1+p2
        // if any products were not already present, or any reactants went absent, will update enabled_reactions
        let new_products = self.urn.config[p1] == 0 || self.urn.config[p2] == 0;
        let absent_reactants = self.urn.config[r1] == 0 || self.urn.config[r2] == 0;
        if new_products || absent_reactants {
            self.update_enabled_reactions();
        }

        // now apply the reaction
        self.urn.add_to_entry(r1, -1);
        self.urn.add_to_entry(r2, -1);
        self.urn.add_to_entry(p1, 1);
        self.urn.add_to_entry(p2, 1);
    }

    /// Updates propensity vector, and returns total propensity:
    /// the probability the next interaction is non-null.
    fn get_total_propensity(&mut self) -> f64 {
        let mut total_propensity = 0.0;
        for j in 0..self.num_enabled_reactions {
            let i = self.enabled_reactions[j];
            let a = self.urn.config[self.reactions[i].0] as f64;
            let b = self.urn.config[self.reactions[i].1] as f64;
            if self.reactions[i].0 == self.reactions[i].1 {
                self.propensities[i] = (a * (a - 1.0) / 2.0) * self.reaction_probabilities[i];
            } else {
                self.propensities[i] = a * b * self.reaction_probabilities[i];
            }
            total_propensity += self.propensities[i];
        }
        total_propensity
    }

    /// Updates :any:`enabled_reactions`, :any:`num_enabled_reactions`, and :any:`silent`.
    fn update_enabled_reactions(&mut self) -> () {
        // flame::start("update_enabled_reactions");
        self.num_enabled_reactions = 0;
        for i in 0..self.reactions.len() {
            let (reactant_1, reactant_2) = (self.reactions[i].0, self.reactions[i].1);
            if (reactant_1 == reactant_2 && self.urn.config[reactant_1] >= 2)
                || (reactant_1 != reactant_2
                    && self.urn.config[reactant_1] >= 1
                    && self.urn.config[reactant_2] >= 1)
            {
                self.enabled_reactions[self.num_enabled_reactions] = i;
                self.num_enabled_reactions += 1;
            }
        }
        self.silent = self.num_enabled_reactions == 0;
        // flame::end("update_enabled_reactions");
    }

    /// Initialize all parameters that depend on the population size n.
    fn set_n_parameters(&mut self) -> () {
        self.logn = (self.n as f64).ln();
        // theoretical optimum for batch_threshold is Theta(sqrt(n / logn) * q) agents / batch
        // let batch_constant = 2_i32.pow(2) as usize;
        let batch_constant = 1 as usize;
        self.batch_threshold = batch_constant
            * ((self.n as f64 / self.logn).sqrt() * (self.q as f64).min((self.n as f64).powf(0.7)))
                as usize;
        // println!("batch_threshold = {}", self.batch_threshold);
        self.batch_threshold = self.n / 2;
        // first rough approximation for probability of successful reaction where we want to do gillespie
        self.gillespie_threshold = 2.0 / (self.n as f64).sqrt();

        // build table for precomputed coll(n, r, u) values
        // Note num_attempted_r_values may be too large; we break early if r >= n.
        // let mut num_r_values = (10.0 * self.logn) as usize;
        let mut num_r_values = (5.0 * self.logn) as usize;
        let num_u_values = num_r_values;

        self.r_constant = (((1.5 * self.batch_threshold as f64) as usize)
            / ((num_r_values - 2) * (num_r_values - 2)))
            .max(1) as usize;

        self.coll_table_r_values = vec![];
        for idx in 0..num_r_values - 1 {
            let r = 2 + self.r_constant * idx * idx;
            if r >= self.n {
                break;
            }
            self.coll_table_r_values.push(r);
        }
        self.coll_table_r_values.push(self.n);
        num_r_values = self.coll_table_r_values.len();

        self.coll_table_u_values = vec![0.0; num_u_values];
        for i in 0..num_u_values {
            self.coll_table_u_values[i] = i as f64 / (num_u_values as f64 - 1.0);
        }

        assert_eq!(
            self.coll_table_r_values.len(),
            num_r_values,
            "self.coll_table_r_values length mismatch",
        );
        assert_eq!(
            self.coll_table_u_values.len(),
            num_u_values,
            "self.coll_table_u_values length mismatch",
        );

        self.coll_table = vec![vec![0; num_u_values]; num_r_values];
        for r_idx in 0..num_r_values {
            for u_idx in 0..num_u_values {
                let r = self.coll_table_r_values[r_idx];
                let u = self.coll_table_u_values[u_idx];
                self.coll_table[r_idx][u_idx] = self.sample_coll(r, u, false, true);
            }
        }
    }

    /// Sample a collision event from the urn
    /// Returns a sample l ~ coll(n, r) from the collision length distribution.
    /// See Section 5.1 in the source paper https://arxiv.org/pdf/2005.03584.
    /// The distribution gives the number of agents needed to pick an agent twice,
    /// when r unique agents have already been selected.
    /// Inversion sampling with binary search is used, based on the formula
    ///     P(l > t) = (n-r)! / (n-r-t)! * n^-t. (slightly incorrect CDF from paper)
    /// NOTE: actually the correct PDF, accounting for the sequential scheduler never picking
    /// the same agent for interactions i,i+1 for odd i (assuming first interaction is i=1), is
    ///     P(l > t) = (n-r)! / (n-r-t)! * n^-ceil(t/2) * (n-1)^-floor(t/2)
    /// We sample a uniform random variable u, and find the value t such that
    ///     P(l > t) < U < P(l > t - 1).
    /// Taking logarithms and using the ln_factorial function, this required formula becomes
    ///     P(l > t) < U
    ///       <-->
    ///     ln_factorial(n-r) - ln_factorial(n-r-t) - t*log(n) < log(u). (still incorrect PDF from paper)
    /// (correcting for correct PDF)
    ///     ln_factorial(n-r) - ln_factorial(n-r-t) - (ceil(t/2)*log(n) + floor(t/2)*log(n-1)) < log(u).
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
    ///     pp: If true, we do not consider collisions of the form i,i+1 for odd i,
    ///         corresponding to the sequential schedule picking an initiator and responder
    ///         who are guaranteed to be distinct.
    /// Returns:
    ///     The number of sampled agents to get the first collision (including the collided agent).
    pub fn sample_coll(&self, r: usize, u: f64, has_bounds: bool, pp: bool) -> usize {
        let mut t_lo: usize;
        let mut t_hi: usize;
        let logu = u.ln();
        assert!(self.n + 1 - r > 0);
        let diff = self.n + 1 - r;

        // let ln_gamma_diff = ln_gamma(diff);
        let ln_gamma_diff = ln_factorial(diff - 1);

        let lhs = ln_gamma_diff - logu;
        // The condition P(l < t) < U becomes
        //     lhs <     lgamma(n-r-t+1) + t*log(n)
        //     lhs < ln_factorial(n-r-t) + t*log(n)

        const PRINT: bool = true;
        let logn_minus_1 = ((self.n - 1) as f64).ln();

        if has_bounds {
            if PRINT {
                use stybulate::{Cell, Headers, Style, Table};
                let mut headers: Vec<String> = vec!["".to_string()];
                let mut first_row = vec![Cell::from(" r\\u")];
                for i in 0..self.coll_table_u_values.len() {
                    // headers.push(format!("{:.2}", self.coll_table_u_values[i]));
                    headers.push(i.to_string());
                    first_row.push(Cell::from(&format!("{:.2}", self.coll_table_u_values[i])));
                }
                let mut table_data: Vec<Vec<Cell>> = vec![first_row];
                for i in 0..self.coll_table.len() {
                    let first = format!("{i:2}:{}", self.coll_table_r_values[i].to_string());
                    let mut row = vec![Cell::from(&first)];
                    for j in 0..self.coll_table[i].len() {
                        let entry = format!("{}", self.coll_table[i][j]);
                        row.push(Cell::from(&entry));
                    }
                    table_data.push(row);
                }
                let headers_ref: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
                let _table = Table::new(Style::Plain, table_data, Some(Headers::from(headers_ref)));
                println!("coll_table:\n{}", _table.tabulate());
            }

            // Look up bounds from coll_table.
            // For r values, we invert the definition of self.coll_table_r_values:
            //   [2 + self.r_constant * (i ** 2) for i in range(self.num_r_values - 1)] + [self.n]
            let i = (((r - 2) as f64) / self.r_constant as f64).sqrt() as usize;
            let i = i.min(self.coll_table_r_values.len() - 2);

            // for u values we similarly invert the definition: np.linspace(0, 1, num_u_values)
            let j = (u * (self.coll_table_u_values.len() - 1) as f64) as usize;

            t_lo = self.coll_table[i + 1][j + 1];
            t_hi = self.coll_table[i][j];
            t_hi = t_hi.min(self.n - r + 1);
            if t_lo == 1 && t_hi == 1 {
                t_lo = 1;
                t_hi = 2;
            }

            if PRINT {
                println!(
                    "n = {}, self.r_constant={}, u={u:.3}, r={r} i={i} j={j} t_lo={t_lo}, t_hi={t_hi} (((r - 2) as f64) / self.r_constant as f64).sqrt()={:.3}",
                    self.n,
                    self.r_constant,
                    (((r - 2) as f64) / self.r_constant as f64).sqrt()
                );
                println!(
                    "lhs = {lhs:.3}, logn={:.1}, t_lo*logn={:.1}, log_factorial(n-r-t_lo)={:.1}, log_factorial(n-r-t_lo) + t_lo*logn={:.3}",
                    self.logn,
                    t_lo as f64 * self.logn,
                    ln_factorial(self.n - r - t_lo),
                    ln_factorial(self.n - r - t_lo) + (t_lo as f64 * self.logn)
                );
            }
            assert!(t_lo < t_hi);
            assert!(self.coll_table_r_values[i] <= r);
            assert!(r <= self.coll_table_r_values[i + 1]);
            assert!(self.coll_table_u_values[j] <= u);
            assert!(u <= self.coll_table_u_values[j + 1]);

            if t_lo < t_hi - 1 {
                assert!(lhs >= ln_factorial(self.n - r - t_lo - 1) + (t_lo as f64 * logn_minus_1));
                assert!(lhs < ln_factorial(self.n - r - t_hi) + (t_hi as f64 * self.logn));
            }
        } else {
            // When building the table, we start with bounds that always hold.
            if r >= self.n {
                return 1;
            }
            t_lo = 0;
            t_hi = self.n - r;
        }

        // We maintain the invariant that P(l > t_lo) >= u and P(l > t_hi) < u
        // Equivalently, lhs >= lgamma(n-r-t_lo+1) + t_lo*logn and
        //               lhs <  lgamma(n-r-t_hi+1) + t_hi*logn
        // Equivalently, lhs >= ln_factorial(n-r-t_lo) + t_lo*logn and
        //               lhs <  ln_factorial(n-r-t_hi) + t_hi*logn
        // Correcting for PDF,
        //               lhs >= ln_factorial(n-r-t_lo) + ceil(t_lo/2)*logn + floor(t_hi/2)*log(n-1), and
        //               lhs <  ln_factorial(n-r-t_hi) + ceil(t_lo/2)*logn + floor(t_hi/2)*log(n-1)
        while t_lo < t_hi - 1 {
            let t_mid = (t_lo + t_hi) / 2;

            // let ln_gamma_nr1 = ln_gamma(self.n - r + 1 - t_mid);
            let ln_gamma_nr1 = ln_factorial(self.n - r - t_mid);

            let rhs: f64;
            if pp {
                // ceil(t_mid / 2) * logn + floor(t_mid / 2) * log(n - 1)
                // This corrects the PDF of the distribution replacing the term
                // n^-t with n^-ceil(t/2) * (n-1)^-floor(t/2), to account for the
                // fact that the sequential scheduler cannot pick the same agent
                // at indices i,i+1 for odd i, assuming we start at index i=1.
                rhs = ln_gamma_nr1
                    + (((t_mid + 1) / 2) as f64) * self.logn
                    + ((t_mid / 2) as f64) * logn_minus_1;
            } else {
                rhs = ln_gamma_nr1 + (t_mid as f64) * self.logn;
            }

            if lhs < rhs {
                t_hi = t_mid;
            } else {
                t_lo = t_mid;
            }
        }

        t_hi
    }

    ///     init_array: A 2D length-q integer array of counts representing the initial configuration.
    ///     delta: A 2D q x q x 2 array representing the transition function.
    ///         Delta[i, j] gives contains the two output states.
    ///     null_transitions: A 2D q x q boolean array where entry [i, j] says if these states have a null interaction.
    ///     random_transitions: A 2D q x q x 2 array. Entry [i, j, 0] is the number of possible outputs if
    ///         transition [i, j] is random, otherwise it is 0. Entry [i, j, 1] gives the starting index to find
    ///         the outputs in the array random_outputs if it is random.
    ///     random_outputs: A ? x 2 array containing all (out1,out2) outputs of random transitions,
    ///         whose indexing information is contained in random_transitions.
    ///     transition_probabilities: A 1D length-? array containing all random transition probabilities,
    ///         whose indexing matches random_outputs.
    ///     seed (optional): An integer seed for the pseudorandom number generator.

    /// This is an easier to use constructor taking native Rust types instead of numpy arrays,
    /// but otherwise it works similarly to the `new` constructor.
    pub fn from_delta_random(
        delta: Vec<Vec<(State, State)>>,
        init_config: Vec<usize>,
        random_transitions: Vec<Vec<(usize, usize)>>,
        random_outputs: Vec<(State, State)>,
        transition_probabilities: Vec<f64>,
        transition_order: String,
        gillespie: bool,
        seed: Option<u64>,
    ) -> Self {
        let mut delta = delta.clone();
        let mut random_transitions = random_transitions.clone();

        let config = init_config.clone();
        let n = config.iter().sum();
        let q = config.len() as State;

        assert_eq!(delta.len(), q, "delta shape mismatch");
        assert_eq!(delta[0].len(), q, "delta shape mismatch");

        let mut null_transitions = vec![vec![false; q]; q];
        for i1 in 0..q {
            for i2 in 0..q {
                let (o1, o2) = delta[i1][i2];
                null_transitions[i1][i2] = i1 == o1 && i2 == o2;
            }
        }
        let transition_order = transition_order.to_lowercase();
        if ["symmetric".to_string(), "symmetric_enforced".to_string()].contains(&transition_order) {
            for i1 in 0..q {
                for i2 in 0..q {
                    if null_transitions[i1][i2] {
                        // Set the output for i, j to be equal to j, i if null
                        null_transitions[i1][i2] = null_transitions[i2][i1];
                        delta[i1][i2] = delta[i2][i1];
                        random_transitions[i1][i2] = random_transitions[i2][i1];
                    } else if transition_order.to_lowercase() == "symmetric_enforced"
                        && !null_transitions[i2][i1]
                    {
                        // If i, j and j, i are both non-null, with symmetric_enforced, check outputs are equal
                        let (o1, o2) = delta[i1][i2];
                        let (o1_p, o2_p) = delta[i2][i1];
                        if (o1, o2) != (o1_p, o2_p) {
                            panic!("Asymmetric interaction despite symmetric_enforced: {i1},{i2} -> {o1},{o2} {i2},{i1} -> {o2_p},{o1_p}");
                        }
                        if random_transitions[i1][i2] != random_transitions[i2][i1] {
                            panic!("Asymmetric interaction despite symmetric_enforced for inputs: {i1},{i2}");
                        }
                    }
                }
            }
        }

        // is_random is true if any num in pair (num, idx) in random_transitions is non-zero
        let mut is_random = false;
        // random_depth is the maximum number of outputs for any randomized transition
        let mut random_depth = 1;
        for random_transitions_inner in &random_transitions {
            for &(num, _) in random_transitions_inner {
                if num != 0 {
                    is_random = true;
                    random_depth = random_depth.max(num);
                }
            }
        }

        if is_random {
            assert_eq!(
                random_outputs.len(),
                transition_probabilities.len(),
                "random_outputs and transition_probabilities length mismatch"
            );
        }

        let t = 0;
        let rng = if let Some(s) = seed {
            SmallRng::seed_from_u64(s)
        } else {
            SmallRng::from_os_rng()
        };

        let urn = Urn::new(config.clone(), seed);
        let updated_counts = Urn::new(vec![0; q], seed);
        let row_sums = vec![0; q];
        let row = vec![0; q];
        let m = vec![0; random_depth];
        let silent = false;
        let do_gillespie = false; // this changes during run
        let gillespie_always = gillespie; // this never changes; if True we always do Gillespie steps

        let mut reactions: Vec<(usize, usize, usize, usize)> = vec![];
        let mut reaction_probabilities = vec![];
        for i in 0..q {
            for j in 0..=i {
                // check if interaction is symmetric
                let mut symmetric = false;
                // Check that entries in delta array match
                let (mut o1, mut o2) = delta[i][j];
                if o1 > o2 {
                    (o1, o2) = (o2, o1);
                }
                let (mut o1_p, mut o2_p) = delta[j][i];
                if o1_p > o2_p {
                    (o1_p, o2_p) = (o2_p, o1_p);
                }
                if o1 == o1_p && o2 == o2_p {
                    // Check if those really were matching deterministic transitions
                    if !is_random
                        || (random_transitions[i][j].0 == 0 && random_transitions[j][i].0 == 0)
                    {
                        symmetric = true;
                    } else if is_random
                        && random_transitions[i][j].0 == random_transitions[j][i].0
                        && random_transitions[i][j].0 > 0
                    {
                        let (a, b) = (random_transitions[i][j].1, random_transitions[j][i].1);
                        symmetric = true;
                        for k in 0..random_transitions[i][j].0 {
                            let (mut o1, mut o2) = random_outputs[a + k];
                            if o1 > o2 {
                                (o1, o2) = (o2, o1);
                            }
                            let (mut o1_p, mut o2_p) = random_outputs[b + k];
                            if o1_p > o2_p {
                                (o1_p, o2_p) = (o2_p, o1_p);
                            }
                            if o1 != o1_p || o2 != o2_p {
                                symmetric = false;
                                // break;
                            }
                        }
                    }
                }
                // Other cases are not symmetric, such as a different number of random outputs based on order
                let indices = if symmetric {
                    vec![(i, j, 1.0)]
                } else {
                    // if interaction is not symmetric, each distinct order gets added as reactions with half probability
                    vec![(i, j, 0.5), (j, i, 0.5)]
                };
                for (a, b, p) in indices.iter() {
                    let a = *a;
                    let b = *b;
                    let p = *p;
                    if !null_transitions[a][b] {
                        let (num_outputs, start_idx) = random_transitions[a][b];
                        if is_random && num_outputs > 0 {
                            for k in 0..num_outputs {
                                let output = random_outputs[start_idx + k];
                                if output != (a, b) {
                                    reactions.push((a, b, output.0, output.1));
                                    reaction_probabilities
                                        .push(transition_probabilities[start_idx + k] * p);
                                }
                            }
                        } else {
                            reactions.push((a, b, o1, o2));
                            reaction_probabilities.push(p);
                        }
                    }
                }
            }
        }

        // next three fields are only used with Gillespie steps;
        // they will be set accordingly if we switch to Gillespie
        let propensities = vec![0.0; reactions.len()];
        let enabled_reactions = vec![0; reactions.len()];
        let num_enabled_reactions = 0;

        // below here we give meaningless default values to the other fields and rely on
        // set_n_parameters and get_enabled_reactions to set them to the correct values
        let logn = 0.0;
        let batch_threshold = 0;
        let gillespie_threshold = 0.0;
        let coll_table = vec![vec![0; 1]; 1];
        let coll_table_r_values = vec![0; 1];
        let coll_table_u_values = vec![0.0; 1];
        let r_constant = 0;

        let mut sim = SimulatorMultiBatch {
            n,
            t,
            q,
            delta,
            null_transitions,
            is_random,
            random_transitions,
            random_outputs,
            transition_probabilities,
            random_depth,
            rng,
            urn,
            updated_counts,
            logn,
            batch_threshold,
            row_sums,
            row,
            m,
            do_gillespie,
            silent,
            reactions,
            enabled_reactions,
            num_enabled_reactions,
            propensities,
            reaction_probabilities,
            gillespie_threshold,
            coll_table,
            coll_table_r_values,
            coll_table_u_values,
            r_constant,
            gillespie_always,
        };
        sim.set_n_parameters();
        sim.update_enabled_reactions();
        sim
    }

    /// This one especially is easier since it assumes a deterministic transition function.
    pub fn from_delta_deterministic(
        delta: Vec<Vec<(State, State)>>,
        init_config: Vec<usize>,
        transition_order: String,
        gillespie: bool,
        seed: Option<u64>,
    ) -> Self {
        let random_transitions: Vec<Vec<(usize, usize)>> =
            vec![vec![(0, 0); delta.len()]; delta.len()];
        let random_outputs: Vec<(State, State)> = vec![(0, 0); 1];
        let transition_probabilities: Vec<f64> = vec![0.0; 1];
        SimulatorMultiBatch::from_delta_random(
            delta,
            init_config,
            random_transitions,
            random_outputs,
            transition_probabilities,
            transition_order,
            gillespie,
            seed,
        )
    }
}
