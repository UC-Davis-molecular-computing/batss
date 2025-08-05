use std::time::{Duration, Instant};

use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardUniform;

use crate::urn::sample_discrete_uniform;

use crate::simulator_abstract::Simulator;

type State = usize;

#[pyclass(extends = Simulator)]
pub struct SimulatorSequentialArray {
    #[pyo3(get, set)]
    pub config: Vec<State>,
    #[pyo3(get, set)]
    pub n: usize,
    #[pyo3(get, set)]
    pub t: usize,
    pub q: usize,
    #[pyo3(get, set)]
    pub delta: Vec<Vec<(State, State)>>,
    #[pyo3(get, set)]
    pub null_transitions: Vec<Vec<bool>>,
    pub is_random: bool,
    pub random_transitions: Vec<Vec<(State, State)>>,
    pub random_outputs: Vec<(usize, usize)>,
    pub transition_probabilities: Vec<f64>,
    pub random_depth: usize,
    rng: SmallRng,
    #[pyo3(get, set)]
    pub population: Vec<State>,
}

#[pymethods]
impl SimulatorSequentialArray {
    #[new]
    #[pyo3(signature = (init_config, delta, null_transitions, random_transitions, random_outputs, transition_probabilities, gillespie=false, seed=None))]
    pub fn new(
        init_config: PyReadonlyArray1<State>,
        delta: PyReadonlyArray3<State>,
        null_transitions: PyReadonlyArray2<bool>,
        random_transitions: PyReadonlyArray3<State>,
        random_outputs: PyReadonlyArray2<State>,
        transition_probabilities: PyReadonlyArray1<f64>,
        gillespie: bool,
        seed: Option<u64>,
    ) -> (Self, Simulator) {
        assert!(
            gillespie == false,
            "gillespie = True is not supported for SimulatorSequentialArray"
        );
        let config = init_config.to_vec().unwrap();
        let transition_probabilities = transition_probabilities.to_vec().unwrap();
        let n: usize = config.iter().sum();
        let q = config.len() as State;
        let t = 0;
        let rng = if let Some(s) = seed {
            SmallRng::seed_from_u64(s)
        } else {
            SmallRng::from_os_rng()
        };

        assert_eq!(delta.shape()[0], q as usize, "delta shape mismatch");
        assert_eq!(delta.shape()[1], q as usize, "delta shape mismatch");
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

        let mut null_transitions_vec = Vec::with_capacity(q);
        for i in 0..q {
            let mut null_inner_vec = Vec::with_capacity(q);
            for j in 0..q {
                let is_null = *null_transitions.get([i, j]).unwrap();
                null_inner_vec.push(is_null);
            }
            null_transitions_vec.push(null_inner_vec);
        }
        let null_transitions = null_transitions_vec;

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

        assert_eq!(
            random_outputs.len(),
            transition_probabilities.len(),
            "random_outputs and transition_probabilities length mismatch"
        );

        let mut sim = SimulatorSequentialArray {
            config,
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
            population: vec![0; n],
        };
        sim.make_population();
        (sim, Simulator::default())
    }

    pub fn make_population(&mut self) -> () {
        let mut k = 0;
        for state in 0..self.q {
            for _ in 0..self.config[state] {
                self.population[k] = state;
                k += 1;
            }
        }
        ()
    }

    /// Run the simulation for a specified number of steps or until max time is reached
    #[pyo3(signature = (t_max, max_wallclock_time=3600.0))]
    pub fn run(&mut self, t_max: usize, max_wallclock_time: f64) -> PyResult<()> {
        let max_wallclock_milliseconds: u64 = (max_wallclock_time * 1_000.0).ceil() as u64;
        let duration = Duration::from_millis(max_wallclock_milliseconds);
        let start_time = Instant::now();
        let run_until_silent = self.t == t_max && max_wallclock_time == 0.0;
        while run_until_silent || self.t < t_max && start_time.elapsed() < duration {
            // rejection sampling to quickly get distinct pair
            let i: usize = sample_discrete_uniform(&mut self.rng, 0, self.n - 1);
            let mut j = sample_discrete_uniform(&mut self.rng, 0, self.n - 1);
            while i == j {
                j = sample_discrete_uniform(&mut self.rng, 0, self.n - 1);
            }
            let in1 = self.population[i];
            let in2 = self.population[j];
            let out1: usize;
            let out2: usize;
            if !self.null_transitions[in1][in2] {
                let num_outputs = self.random_transitions[in1][in2].0;
                if self.is_random && num_outputs != 0 {
                    let k = self.random_transitions[in1][in2].1;
                    // sample from a probability distribution whose support is [k, k+1, ..., k+num_outputs-1],
                    // where Pr[k+i] = transition_probabilities[k+i]
                    let mut u = self.rng.sample::<f64, _>(StandardUniform)
                        - self.transition_probabilities[k];
                    let mut k = k;
                    while u > 0.0 {
                        k += 1;
                        u -= self.transition_probabilities[k];
                    }
                    (out1, out2) = self.random_outputs[k];
                } else {
                    (out1, out2) = self.delta[in1][in2];
                }
                self.population[i] = out1;
                self.population[j] = out2;
                self.config[in1] -= 1;
                self.config[in2] -= 1;
                self.config[out1] += 1;
                self.config[out2] += 1;
            }
            self.t += 1;
        }
        Ok(())
    }

    /// Run the simulation until it is silent, i.e., no reactions are applicable.
    #[pyo3()]
    pub fn run_until_silent(&mut self) -> PyResult<()> {
        return self.run(0, 0.0);
    }

    #[getter]
    pub fn silent(&self) -> bool {
        let mut states_present: Vec<State> = vec![];
        for i in 0..self.q {
            if self.config[i] > 0 {
                states_present.push(i);
            }
        }

        // println!("self.delta: {:?}", self.delta);
        // println!("self.population: {:?}", self.population);
        // println!("self.config: {:?}", self.config);
        // println!("null_transitions: {:?}", self.null_transitions);
        // println!("states_present: {:?}", states_present);

        // Check if all transitions between states_present are null
        for &i in &states_present {
            for &j in &states_present {
                if !self.null_transitions[i][j] {
                    return false; // Found a non-null transition, not silent
                }
            }
        }

        true // All transitions are null, system is silent
    }

    /// Reset the simulation with a new configuration
    // py: Python<'_>,
    #[pyo3(signature = (config, t=0))]
    pub fn reset(&mut self, config: PyReadonlyArray1<State>, t: usize) -> PyResult<()> {
        self.config = config.to_vec().unwrap();
        self.t = t;
        self.n = self.config.iter().sum();
        self.make_population();

        Ok(())
    }

    #[pyo3(signature = (filename=None))]
    pub fn write_profile(&self, filename: Option<String>) -> PyResult<()> {
        panic!("write_profile({filename:?}) not implemented for SimulatorSequentialArray");
    }
}
