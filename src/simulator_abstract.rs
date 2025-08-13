use pyo3::prelude::*;

use numpy::PyReadonlyArray1;

type State = usize;

// An abstract class defining the API for what a simulator needs to do.
#[pyclass(subclass)]
#[derive(Default)]
pub struct Simulator {
    /// The population size (sum of values in urn.config).
    #[pyo3(get, set)]
    pub n: u64,
    /// The current number of elapsed interaction steps.
    #[pyo3(get, set)]
    pub t: u64,
    /// A q x q array of pairs (c,d) representing the transition function.
    /// delta[a][b] gives contains the two output states for a
    /// deterministic transition a,b --> c,d.
    #[pyo3(get, set)]
    pub delta: Vec<Vec<(State, State)>>,
    /// A boolean determining if the configuration is silent (all interactions are null).
    #[pyo3(get, set)]
    pub silent: bool,
}

#[pymethods]
impl Simulator {
    /// Initializes the simulator.
    #[new]
    pub fn new() -> Self {
        unimplemented!()
    }

    /// Gets the configuration of the simulator.
    #[getter]
    pub fn config(&self) -> Vec<State> {
        unimplemented!()
    }

    /// Run the simulation for a specified number of steps or until max time is reached.
    #[allow(unused_variables)]
    #[pyo3(signature = (t_max, max_wallclock_time=3600.0))]
    pub fn run(&mut self, t_max: u64, max_wallclock_time: f64) -> PyResult<()> {
        unimplemented!()
    }

    /// Run the simulation until it is silent, i.e., no reactions are applicable.
    #[pyo3()]
    pub fn run_until_silent(&mut self) {
        unimplemented!()
    }

    /// Reset the simulation with a new configuration
    /// Sets all parameters necessary to change the configuration.
    /// Args:
    ///     config: The configuration array to reset to.
    ///     t: The new value of :any:`t`. Defaults to 0.
    #[pyo3(signature = (config, t=0))]
    #[allow(unused_variables)]
    pub fn reset(&mut self, config: PyReadonlyArray1<State>, t: u64) -> PyResult<()> {
        unimplemented!()
    }
}
