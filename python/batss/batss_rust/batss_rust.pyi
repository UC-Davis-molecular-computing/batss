"""
batss: A Python package with Rust backend for simulation of chemical reaction networks through a fast batchign algorithm.
"""

# Re-export everything from Python modules
from batss.simulation import *
from batss.snapshot import *
from batss.crn import *

import numpy as np
import numpy.typing as npt

from abc import ABC

RustState: TypeAlias = int
"""
Type alias for how states are represented in internally in the Rust simulators, 
as integers 0,1,...,num_states-1.
"""

class Simulator(ABC):
    config: list[RustState]
    """TODO"""

    n: int
    """TODO"""

    t: int
    """TODO"""

    delta: list[list[tuple[RustState, RustState]]]
    """TODO"""

    silent: bool
    """TODO"""

    k0_manual_multiplier: float
    """
    Experiment override (BatchSimulator only). If > 0, K is reset toward ``round(k0_manual_multiplier
    * n)`` instead of the throughput-optimal ``min(2n, crossover)``; used to sweep K and locate the
    batch-count optimum. 0 (default) disables the override.
    """

    k_resets: int
    """Number of K resets performed so far (observability; BatchSimulator only, read-only)."""

    heuristic_gillespie_switching: int
    """
    Which batch/Gillespie switching heuristic to use (BatchSimulator only): 0 = wall-clock
    measurement (default), 1 = the simpler reaction-count proxy, 2 = the experimental
    deterministic prospective-batch score. Must be set before :meth:`run`.
    """

    proxy_threshold: float
    """
    Threshold for the proxy or prospective score (BatchSimulator only): prefer Gillespie when the
    selected estimate of active reactions in the next batch falls below it.
    """

    def run_until_silent(self) -> None:
        """TODO"""
        ...

    def reset(self, config: npt.NDArray[np.uint], t: int = 0) -> None:
        """TODO"""
        ...

    def run(self, t_max: int | float, max_wallclock_time: float = 3600.0) -> None:
        """
        Run the simulation for a specified number of steps or until max time is reached.

        Args:
            t_max: Maximum number of simulation steps to execute or continuous time to simulate
            max_wallclock_time: Maximum wall clock time in seconds before stopping (default: 1 hour)
        """
        ...

    def active_reaction_probability(self) -> float:
        """
        The probability that the next sampled reaction is active (i.e. actually changes the
        configuration in the original CRN). It is a function of the current configuration -- the full
        state including the filler-species (K) count, not the original species counts alone. Because K
        drifts over a run (reset toward its target only when K leaves a multiplicative band around it,
        and frozen during Gillespie phases), the same original-species counts can give different values
        as K drifts. Well-defined in both batch and Gillespie phases.
        """
        ...

    def __init__(
        self,
        init_config: npt.NDArray[np.uint],
        delta: npt.NDArray[np.uint] | None,
        random_transitions: npt.NDArray[np.uint] | None,
        random_outputs: npt.NDArray[np.uint] | None,
        transition_probabilities: npt.NDArray[np.float64] | None,
        transition_order: str | None,
        gillespie: bool | None = False,
        seed: int | None = 3,
        crn=None,
        k=None,
        w=None,
    ) -> None:
        """TODO"""
        ...

class EngineCallBenchmark:
    """Separated timings and work counts from one frozen-state engine call."""

    gillespie: bool
    preparation_seconds: float
    setup_seconds: float
    engine_seconds: float
    postprocess_seconds: float
    continuous_time_advanced: float
    total_reactions: int
    active_reactions: int
    k_rebuilt: bool

class BatchSimulator(Simulator):
    """
    Simulator for CRNs using TODO cite paper once citeable.
    """

    continuous_time: float
    """TODO"""

    do_gillespie: bool
    """TODO"""

    reactions: list[list[RustState]]
    """TODO"""

    enabled_reactions: list[int]
    """TODO"""

    num_enabled_reactions: int
    """TODO"""

    reaction_probabilities: list[float]
    """TODO"""

    def get_enabled_reactions(self) -> None:
        """TODO"""
        ...

    def get_total_propensity(self) -> float:
        """TODO"""
        ...

    def prospective_batch_score(self) -> float:
        """
        Experimental deterministic estimate of active reactions in a batch prepared at the
        prospective K reset target, without rebuilding transition arrays.
        """
        ...
    def benchmark_engine_call(
        self,
        gillespie: bool,
        gillespie_reactions: int | None = None,
    ) -> EngineCallBenchmark:
        """
        Benchmark one batch or Gillespie block from a canonicalized frozen state.

        This mutates the simulator; use a fresh or reset simulator for each paired trial.
        """
        ...

    def clear_profile(self) -> None:
        """Clear Flame spans accumulated by the current thread (no-op without the flm feature)."""
        ...

    def debug_prospective_n(self) -> int: ...
    def debug_q(self) -> int: ...
    def debug_reactant_sets(self) -> int: ...
    def debug_output_branches(self) -> int: ...
