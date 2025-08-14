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

    discrete_steps_no_nulls: int
    """TODO"""

    discrete_steps_no_nulls_last_run: int
    """TODO"""

    discrete_steps_total: int
    """TODO"""

    discrete_steps_total_last_run: int
    """TODO"""

    def run_until_silent(self) -> None:
        """TODO"""
        ...

    def reset(
            self,
            config: npt.NDArray[np.uint],
            t: int = 0
    ) -> None: 
        """TODO"""
        ...

        
    def run(
            self,
            t_max: int | float,
            max_wallclock_time: float = 3600.0
    ) -> None:
        """
        Run the simulation for a specified number of steps or until max time is reached.
        
        Args:
            t_max: Maximum number of simulation steps to execute or continuous time to simulate
            max_wallclock_time: Maximum wall clock time in seconds before stopping (default: 1 hour)
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
            crn = None,
            k = None,
            w = None,
    ) -> None:
        """TODO"""
        ...


class SimulatorCRNMultiBatch(Simulator):
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