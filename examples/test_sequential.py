"""
Basic test script for the ppsim package.
"""

import numpy as np
from ppsim import SimulatorSequentialArray

def test_simulator_sequential_array():
    """Test the SimulatorSequentialArray class."""
    # Create a simulator instance
    init_config = np.array([12,1,1,1], dtype=np.uint)
    delta = np.array([
        [[0,0], [1,1], [2,2], [3,3]],
        [[1,1], [1,1], [2,2], [3,3]],
        [[2,2], [2,2], [2,2], [3,3]],
        [[3,3], [3,3], [3,3], [3,3]],
    ], dtype=np.uint)
    null_transitions = np.zeros((4,4), dtype=np.bool)
    random_transitions = np.zeros((4,4,2), dtype=np.uint)
    random_outputs = np.zeros((1, 2), dtype=np.uint)
    transition_probabilities = np.ones(1, dtype=np.float64)
    sim = SimulatorSequentialArray(
        init_config,
        delta,
        null_transitions,
        random_transitions,
        random_outputs,
        transition_probabilities,
        gillespie=False,
        seed=0,
    )

    # print(f'{sim=}')
    # print(f'{sim.config=}')
    # print(f'{sim.delta=}')
    # print(f'{sim.population=}')
    
    # Create a test configuration array
    config = np.array([12, 1, 1, 1], dtype=np.uint)
    
    # Reset the simulator with the configuration
    # sim.reset(config, t=0)
    
    # Run the simulator
    sim.run(t_max=10**1)
    print(f"config: {sim.config} pop: {sim.population} init_config: {config}")
    sim.reset(config, t=0)
    print(f"config: {sim.config} pop: {sim.population}")
    sim.run(t_max=10**1, max_wallclock_time=1)
    print(f"config: {sim.config} pop: {sim.population}")

if __name__ == "__main__":
    test_simulator_sequential_array()
