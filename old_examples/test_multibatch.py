"""
Basic test script for the ppsim package.
"""

import numpy as np
from ppsim import SimulatorMultiBatch

    

def test_simulator_multi_batch():
    """Test the SimulatorMultiBatch class."""
    init_config = np.array([12,1,1], dtype=np.uint)
    delta = np.array([
        [[0,0], [1,1], [2,2]],
        [[1,1], [1,1], [2,2]],
        [[2,2], [2,2], [2,2]],
    ], dtype=np.uint)
    null_transitions = np.zeros((3,3), dtype=bool)
    random_transitions = np.zeros((3,3,2), dtype=np.uint)
    random_outputs = np.zeros((1, 2), dtype=np.uint)
    transition_probabilities = np.ones(1, dtype=np.float64)
    sim = SimulatorMultiBatch(
        init_config=init_config,
        delta=delta,
        null_transitions=null_transitions,
        random_transitions=random_transitions,
        random_outputs=random_outputs,
        transition_probabilities=transition_probabilities,
        seed=0,
    )
    
    sim.run(t_max=100, max_wallclock_time=10)
    print(sim.config)

if __name__ == "__main__":
    test_simulator_multi_batch()
