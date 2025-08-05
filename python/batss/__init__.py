"""
batss: A Python package with Rust backend for simulation of chemical reaction networks through a fast batchign algorithm.

This package provides tools for simulating population protocols and chemical reaction networks.
The core simulation engine is implemented in Rust for performance, with a Python interface for
ease of use and visualization.

Modules:
    - :mod:`batss.simulation`: Module for simulating chemical reaction networks
    - :mod:`batss.crn`: Module for expressing population protocols using chemical reaction network notation
    - :mod:`batss.snapshot`: Module for visualizing species counts during or after simulation

Example:
    .. code-block:: python
    
        from batss import species, Simulation
        
        # Define a simple protocol
        a, b, u = species('A B U')
        approx_majority = [
            a + b >> 2 * u,
            a + u >> 2 * a,
            b + u >> 2 * b,
        ]
        
        # Initialize and run simulation
        n = 10 ** 5
        a_init = int(0.51 * n)
        b_init = n - a
        init_config = {a: a_init, b: b_init}
        sim = Simulation(init_config=init_config, rule=approx_majority)
        sim.run()
        sim.history.plot()
"""

from importlib.metadata import version
__version__ = version("batss")

# Import order matters to avoid circular imports
from batss.crn import *
from batss.simulation import *
from batss.snapshot import *
