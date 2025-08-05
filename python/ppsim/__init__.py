"""
ppsim: A Python package with Rust backend for simulation.

This package provides tools for simulating population protocols and chemical reaction networks.
The core simulation engine is implemented in Rust for performance, with a Python interface for
ease of use and visualization.

Modules:
    - :mod:`ppsim.simulation`: Module for simulating population protocols
    - :mod:`ppsim.crn`: Module for expressing population protocols using chemical reaction network notation
    - :mod:`ppsim.snapshot`: Module for visualizing protocols during or after simulation

Example:
    .. code-block:: python
    
        from ppsim import species, Simulation
        
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
__version__ = version("ppsim")

# Import order matters to avoid circular imports
from ppsim.crn import *
from ppsim.simulation import *
from ppsim.snapshot import *
