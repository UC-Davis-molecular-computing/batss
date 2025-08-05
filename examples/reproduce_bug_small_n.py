import sys
import importlib.util
import os
from pathlib import Path

if False:
    # Path to your renamed .pyd file
    custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64_2.pyd")

    # Define a custom finder and loader for .pyd files
    class CustomPydFinder:
        @classmethod
        def find_spec(cls, fullname, path=None, target=None):
            # Only handle the specific module we want to redirect
            if fullname == "batss.batss_rust.batss_rust":
                return importlib.util.spec_from_file_location(fullname, str(custom_pyd_path))
            return None

    # Register our custom finder at the beginning of the meta_path
    sys.meta_path.insert(0, CustomPydFinder)



import batss as pp
import numpy as np
import gpac as gp
import polars as pl
from matplotlib import pyplot as plt
import rebop as rb
import timeit

def main():
    import matplotlib.pyplot as plt

    make_and_save_plot(3)
    
def make_and_save_plot(pop_exponent: int) -> None:
    seed = 4
    n = int(10 ** pop_exponent)
    p = 0.5
    r_init = int(n * p)
    f_init = n - r_init
    inits = {'R': r_init, 'F': f_init}
    end_time = 20.0
    num_samples = 10**3
    print(f'done with rebop')

    r,f = pp.species('R F')
    rxns = [
        (r+f >> 2*f).k(1),
        (r >> 2*r).k(1),
        (f >> None).k(1),
    ]
    
    inits = {r: r_init, f: f_init}
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)

    print(f'running ppsim with n = 10^{pop_exponent}')
    sim.run(end_time, end_time / num_samples)

if __name__ == "__main__":
    main()