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

def main():
    a, b, u = pp.species('A B U')
    approx_majority = [
        a + b >> 2 * u,
        a + u >> 2 * a,
        b + u >> 2 * b,
    ]

    pop_exponent = 4
    n = 10 ** pop_exponent
    p = 0.51
    a_init = int(n * p)
    b_init = n - a_init
    inits = {a: a_init, b: b_init}
    end_time = 5

    # for trials_exponent in range(3, 7):
    # for trials_exponent in range(4, 8):
    for trials_exponent in range(8, 9):
        print(f'*************\nCollecting sequential data for pop size 10^{pop_exponent} with 10^{trials_exponent} trials\n')
        trials = 10 ** trials_exponent
        sim = pp.Simulation(inits, approx_majority, simulator_method='sequential')
        results_sequential = sim.sample_future_configuration(end_time, num_samples=trials)
        df = pl.DataFrame(results_sequential)
        fn = f'examples/sequential_samples_n10e{pop_exponent}_trials10e{trials_exponent}.parquet'
        df.write_parquet(fn, compression="zstd")

if __name__ == "__main__":
    main()