import sys
from pathlib import Path

if False:
    import importlib.util
    custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64_bt-n-over-2.pyd")

    class CustomPydFinder:
        @classmethod
        def find_spec(cls, fullname, path=None, target=None):
            # Only handle the specific module we want to redirect
            if fullname == "ppsim.ppsim_rust.ppsim_rust":
                return importlib.util.spec_from_file_location(fullname, str(custom_pyd_path))
            return None

    sys.meta_path.insert(0, CustomPydFinder)



import ppsim as pp
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

def generate_ppsim_samples(pop_exponent: int, trials_exponent: int, end_time: int, dir: str) -> pl.DataFrame:
    a,b,u = pp.species('A B U')
    approx_majority = [
        a+b >> 2*u,
        a+u >> 2*a,
        b+u >> 2*b,
    ]

    n = 10 ** pop_exponent
    p = 0.51
    a_init = int(n * p)
    b_init = n - a_init
    inits = {a: a_init, b: b_init}
    trials = 10 ** trials_exponent
    sim = pp.Simulation(inits, approx_majority)

    results_ppsim = sim.sample_future_configuration(end_time, num_samples = trials)
    
    df = pl.DataFrame(results_ppsim)
    # fn = f'{dir}/ppsim_samples_n10e{pop_exponent}_trials10e{trials_exponent}_bt-n-over-2.parquet'
    # df.write_parquet(fn, compression="zstd")

    return df

def main():
    import json
    compare_to, pop_exponent, trials_exponent = cl_args()
    
    dir = 'examples'
    end_time = 5
    
    print(f'Comparing multibatch to {compare_to} samples with population size '
          f'10^{pop_exponent} and 10^{trials_exponent} trials at time {end_time}.')
    
    state = 'A'
    # state = 'B'
    # state = 'U'

    
    fn_noext = f'{compare_to}_samples_n10e{pop_exponent}_trials10e{trials_exponent}'
    fn = f'{fn_noext}.json'
    results_other = json.load(open(f'{dir}/{fn}'))
    counts_other = []
    for amount, count in results_other[state].items():
        counts_other.extend([amount] * count)
    counts_other = np.array(counts_other, dtype=np.uint64)
    
    results_ppsim = generate_ppsim_samples(pop_exponent, trials_exponent, end_time, dir)
    
    fig, ax = plt.subplots(figsize = (10,4))
    
    min_other = counts_other.min()
    max_other = counts_other.max()
    min_ppsim = results_ppsim[state].min()
    max_ppsim = results_ppsim[state].max()
    minimum = min(min_other, min_ppsim) # type: ignore
    maximum = max(max_other, max_ppsim) # type: ignore
    n = 10 ** pop_exponent
    if minimum < 0.1 * n:
        minimum = 0
    if maximum > 0.9 * n:
        maximum = n
    if minimum < 0.2 * n and maximum > 0.8 * n:
        minimum = 0
        maximum = n

    print(f'minimum: {minimum}, maximum: {maximum}')
    bins = np.linspace(minimum, maximum, 20) # type: ignore

    ax.hist([results_ppsim[state], counts_other], 
            # bins = np.linspace(int(n*0.32), int(n*.43), 20), # type: ignore
            bins = bins, # type: ignore
            alpha = 1, label=['multibatch', compare_to]) #, density=True, edgecolor = 'k', linewidth = 0.5)
    ax.legend()

    ax.set_xlabel(f'count of state {state}')
    ax.set_ylabel(f'empirical probability')
    ax.set_title(f'multibatch vs {compare_to} state {state} distribution sampled at time {end_time} ($10^{trials_exponent}$ samples, n=$10^{pop_exponent}$)')
    
    # plt.ylim(0, 200_000)

    pdf_fn = f'{dir}/multibatch_vs_{compare_to}_n10e{pop_exponent}_trials10e{trials_exponent}_bt-n-over-2_sample_coll-no-precompute.pdf'
    # plt.savefig(pdf_fn, bbox_inches='tight')
    plt.show()

def cl_args() -> tuple[str, int, int]:
    if len(sys.argv) == 4:
        compare_to = sys.argv[1]
        pop_exponent = int(sys.argv[2])
        trials_exponent = int(sys.argv[3])
    else:
        raise ValueError("Please provide exactly three arguments: <rebop/sequential> <pop_exponent> <trials_exponent>")
    
    if compare_to not in ['rebop', 'sequential', 'r', 's']:
        raise ValueError("Please provide 'rebop'/'r' or 'sequential'/'s' as an argument.")

    if compare_to == 'r':
        compare_to = 'rebop'
    elif compare_to == 's':
        compare_to = 'sequential'
    else:
        assert False
    
    return compare_to,pop_exponent,trials_exponent

def compare_stats():
    compare_name, pop_exponent, trials_exponent = cl_args()
    dir = 'examples'
    end_time = 5

    ppsim_name = 'ppsim'
    other_fn = f'{compare_name}_samples_n10e{pop_exponent}_trials10e{trials_exponent}.parquet'
    ppsim_fn = f'{ppsim_name}_samples_n10e{pop_exponent}_trials10e{trials_exponent}_bt-n-over-2.parquet'
    results_other = pl.read_parquet(f'{dir}/{other_fn}')
    results_ppsim = pl.read_parquet(f'{dir}/{ppsim_fn}')
    a_other = results_other['A'].to_numpy()
    a_ppsim = results_ppsim['A'].to_numpy()
    
    print(f'Statistics for count of state "A", comparing {compare_name} and ppsim samples, '
          f'n=10^{pop_exponent}, trials=10^{trials_exponent}, at time {end_time}.')

    for name, counts in [(compare_name, a_other), (ppsim_name, a_ppsim)]:
        mean = float(np.mean(counts))
        stddev = float(np.std(counts))
        print(f'{name:10}  mean: {mean:.3f}  stddev: {stddev:.3f}')

def get_smallest_uint_type(max_val):
    if max_val <= 255:
        return pl.UInt8
    elif max_val <= 65535:
        return pl.UInt16
    elif max_val <= 4294967295:
        return pl.UInt32
    else:
        return pl.UInt64

            

if __name__ == "__main__":
    main()
    # compare_stats()