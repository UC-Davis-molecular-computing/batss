import math
import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt, ticker
import time
import rebop as rb
from scipy import stats
import json
import gillespy2 as gl

import importlib.util
from pathlib import Path
import sys
if False:
    # Path to your renamed .pyd file
    custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64_rebop.pyd")
    # custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64_f128.pyd")

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

def measure_time(fn, trials=1) -> float:
    """
    Measure the time taken by a function over a number of trials.
    """
    start_time = time.perf_counter()
    for _ in range(trials):
        fn()
    end_time = time.perf_counter()
    return (end_time - start_time) / trials

def write_results(fn: str, times: list[float], ns: list[int]):
    results = list(zip(ns, times))
    with open(fn, 'w') as f:
        json.dump(results, f, indent=4)

def create_rebop_running_time_data(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    rebop_times = []
    ns_rebop = []
    seed = 1

    print('creating rebop data')
    # for pop_exponent_increment in tqdm(range(num_ns)):
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f'n = 10^{pop_exponent}')
        
        crn = rb.Gillespie()
        crn.add_reaction(0.1 ** pop_exponent, ['R', 'F'], ['F', 'F'])
        crn.add_reaction(1, ['R'], ['R', 'R'])
        crn.add_reaction(1, ['F'], [])

        predator_fraction = 0.5
        n = int(10 ** pop_exponent)
        
        r_init = int(n * (1 - predator_fraction))
        f_init = n - r_init
        rebop_inits = {'R': r_init, 'F': f_init}

        def run_rebop():
            crn.run(rebop_inits, end_time, 1, rng=seed)
        
        if pop_exponent == min_pop_exponent:
            # for some reason the first time it runs, rebop takes a long time
            run_rebop()
            run_rebop()
        print('rebop')
        rebop_times.append(measure_time(run_rebop, num_trials))
        ns_rebop.append(n)
        write_results(fn, rebop_times, ns_rebop)

def create_gillespy_running_time_data(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    gl_times = []
    gl_ns = []
    seed = 1

    print('creating Gillespy2 data')
    # for pop_exponent_increment in tqdm(range(num_ns)):
    r,f = pp.species('R F')
    ppsim_rxns = [
        (r+f >> 2*f).k(1),
        (r >> 2*r).k(1),
        (f >> None).k(1),
    ]
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f'n = 10^{pop_exponent}')
        
        n = int(10 ** pop_exponent)
        predator_fraction = 0.5
        r_init = int(n * (1 - predator_fraction))
        f_init = n - r_init
        ppsim_inits = {r: r_init, f: f_init}
        gl_crn = pp.gillespy2_format(ppsim_inits, ppsim_rxns, n)
        gl_crn.timespan((0, end_time))
        # print(f'gl_crn = {gl_crn}')
        
        # results = gl_crn.run(t=end_time, increment=0.1, algorithm='SSA')
        # print(f'results = {results}')

        def run_gillespy():
            gl_crn.run(algorithm='SSA')
        
        measured_time = measure_time(run_gillespy, num_trials)
        gl_times.append(measured_time)
        gl_ns.append(n)
        print(f'Gillespy2 took {measured_time} seconds to run with n = 10^{pop_exponent}')
        write_results(fn, gl_times, gl_ns)


def create_ppsim_running_time_data(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    ppsim_times = []
    ns_ppsim = []
    seed = 1
    r,f = pp.species('R F')
    rxns = [
        (r+f >> 2*f).k(1),
        (r >> 2*r).k(1),
        (f >> None).k(1),
    ]

    print('creating ppsim data')
    # for pop_exponent_increment in tqdm(range(num_ns)):
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f'n = 10^{pop_exponent}')
        
        predator_fraction = 0.5
        n = int(10 ** pop_exponent)
        r_init = int(n * (1 - predator_fraction))
        f_init = n - r_init
        ppsim_inits = {r: r_init, f: f_init}
        sim = pp.Simulation(ppsim_inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
        
        def run_ppsim():
            sim.run(end_time, 0.1)
        
        if pop_exponent == min_pop_exponent:
            run_ppsim()
        ppsim_times.append(measure_time(run_ppsim, num_trials))
        ns_ppsim.append(n)
        write_results(fn, ppsim_times, ns_ppsim)

def read_results(fn: str) -> tuple[list[int], list[float]]:
    with open(fn, 'r') as f:
        data = json.load(f)
    ns = [item[0] for item in data]
    times = [item[1] for item in data]
    return ns, times

def plot_results(fn_rebop_data: str, fn_rebop_rust_data: str, fn_ppsim_data_f64: str, fn_ppsim_data_f128: str, fn_out: str):
    # figsize = (6,4)
    figsize = (5,4)
    _, ax = plt.subplots(figsize = figsize)
    import matplotlib
    # matplotlib.rcParams.update({'font.size': 14}) # default font is too small for paper figures
    # matplotlib.rcParams['mathtext.fontset'] = 'cm' # use Computer Modern font for LaTeX
    rebop_ns, rebop_times = read_results(fn_rebop_data)
    rebop_rust_ns, rebop_rust_times = read_results(fn_rebop_rust_data)
    ppsim_ns_f64, ppsim_times_f64 = read_results(fn_ppsim_data_f64)
    ppsim_ns_f128, ppsim_times_f128 = read_results(fn_ppsim_data_f128)
    ax.loglog(ppsim_ns_f64, ppsim_times_f64, label="batching (f64)", marker="o")
    ax.loglog(ppsim_ns_f128, ppsim_times_f128, label="batching (f128)", marker=".")
    ax.loglog(rebop_ns, rebop_times, label="rebop (Python)", marker="^")
    ax.loglog(rebop_rust_ns, rebop_rust_times, label="rebop (Rust)", marker="s")
    ax.set_xlabel(f'initial molecular count')
    ax.set_ylabel(f'run time (s)')
    ax.set_xticks([10**i for i in range(3, 15)])
    ax.legend(loc='upper left')
    ax.set_ylim(bottom=None, top=10**5)
    
    ax.yaxis.set_major_locator(ticker.LogLocator(numticks=999))
    ax.yaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))
    
    plt.savefig(fn_out, bbox_inches='tight')
    plt.show()
    # print(stats.linregress([math.log(x) for x in ns_ppsim], [math.log(x) for x in ppsim_times]))
    # print(stats.linregress([math.log(x) for x in ns_ppsim], [math.log(x) for x in rebop_times]))
    # print(ns_ppsim)
    # print(ppsim_times)
    # print(rebop_times)
    return

# def test_time_scaling_vs_end_time():
#     ppsim_times = []
#     rebop_times = []
#     times = []
#     num_times = 20
#     pop_exponent = 10
#     n = 10 ** pop_exponent
#     min_time_exponent = -5
#     max_time_exponent = -2
#     num_checkpoints = 1
#     num_trials = 1
#     for time_exponent_increment in tqdm(range(num_times)):
#         time_exponent = min_time_exponent + (max_time_exponent - min_time_exponent) * (time_exponent_increment / (float(num_times - 1.0)))
#         a,b = pp.species('A B')
        
#         predator_fraction = 0.5 

#         rxns = [
#             (a+b >> 2*b).k(0.1 ** pop_exponent),
#             (a >> 2*a).k(1),
#             (b >> None).k(1),
#         ]

        
#         def timefn(t):
#             a_init = int(n * (1 - predator_fraction))
#             b_init = n - a_init
#             inits = {a: a_init, b: b_init}
#             seed=random.randint(1, 10000)
#             sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=4)
#             sim.sample_future_configuration(t, num_samples=1)
#             # sim.run(t, t / float(num_checkpoints))
#         # sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True)
        
        

#         def timefnrebop(t):
#             crn = rb.Gillespie()
#             crn.add_reaction(0.1 ** pop_exponent, ['A', 'B'], ['B', 'B'])
#             crn.add_reaction(1, ['A'], ['A', 'A'])
#             crn.add_reaction(1, ['B'], [])
#             a_init = int(n * (1 - predator_fraction))
#             b_init = n - a_init
#             inits = {"A": a_init, "B": b_init}
#             results_rebop = crn.run(inits, t, 1)
#             get_rebop_samples(pop_exponent, 1, b_init, "B", t)
        
#         end_time = 10 ** time_exponent
#         # end_time = 0.1 + 0.1 * time_exponent_increment
#         ppsim_times.append(timeit.timeit(lambda: timefn(end_time), number=num_trials))
#         rebop_times.append(timeit.timeit(lambda: timefnrebop(end_time), number=num_trials))
#         times.append(end_time)
        
#     fig, ax = plt.subplots(figsize = (10,4))
#     ax.plot(times, ppsim_times, label="ppsim run time")
#     ax.plot(times, rebop_times, label="rebop run time")
#     # sim.simulator.write_profile() # type: ignore
#     ax.set_xlabel(f'Simulated continuous time units)')
#     ax.set_ylabel(f'Run time (s)')
#     # ax.set_xscale("log")
#     # ax.set_yscale("log")
#     ax.legend()
#     print(stats.linregress(times, rebop_times))

#     plt.show()
#     return

def fn_count_samples(alg: str, pop_exponent: int, trials_exponent: int, species: str, final_time: float) -> str:
    return f'data/lk_{alg}_{species}-counts_time{final_time}_n1e{pop_exponent}_trials1e{trials_exponent}.json'

def write_rebop_count_samples(pop_exponent: int, trials_exponent: int, species: str, final_time: float) -> None:
    fn = fn_count_samples('rebop', pop_exponent, trials_exponent, species, final_time)
    print(f'collecting rebop data with n = 10^{pop_exponent} for {trials_exponent} trials')
    print(f'writing to {fn}')
    n = 10 ** pop_exponent
    crn = rb.Gillespie()
    crn.add_reaction(0.1 ** pop_exponent, ['R', 'F'], ['F', 'F'])
    crn.add_reaction(1, ['R'], ['R', 'R'])
    crn.add_reaction(1, ['F'], [])
    from collections import defaultdict
    counts = defaultdict(int)
    for _ in tqdm(range(10**trials_exponent)):
        r_init = n // 2
        f_init = n // 2
        inits = {'R': r_init, 'F': f_init}
        # It should be very roughly 1 step every 1/n real time, so to get a particular number
        # of steps, it should be safe to run for, say, 3 times that much time
        while True:
            try:
                results_rebop = crn.run(inits, final_time, 1)
                # print(f"There are {len(results_rebop[state])} total steps in rebop simulation.")
                # print(results_rebop[state])
                count = int(results_rebop[species][-1])
                counts[count] += 1
                break
            except IndexError:
                pass
                print("Index error caught and ignored. Rebop distribution may be slightly off.")
        counts = sort_dict_by_key(counts)
    with open(fn, 'w') as f:
        json.dump(counts, f, indent=4)

def sort_dict_by_key(d: dict) -> dict:
    """
    Sort a dictionary by its keys.
    """
    return dict(sorted(d.items(), key=lambda item: item[0]))

def write_ppsim_count_samples(pop_exponent: int, trials_exponent: int, species: str, final_time: float) -> None:
    print(f'collecting ppsim data with n = 10^{pop_exponent} for {trials_exponent} trials')
    fn = fn_count_samples('ppsim', pop_exponent, trials_exponent, species, final_time)
    n = 10 ** pop_exponent
    r,f = pp.species('R F')
    rxns = [
        (r+f >> 2*f).k(1),
        (r >> 2*r).k(1),
        (f >> None).k(1),
    ]
    a_init = n // 2
    b_init = n - a_init
    inits = {r: a_init, f: b_init}
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=4) #type: ignore
    from collections import defaultdict
    counts = defaultdict(int)
    trials = 10**trials_exponent
    results_batching = sim.sample_future_configuration(final_time, num_samples=trials)
    count_list: list[int] = results_batching[species].squeeze().tolist() # type: ignore
    counts = defaultdict(int)
    for count in count_list:
        counts[count] += 1
    
    counts = sort_dict_by_key(counts)
    with open(fn, 'w') as f:
        json.dump(counts, f, indent=4)

def read_count_samples(fn: str) -> list[int]:
    """
    Read the count samples from a JSON file.
    """
    with open(fn, 'r') as f:
        counts = json.load(f)
    count_list = []
    for count, num_samples_with_count in counts.items():
        count_list.extend([int(count)] * num_samples_with_count)
    return count_list

def plot_rebop_ppsim_histogram(pop_exponent: int, trials_exponent: int, species: str, final_time: float):
    rebop_fn = fn_count_samples('rebop', pop_exponent, trials_exponent, species, final_time)
    ppsim_fn = fn_count_samples('ppsim', pop_exponent, trials_exponent, species, final_time)
    
    rebop_counts = read_count_samples(rebop_fn)
    ppsim_counts = read_count_samples(ppsim_fn)
    
    fig, ax = plt.subplots(figsize = (10,4))
    # print((results_batching).shape)
    # print((results_batching[state].squeeze().tolist()))
    # print(results_rebop) 
    # print([results_batching[state].squeeze().tolist(), results_rebop])
    # ax.hist(results_rebop)
    ax.hist([ppsim_counts, rebop_counts], # type: ignore
            bins = 20,
            alpha = 1, label=['batching', 'rebop']) #, density=True, edgecolor = 'k', linewidth = 0.5)
    ax.legend()

    ax.set_xlabel(f'Count of species {species}')
    ax.set_ylabel(f'Number of samples')
    ax.set_title(f'Species {species} distribution sampled at simulated time {final_time}'
                 f'(n=$10^{pop_exponent}$; trials=$10^{trials_exponent}$)')
    
    # plt.ylim(0, 200_000)
    pdf_fn = fn_count_samples('ppsim-vs-rebop', pop_exponent, trials_exponent, species, final_time)
    pdf_fn = pdf_fn.replace('.json', '.pdf')
    plt.savefig(pdf_fn, bbox_inches='tight')
    plt.show()


def main():
    # create_rebop_running_time_data("data/lotka_volterra_time1_times_rebop.json", 3, 12, 1.0)
    # create_ppsim_running_time_data("data/lotka_volterra_time1_times_ppsim_f128.json", 3, 14, 1.0)
    plot_results('data/lotka_volterra_time1_times_rebop.json', 
                 'data/lotka_volterra_time1_times_rebop_rust.json',
                 'data/lotka_volterra_time1_times_ppsim_f64.json',
                 'data/lotka_volterra_time1_times_ppsim_f128.json',
                 'data/lotka_volterra_scaling_time1.pdf')
    # test_distribution()

    pop_exponent = 4
    trials_exponent = 3
    final_time = 1.0
    species = 'F'
    # write_rebop_count_samples(pop_exponent, trials_exponent, species, final_time)
    # write_ppsim_count_samples(pop_exponent, trials_exponent, species, final_time)
    # plot_rebop_ppsim_histogram(pop_exponent, trials_exponent, species, final_time)


if __name__ == "__main__":
    main()