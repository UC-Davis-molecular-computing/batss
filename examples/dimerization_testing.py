import math
import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import rebop as rb
from scipy import stats
import json
from collections import defaultdict

import importlib.util
from pathlib import Path
import sys
if False:
    # Path to your renamed .pyd file
    # custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64_rebop.pyd")
    # custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64_f128.pyd")
    # custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64_2.pyd")
    # custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64_bugfix_jul10.pyd")
    custom_pyd_path = Path("C:/Dropbox/git/ppsim-rust/python/ppsim/ppsim_rust/ppsim_rust.cp312-win_amd64)bugfix_jul10_fastpp.pyd")

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

def write_running_time_results(fn: str, times: list[float], ns: list[int]):
    results = list(zip(ns, times))
    with open(fn, 'w') as f:
        json.dump(results, f, indent=4)

def create_rebop_running_time_data(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    rebop_times = []
    ns_rebop = []
    seed = 1

    print('creating rebop running time data for dimerization')
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f'n = 10^{pop_exponent}')
        
        crn = rb.Gillespie()
        crn.add_reaction(0.1 ** pop_exponent, ['M', 'M'], ['D'])
        crn.add_reaction(1, ['D'], ['M', 'M'])

        predator_fraction = 0.5
        n = int(10 ** pop_exponent)
        
        a_init = int(n * (1 - predator_fraction))
        b_init = n - a_init
        rebop_inits = {"A": a_init, "B": b_init}

        def run_rebop():
            crn.run(rebop_inits, end_time, 1, rng=seed)
        
        if pop_exponent == min_pop_exponent:
            # for some reason the first time it runs, rebop takes a long time
            run_rebop()
            run_rebop()
        print('rebop')
        rebop_times.append(measure_time(run_rebop, num_trials))
        ns_rebop.append(n)
        write_running_time_results(fn, rebop_times, ns_rebop)

def create_ppsim_running_time_data(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    ppsim_times = []
    ns_ppsim = []
    seed = 1
    m,d = pp.species('M D')
    rxns = [ (m+m | d).k(1) ]

    print('creating ppsim running time data for dimerization')
    # for pop_exponent_increment in tqdm(range(num_ns)):
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f'n = 10^{pop_exponent}')
        
        predator_fraction = 0.5
        n = int(10 ** pop_exponent)
        ppsim_inits = { m: n }
        sim = pp.Simulation(ppsim_inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
        
        def run_ppsim():
            sim.run(end_time, 0.1)
        
        if pop_exponent == min_pop_exponent:
            run_ppsim()
        ppsim_times.append(measure_time(run_ppsim, num_trials))
        ns_ppsim.append(n)
        write_running_time_results(fn, ppsim_times, ns_ppsim)

def read_running_time_results(fn: str) -> tuple[list[int], list[float]]:
    with open(fn, 'r') as f:
        data = json.load(f)
    ns = [item[0] for item in data]
    times = [item[1] for item in data]
    return ns, times

def plot_results(fn_rebop_data: str, fn_ppsim_data_f64: str, fn_ppsim_data_f128: str, fn_out: str):
    # figsize = (6,4)
    figsize = (5,4)
    _, ax = plt.subplots(figsize = figsize)
    import matplotlib
    # matplotlib.rcParams.update({'font.size': 14}) # default font is too small for paper figures
    # matplotlib.rcParams['mathtext.fontset'] = 'cm' # use Computer Modern font for LaTeX
    rebop_ns, rebop_times = read_running_time_results(fn_rebop_data)
    ppsim_ns_f64, ppsim_times_f64 = read_running_time_results(fn_ppsim_data_f64)
    ppsim_ns_f128, ppsim_times_f128 = read_running_time_results(fn_ppsim_data_f128)
    ax.loglog(ppsim_ns_f64, ppsim_times_f64, label="batching f64 run time", marker="o")
    ax.loglog(ppsim_ns_f128, ppsim_times_f128, label="batching f128 run time", marker="o")
    ax.loglog(rebop_ns, rebop_times, label="rebop run time", marker="o")
    ax.set_xlabel(f'Initial molecular count')
    ax.set_ylabel(f'Run time (s)')
    ax.set_xticks([10**i for i in range(3, 15)])
    ax.set_ylim(bottom=None, top=10**5)
    ax.legend(loc='upper left')
    
    plt.savefig(fn_out, bbox_inches='tight')
    return

def fn_count_samples(alg: str, pop_exponent: int, trials_exponent: int, species: str, final_time: float) -> str:
    return f'data/dimer_{species}-counts_time{final_time}_n1e{pop_exponent}_trials1e{trials_exponent}_{alg}.json'

def write_rebop_count_samples(crn: rb.Gillespie, inits: dict[str, int], 
                              pop_exponent: int, trials_exponent: int, species_name: str, final_time: float) -> None:
    fn = fn_count_samples('rebop', pop_exponent, trials_exponent, species_name, final_time)
    print(f'collecting rebop data with n = 10^{pop_exponent} for {trials_exponent} trials')
    print(f'writing to {fn}')
    counts = defaultdict(int)
    for _ in tqdm(range(10**trials_exponent)):
        while True:
            try:
                results_rebop = crn.run(inits, final_time, 1)
                count = int(results_rebop[species_name][-1])
                counts[count] += 1
                break
            except IndexError:
                pass
                print("Index error caught and ignored. Rebop distribution may be slightly off.")
    counts = sort_dict_by_key(counts)
    with open(fn, 'w') as f:
        json.dump(counts, f, indent=4)

def sort_dict_by_key(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda item: item[0]))

def write_ppsim_count_samples(sim: pp.Simulation, 
                              pop_exponent: int, trials_exponent: int, species_name: str, final_time: float) -> None:
    fn = fn_count_samples('ppsim', pop_exponent, trials_exponent, species_name, final_time)
    print(f'collecting ppsim data with n = 10^{pop_exponent} for {trials_exponent} trials')
    print(f'writing to {fn}')
    trials = 10**trials_exponent
    results_batching = sim.sample_future_configuration(final_time, num_samples=trials)
    count_list: list[int] = results_batching[species_name].squeeze().tolist() # type: ignore
    counts = defaultdict(int)
    for count in count_list:
        counts[count] += 1
    
    counts = sort_dict_by_key(counts)
    with open(fn, 'w') as f:
        json.dump(counts, f, indent=4)

def read_count_samples(fn: str) -> dict[int, int]:
    """
    Read the count samples from a JSON file.
    """
    with open(fn, 'r') as f:
        counts = json.load(f)
    count_dict = {}
    for count, num_samples_with_count in counts.items():
        count_dict[int(count)] = num_samples_with_count
    return count_dict

def plot_rebop_ppsim_histogram(pop_exponent: int, trials_exponent: int, species_name: str, final_time: float):
    rebop_fn = fn_count_samples('rebop', pop_exponent, trials_exponent, species_name, final_time)
    ppsim_fn = fn_count_samples('ppsim', pop_exponent, trials_exponent, species_name, final_time)
    
    rebop_counts = read_count_samples(rebop_fn)
    ppsim_counts = read_count_samples(ppsim_fn)

    _, ax = plt.subplots(figsize = (7,3))

    # Get the range of values and calculate total trials for normalization
    all_values = set(rebop_counts.keys()) | set(ppsim_counts.keys())
    min_val = min(all_values)
    max_val = max(all_values)
    
    # Calculate total number of trials for each method (for normalization to empirical probability)
    total_ppsim_trials = sum(ppsim_counts.values())
    total_rebop_trials = sum(rebop_counts.values())
    
    # Create arrays for the bar plot
    values = list(range(min_val, max_val + 1))
    ppsim_probs = [ppsim_counts.get(val, 0) / total_ppsim_trials for val in values]
    rebop_probs = [rebop_counts.get(val, 0) / total_rebop_trials for val in values]
    
    # Create side-by-side bars
    bar_width = 0.4
    x_positions = np.array(values)
    
    ax.bar(x_positions - bar_width/2, ppsim_probs, bar_width, label='batching', alpha=1)
    ax.bar(x_positions + bar_width/2, rebop_probs, bar_width, label='rebop', alpha=1)
    
    ax.legend()
    ax.set_xlim((5,35))
    ax.set_xticks(range(5, 36, 5))
    ax.set_xlabel(f'count of species ${species_name}$')
    ax.set_ylabel(f'empirical probability')
    ax.set_title(f'count of species ${species_name}$ sampled at time {final_time} '
                 f'($n$=$10^{pop_exponent}$; trials=$10^{trials_exponent}$)')
    
    pdf_fn = fn_count_samples('ppsim-vs-rebop', pop_exponent, trials_exponent, species_name, final_time)
    pdf_fn = pdf_fn.replace('.json', '.pdf')
    plt.savefig(pdf_fn, bbox_inches='tight')
    plt.show()

# change what's in the next two functions to change the CRN tested
def rebop_dimerization_with_inits(pop_exponent: int) -> tuple[rb.Gillespie, dict[str, int]]:
    crn = rb.Gillespie()
    crn.add_reaction(0.1 ** pop_exponent, ['M', 'M'], ['D'])
    crn.add_reaction(1, ['D'], ['M', 'M'])
    
    n = int(10 ** pop_exponent)
    m_init = n
    inits = {'M': m_init}
    
    return crn, inits

def plot_dimerization_crn_ppsim_with_null(pop_exponent: int, seed: int) -> None:
    figsize = (12, 6)
    sim = ppsim_dimerization_crn(pop_exponent, seed)
    print(f'running ppsim with n = 10^{pop_exponent}')

    sim.run(2, 0.01)

    # total steps starts with 0 (and for some reason ends with 0, I don't get that),
    # so we slice it to remove the first and last element to avoid dividing by zero.
    total_steps = np.array(sim.discrete_steps_total_last_run)[1:-1]
    non_null_steps = np.array(sim.discrete_steps_no_nulls_last_run)[1:-1]
    null_steps = total_steps - non_null_steps
    null_fractions = null_steps / total_steps
    times = sim.history.index.tolist()[1:-1] # make same length as null_fractions
    

    f, ax = plt.subplots(figsize=figsize)

    blue, orange, green, red  = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'
    # Create the primary plot with R and F (left y-axis)
    ax.plot(sim.history['M'], label='M', color=blue)
    ax.plot(sim.history['D'], label='D', color=orange)

    # Set up the left y-axis
    ax.set_ylabel('counts')

    # Create a second y-axis that shares the same x-axis
    ax2 = ax.twinx()

    # Plot null_fractions on the second y-axis
    ax2.plot(times, null_fractions, label='passive', color=red)

    # Set up the right y-axis
    ax2.set_ylabel('fraction of passive reactions')
    ax2.set_ylim(0.0, 1.0)

    # Create a single legend with handles from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='lower right')
    # plt.savefig(f'data/dimerization_plot_with_passive_reactions_n1e{pop_exponent}.pdf', bbox_inches='tight')
    plt.show()


def ppsim_dimerization_crn(pop_exponent: int, seed: int) -> pp.Simulation:
    m, d = pp.species('M D')
    rxns = [
        (m + m >> d).k(1),
        (d >> m + m).k(1),
    ]
    
    n = int(10 ** pop_exponent)
    m_init = n
    inits = {m: m_init}
    
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
    
    return sim

def ppsim_vilar_crn(pop_exponent: int, seed: int) -> pp.Simulation:
    a, da, dap, dr, drp, ma, mr, r, c = pp.species('A DA DAp DR DRp MA MR R C')
    rxns = [
        (da >> dap + a).k(50),
        (dr >> drp + r).k(0.01),
        (dap >> da + a).k(50),
        (drp >> dr + a).k(100),
        (ma >> ma + a).k(50),
        (mr >> mr + r).k(5),
        (a + da >> dap).k(1),
        (a + dr >> drp).k(1),
        (a + r >> c).k(2),
        (c >> r).k(1),
        (a >> None).k(1),
        (ma >> None).k(10),
        (mr >> None).k(0.5),
        (r >> None).k(0.2)
    ]
    n = int(10 ** pop_exponent)
    inits = {da: n//2, dr: n//2, a: 0, c: 0, dap: 0, drp: 0, ma: 0, mr: 0, r: 0}
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
    return sim

import rebop as rb

def rebop_vilar_with_inits(pop_exponent: int) -> tuple[rb.Gillespie, dict[str, int]]:
    """
    Create a rebop Gillespie simulation of the Vilar oscillator.
    
    Args:
        pop_exponent: Population size will be 10^pop_exponent
        
    Returns:
        tuple of (rebop.Gillespie object, initial conditions dict)
    """
    crn = rb.Gillespie()
    
    n = int(10 ** pop_exponent)
    
    # Rate constants from the Vilar model
    # Unimolecular rates (no volume scaling needed)
    alphaA = 50      # DA -> DA + MA
    alphaAp = 500    # DAp -> DAp + MA  
    alphaR = 0.01    # DR -> DR + MR
    alphaRp = 50     # DRp -> DRp + MR
    betaA = 50       # MA -> MA + A
    betaR = 5        # MR -> MR + R
    thetaA = 50      # DAp -> DA + A
    thetaR = 100     # DRp -> DR + A
    deltaA = 1       # A -> ∅ and C -> R
    deltaMA = 10     # MA -> ∅
    deltaMR = 0.5    # MR -> ∅
    deltaR = 0.2     # R -> ∅
    
    # Bimolecular rates (divide by volume n for discrete simulation)
    gammaA = 1 / n   # A + DA -> DAp
    gammaR = 1 / n   # A + DR -> DRp  
    gammaC = 2 / n   # A + R -> C
    
    # Transcription reactions (unimolecular)
    crn.add_reaction(alphaA, ['DA'], ['DA', 'MA'])
    crn.add_reaction(alphaAp, ['DAp'], ['DAp', 'MA'])
    crn.add_reaction(alphaR, ['DR'], ['DR', 'MR'])
    crn.add_reaction(alphaRp, ['DRp'], ['DRp', 'MR'])
    
    # Translation reactions (unimolecular)
    crn.add_reaction(betaA, ['MA'], ['MA', 'A'])
    crn.add_reaction(betaR, ['MR'], ['MR', 'R'])
    
    # Binding reactions (bimolecular - volume scaled)
    crn.add_reaction(gammaA, ['A', 'DA'], ['DAp'])
    crn.add_reaction(gammaR, ['A', 'DR'], ['DRp'])
    crn.add_reaction(gammaC, ['A', 'R'], ['C'])
    
    # Unbinding reactions (unimolecular)
    crn.add_reaction(thetaA, ['DAp'], ['DA', 'A'])
    crn.add_reaction(thetaR, ['DRp'], ['DR', 'A'])
    
    # Degradation reactions (unimolecular)
    crn.add_reaction(deltaA, ['A'], [])
    crn.add_reaction(deltaMA, ['MA'], [])
    crn.add_reaction(deltaMR, ['MR'], [])
    crn.add_reaction(deltaR, ['R'], [])
    
    # Complex dissociation (unimolecular)
    crn.add_reaction(deltaA, ['C'], ['R'])
    
    # Initial conditions: split n half and half between DA and DR
    inits = {
        'A': 0,
        'C': 0, 
        'DA': n // 2,
        'DAp': 0,
        'DR': n // 2,
        'DRp': 0,
        'MA': 0,
        'MR': 0,
        'R': 0
    }
    
    return crn, inits

def plot_dimerization_crn(pop_exponent: int, seed: int, num_runs: int = 1) -> None:
    import gpac as gp
    import numpy as np

    n = 10**pop_exponent
    m,d = gp.species('M D')
    rxns = [ m+m | d ]
    import matplotlib.pyplot as plt
    inits_discrete = { m: n }
    tmax = 2
    
    # Generate datasets for the specified number of runs
    datasets = []
    for i in range(num_runs):
        data = gp.rebop_crn_counts(rxns, inits_discrete, tmax, seed=seed+i)
        datasets.append(data)
    
    # Print first dataset for reference
    print(datasets[0])
    
    #gp.plot_gillespie(rxns, inits_discrete, tmax, figsize=(4,4), seed=seed, latex_legend=True)
    
    # Plot all datasets in a single plot
    import matplotlib
    matplotlib.rcParams.update({'font.size': 16})  # Increase font size
    plt.figure(figsize=(5, 5))
    
    plt.axvline(x=0.5, color='g', linestyle='--')

    # after first plot, color the rest shaded
    alpha = 0.3
    for i, data in enumerate(datasets):
        if i == 0:
            plt.plot(data.time, data.D, color='C1', label='$D$', alpha=1)
            plt.plot(data.time, data.M, color='C0', label='$M$', alpha=1)
        else:
            plt.plot(data.time, data.D, color='C1', alpha=alpha)
            plt.plot(data.time, data.M, color='C0', alpha=alpha)

    # # Plot D counts - all same color (first default color), only label the first one
    # for i, data in enumerate(datasets):
    #     if i == 0:
    #         plt.plot(data.time, data.D, color='C0', label='$D$')
    #     else:
    #         plt.plot(data.time, data.D, color='C0')
    
    # # Plot M counts - all same color (second default color), only label the first one
    # for i, data in enumerate(datasets):
    #     if i == 0:
    #         plt.plot(data.time, data.M, color='C1', label='$M$')
    #     else:
    #         plt.plot(data.time, data.M, color='C1')
    
    plt.xlabel('time')
    plt.ylabel('count')
    plt.legend()
    plt.savefig('data/dimerization_counts_vs_time.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    pop_exponent = 2
    trials_exponent = 9
    final_time = 0.5
    species_name = 'D'
    seed = 1
    num_runs = 10  # Number of simulation runs to plot
    rebop_crn, rebop_inits = rebop_dimerization_with_inits(pop_exponent)
    ppsim_sim = ppsim_dimerization_crn(pop_exponent, seed)

    # plot_dimerization_crn_ppsim_with_null(pop_exponent, seed)
    
    plot_dimerization_crn(pop_exponent, seed, num_runs)
    
    # ppsim_sim.run(final_time, 0.01) # type: ignore
    # print(f'done with ppsim')
    # ppsim_sim.history.plot(figsize=(10, 4)) # type: ignore
    # plt.show()
    
    # write_rebop_count_samples(rebop_crn, rebop_inits, pop_exponent, trials_exponent, species_name, final_time)
    # write_ppsim_count_samples(ppsim_sim, pop_exponent, trials_exponent, species_name, final_time)
    # plot_rebop_ppsim_histogram(pop_exponent, trials_exponent, species_name, final_time)

if __name__ == "__main__":
    main()