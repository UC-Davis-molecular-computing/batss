import math
import numpy as np
import ppsim as pp
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import timeit
import time
import rebop as rb
from scipy import stats

def measure_time(fn, trials=1):
    """
    Measure the time taken by a function over a number of trials.
    """
    start_time = time.perf_counter()
    for _ in range(trials):
        fn()
    end_time = time.perf_counter()
    return (end_time - start_time) / trials

def test_time_scaling_vs_population():
    ppsim_times = []
    rebop_times = []
    ns_rebop = []
    ns_ppsim = []
    min_pop_exponent = 3
    max_pop_exponent_rebop = 10
    max_pop_exponent = 12
    figsize = (5,3)
    num_trials = 1
    end_time = 1.0
    seed = 1
    # for pop_exponent_increment in tqdm(range(num_ns)):
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f'n = 10^{pop_exponent}')
        a,b = pp.species('A B')
        
        predator_fraction = 0.5

        rxns = [
            (a+b >> 2*b).k(0.1 ** pop_exponent),
            (a >> 2*a).k(1),
            (b >> None).k(1),
        ]
        
        a_init = 1
        b_init = 0
        fake_inits = {a: a_init, b: b_init}
        sim = pp.Simulation(fake_inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
        
        def run_ppsim(n):
            a_init = int(n * (1 - predator_fraction))
            b_init = n - a_init
            inits = {a: a_init, b: b_init}
            sim.reset(inits) # type: ignore
            # s = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)
            sim.run(end_time, 0.5)
            # sim.simulator.write_profile() # type: ignore
        # sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True)
        
        crn = rb.Gillespie()
        crn.add_reaction(0.1 ** pop_exponent, ['A', 'B'], ['B', 'B'])
        crn.add_reaction(1, ['A'], ['A', 'A'])
        crn.add_reaction(1, ['B'], [])

        def run_rebop(n, t):
            a_init = int(n * (1 - predator_fraction))
            b_init = n - a_init
            inits = {"A": a_init, "B": b_init}
            crn.run(inits, t, 1, rng=seed)
        
        n = int(10 ** pop_exponent)
        
        if pop_exponent <= max_pop_exponent_rebop:
            if pop_exponent == min_pop_exponent:
                # for some reason the first time it runs, rebop takes a long time
                run_rebop(n, end_time)
                run_rebop(n, end_time)
            print('rebop')
            rebop_times.append(measure_time(lambda: run_rebop(n, end_time), num_trials))
            ns_rebop.append(n)
        print('ppsim')
        if pop_exponent == min_pop_exponent:
            run_ppsim(n)
        ppsim_times.append(measure_time(lambda: run_ppsim(n), num_trials))
        ns_ppsim.append(n)
        
    _, ax = plt.subplots(figsize = figsize)
    import matplotlib
    # matplotlib.rcParams.update({'font.size': 16}) # default font is too small for paper figures
    matplotlib.rcParams['mathtext.fontset'] = 'cm' # use Computer Modern font for LaTeX
    ax.loglog(ns_ppsim, ppsim_times, label="batching run time", marker="o")
    ax.loglog(ns_rebop, rebop_times, label="rebop run time", marker="o")
    ax.set_xlabel(f'Initial molecular count')
    ax.set_ylabel(f'Run time (s)')
    ax.set_ylim(bottom=None, top=10**3)
    ax.legend()
    plt.savefig("lotka_volterra_scaling_f128.pdf", bbox_inches='tight')
    # plt.savefig("lotka_volterra_scaling_f64.pdf", bbox_inches='tight')
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

def get_rebop_samples(pop_exponent, trials, predator_count, state, final_time):
    n = 10 ** pop_exponent
    output = []
    total_time_inside_simulation = 0.0
    total_time_outside = 0.0
    for _ in tqdm(range(trials)):
        crn = rb.Gillespie()
        crn.add_reaction(0.1 ** pop_exponent, ['A', 'B'], ['B', 'B'])
        crn.add_reaction(1, ['A'], ['A', 'A'])
        crn.add_reaction(1, ['B'], [])
        b_init = predator_count
        a_init = n - b_init
        inits = {"A": a_init, "B": b_init}
        # It should be very roughly 1 step every 1/n real time, so to get a particular number
        # of steps, it should be safe to run for, say, 3 times that much time
        while True:
            try:
                results_rebop = crn.run(inits, final_time, 1)
                # print(f"There are {len(results_rebop[state])} total steps in rebop simulation.")
                # print(results_rebop[state])
                output.append(int(results_rebop[state][-1]))
                break
            except IndexError:
                pass
                print("Index error caught and ignored. Rebop distribution may be slightly off.")
    return output

def test_distribution():
    pop_exponent = 10.69
    trials_exponent = 0
    final_time_exponent = -2
    a,b = pp.species('A B')
    
    final_time = 10 ** final_time_exponent
    rxns = [
        (a+b >> 2*b).k(0.1 ** pop_exponent),
        (a >> 2*a).k(1),
        (b >> None).k(1),
    ]

    n = 10 ** pop_exponent
    print(f"n = {n}")
    a_init = n // 2
    b_init = n - a_init
    inits = {a: a_init, b: b_init}
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=4) #type: ignore
    
    trials = 3
    
    # The simulator multiplies by n currently so just gonna be lazy here.
    # state = 'A'
    state = 'B'
    results_batching = sim.sample_future_configuration(final_time, num_samples=trials)
    print(f"total reactions simulated by batching: {sim.simulator.discrete_steps_no_nulls}") #type: ignore
    # sim.simulator.write_profile() # type: ignore
    # results_rebop = get_rebop_samples(pop_exponent, trials, b_init, state, final_time)
    # fig, ax = plt.subplots(figsize = (10,4))
    # # print((results_batching).shape)
    # # print((results_batching[state].squeeze().tolist()))
    # # print(results_rebop) 
    # # print([results_batching[state].squeeze().tolist(), results_rebop])
    # # ax.hist(results_rebop)
    # ax.hist([results_batching[state].squeeze().tolist(), results_rebop], # type: ignore
    #         #bins = np.linspace(int(n*0.32), int(n*.43), 20), # type: ignore
    #         alpha = 1, label=['ppsim', 'rebop']) #, density=True, edgecolor = 'k', linewidth = 0.5)
    # ax.legend()

    # ax.set_xlabel(f'Count of state {state}')
    # ax.set_ylabel(f'Number of samples')
    # ax.set_title(f'State {state} distribution sampled at simulated time {final_time} ($10^{trials_exponent}$ samples)')
    
    # # plt.ylim(0, 200_000)

    # # plt.savefig(pdf_fn, bbox_inches='tight')
    # plt.show()


def main():
    test_time_scaling_vs_population()
    # test_distribution()

if __name__ == "__main__":
    main()