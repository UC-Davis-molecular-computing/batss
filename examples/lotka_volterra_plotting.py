import sys
import importlib.util
import os
from pathlib import Path
import gpac as gp

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

    # Get the default color cycle
    # default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # print(default_colors)
    # return
    # for pop_exponent in [3,4,5,8]:
    for pop_exponent in [5]:
        #XXX: pop_exponent 5 and 6 these show the slowdown bug in ppsim
        # going to time 20, for n=10^5 around time 6.966 (35% progress bar)
        # and for n=10^6, around time 13.718 (69% progress bar),
        # there is a massive slowdown in ppsim compared to the other time intervals.
        # Despite using the same random seed each time, the exact times when this
        # happens is stochastic, but it is always around the same time.
        make_and_save_plot(pop_exponent)
    
def make_and_save_plot(pop_exponent: int) -> None:
    seed = 1
    figsize = (4, 2)
    crn = rb.Gillespie()
    crn.add_reaction(.1 ** pop_exponent, ['R', 'F'], ['F', 'F'])
    crn.add_reaction(1, ['R'], ['R', 'R'])
    crn.add_reaction(1, ['F'], [])
    n = int(10 ** pop_exponent)
    p = 0.5
    r_init = int(n * p)
    f_init = n - r_init
    inits = {'R': r_init, 'F': f_init}
    end_time = 20.0
    num_samples = 10**3
    results_rebop = {}
    print(f'running rebop with n = 10^{pop_exponent}')
    results_rebop = crn.run(inits, end_time, num_samples, rng=seed)

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

    gp_r, gp_f = gp.species('R F')
    gp_rxns = [
        (gp_r+gp_f >> 2*gp_f).k(1),
        (gp_r >> 2*gp_r).k(1),
        (gp_f >> gp.empty).k(1),
    ]
    gp_inits = {gp_r: 0.5, gp_f: 0.5}
    t_eval = np.linspace(0, 20, num_samples)
    # Here's something funny: the default RK45 method makes thegpac data bad;
    # so it's more precise to rely on the discrete model than the ODE model.
    gp_data = gp.integrate_crn_odes(gp_rxns, gp_inits, t_eval, method='Radau')
    gp_times = gp_data.t
    gp_r_counts = gp_data.y[0] * n
    gp_f_counts = gp_data.y[1] * n


    # sim.history.plot(figsize = (15,4))
    # plt.ylim(0, 2.1 * n)
    # plt.title('lotka volterra (with batching)')
    
    # print(f"Total reactions simulated: {sampling_increment * len(results_rebop['R'])}")

    # f, ax = plt.subplots()
    f, ax = plt.subplots(figsize=figsize)

    blue, orange, green, red  = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'
    ax.plot(sim.history['R'], label = 'R (batching)', color=blue)
    ax.plot(sim.history['F'], label = 'F (batching)', color=orange)
    ax.plot(results_rebop['time'], results_rebop['R'], label='R (rebop)', color=red, linestyle='--')
    ax.plot(results_rebop['time'], results_rebop['F'], label='F (rebop)', color=green, linestyle='--')
    ax.plot(gp_times, gp_r_counts, label = 'R (ODE)', color='C4', linestyle=':')
    ax.plot(gp_times, gp_f_counts, label = 'F (ODE)', color='C5', linestyle=':')
    # ax.legend(loc='center left')
    ax.set_ylim(0, 2.6 * n)

    # Force scientific notation on y-axis
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # Always use scientific notation
    ax.yaxis.set_major_formatter(formatter)

    plt.savefig(f'data/lotka_volterra_counts_time20_n1e{pop_exponent}.pdf', bbox_inches='tight')
    plt.show()

    # make separate file for legend to share among all 4 subplots
    # handles, labels = plt.gca().get_legend_handles_labels()
    # figLegend = plt.figure()
    # figLegend.legend(handles, labels, ncols=3, loc='center')
    # print(f'saving legend for lotka volterra n=10^{pop_exponent}')
    # figLegend.savefig(f'data/lotka_volterra_counts_time20_legend.pdf', format='pdf', bbox_inches='tight')
    

def plot_null_reactions(pop_exponent: int, seed: int) -> None:
    figsize = (12, 6)
    n = int(10 ** pop_exponent)
    p = 0.5
    r_init = int(n * p)
    f_init = n - r_init
    inits = {'R': r_init, 'F': f_init}
    end_time = 20.0
    num_samples = 10**3
    
    r,f = pp.species('R F')
    rxns = [
        (r+f >> 2*f).k(1.5), # blows up population size less than choosing 1; David S used this in PNAS paper
        (r >> 2*r).k(1),
        (f >> None).k(1),
    ]
    
    inits = {r: r_init, f: f_init}
    sim = pp.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)

    print(f'running ppsim with n = 10^{pop_exponent}')
    sim.run(end_time, end_time / num_samples)

    
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
    ax.plot(sim.history['R'], label='R', color=blue)
    ax.plot(sim.history['F'], label='F', color=orange)

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
    plt.savefig(f'data/lotka_volterra_plot_with_passive_reactions_n1e{pop_exponent}.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # main()
    pop_exponent = 8
    seed = 1
    plot_null_reactions(pop_exponent, seed)