import sys
import importlib.util
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


import batss as bt
from matplotlib import pyplot as plt
import rebop as rb
import numpy as np


def main():
    crn = rb.Gillespie()
    pop_exponent = 4
    crn.add_reaction(30, ["X1"], ["X1", "X1"])
    crn.add_reaction(0.5 * 0.1**pop_exponent, ["X1", "X1"], ["X1"])
    crn.add_reaction(1 * 0.1**pop_exponent, ["X2", "X1"], ["X2", "X2"])
    crn.add_reaction(10, ["X2"], [])
    crn.add_reaction(1 * 0.1**pop_exponent, ["X1", "X3"], [])
    crn.add_reaction(16.5, ["X3"], ["X3", "X3"])
    crn.add_reaction(0.5 * 0.1**pop_exponent, ["X3", "X3"], ["X3"])
    n = int(10**pop_exponent)
    x1_init = int(n / 3)
    x2_init = int(n / 3)
    x3_init = int(n / 3)
    inits = {"X1": x1_init, "X2": x2_init, "X3": x3_init}
    end_time = 1
    num_samples = 500
    results_rebop = {}
    results_rebop = crn.run(inits, end_time, num_samples)

    x1, x2, x3 = bt.species("X1 X2 X3")
    rxns = [
        (x1 >> 2 * x1).k(30),
        (2 * x1 >> x1).k(0.5),
        (x2 + x1 >> 2 * x2).k(1),
        (x2 >> None).k(10),
        (x1 + x3 >> None).k(1),
        (x3 >> 2 * x3).k(16.5),
        (2 * x3 >> x3).k(0.5),
    ]
    inits = {x1: x1_init, x2: x2_init, x3: x3_init}
    print(inits)  # type: ignore
    sim = bt.Simulation(inits, rxns, simulator_method="crn", continuous_time=True)
    print(sim.simulator.transition_probabilities)  # type: ignore

    sim.run(end_time, end_time / num_samples)
    print(sim.simulator.transition_probabilities)  # type: ignore
    # sim.history.plot(figsize = (15,4))
    # plt.ylim(0, 2.1 * n)
    # plt.title('lotka volterra (with batching)')

    # print(f"Total reactions simulated: {num_samples * len(results_rebop['X1'])}")

    f, ax = plt.subplots()

    ax.plot(results_rebop["time"], results_rebop["X1"], label="X1 (rebop)")
    ax.plot(results_rebop["time"], results_rebop["X2"], label="X2 (rebop)")
    ax.plot(results_rebop["time"], results_rebop["X3"], label="X3 (rebop)")
    # print(sim.history)
    # print(results_rebop)
    # print(np.linspace(0, end_time, num_samples + 1))
    # print(sim.history['A'])
    # ax.plot(sim.history['K'], label = 'K (ppsim)')
    ax.plot(sim.history["X1"], label="X1 (ppsim)")
    ax.plot(sim.history["X2"], label="X2 (ppsim)")
    ax.plot(sim.history["X3"], label="X3 (ppsim)")
    # ax2.plot(np.linspace(0, end_time, num_samples + 1), sim.history['A'], label='A (ppsim)')
    # ax2.plot(np.linspace(0, end_time, num_samples + 1), sim.history['B'], label='B (ppsim)')
    # ax.hist([results_rebop['A'], results_rebop['B']], bins = np.linspace(0, n, 20),
    #         alpha = 1, label=['A', 'B']) #, density=True, edgecolor = 'k', linewidth = 0.5)
    ax.legend()
    # sim.simulator.write_profile() # type: ignore

    plt.show()
    # We could just write gpac reactions directly, but this is ensuring the gpac_format function works.
    # gp_rxns, gp_inits = pp.gpac_format(lotka_volterra, inits)
    # print('Reactions:')
    # for rxn in gp_rxns:
    #     print(rxn)
    # print('Initial conditions:')
    # for sp, count in gp_inits.items():
    #     print(f'{sp}: {count}')

    # # for trials_exponent in range(3, 7):
    # # for trials_exponent in range(3, 8):
    # print(f'*************\nCollecting rebop data for pop size 10^{pop_exponent} with 10^{trials_exponent} trials\n')
    # results_rebop = gp.rebop_crn_counts(gp_rxns, gp_inits, end_time)
    # df = pl.DataFrame(results_rebop).to_pandas()
    # df.plot(figsize=(10,5)) # .plot(figsize = (6, 4))
    # plt.title('approximate majority (ppsim)')
    # plt.show()
    # print("Done!")


def plot_null_reactions(pop_exponent: int, seed: int) -> None:
    n = int(10**pop_exponent)
    x1_init = int(n / 3)
    x2_init = int(n / 3)
    x3_init = int(n / 3)
    inits = {"X1": x1_init, "X2": x2_init, "X3": x3_init}

    x1, x2, x3 = bt.species("X1 X2 X3")
    rxns = [
        (x1 >> 2 * x1).k(30),
        (2 * x1 >> x1).k(0.5),
        (x2 + x1 >> 2 * x2).k(1),
        (x2 >> None).k(10),
        (x1 + x3 >> None).k(1),
        (x3 >> 2 * x3).k(16.5),
        (2 * x3 >> x3).k(0.5),
    ]
    inits = {x1: x1_init, x2: x2_init, x3: x3_init}
    figsize = (12, 6)
    end_time = 10.0
    num_samples = 10**3

    sim = bt.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)

    print(f"running batss with n = 10^{pop_exponent}")
    sim.run(end_time, end_time / num_samples)

    # total steps starts with 0 (and for some reason ends with 0, I don't get that),
    # so we slice it to remove the first and last element to avoid dividing by zero.
    total_steps = np.array(sim.discrete_steps_total_last_run)[1:-1]
    non_null_steps = np.array(sim.discrete_steps_no_nulls_last_run)[1:-1]
    null_steps = total_steps - non_null_steps
    null_fractions = null_steps / total_steps
    non_null_fractions = non_null_steps / total_steps
    times = sim.history.index.tolist()[1:-1]  # make same length as null_fractions

    f, ax = plt.subplots(figsize=figsize)

    blue, orange, green, red = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
    # Create the primary plot with species counts (left y-axis)
    ax.plot(sim.history["X1"], label="X1", color=blue)
    ax.plot(sim.history["X2"], label="X2", color=orange)
    ax.plot(sim.history["X3"], label="X3", color=green)

    # Set up the left y-axis
    ax.set_ylabel("counts")
    ax.set_xlabel("Simulated time (Rossler)")

    # Create a second y-axis that shares the same x-axis
    ax2 = ax.twinx()

    # Plot null_fractions on the second y-axis
    ax2.plot(times, non_null_fractions, label="non-passive", color=red)

    # Set up the right y-axis
    ax2.set_ylabel("fraction of real (non-passive) reactions")
    ax2.set_ylim(0.0, 1.0)

    # Create a single legend with handles from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower right")
    plt.savefig(f"data/rossler_plot_with_passive_reactions_n1e{pop_exponent}.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
