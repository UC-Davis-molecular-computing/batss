from tqdm import tqdm
from matplotlib import pyplot as plt, ticker
import time
import rebop as rb
import json

import importlib.util
from pathlib import Path
import sys

if False:
    # Path to your renamed .pyd file
    custom_pyd_path = Path("C:/Dropbox/git/batss-rust/python/batss/batss_rust/batss_rust.cp312-win_amd64_rebop.pyd")
    # custom_pyd_path = Path("C:/Dropbox/git/batss-rust/python/batss/batss_rust/batss_rust.cp312-win_amd64_f128.pyd")

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

import batss


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
    with open(fn, "w") as f:
        json.dump(results, f, indent=4)


def create_rebop_running_time_data_LV(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    rebop_times = []
    ns_rebop = []
    seed = 1

    print("creating rebop data")
    # for pop_exponent_increment in tqdm(range(num_ns)):
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f"n = 10^{pop_exponent}")

        crn = rb.Gillespie()
        crn.add_reaction(0.1**pop_exponent, ["R", "F"], ["F", "F"])
        crn.add_reaction(1, ["R"], ["R", "R"])
        crn.add_reaction(1, ["F"], [])

        predator_fraction = 0.5
        n = int(10**pop_exponent)

        r_init = int(n * (1 - predator_fraction))
        f_init = n - r_init
        rebop_inits = {"R": r_init, "F": f_init}

        def run_rebop():
            crn.run(rebop_inits, end_time, 1, rng=seed)

        if pop_exponent == min_pop_exponent:
            # for some reason the first time it runs, rebop takes a long time
            run_rebop()
            run_rebop()
        print("rebop")
        runtime = measure_time(run_rebop, num_trials)
        rebop_times.append(runtime)
        print(f"Rebop took {runtime}s.")
        ns_rebop.append(n)
        write_results(fn, rebop_times, ns_rebop)


def create_rebop_running_time_data_rossler(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    rebop_times = []
    ns_rebop = []
    seed = 1

    print("creating rebop data")
    # for pop_exponent_increment in tqdm(range(num_ns)):
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f"n = 10^{pop_exponent}")

        crn = rb.Gillespie()
        crn.add_reaction(30, ["X1"], ["X1", "X1"])
        crn.add_reaction(0.5 * 0.1**pop_exponent, ["X1", "X1"], ["X1"])
        crn.add_reaction(1 * 0.1**pop_exponent, ["X2", "X1"], ["X2", "X2"])
        crn.add_reaction(10, ["X2"], [])
        crn.add_reaction(1 * 0.1**pop_exponent, ["X1", "X3"], [])
        crn.add_reaction(16.5, ["X3"], ["X3", "X3"])
        crn.add_reaction(0.5 * 0.1**pop_exponent, ["X3", "X3"], ["X3"])

        n = int(10**pop_exponent)
        n = int(10**pop_exponent)
        x1_init = int(n / 3)
        x2_init = int(n / 3)
        x3_init = int(n / 3)

        rebop_inits = {"X1": x1_init, "X2": x2_init, "X3": x3_init}

        def run_rebop():
            crn.run(rebop_inits, end_time, 1, rng=seed)

        if pop_exponent == min_pop_exponent:
            # for some reason the first time it runs, rebop takes a long time
            run_rebop()
            run_rebop()
        print("rebop")
        runtime = measure_time(run_rebop, num_trials)
        rebop_times.append(runtime)
        print(f"Rebop took {runtime}s.")
        ns_rebop.append(n)
        write_results(fn, rebop_times, ns_rebop)


def create_rebop_running_time_data_oregonator(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    rebop_times = []
    ns_rebop = []
    seed = 1

    print("creating rebop data")
    # for pop_exponent_increment in tqdm(range(num_ns)):
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f"n = 10^{pop_exponent}")

        crn = rb.Gillespie()
        crn.add_reaction(0.1, ["X2"], ["X1"])
        crn.add_reaction(1000 * (0.1**pop_exponent), ["X2", "X1"], [])
        crn.add_reaction(520, ["X1"], ["X1", "X1", "X3"])
        crn.add_reaction(4 * (10**1) * (0.1**pop_exponent), ["X1", "X1"], [])
        crn.add_reaction(443.324, ["X3"], ["X2"])
        crn.add_reaction(2.676, ["X3"], [])

        n = int(10**pop_exponent)
        n = int(10**pop_exponent)
        x1_init = int(n / 3)
        x2_init = int(n / 3)
        x3_init = int(n / 3)

        rebop_inits = {"X1": x1_init, "X2": x2_init, "X3": x3_init}

        def run_rebop():
            crn.run(rebop_inits, end_time, 1, rng=seed)

        if pop_exponent == min_pop_exponent:
            # for some reason the first time it runs, rebop takes a long time
            run_rebop()
            run_rebop()
        print("rebop")
        runtime = measure_time(run_rebop, num_trials)
        rebop_times.append(runtime)
        print(f"Rebop took {runtime}s.")
        ns_rebop.append(n)
        write_results(fn, rebop_times, ns_rebop)


def create_gillespy_running_time_data(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    gl_times = []
    gl_ns = []
    seed = 1

    print("creating Gillespy2 data")
    # for pop_exponent_increment in tqdm(range(num_ns)):
    r, f = batss.species("R F")
    batss_rxns = [
        (r + f >> 2 * f).k(1),
        (r >> 2 * r).k(1),
        (f >> None).k(1),
    ]
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f"n = 10^{pop_exponent}")

        n = int(10**pop_exponent)
        predator_fraction = 0.5
        r_init = int(n * (1 - predator_fraction))
        f_init = n - r_init
        batss_inits = {r: r_init, f: f_init}
        gl_crn = batss.gillespy2_format(batss_inits, batss_rxns, n)
        gl_crn.timespan((0, end_time))
        # print(f'gl_crn = {gl_crn}')

        # results = gl_crn.run(t=end_time, increment=0.1, algorithm='SSA')
        # print(f'results = {results}')

        def run_gillespy():
            gl_crn.run(algorithm="SSA")

        measured_time = measure_time(run_gillespy, num_trials)
        gl_times.append(measured_time)
        gl_ns.append(n)
        print(f"Gillespy2 took {measured_time} seconds to run with n = 10^{pop_exponent}")
        write_results(fn, gl_times, gl_ns)


def create_batss_running_time_data_LV(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    batss_times = []
    ns_batss = []
    seed = 1
    r, f = batss.species("R F")
    rxns = [
        (r + f >> 2 * f).k(1),
        (r >> 2 * r).k(1),
        (f >> None).k(1),
    ]

    print("creating batss data")
    # for pop_exponent_increment in tqdm(range(num_ns)):
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f"n = 10^{pop_exponent}")

        predator_fraction = 0.5
        n = int(10**pop_exponent)
        r_init = int(n * (1 - predator_fraction))
        f_init = n - r_init
        batss_inits = {r: r_init, f: f_init}
        sim = batss.Simulation(batss_inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)

        def run_batss():
            sim.run(end_time, 0.5 * end_time)

        if pop_exponent == min_pop_exponent:
            run_batss()
        runtime = measure_time(run_batss, num_trials)
        batss_times.append(runtime)
        print(f"batss took {runtime}s.")
        ns_batss.append(n)
        write_results(fn, batss_times, ns_batss)


def create_batss_running_time_data_rossler(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    batss_times = []
    ns_batss = []
    seed = 1
    x1, x2, x3 = batss.species("X1 X2 X3")
    rxns = [
        (x1 >> 2 * x1).k(30),
        (2 * x1 >> x1).k(0.5),
        (x2 + x1 >> 2 * x2).k(1),
        (x2 >> None).k(10),
        (x1 + x3 >> None).k(1),
        (x3 >> 2 * x3).k(16.5),
        (2 * x3 >> x3).k(0.5),
    ]

    print("creating batss data")
    # for pop_exponent_increment in tqdm(range(num_ns)):
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f"n = 10^{pop_exponent}")

        n = int(10**pop_exponent)
        x1_init = int(n / 3)
        x2_init = int(n / 3)
        x3_init = int(n / 3)
        batss_inits = {x1: x1_init, x2: x2_init, x3: x3_init}
        sim = batss.Simulation(batss_inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)

        def run_batss():
            sim.run(end_time, 0.5 * end_time)

        if pop_exponent == min_pop_exponent:
            run_batss()
        runtime = measure_time(run_batss, num_trials)
        batss_times.append(runtime)
        print(f"batss took {runtime}s.")
        ns_batss.append(n)
        write_results(fn, batss_times, ns_batss)


def create_batss_running_time_data_oregonator(fn: str, min_pop_exponent: int, max_pop_exponent: int, end_time: float):
    num_trials = 1
    batss_times = []
    ns_batss = []
    seed = 1
    x1, x2, x3 = batss.species("X1 X2 X3")
    rxns = [
        (x2 >> x1).k(0.1),
        (x2 + x1 >> None).k(1000),
        (x1 >> 2 * x1 + x3).k(520),
        (2 * x1 >> None).k(4 * 10**1),
        (x3 >> x2).k(443.324),
        (x3 >> None).k(2.676),
    ]
    print("creating batss data")
    # for pop_exponent_increment in tqdm(range(num_ns)):
    for pop_exponent in range(min_pop_exponent, max_pop_exponent + 1):
        print(f"n = 10^{pop_exponent}")

        n = int(10**pop_exponent)
        x1_init = int(n / 3)
        x2_init = int(n / 3)
        x3_init = int(n / 3)
        batss_inits = {x1: x1_init, x2: x2_init, x3: x3_init}
        sim = batss.Simulation(batss_inits, rxns, simulator_method="crn", continuous_time=True, seed=seed)

        def run_batss():
            sim.run(end_time, 0.5 * end_time)

        if pop_exponent == min_pop_exponent:
            run_batss()
        runtime = measure_time(run_batss, num_trials)
        batss_times.append(runtime)
        print(f"batss took {runtime}s.")
        ns_batss.append(n)
        write_results(fn, batss_times, ns_batss)


def read_results(fn: str) -> tuple[list[int], list[float]]:
    with open(fn, "r") as f:
        data = json.load(f)
    ns = [item[0] for item in data]
    times = [item[1] for item in data]
    return ns, times


def plot_results(
    fn_rebop_data: str, fn_rebop_rust_data: str, fn_batss_data_f64: str, fn_batss_data_f128: str, fn_out: str
):
    # figsize = (6,4)
    figsize = (5, 4)
    _, ax = plt.subplots(figsize=figsize)
    # matplotlib.rcParams.update({'font.size': 14}) # default font is too small for paper figures
    # matplotlib.rcParams['mathtext.fontset'] = 'cm' # use Computer Modern font for LaTeX
    rebop_ns, rebop_times = read_results(fn_rebop_data)
    rebop_rust_ns, rebop_rust_times = read_results(fn_rebop_rust_data)
    batss_ns_f64, batss_times_f64 = read_results(fn_batss_data_f64)
    batss_ns_f128, batss_times_f128 = read_results(fn_batss_data_f128)
    # ax.loglog(batss_ns_f64, batss_times_f64, label="batching (f64)", marker="o")
    ax.loglog(batss_ns_f128, batss_times_f128, label="batching", marker=".")
    ax.loglog(rebop_ns, rebop_times, label="rebop (Python)", marker="^")
    ax.loglog(rebop_rust_ns, rebop_rust_times, label="rebop (Rust)", marker="s")
    ax.set_xlabel("initial molecular count (Oregonator)")
    ax.set_ylabel("run time (s)")
    ax.set_xticks([10**i for i in range(3, 12)])
    ax.minorticks_off()
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=None, top=10**5)

    ax.yaxis.set_major_locator(ticker.LogLocator(numticks=999))
    ax.yaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))

    plt.savefig(fn_out, bbox_inches="tight")
    plt.show()
    # print(stats.linregress([math.log(x) for x in ns_batss], [math.log(x) for x in batss_times]))
    # print(stats.linregress([math.log(x) for x in ns_batss], [math.log(x) for x in rebop_times]))
    # print(ns_batss)
    # print(batss_times)
    # print(rebop_times)
    return


def fn_count_samples(alg: str, pop_exponent: int, trials_exponent: int, species: str, final_time: float) -> str:
    return f"data/lk_{alg}_{species}-counts_time{final_time}_n1e{pop_exponent}_trials1e{trials_exponent}.json"


def write_rebop_count_samples(pop_exponent: int, trials_exponent: int, species: str, final_time: float) -> None:
    fn = fn_count_samples("rebop", pop_exponent, trials_exponent, species, final_time)
    print(f"collecting rebop data with n = 10^{pop_exponent} for {trials_exponent} trials")
    print(f"writing to {fn}")
    n = 10**pop_exponent
    crn = rb.Gillespie()
    crn.add_reaction(0.1**pop_exponent, ["R", "F"], ["F", "F"])
    crn.add_reaction(1, ["R"], ["R", "R"])
    crn.add_reaction(1, ["F"], [])
    from collections import defaultdict

    counts = defaultdict(int)
    for _ in tqdm(range(10**trials_exponent)):
        r_init = n // 2
        f_init = n // 2
        inits = {"R": r_init, "F": f_init}
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
    with open(fn, "w") as f:
        json.dump(counts, f, indent=4)


def sort_dict_by_key(d: dict) -> dict:
    """
    Sort a dictionary by its keys.
    """
    return dict(sorted(d.items(), key=lambda item: item[0]))


def write_batss_count_samples(pop_exponent: int, trials_exponent: int, species: str, final_time: float) -> None:
    print(f"collecting batss data with n = 10^{pop_exponent} for {trials_exponent} trials")
    fn = fn_count_samples("batss", pop_exponent, trials_exponent, species, final_time)
    n = 10**pop_exponent
    r, f = batss.species("R F")
    rxns = [
        (r + f >> 2 * f).k(1),
        (r >> 2 * r).k(1),
        (f >> None).k(1),
    ]
    a_init = n // 2
    b_init = n - a_init
    inits = {r: a_init, f: b_init}
    sim = batss.Simulation(inits, rxns, simulator_method="crn", continuous_time=True, seed=4)  # type: ignore
    from collections import defaultdict

    counts = defaultdict(int)
    trials = 10**trials_exponent
    results_batching = sim.sample_future_configuration(final_time, num_samples=trials)
    count_list: list[int] = results_batching[species].squeeze().tolist()  # type: ignore
    counts = defaultdict(int)
    for count in count_list:
        counts[count] += 1

    counts = sort_dict_by_key(counts)
    with open(fn, "w") as f:
        json.dump(counts, f, indent=4)


def read_count_samples(fn: str) -> list[int]:
    """
    Read the count samples from a JSON file.
    """
    with open(fn, "r") as f:
        counts = json.load(f)
    count_list = []
    for count, num_samples_with_count in counts.items():
        count_list.extend([int(count)] * num_samples_with_count)
    return count_list


def plot_rebop_batss_histogram(pop_exponent: int, trials_exponent: int, species: str, final_time: float):
    rebop_fn = fn_count_samples("rebop", pop_exponent, trials_exponent, species, final_time)
    batss_fn = fn_count_samples("batss", pop_exponent, trials_exponent, species, final_time)

    rebop_counts = read_count_samples(rebop_fn)
    batss_counts = read_count_samples(batss_fn)

    fig, ax = plt.subplots(figsize=(10, 4))
    # print((results_batching).shape)
    # print((results_batching[state].squeeze().tolist()))
    # print(results_rebop)
    # print([results_batching[state].squeeze().tolist(), results_rebop])
    # ax.hist(results_rebop)
    ax.hist(
        [batss_counts, rebop_counts],  # type: ignore
        bins=20,
        alpha=1,
        label=["batching", "rebop"],
    )  # , density=True, edgecolor = 'k', linewidth = 0.5)
    ax.legend()

    ax.set_xlabel(f"Count of species {species}")
    ax.set_ylabel("Number of samples")
    ax.set_title(
        f"Species {species} distribution sampled at simulated time {final_time}"
        f"(n=$10^{pop_exponent}$; trials=$10^{trials_exponent}$)"
    )

    # plt.ylim(0, 200_000)
    pdf_fn = fn_count_samples("batss-vs-rebop", pop_exponent, trials_exponent, species, final_time)
    pdf_fn = pdf_fn.replace(".json", ".pdf")
    plt.savefig(pdf_fn, bbox_inches="tight")
    plt.show()


def main():
    # create_batss_running_time_data_LV("data/lotka_volterra_time1_times_batss_f128.json", 3, 14, 1.0)
    # create_rebop_running_time_data_LV("data/lotka_volterra_time1_times_rebop_fix.json", 3, 11, 1.0)
    # create_rebop_running_time_data_rossler("data/rossler_time1_times_rebop.json", 3, 7, 1.0)
    # create_batss_running_time_data_rossler("data/rossler_time1_times_batss_f128.json", 3, 9, 1.0)
    # create_rebop_running_time_data_oregonator("data/oregonator_time1_times_rebop2.json", 3, 10, 1.0)
    # create_batss_running_time_data_oregonator("data/oregonator_time1_times_batss_f1282.json", 3, 10, 1.0)

    # create_rebop_running_time_data_oregonator("data/oregonator_time1_times_rebop2.json", 3, 4, 1.0)
    create_batss_running_time_data_oregonator("data/oregonator_time1_times_batss_f1282.json", 3, 4, 1.0)

    # plot_results(
    #     "data/oregonator_time1_times_rebop2.json",
    #     "data/oregonator_time1_times_rebop_rust.json",
    #     "data/oregonator_time1_times_batss_f1282.json",
    #     "data/oregonator_time1_times_batss_f1282.json",
    #     "data/oregonator_scaling_time1.pdf",
    # )
    # test_distribution()

    pop_exponent = 4
    trials_exponent = 3
    final_time = 1.0
    species = "F"
    # write_rebop_count_samples(pop_exponent, trials_exponent, species, final_time)
    # write_batss_count_samples(pop_exponent, trials_exponent, species, final_time)
    # plot_rebop_batss_histogram(pop_exponent, trials_exponent, species, final_time)


if __name__ == "__main__":
    main()
