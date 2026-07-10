"""
Dimerization CRN benchmarks and plots, using ``batss.testing``.

Port of the benchmarking/trajectory pieces of ``dimerization_testing.py`` onto
the reusable :class:`~batss.testing.CRNSpec` interface.
"""

from __future__ import annotations

from typing import Any

import batss as bt
from batss.benchmarking import CRNSpec, benchmark_runtimes, plot_runtimes, plot_trajectory

DATA_DIR = "data"

# Values for ``sim.simulator.heuristic_gillespie_switching`` -- which batch/Gillespie switching
# rule to use. Set e.g. ``sim.simulator.heuristic_gillespie_switching = HEURISTIC_PROXY`` to
# A/B-test the simpler rule empirically.
HEURISTIC_WALLCLOCK = 0  # default: switch by measured wall-clock per unit continuous time
HEURISTIC_PROXY = 1  # simpler: the original reaction-count rule only (tunable via proxy_threshold)


def dimerization_spec() -> CRNSpec:
    m, d = bt.species("M D")
    rxns = [
        (m + m >> d).k(1),
        (d >> m + m).k(1),
    ]
    return CRNSpec(
        name="dimerization",
        rxns=rxns,
        inits_from_n=lambda n: {m: n},
        benchmark_end_time=0.5,
    )


def oregonator_spec() -> CRNSpec:
    """Oregonator (Belousov-Zhabotinsky) oscillator, in the count~n regime.

    The two bimolecular rates are reduced from the textbook values (1.6e9, 4e7 -- calibrated
    for concentrations ~1e-7, which collapse everything to 0 at count~n) to 1000/40 so the
    limit cycle lives at count~n. The initial condition sits on that limit cycle so it
    oscillates immediately; its concentrations sum to ~0.54 (not 1), which is why batss uses
    ``volume=n`` (see ``_batss_sim``). This is the CRN that exposes the batch/Gillespie
    switching cost: batch mode is far slower than Gillespie here below a large crossover n.
    """
    x1, x2, x3 = bt.species("X1 X2 X3")
    rxns = [
        (x2 >> x1).k(0.0871),
        (x2 + x1 >> None).k(1000),
        (x1 >> 2 * x1 + x3).k(520),
        (2 * x1 >> None).k(40),
        (x3 >> x2).k(443.324),
        (x3 >> None).k(2.676),
    ]
    return CRNSpec(
        name="oregonator",
        rxns=rxns,
        inits_from_n=lambda n: {x1: int(0.0029 * n), x2: int(0.5358 * n), x3: int(0.0034 * n)},
        benchmark_end_time=1.0,
    )


def mode_split(sim: bt.Simulation) -> dict[str, float]:
    """Per-mode wall-clock accumulators from the Rust simulator, after a run.

    Read-only observability for confirming which engine the wall-clock-aware heuristic chose:
    wall-clock seconds and continuous time advanced in each of batch / Gillespie mode (and
    their ratio w/dt), call counts, number of mode switches, and total switch overhead. These
    have no effect on the decision.
    """
    sim_obj: Any = sim.simulator  # pyo3-defined getters aren't visible to static analysis
    s: Any = sim_obj.switch

    def wdt(wall: float, ct: float) -> float:
        return wall / ct if ct > 0 else float("nan")

    return {
        "batch_seconds": s.batch_wallclock_seconds,
        "gillespie_seconds": s.gillespie_wallclock_seconds,
        "batch_continuous_time": s.batch_continuous_time,
        "gillespie_continuous_time": s.gillespie_continuous_time,
        "batch_w_per_dt": wdt(s.batch_wallclock_seconds, s.batch_continuous_time),
        "gillespie_w_per_dt": wdt(s.gillespie_wallclock_seconds, s.gillespie_continuous_time),
        "batch_calls": s.batch_calls,
        "gillespie_calls": s.gillespie_calls,
        "mode_switches": s.mode_switches,
        "switch_overhead_seconds": s.switch_overhead_seconds,
    }


def main() -> None:
    spec = dimerization_spec()

    # 1) runtime scaling: batss vs rebop
    benchmark_runtimes(
        spec,
        pop_sizes=[10**e for e in range(3, 11)],
        data_dir=DATA_DIR,
    )
    plot_runtimes(
        spec,
        data_dir=DATA_DIR,
    )

    # 2) trajectory + passive fraction
    plot_trajectory(
        spec,
        n=10**2,
        end_time=2.0,
        data_dir=DATA_DIR,
        seed=1,
        num_samples=200,
    )


if __name__ == "__main__":
    main()
