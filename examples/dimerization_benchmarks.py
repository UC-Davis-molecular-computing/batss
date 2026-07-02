"""
Dimerization CRN benchmarks and plots, using ``batss.testing``.

Port of the benchmarking/trajectory pieces of ``dimerization_testing.py`` onto
the reusable :class:`~batss.testing.CRNSpec` interface.
"""

from __future__ import annotations

import batss as bt
from batss.benchmarking import CRNSpec, benchmark_runtimes, plot_runtimes, plot_trajectory

DATA_DIR = "data"


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
