"""
Dimerization CRN benchmarks and plots, using ``batss.testing``.

Port of the benchmarking/trajectory pieces of ``dimerization_testing.py`` onto
the reusable :class:`~batss.testing.CRNSpec` interface.
"""

from __future__ import annotations

import batss as bt
from batss.testing import CRNSpec, benchmark_runtimes, plot_runtimes, plot_trajectory

CACHE_DIR = "data"


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
    )


def main() -> None:
    spec = dimerization_spec()

    # 1) runtime scaling: batss vs rebop
    end_time = 0.5
    benchmark_runtimes(
        spec,
        pop_sizes=[10**e for e in range(3, 11)],
        end_time=end_time,
        cache_dir=CACHE_DIR,
    )
    plot_runtimes(
        spec,
        end_time=end_time,
        cache_dir=CACHE_DIR,
        out_path=f"{CACHE_DIR}/dimerization_runtime_t{end_time}.pdf",
    )

    # 2) trajectory + passive fraction
    plot_trajectory(
        spec,
        n=10**2,
        end_time=2.0,
        seed=1,
        num_samples=200,
        out_path=f"{CACHE_DIR}/dimerization_trajectory_n1e2.pdf",
    )


if __name__ == "__main__":
    main()
