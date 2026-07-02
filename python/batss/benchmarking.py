"""
Utilities for benchmarking and plotting CRNs with batss (and comparing against
rebop's Python bindings).

Define a CRN once as a :class:`CRNSpec`, then hand it to

* :func:`benchmark_runtimes` + :func:`plot_runtimes` to measure how run time
  scales with population size ``n``, batss vs rebop, and
* :func:`plot_trajectory` to plot species counts over time from a single batss
  run, optionally overlaying the fraction of passive (null) reactions on a
  dashed second y-axis.

The runtime benchmark caches per-(backend, n) measurements to JSON so reruns
skip work that's already cached.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import rebop as rb
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from batss.crn import Reaction, Specie
from batss.simulation import Simulation

__all__ = [
    "CRNSpec",
    "benchmark_runtimes",
    "plot_runtimes",
    "plot_trajectory",
]


@dataclass
class CRNSpec:
    """Single-source description of a CRN for benchmarking and plotting.

    ``rxns`` uses batss notation — e.g. ``[(r + f >> 2*f).k(1), (r >> 2*r).k(1)]``.
    For the rebop comparison, bimolecular (and higher-order) rate constants
    are automatically divided by ``n**(order-1)``; batss is left alone since
    it already expects rates in that form.

    ``inits_from_n`` maps ``n = 10**pop_exponent`` to a dict of initial counts
    keyed by the same :class:`Specie` objects used in ``rxns``.
    """

    name: str
    """Short identifier used in cache filenames and default plot titles."""

    rxns: list[Reaction]
    """Reactions in batss notation."""

    inits_from_n: Callable[[int], dict[Specie, int]]
    """Given ``n``, return initial counts keyed by the Species used in rxns."""

    benchmark_end_time: float
    """Simulated end time used by :func:`benchmark_runtimes` and :func:`plot_runtimes`."""


def _spec_species_names(spec: CRNSpec) -> list[str]:
    """Species that appear in spec.rxns, in first-seen order."""
    seen: dict[str, None] = {}
    for rxn in spec.rxns:
        for s in rxn.reactants.species:
            seen.setdefault(s.name, None)
        for s in rxn.products.species:
            seen.setdefault(s.name, None)
    return list(seen)


def _batss_sim(spec: CRNSpec, n: int, seed: int) -> Simulation:
    return Simulation(
        spec.inits_from_n(n),
        spec.rxns,
        simulator_method="crn",
        continuous_time=True,
        seed=seed,
    )


def _rebop_crn(spec: CRNSpec, n: int) -> tuple[rb.Gillespie, dict[str, int]]:
    crn = rb.Gillespie()
    all_names: set[str] = set()
    for rxn in spec.rxns:
        reactants = [s.name for s in rxn.reactants.species]
        products = [s.name for s in rxn.products.species]
        all_names.update(reactants)
        all_names.update(products)

        fwd_scale = n ** (len(reactants) - 1) if len(reactants) >= 1 else 1
        crn.add_reaction(rxn.rate_constant / fwd_scale, reactants, products)
        if rxn.reversible:
            rev_scale = n ** (len(products) - 1) if len(products) >= 1 else 1
            crn.add_reaction(rxn.rate_constant_reverse / rev_scale, products, reactants)

    inits: dict[str, int] = {s.name: c for s, c in spec.inits_from_n(n).items()}
    for name in all_names:
        inits.setdefault(name, 0)
    return crn, inits


# --- plot 1: runtime vs population size ---------------------------------------


def _runtime_path(data_dir: Path, spec: CRNSpec, backend: str, end_time: float) -> Path:
    return data_dir / f"{spec.name}_runtime_{backend}_t{end_time}.json"


def _load_runtime_json(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {int(n): float(t) for n, t in json.load(f)}


def _save_runtime_json(path: Path, data: dict[int, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([[n, t] for n, t in sorted(data.items())], f, indent=4)


def benchmark_runtimes(
    spec: CRNSpec,
    pop_sizes: Iterable[int],
    data_dir: str | Path = "data",
    backends: tuple[str, ...] = ("batss", "rebop"),
    seed: int = 1,
    overwrite: bool = False,
) -> None:
    """Time one run to ``spec.benchmark_end_time`` for each (backend, n) and cache to JSON.

    Already-cached (backend, n) pairs are skipped unless ``overwrite=True``.
    The first timed run for each backend is preceded by a warm-up run whose time
    is discarded (rebop's first call in a process is notably slower than
    subsequent ones).
    """
    end_time = spec.benchmark_end_time
    data_dir = Path(data_dir)
    for backend in backends:
        path = _runtime_path(data_dir, spec, backend, end_time)
        data = {} if overwrite else _load_runtime_json(path)
        print(f"benchmarking {spec.name} on {backend} -> {path}")

        warmed_up = False
        for n in pop_sizes:
            if n in data:
                print(f"  n={n:,}: cached {data[n]:.4g}s, skipping")
                continue

            if backend == "batss":
                sim = _batss_sim(spec, n, seed)

                def run() -> None:
                    sim.run(end_time, end_time, timer=False)
            elif backend == "rebop":
                crn, inits = _rebop_crn(spec, n)

                def run() -> None:
                    crn.run(inits, end_time, 1, rng=seed)
            else:
                raise ValueError(f"unknown backend: {backend!r}")

            if not warmed_up:
                run()
                warmed_up = True

            t0 = time.perf_counter()
            run()
            elapsed = time.perf_counter() - t0
            print(f"  n={n:,}: {elapsed:.4g}s")
            data[n] = elapsed
            _save_runtime_json(path, data)


def plot_runtimes(
    spec: CRNSpec,
    data_dir: str | Path = "data",
    backends: tuple[str, ...] = ("batss", "rebop"),
    pop_sizes: Iterable[int] | None = None,
    figsize: tuple[float, float] = (5, 4),
    ax: Axes | None = None,
) -> Axes:
    """Load cached runtimes and overlay them on log-log axes.

    If ``pop_sizes`` is given, only those ``n`` values are plotted (useful when
    the JSON cache contains data from earlier runs at larger ``n`` that you
    don't want on this plot). If ``None``, every cached point is plotted.

    The figure is saved to ``<data_dir>/<spec.name>_runtime_t<end_time>.pdf``.
    """
    end_time = spec.benchmark_end_time
    data_dir = Path(data_dir)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    wanted = set(pop_sizes) if pop_sizes is not None else None
    markers = {"batss": "o", "rebop": "^"}
    for backend in backends:
        path = _runtime_path(data_dir, spec, backend, end_time)
        data = _load_runtime_json(path)
        if not data:
            print(f"no runtime data for {backend} at {path}")
            continue
        ns = sorted(n for n in data if wanted is None or n in wanted)
        ts = [data[n] for n in ns]
        ax.loglog(ns, ts, label=backend, marker=markers.get(backend, "o"))

    ax.set_xlabel("initial molecular count")
    ax.set_ylabel("run time (s)")
    ax.set_title(f"{spec.name}: run time to simulate t={end_time}")
    ax.legend(loc="upper left")

    out_path = data_dir / f"{spec.name}_runtime_t{end_time}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    return ax


# --- plot 2: trajectory + passive-reaction fraction ---------------------------


def _passive_fractions(sim: Simulation) -> tuple[list[float], list[float]]:
    # Intervals with zero total steps (always the first and last, sometimes
    # others at small n) would divide by zero — drop them.
    total = np.array(sim.discrete_batched_steps_total_last_run)
    non_null = np.array(sim.discrete_batched_steps_no_passives_last_run)
    all_times = sim.history.index.tolist()
    times = [t for t, n in zip(all_times, total) if n > 0]
    fractions = [(n - m) / n for n, m in zip(total, non_null) if n > 0]
    return times, fractions


def plot_trajectory(
    spec: CRNSpec,
    n: int,
    end_time: float,
    data_dir: str | Path = "data",
    seed: int = 1,
    num_samples: int = 1000,
    species: list[str] | None = None,
    with_passive: bool = True,
    figsize: tuple[float, float] = (8, 4),
    ax: Axes | None = None,
) -> Axes:
    """Simulate ``spec`` with batss at population size ``n`` and plot counts vs time.

    The count y-axis starts at 0 (so the x-axis sits at y=0). When
    ``with_passive=True``, the fraction of passive (null) reactions is drawn as
    a dashed line on a second y-axis (range [0, 1]) so it's visually distinct
    from the count curves.

    ``species`` is a list of species-name strings to plot; defaults to the
    species named in ``spec.rxns``.

    The figure is saved to ``<data_dir>/<spec.name>_trajectory_n<n>_t<end_time>.pdf``.
    """
    # write n as $10^e$ if n is a power of 10, otherwise with commas
    exp = int(round(np.log10(n))) if n > 0 else -1
    n_str = f"$10^{{{exp}}}$" if exp >= 0 and 10**exp == n else f"{n:,}"
    sim = _batss_sim(spec, n, seed)
    print(f"running batss for {spec.name} at n={n_str}")
    sim.run(end_time, end_time / num_samples)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    species_to_plot = species if species is not None else _spec_species_names(spec)
    for sp in species_to_plot:
        ax.plot(sim.history[sp], label=sp)
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()

    if with_passive:
        times, fractions = _passive_fractions(sim)
        ax2 = ax.twinx()
        ax2.plot(
            times,
            fractions,
            label="passive",
            linestyle="--",
            color="#d62728",
        )
        ax2.set_ylabel("fraction of passive reactions")
        ax2.set_ylim(0.0, 1.0)
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    ax.legend(handles, labels, loc="lower right")
    ax.set_title(f"{spec.name}: n={n_str}")

    out_path = Path(data_dir) / f"{spec.name}_trajectory_n{n}_t{end_time}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    return ax
