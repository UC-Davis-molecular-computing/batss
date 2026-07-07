"""
Utilities for benchmarking and plotting CRNs with batss (and comparing against
rebop's Python bindings).

Define a CRN once as a :class:`CRNSpec`, then hand it to

* :func:`benchmark_runtimes` + :func:`plot_runtimes` to measure how run time
  scales with population size ``n``, batss vs rebop, and
* :func:`plot_trajectory` to plot species counts over time from a single batss
  run, optionally overlaying the fraction of non-passive (real) reactions on a
  dashed second y-axis.

The runtime benchmark caches per-(backend, n) measurements to JSON so reruns
skip work that's already cached.
"""

from __future__ import annotations

import json
import os
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
    # Pass volume=n explicitly rather than letting batss default it to the sum of
    # initial counts. This keeps the volume convention identical to _rebop_crn
    # (which divides a k-reactant rate by n**(k-1)) even when the initial counts
    # don't sum to n -- e.g. an initial condition placed on a limit cycle, whose
    # concentrations sum to less than 1.
    return Simulation(
        spec.inits_from_n(n),
        spec.rxns,
        simulator_method="crn",
        continuous_time=True,
        seed=seed,
        volume=n,
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


def _pow10_exponent(n: int) -> int | None:
    """Return ``k`` if ``n == 10**k`` for some integer ``k >= 0``, else ``None``."""
    if n <= 0:
        return None
    exp = int(round(np.log10(n)))
    return exp if exp >= 0 and 10**exp == n else None


def _format_pop_size(n: int) -> str:
    """Render ``n`` as ``'10^k'`` when it is a power of ten, else comma-grouped (``'1,000,000'``)."""
    exp = _pow10_exponent(n)
    return f"10^{exp}" if exp is not None else f"{n:,}"


# --- plot 1: runtime vs population size ---------------------------------------


def _runtime_path(data_dir: Path, spec: CRNSpec, backend: str, end_time: float, proxy: bool = False) -> Path:
    # The proxy heuristic caches to its own file so switching heuristics doesn't silently reuse (or
    # overwrite) the wall-clock results. Only batss has a heuristic, so ``proxy`` only tags batss.
    suffix = "_proxy" if proxy else ""
    return data_dir / f"{spec.name}_runtime_{backend}{suffix}_t{end_time}.json"


def _load_runtime_json(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {int(n): float(t) for n, t in json.load(f)}


def _save_runtime_json(path: Path, data: dict[int, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([[n, t] for n, t in sorted(data.items())], indent=4)
    # Write to a sibling temp file and atomically replace the target. This avoids the
    # window where a cloud-sync client (this repo lives under Dropbox) or antivirus grabs
    # the file mid-write, which on Windows surfaces intermittently as an OSError on open()
    # or replace() (e.g. [Errno 22] Invalid argument, or "os error 32"). Retry a few times
    # with a short backoff so a transient lock doesn't discard a measurement that may have
    # taken tens of seconds to compute. The spaces/non-ASCII in the filename are NOT the
    # cause -- such names open fine; it is the concurrent-access race that fails.
    tmp = path.with_name(path.name + ".tmp")
    last_err: OSError | None = None
    for attempt in range(5):
        try:
            with tmp.open("w") as f:
                f.write(payload)
            os.replace(tmp, path)
            return
        except OSError as err:
            last_err = err
            try:
                tmp.unlink()
            except OSError:
                pass
            time.sleep(0.2 * (attempt + 1))
    assert last_err is not None
    raise last_err


def benchmark_runtimes(
    spec: CRNSpec,
    pop_sizes: Iterable[int],
    data_dir: str | Path = "data",
    backends: tuple[str, ...] = ("batss", "rebop"),
    seed: int = 1,
    overwrite: bool = False,
    heuristic: bool = False,
    show_progress: bool = True,
) -> None:
    """Time one run to ``spec.benchmark_end_time`` for each (backend, n) and cache to JSON.

    Already-cached (backend, n) pairs are skipped unless ``overwrite=True``.
    The first timed run for each backend is preceded by a warm-up run whose time
    is discarded (rebop's first call in a process is notably slower than
    subsequent ones).

    ``show_progress`` (default True) toggles batss's live progress bar during each
    run (the ``timer`` snapshot in :meth:`Simulation.run`). It has no effect on the
    rebop backend, which has no such bar. The bar's display overhead is included in
    the timed measurement, so set it False for the least-perturbed timings.
    """
    end_time = spec.benchmark_end_time
    data_dir = Path(data_dir)
    # Materialize pop_sizes (it is iterated once per backend, plus once more here) and pad every
    # "n=..." label to a common width so the reported times line up in a column.
    pop_sizes = list(pop_sizes)
    label_width = max((len(_format_pop_size(n)) for n in pop_sizes), default=0)
    for backend in backends:
        path = _runtime_path(data_dir, spec, backend, end_time, proxy=heuristic and backend == "batss")
        data = {} if overwrite else _load_runtime_json(path)
        print(f"benchmarking {spec.name} on {backend} -> {path}", flush=True)

        warmed_up = False
        for n in pop_sizes:
            label = f"n={_format_pop_size(n):<{label_width}}"
            if n in data:
                print(f"  {label}: cached {data[n]:.4g}s, skipping", flush=True)
                continue

            if backend == "batss":
                sim = _batss_sim(spec, n, seed)
                sim.simulator.heuristic = 1 if heuristic else 0

                def run() -> None:
                    # A progress bar can only advance when the Python run loop regains control,
                    # which happens once per stopping_interval. With the whole run in a single
                    # simulator.run() call the bar would sit at 0% until it finished, so when
                    # showing progress we split the run into ~100 checkpoints. This records no
                    # extra history (history_interval is still the full end_time); it only yields
                    # to refresh the bar, at a small timing cost (hence show_progress=False for
                    # the cleanest timings).
                    stopping = end_time / 100 if show_progress else end_time
                    sim.run(end_time, end_time, stopping_interval=stopping, timer=show_progress)
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
            print(f"  {label}: {elapsed:.4g}s", flush=True)
            data[n] = elapsed
            _save_runtime_json(path, data)


def plot_runtimes(
    spec: CRNSpec,
    data_dir: str | Path = "data",
    backends: tuple[str, ...] = ("batss", "rebop"),
    pop_sizes: Iterable[int] | None = None,
    figsize: tuple[float, float] = (5, 4),
    ax: Axes | None = None,
    heuristic: bool = False,
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
        proxy = heuristic and backend == "batss"
        path = _runtime_path(data_dir, spec, backend, end_time, proxy=proxy)
        data = _load_runtime_json(path)
        if not data:
            print(f"no runtime data for {backend} at {path}")
            continue
        ns = sorted(n for n in data if wanted is None or n in wanted)
        ts = [data[n] for n in ns]
        label = f"{backend} (proxy)" if proxy else backend
        ax.loglog(ns, ts, label=label, marker=markers.get(backend, "o"))

    ax.set_xlabel("initial molecular count")
    ax.set_ylabel("run time (s)")
    ax.set_title(f"{spec.name}: run time to simulate t={end_time}")
    ax.legend(loc="upper left")

    out_path = data_dir / f"{spec.name}_runtime_t{end_time}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    return ax


# --- plot 2: trajectory + non-passive-reaction fraction -----------------------


def plot_trajectory(
    spec: CRNSpec,
    n: int,
    end_time: float,
    backend: str = "batss",
    data_dir: str | Path = "data",
    seed: int = 1,
    num_samples: int = 1000,
    species: list[str] | None = None,
    show_non_passive: bool = False,
    figsize: tuple[float, float] = (8, 4),
    ax: Axes | None = None,
    loc: str = "best",
    heuristic: bool = False,
) -> Axes:
    """Simulate ``spec`` with ``backend`` at population size ``n`` and plot counts vs time.

    ``backend`` is either ``"batss"`` (the default) or ``"rebop"``.

    The count y-axis starts at 0 (so the x-axis sits at y=0). When
    ``show_non_passive=True`` and ``backend == "batss"``, the fraction of
    non-passive (real) reactions is drawn as a dashed line on a second y-axis
    (range [0, 1]). This also forces the simulator into pure batching mode (it
    never switches to Gillespie): the non-passive fraction is a batch-algorithm
    quantity -- Gillespie simulates only real reactions, so during a Gillespie
    phase the value would reflect a frozen filler count K rather than active
    batching. rebop has no notion of passive reactions, so that line is omitted
    (with a printed note) for that backend.

    ``species`` is a list of species-name strings to plot; defaults to the
    species named in ``spec.rxns``.

    The figure is saved to
    ``<data_dir>/<spec.name>_trajectory_<backend>_n<n>_t<end_time>.pdf``.
    """
    if backend not in ("batss", "rebop"):
        raise ValueError(f"unknown backend: {backend!r}")

    # write n as $10^e$ if n is a power of 10, otherwise with commas
    exp = _pow10_exponent(n)
    n_str = f"$10^{{{exp}}}$" if exp is not None else f"{n:,}"

    species_to_plot = species if species is not None else _spec_species_names(spec)

    print(f"running {backend} for {spec.name} at n={n_str}")
    sim: Simulation | None = None
    if backend == "batss":
        sim = _batss_sim(spec, n, seed)
        if show_non_passive:
            # Force pure batching mode so the non-passive fraction is measured where it is meaningful.
            # The proxy heuristic with threshold 0 never switches to Gillespie: it would require a
            # batch's expected non-passive reaction count to fall below 0, and it is always >= 0.
            sim.simulator.heuristic = 1
            sim.simulator.proxy_threshold = 0.0
        else:
            sim.simulator.heuristic = 1 if heuristic else 0  # must be set BEFORE run(), not after
        sim.run(end_time, end_time / num_samples)
        times = sim.history.index
        counts = {sp: sim.history[sp] for sp in species_to_plot}
    else:
        crn, inits = _rebop_crn(spec, n)
        ds = crn.run(inits, end_time, num_samples, rng=seed)
        times = ds["time"]
        counts = {sp: ds[sp] for sp in species_to_plot}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for sp in species_to_plot:
        ax.plot(times, counts[sp], label=sp)
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()

    if show_non_passive and backend == "rebop":
        print(
            "note: the non-passive reaction fraction is only available for "
            "backend='batss' (rebop has no passive reactions); skipping it."
        )
    if show_non_passive and sim is not None:
        # The non-passive reaction fraction at each recorded time, measured directly from that
        # snapshot's configuration (parallel to sim.history.index).
        ax2 = ax.twinx()
        ax2.plot(
            sim.history.index,
            sim.non_passive_fractions,
            label="non-passive",
            linestyle="--",
            color="#d62728",
        )
        ax2.set_ylabel("fraction of non-passive reactions")
        ax2.set_ylim(0.0, 1.0)
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    ax.legend(handles, labels, loc=loc)
    ax.set_title(f"{spec.name}: {backend}, n={n_str}")

    out_path = Path(data_dir) / f"{spec.name}_trajectory_{backend}_n{n}_t{end_time}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    return ax
