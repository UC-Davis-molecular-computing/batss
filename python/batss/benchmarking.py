"""
Utilities for benchmarking and plotting CRNs with batss (and comparing against
rebop's Python bindings).

Define a CRN once as a :class:`CRNSpec`, then hand it to

* :func:`benchmark_runtimes` + :func:`plot_runtimes` to measure how run time
  scales with population size ``n``, batss vs rebop, and
* :func:`plot_trajectory` to plot species counts over time from a single batss
  run, optionally overlaying the fraction of active (real) reactions on a
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
from matplotlib.figure import Figure

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
    progress_bar_above_size: int = 1_000_000,
) -> None:
    """Time one run to ``spec.benchmark_end_time`` for each (backend, n) and cache to JSON.

    Every size in ``pop_sizes`` is always (re)computed -- a specified size is never skipped, even
    if it is already in the JSON. ``overwrite`` decides only what happens to the *other* sizes:

    * ``overwrite=False`` (default): the existing JSON is loaded first, so sizes not in
      ``pop_sizes`` are kept and the freshly computed ``pop_sizes`` are merged in (overwriting any
      old values for those sizes). Use this to refresh specific points while keeping the rest.
    * ``overwrite=True``: the file is started fresh, so it ends up holding only ``pop_sizes`` and
      any previously cached sizes are discarded.

    The first timed run for each backend is preceded by a warm-up run whose time is discarded
    (rebop's first call in a process is notably slower than subsequent ones). The warm-up runs on
    the same simulation object as the timed run, which is then rewound to the initial configuration
    with :meth:`Simulation.reset` so the timed run still simulates ``0..end_time``.

    ``progress_bar_above_size`` (default 1_000_000) shows batss's live progress bar (the ``timer``
    snapshot in :meth:`Simulation.run`) only for ``n >= progress_bar_above_size``. The bar splits a
    run into ~100 checkpoints whose overhead is charged to the timed measurement -- and only to
    batss, since rebop has no such bar -- which noticeably inflates fast small-n timings and biases
    the batss-vs-rebop comparison; above the threshold a run is slow enough that the overhead is
    negligible and the feedback is worth it. Set it very high to never show the bar, or to 0 to
    always show it.
    """
    end_time = spec.benchmark_end_time
    data_dir = Path(data_dir)
    # Materialize pop_sizes (it is iterated once per backend, plus once more here) and pad every
    # "n=..." label to a common width so the reported times line up in a column.
    pop_sizes = list(pop_sizes)
    label_width = max((len(_format_pop_size(n)) for n in pop_sizes), default=0)
    for backend in backends:
        path = _runtime_path(data_dir, spec, backend, end_time, proxy=heuristic and backend == "batss")
        # overwrite=True starts a brand-new file holding only the pop_sizes computed below;
        # overwrite=False loads the existing file so sizes outside pop_sizes are preserved. Either
        # way every size in pop_sizes is (re)computed -- a specified size is never skipped.
        data = {} if overwrite else _load_runtime_json(path)
        print(f"benchmarking {spec.name} on {backend} -> {path}", flush=True)

        warmed_up = False
        for n in pop_sizes:
            label = f"n={_format_pop_size(n):<{label_width}}"

            # Show the progress bar only for large n, where a run takes long enough to want feedback
            # and the bar's overhead is negligible; small-n runs stay bar-free for clean timings.
            show_progress = n >= progress_bar_above_size

            if backend == "batss":
                sim = _batss_sim(spec, n, seed)
                sim.simulator.heuristic_gillespie_switching = 1 if heuristic else 0

                def run() -> None:
                    # A progress bar can only advance when the Python run loop regains control,
                    # which happens once per stopping_interval. With the whole run in a single
                    # simulator.run() call the bar would sit at 0% until it finished, so when
                    # showing progress we split the run into ~100 checkpoints. This records no
                    # extra history (history_interval is still the full end_time); it only yields
                    # to refresh the bar, at a small timing cost (hence the bar is suppressed below
                    # progress_bar_above_size for the cleanest small-n timings).
                    stopping = end_time / 100 if show_progress else end_time
                    sim.run(end_time, end_time, stopping_interval=stopping, timer=show_progress)

                def rewind() -> None:
                    sim.reset()
            elif backend == "rebop":
                crn, inits = _rebop_crn(spec, n)

                def run() -> None:
                    crn.run(inits, end_time, 1, rng=seed)

                def rewind() -> None:
                    pass  # every crn.run starts over from inits; nothing to rewind
            else:
                raise ValueError(f"unknown backend: {backend!r}")

            if not warmed_up:
                # Warm up on the SAME objects the timed run will use, then rewind to the initial
                # configuration at t=0. The warm-up must share the sim with the timed run (a
                # throwaway sim would leave the timed object's own first-run costs unpaid), but
                # Simulation.run's run_until is relative, so without the rewind the timed run
                # would simulate end_time..2*end_time instead of 0..end_time.
                run()
                rewind()
                warmed_up = True

            t0 = time.perf_counter()
            run()
            elapsed = time.perf_counter() - t0
            print(f"  {label}: {elapsed:.4g}s", flush=True)
            data[n] = elapsed
            _save_runtime_json(path, data)


def _grow_figure_to_fit_xticks(ax: Axes, max_scale: float = 2.0, step: float = 0.05) -> None:
    """Grow the figure (both dimensions, so the aspect is preserved) until the x tick labels no
    longer overlap. A wide molecular-count range shows many decade labels that collide at a fixed
    figure size; enlarging keeps every label at full font size rather than dropping or shrinking
    them. With ``bbox_inches='tight'`` on save, only the intrinsic PDF size changes, not the layout.
    """
    fig = ax.get_figure()
    if not isinstance(fig, Figure):  # a SubFigure can't be resized; nothing to do
        return
    w0, h0 = fig.get_size_inches()
    scale = 1.0

    def overlapping() -> bool:
        fig.canvas.draw()
        boxes = [t.get_window_extent() for t in ax.get_xticklabels() if t.get_text()]
        return any(left.x1 > right.x0 for left, right in zip(boxes, boxes[1:]))

    while overlapping() and scale < max_scale:
        scale += step
        fig.set_size_inches(w0 * scale, h0 * scale)


def plot_runtimes(
    spec: CRNSpec,
    data_dir: str | Path = "data",
    backends: tuple[str, ...] = ("batss", "rebop"),
    pop_sizes: Iterable[int] | None = None,
    figsize: tuple[float, float] = (5, 4),
    ax: Axes | None = None,
    heuristic: bool = False,
    fontsize: float = 18,
    title: str | None = None,
    equal_aspect: bool = False,
    xtick_step: int | None = None,
) -> Axes:
    """Load cached runtimes and overlay them on log-log axes.

    If ``pop_sizes`` is given, only those ``n`` values are plotted (useful when
    the JSON cache contains data from earlier runs at larger ``n`` that you
    don't want on this plot). If ``None``, every cached point is plotted.

    ``fontsize`` sets the axis-label/title size (ticks and legend are 2pt
    smaller); the default is sized for inclusion in a paper at roughly a third
    of the text width. ``title`` overrides the default plot title; pass ``""``
    to omit the title entirely.

    ``xtick_step`` is the number of decades between labelled x ticks (1 = every
    decade, the default). Every decade is labelled; if the labels would overlap at
    the current size (a wide n range), the figure is grown in both dimensions
    (keeping its aspect) until they fit, rather than dropping labels or shrinking
    the font. Pass ``xtick_step=2`` to instead thin to every other decade.

    ``equal_aspect`` gives the log-log axes equal decade lengths on both axes, so
    an O(n) runtime (rebop) reads as a slope-1 line and an O(sqrt(n)) runtime
    (batss while it stays in batch mode) as slope-1/2. Pair it with a taller
    ``figsize`` at the same width; ``bbox_inches='tight'`` trims the surplus so
    the saved height is whatever equal decades require.

    The figure is saved to ``<data_dir>/<spec.name>_runtime_t<end_time>.pdf``.
    """
    end_time = spec.benchmark_end_time
    data_dir = Path(data_dir)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    wanted = set(pop_sizes) if pop_sizes is not None else None
    markers = {"batss": "o", "rebop": "^"}
    all_ns: set[int] = set()
    for backend in backends:
        proxy = heuristic and backend == "batss"
        path = _runtime_path(data_dir, spec, backend, end_time, proxy=proxy)
        data = _load_runtime_json(path)
        if not data:
            print(f"no runtime data for {backend} at {path}")
            continue
        ns = sorted(n for n in data if wanted is None or n in wanted)
        ts = [data[n] for n in ns]
        all_ns.update(ns)
        label = f"{backend} (proxy)" if proxy else backend
        ax.loglog(ns, ts, label=label, marker=markers.get(backend, "o"))

    # Label every decade by default; a wide range that would collide is handled by growing the
    # figure below (see _grow_figure_to_fit_xticks), not by dropping labels.
    exps = [e for e in (_pow10_exponent(n) for n in all_ns) if e is not None]
    if xtick_step is None:
        xtick_step = 1
    if exps:
        ax.set_xticks([10.0**e for e in range(min(exps), max(exps) + 1, xtick_step)])

    ax.set_xlabel("initial molecular count", fontsize=fontsize)
    ax.set_ylabel("run time (s)", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)
    if title is None:
        title = f"{spec.name}: run time to simulate t={end_time}"
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.legend(loc="upper left", fontsize=fontsize - 2)

    if equal_aspect:
        # On log-log axes 'equal' makes one decade the same physical length on both axes,
        # so runtime slopes are readable directly. adjustable='box' shrinks the box (not the
        # data limits) to satisfy it; a tall figure keeps the width and grows the height.
        ax.set_aspect("equal", adjustable="box")

    _grow_figure_to_fit_xticks(ax)

    out_path = data_dir / f"{spec.name}_runtime_t{end_time}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    return ax


# --- plot 2: trajectory + active-reaction fraction -----------------------


def plot_trajectory(
    spec: CRNSpec,
    n: int,
    end_time: float,
    backend: str = "batss",
    data_dir: str | Path = "data",
    seed: int = 1,
    num_samples: int = 1000,
    species: list[str] | None = None,
    show_active: bool = False,
    figsize: tuple[float, float] = (8, 4),
    ax: Axes | None = None,
    loc: str = "best",
    heuristic: bool = False,
    fontsize: float = 18,
    title: str | None = None,
) -> Axes:
    """Simulate ``spec`` with ``backend`` at population size ``n`` and plot counts vs time.

    ``backend`` is either ``"batss"`` (the default) or ``"rebop"``.

    The count y-axis starts at 0 (so the x-axis sits at y=0). When
    ``show_active=True`` and ``backend == "batss"``, the fraction of
    active (real) reactions is drawn as a dashed line on a second y-axis
    (range [0, 1]). This also forces the simulator into pure batching mode (it
    never switches to Gillespie): the active fraction is a batch-algorithm
    quantity -- Gillespie simulates only real reactions, so during a Gillespie
    phase the value would reflect a frozen filler count K rather than active
    batching. rebop has no notion of passive reactions, so that line is omitted
    (with a printed note) for that backend.

    ``species`` is a list of species-name strings to plot; defaults to the
    species named in ``spec.rxns``.

    ``fontsize`` sets the axis-label/title size (ticks and legend are 2pt
    smaller); the default is sized for inclusion in a paper at roughly half
    the text width. ``title`` overrides the default plot title; pass ``""``
    to omit the title entirely.

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
        if show_active:
            # Force pure batching mode so the active fraction is measured where it is meaningful.
            # The proxy heuristic with threshold 0 never switches to Gillespie: it would require a
            # batch's expected active reaction count to fall below 0, and it is always >= 0.
            sim.simulator.heuristic_gillespie_switching = 1
            sim.simulator.proxy_threshold = 0.0
        else:
            # must be set BEFORE run(), not after
            sim.simulator.heuristic_gillespie_switching = 1 if heuristic else 0
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
    ax.set_xlabel("time", fontsize=fontsize)
    ax.set_ylabel("count", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)
    ax.yaxis.get_offset_text().set_fontsize(fontsize - 2)
    ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()

    if show_active and backend == "rebop":
        print(
            "note: the active reaction fraction is only available for "
            "backend='batss' (rebop has no passive reactions); skipping it."
        )
    if show_active and sim is not None:
        # The active reaction fraction at each recorded time, measured directly from that
        # snapshot's configuration (parallel to sim.history.index).
        ax2 = ax.twinx()
        ax2.plot(
            sim.history.index,
            sim.active_fractions,
            label="active",
            linestyle="--",
            color="#d62728",
        )
        ax2.set_ylabel("active fraction", fontsize=fontsize)
        ax2.tick_params(axis="y", which="major", labelsize=fontsize - 2)
        ax2.set_ylim(0.0, 1.0)
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    ax.legend(handles, labels, loc=loc, fontsize=fontsize - 2)
    if title is None:
        title = f"{spec.name}: {backend}, n={n_str}"
    if title:
        ax.set_title(title, fontsize=fontsize)

    out_path = Path(data_dir) / f"{spec.name}_trajectory_{backend}_n{n}_t{end_time}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    return ax
