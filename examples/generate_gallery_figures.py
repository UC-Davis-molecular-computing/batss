"""Regenerate the per-CRN gallery figures for the paper (\\Cref{fig:crn_gallery}).

For each of the five CRNs it writes two PDFs, sized and titled for the paper (the CRN
name is typeset in LaTeX above each row, so titles are omitted here rather than cropped):

    fig_traj_<slug>_crop.pdf     trajectory (counts + active-reaction fraction), from plot_trajectory
    fig_runtime_<slug>_crop.pdf  batss-vs-rebop runtime scaling, from plot_runtimes

The runtime figures read the cached benchmark JSON in ``DATA_DIR``; regenerate that first with
``benchmark_runtimes`` (see benchmark.ipynb) if the timings changed. ``FONTSIZE`` is tuned for the
small per-CRN panels; bump it if the paper layout shrinks them further.

    python generate_gallery_figures.py [output_dir]

``output_dir`` defaults to the Overleaf figures folder.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import batss as bt
from batss.benchmarking import CRNSpec, plot_runtimes, plot_trajectory

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTDIR = Path(
    r"C:/Dropbox/Apps/Overleaf/Efficient Simulation of Chemical Reaction Networks/figures"
)
FONTSIZE = 20
TRAJECTORY_N = 10**5  # population size for every trajectory panel


def gallery_specs() -> list[tuple[CRNSpec, str, float, str, bool]]:
    """The five gallery CRNs, each with its filename slug, trajectory end time, trajectory legend
    location, and whether its runtime plot uses equal log-decade aspect. Reaction definitions, end
    times, and legend placements match benchmark.ipynb (and the cached JSON filenames).

    equal_aspect=True (taller plot) is used only where batss and rebop scale differently, so the
    slope contrast is worth reading off the axes (Dimerization, Lotka-Volterra: batss ~sqrt(n) vs
    rebop ~n; Rossler: a milder gap). Oregonator and Brusselator use the shorter default aspect
    because there batss tracks rebop at ~slope 1 -- there is no contrast to show."""
    m, d = bt.species("M D")
    dimer = CRNSpec(name="Dimerization",
        rxns=[(m + m >> d).k(1), (d >> m + m).k(1)],
        inits_from_n=lambda n: {m: n}, benchmark_end_time=0.5)

    r, f = bt.species("R F")
    lv = CRNSpec(name="Lotka Volterra",
        rxns=[(r + f >> 2 * f).k(1), (r >> 2 * r).k(1), (f >> None).k(1)],
        inits_from_n=lambda n: {r: n // 2, f: n - n // 2}, benchmark_end_time=1.0)

    x1, x2, x3 = bt.species("X1 X2 X3")
    oreg = CRNSpec(name="Oregonator",
        rxns=[(x2 >> x1).k(0.0871), (x2 + x1 >> None).k(1000), (x1 >> 2 * x1 + x3).k(520),
              (2 * x1 >> None).k(40), (x3 >> x2).k(443.324), (x3 >> None).k(2.676)],
        inits_from_n=lambda n: {x1: int(0.0029 * n), x2: int(0.5358 * n), x3: int(0.0034 * n)},
        benchmark_end_time=1.0)

    y1, y2, y3 = bt.species("X1 X2 X3")
    ross = CRNSpec(name="Rössler-Willamowski",
        rxns=[(y1 >> 2 * y1).k(30), (2 * y1 >> y1).k(0.5), (y2 + y1 >> 2 * y2).k(1),
              (y2 >> None).k(10), (y1 + y3 >> None).k(1), (y3 >> 2 * y3).k(16.5), (2 * y3 >> y3).k(0.5)],
        inits_from_n=lambda n: {y1: n // 3, y2: n // 3, y3: n // 3}, benchmark_end_time=1.0)

    a, b, x, y = bt.species("A B X Y")
    brus = CRNSpec(name="Order-3 Brusselator",
        rxns=[(a >> a + x).k(1), (2 * x + y >> 3 * x).k(1), (b + x >> b + y).k(1), (x >> None).k(1)],
        inits_from_n=lambda n: {a: n, b: 3 * n, x: n, y: n}, benchmark_end_time=40.0)

    return [
        (lv, "lotka_volterra", 20.0, "lower right", True),
        (oreg, "oregonator", 5.0, "center right", False),
        (ross, "rossler", 8.0, "upper right", True),
        (brus, "brusselator", 20.0, "upper right", False),
        (dimer, "dimerization", 2.0, "upper right", True),
    ]


def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for spec, slug, traj_end, loc, equal_aspect in gallery_specs():
        plt.close("all")
        plot_trajectory(spec, n=TRAJECTORY_N, end_time=traj_end, data_dir=DATA_DIR, seed=1,
                        num_samples=500, show_active=True, fontsize=FONTSIZE, title="", loc=loc)
        plt.savefig(outdir / f"fig_traj_{slug}_crop.pdf", bbox_inches="tight")

        plt.close("all")
        # Equal-aspect plots need vertical room; bbox_inches='tight' then crops to the equal-decade
        # box, so a tall canvas keeps the width and lets the height settle where it must.
        _, ax = plt.subplots(figsize=(5, 9) if equal_aspect else (5, 4))
        plot_runtimes(spec, data_dir=DATA_DIR, ax=ax, fontsize=FONTSIZE, title="",
                      equal_aspect=equal_aspect)
        plt.savefig(outdir / f"fig_runtime_{slug}_crop.pdf", bbox_inches="tight")
        print(f"{slug}: wrote fig_traj + fig_runtime")
    plt.close("all")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTDIR)
