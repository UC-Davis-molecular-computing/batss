"""Regenerate the animated GIFs in README_files/ that stand in for README.ipynb's slider cells.

    python scripts/make_readme_gifs.py

`scripts/make_readme.py --execute` calls this automatically, so you only need to run it directly when
iterating on the GIFs themselves and you don't want to sit through a full notebook re-execution.

Two cells of README.ipynb produce an ipywidgets slider. A slider cannot survive `nbconvert` (and GitHub
cannot render a live widget inside a notebook either), so `make_readme.py` strips the dead widget repr and
the README embeds one of these GIFs instead, showing what dragging the slider actually looks like.

The GIFs are rendered offscreen with matplotlib's PillowWriter rather than screen-recorded, so regenerating
them is a single command. Note that it re-runs the simulations, and the simulator is not reproducible even
with a fixed seed (the batch/Gillespie switch decides by measured wall-clock time -- see issue #14), so the
GIFs come out different every time. That is why `make_readme.py` only regenerates them under `--execute`:
an unconditional rebuild would rewrite a couple of hundred KB of binary on every conversion.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from batss import Simulation, species  # noqa: E402
from batss.snapshot import StatePlotter  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "README_files"
FPS = 20

r, f = species("R F")
rxns = [r >> 2 * r, r + f >> 2 * f, f >> None]


def bar_slider_gif() -> None:
    """The `widgets.interact(plot_row, ...)` cell: a barplot of one configuration, scrubbed over time."""
    n = 10**7
    sim = Simulation({r: n // 2, f: n - n // 2}, rxns, seed=0)
    sim.run(10.0, 0.01)
    history = sim.history
    frames = range(0, len(history), 10)  # 1001 rows -> ~100 frames

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(list(history.columns), [0] * len(history.columns), color=["#1f77b4", "#ff7f0e"])
    ax.set_ylim(0, sim.n)
    ax.set_xlabel("species")
    ax.set_ylabel("count")

    def draw(row: int):
        for bar, value in zip(bars, history.iloc[row]):
            bar.set_height(value)
        ax.set_title(f"Lotka-Volterra at time {history.index[row]:.2f}")
        return bars

    animation = FuncAnimation(fig, draw, frames=frames, blit=False)
    animation.save(OUT / "lv_bar_slider.gif", writer=PillowWriter(fps=FPS), dpi=80)
    plt.close(fig)


def snapshot_slider_gif() -> None:
    """The `sim.snapshot_slider('time')` cell: batss driving its own StatePlotter through the history."""
    n = 10**5
    sim = Simulation({r: n // 2, f: n - n // 2}, rxns, seed=1)
    sim.add_snapshot(StatePlotter())
    sim.run(20.0, 0.1)

    snapshot = sim.snapshots[0]
    assert snapshot.fig is not None
    frames = range(0, len(sim.configs), 2)  # 201 configs -> ~100 frames

    # set_snapshot_index is exactly what the slider calls, so this animates the real widget behaviour.
    animation = FuncAnimation(snapshot.fig, sim.set_snapshot_index, frames=frames, blit=False)
    animation.save(OUT / "lv_snapshot_slider.gif", writer=PillowWriter(fps=FPS), dpi=80)
    plt.close(snapshot.fig)


def main() -> None:
    """Regenerate every GIF in README_files/. Takes about 12 seconds."""
    print("regenerating README GIFs ...")
    bar_slider_gif()
    snapshot_slider_gif()
    for gif in sorted(OUT.glob("*.gif")):
        print(f"wrote {gif.relative_to(OUT.parent)}  ({gif.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
