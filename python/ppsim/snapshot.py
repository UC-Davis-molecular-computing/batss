"""
A module for :class:`Snapshot` objects used to visualize the protocol during or after
the simulation has run.


:class:`Snapshot` is a base class for snapshot objects that get are updated by :class:`ppsim.simulation.Simulation`.

:class:`Plotter` is a subclass of :class:`Snapshot` that creates a matplotlib figure and axis.
It also gives the option for a state_map function which maps states to the categories which
will show up in the plot.

:class:`StatePlotter` is a subclass of :class:`Plotter` that creates a barplot of the counts
in categories.

:class:`HistoryPlotter` is a subclass of :class:`Plotter` that creates a lineplot of the counts
in categories over time.
"""

from typing import Callable, Hashable, Any, TYPE_CHECKING, Optional, ForwardRef

if TYPE_CHECKING:
    from ppsim.simulation import Simulation
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm
from dataclasses import dataclass
from abc import ABC


@dataclass
class Snapshot(ABC):
    """Abstract base class for snapshot objects."""

    simulation: Optional["Simulation"]
    """
    The :class:`ppsim.simulation.Simulation` object that initialized and will update the :class:`Snapshot`.
    This attribute gets set when the :class:`ppsim.simulation.Simulation` object calls 
    :meth:`ppsim.simulation.Simulation.add_snapshot`.
    """

    update_time: float
    """
    How many seconds will elapse between calls to update while in the 
    :meth:`ppsim.simulation.Simulation.run` method of :class:`ppsim.simulation.Simulation`.
    """

    time: float
    """
    The time at the current snapshot. Changes when :meth:`Snapshot.update` is called.
    """

    config: np.ndarray | None
    """
    The configuration array at the current snapshot. Changes when
    :meth:`Snapshot.update` is called.
    """

    next_snapshot_time: float
    """
    The time at which the next snapshot will be taken.
    """

    def __init__(self, update_time: float = 0.1) -> None:
        """
        Init constructor for the base class.

        Parameters can be passed in here, and any attributes that can be defined
        without the parent :class:`ppsim.simulation.Simulation` object can be instantiated here, 
        such as :data:`Snapshot.update_time`.
        """
        self.simulation = None
        self.update_time = update_time
        self.time = 0.0
        self.config = None
        self.next_snapshot_time = 0.0

    def initialize(self) -> None:
        """Method which is called once during :meth:`ppsim.simulation.Simulation.add_snapshot`. 
        Should only be called after :meth:`ppsim.simulation.Simulation.add_snapshot` is called.

        Any initialization that requires accessing the data in :data:`Snapshot.simulation`
        should go here.
        """
        if self.simulation is None:
            raise ValueError('self.simulation is None, cannot call self.initialize until using sim.add_snapshot')

    def update(self, index: int | None = None) -> None:
        """Method which is called while :data:`Snapshot.simulation` is running.

        Args:
            index: An optional integer index. If present, the snapshot will use the
                data from configuration :data:`ppsim.simulation.Simulation.configs` ``[index]`` and time
                :data:`ppsim.simulation.Simulation.times` ``[index]``. Otherwise, the snapshot will use the current
                configuration :meth:`ppsim.simulation.Simulation.config_array` and current time.
        """
        if self.simulation is None:
            raise ValueError('self.simulation is None, cannot call self.update until using sim.add_snapshot')
        if index is not None:
            self.time = self.simulation.times[index]
            self.config = self.simulation.configs[index]
        else:
            self.time = self.simulation.time
            self.config = self.simulation.config_array


@dataclass
class TimeUpdate(Snapshot):
    """
    Simple :class:`Snapshot` that prints the current time in the :class:`ppsim.simulation.Simulation`.

    When calling :class:`ppsim.simulation.Simulation.run`, if :data:`ppsim.simulation.Simulation.snapshots` is empty, 
    then this object will get added to provide a basic progress update.
    """

    pbar: tqdm
    """
    The progress bar object that gets created.
    """

    start_time: float
    """
    The starting time of the simulation.
    """

    def __init__(self, time_bound: float | None = None, update_time: float = 0.2) -> None:
        super().__init__(update_time)
        self.pbar = tqdm(total=time_bound, position=0, leave=False, unit=' time simulated')
        self.start_time = 0

    def initialize(self) -> None:
        super().initialize()
        assert self.simulation is not None
        self.start_time = self.simulation.time

    def update(self, index: int | None = None) -> None:
        super().update(index)
        new_n = round(self.time - self.start_time, 3)
        self.pbar.update(new_n - self.pbar.n)


@dataclass
class Plotter(Snapshot):
    """
    Base class for a :class:`Snapshot` which will make a plot.

    Gives the option to map states to categories, for an easy way to visualize
    relevant subsets of the states rather than the whole state set.
    These require an interactive matplotlib backend to work.
    """

    fig: Optional["Figure"]
    """
    The matplotlib figure that is created.
    """

    ax: Optional["Axes"]
    """
    The matplotlib axis object that is created. Modifying properties
    of this object is the most direct way to modify the plot.
    """

    yscale: str
    """
    The scale used for the yaxis, passed into ax.set_yscale.
    """

    state_map: Callable[[Hashable], Any]
    """
    A function mapping states to categories, which acts as a filter
    to view a subset of the states or just one field of the states.
    This function should take a state as input and return a category.
    If not specified in constructor, then the state itself will be used as the category.
    """

    categories: list[Any]
    """
    A list which holds the set ``{state_map(state)}`` for all states
    in :data:`Snapshot.simulation.state_list`.
    """

    _matrix: np.ndarray | None
    """
    A (# states)x(# categories) matrix such that for the configuration
    array (indexed by states), ``matrix * config`` gives an array
    of counts of categories. Used internally to get counts of categories.
    """

    def __init__(self, state_map: Callable[[Hashable], Any] | None = None, update_time: float = 0.5,
                 yscale: str = 'linear') -> None:
        """Initializes the :class:`Plotter`.

        Args:
            state_map: An optional function mapping states to categories.
                If None, then the state itself will be used as the category.
            update_time: How many seconds will elapse between calls to update while
                :class:`ppsim.simulation.Simulation.run` method.
            yscale: The scale used for the yaxis, passed into ax.set_yscale.
                Defaults to 'linear'.
        """
        super().__init__(update_time)
        self._matrix = None
        # identity function for state_map if not specified; i.e., the catagories are the states themselves
        self.state_map = state_map if state_map is not None else lambda x: x
        self.yscale = yscale

    def _update_categories_and_matrix(self) -> None:
        """An internal function called to update :data:`Plotter.categories` and `Plotter._matrix`."""
        self.categories = []
        assert self.simulation is not None

        for state in self.simulation.state_list:
            if self.state_map(state) is not None and self.state_map(state) not in self.categories:
                self.categories.append(self.state_map(state))
        self.categories = natsorted(self.categories, key=lambda x: repr(x))

        categories_dict = {j: i for i, j in enumerate(self.categories)}
        self._matrix = np.zeros((len(self.simulation.state_list), len(self.categories)), dtype=np.int64)
        for i, state in enumerate(self.simulation.state_list):
            m = self.state_map(state)
            if m is not None:
                self._matrix[i, categories_dict[m]] += 1

    def initialize(self) -> None:
        """Initializes the plotter by creating a fig and ax."""
        # Only do matplotlib import when necessary
        super().initialize()
        self.fig, self.ax = plt.subplots()
        self._update_categories_and_matrix()


class StatePlotter(Plotter):
    """:class:`Plotter` which produces a barplot of counts."""

    def initialize(self) -> None:
        """Initializes the barplot.

        If :data:`Plotter.state_map` gets changed, call :meth:`Plotter.initialize` to update the barplot to
            show the new set :data:`Plotter.categories`.
        """
        super().initialize()
        assert self.simulation is not None
        assert self.fig is not None
        import seaborn as sns
        self.ax = sns.barplot(x=[str(c) for c in self.categories], y=np.zeros(len(self.categories)))
        assert self.ax is not None
        # rotate the x-axis labels if any of the label strings have more than 2 characters
        if max([len(str(c)) for c in self.categories]) > 2:
            for tick in self.ax.get_xticklabels():
                tick.set_rotation(90)
        self.ax.set_yscale(self.yscale)
        if self.yscale in ['symlog', 'log']:
            self.ax.set_ylim(0, 2 * self.simulation.simulator.n)
        else:
            self.ax.set_ylim(0, self.simulation.simulator.n)

    def update(self, index: int | None = None) -> None:
        """Update the heights of all bars in the plot."""
        super().update(index)
        assert self.fig is not None
        assert self.ax is not None
        assert self.config is not None
        if self._matrix is not None:
            heights = np.matmul(self.config, self._matrix)
        else:
            heights = self.config
        for i, rect in enumerate(self.ax.patches):
            rect.set_height(heights[i]) # type: ignore

        self.ax.set_title(f'Time {self.time: .3f}')
        self.fig.tight_layout()
        self.fig.canvas.draw()


class HistoryPlotter(Plotter):
    """Plotter which produces a lineplot of counts over time."""

    def update(self, index: int | None = None) -> None:
        """Make a new history plot."""
        super().update(index)
        assert self.simulation is not None
        assert self.fig is not None
        assert self.ax is not None
        self.ax.clear()
        if self._matrix is not None:
            df = pd.DataFrame(data=np.matmul(self.simulation.history.to_numpy(), self._matrix),
                              columns=self.categories,
                              index=self.simulation.history.index)
        else:
            df = self.simulation.history
        df.plot(ax=self.ax)

        self.ax.set_yscale(self.yscale)
        if self.yscale in ['symlog', 'log']:
            self.ax.set_ylim(0, 2 * self.simulation.simulator.n)
        else:
            self.ax.set_ylim(0, 1.1 * self.simulation.simulator.n)

        # rotate the x labels if they are time units
        if self.simulation.time_units:
            for tick in self.ax.get_xticklabels():
                tick.set_rotation(45)
        self.fig.tight_layout()
        self.fig.canvas.draw()
