"""
A module for simulating population protocols.

The main class :class:`Simulation` is created with a description of the protocol and the initial condition,
and is responsible for running the simulation.

The general syntax is

.. code-block:: python

    a, b, u = 'A', 'B', 'U'
    approx_majority = {
        (a,b): (u,u),
        (a,u): (a,a),
        (b,u): (b,b),
    }
    n = 10 ** 5
    init_config = {a: 0.51 * n, b: 0.49 * n}
    sim = Simulation(init_config=init_config, rule=approx_majority)
    sim.run()
    sim.history.plot()

More examples given in https://github.com/UC-Davis-molecular-computing/ppsim/tree/main/examples

:func:`time_trials` is a convenience function used for gathering data about the
convergence time of a protocol.
"""

import dataclasses
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Callable, Iterable, Any, cast, TypeAlias, TypeVar
from collections.abc import Hashable
from natsort import natsorted
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm.auto import tqdm

from ppsim.crn import Reaction, reactions_to_dict, CRN, convert_to_uniform, catalyst_specie, waste_specie, extra_species
from ppsim.snapshot import Snapshot, TimeUpdate
from ppsim.ppsim_rust import Simulator, SimulatorSequentialArray, SimulatorMultiBatch, SimulatorCRNMultiBatch

# TODO: these names are not showing up in the mouseover information
State: TypeAlias = Hashable
# the plays nicer with generic collections like dict[StateTypeVar, int] than using State
# "Dictionaries are invariant in their key type" is the jargon.
StateTypeVar = TypeVar("StateTypeVar", bound=State) 
Output: TypeAlias = tuple[State, State] | dict[tuple[State, State], float]
TransitionFunction: TypeAlias = Callable[[State, State], Output]
Rule: TypeAlias = TransitionFunction | dict[tuple[State, State], Output] | Iterable[Reaction]
"""
Type representing transition rules for protocol. Is one of three types: 

- a function that takes two states and returns either a tuple of two states or a dictionary mapping pairs of states to probabilities.
- a dictionary mapping pairs of states to either a tuple of two states or a dictionary mapping pairs of states to probabilities.
- a list of :class:`ppsim.crn.Reaction` objects describing a CRN, which will be passed as a CRN if the simulator method is CRN,
or will be converted into a population protocol if it's using original ppsim-style multibatching.
"""

ConvergenceDetector: TypeAlias = Callable[[dict[State, int]], bool]

# TODO: give other option for when the number of reachable states is large or unbounded
def state_enumeration(init_dist: dict[StateTypeVar, int], rule: Callable[[State, State], Output]) -> set[State]:
    """
    Finds all reachable states by breadth-first search.

    Args:
        init_dist: dictionary mapping states to counts
            (states are any hashable type, commonly NamedTuple or String)
        rule: function mapping a pair of states to either a pair of states
            or to a dictionary mapping pairs of states to probabilities

    Returns:
        a set of all reachable states
    """
    checked_states: set[State] = set()
    unchecked_states: set[State] = set(init_dist.keys())
    while len(unchecked_states) > 0:
        unchecked_state = unchecked_states.pop()
        if unchecked_state not in checked_states:
            checked_states.add(unchecked_state)
        for checked_state in checked_states:
            for new_states in [rule(checked_state, unchecked_state),
                               rule(unchecked_state, checked_state)]:
                assert new_states is not None
                if isinstance(new_states, dict):
                    # if the output is a distribution
                    new_states_list = []
                    for new_state1, new_state2 in new_states.keys():
                        new_states_list.extend([new_state1, new_state2])
                else:
                    new_states_list = list(new_states)

                for new_state in new_states_list:
                    if new_state not in checked_states and new_state not in unchecked_states:
                        unchecked_states.add(new_state)
    return checked_states


@dataclass
class Simulation:
    """Class to simulate a population protocol."""

    state_list: list[State]
    """
    A sorted list of all reachable states.
    """

    state_dict: dict[State, int]
    """
    Maps states to their integer index to be used in array representations.
    """

    simulator: Simulator
    """An internal Simulator that performs the simulation steps.
    
    This is either a SimulatorMultiBatch or SimulatorSequentialArray instance."""

    configs: list[npt.NDArray[np.uint]]
    """
    A list of all configurations that have been recorded during the simulation, as integer arrays.
    """

    time: float
    """The current time."""

    times: list[float]
    """A list of all the corresponding times for configs."""

    discrete_steps_total: list[int]
    """
    Parallel to self.times and self.configs, this is how many total steps were taken up to that time,
    including steps that simulated passive reactions. This is a cumulative list, i.e.,
    self.discrete_steps_total[i] is the total number of steps taken up to time self.times[i].
    """

    discrete_steps_total_last_run: list[int]
    """
    Like :data:`Simulation.discrete_steps_total`, but only since the last call to 
    :meth:`Simulation.simulator.run`.
    """

    discrete_steps_no_nulls: list[int]
    """
    Parallel to self.times and self.configs, this is how many total steps were taken up to that time,
    NOT including steps that simulated passive reactions. Cumulative list similarly to self.discrete_steps_total.
    """

    discrete_steps_no_nulls_last_run: list[int]
    """
    Like :data:`Simulation.discrete_steps_total`, but only since the last call to 
    :meth:`Simulation.simulator.run`.
    """

    steps_per_time_unit: float
    """Number of simulated interactions per time unit."""

    time_units: str | None
    """
    The units that time is in. 
    
    For options see https://pandas.pydata.org/docs/reference/api/pandas.to_timedelta.html
    """
    continuous_time: bool
    """
    Whether continuous time is used. 
    The regular discrete time model considers :data:`Simulation.steps_per_time_unit` steps to be 1 unit of time.
    The continuous time model is a Poisson process, 
    with expected :data:`Simulation.steps_per_time_unit` steps per 1 unit of time.
    """

    column_names: pd.MultiIndex | list[str]
    """
    Columns representing all states for pandas dataframe.
    If the State is a tuple, NamedTuple, or dataclass, this will be a
    pandas MultiIndex based on the various fields.
    Otherwise it is list of str(State) for each State.
    """

    snapshots: list[Snapshot]
    """
    A list of :class:`ppsim.snapshot.Snapshot` objects, which get periodically called during the 
    running of the simulation to give live updates.
    """

    rng: np.random.Generator
    """
    A numpy random generator used to sample random variables outside Rust code.
    """

    seed: int | None
    """
    The optional integer seed used for rng and inside Rust code.
    """

    _method: type[SimulatorMultiBatch] | type[SimulatorSequentialArray] | type[SimulatorCRNMultiBatch]

    simulator_method: str

    #TODO: this manual constructor defeats some of the purpose of dataclasses; make more default values and
    # replace this with __post_init__ 
    def __init__(
            self, 
            init_config: dict[StateTypeVar, int], 
            rule: Rule, 
            *,
            simulator_method: str = "MultiBatch",
            transition_order: str = "symmetric", 
            seed: int | None = None,
            volume: float | None = None, 
            continuous_time: bool = False, 
            time_units: str | None = None,
            crn: CRN | None = None,
            **kwargs
    ) -> None:
        """
        Initialize a Simulation.

        Args:
            init_config: The starting configuration, as a
                dictionary mapping states to counts.
            
            rule: A representation of the transition rule. The first two options are
                a dictionary, whose keys are tuples of 2 states and values are their
                outputs, or a function which takes pairs of states as input. For a
                deterministic transition function, the output is a tuple of 2 states.
                For a probabilistic transition function, the output is a dictionary
                mapping tuples of states to probabilities. Inputs that are not present
                in the dictionary, or return None from the function, are interpreted as
                null transitions that return the same pair of states as the output.
                The third option is a list of :class:`ppsim.crn.Reaction` objects describing a CRN,
                which will be parsed into an equivalent population protocol.
            
            simulator_method: Which Simulator method to use, either ``'MultiBatch'``
                or ``'Sequential'`` or ``'Gillespie'`` or ``'CRN'``.
                - ``'MultiBatch'``:
                    :class:`ppsim_rust.SimulatorMultiBatch` does O(sqrt(n)) interaction steps in parallel
                    using batching, and is much faster for large population sizes and
                    relatively small state sets.
                - ``'Gillespie'``:
                    uncondtionally uses the Gillespie algorithm. Still uses the multibatch 
                    simulator, but instructs it to always use the Gillespie algorithm.
                - ``'Sequential'``:
                    :class:`ppsim_rust.SimulatorSequentialArray` represents the population as an array of
                    agents, and simulates each interaction step by choosing a pair of agents
                    to update. Defaults to 'MultiBatch'.
                - ``'CRN'``:
                    :class:`ppsim_rust.SimulatorCRNMultiBatch` does parallel batching for arbitrary
                    CRNs, and should be faster than Gillespie on large population sizes and
                    small species sets.
            
            transition_order: Should the rule be interpreted as being symmetric,
                either ``'asymmetric'``, ``'symmetric'``, or ``'symmetric_enforced'``.
                Defaults to 'symmetric'.

                ``'asymmetric'``:
                    Ordering of the inputs matters, and all inputs not
                    explicitly given as assumed to be null interactions.

                ``'symmetric'``:
                    The input pairs are interpreted as unordered. If rule(a, b)
                    returns None, while rule(b, a) has a non-null output, then the
                    output of rule(a, b) is assumed to be the same as rule(b, a).
                    If rule(a, b) and rule(b, a) are each given, there is no effect.
                    Asymmetric interactions can be explicitly included this way.

                ``'symmetric_enforced'``:
                    The same as symmetric, except that if rule(a, b)
                    and rule(b, a) are non-null and do not give the same set of outputs,
                    a ValueError is raised.
            
            seed: An optional integer used as the seed for all pseudorandom number
                generation. Defaults to None.
            
            volume: If a list of :class:`ppsim.crn.Reaction` objects is given for a CRN, then
                the parameter volume can be passed in here. Defaults to None.
                If None, the volume will be assumed to be the population size n.
            
            continuous_time: Whether continuous time is used. Defaults to False.
                If a CRN as a list of reactions is passed in, this will be set to True.
            
            time_units: An optional string given the units that time is in. Defaults to None.
                This must be a valid string to pass as unit to pandas.to_timedelta.
            
            **kwargs: If `rule` is a function, any extra function parameters are passed in here,
                beyond the first two arguments representing the two agents. For example, if `rule` is
                defined:

                .. code-block:: python

                    def rule(sender: int, receiver: int, threshold: int) -> Tuple[int, int]:
                        if sender + receiver > threshold:
                            return 0, 0
                        else:
                            return sender, receiver+1

                To use a threshold of 20 in each interaction, in the :class:`Simulation` constructor, use

                .. code-block:: python

                    sim = Simulation(init_config, rule, threshold=20)

        """
        self.discrete_steps_total = []
        self.discrete_steps_total_last_run = []
        self.discrete_steps_no_nulls = []
        self.discrete_steps_no_nulls_last_run = []
        self.simulator_method = simulator_method
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n = sum(init_config.values())
        self.steps_per_time_unit = self.n
        self.time_units = time_units
        self.continuous_time = continuous_time
        # if rule is iterable of Reactions from the crn module, then convert to dict
        rule_is_reaction_iterable = True
        try:
            for reaction in rule:  # type: ignore
                if not isinstance(reaction, Reaction):
                    rule_is_reaction_iterable = False
                    break
        except TypeError:
            # might end up here if rule is not even iterable, e.g., is a function
            rule_is_reaction_iterable = False
        if rule_is_reaction_iterable and not simulator_method.lower() == 'crn':
            if volume is None:
                volume = self.n
            rule, rate_max = reactions_to_dict(rule, self.n, volume)  # type: ignore
            transition_order = 'asymmetric'
            self.steps_per_time_unit *= rate_max
            # Default to continuous time for lists of reactions
            self.continuous_time = True
            

        self._rule = rule
        self._rule_kwargs = kwargs

        # Get a list of all reachable states, use the natsort library to put in a nice order.
        if isinstance(self._rule, dict):
            self._rule = cast(dict[tuple[State, State], Output], self._rule)
            # If the rule is a dict, we can loop over the entries to get all states
            states: list[State] = []
            for inpt, output in self._rule.items():
                states.extend(inpt)
                if isinstance(output, dict):
                    for pair in output.keys():
                        states.extend(pair)
                else:
                    states.extend(output)
            state_list = list(set(states))
        elif simulator_method.lower() == 'crn':
            # TODO: a lot of this code is really really written with the assumption of operating
            # on population protocols. It'll take some significant refactoring to get it to not
            # be ugly to also work easily with general CRNs. For now, I'm letting it be ugly.
            # All of the code paths are based on whether or not simulator_method.lower() == 'crn'.
            states = []
            for reaction in rule: # type: ignore
                states += reaction.reactants.species
                states += reaction.products.species
            state_list = list(set(states))
            for init_state in init_config.keys():
                if init_state not in state_list:
                    raise ValueError(f'state "{init_state}" not found in rule reactions: {rule}')

        else:
            # Otherwise, we use breadth-first search to find all reachable states
            state_list = list(state_enumeration(init_config, self.rule))
        # We use the natsorted library to put state_list in a reasonable order
        self.state_list = natsorted(state_list, key=lambda x: repr(x))
        self.state_dict = {state: i for i, state in enumerate(self.state_list)}

        
        if simulator_method.lower() == "crn":
            if volume is None:
                volume = self.n
            # Build a CRN and modify it to make all reactions uniform in order and generativity.
            crn = CRN(list(rule), self.state_list) # type: ignore
            self._crn = convert_to_uniform(crn, volume)
            # TODO we probably want to keep track of these separately, because we don't want to
            # report the counts of K and W to the end user, typically.
            self.state_list = natsorted(state_list, key=lambda x: repr(x)) + extra_species()
            self.state_dict = {state: i for i, state in enumerate(self.state_list)}

        if simulator_method.lower() in ('multibatch', 'gillespie'):
            self._method = SimulatorMultiBatch
        elif simulator_method.lower() == 'sequential':
            self._method = SimulatorSequentialArray
        elif simulator_method.lower() == 'crn':
            self._method = SimulatorCRNMultiBatch
        else:
            raise ValueError('simulator_method must be multibatch, sequential, crn, or gillespie.')
        self._transition_order = transition_order
        self.initialize_simulator(self.array_from_dict(init_config))

        # Check an arbitrary state to see if it has fields.
        # This will be true for either a tuple, NamedTuple, or dataclass.
        state = next(iter(init_config.keys()))
        field_names: None | list[str]
        # Check for dataclass.
        if dataclasses.is_dataclass(state):
            field_names = [field.name for field in dataclasses.fields(state)]  # noqa
            tuples = [dataclasses.astuple(state) for state in self.state_list]  # type: ignore
        else:
            # Check for NamedTuple.
            field_names = getattr(state, '_fields', None)
            tuples = self.state_list
        # Check also for tuple.
        if (field_names is not None and len(field_names) > 1) \
                or (isinstance(state, tuple) and len(state) > 1):
            # Make a MultiIndex only if there are multiple fields
            self.column_names = pd.MultiIndex.from_tuples(tuples, names=field_names) # type: ignore
        else:
            self.column_names = [str(i) for i in self.state_list]
        self.configs = []
        self.times = []
        self.time = 0.0
        self.add_config()
        # private history dataframe is initially empty, updated by the getter of property self.history
        self._history = pd.DataFrame(data=self.configs, index=pd.Index(self.times_in_units(self.times)),
                                     columns=self.column_names)
        self.snapshots = []

    def rule(self, a: State, b: State) -> Output:
        """The rule, as a function of two input states."""
        # If the input rule was a dict
        rule = self._rule
        if isinstance(rule, dict):
            rule = cast(dict[tuple[State, State], Output], rule)
            return rule.get((a, b), (a, b))
            # if (a, b) in self._rule:
            #     return self._rule[(a, b)]
        # If the input rule was a function, with possible kwargs
        elif callable(rule):
            # Make a fresh copy in the case of a dataclass in case the function mutates a, b
            if dataclasses.is_dataclass(a):
                a, b = dataclasses.replace(a), dataclasses.replace(b)  # type: ignore
            # TODO: this doesn't help if the dataclass has more classes as fields
            #   using deepcopy fixes this problem, but is significantly slower
            # a, b = deepcopy(a), deepcopy(b)
            output = rule(a, b, **self._rule_kwargs)
            # If function just mutates a, b but doesn't return, then return new a, b values
            return (a, b) if output is None else output
        else:
            raise TypeError("rule must be either a dict or a callable.")

    def initialize_simulator(self, config: npt.NDArray[np.uint]) -> None:
        """
        Build the data structures necessary to instantiate the simulator class.

        Args:
            config: The config array to instantiate the simulator.
        """
        use_crn_mode = self.simulator_method.lower() == 'crn'
        q = len(self.state_list)
        delta = None
        reactions = None
        k_state = None
        w_state = None
        null_transitions = None
        random_transitions = None
        random_outputs = None # type: ignore
        random_outputs_arr = None
        transition_probabilities = None
        if use_crn_mode:
            reactions = self._crn.exportable_reactions()
            k_state = self.state_list.index(catalyst_specie())
            w_state = self.state_list.index(waste_specie())
        else:
            delta = np.zeros((q, q, 2), dtype=np.uint)
            null_transitions = np.zeros((q, q), dtype=np.bool_)
            random_transitions = np.zeros((q, q, 2), dtype=np.uint)
            random_outputs: list[tuple[State, State]] = []
            transition_probabilities = []
            for i, a in enumerate(self.state_list):
                for j, b in enumerate(self.state_list):
                    output = self.rule(a, b)
                    assert output is not None
                    # when output is a distribution
                    if isinstance(output, dict):
                        s = sum(output.values())
                        assert s <= 1 + 2 ** -20, "The sum of output probabilities must be <= 1."
                        # ensure probabilities sum to 1
                        if 1 - s:
                            if (a, b) in output.keys():
                                output[(a, b)] += 1 - s
                            else:
                                output[(a, b)] = 1 - s
                        if len(output) == 1:
                            # distribution only had 1 output, not actually random
                            output = next(iter(output.keys()))
                        else:
                            # add (number of outputs, index to outputs)
                            random_transitions[i, j] = (len(output), len(random_outputs))
                            for (x, y) in output.keys():
                                random_outputs.append((self.state_dict[x], self.state_dict[y]))
                            transition_probabilities.extend(list(output.values()))
                    if set(output) == {a, b}:
                        null_transitions[i, j] = True
                        delta[i, j] = (i, j)
                    elif isinstance(output, tuple):
                        delta[i, j] = (self.state_dict[output[0]], self.state_dict[output[1]])

            # if random_outputs is empty, this makes a 0D ndarray, but we need a 2D array,
            # with len (shape[0]) = 0, so that the first dimension's length will match with transition_probabilities
            random_outputs_arr = np.array(random_outputs, dtype=np.uint) \
                if len(random_outputs) > 0 else np.empty(shape=(0,2), dtype=np.uint)

            transition_probabilities = np.array(transition_probabilities, dtype=float)

            if self._transition_order.lower() in ['symmetric', 'symmetric_enforced']:
                for i in range(q):
                    for j in range(q):
                        # Set the output for i, j to be equal to j, i if null
                        if null_transitions[i, j]:
                            null_transitions[i, j] = null_transitions[j, i]
                            delta[i, j] = delta[j, i]
                            random_transitions[i, j] = random_transitions[j, i]
                        # If i, j and j, i are both non-null, with symmetric_enforced, check outputs are equal
                        elif self._transition_order.lower() == 'symmetric_enforced' \
                                and not null_transitions[j, i]:
                            if sorted(delta[i, j]) != sorted(delta[j, i]) or \
                                    random_transitions[i, j, 0] != random_transitions[j, i, 0]:
                                a, b = self.state_list[i], self.state_list[j]
                                raise ValueError(f'''Asymmetric interaction:
                                                {a, b} -> {self.rule(a, b)}
                                                {b, a} -> {self.rule(b, a)}''')

        gillespie = self.simulator_method.lower() == 'gillespie' if not use_crn_mode else None
        transition_order = self._transition_order if not use_crn_mode else None

        self.simulator = self._method(
            config, delta, 
            random_transitions, random_outputs_arr, transition_probabilities, 
            transition_order,
            gillespie, self.seed, reactions, k_state, w_state
        )

    def array_from_dict(self, d: dict) -> npt.NDArray[np.uint]:
        """Convert a configuration dictionary to an array.

        Args:
            d: A dictionary mapping states to counts.

        Returns:
            An array giving counts of all states, in the order of
            self.state_list.
        """

        a = np.zeros(len(self.state_list), dtype=np.uint)
        for k in d.keys():
            a[self.state_dict[k]] += d[k]
        return a
    
    def silent(self) -> bool:
        """Check if the configuration is silent.

        Returns:
            True if the configuration is silent, False otherwise.
        """
        return self.simulator.silent

    def run(self, run_until: float | ConvergenceDetector | None = None,
            history_interval: float | Callable[[float], float] = 1.,
            stopping_interval: float = 1., timer: bool = True) -> None:
        """Runs the simulation.

        Can give a fixed amount of time to run the simulation, or a function that checks
        the configuration for convergence.

        Args:
            run_until: The stop condition. To run for a fixed amount of time, give
                a numerical value. To run until a convergence criterion, give a function
                mapping a configuration (as a dictionary mapping states to counts) to a
                boolean. The run will stop when the function returns True.
                Defaults to None. If None, the simulation will run until the configuration
                is silent (all transitions are null). This only works with the multibatch
                simulator method, if another simulator method is given, then using None will
                raise a ValueError.
            
            history_interval: The length to run the simulator before recording data,
                in current time units. Defaults to 1.
                This can either be a float, or a function that takes the current time and
                and returns a float.
            
            stopping_interval: The length to run the simulator before checking for the stop
                condition.
            
            timer: If True, and there are no other snapshot objects, a default :class:`ppsim.snapshot.TimeUpdate`
                snapshot will be created to print updates with the current time.
                Defaults to True.
        """
        if len(self.snapshots) == 0 and timer is True:
            if isinstance(run_until, (float, int)):
                self.add_snapshot(TimeUpdate(time_bound=run_until))
            else:
                self.add_snapshot(TimeUpdate())

        end_time = None
        # stop_condition() returns True when it is time to stop
        if run_until is None:
            if not isinstance(self.simulator, SimulatorMultiBatch):
                raise ValueError('Running until silence only works with multibatch simulator.')

            def stop_condition():
                return self.simulator.silent
        elif isinstance(run_until, (float, int)):
            end_time = self.time + run_until

            def stop_condition():
                return self.time >= end_time
        elif callable(run_until):

            def stop_condition():
                return run_until(self.config_dict)
        else:
            raise TypeError('run_until must be a float, int, function, or None.')

        # Stop if stop_condition is already met
        if self.silent() or stop_condition():
            return

        def get_next_history_time():
            # Get the next time that will be recorded to self.times and self.history
            if callable(history_interval):
                length = history_interval(self.time)
            else:
                length = history_interval
            if length <= 0:
                raise ValueError('history_interval must always be strictly positive.')
            return self.time + length

        if stopping_interval <= 0:
            raise ValueError('stopping_interval must always be strictly positive.')

        next_history_time = get_next_history_time()

        def get_next_time():
            # Get the next time simulator will run until
            t = min(next_history_time, self.time + stopping_interval)
            if end_time is not None:
                t = min(t, end_time)
            return t

        next_time = get_next_time()
        # The next step that the simulator will be run until, which corresponds to parallel time next_time
        next_step = self.time_to_steps(next_time)

        for snapshot in self.snapshots:
            snapshot.next_snapshot_time = perf_counter() + snapshot.update_time

        # add max_wall_clock to be the minimum snapshot update time, to put a time bound on calls to simulator.run
        max_wallclock_time = [min([s.update_time for s in self.snapshots])] if len(self.snapshots) > 0 else []
        while not self.silent() and not stop_condition():
        # while not stop_condition(): # XXX: this while condition was from the Cython implementation; 
                                      # not sure why it didn't cause a bug there
            if self.time >= next_time:
                next_time = get_next_time()
                next_step = self.time_to_steps(next_time)
            current_step = self.simulator.t
            # the next line is overly clever: max_wallclock_time is a list of length 1 or 0;
            # if 0, the default value is used; if 1, the value is used with the * unpacking operator
            if self.simulator_method.lower() == 'crn':
                self.simulator.run(next_time, 10000)
                # print(f"First thing: {self.simulator.continuous_time} and {max_wallclock_time}") # type: ignore
                # print(f"Second thing: {next_time}")
                # print(f"Third thing: {self.simulator.config}")
                assert self.simulator.continuous_time == next_time, "Haven't yet implemented behavior when crn simulation runs past max_wallclock_time" # type: ignore
                self.time = next_time
            else:
                self.simulator.run(next_step, *max_wallclock_time)
                if self.simulator.t == next_step:
                    self.time = next_time
                elif self.simulator.t < next_step:
                    # simulator exited early from hitting max_wallclock_time
                    # add a fraction of the time until next_time equal to the fractional progress made by simulator
                    self.time += (next_time - self.time) * (self.simulator.t - current_step) / (next_step - current_step)
                else:
                    raise RuntimeError(f'The simulator ran to step {self.simulator.t} past the next step {next_step}.')
            if self.time >= next_history_time:
                assert self.time == next_history_time, \
                    f'self.time = {self.time} overshot next_history_time = {next_history_time}'
                self.add_config()
                next_history_time = get_next_history_time()
            for snapshot in self.snapshots:
                if perf_counter() >= snapshot.next_snapshot_time:
                    snapshot.update()
                    snapshot.next_snapshot_time = perf_counter() + snapshot.update_time

        # add the final configuration if it wasn't already recorded
        if self.time > self.times[-1]:
            self.add_config()
        # final updates for all snapshots
        for snapshot in self.snapshots:
            snapshot.update()

        if len(self.snapshots) == 1:
            snapshot = self.snapshots[0]
            if isinstance(snapshot, TimeUpdate):
                snapshot.pbar.close()
                self.snapshots.pop()
            # print()

    @property
    def reactions(self) -> str:
        """
        A string showing all non-null transitions in reaction notation.

        Each reaction is separated by newlines, so that ``print(self.reactions)`` will display all reactions.
        Only works with simulator method multibatch, otherwise will raise a ValueError.
        """
        if type(self.simulator) != SimulatorMultiBatch:
            raise ValueError('reactions must be defined by multibatch simulator.')
        w = max([len(str(state)) for state in self.state_list])
        reactions = [self._reaction_string(r, p, w) for (r, p) in
                     zip(self.simulator.reactions, self.simulator.reaction_probabilities)]
        return '\n'.join(reactions)

    @property
    def enabled_reactions(self) -> str:
        """
        A string showing all non-null transitions that are currently enabled.

        This can only check the current configuration in self.simulator.
        Each reaction is separated by newlines, so that ``print(self.enabled_reactions)``
        will display all enabled reactions.
        """
        if type(self.simulator) != SimulatorMultiBatch:
            raise ValueError('reactions must be defined by multibatch simulator.')
        w = max([len(str(state)) for state in self.state_list])
        self.simulator.get_enabled_reactions()

        reactions = []
        for i in range(self.simulator.num_enabled_reactions):
            r = self.simulator.reactions[self.simulator.enabled_reactions[i]]
            p = self.simulator.reaction_probabilities[self.simulator.enabled_reactions[i]]
            reactions.append(self._reaction_string(r, p, w))
        return '\n'.join(reactions)

    def _reaction_string(self, reaction, p: float = 1, w: int = 1) -> str:
        """A string representation of a reaction."""

        reactants = [self.state_list[i] for i in sorted(reaction[0:2])]
        products = [self.state_list[i] for i in sorted(reaction[2:])]
        s = '{0}, {1}  -->  {2}, {3}'.format(*[str(x).rjust(w) for x in reactants + products])
        if p < 1:
            s += f'      with probability {p}'
        return s

    # TODO: If this changes n, then the timescale must change
    def reset(self, init_config: dict[State, int] | None = None) -> None:
        """Reset the simulation.

        Args:
            init_config: The configuration to reset to. Defaults to None.
                If None, will use the old initial configuration.
        """
        if init_config is None:
            config = self.configs[0]
        else:
            config = np.zeros(len(self.state_list), dtype=np.uint)
            for k in init_config.keys():
                config[self.state_dict[k]] += init_config[k]
        self.configs = [config]
        self.times = [0]
        self.time = 0
        self._history = pd.DataFrame(data=self.configs, index=pd.Index(self.times, name='time'),
                                     columns=self._history.columns)
        self.simulator.reset(config)

    # TODO: If this changes n, then the timescale must change
    def set_config(self, config: dict[State, int] | np.ndarray) -> None:
        """Change the current configuration.

        Args:
            config: The configuration to change to. This can be a dictionary,
                mapping states to counts, or an array giving counts in the order
                of :data:`Simulation.state_list`.
        """
        if type(config) is dict:
            config_array = self.array_from_dict(config)
        else:
            config_array = np.array(config, dtype=np.uint)
        self.simulator.reset(config_array, self.simulator.t)
        self.add_config()

    def time_to_steps(self, time: float) -> int:
        """Convert simulated time into number of simulated interaction steps.

        Args:
            time: The amount of time to convert.
        """
        if self.continuous_time:
            # In continuous time the number of interactions to simulate is a Poisson random variable
            # The last recorded simulated time was self.time, and at this point we had simulated self.simulator.t
            # total interactions. We first compute the expected number of additional steps to simulate.
            expected_steps = (time - self.time) * self.steps_per_time_unit
            return self.simulator.t + self.rng.poisson(expected_steps)
        else:
            # In discrete time we multiply to convert
            return math.floor(time * self.steps_per_time_unit)

    @property
    def config_dict(self) -> dict[State, int]:
        """
        The current configuration, as a dictionary mapping states to counts.
        """
        # return {self.state_list[i]: self.simulator.config[i] for i in np.nonzero(self.simulator.config)[0]}
        return {self.state_list[state_idx]: self.simulator.config[state_idx]
                for state_idx in range(len(self.simulator.config)) if self.simulator.config[state_idx] > 0}

    @property
    def config_array(self) -> np.ndarray:
        """
        The current configuration in the simulator, as an array of counts.

        The array is given in the same order as self.state_list. The index of state s
        is self.state_dict[s].
        """
        return np.asarray(self.simulator.config)

    @property
    def history(self) -> pd.DataFrame:
        """A pandas dataframe containing the history of all recorded configurations."""
        h = len(self._history)
        if h < len(self.configs):
            new_history = pd.DataFrame(data=self.configs[h:], index=pd.Index(self.times_in_units(self.times[h:])),
                                       columns=self._history.columns)
            self._history = pd.concat([self._history, new_history])
            if self.time_units is None:
                if self.continuous_time:
                    self._history.index.name = 'time (continuous units)'
                else:
                    n = "n" if self.n == self.steps_per_time_unit else str(self.steps_per_time_unit)
                    self._history.index.name = f'time ({n} interactions)'
        return self._history

    @property
    def null_probability(self) -> float:
        """The probability the next interaction is null."""
        if type(self.simulator) != SimulatorMultiBatch:
            raise ValueError('null probability requires by multibatch simulator.')
        self.simulator.get_enabled_reactions()
        n = self.simulator.n
        return 1 - self.simulator.get_total_propensity() / (n * (n - 1) / 2)

    def times_in_units(self, times: Iterable[float]) -> Iterable[Any]:
        """If :data:`Simulation.time_units` is defined, convert time list to appropriate units."""
        if self.time_units is not None:
            if not isinstance(times, list):
                times = list(times)
            times = cast(list[float], times)
            # type ignore on next line is because pandas.to_timedelta declares UnitChoices as a literal,
            # but here we are using a string variable to specify instead of a string literal
            return pd.to_timedelta(times, unit=self.time_units)  # type: ignore
        else:
            return times

    def add_config(self) -> None:
        """Appends the current simulator configuration and time."""
        self.configs.append(np.array(self.simulator.config))
        self.times.append(self.time)

        self.discrete_steps_total.append(self.simulator.discrete_steps_total)
        self.discrete_steps_no_nulls.append( self.simulator.discrete_steps_no_nulls)

        self.discrete_steps_total_last_run.append(self.simulator.discrete_steps_total_last_run)
        self.discrete_steps_no_nulls_last_run.append( self.simulator.discrete_steps_no_nulls_last_run)

    def set_snapshot_time(self, time: float) -> None:
        """Updates all snapshots to the nearest recorded configuration to a specified time.

        Args:
            time: The parallel time to update the snapshots to.
        """
        index = int(np.searchsorted(self.times, time))
        for snapshot in self.snapshots:
            snapshot.update(index=index)

    def set_snapshot_index(self, index: int) -> None:
        """Updates all snapshots to the configuration :data:`Simulation.configs` ``[index]``.

        Args:
            index: The index of the configuration.
        """
        for snapshot in self.snapshots:
            snapshot.update(index=index)

    def add_snapshot(self, snap: Snapshot) -> None:
        """Add a new :class:`ppsim.snapshot.Snapshot` to :data:`Simulation.snapshots`.

        Args:
            snap: The :class:`ppsim.snapshot.Snapshot` object to be added.
        """
        snap.simulation = self
        snap.initialize()
        snap.update()
        self.snapshots.append(snap)

    def snapshot_slider(self, var: str = 'index') -> "Any":
        """Returns a slider that updates all :class:`ppsim.snapshot.Snapshot` objects.

        Returns a slider from the ipywidgets library.

        Args:
            var: What variable the slider uses, either ``'index'`` or ``'time'``.
        """
        import ipywidgets as widgets  # type: ignore
        if var.lower() == 'index':
            return widgets.interactive(self.set_snapshot_index,
                                       index=widgets.IntSlider(min=0,
                                                               max=len(self.times) - 1,
                                                               layout=widgets.Layout(width='100%'),
                                                               step=1))
        elif var.lower() == 'time':
            return widgets.interactive(self.set_snapshot_time,
                                       time=widgets.FloatSlider(min=self.times[0],
                                                                max=self.times[-1],
                                                                layout=widgets.Layout(width='100%'),
                                                                step=0.01))
        else:
            raise ValueError("var must be either 'index' or 'time'.")

    def sample_silence_time(self) -> float:
        """Starts a new trial from the initial distribution and return time until silence."""
        if not isinstance(self.simulator, SimulatorMultiBatch):
            raise ValueError('silence time can only be found by multibatch simulator.')
        self.simulator.run_until_silent()
        return self.time

    def sample_future_configuration(self, time: float, num_samples: int = 100) -> pd.DataFrame:
        """Repeatedly samples the configuration at a fixed future time.

        Args:
            time: The amount of time ahead to sample the configuration.
            num_samples: The number of samples to get.

        Returns:
            A dataframe whose rows are the sampled configuration.
        """
        samples = []
        t = self.simulator.t
        for _ in tqdm(range(num_samples)):
            if self.simulator_method.lower() == 'crn':
                self.simulator.reset(np.array(self.configs[-1], dtype=np.uint), t)
                end_time = t + time
                self.simulator.run(end_time)
                samples.append(np.array(self.simulator.config))
            else:
                self.simulator.reset(np.array(self.configs[-1], dtype=np.uint), t)
                end_step = t + self.time_to_steps(time)
                self.simulator.run(end_step)
                samples.append(np.array(self.simulator.config))
        return pd.DataFrame(data=samples, index=pd.Index(range(num_samples), name='trial #'),
                            columns=self._history.columns)

    def __getstate__(self):
        """Returns information to be pickled."""
        # Clear _history such that it can be regenerated by self.history
        d = dict(self.__dict__)
        d['_history'] = pd.DataFrame(data=self.configs[0:1], index=pd.Index(self.times_in_units(self.times[0:1])),
                                     columns=self._history.columns)
        del d['simulator']
        return d

    def __setstate__(self, state) -> None:
        """Instantiates from the pickled state information."""
        self.__dict__ = state
        self.initialize_simulator(self.configs[-1])


InitialCondition: TypeAlias = dict[State, int]


def time_trials(
        rule: Rule, ns: list[int],
        initial_conditions: Callable[[int], InitialCondition] | list[InitialCondition],
        convergence_condition: ConvergenceDetector | None = None,
        convergence_check_interval: float = 0.1,
        num_trials: int = 100,
        max_wallclock_time: float = 60 * 60 * 24,
        **kwargs
) -> pd.DataFrame:
    """Gathers data about the convergence time of a rule.

    Args:
        rule: The rule that is used to generate the :class:`Simulation` object.
        ns: A list of population sizes n to sample from.
            This should be in increasing order.
        initial_conditions: An initial condition is a dict mapping states to counts.
            This can either be a list of initial conditions, or a function mapping
            population size n to an initial condition of n agents.
        convergence_condition: A boolean function that takes a configuration dict
            (mapping states to counts)
            as input and returns True if that configuration has converged.
            Defaults to None. If None, the simulation will run until silent
            (all transitions are null), and the data will be for silence time.
        convergence_check_interval: How often (in parallel time) the simulation will
            run between convergence checks. Defaults to 0.1.
            Smaller values give better resolution, but spend more time checking for
            convergence.
        num_trials: The maximum number of trials that will be done for each
            population size n, if there is sufficient time. Defaults to 100.
            If you want to ensure that you get the full num_trials samples for
            each value n, use a large value for time_bound.
        max_wallclock_time: A bound (in seconds) for how long this function will run.
            Each value n is given a time budget based on the time remaining, and
            will stop before doing num_trials runs when this time budget runs out.
            Defaults to 60 * 60 * 24 (one day).
        **kwargs: Other keyword arguments to pass into :class:`Simulation`.

    Returns:
        A pandas dataframe giving the data from each trial, with two columns
        ``'n''`` and ``'time'``. A good way to visualize this dataframe is using the
        seaborn library, calling ``sns.lineplot(x='n', y='time', data=df)``.
    """
    d: dict[str, list[float]] = {'n': [], 'time': []}
    end_time = perf_counter() + max_wallclock_time
    if callable(initial_conditions):
        initial_conditions = [initial_conditions(n) for n in ns]
    for i in tqdm(range(len(ns))):
        sim = Simulation(initial_conditions[i], rule, **kwargs)
        t = perf_counter()
        time_limit = t + (end_time - t) / (len(ns) - i)
        j = 0
        while j < num_trials and perf_counter() < time_limit:
            j += 1
            sim.reset(initial_conditions[i])
            sim.run(convergence_condition, stopping_interval=convergence_check_interval, timer=False)
            d['n'].append(ns[i])
            d['time'].append(sim.time)

    return pd.DataFrame(data=d)
