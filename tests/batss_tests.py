"""Unit tests for the batss package.

Modernized from the old ``ppsim_tests.py`` (the package was renamed ppsim -> batss):

- The import now comes from ``batss`` instead of the dead ``ppsim`` module.
- The ``TestCRN`` cases (which exercise ``reactions_to_dict``) are unchanged; they
  already use the current species/reaction API.
- The old string-based population-protocol ``test_fratricide`` was removed: the
  bare-string state API is gone and the CRN-only simulator no longer supports the
  population-protocol engine or ``enabled_reactions``. It is replaced by
  ``TestCRNSimulation``, which drives the actual ``crn`` simulator end to end,
  including a regression test for the Gillespie-sync hang (see
  ``CHANGES_2026-07-03.md``).
"""

import unittest
from typing import List, Dict, Any

import numpy
from batss import species, reactions_to_dict, Reaction, Simulation


class TestCRN(unittest.TestCase):
    # tests creating protocols using CRN notation

    def setUp(self) -> None:
        self.n = 10
        self.volume = 10
        self.bimol_correction = (self.n - 1) / (2 * self.volume)

    def test_unit_rates_deterministic(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            a + b >> 2 * b,
            b + c >> 2 * c,
            c + a >> a + a,
        ]
        transitions, max_rate = reactions_to_dict(rxns, self.n, self.volume)
        self.assertAlmostEqual(0.45, max_rate)
        self.assertEqual(6, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((b, a), transitions.keys())
        self.assertIn((b, c), transitions.keys())
        self.assertIn((c, b), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        self.assertEqual((b, b), transitions[(a, b)])
        self.assertEqual((b, b), transitions[(b, a)])
        self.assertEqual((c, c), transitions[(b, c)])
        self.assertEqual((c, c), transitions[(c, b)])
        self.assertEqual((a, a), transitions[(c, a)])
        self.assertEqual((a, a), transitions[(a, c)])

    def test_nonunit_rates_deterministic(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            (a + b >> 2 * b).k(2),
            (b + c >> 2 * c).k(3),
            (c + a >> a + a).k(5),
        ]
        transitions, max_rate = reactions_to_dict(rxns, self.n, self.volume)
        self.assertAlmostEqual(2.25, max_rate)
        self.assertEqual(6, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((b, c), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        self.assertIn((b, a), transitions.keys())
        self.assertIn((c, b), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        assertDeepAlmostEqual(self, {(b, b): 2 / 5}, transitions[(a, b)])
        assertDeepAlmostEqual(self, {(b, b): 2 / 5}, transitions[(b, a)])
        assertDeepAlmostEqual(self, {(c, c): 3 / 5}, transitions[(b, c)])
        assertDeepAlmostEqual(self, {(c, c): 3 / 5}, transitions[(c, b)])
        self.assertEqual((a, a), transitions[(c, a)])
        self.assertEqual((a, a), transitions[(a, c)])

    def test_unit_rates_randomized(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            a + b >> 2 * b,
            a + b >> 2 * c,
            a + c >> a + a,
        ]
        transitions, max_rate = reactions_to_dict(rxns, self.n, self.volume)
        self.assertAlmostEqual(0.9, max_rate)
        self.assertEqual(4, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((b, a), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        assertDeepAlmostEqual(self, {(b, b): 1 / 2, (c, c): 1 / 2}, transitions[(a, b)])
        assertDeepAlmostEqual(self, {(b, b): 1 / 2, (c, c): 1 / 2}, transitions[(b, a)])
        assertDeepAlmostEqual(self, {(a, a): 1 / 2}, transitions[(a, c)])
        assertDeepAlmostEqual(self, {(a, a): 1 / 2}, transitions[(c, a)])

    def test_nonunit_rates_randomized(self) -> None:
        a, b, c = species('A B C')
        rxns = [
            (a + b >> 2 * b).k(2),
            (a + b >> 2 * c).k(3),
            (a + c >> a + a).k(4),
        ]
        transitions, max_rate = reactions_to_dict(rxns, self.n, self.volume)
        self.assertAlmostEqual(2.25, max_rate)
        self.assertEqual(4, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((b, a), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        assertDeepAlmostEqual(self, {(b, b): 2 / 5, (c, c): 3 / 5}, transitions[(a, b)])
        assertDeepAlmostEqual(self, {(b, b): 2 / 5, (c, c): 3 / 5}, transitions[(b, a)])
        assertDeepAlmostEqual(self, {(a, a): 4 / 5}, transitions[(a, c)])
        assertDeepAlmostEqual(self, {(a, a): 4 / 5}, transitions[(c, a)])

    def test_unimolecular(self) -> None:
        a, b, c, d = species('A B C D')
        rxns = [
            (a + d >> 2 * b).k(3),
            (a + c >> 2 * c).k(2),
            (b >> d).k(2),
        ]
        transitions, max_rate = reactions_to_dict(rxns, self.n, self.volume)
        self.assertAlmostEqual(2.0, max_rate)
        self.assertEqual(8, len(transitions))
        self.assertIn((a, d), transitions.keys())
        self.assertIn((d, a), transitions.keys())
        self.assertIn((a, c), transitions.keys())
        self.assertIn((c, a), transitions.keys())
        self.assertIn((b, a), transitions.keys())
        self.assertIn((b, b), transitions.keys())
        self.assertIn((b, c), transitions.keys())
        self.assertIn((b, d), transitions.keys())
        assertDeepAlmostEqual(self, {(b, b): 3 / 2 * self.bimol_correction}, transitions[(a, d)])
        assertDeepAlmostEqual(self, {(b, b): 3 / 2 * self.bimol_correction}, transitions[(d, a)])
        assertDeepAlmostEqual(self, {(c, c): 2 / 2 * self.bimol_correction}, transitions[(a, c)])
        assertDeepAlmostEqual(self, {(c, c): 2 / 2 * self.bimol_correction}, transitions[(c, a)])
        assertDeepAlmostEqual(self, (d, a), transitions[(b, a)])
        assertDeepAlmostEqual(self, (d, b), transitions[(b, b)])
        assertDeepAlmostEqual(self, (d, c), transitions[(b, c)])
        assertDeepAlmostEqual(self, (d, d), transitions[(b, d)])

    def test_reversible_rxn(self) -> None:
        a, b, c, d = species('A B C D')
        rxn: Reaction = a + b | c + d
        self.assertTrue(rxn.reversible)
        transitions, max_rate = reactions_to_dict([rxn], self.n, self.volume)
        self.assertEqual(4, len(transitions))
        self.assertIn((a, b), transitions.keys())
        self.assertIn((c, d), transitions.keys())
        self.assertEqual((c, d), transitions[(a, b)])
        self.assertEqual((a, b), transitions[(c, d)])


class TestCRNSimulation(unittest.TestCase):
    # end-to-end tests of the crn simulator (batss_rust.BatchSimulator)

    def test_fratricide_converges(self) -> None:
        """The reaction 2 L -> L + F is the CRN form of the fratricide population
        protocol. Starting from 20 L it must converge deterministically to a single
        surviving L and 19 F (each reaction consumes an L, and no reaction is possible
        once only one L remains). This replaces the old string-API population-protocol
        test. Special bookkeeping species (K, W) are filtered out of the config.
        """
        L, F = species('L F')
        sim = Simulation({L: 20}, [2 * L >> L + F], seed=1)
        sim.run()  # run_until=None: run until silent (no reaction possible)
        real_config = {sp: count for sp, count in sim.config_dict.items() if not sp.is_special_specie}
        self.assertEqual({L: 1, F: 19}, real_config)

    def test_gillespie_switch_reaches_target_time(self) -> None:
        """Regression test for a hang in the crn simulator.

        When the simulator switched to Gillespie mode it failed to sync rebop's
        updated species counts back into its internal urn, so propensity (and the
        batch/Gillespie switching heuristic) kept using the stale configuration from
        when Gillespie mode was entered. For a stiff CRN like the Oregonator that
        made the per-call time advance minuscule, so the simulation appeared to hang.

        The Oregonator at n=1000 switches to Gillespie almost immediately. On a
        correct build the raw simulator reaches t_max in well under a second; the 15s
        wallclock cap turns a regression (time crawling forward) into a fast test
        failure instead of an indefinite hang. See CHANGES_2026-07-03.md.
        """
        x1, x2, x3 = species('X1 X2 X3')
        rxns = [
            (x2 >> x1).k(0.0871),
            (x2 + x1 >> None).k(1.6 * 10**9),
            (x1 >> 2 * x1 + x3).k(520),
            (2 * x1 >> None).k(4 * 10**7),
            (x3 >> x2).k(443.324),
            (x3 >> None).k(2.676),
        ]
        n = 1000
        inits = {x1: n // 6, x2: 4 * n // 6, x3: n // 6}
        sim = Simulation(inits, rxns, simulator_method='crn', continuous_time=True, seed=1)
        end_time = 8.0
        sim.simulator.run(end_time, 15.0)  # (t_max, max_wallclock_seconds)
        self.assertGreaterEqual(
            sim.simulator.continuous_time,
            end_time,
            "crn simulator failed to reach t_max within the wallclock cap; "
            "the Gillespie-sync hang may have regressed",
        )

    def test_prospective_switch_replays_exactly(self) -> None:
        """The deterministic prospective score must replay a run that crosses its threshold.

        A -> 2A has prospective K=n, order 1, and generativity 1, so its score is
        sqrt(pi*n/8). Starting from n=100 it crosses threshold 10 near n=255, exercising
        batch -> Gillespie -> batch without using wall-clock measurements.
        """
        a, = species('A')
        rxns = [(a >> 2 * a).k(1)]

        def run_once() -> Simulation:
            sim = Simulation({a: 100}, rxns, simulator_method='crn', continuous_time=True, seed=17)
            self.assertEqual(sim.simulator.heuristic_gillespie_switching, 0)
            sim.simulator.heuristic_gillespie_switching = 2
            sim.simulator.proxy_threshold = 10.0
            self.assertAlmostEqual(
                sim.simulator.prospective_batch_score(),
                numpy.sqrt(numpy.pi * sim.simulator.n / 8),
            )
            sim.run(2.0, history_interval=0.25, timer=False)
            self.assertGreaterEqual(sim.simulator.continuous_time, 2.0)
            return sim

        first = run_once()
        second = run_once()

        self.assertTrue(first.history.equals(second.history))
        numpy.testing.assert_array_equal(first.simulator.config, second.simulator.config)
        for name in (
            'batch_calls',
            'gillespie_calls',
            'mode_switches',
            'batch_continuous_time',
            'gillespie_continuous_time',
        ):
            self.assertEqual(getattr(first.simulator.switch, name), getattr(second.simulator.switch, name))
        self.assertEqual(first.simulator.do_gillespie, second.simulator.do_gillespie)
        self.assertEqual(first.simulator.k_resets, second.simulator.k_resets)
        self.assertGreater(first.simulator.switch.batch_calls, 0)
        self.assertGreater(first.simulator.switch.gillespie_calls, 0)
        self.assertEqual(first.simulator.switch.mode_switches, 2)

    def test_prospective_score_uses_reset_k_during_gillespie(self) -> None:
        """A frozen Gillespie K must not feed back into the prospective batch score."""
        a, = species('A')
        sim = Simulation(
            {a: 100},
            [(a >> 2 * a).k(1)],
            simulator_method='crn',
            continuous_time=True,
            seed=17,
        )
        sim.simulator.heuristic_gillespie_switching = 2
        sim.simulator.proxy_threshold = float('inf')
        sim.simulator.run(1.0, 15.0)

        self.assertTrue(sim.simulator.do_gillespie)
        self.assertGreaterEqual(sim.simulator.continuous_time, 1.0)
        self.assertAlmostEqual(
            sim.simulator.prospective_batch_score(),
            numpy.sqrt(numpy.pi * sim.simulator.n / 8),
        )

    def test_combined_reaction_order_replays_exactly(self) -> None:
        """Combined reactant sets have a canonical order before rebop registration.

        A randomized HashMap iteration order previously swapped these two equally rated
        Gillespie reactions between fresh simulators, producing different fixed-seed results.
        """
        a, b, c, d = species('A B C D')
        rxns = [(a >> b).k(1), (c >> d).k(1)]

        def run_once() -> Simulation:
            sim = Simulation(
                {a: 100, c: 100},
                rxns,
                simulator_method='crn',
                continuous_time=True,
                seed=17,
            )
            sim.simulator.heuristic_gillespie_switching = 2
            sim.simulator.proxy_threshold = float('inf')
            sim.run(1.0, history_interval=0.25, timer=False)
            self.assertGreater(sim.simulator.switch.gillespie_calls, 0)
            return sim

        first = run_once()
        second = run_once()

        self.assertTrue(first.history.equals(second.history))
        numpy.testing.assert_array_equal(first.simulator.config, second.simulator.config)
        self.assertEqual(first.simulator.switch.batch_calls, second.simulator.switch.batch_calls)
        self.assertEqual(first.simulator.switch.gillespie_calls, second.simulator.switch.gillespie_calls)
        self.assertEqual(first.simulator.switch.mode_switches, second.simulator.switch.mode_switches)

    def test_frozen_state_engine_oracle_counts_work(self) -> None:
        """The experimental timing oracle reports exact work without mixing in setup."""
        a, = species('A')

        def make_sim(seed: int) -> Simulation:
            return Simulation(
                {a: 100},
                [(a >> 2 * a).k(1)],
                simulator_method='crn',
                continuous_time=True,
                seed=seed,
                volume=100,
            )

        batch_sim = make_sim(17)
        self.assertEqual(batch_sim.simulator.debug_prospective_n(), 200)
        self.assertEqual(batch_sim.simulator.debug_q(), 3)
        self.assertEqual(batch_sim.simulator.debug_reactant_sets(), 1)
        self.assertEqual(batch_sim.simulator.debug_output_branches(), 1)
        batch = batch_sim.simulator.benchmark_engine_call(False)
        self.assertFalse(batch.gillespie)
        self.assertGreater(batch.total_reactions, 0)
        self.assertLessEqual(batch.active_reactions, batch.total_reactions)
        self.assertGreater(batch.continuous_time_advanced, 0)
        self.assertGreaterEqual(batch.engine_seconds, 0)
        self.assertGreaterEqual(batch.postprocess_seconds, 0)

        gillespie_sim = make_sim(19)
        gillespie = gillespie_sim.simulator.benchmark_engine_call(True, 7)
        self.assertTrue(gillespie.gillespie)
        self.assertEqual(gillespie.total_reactions, 7)
        self.assertEqual(gillespie.active_reactions, 7)
        self.assertGreater(gillespie.continuous_time_advanced, 0)
        self.assertGreaterEqual(gillespie.setup_seconds, 0)
        self.assertGreaterEqual(gillespie.engine_seconds, 0)



# taken from https://stackoverflow.com/questions/23549419/assert-that-two-dictionaries-are-almost-equal
def assertDeepAlmostEqual(test_case: unittest.TestCase, expected: Any, actual: Any,
                          *args: List, **kwargs: Dict) -> None:
    """
    Assert that two complex structures have almost equal contents.

    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).

    Usage:

    .. code-block:: python

        from util import assertDeepAlmostEqual

        assertDeepAlmostEqual(self, expected, actual)

    Args:
        test_case: TestCase object on which we can call all of the basic 'assert' methods.
        expected: expected value
        actual: actual value
        *args: args to pass to TestCase.assertAlmostEqual
        **kwargs: kwargs to pass to TestCase.assertAlmostEqual
    """
    is_root = '__trace' not in kwargs
    trace = kwargs.pop('__trace', 'ROOT')
    try:
        if isinstance(expected, (int, float, numpy.long, complex)):
            test_case.assertAlmostEqual(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, numpy.ndarray)):
            test_case.assertEqual(len(expected), len(actual))
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assertDeepAlmostEqual(test_case, v1, v2,
                                      __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            test_case.assertEqual(set(expected), set(actual))
            for key in expected:
                assertDeepAlmostEqual(test_case, expected[key], actual[key],
                                      __trace=repr(key), *args, **kwargs)
        else:
            test_case.assertEqual(expected, actual)
    except AssertionError as exc:
        exc.__dict__.setdefault('traces', []).append(trace)
        if is_root:
            trace = ' -> '.join(reversed(exc.traces))
            exc = AssertionError("%s\nTRACE: %s" % (str(exc), trace))
        raise exc


class TestCollisionSamplerNonzeroR(unittest.TestCase):
    """Characterization tests for issue #16: ``sample_collision(r, u, ...)`` with ``r > 0``.

    ``r`` is the number of agents already drawn in the current batch. The engine always passes
    ``r = 0`` (``batch_step`` is the only caller); ``r > 0`` is left over from the multibatching
    scheme inherited from ppsim, which batss does not use.

    With ``r > 0`` a collision-free run length of **zero** is a legitimate outcome -- the very next
    o-tuple can collide with the ``r`` already-drawn agents -- but the sampler asserts it can never
    happen, so it panics instead of returning 0. These tests pin that behaviour, including evidence
    that the binary search is computing the *right* answer and only the assertion is wrong.

    **These tests describe a bug.** When issue #16 is fixed they must change: if ``r`` is removed,
    delete this class; if ``r`` is kept, they should assert that 0 is returned rather than that a
    panic is raised.
    """

    N_FOR_TESTS = 200_000

    @staticmethod
    def _uniform_sampler(n: int = N_FOR_TESTS) -> Any:
        """A CRN that is already uniform at (o, g) = (2, 1), so no padding surprises."""
        a, b = species('A B')
        rxns = [
            (2 * a >> 3 * a).k(1.0),
            (a + b >> 2 * a + b).k(1.0),
            (2 * b >> 2 * a + b).k(1.0),
        ]
        sim = Simulation({a: n // 2, b: n - n // 2}, rxns, volume=n)
        return sim.simulator

    def test_sample_collision_panics_for_nonzero_r(self) -> None:
        """r > 0 panics with a message blaming floating point, though nothing here is imprecise."""
        rust = self._uniform_sampler()
        n_padded = int(rust.debug_prospective_n())
        with self.assertRaises(BaseException) as ctx:
            for i in range(1, 101):
                rust.sample_collision(n_padded // 2, i / 101.0, False)
        # pyo3 surfaces a Rust panic as pyo3_runtime.PanicException, which is not importable here.
        self.assertEqual('PanicException', type(ctx.exception).__name__)
        self.assertIn('Binary search should never return t_lo = 0', str(ctx.exception))

    def test_sample_collision_never_panics_for_r_zero(self) -> None:
        """The value the engine actually uses is unaffected: r = 0 is sound."""
        rust = self._uniform_sampler()
        for i in range(1, 1001):
            length = rust.sample_collision(0, i / 1001.0, False)
            self.assertGreaterEqual(length, 1)

    def test_panic_rate_equals_immediate_collision_probability(self) -> None:
        """The search is right; the assertion is wrong.

        A run length of 0 means the next o-tuple collides with one of the r already-drawn agents,
        which happens with probability ``1 - (1 - r/N)**o``. The panic rate matches that closed form
        to three decimal places, so the binary search is landing on the correct answer and the
        assertion is rejecting it. A genuine precision bug would not track this curve.
        """
        rust = self._uniform_sampler()
        n_padded = int(rust.debug_prospective_n())
        order = int(rust.debug_o())
        trials = 2000  # u is swept on a regular grid, so the resolution here is 1/trials
        for fraction in (0.1, 0.25, 0.5, 0.75):
            r = int(n_padded * fraction)
            panics = 0
            for i in range(1, trials + 1):
                try:
                    rust.sample_collision(r, i / (trials + 1.0), False)
                except BaseException:  # noqa: BLE001 - PanicException is not importable
                    panics += 1
            expected = 1.0 - (1.0 - r / n_padded) ** order
            self.assertAlmostEqual(
                panics / trials,
                expected,
                delta=2.0 / trials,
                msg=f'at r/N={fraction}, panic rate {panics / trials} should equal the '
                    f'immediate-collision probability {expected}',
            )


if __name__ == '__main__':
    unittest.main()
