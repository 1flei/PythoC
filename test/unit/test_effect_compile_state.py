"""Unit tests for explicit effect compile state.

Covers:
- effect usage recording onto ActiveCompileFrame (replaces the old
  thread-local usage tracker),
- the compilation-context view over the current binding state,
- per-thread isolation of the Effect suffix/override stacks.
"""

import builtins
import threading
import unittest

from pythoc.context import ActiveCompileFrame, FunctionBindingState
from pythoc.effect import (
    Effect,
    capture_effect_override_names,
    get_current_compilation_context,
    pop_compilation_context,
    push_compilation_context,
    record_effect_usage,
)


class TestEffectUsageRecording(unittest.TestCase):

    def test_frame_carries_usage_set(self):
        frame = ActiveCompileFrame()
        self.assertEqual(frame.effect_usage, set())

    def test_records_into_compile_frame(self):
        frame = ActiveCompileFrame()
        record_effect_usage('rng', frame)
        record_effect_usage('mem', frame)
        record_effect_usage('rng', frame)
        self.assertEqual(frame.effect_usage, {'rng', 'mem'})

    def test_without_frame_is_dropped(self):
        # No active compilation: nothing to attribute the usage to.
        record_effect_usage('rng')
        record_effect_usage('rng', None)


class TestCompilationContextView(unittest.TestCase):

    def tearDown(self):
        # Never leak a pushed context into other tests on this thread.
        while get_current_compilation_context() is not None:
            pop_compilation_context()

    def test_view_over_binding_state(self):
        st = FunctionBindingState(
            compile_suffix='csuf',
            effect_suffix='esuf',
            group_key=('file.py', None, None, 'esuf'),
            captured_effect_context={'rng': object()},
            effect_override_names={'rng'},
        )
        self.assertIsNone(get_current_compilation_context())
        push_compilation_context(st)
        try:
            ctx = get_current_compilation_context()
            self.assertEqual(ctx['compile_suffix'], 'csuf')
            self.assertEqual(ctx['effect_suffix'], 'esuf')
            self.assertEqual(ctx['group_key'], ('file.py', None, None, 'esuf'))
            self.assertEqual(set(ctx['effect_overrides']), {'rng'})
            self.assertEqual(ctx['effect_override_names'], {'rng'})
        finally:
            pop_compilation_context()
        self.assertIsNone(get_current_compilation_context())

    def test_empty_override_names_normalised(self):
        st = FunctionBindingState(effect_suffix='esuf')
        push_compilation_context(st)
        try:
            ctx = get_current_compilation_context()
            self.assertEqual(ctx['effect_override_names'], set())
        finally:
            pop_compilation_context()


class TestEffectStackThreadIsolation(unittest.TestCase):
    """Two threads inside their own `with effect(...)` blocks must not
    observe each other's suffix or override names."""

    def setUp(self):
        self.effect = Effect()
        self._saved_import = builtins.__import__

    def tearDown(self):
        # Parallel EffectContext enter/exit races on the process-wide
        # __import__ hook installation (out of scope for per-thread
        # stacks); restore whatever was active before the test.
        builtins.__import__ = self._saved_import

    def _run_pair(self, worker):
        entered = threading.Barrier(2)
        leave = threading.Barrier(2)
        observed = [None, None]
        errors = []

        def run(index, *args):
            try:
                worker(index, entered, leave, observed, *args)
            except Exception as exc:
                errors.append(exc)

        t0 = threading.Thread(target=run, args=(0, 'alpha'))
        t1 = threading.Thread(target=run, args=(1, 'beta'))
        t0.start()
        t1.start()
        t0.join(timeout=10)
        t1.join(timeout=10)
        self.assertFalse(t0.is_alive() or t1.is_alive(), "barrier deadlock")
        self.assertFalse(errors)
        return observed

    def test_suffix_stacks_are_isolated_between_threads(self):
        eff = self.effect

        def worker(index, entered, leave, observed, suffix):
            with eff(suffix=suffix):
                entered.wait(timeout=5)
                # Both threads are inside their contexts now; each must
                # still see only its own suffix.
                observed[index] = (
                    eff._get_current_suffix(), list(eff._suffix_stack))
                leave.wait(timeout=5)

        observed = self._run_pair(worker)
        self.assertEqual(observed[0], ('alpha', ['alpha']))
        self.assertEqual(observed[1], ('beta', ['beta']))
        self.assertIsNone(eff._get_current_suffix())

    def test_override_name_stacks_are_isolated_between_threads(self):
        # capture_effect_override_names() reads the active session's
        # Effect, so this test drives contexts on that instance directly.
        # Threads start with an empty context, so each worker explicitly
        # activates the shared session first; the suffix/override stacks
        # stay per-thread on the shared Effect.
        from pythoc.effect import effect as eff
        from pythoc.session import CompileSession
        session = CompileSession.current()

        def worker(index, entered, leave, observed, tag):
            token = session.activate()
            try:
                name = 'eff_' + tag
                with eff(suffix=tag, **{name: object()}):
                    entered.wait(timeout=5)
                    observed[index] = capture_effect_override_names()
                    leave.wait(timeout=5)
            finally:
                session.deactivate(token)

        observed = self._run_pair(worker)
        self.assertEqual(observed[0], {'eff_alpha'})
        self.assertEqual(observed[1], {'eff_beta'})
        self.assertEqual(capture_effect_override_names(), set())


if __name__ == '__main__':
    unittest.main()
