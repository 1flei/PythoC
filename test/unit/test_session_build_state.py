"""Unit tests for build-layer singletons owned by the active CompileSession."""

import unittest

from pythoc.session import CompileSession


class TestBuildSingletonsOnActiveSession(unittest.TestCase):
    """Accessors delegate to the active session and keep their semantics."""

    def test_output_manager_attached_to_active_session(self):
        from pythoc.build.output_manager import get_output_manager
        om = get_output_manager()
        self.assertIs(CompileSession.current().output_manager, om)
        # Stable identity across calls.
        self.assertIs(get_output_manager(), om)

    def test_dependency_tracker_attached_to_active_session(self):
        from pythoc.build.deps import (
            get_dependency_tracker,
            get_group_level_dependency_tracker,
        )
        tracker = get_dependency_tracker()
        self.assertIs(CompileSession.current().dependency_tracker, tracker)
        self.assertIs(get_dependency_tracker(), tracker)
        # Backward-compatible alias resolves to the same instance.
        self.assertIs(get_group_level_dependency_tracker(), tracker)

    def test_multi_so_executor_attached_to_active_session(self):
        from pythoc.native_executor import get_multi_so_executor
        executor = get_multi_so_executor()
        self.assertIs(CompileSession.current().multi_so_executor, executor)
        self.assertIs(get_multi_so_executor(), executor)

    def test_registry_reference_point(self):
        # Each session owns its registry; the accessor resolves the
        # active session's instance.
        from pythoc.registry import get_unified_registry
        self.assertIs(CompileSession.current().registry, get_unified_registry())

    def test_accessors_follow_activation(self):
        # A newly activated session gets its own lazily created singletons.
        from pythoc.build.output_manager import get_output_manager
        outer_om = get_output_manager()
        session = CompileSession()
        with session:
            inner_om = get_output_manager()
            self.assertIs(session.output_manager, inner_om)
            self.assertIsNot(inner_om, outer_om)
        self.assertIs(get_output_manager(), outer_om)


class TestIndependentSessions(unittest.TestCase):
    """Fresh sessions can hold their own independent build singletons."""

    def test_independent_output_manager_and_tracker(self):
        from pythoc.build.output_manager import OutputManager
        from pythoc.build.deps import DependencyTracker

        s1 = CompileSession()
        s2 = CompileSession()
        self.assertIsNone(s1.output_manager)
        self.assertIsNone(s1.dependency_tracker)

        s1.output_manager = OutputManager()
        s2.output_manager = OutputManager()
        s1.dependency_tracker = DependencyTracker()
        s2.dependency_tracker = DependencyTracker()

        self.assertIsNot(s1.output_manager, s2.output_manager)
        self.assertIsNot(s1.dependency_tracker, s2.dependency_tracker)

        # State is independent between the two instances.
        s1.dependency_tracker.record_group_dependency(
            ('a', None, None, None), ('b', None, None, None), 'function_ref')
        self.assertNotEqual(
            s1.dependency_tracker._group_deps,
            s2.dependency_tracker._group_deps,
        )

        # The active session's accessor is unaffected by other sessions.
        from pythoc.build.output_manager import get_output_manager
        self.assertIsNot(get_output_manager(), s1.output_manager)
        self.assertIs(get_output_manager(), CompileSession.current().output_manager)

    def test_fresh_session_has_own_registry(self):
        from pythoc.registry import get_unified_registry
        fresh = CompileSession()
        self.assertIsNot(fresh.registry, get_unified_registry())
        self.assertIsNot(fresh.registry, CompileSession().registry)


class TestBindingStateSession(unittest.TestCase):
    """The @compile wrapper binds the active session at decoration time."""

    def test_binding_state_session_field(self):
        from pythoc.context import FunctionBindingState
        self.assertIsNone(FunctionBindingState().session)

    def test_compile_wrapper_captures_active_session(self):
        from pythoc import compile, i32

        @compile
        def _binding_session_probe(x: i32) -> i32:
            return x

        self.assertIs(
            _binding_session_probe._binding.session,
            CompileSession.current(),
        )


if __name__ == '__main__':
    unittest.main()
