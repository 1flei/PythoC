"""
Unit tests for per-session forward reference namespaces.
"""

import unittest

from pythoc.session import CompileSession
from pythoc.forward_ref import (
    clear_forward_ref_state,
    get_pending_callbacks,
    is_type_defined,
    mark_type_defined,
    register_forward_ref_callback,
)


class TestSessionForwardRefIsolation(unittest.TestCase):
    """ForwardRefNamespace state is isolated between CompileSession instances."""

    def test_sessions_have_independent_namespaces(self):
        s1 = CompileSession()
        s2 = CompileSession()

        self.assertIsNot(s1.forward_refs, s2.forward_refs)

        marker = object()
        s1.forward_refs.mark_type_defined("Node", marker)

        self.assertTrue(s1.forward_refs.is_type_defined("Node"))
        self.assertIs(s1.forward_refs.get_defined_type("Node"), marker)
        self.assertFalse(s2.forward_refs.is_type_defined("Node"))
        self.assertIsNone(s2.forward_refs.get_defined_type("Node"))

    def test_callbacks_are_session_local(self):
        s1 = CompileSession()
        s2 = CompileSession()
        fired = []

        s1.forward_refs.register_callback("Node", lambda obj: fired.append(("s1", obj)))
        s2.forward_refs.register_callback("Node", lambda obj: fired.append(("s2", obj)))

        marker = object()
        s1.forward_refs.mark_type_defined("Node", marker)

        self.assertEqual(fired, [("s1", marker)])
        # s2 still waits for its own definition
        self.assertEqual(s2.forward_refs.get_pending_callbacks(), {"Node": 1})

    def test_snapshot_is_a_copy(self):
        s1 = CompileSession()
        s1.forward_refs.mark_type_defined("Node", object())
        snapshot = s1.forward_refs.defined_types_snapshot()
        snapshot["Injected"] = object()
        self.assertFalse(s1.forward_refs.is_type_defined("Injected"))


class TestModuleLevelDelegation(unittest.TestCase):
    """Module-level functions delegate to the active session's namespace."""

    def setUp(self):
        clear_forward_ref_state()

    def tearDown(self):
        clear_forward_ref_state()

    def test_module_functions_use_active_session(self):
        marker = object()
        mark_type_defined("Node", marker)

        namespace = CompileSession.current().forward_refs
        self.assertTrue(namespace.is_type_defined("Node"))
        self.assertIs(namespace.get_defined_type("Node"), marker)
        self.assertTrue(is_type_defined("Node"))

    def test_callback_fires_on_mark(self):
        fired = []
        register_forward_ref_callback("Node", fired.append)
        self.assertEqual(get_pending_callbacks(), {"Node": 1})

        marker = object()
        mark_type_defined("Node", marker)

        self.assertEqual(fired, [marker])
        self.assertEqual(get_pending_callbacks(), {})

    def test_callback_fires_immediately_when_defined(self):
        marker = object()
        mark_type_defined("Node", marker)

        fired = []
        register_forward_ref_callback("Node", fired.append)
        self.assertEqual(fired, [marker])

    def test_callback_exception_does_not_block_others(self):
        fired = []

        def bad_callback(_obj):
            raise RuntimeError("boom")

        register_forward_ref_callback("Node", bad_callback)
        register_forward_ref_callback("Node", fired.append)
        mark_type_defined("Node", object())

        self.assertEqual(len(fired), 1)


if __name__ == "__main__":
    unittest.main()
