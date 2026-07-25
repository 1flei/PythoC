"""Unit tests for per-session unified registry isolation.

The mutable registry tables (structs, compilers, source files, link
libraries/objects) are owned by each CompileSession; the builtin entity
table is a process-level frozen structure shared by all sessions.
"""

import unittest

from pythoc.session import CompileSession
from pythoc.registry import (
    StructInfo,
    get_unified_registry,
    get_builtin_entity,
)


def _make_struct(name):
    return StructInfo(name=name, fields=[("x", None)])


class TestRegistrySessionIsolation(unittest.TestCase):
    """Mutable tables are isolated between sessions."""

    def test_structs_isolated_between_sessions(self):
        s1 = CompileSession()
        s2 = CompileSession()

        with s1:
            get_unified_registry().register_struct(_make_struct("OnlyInS1"))
            self.assertTrue(get_unified_registry().has_struct("OnlyInS1"))

        with s2:
            self.assertFalse(get_unified_registry().has_struct("OnlyInS1"))
            self.assertIsNone(get_unified_registry().get_struct("OnlyInS1"))

        # Back in s1 the struct is still there.
        with s1:
            self.assertTrue(get_unified_registry().has_struct("OnlyInS1"))

    def test_same_name_struct_distinct_per_session(self):
        s1 = CompileSession()
        s2 = CompileSession()

        with s1:
            get_unified_registry().register_struct(_make_struct("Point"))
            info1 = get_unified_registry().get_struct("Point")
        with s2:
            get_unified_registry().register_struct(_make_struct("Point"))
            info2 = get_unified_registry().get_struct("Point")

        self.assertIsNot(info1, info2)

    def test_compilers_isolated_between_sessions(self):
        s1 = CompileSession()
        s2 = CompileSession()
        marker = object()

        with s1:
            get_unified_registry().register_compiler("file_a.py", marker)
            self.assertIs(get_unified_registry().get_compiler("file_a.py"), marker)

        with s2:
            self.assertIsNone(get_unified_registry().get_compiler("file_a.py"))

    def test_link_libraries_isolated_between_sessions(self):
        s1 = CompileSession()
        s2 = CompileSession()

        with s1:
            get_unified_registry().add_link_library("m")
            get_unified_registry().add_link_object("/tmp/fake_s1.o")

        with s2:
            self.assertNotIn("m", get_unified_registry().get_link_libraries())
            self.assertNotIn("/tmp/fake_s1.o",
                             get_unified_registry().get_link_objects())


class TestBuiltinEntityTableShared(unittest.TestCase):
    """The frozen builtin entity table is shared by all sessions."""

    def test_builtin_entities_visible_in_any_session(self):
        from pythoc.builtin_entities import i32, ptr

        s1 = CompileSession()
        s2 = CompileSession()
        with s1:
            self.assertIs(get_unified_registry().get_builtin_entity("i32"), i32)
        with s2:
            self.assertIs(get_unified_registry().get_builtin_entity("ptr"), ptr)

    def test_builtin_entity_query_needs_no_session(self):
        from pythoc.builtin_entities import i32

        # Module-level query functions read the frozen table directly and
        # work without an active session (import-time semantics).
        token = _unset_active_session()
        try:
            self.assertIs(get_builtin_entity("i32"), i32)
        finally:
            _restore_active_session(token)


class TestGetUnifiedRegistryRequiresSession(unittest.TestCase):
    """get_unified_registry() raises when no session is active."""

    def test_raises_without_active_session(self):
        token = _unset_active_session()
        try:
            with self.assertRaises(RuntimeError) as ctx:
                get_unified_registry()
            self.assertIn("pythoc.init()", str(ctx.exception))
        finally:
            _restore_active_session(token)

    def test_clear_all_scoped_to_one_session(self):
        s1 = CompileSession()
        s2 = CompileSession()

        with s1:
            get_unified_registry().register_struct(_make_struct("ToClear"))
        with s2:
            get_unified_registry().register_struct(_make_struct("ToKeep"))

        with s1:
            get_unified_registry().clear_all()
            self.assertFalse(get_unified_registry().has_struct("ToClear"))

        with s2:
            self.assertTrue(get_unified_registry().has_struct("ToKeep"))

        # The frozen builtin table survives clear_all.
        from pythoc.builtin_entities import i32
        with s1:
            self.assertIs(get_unified_registry().get_builtin_entity("i32"), i32)


def _unset_active_session():
    """Temporarily clear the active session; returns a token to restore."""
    from pythoc.session import _current_session
    return _current_session.set(None)


def _restore_active_session(token):
    from pythoc.session import _current_session
    _current_session.reset(token)


if __name__ == '__main__':
    unittest.main()
