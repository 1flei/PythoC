"""Unit tests for pythoc.session -- explicit CompileSession activation."""

import contextvars
import threading
import unittest

from pythoc.session import CompileSession


def _run_in_empty_context(fn):
    """Run fn in a fresh context with no active session.

    Importing pythoc installs a session in this context, so "no active
    session" cases must run in an empty context where the session
    ContextVar is unset.
    """
    return contextvars.Context().run(fn)


class TestCurrentSession(unittest.TestCase):

    def test_current_raises_without_session(self):
        def probe():
            with self.assertRaises(RuntimeError) as cm:
                CompileSession.current()
            self.assertIn('pythoc.init()', str(cm.exception))
            self.assertIn('with CompileSession():', str(cm.exception))
        _run_in_empty_context(probe)

    def test_active_returns_none_without_session(self):
        self.assertIsNone(_run_in_empty_context(CompileSession.active))

    def test_with_block_activates_and_restores(self):
        session = CompileSession()
        with session:
            self.assertIs(CompileSession.current(), session)
            self.assertIs(CompileSession.active(), session)

    def test_nested_sessions_restore_in_order(self):
        outer = CompileSession()
        inner = CompileSession()
        with outer:
            self.assertIs(CompileSession.current(), outer)
            with inner:
                self.assertIs(CompileSession.current(), inner)
            self.assertIs(CompileSession.current(), outer)

    def test_two_sessions_are_isolated(self):
        def probe():
            s1 = CompileSession()
            s2 = CompileSession()
            with s1:
                s1.forward_refs.mark_type_defined('Node', object())
                self.assertTrue(s1.forward_refs.is_type_defined('Node'))
                self.assertFalse(s2.forward_refs.is_type_defined('Node'))
            with s2:
                self.assertFalse(
                    CompileSession.current().forward_refs.is_type_defined('Node'))
        _run_in_empty_context(probe)

    def test_activate_deactivate_roundtrip(self):
        def probe():
            session = CompileSession()
            token = session.activate()
            self.assertIs(CompileSession.current(), session)
            session.deactivate(token)
            self.assertIsNone(CompileSession.active())
        _run_in_empty_context(probe)

    def test_activation_is_context_local(self):
        # A session activated on another thread must not leak here, and
        # vice versa (ContextVar semantics, not threading.local).
        session = CompileSession()
        seen = []

        def worker():
            with session:
                seen.append(CompileSession.current())

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)
        self.assertEqual(seen, [session])
        self.assertIsNot(CompileSession.active(), session)


class TestInit(unittest.TestCase):

    def test_init_returns_active_session(self):
        import pythoc
        # The unit runner installed a session before discovery; init()
        # must return it unchanged (idempotent).
        self.assertIs(pythoc.init(), CompileSession.current())
        self.assertIs(pythoc.init(), pythoc.init())

    def test_init_creates_and_activates_when_absent(self):
        def probe():
            import pythoc
            session = pythoc.init()
            self.assertIsInstance(session, CompileSession)
            self.assertIs(CompileSession.current(), session)
            self.assertIs(pythoc.init(), session)
        _run_in_empty_context(probe)


class TestEffectProxy(unittest.TestCase):

    def test_proxy_forwards_public_api(self):
        def probe():
            from pythoc import effect
            impl = object()
            with CompileSession():
                effect.default(rng=impl)
                self.assertTrue(effect.has_effect('rng'))

                effect.rng = impl
                self.assertIs(effect.get_effect_impl('rng'), impl)
                self.assertTrue(effect.is_direct_assignment('rng'))

                other = object()
                with effect(rng=other, suffix='proxy_smoke'):
                    self.assertIs(effect.get_effect_impl('rng'), impl)
                self.assertIs(effect.get_effect_impl('rng'), impl)
        _run_in_empty_context(probe)

    def test_effect_state_is_per_session(self):
        def probe():
            from pythoc import effect
            impl = object()
            s1 = CompileSession()
            s2 = CompileSession()
            with s1:
                effect.rng = impl
                self.assertIs(s1.effects.get_effect_impl('rng'), impl)
            self.assertIsNot(s1.effects, s2.effects)
            with s2:
                self.assertFalse(effect.has_effect('rng'))
        _run_in_empty_context(probe)

    def test_proxy_raises_without_session(self):
        def probe():
            from pythoc import effect
            with self.assertRaises(RuntimeError) as cm:
                effect.has_effect('rng')
            self.assertIn('pythoc.init()', str(cm.exception))
        _run_in_empty_context(probe)


class TestCompileRequiresSession(unittest.TestCase):

    def test_compile_without_session_raises_actionable_error(self):
        def probe():
            from pythoc import compile, i32
            with self.assertRaises(RuntimeError) as cm:
                @compile
                def _no_session_probe(x: i32) -> i32:
                    return x
            self.assertIn('pythoc.init()', str(cm.exception))
        _run_in_empty_context(probe)

    def test_compile_captures_active_session(self):
        from pythoc import compile, i32
        session = CompileSession()
        with session:
            @compile
            def _session_capture_probe(x: i32) -> i32:
                return x
        self.assertIs(_session_capture_probe._binding.session, session)


class TestImportInstallsSession(unittest.TestCase):
    """import pythoc installs a convenience session: plain scripts need no
    explicit init.  Verified in a subprocess because this process already
    has an active session."""

    def _run_fresh_python(self, code):
        import subprocess
        import sys
        return subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True, timeout=120)

    def test_import_pythoc_provides_active_session(self):
        proc = self._run_fresh_python(
            "import pythoc\n"
            "from pythoc.session import CompileSession\n"
            "assert CompileSession.active() is not None\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_plain_script_compiles_without_init(self):
        # @compile needs inspect.getsource(), so the script must be a real
        # file rather than a -c string.
        import subprocess
        import sys
        import tempfile
        import os
        script = (
            "from pythoc import compile, i32\n"
            "@compile\n"
            "def add(a: i32, b: i32) -> i32:\n"
            "    return a + b\n"
            "assert add(20, 22) == 42\n")
        fd, path = tempfile.mkstemp(suffix='.py', prefix='pc_plain_script_')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(script)
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
