"""Python -> PythoC vectorcall entry, without changing native calls or AOT."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

from pythoc import bool, compile, f64, i32, i64
from pythoc import python_call


@compile
def add(a: i64, b: i64) -> i64:
    return a + b


@compile
def add_default(a: i64, b: i64 = 5) -> i64:
    return a + b


@compile
def answer() -> i64:
    return 42


@compile
def addf(a: f64, b: f64) -> f64:
    return a + b


@compile
def narrow(a: i32, b: i32) -> i32:
    return a + b


@compile
def echo_bool(flag: bool) -> bool:
    return flag


@compile
def concurrent_add(a: i64, b: i64) -> i64:
    return a + b


@compile
def failed_add(a: i64, b: i64) -> i64:
    return a + b


@compile
def reentrant_add(a: i64, b: i64) -> i64:
    return a + b


def _load_callee():
    path = os.path.join(os.path.dirname(__file__), 'python_call_callee.py')
    spec = importlib.util.spec_from_file_location('python_call_callee', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inc


inc = _load_callee()


@compile
def twice(x: i64) -> i64:
    return inc(x) + inc(x)


class TestPythonVectorcall(unittest.TestCase):
    def test_scalar_call_uses_generated_adapter(self):
        self.assertEqual(add(1, 2), 3)
        self.assertTrue(add.is_fast_bound())
        self.assertEqual(add(a=4, b=5), 9)
        self.assertEqual(add(7, b=1), 8)
        self.assertEqual(answer(), 42)
        self.assertTrue(answer.is_fast_bound())
        self.assertEqual(add_default(10), 15)
        self.assertEqual(add_default(10, b=2), 12)
        self.assertAlmostEqual(addf(1.5, 2.25), 3.75)
        self.assertEqual(narrow(20, 22), 42)
        self.assertEqual(echo_bool(True), True)
        self.assertEqual(echo_bool(False), False)
        libraries = {
            item._pythoc_adapter_lib._name
            for item in (add, add_default, answer, addf, narrow, echo_bool)
        }
        self.assertEqual(len(libraries), 1)

    def test_boundary_errors(self):
        with self.assertRaises(OverflowError):
            narrow(2147483648, 0)
        with self.assertRaises(TypeError):
            add(1, 2, 3)
        with self.assertRaises(TypeError):
            add(1, c=2)

    def test_concurrent_first_call_waits(self):
        started = threading.Event()
        proceed = threading.Event()
        original = concurrent_add._pythoc_resolve
        results = []
        errors = []

        def delayed_resolve():
            started.set()
            if not proceed.wait(5):
                raise RuntimeError('test resolve wait timed out')
            original()

        def invoke():
            try:
                results.append(concurrent_add(20, 22))
            except BaseException as error:
                errors.append(error)

        concurrent_add._pythoc_resolve = delayed_resolve
        first = threading.Thread(target=invoke)
        second = threading.Thread(target=invoke)
        first.start()
        self.assertTrue(started.wait(5))
        second.start()
        proceed.set()
        first.join(5)
        second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [42, 42])

    def test_resolve_failure_is_preserved(self):
        error = LookupError('resolve marker')
        with mock.patch.object(
            python_call,
            'resolve_compiled_callable',
            side_effect=error,
        ):
            with self.assertRaises(LookupError) as first:
                failed_add(1, 2)
        with self.assertRaises(LookupError) as second:
            failed_add(1, 2)
        self.assertIs(first.exception, error)
        self.assertIs(second.exception, error)

    def test_reentrant_resolve_fails_without_deadlock(self):
        def recurse(wrapper):
            wrapper(1, 2)

        with mock.patch.object(
            python_call,
            'resolve_compiled_callable',
            side_effect=recurse,
        ):
            with self.assertRaisesRegex(RuntimeError, 're-entrant'):
                reentrant_add(1, 2)

    def test_pythoc_call_stays_native(self):
        self.assertEqual(twice(3), 8)
        self.assertTrue(twice.is_fast_bound())
        obj = os.path.join(
            'build', 'test', 'integration', 'test_python_vectorcall.o'
        )
        relocations = subprocess.check_output(
            ['objdump', '-r', obj],
            text=True,
        )
        symbols = subprocess.check_output(['nm', obj], text=True)
        self.assertIn('inc', relocations)
        self.assertNotIn('PyObject', symbols)
        self.assertNotIn('pythoc_pyadapter_', symbols)

    def test_native_aot_has_no_python_adapter(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        sample = os.path.join(
            os.path.dirname(__file__),
            'python_extension_aot_sample.py',
        )
        env = os.environ.copy()
        env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
        built = subprocess.run(
            [sys.executable, sample],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            built.returncode,
            0,
            built.stdout + '\n' + built.stderr,
        )
        self.assertIn('AOT_OK', built.stdout)

        manifest = os.path.join(
            root, 'build', 'python_ext_check', '_pythoc_manifest.json'
        )
        use_env = dict(env)
        use_env['PYTHOC_NATIVE_MANIFEST'] = manifest
        use_env['PYTHOC_NATIVE_USE'] = '1'
        used = subprocess.run(
            [sys.executable, sample],
            cwd=root,
            env=use_env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            used.returncode,
            0,
            used.stdout + '\n' + used.stderr,
        )

    def test_adapter_artifacts_are_reused_across_processes(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        sample = os.path.join(
            os.path.dirname(__file__),
            'python_adapter_cache_sample.py',
        )
        env = os.environ.copy()
        env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')

        def run_sample():
            result = subprocess.run(
                [sys.executable, sample],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                result.stdout + '\n' + result.stderr,
            )
            line = next(
                item for item in result.stdout.splitlines()
                if item.startswith('CACHE_STATE=')
            )
            return json.loads(line.partition('=')[2])

        first = run_sample()
        time.sleep(0.02)
        second = run_sample()
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
