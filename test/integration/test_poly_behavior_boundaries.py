"""Poly ownership, real compile failures and stable cross-module dispatch."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import pythoc
from pythoc import compile, consume, enum, f64, i32, linear, static
from pythoc.std.poly import Poly
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


FIXTURES = Path(__file__).resolve().parent / 'fixtures'


@compile
class Counts:
    made: static[i32] = 0
    released: static[i32] = 0


@compile
class Owner:
    value: i32
    token: linear


@enum(i32)
class Left:
    Owned: Owner
    Copy: i32


@enum(i32)
class Right:
    Owned: Owner
    Copy: i32


@compile
def acquire(value: i32) -> Owner:
    Counts.made += 1
    item: Owner = (value, linear())
    return item


@compile
def release(item: Owner) -> i32:
    consume(item.token)
    Counts.released += 1
    return item.value


@compile
def owned_owned(a: Owner, b: Owner) -> i32:
    return 1000 + release(a) * 10 + release(b)


@compile
def owned_copy(a: Owner, b: i32) -> i32:
    return 2000 + release(a) * 10 + b


@compile
def copy_owned(a: i32, b: Owner) -> i32:
    return 3000 + a * 10 + release(b)


@compile
def copy_copy(a: i32, b: i32) -> i32:
    return 4000 + a * 10 + b


combine = Poly(owned_owned, owned_copy, copy_owned, copy_copy)


@compile
def compute(a: i32, b: i32) -> i32:
    Counts.made = 0
    Counts.released = 0
    left: Left
    right: Right
    if a != 0:
        left = Left(Left.Owned, acquire(3))
    else:
        left = Left(Left.Copy, 3)
    if b != 0:
        right = Right(Right.Owned, acquire(7))
    else:
        right = Right(Right.Copy, 7)
    result = combine(left, right)
    return result * 100 + Counts.made * 10 + Counts.released


class PolyBehaviorBoundariesTest(DeferredTestCase):
    def run_script(self, script, cwd):
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([
            str(Path(pythoc.__file__).resolve().parent.parent),
            env.get('PYTHONPATH', ''),
        ])
        env['PC_RAISE_ON_ERROR'] = '1'
        env['PC_FAIL_ON_BUILD_TIME_QUEUE'] = '0'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        result = subprocess.run(
            [sys.executable, str(script)], cwd=cwd, env=env,
            capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)

    def test_two_linear_enum_arguments_cover_all_payload_combinations(self):
        for a, b, expected in ((1, 1, 103722), (1, 0, 203711),
                               (0, 1, 303711), (0, 0, 403700)):
            with self.subTest(a=a, b=b):
                self.assertEqual(compute(a, b), expected)

    def test_rejects_poly_missing_overload(self):
        with CompileSession(), self.assertRaisesRegex(
                ValueError, 'missing function overloads'):
            @enum(i32)
            class Input:
                Integer: i32
                Float: f64

            @compile
            def integer(value: i32) -> i32:
                return value

            operation = Poly(integer)

            @compile
            def main() -> i32:
                return operation(Input(Input.Integer, 7))

            flush_all_pending_outputs()

    def test_rejects_poly_inconsistent_returns(self):
        with CompileSession(), self.assertRaisesRegex(
                ValueError, 'inconsistent return types'):
            @enum(i32)
            class Input:
                Integer: i32
                Float: f64

            @compile
            def integer(value: i32) -> i32:
                return value

            @compile
            def floating(value: f64) -> f64:
                return value

            operation = Poly(integer, floating)

            @compile
            def main() -> i32:
                return operation(Input(Input.Integer, 7))

            flush_all_pending_outputs()

    def test_rejects_poly_payloadless_variant(self):
        with CompileSession(), self.assertRaisesRegex(
                ValueError, 'requires a payload'):
            @enum(i32)
            class Input:
                Integer: i32
                Absent: None

            @compile
            def integer(value: i32) -> i32:
                return value

            operation = Poly(integer)

            @compile
            def main() -> i32:
                return operation(Input(Input.Integer, 7))

            flush_all_pending_outputs()

    def test_rejects_poly_linear_use_after_dispatch(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'already consumed'):
            @compile
            class Owner:
                value: i32
                token: linear

            @enum(i32)
            class Input:
                Owned: Owner
                Copy: i32

            @compile
            def owned(item: Owner) -> i32:
                consume(item.token)
                return item.value

            @compile
            def copy(item: i32) -> i32:
                return item

            operation = Poly(owned, copy)

            @compile
            def main() -> i32:
                owner: Owner = (7, linear())
                value = Input(Input.Owned, owner)
                a = operation(value)
                return a + operation(value)

            flush_all_pending_outputs()

    def test_cross_module_same_named_handlers_survive_a_warm_process(self):
        with tempfile.TemporaryDirectory(prefix='pythoc-poly-warm-') as temp:
            root = Path(temp) / 'modules'
            shutil.copytree(FIXTURES / 'poly_cross_module', root)
            self.run_script(root / 'probe.py', root)
            before = {str(path.relative_to(root)): path.stat().st_mtime_ns
                      for path in (root / 'build').rglob('*.o')}
            self.assertTrue(before, 'Cold run produced no object artifacts')
            self.assertTrue(any('poly' in name for name in before), before)
            self.run_script(root / 'probe.py', root)
            after = {str(path.relative_to(root)): path.stat().st_mtime_ns
                     for path in (root / 'build').rglob('*.o')}
            self.assertEqual(
                before, after, 'Warm run recompiled dispatch artifacts')


if __name__ == '__main__':
    unittest.main()
