"""Poly expands in the caller without creating pending native compilation work."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pythoc
from pythoc import compile, consume, enum, f64, i32, linear, static, void
from pythoc.build.output_manager import get_output_manager, flush_all_pending_outputs
from pythoc.std.poly import Poly


FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'poly_inline.py'


@enum(i32)
class Input:
    Integer: i32
    Float: f64


@compile
class Counts:
    evaluated: static[i32] = 0
    visited: static[i32] = 0


@compile
class Owner:
    value: i32
    token: linear


@compile
def integer(value: i32) -> i32:
    return value


@compile
def floating(value: f64) -> i32:
    return i32(value) * 10


@compile
def make_input() -> Input:
    Counts.evaluated += 1
    return Input(Input.Integer, Counts.evaluated)


@compile
def visit_integer(value: i32) -> void:
    Counts.visited += value


@compile
def visit_floating(value: f64) -> void:
    Counts.visited += i32(value) * 10


@compile
def integer_owner(value: i32) -> Owner:
    result: Owner = (value, linear())
    return result


@compile
def floating_owner(value: f64) -> Owner:
    result: Owner = (i32(value) * 10, linear())
    return result


@compile
def combine_integer(value: i32, owner: Owner) -> i32:
    consume(owner.token)
    return value + owner.value


@compile
def combine_floating(value: f64, owner: Owner) -> i32:
    consume(owner.token)
    return i32(value) * 10 + owner.value


dispatch = Poly(integer, floating)
visit = Poly(visit_integer, visit_floating)
owners = Poly(integer_owner, floating_owner)
combine = Poly(combine_integer, combine_floating)


@compile
def automatic_dispatch() -> i32:
    result = dispatch(Input(Input.Integer, 7))
    result += dispatch(Input(Input.Float, 2.0))
    return result


@compile
def explicit_entry() -> i32:
    return dispatch._call(Input(Input.Integer, 3)) + dispatch._call(i32(5))


@compile
def evaluates_once() -> i32:
    Counts.evaluated = 0
    result = dispatch(make_input())
    result += dispatch._call(make_input())
    return result * 10 + Counts.evaluated


@compile
def void_results() -> i32:
    Counts.visited = 0
    visit(Input(Input.Integer, 3))
    visit._call(Input(Input.Float, 2.0))
    return Counts.visited


@compile
def owned_results(tag: i32) -> i32:
    source: Input
    if tag == 0:
        source = Input(Input.Integer, 3)
    else:
        source = Input(Input.Float, 2.0)
    result = owners(source)
    consume(result.token)
    return result.value


@compile
def mixed_owned_argument(tag: i32) -> i32:
    owner: Owner = (7, linear())
    source: Input
    if tag == 0:
        source = Input(Input.Integer, 3)
    else:
        source = Input(Input.Float, 2.0)
    return combine(source, owner)


class PolyInlineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manager = get_output_manager()
        with patch.object(manager, 'queue_compilation',
                          side_effect=AssertionError('Poly queued native work during expansion')):
            flush_all_pending_outputs()

    def run_probe(self, *, aot=False):
        with tempfile.TemporaryDirectory(prefix='poly-inline-') as temp:
            env = os.environ.copy()
            compiler_root = str(Path(pythoc.__file__).resolve().parent.parent)
            env['PYTHONPATH'] = os.pathsep.join(
                [compiler_root, env.get('PYTHONPATH', '')])
            env['PC_FAIL_ON_BUILD_TIME_QUEUE'] = '1'
            env['PC_RAISE_ON_ERROR'] = '1'
            command = [sys.executable, str(FIXTURE)]
            if aot:
                command.append('--aot')
            result = subprocess.run(command, cwd=temp, env=env,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            if aot:
                executable = Path(temp) / ('native.exe' if os.name == 'nt' else 'native')
                execution = subprocess.run([str(executable)], env=env,
                                           capture_output=True, text=True, timeout=30)
                self.assertEqual(execution.returncode, 0,
                                 execution.stdout + execution.stderr)
            self.assertFalse(any(path.name.startswith('poly.')
                                 for path in Path(temp, 'build').rglob('*.o')))

    def test_no_preparation_is_required(self):
        self.assertEqual(automatic_dispatch(), 27)
        self.assertEqual(explicit_entry(), 8)

    def test_arguments_are_evaluated_once(self):
        self.assertEqual(evaluates_once(), 32)

    def test_void_results(self):
        self.assertEqual(void_results(), 23)

    def test_owned_results(self):
        self.assertEqual(owned_results(0), 3)
        self.assertEqual(owned_results(1), 20)

    def test_non_enum_owner_is_transferred_only_on_the_selected_branch(self):
        self.assertEqual(mixed_owned_argument(0), 10)
        self.assertEqual(mixed_owned_argument(1), 27)

    def test_guarded_development_build(self):
        self.run_probe()

    def test_guarded_aot_build(self):
        self.run_probe(aot=True)


if __name__ == '__main__':
    unittest.main()
