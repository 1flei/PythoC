"""Automatic Poly calls and explicit preparation with the build queue guard."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pythoc
from pythoc import compile, enum, f64, i32
from pythoc.std.poly import Poly


FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'poly_preparation.py'


@enum(i32)
class Input:
    Integer: i32
    Float: f64


@compile
def integer(value: i32) -> i32:
    return value


@compile
def floating(value: f64) -> i32:
    return i32(value) * 10


dispatch = Poly(integer, floating)


@compile
def automatic_dispatch() -> i32:
    result = dispatch(Input(Input.Integer, 7))
    result += dispatch(Input(Input.Float, 2.0))
    return result


class PolyPreparationTest(unittest.TestCase):
    def run_probe(self, *, guarded=False, prepare=False, aot=False):
        with tempfile.TemporaryDirectory(prefix='poly-preparation-') as temp:
            env = os.environ.copy()
            compiler_root = str(Path(pythoc.__file__).resolve().parent.parent)
            env['PYTHONPATH'] = os.pathsep.join(
                [compiler_root, env.get('PYTHONPATH', '')])
            env['PC_FAIL_ON_BUILD_TIME_QUEUE'] = '1' if guarded else '0'
            env['PC_RAISE_ON_ERROR'] = '1'
            command = [sys.executable, str(FIXTURE)]
            if prepare:
                command.append('--prepare')
            if aot:
                command.append('--aot')
            result = subprocess.run(
                command, cwd=temp, env=env,
                capture_output=True, text=True, timeout=120)
            if aot and result.returncode == 0:
                name = 'native.exe' if os.name == 'nt' else 'native'
                executable = Path(temp) / name
                execution = subprocess.run([str(executable)], env=env,
                                           capture_output=True, text=True,
                                           timeout=30)
                self.assertEqual(execution.returncode, 0,
                                 execution.stdout + execution.stderr)
            return result

    def test_default_call_needs_no_preparation(self):
        self.assertEqual(automatic_dispatch(), 27)

    def test_aot_needs_no_preparation(self):
        result = self.run_probe(aot=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_explicit_preparation_satisfies_build_queue_guard(self):
        unprepared = self.run_probe(guarded=True)
        self.assertNotEqual(unprepared.returncode, 0)
        self.assertIn('queue_compilation called during object build',
                      unprepared.stdout + unprepared.stderr)
        prepared = self.run_probe(guarded=True, prepare=True)
        self.assertEqual(prepared.returncode, 0,
                         prepared.stdout + prepared.stderr)


if __name__ == '__main__':
    unittest.main()
