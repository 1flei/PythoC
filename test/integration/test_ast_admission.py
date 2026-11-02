"""Unsupported syntax must fail in source and inline expansion paths."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pythoc


FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'ast_admission_case.py'


class AstAdmissionTest(unittest.TestCase):
    def run_case(self, case):
        with tempfile.TemporaryDirectory(prefix='ast-admission-') as temp:
            env = os.environ.copy()
            compiler_root = str(Path(pythoc.__file__).resolve().parent.parent)
            env['PYTHONPATH'] = os.pathsep.join(
                [compiler_root, env.get('PYTHONPATH', '')]
            )
            env['PC_RAISE_ON_ERROR'] = '1'
            return subprocess.run(
                [sys.executable, str(FIXTURE), case],
                cwd=temp, env=env, capture_output=True, text=True,
                timeout=90,
            )

    def test_unsupported_syntax_is_diagnosed(self):
        cases = {
            'assert': 'Assert',
            'try': 'Try',
            'raise': 'Raise',
            'import': 'Import',
            'delete': 'Delete',
            'named_expression': 'NamedExpr',
            'inline': 'Assert',
            'generated': 'Assert',
        }
        for case, syntax in cases.items():
            with self.subTest(case=case):
                result = self.run_case(case)
                output = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, output)
                self.assertIn(f'Unsupported Python syntax: {syntax}', output)
                self.assertIn('SyntaxError:', output)

    def test_pass_remains_a_valid_noop(self):
        result = self.run_case('pass')
        self.assertEqual(result.returncode, 0,
                         result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
