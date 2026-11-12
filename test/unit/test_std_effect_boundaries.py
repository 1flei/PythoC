"""Keep direct libc allocation imports at the default memory provider leaf."""

import ast
from importlib.util import resolve_name, module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

from pythoc import CompileSession, effect


class TestStdEffectBoundaries(unittest.TestCase):
    def test_result_import_does_not_choose_an_error_policy(self):
        path = Path(__file__).resolve().parents[2] / 'pythoc' / 'std' / 'result.py'
        spec = spec_from_file_location('pythoc.std._result_policy_probe', path)
        module = module_from_spec(spec)
        with CompileSession():
            spec.loader.exec_module(module)
            self.assertIsNone(effect.get_effect_impl('errno'))

    def test_libc_allocators_are_confined_to_default_mem(self):
        std = Path(__file__).resolve().parents[2] / 'pythoc' / 'std'
        allocation_names = {
            'malloc', 'calloc', 'realloc', 'free', 'aligned_alloc',
            'posix_memalign', '*',
        }
        violations = []
        for path in sorted(std.rglob('*.py')):
            if path == std / 'mem.py':
                continue
            package = '.'.join(
                ('pythoc', 'std', *path.relative_to(std).parts[:-1]))
            source = path.read_text(encoding='utf-8')
            for node in ast.walk(ast.parse(source, filename=str(path))):
                forbidden = False
                if isinstance(node, ast.Import):
                    forbidden = any(
                        alias.name == 'pythoc.libc.stdlib'
                        for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = resolve_name(
                        '.' * node.level + (node.module or ''), package)
                    names = {alias.name for alias in node.names}
                    forbidden = (
                        (module == 'pythoc.libc.stdlib'
                         and bool(names & allocation_names))
                        or (module == 'pythoc.libc'
                            and bool(names & {'stdlib', '*'}))
                    )
                if forbidden:
                    violations.append(f'{path.relative_to(std)}:{node.lineno}')
        self.assertEqual(
            violations, [],
            'std consumers must use the memory provider interface')


if __name__ == '__main__':
    unittest.main()
