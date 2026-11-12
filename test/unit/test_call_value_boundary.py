"""Keep ordinary handlers on the evaluated-value side of the call boundary."""

import ast
from pathlib import Path
import unittest

from llvmlite import ir

from pythoc import func, i32
from pythoc.callable_lowering import get_compile_time_callable
from pythoc.type_resolver import TypeResolver
from pythoc.valueref import wrap_python_constant, wrap_value


ROOT = Path(__file__).resolve().parents[2] / 'pythoc'


class CallValueBoundaryTest(unittest.TestCase):
    def test_handlers_do_not_read_call_structure(self):
        violations = []
        for path in ROOT.rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for function in ast.walk(tree):
                if not isinstance(function, ast.FunctionDef):
                    continue
                if function.name not in ('handle_call', 'handle_call_method',
                                         'handle_type_call', '_eval_call',
                                         'handle_type_conversion'):
                    continue
                names = {arg.arg for arg in function.args.args
                         if arg.arg in ('node', 'call_node')}
                for node in ast.walk(function):
                    if (isinstance(node, ast.Attribute)
                            and isinstance(node.value, ast.Name)
                            and node.value.id in names
                            and node.attr in ('args', 'keywords', 'func')):
                        violations.append(f'{path.relative_to(ROOT)}:{node.lineno}')
        self.assertEqual(violations, [])

    def test_std_does_not_implement_compiler_handlers(self):
        violations = []
        for path in (ROOT / 'std').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if (isinstance(node, ast.FunctionDef)
                        and node.name.startswith('handle_')):
                    violations.append(f'{path.relative_to(ROOT)}:{node.lineno}')
        self.assertEqual(violations, [])

    def test_type_value_resolution_rejects_syntax(self):
        resolver = TypeResolver()
        self.assertIs(resolver.resolve_type_value(wrap_python_constant(i32)), i32)
        with self.assertRaises(TypeError):
            resolver.resolve_type_value(ast.parse('i32', mode='eval').body)

    def test_loaded_pointer_cannot_reuse_symbol_identity(self):
        def predicate(value):
            return value > 0

        module = ir.Module()
        symbol = ir.Function(module, ir.FunctionType(ir.IntType(32), []), 'pred')
        reference = wrap_value(symbol, kind='pointer', type_hint=func[i32])
        symbol._pc_python_callable = predicate
        self.assertIs(get_compile_time_callable(reference), predicate)
        self.assertIs(get_compile_time_callable(wrap_python_constant(predicate)),
                      predicate)
        loaded = reference.clone(value=ir.Constant(symbol.type, None))
        with self.assertRaisesRegex(TypeError, 'no static identity'):
            get_compile_time_callable(loaded)
        other = ir.Function(module, symbol.function_type, 'other')
        with self.assertRaisesRegex(TypeError, 'no static identity'):
            get_compile_time_callable(reference.clone(value=other))


if __name__ == '__main__':
    unittest.main()
