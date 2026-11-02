"""Run one syntax-admission case per process and compilation group."""

import ast
import sys

from pythoc import compile, i32, inline
from pythoc.build.output_manager import flush_all_pending_outputs


case = sys.argv[1]

if case == 'assert':
    @compile
    def selected(value: i32) -> i32:
        assert value > 0
        return value
elif case == 'try':
    @compile
    def selected(value: i32) -> i32:
        try:
            return value + 1
        except ValueError:
            return value + 2
elif case == 'raise':
    @compile
    def selected(value: i32) -> i32:
        raise ValueError('unsupported')
elif case == 'import':
    @compile
    def selected(value: i32) -> i32:
        import math
        return value
elif case == 'delete':
    @compile
    def selected(value: i32) -> i32:
        del value
        return 1
elif case == 'named_expression':
    @compile
    def selected(value: i32) -> i32:
        if (assigned := value) > 0:
            return assigned
        return 0
elif case == 'inline':
    @inline
    def inline_assert(value):
        assert value > 0
        return value

    @compile
    def selected(value: i32) -> i32:
        return inline_assert(value)
elif case == 'pass':
    @compile
    def selected(value: i32) -> i32:
        pass
        return value + 1
elif case == 'generated':
    from pythoc import meta

    generated = meta.func(
        name='generated_assert',
        params=[('value', i32)],
        return_type=i32,
        body=[
            ast.Assert(test=ast.Compare(
                left=ast.Name(id='value', ctx=ast.Load()),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=0)],
            )),
            ast.Return(value=ast.Name(id='value', ctx=ast.Load())),
        ],
    )
    selected = meta.compile_generated(generated, user_globals=globals())
else:
    raise ValueError(case)

flush_all_pending_outputs()
if case == 'pass':
    assert selected(3) == 4
