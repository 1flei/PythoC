"""Validate predicates and yield a refinement of the evaluated input values."""

import ast
import inspect

from .base import BuiltinFunction
from .python_type import PythonType
from .._pc_intrinsics import _intrinsic_name
from ..logger import logger
from ..valueref import wrap_value


class refine(BuiltinFunction):
    """refine(value, predicates/tags...) or refine(values..., predicate).

    The call boundary evaluates arguments. The synthetic yield function
    binds those values once; predicate checks and assume share its locals.
    """

    # The generated parameter bindings perform the ownership transfer.
    defer_linear_transfer = True

    @classmethod
    def get_name(cls):
        return 'refine'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        if len(args) < 2:
            logger.error('refine() requires at least 2 arguments',
                         node=node, exc_type=TypeError)

        func_ast = cls._create_refine_inline_ast(args, node)

        def placeholder(*args, **kwargs):
            raise RuntimeError('refine() must be expanded in a for loop')

        placeholder._is_yield_generated = True
        placeholder._pc_compile_level_callable = True
        placeholder._original_ast = func_ast
        result = wrap_value(placeholder, kind='python',
                            type_hint=PythonType(placeholder))
        result._yield_inline_info = {
            'func_obj': None,
            'original_ast': func_ast,
            'call_node': node,
            'call_args': args,
        }
        return result

    @classmethod
    def _create_refine_inline_ast(cls, args, node):
        """Build checks over formal bindings, never over source expressions."""
        if not args[-1].is_python_value():
            logger.error(
                'refine() predicates and tags must be compile-time values',
                node=node, exc_type=TypeError)
        last = args[-1].get_python_value()
        arity = (len(inspect.signature(last).parameters)
                 if callable(last) else 0)
        multi = arity > 1 and len(args) == arity + 1
        constraint_start = arity if multi else 1
        constraints = []
        for argument in args[constraint_start:]:
            if not argument.is_python_value():
                logger.error(
                    'refine() predicates and tags must be compile-time values',
                    node=node, exc_type=TypeError)
            constraints.append(argument.get_python_value())

        if any(not (callable(value) or isinstance(value, str))
               for value in constraints):
            logger.error(
                'refine() constraints must be predicates or string tags',
                node=node, exc_type=TypeError)

        parameters = [f'_refine_arg_{index}' for index in range(len(args))]

        def reference(index):
            return ast.Name(id=parameters[index], ctx=ast.Load())

        checks = [
            ast.Call(
                func=reference(index),
                args=[reference(i) for i in range(constraint_start)],
                keywords=[],
            )
            for index in range(constraint_start, len(args))
            if not isinstance(constraints[index - constraint_start], str)
        ]
        assume_call = ast.Call(
            func=_intrinsic_name('assume'),
            args=[reference(index) for index in range(len(args))],
            keywords=[],
        )
        body = [ast.Expr(value=ast.Yield(value=assume_call))]
        if checks:
            condition = checks[0] if len(checks) == 1 else ast.BoolOp(
                op=ast.And(), values=checks)
            body = [ast.If(test=condition, body=body, orelse=[])]

        func_ast = ast.FunctionDef(
            name='__refine_inline__',
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=name) for name in parameters],
                kwonlyargs=[], kw_defaults=[], defaults=[], vararg=None,
                kwarg=None,
            ),
            body=body, decorator_list=[], returns=None,
        )
        return ast.fix_missing_locations(ast.copy_location(func_ast, node))
