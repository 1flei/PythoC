"""Integer bit-operation builtins (byte swap and bit counting).

These lower to LLVM intrinsics (``llvm.bswap``, ``llvm.ctpop``,
``llvm.ctlz``, ``llvm.cttz``) declared on demand for the argument's exact
integer width.
"""

import ast

from llvmlite import ir

from .base import BuiltinFunction
from ..logger import logger
from ..valueref import ensure_ir, wrap_value, get_type_hint


def _declare_intrinsic(visitor, intrinsic: str, ret_type):
    """Declare an LLVM intrinsic in the current module (idempotent)."""
    name = "%s.i%d" % (intrinsic, ret_type.width)
    fn = visitor.module.globals.get(name)
    if fn is None:
        fn = ir.Function(
            visitor.module, ir.FunctionType(ret_type, (ret_type,)), name,
        )
    return fn


def _int_arg_width(visitor, args, builtin_name, node):
    """Validate a single integer argument; return (llvm_value, width)."""
    if len(args) != 1:
        logger.error(
            f"{builtin_name}() takes exactly 1 argument",
            node=node, exc_type=TypeError,
        )
    value = ensure_ir(args[0])
    if not isinstance(value.type, ir.IntType):
        logger.error(
            f"{builtin_name}() requires an integer argument, got {value.type}",
            node=node, exc_type=TypeError,
        )
    return value, value.type.width


class bswap(BuiltinFunction):
    """bswap(N) -> N: byte-reverse an integer of any width."""

    _borrows_args = True

    @classmethod
    def get_name(cls) -> str:
        return 'bswap'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        value, width = _int_arg_width(visitor, args, 'bswap', node)
        fn = _declare_intrinsic(visitor, 'llvm.bswap', ir.IntType(width))
        result = visitor.builder.call(fn, [value])
        return wrap_value(result, kind='value', type_hint=get_type_hint(args[0]))


class popcount(BuiltinFunction):
    """popcount(N) -> N: number of set bits (same width as input)."""

    _borrows_args = True

    @classmethod
    def get_name(cls) -> str:
        return 'popcount'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        value, width = _int_arg_width(visitor, args, 'popcount', node)
        fn = _declare_intrinsic(visitor, 'llvm.ctpop', ir.IntType(width))
        result = visitor.builder.call(fn, [value])
        return wrap_value(result, kind='value', type_hint=get_type_hint(args[0]))


class ctlz(BuiltinFunction):
    """ctlz(N) -> N: count leading zeros (same width as input).

    Declared in the one-argument form; llvmlite's AutoUpgrade rewrites it
    to the modern two-argument form with ``is_zero_undef=false`` on parse,
    so zero input is well-defined (returns the full width).
    """

    _borrows_args = True

    @classmethod
    def get_name(cls) -> str:
        return 'ctlz'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        value, width = _int_arg_width(visitor, args, 'ctlz', node)
        fn = _declare_intrinsic(visitor, 'llvm.ctlz', ir.IntType(width))
        result = visitor.builder.call(fn, [value])
        return wrap_value(result, kind='value', type_hint=get_type_hint(args[0]))


class cttz(BuiltinFunction):
    """cttz(N) -> N: count trailing zeros (same width as input).

    Same one-argument declaration as ctlz (well-defined at zero).
    """

    _borrows_args = True

    @classmethod
    def get_name(cls) -> str:
        return 'cttz'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        value, width = _int_arg_width(visitor, args, 'cttz', node)
        fn = _declare_intrinsic(visitor, 'llvm.cttz', ir.IntType(width))
        result = visitor.builder.call(fn, [value])
        return wrap_value(result, kind='value', type_hint=get_type_hint(args[0]))


__all__ = [
    'bswap',
    'popcount',
    'ctlz',
    'cttz',
]
