"""
LLVM va_list builtins for C ABI varargs interop.

These are explicit primitives for operating on C-style va_list.
Only needed for bare *args (no type annotation) functions that
must maintain C ABI compatibility.

Usage in pythoc:
    @compile
    def my_variadic(count: i32, *args) -> i32:
        ap = va_start()
        val = va_arg(ap, i32)
        va_end(ap)
        return val
"""

import ast

from .base import BuiltinFunction
from .types import ptr, i8, void
from ..valueref import wrap_value
from ..logger import logger


class va_start(BuiltinFunction):
    """va_start() -> ptr[i8]

    Initialize a va_list for the current varargs function.
    Returns a pointer to the va_list (i8*).
    """

    @classmethod
    def get_name(cls) -> str:
        return 'va_start'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 0:
            logger.error(
                "va_start() takes no arguments", node=node, exc_type=TypeError,
            )

        ap = visitor.builder.va_start()
        return wrap_value(ap, kind='value', type_hint=ptr[i8])


class va_arg(BuiltinFunction):
    """va_arg(ap, T) -> T

    Read the next argument of type T from the va_list.
    Uses the LLVM va_arg instruction which is lowered to correct
    platform-specific ABI code by the LLVM backend.
    """

    @classmethod
    def get_name(cls) -> str:
        return 'va_arg'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(node.args) != 2:
            logger.error(
                "va_arg() takes exactly 2 arguments: va_arg(ap, T)",
                node=node, exc_type=TypeError,
            )

        from ..valueref import ensure_ir
        ap_val = ensure_ir(args[0])

        # Second arg: the target type (parse from AST node, like sizeof does)
        target_pc_type = visitor.type_resolver.parse_annotation(node.args[1])
        if target_pc_type is None:
            logger.error(
                f"va_arg() second argument must be a type",
                node=node, exc_type=TypeError,
            )

        target_llvm_type = target_pc_type.get_llvm_type(visitor.module.context)
        value = visitor.builder.va_arg(ap_val, target_llvm_type, name="va.arg")

        return wrap_value(value, kind='value', type_hint=target_pc_type)


class va_end(BuiltinFunction):
    """va_end(ap) -> void

    Clean up a va_list. Must be called before the function returns.
    """

    @classmethod
    def get_name(cls) -> str:
        return 'va_end'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 1:
            logger.error(
                "va_end() takes exactly 1 argument", node=node, exc_type=TypeError,
            )

        from ..valueref import ensure_ir
        ap_val = ensure_ir(args[0])

        visitor.builder.va_end(ap_val)

        return wrap_value(None, kind='python', type_hint=void)
