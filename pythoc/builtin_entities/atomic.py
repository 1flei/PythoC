import ast

from llvmlite import ir

from .base import BuiltinFunction
from .types import i32, i64, void
from ..logger import logger
from ..valueref import ensure_ir, wrap_value


class atomic_load_i64(BuiltinFunction):
    @classmethod
    def get_name(cls) -> str:
        return 'atomic_load_i64'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 1:
            logger.error(
                "atomic_load_i64() takes exactly 1 argument",
                node=node, exc_type=TypeError,
            )
        ptr_value = ensure_ir(args[0])
        result = visitor.builder.load_atomic(
            ptr_value, ordering='seq_cst', align=8, typ=ir.IntType(64),
        )
        return wrap_value(result, kind='value', type_hint=i64)


class atomic_load_i32(BuiltinFunction):
    @classmethod
    def get_name(cls) -> str:
        return 'atomic_load_i32'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 1:
            logger.error(
                "atomic_load_i32() takes exactly 1 argument",
                node=node, exc_type=TypeError,
            )
        ptr_value = ensure_ir(args[0])
        result = visitor.builder.load_atomic(
            ptr_value, ordering='seq_cst', align=4, typ=ir.IntType(32),
        )
        return wrap_value(result, kind='value', type_hint=i32)


class atomic_store_i64(BuiltinFunction):
    @classmethod
    def get_name(cls) -> str:
        return 'atomic_store_i64'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 2:
            logger.error(
                "atomic_store_i64() takes exactly 2 arguments",
                node=node, exc_type=TypeError,
            )
        ptr_value = ensure_ir(args[0])
        value = visitor.implicit_coercer.coerce(args[1], i64, node)
        visitor.builder.store_atomic(
            ensure_ir(value), ptr_value, ordering='seq_cst', align=8,
        )
        return wrap_value(None, kind='python', type_hint=void)


class atomic_store_i32(BuiltinFunction):
    @classmethod
    def get_name(cls) -> str:
        return 'atomic_store_i32'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 2:
            logger.error(
                "atomic_store_i32() takes exactly 2 arguments",
                node=node, exc_type=TypeError,
            )
        ptr_value = ensure_ir(args[0])
        value = visitor.implicit_coercer.coerce(args[1], i32, node)
        visitor.builder.store_atomic(
            ensure_ir(value), ptr_value, ordering='seq_cst', align=4,
        )
        return wrap_value(None, kind='python', type_hint=void)


class atomic_fetch_add_i64(BuiltinFunction):
    @classmethod
    def get_name(cls) -> str:
        return 'atomic_fetch_add_i64'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 2:
            logger.error(
                "atomic_fetch_add_i64() takes exactly 2 arguments",
                node=node, exc_type=TypeError,
            )
        ptr_value = ensure_ir(args[0])
        value = visitor.implicit_coercer.coerce(args[1], i64, node)
        result = visitor.builder.atomic_rmw(
            'add', ptr_value, ensure_ir(value), ordering='seq_cst',
        )
        return wrap_value(result, kind='value', type_hint=i64)


class atomic_cas_i64(BuiltinFunction):
    @classmethod
    def get_name(cls) -> str:
        return 'atomic_cas_i64'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 3:
            logger.error(
                "atomic_cas_i64() takes exactly 3 arguments",
                node=node, exc_type=TypeError,
            )
        ptr_value = ensure_ir(args[0])
        expected_ptr = ensure_ir(args[1])
        expected = visitor.builder.load_atomic(
            expected_ptr, ordering='seq_cst', align=8, typ=ir.IntType(64),
        )
        desired = visitor.implicit_coercer.coerce(args[2], i64, node)
        pair = visitor.builder.cmpxchg(
            ptr_value, expected, ensure_ir(desired),
            ordering='seq_cst', failordering='seq_cst',
        )
        old_value = visitor.builder.extract_value(pair, 0)
        success = visitor.builder.extract_value(pair, 1)
        visitor.builder.store_atomic(
            old_value, expected_ptr, ordering='seq_cst', align=8,
        )
        result = visitor.builder.zext(success, ir.IntType(32))
        return wrap_value(result, kind='value', type_hint=i32)
