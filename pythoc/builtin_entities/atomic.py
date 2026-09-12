import ast

from llvmlite import ir

from .base import BuiltinFunction
from .types import i32, i64, void
from ..logger import logger
from ..valueref import ensure_ir, wrap_value, get_type_hint


def _atomic_pointee(visitor, arg, param_name, node):
    """Resolve the pointee PC type of a typed-pointer argument.

    Returns (pc_type, llvm_type, align_bytes).  ``ptr[T]`` arguments carry a
    specialized ptr class in their type_hint whose ``pointee_type`` names T;
    forward-reference strings resolve through the session registry.
    """
    type_hint = get_type_hint(arg)
    pointee = getattr(type_hint, 'pointee_type', None)
    if isinstance(pointee, str):
        from ..forward_ref import get_defined_type
        resolved = get_defined_type(pointee)
        if resolved is not None:
            pointee = resolved
    if pointee is None or isinstance(pointee, str):
        logger.error(
            f"{param_name}: first argument must be a typed pointer (ptr[T])",
            node=node, exc_type=TypeError,
        )
    if pointee is void:
        logger.error(
            f"{param_name}: cannot operate on ptr[void]; cast to a concrete "
            f"pointer type first",
            node=node, exc_type=TypeError,
        )
    llvm_type = pointee.get_llvm_type(visitor.module.context)
    size = pointee.get_size_bytes()
    if size is None:
        logger.error(
            f"{param_name}: unsupported pointee type {pointee.get_name()} "
            f"for atomic operation",
            node=node, exc_type=TypeError,
        )
    return pointee, llvm_type, size


def _emit_atomic_rmw(visitor, op, args, param_name, node):
    """Shared lowering for fetch-add/and/or over the generic width path."""
    pointee, _, _ = _atomic_pointee(visitor, args[0], param_name, node)
    ptr_value = ensure_ir(args[0])
    value = visitor.implicit_coercer.coerce(args[1], pointee, node)
    result = visitor.builder.atomic_rmw(
        op, ptr_value, ensure_ir(value), ordering='seq_cst',
    )
    return wrap_value(result, kind='value', type_hint=pointee)


class atomic_load(BuiltinFunction):
    """atomic_load(ptr[T]) -> T

    Seq-cst atomic load of any integer/pointer width.  Generic-width version
    of atomic_load_i32/atomic_load_i64: the width and alignment are derived
    from the typed-pointer argument.
    """

    _borrows_args = True

    @classmethod
    def get_name(cls) -> str:
        return 'atomic_load'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 1:
            logger.error(
                "atomic_load() takes exactly 1 argument",
                node=node, exc_type=TypeError,
            )
        pointee, llvm_type, align = _atomic_pointee(visitor, args[0], 'atomic_load', node)
        ptr_value = ensure_ir(args[0])
        result = visitor.builder.load_atomic(
            ptr_value, ordering='seq_cst', align=align, typ=llvm_type,
        )
        return wrap_value(result, kind='value', type_hint=pointee)


class atomic_store(BuiltinFunction):
    """atomic_store(ptr[T], value) -> void

    Seq-cst atomic store of any integer/pointer width.
    """

    _borrows_args = True

    @classmethod
    def get_name(cls) -> str:
        return 'atomic_store'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 2:
            logger.error(
                "atomic_store() takes exactly 2 arguments",
                node=node, exc_type=TypeError,
            )
        pointee, _, align = _atomic_pointee(visitor, args[0], 'atomic_store', node)
        ptr_value = ensure_ir(args[0])
        value = visitor.implicit_coercer.coerce(args[1], pointee, node)
        visitor.builder.store_atomic(
            ensure_ir(value), ptr_value, ordering='seq_cst', align=align,
        )
        return wrap_value(None, kind='python', type_hint=void)


class atomic_fetch_add(BuiltinFunction):
    """atomic_fetch_add(ptr[T], value) -> T (old value)

    Seq-cst atomic add returning the value before the operation.
    """

    _borrows_args = True

    @classmethod
    def get_name(cls) -> str:
        return 'atomic_fetch_add'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 2:
            logger.error(
                "atomic_fetch_add() takes exactly 2 arguments",
                node=node, exc_type=TypeError,
            )
        return _emit_atomic_rmw(
            visitor, 'add', args, 'atomic_fetch_add', node)


class atomic_fetch_and(BuiltinFunction):
    """atomic_fetch_and(ptr[T], value) -> T (old value)"""

    _borrows_args = True

    @classmethod
    def get_name(cls) -> str:
        return 'atomic_fetch_and'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 2:
            logger.error(
                "atomic_fetch_and() takes exactly 2 arguments",
                node=node, exc_type=TypeError,
            )
        return _emit_atomic_rmw(
            visitor, 'and', args, 'atomic_fetch_and', node)


class atomic_fetch_or(BuiltinFunction):
    """atomic_fetch_or(ptr[T], value) -> T (old value)"""

    _borrows_args = True

    @classmethod
    def get_name(cls) -> str:
        return 'atomic_fetch_or'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 2:
            logger.error(
                "atomic_fetch_or() takes exactly 2 arguments",
                node=node, exc_type=TypeError,
            )
        return _emit_atomic_rmw(
            visitor, 'or', args, 'atomic_fetch_or', node)


class atomic_exchange(BuiltinFunction):
    """atomic_exchange(ptr[T], value) -> T (old value)

    Seq-cst atomic exchange returning the value before the operation.
    """

    _borrows_args = True

    @classmethod
    def get_name(cls) -> str:
        return 'atomic_exchange'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 2:
            logger.error(
                "atomic_exchange() takes exactly 2 arguments",
                node=node, exc_type=TypeError,
            )
        pointee, _, _ = _atomic_pointee(visitor, args[0], 'atomic_exchange', node)
        ptr_value = ensure_ir(args[0])
        value = visitor.implicit_coercer.coerce(args[1], pointee, node)
        result = visitor.builder.atomic_rmw(
            'xchg', ptr_value, ensure_ir(value), ordering='seq_cst',
        )
        return wrap_value(result, kind='value', type_hint=pointee)


class atomic_cas(BuiltinFunction):
    """atomic_cas(ptr[T], expected_ptr[T], desired: T) -> i32

    Seq-cst compare-and-swap.  On failure the observed value is written back
    to ``expected_ptr`` (the C11 __atomic_compare_exchange contract).
    """

    _borrows_args = True

    @classmethod
    def get_name(cls) -> str:
        return 'atomic_cas'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 3:
            logger.error(
                "atomic_cas() takes exactly 3 arguments",
                node=node, exc_type=TypeError,
            )
        pointee, llvm_type, align = _atomic_pointee(
            visitor, args[0], 'atomic_cas', node)
        ptr_value = ensure_ir(args[0])
        expected_ptr = ensure_ir(args[1])
        expected = visitor.builder.load_atomic(
            expected_ptr, ordering='seq_cst', align=align, typ=llvm_type,
        )
        desired = visitor.implicit_coercer.coerce(args[2], pointee, node)
        pair = visitor.builder.cmpxchg(
            ptr_value, expected, ensure_ir(desired),
            ordering='seq_cst', failordering='seq_cst',
        )
        old_value = visitor.builder.extract_value(pair, 0)
        success = visitor.builder.extract_value(pair, 1)
        visitor.builder.store_atomic(
            old_value, expected_ptr, ordering='seq_cst', align=align,
        )
        result = visitor.builder.zext(success, ir.IntType(32))
        return wrap_value(result, kind='value', type_hint=i32)


class atomic_fence(BuiltinFunction):
    """atomic_fence() -> void

    Seq-cst thread fence (C11 atomic_thread_fence equivalent).
    """

    @classmethod
    def get_name(cls) -> str:
        return 'atomic_fence'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 0:
            logger.error(
                "atomic_fence() takes no arguments",
                node=node, exc_type=TypeError,
            )
        visitor.builder.fence('seq_cst')
        return wrap_value(None, kind='python', type_hint=void)


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


class atomic_cas_i32(BuiltinFunction):
    @classmethod
    def get_name(cls) -> str:
        return 'atomic_cas_i32'

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node: ast.Call):
        if len(args) != 3:
            logger.error(
                "atomic_cas_i32() takes exactly 3 arguments",
                node=node, exc_type=TypeError,
            )
        ptr_value = ensure_ir(args[0])
        expected_ptr = ensure_ir(args[1])
        expected = visitor.builder.load_atomic(
            expected_ptr, ordering='seq_cst', align=4, typ=ir.IntType(32),
        )
        desired = visitor.implicit_coercer.coerce(args[2], i32, node)
        pair = visitor.builder.cmpxchg(
            ptr_value, expected, ensure_ir(desired),
            ordering='seq_cst', failordering='seq_cst',
        )
        old_value = visitor.builder.extract_value(pair, 0)
        success = visitor.builder.extract_value(pair, 1)
        visitor.builder.store_atomic(
            old_value, expected_ptr, ordering='seq_cst', align=4,
        )
        result = visitor.builder.zext(success, ir.IntType(32))
        return wrap_value(result, kind='value', type_hint=i32)
