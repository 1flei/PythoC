"""Typed sequentially consistent atomic operations."""

from llvmlite import ir

from .base import BuiltinFunction
from .types import i32, i64, ptr, void
from ..logger import logger
from ..valueref import ensure_ir, wrap_value


def _atomic_pointee(visitor, arg, operation, name, node, fixed=None):
    from ..type_converter import get_base_type, strip_qualifiers
    from ..type_descriptor import atomic_value_is_supported, same_type
    source = get_base_type(arg.type_hint)
    if not isinstance(source, type) or not issubclass(source, ptr):
        logger.error(f'{name}: argument must be a typed data pointer',
                     node=node, exc_type=TypeError)
    pointee = source._resolve_pointee(getattr(visitor, 'ctx', None))
    if pointee is None or isinstance(pointee, str) or pointee is void:
        logger.error(f'{name}: argument must be a complete typed pointer',
                     node=node, exc_type=TypeError)
    if fixed is not None and not same_type(strip_qualifiers(pointee), fixed):
        logger.error(f'{name}: expected a pointer to {fixed.get_name()}',
                     node=node, exc_type=TypeError)
    if not atomic_value_is_supported(pointee, operation):
        logger.error(f'{name}: unsupported atomic {operation} on {pointee.get_name()}',
                     node=node, exc_type=TypeError)
    converted = visitor.type_converter.convert(arg, ptr[pointee], node)
    return (pointee, pointee.get_llvm_type(visitor.module.context),
            pointee.get_size_bytes(), ensure_ir(converted))


def _emit(visitor, operation, pointee, *args, **kwargs):
    from ..ir_helpers import is_volatile
    method = operation + '_volatile' if is_volatile(pointee) else operation
    return getattr(visitor.builder, method)(*args, **kwargs)


def _handle_atomic(visitor, args, node, name, operation, arity, fixed):
    from ..type_converter import strip_qualifiers
    from ..type_descriptor import same_type
    if len(args) != arity:
        logger.error(f'{name}() takes exactly {arity} arguments',
                     node=node, exc_type=TypeError)
    if operation == 'fence':
        visitor.builder.fence('seq_cst')
        return wrap_value(None, kind='python', type_hint=void)
    pointee, llvm_type, align, address = _atomic_pointee(
        visitor, args[0], operation, name, node, fixed)
    if operation == 'load':
        value = _emit(visitor, 'load_atomic', pointee, address,
                      ordering='seq_cst', align=align, typ=llvm_type)
        return wrap_value(value, kind='value', type_hint=pointee)
    if operation == 'cas':
        expected_type, _, expected_align, expected_address = _atomic_pointee(
            visitor, args[1], 'store', name, node, fixed)
        if not same_type(strip_qualifiers(pointee), strip_qualifiers(expected_type)):
            logger.error(f'{name}: expected pointer must have the same value type as target',
                         node=node, exc_type=TypeError)
        expected = _emit(visitor, 'load_atomic', expected_type, expected_address,
                         ordering='seq_cst', align=expected_align, typ=llvm_type)
        desired = visitor.implicit_coercer.coerce(args[2], pointee, node)
        pair = _emit(visitor, 'cmpxchg', pointee, address, expected, ensure_ir(desired),
                     ordering='seq_cst', failordering='seq_cst')
        old = visitor.builder.extract_value(pair, 0)
        success = visitor.builder.extract_value(pair, 1)
        failure = visitor.builder.create_block('atomic.cas.failure')
        merge = visitor.builder.create_block('atomic.cas.merge')
        visitor.builder.cbranch(success, merge, failure)
        visitor.builder.position_at_end(failure)
        _emit(visitor, 'store_atomic', expected_type, old, expected_address,
              ordering='seq_cst', align=expected_align)
        visitor.builder.branch(merge)
        visitor.builder.position_at_end(merge)
        result = visitor.builder.zext(success, ir.IntType(32))
        return wrap_value(result, kind='value', type_hint=i32)
    value = visitor.implicit_coercer.coerce(args[1], pointee, node)
    if operation == 'store':
        _emit(visitor, 'store_atomic', pointee, ensure_ir(value), address,
              ordering='seq_cst', align=align)
        return wrap_value(None, kind='python', type_hint=void)
    result = _emit(visitor, 'atomic_rmw', pointee, operation, address,
                   ensure_ir(value), ordering='seq_cst')
    return wrap_value(result, kind='value', type_hint=pointee)


def _atomic_builtin(name, operation, arity, fixed=None):
    def handle_type_call(cls, visitor, func_ref, args, node):
        return _handle_atomic(visitor, args, node, name, operation, arity, fixed)
    return type(name, (BuiltinFunction,), {
        '__module__': __name__,
        '_borrows_args': True,
        'get_name': classmethod(lambda cls: name),
        'handle_type_call': classmethod(handle_type_call),
    })


atomic_load = _atomic_builtin('atomic_load', 'load', 1)
atomic_store = _atomic_builtin('atomic_store', 'store', 2)
atomic_fetch_add = _atomic_builtin('atomic_fetch_add', 'add', 2)
atomic_fetch_and = _atomic_builtin('atomic_fetch_and', 'and', 2)
atomic_fetch_or = _atomic_builtin('atomic_fetch_or', 'or', 2)
atomic_exchange = _atomic_builtin('atomic_exchange', 'xchg', 2)
atomic_cas = _atomic_builtin('atomic_cas', 'cas', 3)
atomic_fence = _atomic_builtin('atomic_fence', 'fence', 0)
atomic_load_i64 = _atomic_builtin('atomic_load_i64', 'load', 1, i64)
atomic_load_i32 = _atomic_builtin('atomic_load_i32', 'load', 1, i32)
atomic_store_i64 = _atomic_builtin('atomic_store_i64', 'store', 2, i64)
atomic_store_i32 = _atomic_builtin('atomic_store_i32', 'store', 2, i32)
atomic_fetch_add_i64 = _atomic_builtin('atomic_fetch_add_i64', 'add', 2, i64)
atomic_cas_i64 = _atomic_builtin('atomic_cas_i64', 'cas', 3, i64)
atomic_cas_i32 = _atomic_builtin('atomic_cas_i32', 'cas', 3, i32)
