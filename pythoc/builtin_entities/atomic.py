"""Typed sequentially consistent atomic operations."""

from .base import BuiltinFunction
from .types import bool as pc_bool
from .types import i8, i32, i64, ptr, void
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
    address = ensure_ir(converted)
    alignment = pointee.get_size_bytes()
    visitor.builder.require_atomic_alignment(address, alignment, name, node)
    context = visitor.module.context
    logical_type = pointee.get_llvm_type(context)
    # A boolean occupies a byte of storage even though its value is an i1.
    storage_type = logical_type
    if get_base_type(pointee) is pc_bool:
        storage_type = i8.get_llvm_type(context)
        address = visitor.builder.bitcast(address, ptr[i8].get_llvm_type(context))
    return pointee, logical_type, storage_type, alignment, address


def _emit(visitor, operation, pointee, *args, **kwargs):
    from ..ir_helpers import is_volatile
    method = operation + '_volatile' if is_volatile(pointee) else operation
    return getattr(visitor.builder, method)(*args, **kwargs)


def _atomic_operand(visitor, argument, pointee, storage_type, node):
    value = ensure_ir(visitor.implicit_coercer.coerce(argument, pointee, node))
    if value.type != storage_type:
        return visitor.builder.zext(value, storage_type)
    return value


def _atomic_result(visitor, value, pointee, logical_type):
    if value.type != logical_type:
        value = visitor.builder.trunc(value, logical_type)
    return wrap_value(value, kind='value', type_hint=pointee)


def _handle_atomic(visitor, args, node, name, operation, arity, fixed):
    from ..type_converter import strip_qualifiers
    from ..type_descriptor import same_type
    if len(args) != arity:
        logger.error(f'{name}() takes exactly {arity} arguments',
                     node=node, exc_type=TypeError)
    if operation == 'fence':
        visitor.builder.fence('seq_cst')
        return wrap_value(None, kind='python', type_hint=void)
    pointee, llvm_type, storage_type, align, address = _atomic_pointee(
        visitor, args[0], operation, name, node, fixed)
    if operation == 'load':
        value = _emit(visitor, 'load_atomic', pointee, address,
                      ordering='seq_cst', align=align, typ=storage_type)
        return _atomic_result(visitor, value, pointee, llvm_type)
    if operation == 'cas':
        expected_type, _, _, expected_align, expected_address = _atomic_pointee(
            visitor, args[1], 'store', name, node, fixed)
        if not same_type(strip_qualifiers(pointee), strip_qualifiers(expected_type)):
            logger.error(f'{name}: expected pointer must have the same value type as target',
                         node=node, exc_type=TypeError)
        expected = _emit(visitor, 'load_atomic', expected_type, expected_address,
                         ordering='seq_cst', align=expected_align, typ=storage_type)
        desired = _atomic_operand(visitor, args[2], pointee, storage_type, node)
        pair = _emit(visitor, 'cmpxchg', pointee, address, expected, desired,
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
        result = visitor.builder.zext(success, i32.get_llvm_type(visitor.module.context))
        return wrap_value(result, kind='value', type_hint=i32)
    value = _atomic_operand(visitor, args[1], pointee, storage_type, node)
    if operation == 'store':
        _emit(visitor, 'store_atomic', pointee, value, address,
              ordering='seq_cst', align=align)
        return wrap_value(None, kind='python', type_hint=void)
    result = _emit(visitor, 'atomic_rmw', pointee, operation, address,
                   value, ordering='seq_cst')
    return _atomic_result(visitor, result, pointee, llvm_type)


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
atomic_fetch_sub = _atomic_builtin('atomic_fetch_sub', 'sub', 2)
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
