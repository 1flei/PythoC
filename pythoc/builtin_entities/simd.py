"""Fixed-width CPU SIMD value types and intrinsics.

This module intentionally keeps the first pcllm-facing SIMD surface small:
``simd[T, N]`` lowers to an LLVM ``<N x T>`` register on CPU, with arithmetic,
lane extraction, splat construction, FMA, and reductions.
"""

import builtins

from llvmlite import ir

from .base import BuiltinFunction, BuiltinType
from ..logger import logger
from ..valueref import ensure_ir, wrap_value


def _i32(value):
    return ir.Constant(ir.IntType(32), value)


def _i32_type():
    from .types import i32

    return i32


def _vector_suffix(vec_type):
    elem = vec_type.element
    if isinstance(elem, ir.FloatType):
        scalar = "f32"
    elif isinstance(elem, ir.DoubleType):
        scalar = "f64"
    elif isinstance(elem, ir.IntType):
        scalar = "i%d" % elem.width
    elif isinstance(elem, ir.HalfType):
        scalar = "f16"
    elif elem.__class__.__name__ == "BFloatType":
        scalar = "bf16"
    else:
        logger.error(
            "simd intrinsics do not support element type %s" % elem,
            node=None,
            exc_type=TypeError,
        )
    return "v%d%s" % (vec_type.count, scalar)


def _splat(visitor, scalar_ir, vec_type):
    undef = ir.Constant(vec_type, ir.Undefined)
    inserted = visitor.builder.insert_element(undef, scalar_ir, _i32(0))
    mask = ir.Constant(ir.VectorType(ir.IntType(32), vec_type.count),
                       [0] * vec_type.count)
    return visitor.builder.shuffle_vector(inserted, undef, mask)


def _declare(module, name, fn_ty):
    try:
        return module.get_global(name)
    except KeyError:
        return ir.Function(module, fn_ty, name)


def _same_simd_type(left, right):
    return (
        isinstance(left, type)
        and isinstance(right, type)
        and issubclass(left, simd)
        and issubclass(right, simd)
        and left.element_type is right.element_type
        and left.lanes == right.lanes
    )


def _simd_type_argument(arg, fn_name, node):
    simd_type = arg.get_python_value() if arg.is_python_value() else None
    if not (isinstance(simd_type, type) and issubclass(simd_type, simd)):
        logger.error(
            "%s type argument must be simd[T, N]" % fn_name,
            node=node,
            exc_type=TypeError,
        )
    return simd_type


def _vector_pointer(visitor, arg, simd_type, fn_name, node):
    base_ir = ensure_ir(arg)
    if not isinstance(base_ir.type, ir.PointerType):
        logger.error(
            "%s expects a pointer operand" % fn_name,
            node=node,
            exc_type=TypeError,
        )
    return visitor.builder.bitcast(
        base_ir,
        ir.PointerType(
            simd_type.get_llvm_type(visitor.module.context),
            addrspace=getattr(base_ir.type, "addrspace", 0),
        ),
    )


class simd(BuiltinType):
    """simd[T, N]: an LLVM vector register value."""

    _is_signed = False
    element_type = None
    lanes = None

    @classmethod
    def get_name(cls):
        if cls.element_type is None or cls.lanes is None:
            return "simd"
        return "simd[%s, %d]" % (cls.element_type.get_name(), cls.lanes)

    @classmethod
    def get_type_id(cls, _visited=None):
        if cls.element_type is None or cls.lanes is None:
            return "Vx"
        from ..type_id import get_type_id

        return "V%s_%d" % (get_type_id(cls.element_type, _visited), cls.lanes)

    @classmethod
    def get_llvm_type(cls, module_context=None):
        if cls.element_type is None or cls.lanes is None:
            logger.error("simd requires element type and lane count",
                         node=None, exc_type=TypeError)
        return ir.VectorType(cls.element_type.get_llvm_type(module_context),
                             cls.lanes)

    @classmethod
    def get_layout(cls):
        if cls.element_type is None or cls.lanes is None:
            from ..target_layout import TypeLayout
            return TypeLayout(0, 0, 1, 0)
        from ..target_layout import get_target_layout
        return get_target_layout().primitive(cls.get_llvm_type())

    @classmethod
    def handle_type_subscript(cls, items):
        if not isinstance(items, builtins.tuple):
            items = (items,)
        if len(items) != 2:
            logger.error(
                "simd[T, N] requires an element type and a lane count",
                node=None,
                exc_type=TypeError,
            )
        (_, element_type), (_, lanes) = items
        if not (
            isinstance(element_type, type)
            and issubclass(element_type, BuiltinType)
        ):
            logger.error(
                "simd element must be a PythoC builtin type",
                node=None,
                exc_type=TypeError,
            )
        element_ir = element_type.get_llvm_type()
        is_vector_element = isinstance(
            element_ir,
            (ir.IntType, ir.HalfType, ir.FloatType, ir.DoubleType),
        ) or element_ir.__class__.__name__ == "BFloatType"
        if not is_vector_element:
            logger.error(
                "simd element type %s is not numeric"
                % element_type.get_name(),
                node=None,
                exc_type=TypeError,
            )
        if not isinstance(lanes, int) or lanes <= 0:
            logger.error(
                "simd lane count must be a positive integer, got %r" % lanes,
                node=None,
                exc_type=TypeError,
            )
        name = "simd[%s, %d]" % (element_type.get_name(), lanes)
        return type(name, (cls,), {
            "_pc_specialized": True,
            "element_type": element_type,
            "lanes": lanes,
        })

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        if cls.element_type is None or cls.lanes is None:
            logger.error("cannot construct unspecialized simd",
                         node=node, exc_type=TypeError)
        if len(args) != 1:
            logger.error("%s() takes exactly one argument" % cls.get_name(),
                         node=node, exc_type=TypeError)
        scalar = visitor.implicit_coercer.coerce(args[0], cls.element_type,
                                                node)
        result = _splat(visitor, ensure_ir(scalar), cls.get_llvm_type())
        return wrap_value(result, kind="value", type_hint=cls)

    @classmethod
    def _coerce_operand(cls, visitor, operand, node):
        type_hint = getattr(operand, "type_hint", None)
        if isinstance(type_hint, type) and issubclass(type_hint, simd):
            if not _same_simd_type(cls, type_hint):
                logger.error("cannot mix %s and %s" % (
                    cls.get_name(), type_hint.get_name()),
                    node=node,
                    exc_type=TypeError,
                )
            return ensure_ir(operand)
        scalar = visitor.implicit_coercer.coerce(operand, cls.element_type,
                                                node)
        return _splat(visitor, ensure_ir(scalar), cls.get_llvm_type())

    @classmethod
    def _binary(cls, visitor, left, right, node, float_op, int_op):
        lhs = cls._coerce_operand(visitor, left, node)
        rhs = cls._coerce_operand(visitor, right, node)
        op = float_op if cls.element_type.is_float() else int_op
        result = getattr(visitor.builder, op)(lhs, rhs)
        return wrap_value(result, kind="value", type_hint=cls)

    @classmethod
    def handle_add(cls, visitor, left, right, node):
        return cls._binary(visitor, left, right, node, "fadd", "add")

    @classmethod
    def handle_sub(cls, visitor, left, right, node):
        return cls._binary(visitor, left, right, node, "fsub", "sub")

    @classmethod
    def handle_mul(cls, visitor, left, right, node):
        return cls._binary(visitor, left, right, node, "fmul", "mul")

    @classmethod
    def handle_div(cls, visitor, left, right, node):
        int_op = "sdiv" if cls.element_type.is_signed() else "udiv"
        return cls._binary(visitor, left, right, node, "fdiv", int_op)

    @classmethod
    def handle_subscript(cls, visitor, base, index, node):
        idx = visitor.implicit_coercer.coerce(index, _i32_type(), node)
        result = visitor.builder.extract_element(
            ensure_ir(visitor.read_rvalue(base)), ensure_ir(idx))
        return wrap_value(result, kind="value", type_hint=cls.element_type)


def _require_simd(arg, fn_name, node):
    type_hint = getattr(arg, "type_hint", None)
    if not (isinstance(type_hint, type) and issubclass(type_hint, simd)):
        logger.error("%s() expects a simd value" % fn_name,
                     node=node, exc_type=TypeError)
    return type_hint


class simd_splat(BuiltinFunction):
    @classmethod
    def get_name(cls):
        return "simd_splat"

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        if len(args) != 2:
            logger.error("simd_splat(scalar, type) takes two arguments",
                         node=node, exc_type=TypeError)
        simd_type = _simd_type_argument(
            args[1], "simd_splat", node)
        return simd_type.handle_type_call(visitor, func_ref, [args[0]], node)


class simd_load(BuiltinFunction):
    """simd_load(ptr, simd_type) -> unaligned vector load."""

    @classmethod
    def get_name(cls):
        return "simd_load"

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        if len(args) != 2:
            logger.error("simd_load(ptr, type) takes two arguments",
                         node=node, exc_type=TypeError)
        simd_type = _simd_type_argument(args[1], "simd_load", node)
        ptr_ir = _vector_pointer(
            visitor, args[0], simd_type, "simd_load", node)
        result = visitor.builder.load(ptr_ir, align=1)
        return wrap_value(result, kind="value", type_hint=simd_type)


class simd_store(BuiltinFunction):
    """simd_store(ptr, value) -> unaligned vector store."""

    @classmethod
    def get_name(cls):
        return "simd_store"

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        if len(args) != 2:
            logger.error("simd_store(ptr, value) takes two arguments",
                         node=node, exc_type=TypeError)
        simd_type = _require_simd(args[1], "simd_store", node)
        ptr_ir = _vector_pointer(
            visitor, args[0], simd_type, "simd_store", node)
        visitor.builder.store(ensure_ir(args[1]), ptr_ir, align=1)
        from .types import void

        return wrap_value(None, kind="python", type_hint=void)


class simd_fma(BuiltinFunction):
    @classmethod
    def get_name(cls):
        return "simd_fma"

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        if len(args) != 3:
            logger.error("simd_fma(a, b, c) takes three arguments",
                         node=node, exc_type=TypeError)
        simd_type = _require_simd(args[0], "simd_fma", node)
        vec_type = simd_type.get_llvm_type(visitor.module.context)
        if not simd_type.element_type.is_float():
            logger.error("simd_fma requires a floating point simd type",
                         node=node, exc_type=TypeError)
        suffix = _vector_suffix(vec_type)
        fn = _declare(
            visitor.module,
            "llvm.fma." + suffix,
            ir.FunctionType(vec_type, [vec_type, vec_type, vec_type]),
        )
        operands = [simd_type._coerce_operand(visitor, a, node) for a in args]
        result = visitor.builder.call(fn, operands)
        return wrap_value(result, kind="value", type_hint=simd_type)


class simd_reduce_add(BuiltinFunction):
    @classmethod
    def get_name(cls):
        return "simd_reduce_add"

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        if len(args) != 1:
            logger.error("simd_reduce_add(v) takes one argument",
                         node=node, exc_type=TypeError)
        simd_type = _require_simd(args[0], "simd_reduce_add", node)
        vec_type = simd_type.get_llvm_type(visitor.module.context)
        elem = vec_type.element
        suffix = _vector_suffix(vec_type)
        if simd_type.element_type.is_float():
            fn = _declare(
                visitor.module,
                "llvm.vector.reduce.fadd." + suffix,
                ir.FunctionType(elem, [elem, vec_type]),
            )
            result = visitor.builder.call(
                fn, [ir.Constant(elem, 0.0), ensure_ir(args[0])])
        else:
            fn = _declare(
                visitor.module,
                "llvm.vector.reduce.add." + suffix,
                ir.FunctionType(elem, [vec_type]),
            )
            result = visitor.builder.call(fn, [ensure_ir(args[0])])
        return wrap_value(result, kind="value",
                          type_hint=simd_type.element_type)


class simd_reduce_max(BuiltinFunction):
    @classmethod
    def get_name(cls):
        return "simd_reduce_max"

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        if len(args) != 1:
            logger.error("simd_reduce_max(v) takes one argument",
                         node=node, exc_type=TypeError)
        simd_type = _require_simd(args[0], "simd_reduce_max", node)
        vec_type = simd_type.get_llvm_type(visitor.module.context)
        elem = vec_type.element
        suffix = _vector_suffix(vec_type)
        if simd_type.element_type.is_float():
            name = "llvm.vector.reduce.fmax." + suffix
            fn_ty = ir.FunctionType(elem, [vec_type])
        else:
            kind = "smax" if simd_type.element_type.is_signed() else "umax"
            name = "llvm.vector.reduce.%s.%s" % (kind, suffix)
            fn_ty = ir.FunctionType(elem, [vec_type])
        fn = _declare(visitor.module, name, fn_ty)
        result = visitor.builder.call(fn, [ensure_ir(args[0])])
        return wrap_value(result, kind="value",
                          type_hint=simd_type.element_type)


class simd_bf16_dot(BuiltinFunction):
    """simd_bf16_dot(acc, x0, x1, w) -> AVX512 BF16 dot product step."""

    @classmethod
    def get_name(cls):
        return "simd_bf16_dot"

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        if len(args) != 4:
            logger.error("simd_bf16_dot(acc, x0, x1, w) takes four arguments",
                         node=node, exc_type=TypeError)
        from .types import f32, u16

        acc_ty = _require_simd(args[0], "simd_bf16_dot", node)
        x0_ty = _require_simd(args[1], "simd_bf16_dot", node)
        x1_ty = _require_simd(args[2], "simd_bf16_dot", node)
        w_ty = _require_simd(args[3], "simd_bf16_dot", node)
        if acc_ty.element_type is not f32 or acc_ty.lanes != 16:
            logger.error("simd_bf16_dot acc must be simd[f32, 16]",
                         node=node, exc_type=TypeError)
        if x0_ty.element_type is not f32 or x0_ty.lanes != 16:
            logger.error("simd_bf16_dot x0 must be simd[f32, 16]",
                         node=node, exc_type=TypeError)
        if x1_ty.element_type is not f32 or x1_ty.lanes != 16:
            logger.error("simd_bf16_dot x1 must be simd[f32, 16]",
                         node=node, exc_type=TypeError)
        if w_ty.element_type is not u16 or w_ty.lanes != 32:
            logger.error("simd_bf16_dot w must be simd[u16, 32]",
                         node=node, exc_type=TypeError)

        f32x16 = acc_ty.get_llvm_type(visitor.module.context)
        u16x32 = w_ty.get_llvm_type(visitor.module.context)
        cvt = _declare(
            visitor.module,
            "llvm.x86.avx512bf16.cvtne2ps2bf16.512",
            ir.FunctionType(u16x32, [f32x16, f32x16]),
        )
        dot = _declare(
            visitor.module,
            "llvm.x86.avx512bf16.dpbf16ps.512",
            ir.FunctionType(f32x16, [f32x16, u16x32, u16x32]),
        )
        xb = visitor.builder.call(cvt, [ensure_ir(args[2]), ensure_ir(args[1])])
        result = visitor.builder.call(
            dot, [ensure_ir(args[0]), xb, ensure_ir(args[3])])
        return wrap_value(result, kind="value", type_hint=acc_ty)


class simd_f32_to_bf16(BuiltinFunction):
    """simd_f32_to_bf16(x0, x1) -> packed simd[u16, 32]."""

    @classmethod
    def get_name(cls):
        return "simd_f32_to_bf16"

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        if len(args) != 2:
            logger.error("simd_f32_to_bf16(x0, x1) takes two arguments",
                         node=node, exc_type=TypeError)
        from .types import f32, u16

        x0_ty = _require_simd(args[0], "simd_f32_to_bf16", node)
        x1_ty = _require_simd(args[1], "simd_f32_to_bf16", node)
        if x0_ty.element_type is not f32 or x0_ty.lanes != 16:
            logger.error("simd_f32_to_bf16 x0 must be simd[f32, 16]",
                         node=node, exc_type=TypeError)
        if x1_ty.element_type is not f32 or x1_ty.lanes != 16:
            logger.error("simd_f32_to_bf16 x1 must be simd[f32, 16]",
                         node=node, exc_type=TypeError)
        out_ty = simd[u16, 32]
        f32x16 = x0_ty.get_llvm_type(visitor.module.context)
        u16x32 = out_ty.get_llvm_type(visitor.module.context)
        cvt = _declare(
            visitor.module,
            "llvm.x86.avx512bf16.cvtne2ps2bf16.512",
            ir.FunctionType(u16x32, [f32x16, f32x16]),
        )
        result = visitor.builder.call(
            cvt, [ensure_ir(args[1]), ensure_ir(args[0])])
        return wrap_value(result, kind="value", type_hint=out_ty)


class simd_bf16_dot_packed(BuiltinFunction):
    """simd_bf16_dot_packed(acc, x, w) -> AVX512 packed BF16 dot step."""

    @classmethod
    def get_name(cls):
        return "simd_bf16_dot_packed"

    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        if len(args) != 3:
            logger.error(
                "simd_bf16_dot_packed(acc, x, w) takes three arguments",
                node=node,
                exc_type=TypeError,
            )
        from .types import f32, u16

        acc_ty = _require_simd(args[0], "simd_bf16_dot_packed", node)
        x_ty = _require_simd(args[1], "simd_bf16_dot_packed", node)
        w_ty = _require_simd(args[2], "simd_bf16_dot_packed", node)
        if acc_ty.element_type is not f32 or acc_ty.lanes != 16:
            logger.error("simd_bf16_dot_packed acc must be simd[f32, 16]",
                         node=node, exc_type=TypeError)
        if x_ty.element_type is not u16 or x_ty.lanes != 32:
            logger.error("simd_bf16_dot_packed x must be simd[u16, 32]",
                         node=node, exc_type=TypeError)
        if w_ty.element_type is not u16 or w_ty.lanes != 32:
            logger.error("simd_bf16_dot_packed w must be simd[u16, 32]",
                         node=node, exc_type=TypeError)
        f32x16 = acc_ty.get_llvm_type(visitor.module.context)
        u16x32 = x_ty.get_llvm_type(visitor.module.context)
        dot = _declare(
            visitor.module,
            "llvm.x86.avx512bf16.dpbf16ps.512",
            ir.FunctionType(f32x16, [f32x16, u16x32, u16x32]),
        )
        result = visitor.builder.call(
            dot, [ensure_ir(args[0]), ensure_ir(args[1]), ensure_ir(args[2])])
        return wrap_value(result, kind="value", type_hint=acc_ty)
