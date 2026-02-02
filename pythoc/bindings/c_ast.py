"""
C AST types for header parsing (pythoc compiled)

Type-Centric Design:
- CType enum with payload for all type variants
- Pointer/Array/Func are types, not declarators
- QualType wraps type + qualifiers
- All types are zero-copy, referencing source text via Span

Memory Safety:
- Linear types track ownership: alloc returns (proof, ptr), free consumes proof
- Refined types guarantee non-null: CTypeRef, QualTypeRef, etc.
- make_* functions transfer ownership of children to parent

Usage:
- Use match-case to dispatch on CType variants and extract payloads safely
- Use linear alloc/free pairs for memory management
- Refined types (e.g., CTypeRef) guarantee non-null pointers

See docs/c_ast_review.md for design rationale.
"""

from pythoc import (
    compile, i32, i64, i8, ptr, enum, sizeof, void, nullptr, bool, char,
    struct, linear, consume, assume, refined, effect
)
from pythoc.std.refine_wrapper import nonnull_wrap
from pythoc.std.linear_wrapper import linear_wrap
from pythoc.std import mem  # Sets up default mem effect


# =============================================================================
# Span: Zero-copy string reference with lifetime tracking
# =============================================================================

# SpanProof is a refined linear type that ties Span to source buffer lifetime
# To create a Span, you need a SourceProof (representing the source buffer)
# To release a Span, you must return the SourceProof, ensuring source outlives span
SpanProof = refined[linear, "SpanProof"]


@compile
class Span:
    """Zero-copy reference to source text
    
    Lifetime is enforced via linear types:
    - span_from_source(src, src_prf) -> (Span, SpanProof, SourceProof)
    - span_release(span, span_prf, src_prf) -> SourceProof
    
    This ensures Span cannot outlive the source buffer.
    For AST nodes, the SpanProof is consumed when the node is created,
    and the source buffer must outlive the entire AST.
    """
    start: ptr[i8]
    len: i32


span_nonnull, SpanRef = nonnull_wrap(ptr[Span])

# SourceProof represents ownership/lifetime of a source buffer
# When parsing, you create a SourceProof for the source buffer,
# and all Spans must be released before the source buffer can be freed
SourceProof = refined[linear, "SourceProof"]


@compile
def source_begin(source: ptr[i8]) -> SourceProof:
    """Begin a parsing session with a source buffer.
    
    Returns a SourceProof that must be held until all Spans are released.
    The source buffer must remain valid while SourceProof exists.
    """
    return assume(linear(), "SourceProof")


@compile
def source_end(src_prf: SourceProof) -> void:
    """End a parsing session, allowing source buffer to be freed.
    
    All SpanProofs must have been released before calling this.
    """
    consume(src_prf)


@compile
def span_from_cstr(start: ptr[i8], length: i32, src_prf: SourceProof) -> struct[Span, SpanProof, SourceProof]:
    """Create a Span from a C string pointer within the source buffer.
    
    Args:
        start: Pointer into the source buffer
        length: Length of the span
        src_prf: Source proof (passed through)
    
    Returns:
        span: The created Span
        span_prf: Proof that span is valid (must be released via span_release)
        src_prf: Source proof passed through
    """
    s: Span
    s.start = start
    s.len = length
    span_prf: SpanProof = assume(linear(), "SpanProof")
    return s, span_prf, src_prf


@compile
def span_release(s: Span, span_prf: SpanProof, src_prf: SourceProof) -> SourceProof:
    """Release a Span, returning the source proof.
    
    This enforces that Span cannot outlive the source buffer:
    - To release a span, you must have the source proof
    - This proves the source is still alive when the span is released
    """
    consume(span_prf)
    return src_prf


@compile
def span_empty() -> Span:
    """Create an empty span (no source reference, no proof needed)
    
    Empty spans are safe because they don't reference any source buffer.
    """
    s: Span
    s.start = nullptr
    s.len = 0
    return s


@compile
def span_is_empty(s: Span) -> bool:
    """Check if span is empty"""
    return s.len == 0


@compile
def span_eq(a: Span, b: Span) -> bool:
    """Check if two spans have equal content"""
    if a.len != b.len:
        return False
    i: i32 = 0
    while i < a.len:
        if a.start[i] != b.start[i]:
            return False
        i = i + 1
    return True


@compile
def span_eq_cstr(s: Span, cstr: ptr[i8]) -> bool:
    """Check if span equals a C string"""
    i: i32 = 0
    while i < s.len:
        if cstr[i] == 0:
            return False
        if s.start[i] != cstr[i]:
            return False
        i = i + 1
    return cstr[s.len] == 0


# =============================================================================
# Qualifier constants (bitflags)
# =============================================================================

QUAL_NONE: i8 = 0
QUAL_CONST: i8 = 1
QUAL_VOLATILE: i8 = 2
QUAL_RESTRICT: i8 = 4


# =============================================================================
# Storage class constants
# =============================================================================

STORAGE_NONE: i8 = 0
STORAGE_EXTERN: i8 = 1
STORAGE_STATIC: i8 = 2


# =============================================================================
# Forward declarations for compound type payloads
# =============================================================================

@compile
class PtrType:
    """Pointer type payload"""
    pointee: ptr['QualType']
    quals: i8


@compile
class ArrayType:
    """Array type payload"""
    elem: ptr['QualType']
    size: i32  # -1 for [], >=0 for [N]


@compile
class ParamInfo:
    """Function parameter"""
    name: Span
    type: ptr['QualType']


@compile
class FuncType:
    """Function type payload"""
    ret: ptr['QualType']
    params: ptr[ParamInfo]
    param_count: i32
    is_variadic: i8


@compile
class FieldInfo:
    """Struct/union field"""
    name: Span
    type: ptr['QualType']
    bit_width: i32  # -1 if not a bitfield


@compile
class StructType:
    """Struct or union type payload"""
    name: Span
    fields: ptr[FieldInfo]
    field_count: i32
    is_complete: i8


@compile
class EnumValue:
    """Enum constant"""
    name: Span
    value: i64
    has_explicit_value: i8


@compile
class EnumType:
    """Enum type payload"""
    name: Span
    values: ptr[EnumValue]
    value_count: i32
    is_complete: i8


# =============================================================================
# CType: The central type enum
# =============================================================================

@enum(i8)
class CType:
    """Complete C type representation as tagged union"""
    # Primitive types (no payload)
    Void: None
    Char: None
    SChar: None
    UChar: None
    Short: None
    UShort: None
    Int: None
    UInt: None
    Long: None
    ULong: None
    LongLong: None
    ULongLong: None
    Float: None
    Double: None
    LongDouble: None

    # Compound types (with payload)
    Ptr: ptr[PtrType]
    Array: ptr[ArrayType]
    Func: ptr[FuncType]
    Struct: ptr[StructType]
    Union: ptr[StructType]
    Enum: ptr[EnumType]
    Typedef: Span


# =============================================================================
# QualType: Type with qualifiers
# =============================================================================

@compile
class QualType:
    """Qualified type - the standard way to reference a type"""
    type: ptr[CType]
    quals: i8


# =============================================================================
# Top-level declarations
# =============================================================================

@enum(i8)
class DeclKind:
    """Kind of top-level declaration"""
    Func: None
    Var: None
    Typedef: None
    Struct: None
    Union: None
    Enum: None


@compile
class Decl:
    """Top-level declaration"""
    kind: DeclKind
    name: Span
    type: ptr[QualType]
    storage: i8


# =============================================================================
# Refined types (nonnull predicates and refined pointer types)
# =============================================================================

# Force resolve forward references first
_all_struct_types = [PtrType, ArrayType, ParamInfo, FuncType, FieldInfo, 
                     StructType, EnumValue, EnumType, QualType, Decl]
for _t in _all_struct_types:
    if hasattr(_t, '_ensure_field_types_resolved'):
        _t._ensure_field_types_resolved()

# Nonnull predicates and refined types for all AST types
ctype_nonnull, CTypeRef = nonnull_wrap(ptr[CType])
qualtype_nonnull, QualTypeRef = nonnull_wrap(ptr[QualType])
ptrtype_nonnull, PtrTypeRef = nonnull_wrap(ptr[PtrType])
arraytype_nonnull, ArrayTypeRef = nonnull_wrap(ptr[ArrayType])
functype_nonnull, FuncTypeRef = nonnull_wrap(ptr[FuncType])
structtype_nonnull, StructTypeRef = nonnull_wrap(ptr[StructType])
enumtype_nonnull, EnumTypeRef = nonnull_wrap(ptr[EnumType])
paraminfo_nonnull, ParamInfoRef = nonnull_wrap(ptr[ParamInfo])
fieldinfo_nonnull, FieldInfoRef = nonnull_wrap(ptr[FieldInfo])
enumvalue_nonnull, EnumValueRef = nonnull_wrap(ptr[EnumValue])
decl_nonnull, DeclRef = nonnull_wrap(ptr[Decl])


# =============================================================================
# Raw alloc/free functions for linear types
# =============================================================================

@compile
def _ctype_alloc_raw() -> ptr[CType]:
    return ptr[CType](effect.mem.malloc(sizeof(CType)))

@compile
def _ctype_free_raw(p: ptr[CType]) -> void:
    effect.mem.free(p)

@compile
def _qualtype_alloc_raw() -> ptr[QualType]:
    return ptr[QualType](effect.mem.malloc(sizeof(QualType)))

@compile
def _qualtype_free_raw(p: ptr[QualType]) -> void:
    effect.mem.free(p)

@compile
def _ptrtype_alloc_raw() -> ptr[PtrType]:
    return ptr[PtrType](effect.mem.malloc(sizeof(PtrType)))

@compile
def _ptrtype_free_raw(p: ptr[PtrType]) -> void:
    effect.mem.free(p)

@compile
def _arraytype_alloc_raw() -> ptr[ArrayType]:
    return ptr[ArrayType](effect.mem.malloc(sizeof(ArrayType)))

@compile
def _arraytype_free_raw(p: ptr[ArrayType]) -> void:
    effect.mem.free(p)

@compile
def _functype_alloc_raw() -> ptr[FuncType]:
    return ptr[FuncType](effect.mem.malloc(sizeof(FuncType)))

@compile
def _functype_free_raw(p: ptr[FuncType]) -> void:
    effect.mem.free(p)

@compile
def _structtype_alloc_raw() -> ptr[StructType]:
    return ptr[StructType](effect.mem.malloc(sizeof(StructType)))

@compile
def _structtype_free_raw(p: ptr[StructType]) -> void:
    effect.mem.free(p)

@compile
def _enumtype_alloc_raw() -> ptr[EnumType]:
    return ptr[EnumType](effect.mem.malloc(sizeof(EnumType)))

@compile
def _enumtype_free_raw(p: ptr[EnumType]) -> void:
    effect.mem.free(p)

@compile
def _decl_alloc_raw() -> ptr[Decl]:
    return ptr[Decl](effect.mem.malloc(sizeof(Decl)))

@compile
def _decl_free_raw(p: ptr[Decl]) -> void:
    effect.mem.free(p)


# =============================================================================
# Linear-wrapped alloc/free with proof types
# =============================================================================

CTypeProof, ctype_alloc, ctype_free_linear = linear_wrap(
    _ctype_alloc_raw, _ctype_free_raw, struct_name='CTypeProof')

QualTypeProof, qualtype_alloc, qualtype_free_linear = linear_wrap(
    _qualtype_alloc_raw, _qualtype_free_raw, struct_name='QualTypeProof')

PtrTypeProof, ptrtype_alloc, ptrtype_free_linear = linear_wrap(
    _ptrtype_alloc_raw, _ptrtype_free_raw, struct_name='PtrTypeProof')

ArrayTypeProof, arraytype_alloc, arraytype_free_linear = linear_wrap(
    _arraytype_alloc_raw, _arraytype_free_raw, struct_name='ArrayTypeProof')

FuncTypeProof, functype_alloc, functype_free_linear = linear_wrap(
    _functype_alloc_raw, _functype_free_raw, struct_name='FuncTypeProof')

StructTypeProof, structtype_alloc, structtype_free_linear = linear_wrap(
    _structtype_alloc_raw, _structtype_free_raw, struct_name='StructTypeProof')

EnumTypeProof, enumtype_alloc, enumtype_free_linear = linear_wrap(
    _enumtype_alloc_raw, _enumtype_free_raw, struct_name='EnumTypeProof')

DeclProof, decl_alloc, decl_free_linear = linear_wrap(
    _decl_alloc_raw, _decl_free_raw, struct_name='DeclProof')


# =============================================================================
# Array allocation functions (no linear tracking)
# =============================================================================

@compile
def paraminfo_alloc(count: i32) -> ptr[ParamInfo]:
    return ptr[ParamInfo](effect.mem.malloc(sizeof(ParamInfo) * count))

@compile
def fieldinfo_alloc(count: i32) -> ptr[FieldInfo]:
    return ptr[FieldInfo](effect.mem.malloc(sizeof(FieldInfo) * count))

@compile
def enumvalue_alloc(count: i32) -> ptr[EnumValue]:
    return ptr[EnumValue](effect.mem.malloc(sizeof(EnumValue) * count))


# =============================================================================
# Type construction helpers (transfer ownership)
# =============================================================================

@compile
def make_qualtype(ty_prf: CTypeProof, ty: ptr[CType], quals: i8) -> struct[QualTypeProof, ptr[QualType]]:
    """Create a QualType wrapping a CType.
    
    Ownership: Takes ownership of ty (consumes ty_prf), returns ownership of QualType.
    """
    qt_prf, qt = qualtype_alloc()
    qt.type = ty
    qt.quals = quals
    # Transfer CType ownership into QualType - consume the proof
    consume(ty_prf)
    return qt_prf, qt


# Primitive type constructors (no children, simple ownership)
def _make_primitive_api():
    """Factory that generates primitive type constructors.
    
    Returns a PrimitiveApi class with methods like:
      - prim.void() -> struct[CTypeProof, ptr[CType]]
      - prim.int() -> struct[CTypeProof, ptr[CType]]
    """
    _types = [
        ('void', CType.Void),
        ('char', CType.Char),
        ('schar', CType.SChar),
        ('uchar', CType.UChar),
        ('short', CType.Short),
        ('ushort', CType.UShort),
        ('int', CType.Int),
        ('uint', CType.UInt),
        ('long', CType.Long),
        ('ulong', CType.ULong),
        ('longlong', CType.LongLong),
        ('ulonglong', CType.ULongLong),
        ('float', CType.Float),
        ('double', CType.Double),
        ('longdouble', CType.LongDouble),
    ]
    
    class PrimitiveApi:
        """Primitive type constructors"""
        pass
    
    for _name, _tag in _types:
        @compile(suffix=_tag)
        def _make_prim() -> struct[CTypeProof, ptr[CType]]:
            prf, ty = ctype_alloc()
            ty[0] = CType(_tag)
            return prf, ty
        setattr(PrimitiveApi, _name, staticmethod(_make_prim))
    
    return PrimitiveApi


# Create the primitive type API
prim = _make_primitive_api()

# Convenience aliases
make_void_type = prim.void
make_char_type = prim.char
make_schar_type = prim.schar
make_uchar_type = prim.uchar
make_short_type = prim.short
make_ushort_type = prim.ushort
make_int_type = prim.int
make_uint_type = prim.uint
make_long_type = prim.long
make_ulong_type = prim.ulong
make_longlong_type = prim.longlong
make_ulonglong_type = prim.ulonglong
make_float_type = prim.float
make_double_type = prim.double
make_longdouble_type = prim.longdouble


@compile
def make_ptr_type(pointee_prf: QualTypeProof, pointee: ptr[QualType], 
                  ptr_quals: i8) -> struct[CTypeProof, ptr[CType]]:
    """Create a pointer type.
    
    Ownership: Takes ownership of pointee (consumes pointee_prf).
    """
    pt_prf, pt = ptrtype_alloc()
    pt.pointee = pointee
    pt.quals = ptr_quals
    
    ty_prf, ty = ctype_alloc()
    ty[0] = CType(CType.Ptr, pt)
    
    # Transfer ownership: PtrType now owns pointee, CType owns PtrType
    consume(pointee_prf)
    consume(pt_prf)
    return ty_prf, ty


@compile
def make_array_type(elem_prf: QualTypeProof, elem: ptr[QualType], 
                    size: i32) -> struct[CTypeProof, ptr[CType]]:
    """Create an array type. size=-1 for unsized array [].
    
    Ownership: Takes ownership of elem (consumes elem_prf).
    """
    at_prf, at = arraytype_alloc()
    at.elem = elem
    at.size = size
    
    ty_prf, ty = ctype_alloc()
    ty[0] = CType(CType.Array, at)
    
    consume(elem_prf)
    consume(at_prf)
    return ty_prf, ty


@compile
def make_func_type(ret_prf: QualTypeProof, ret: ptr[QualType], 
                   params: ptr[ParamInfo], param_count: i32, 
                   is_variadic: i8) -> struct[CTypeProof, ptr[CType]]:
    """Create a function type.
    
    Ownership: Takes ownership of ret and params array.
    Note: params array ownership is transferred, caller should not free it.
    """
    ft_prf, ft = functype_alloc()
    ft.ret = ret
    ft.params = params
    ft.param_count = param_count
    ft.is_variadic = is_variadic
    
    ty_prf, ty = ctype_alloc()
    ty[0] = CType(CType.Func, ft)
    
    consume(ret_prf)
    consume(ft_prf)
    return ty_prf, ty


@compile
def make_struct_type(name: Span, fields: ptr[FieldInfo], field_count: i32, 
                     is_complete: i8) -> struct[CTypeProof, ptr[CType]]:
    """Create a struct type.
    
    Ownership: Takes ownership of fields array.
    """
    st_prf, st = structtype_alloc()
    st.name = name
    st.fields = fields
    st.field_count = field_count
    st.is_complete = is_complete
    
    ty_prf, ty = ctype_alloc()
    ty[0] = CType(CType.Struct, st)
    
    consume(st_prf)
    return ty_prf, ty


@compile
def make_union_type(name: Span, fields: ptr[FieldInfo], field_count: i32, 
                    is_complete: i8) -> struct[CTypeProof, ptr[CType]]:
    """Create a union type.
    
    Ownership: Takes ownership of fields array.
    """
    st_prf, st = structtype_alloc()
    st.name = name
    st.fields = fields
    st.field_count = field_count
    st.is_complete = is_complete
    
    ty_prf, ty = ctype_alloc()
    ty[0] = CType(CType.Union, st)
    
    consume(st_prf)
    return ty_prf, ty


@compile
def make_enum_type(name: Span, values: ptr[EnumValue], value_count: i32, 
                   is_complete: i8) -> struct[CTypeProof, ptr[CType]]:
    """Create an enum type.
    
    Ownership: Takes ownership of values array.
    """
    et_prf, et = enumtype_alloc()
    et.name = name
    et.values = values
    et.value_count = value_count
    et.is_complete = is_complete
    
    ty_prf, ty = ctype_alloc()
    ty[0] = CType(CType.Enum, et)
    
    consume(et_prf)
    return ty_prf, ty


@compile
def make_typedef_type(name: Span) -> struct[CTypeProof, ptr[CType]]:
    """Create a typedef reference type."""
    ty_prf, ty = ctype_alloc()
    ty[0] = CType(CType.Typedef, name)
    return ty_prf, ty


# =============================================================================
# Free functions (recursive, consumes proof)
# =============================================================================

@compile
def free_fields(fields: ptr[FieldInfo], count: i32) -> void:
    """Free field array and their types (internal helper)"""
    i: i32 = 0
    while i < count:
        if fields[i].type != nullptr:
            # Note: We can't track individual field type proofs here,
            # so we use raw free. The parent proof covers this.
            _qualtype_free_deep(fields[i].type)
        i = i + 1
    effect.mem.free(fields)


@compile
def free_params(params: ptr[ParamInfo], count: i32) -> void:
    """Free parameter array and their types (internal helper)"""
    i: i32 = 0
    while i < count:
        if params[i].type != nullptr:
            _qualtype_free_deep(params[i].type)
        i = i + 1
    effect.mem.free(params)


@compile
def _ctype_free_deep(ty: ptr[CType]) -> void:
    """Recursively free a CType and all its children (internal).
    
    Uses match-case to safely dispatch on CType variants.
    """
    if ty == nullptr:
        return

    match ty[0]:
        case (CType.Ptr, pt):
            if pt != nullptr:
                if pt.pointee != nullptr:
                    _qualtype_free_deep(pt.pointee)
                effect.mem.free(pt)
        case (CType.Array, at):
            if at != nullptr:
                if at.elem != nullptr:
                    _qualtype_free_deep(at.elem)
                effect.mem.free(at)
        case (CType.Func, ft):
            if ft != nullptr:
                if ft.ret != nullptr:
                    _qualtype_free_deep(ft.ret)
                if ft.params != nullptr:
                    free_params(ft.params, ft.param_count)
                effect.mem.free(ft)
        case (CType.Struct, st):
            if st != nullptr:
                if st.fields != nullptr:
                    free_fields(st.fields, st.field_count)
                effect.mem.free(st)
        case (CType.Union, st):
            if st != nullptr:
                if st.fields != nullptr:
                    free_fields(st.fields, st.field_count)
                effect.mem.free(st)
        case (CType.Enum, et):
            if et != nullptr:
                if et.values != nullptr:
                    effect.mem.free(et.values)
                effect.mem.free(et)
        case _:
            # Typedef and primitives have no heap-allocated payload
            pass

    effect.mem.free(ty)


@compile
def _qualtype_free_deep(qt: ptr[QualType]) -> void:
    """Free a QualType and its underlying CType (internal)"""
    if qt == nullptr:
        return
    _ctype_free_deep(qt.type)
    effect.mem.free(qt)


@compile
def ctype_free(prf: CTypeProof, ty: ptr[CType]) -> void:
    """Free a CType tree, consuming the ownership proof.
    
    This is the safe API - proof ensures you own the memory.
    """
    _ctype_free_deep(ty)
    consume(prf)


@compile
def qualtype_free(prf: QualTypeProof, qt: ptr[QualType]) -> void:
    """Free a QualType and its underlying CType, consuming proof."""
    _qualtype_free_deep(qt)
    consume(prf)


@compile
def decl_free(prf: DeclProof, d: ptr[Decl]) -> void:
    """Free a declaration and its type, consuming proof."""
    if d != nullptr:
        _qualtype_free_deep(d.type)
        effect.mem.free(d)
    consume(prf)


# =============================================================================
# Shallow clone for simple types (primitives, typedef)
# =============================================================================

@compile
def qualtype_clone_shallow(qt: ptr[QualType]) -> struct[QualTypeProof, ptr[QualType]]:
    """Clone a QualType with shallow copy of CType.

    This is safe for:
    - Primitive types (no payload to share)
    - Typedef (Span is a view, not owned)

    NOT safe for Ptr/Array/Func/Struct/Union/Enum with heap payloads.
    Use this only when you know the underlying CType has no heap children.

    Returns (proof, cloned_ptr). Caller takes ownership via proof.
    """
    new_qt_prf, new_qt = qualtype_alloc()
    new_qt.quals = qt.quals

    # Allocate new CType and copy the tag (shallow)
    new_ty_prf, new_ty = ctype_alloc()
    new_ty[0] = qt.type[0]  # Copy enum tag + payload
    consume(new_ty_prf)

    new_qt.type = new_ty
    return new_qt_prf, new_qt


# =============================================================================
# Deep clone for ownership safety
# =============================================================================

@compile
def _clone_params_deep(params: ptr[ParamInfo], count: i32) -> ptr[ParamInfo]:
    """Deep clone ParamInfo array (names + deep-cloned types)."""
    if params == nullptr or count <= 0:
        return nullptr

    out: ptr[ParamInfo] = paraminfo_alloc(count)
    i: i32 = 0
    while i < count:
        out[i].name = params[i].name
        if params[i].type != nullptr:
            qt_prf, qt = qualtype_clone_deep(params[i].type)
            out[i].type = qt
            consume(qt_prf)
        else:
            out[i].type = nullptr
        i = i + 1

    return out


@compile
def _clone_fields_deep(fields: ptr[FieldInfo], count: i32) -> ptr[FieldInfo]:
    """Deep clone FieldInfo array (names/bit_width + deep-cloned types)."""
    if fields == nullptr or count <= 0:
        return nullptr

    out: ptr[FieldInfo] = fieldinfo_alloc(count)
    i: i32 = 0
    while i < count:
        out[i].name = fields[i].name
        out[i].bit_width = fields[i].bit_width
        if fields[i].type != nullptr:
            qt_prf, qt = qualtype_clone_deep(fields[i].type)
            out[i].type = qt
            consume(qt_prf)
        else:
            out[i].type = nullptr
        i = i + 1

    return out


@compile
def ctype_clone_deep(ty: ptr[CType]) -> struct[CTypeProof, ptr[CType]]:
    """Deep clone a CType tree.

    The returned CType has independent ownership and can be deep-freed safely.
    """
    if ty == nullptr:
        return prim.void()

    match ty[0]:
        case (CType.Ptr, pt):
            if pt != nullptr and pt.pointee != nullptr:
                pointee_prf, pointee = qualtype_clone_deep(pt.pointee)
                return make_ptr_type(pointee_prf, pointee, pt.quals)
            base_prf, base_ty = prim.void()
            base_qt_prf, base_qt = make_qualtype(base_prf, base_ty, QUAL_NONE)
            return make_ptr_type(base_qt_prf, base_qt, QUAL_NONE)

        case (CType.Array, at):
            if at != nullptr and at.elem != nullptr:
                elem_prf, elem = qualtype_clone_deep(at.elem)
                return make_array_type(elem_prf, elem, at.size)
            base_prf, base_ty = prim.void()
            base_qt_prf, base_qt = make_qualtype(base_prf, base_ty, QUAL_NONE)
            return make_array_type(base_qt_prf, base_qt, -1)

        case (CType.Func, ft):
            ret_prf: QualTypeProof
            ret: ptr[QualType]
            if ft != nullptr and ft.ret != nullptr:
                ret_prf, ret = qualtype_clone_deep(ft.ret)
            else:
                base_prf, base_ty = prim.void()
                ret_prf, ret = make_qualtype(base_prf, base_ty, QUAL_NONE)

            new_params: ptr[ParamInfo] = nullptr
            param_count: i32 = 0
            is_variadic: i8 = 0
            if ft != nullptr:
                param_count = ft.param_count
                is_variadic = ft.is_variadic
                if ft.params != nullptr and ft.param_count > 0:
                    new_params = _clone_params_deep(ft.params, ft.param_count)

            return make_func_type(ret_prf, ret, new_params, param_count, is_variadic)

        case (CType.Struct, st):
            if st != nullptr:
                new_fields: ptr[FieldInfo] = _clone_fields_deep(st.fields, st.field_count)
                return make_struct_type(st.name, new_fields, st.field_count, st.is_complete)
            return make_struct_type(span_empty(), nullptr, 0, 0)

        case (CType.Union, st):
            if st != nullptr:
                new_fields: ptr[FieldInfo] = _clone_fields_deep(st.fields, st.field_count)
                return make_union_type(st.name, new_fields, st.field_count, st.is_complete)
            return make_union_type(span_empty(), nullptr, 0, 0)

        case (CType.Enum, et):
            if et != nullptr and et.values != nullptr and et.value_count > 0:
                out_vals: ptr[EnumValue] = enumvalue_alloc(et.value_count)
                i: i32 = 0
                while i < et.value_count:
                    out_vals[i].name = et.values[i].name
                    out_vals[i].value = et.values[i].value
                    out_vals[i].has_explicit_value = et.values[i].has_explicit_value
                    i = i + 1
                return make_enum_type(et.name, out_vals, et.value_count, et.is_complete)
            return make_enum_type(span_empty(), nullptr, 0, 0)

        case _:
            prf, out = ctype_alloc()
            out[0] = ty[0]
            return prf, out


@compile
def qualtype_clone_deep(qt: ptr[QualType]) -> struct[QualTypeProof, ptr[QualType]]:
    """Deep clone a QualType and its owned CType tree."""
    if qt == nullptr:
        base_prf, base_ty = prim.void()
        return make_qualtype(base_prf, base_ty, QUAL_NONE)

    ty_prf, ty = ctype_clone_deep(qt.type)
    return make_qualtype(ty_prf, ty, qt.quals)
