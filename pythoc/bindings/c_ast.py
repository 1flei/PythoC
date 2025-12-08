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
    struct, linear, consume, assume, refined
)
from pythoc.std.refine_wrapper import nonnull_wrap
from pythoc.std.linear_wrapper import linear_wrap
from pythoc.libc.stdlib import malloc, free


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
# Metaprogramming: Generate refined types, linear alloc/free, and API
# =============================================================================

def _make_ast_memory_api():
    """Generate all memory management infrastructure via metaprogramming.
    
    For each AST type, generates:
    - nonnull predicate and refined type (e.g., ctype_nonnull, CTypeRef)
    - raw alloc/free functions
    - linear-wrapped alloc/free with proof types
    
    Returns dict with all generated items to be exported to module namespace.
    """
    # Types that need single-element allocation with linear tracking
    _linear_types = [
        ('ctype', CType),
        ('qualtype', QualType),
        ('ptrtype', PtrType),
        ('arraytype', ArrayType),
        ('functype', FuncType),
        ('structtype', StructType),
        ('enumtype', EnumType),
        ('decl', Decl),
    ]
    
    # Types that need array allocation (no linear tracking for now)
    _array_types = [
        ('paraminfo', ParamInfo),
        ('fieldinfo', FieldInfo),
        ('enumvalue', EnumValue),
    ]
    
    # All types that need nonnull refined types
    _all_types = _linear_types + _array_types
    
    exports = {}
    
    # Generate nonnull predicates and refined types
    for name, typ in _all_types:
        pred, ref_type = nonnull_wrap(ptr[typ])
        exports[f'{name}_nonnull'] = pred
        exports[f'{typ.__name__}Ref'] = ref_type
    
    # Generate raw alloc/free and linear-wrapped versions for linear types
    for name, typ in _linear_types:
        # Use unique function names to avoid symbol conflicts in linear_wrap
        type_name = typ.__name__
        
        # Generate raw alloc with unique name
        @compile(suffix=typ)
        def _alloc_raw() -> ptr[typ]:
            return ptr[typ](malloc(sizeof(typ)))
        _alloc_raw.__name__ = f'{name}_alloc_raw'
        
        # Generate raw free with unique name
        @compile(suffix=typ)
        def _free_raw(p: ptr[typ]) -> void:
            free(p)
        _free_raw.__name__ = f'{name}_free_raw'
        
        # Wrap with linear_wrap
        proof_type, alloc_fn, free_fn = linear_wrap(
            _alloc_raw, _free_raw, struct_name=f'{type_name}Proof')
        
        exports[f'{type_name}Proof'] = proof_type
        exports[f'{name}_alloc'] = alloc_fn
        exports[f'{name}_free_linear'] = free_fn
    
    # Generate array allocation functions
    for name, typ in _array_types:
        @compile(suffix=typ)
        def _array_alloc(count: i32) -> ptr[typ]:
            return ptr[typ](malloc(sizeof(typ) * count))
        exports[f'{name}_alloc'] = _array_alloc
    
    # Build alloc API class
    class AllocApi:
        """Allocation functions for C AST types
        
        Linear allocation (returns proof + ptr):
            prf, ty = alloc.ctype()
            # ... use ty ...
            ctype_free_linear(prf, ty)
        
        Array allocation:
            params = alloc.paraminfo(count)
        """
        pass
    
    for name, _ in _linear_types + _array_types:
        setattr(AllocApi, name, staticmethod(exports[f'{name}_alloc']))
    
    exports['alloc'] = AllocApi
    
    return exports


# Force resolve forward references before metaprogramming
# This ensures all types are fully resolved when linear_wrap calls get_type_id
_all_struct_types = [PtrType, ArrayType, ParamInfo, FuncType, FieldInfo, 
                     StructType, EnumValue, EnumType, QualType, Decl]
for _t in _all_struct_types:
    if hasattr(_t, '_ensure_field_types_resolved'):
        _t._ensure_field_types_resolved()

# Execute metaprogramming and export to module namespace
_ast_api = _make_ast_memory_api()
globals().update(_ast_api)

# Re-export commonly used items for IDE autocomplete
CTypeRef = _ast_api['CTypeRef']
QualTypeRef = _ast_api['QualTypeRef']
PtrTypeRef = _ast_api['PtrTypeRef']
ArrayTypeRef = _ast_api['ArrayTypeRef']
FuncTypeRef = _ast_api['FuncTypeRef']
StructTypeRef = _ast_api['StructTypeRef']
EnumTypeRef = _ast_api['EnumTypeRef']
ParamInfoRef = _ast_api['ParamInfoRef']
FieldInfoRef = _ast_api['FieldInfoRef']
EnumValueRef = _ast_api['EnumValueRef']
DeclRef = _ast_api['DeclRef']

CTypeProof = _ast_api['CTypeProof']
QualTypeProof = _ast_api['QualTypeProof']
PtrTypeProof = _ast_api['PtrTypeProof']
ArrayTypeProof = _ast_api['ArrayTypeProof']
FuncTypeProof = _ast_api['FuncTypeProof']
StructTypeProof = _ast_api['StructTypeProof']
EnumTypeProof = _ast_api['EnumTypeProof']
DeclProof = _ast_api['DeclProof']

ctype_alloc = _ast_api['ctype_alloc']
qualtype_alloc = _ast_api['qualtype_alloc']
ptrtype_alloc = _ast_api['ptrtype_alloc']
arraytype_alloc = _ast_api['arraytype_alloc']
functype_alloc = _ast_api['functype_alloc']
structtype_alloc = _ast_api['structtype_alloc']
enumtype_alloc = _ast_api['enumtype_alloc']
decl_alloc = _ast_api['decl_alloc']
paraminfo_alloc = _ast_api['paraminfo_alloc']
fieldinfo_alloc = _ast_api['fieldinfo_alloc']
enumvalue_alloc = _ast_api['enumvalue_alloc']

ctype_free_linear = _ast_api['ctype_free_linear']
qualtype_free_linear = _ast_api['qualtype_free_linear']

alloc = _ast_api['alloc']


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
    free(fields)


@compile
def free_params(params: ptr[ParamInfo], count: i32) -> void:
    """Free parameter array and their types (internal helper)"""
    i: i32 = 0
    while i < count:
        if params[i].type != nullptr:
            _qualtype_free_deep(params[i].type)
        i = i + 1
    free(params)


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
                free(pt)
        case (CType.Array, at):
            if at != nullptr:
                if at.elem != nullptr:
                    _qualtype_free_deep(at.elem)
                free(at)
        case (CType.Func, ft):
            if ft != nullptr:
                if ft.ret != nullptr:
                    _qualtype_free_deep(ft.ret)
                if ft.params != nullptr:
                    free_params(ft.params, ft.param_count)
                free(ft)
        case (CType.Struct, st):
            if st != nullptr:
                if st.fields != nullptr:
                    free_fields(st.fields, st.field_count)
                free(st)
        case (CType.Union, st):
            if st != nullptr:
                if st.fields != nullptr:
                    free_fields(st.fields, st.field_count)
                free(st)
        case (CType.Enum, et):
            if et != nullptr:
                if et.values != nullptr:
                    free(et.values)
                free(et)
        case _:
            # Typedef and primitives have no heap-allocated payload
            pass

    free(ty)


@compile
def _qualtype_free_deep(qt: ptr[QualType]) -> void:
    """Free a QualType and its underlying CType (internal)"""
    if qt == nullptr:
        return
    _ctype_free_deep(qt.type)
    free(qt)


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
        free(d)
    consume(prf)
