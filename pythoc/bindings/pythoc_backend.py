"""
Pythoc Backend - Generate pythoc code from C AST (pythoc compiled)

This module provides code generation from C header AST to pythoc source code.
It translates C type declarations, structs, enums, and function signatures
into equivalent pythoc code.

Design:
- All code generation functions are @compile decorated
- Uses StringBuffer for dynamic string building
- Generates @compile decorated classes for structs/unions
- Generates @enum decorated classes for enums
- Generates @extern decorated function declarations
- Type mapping from C primitives to pythoc types

Usage:
    from pythoc.bindings.pythoc_backend import (
        StringBuffer, strbuf_init, strbuf_destroy, strbuf_to_cstr,
        emit_module_header, emit_decl
    )
    from pythoc.bindings.c_parser import parse_declarations
    from pythoc.bindings.c_ast import decl_free
    
    @compile
    def generate_bindings(source: ptr[i8]) -> ptr[i8]:
        buf: StringBuffer
        strbuf_init(ptr(buf))
        
        emit_module_header(ptr(buf))
        for decl_prf, decl in parse_declarations(source):
            emit_decl(ptr(buf), decl)
            decl_free(decl_prf, decl)
        
        result: ptr[i8] = strbuf_to_cstr(ptr(buf))
        # Note: caller should copy result before destroying buf
        strbuf_destroy(ptr(buf))
        return result
"""

from pythoc import (
    compile, inline, i32, i64, i8, bool, ptr, array, nullptr, sizeof, void,
    char, refine, assume, struct, consume, linear
)
from pythoc.libc.stdlib import malloc, free, realloc
from pythoc.libc.string import memcpy, strlen, strcpy, strcat
from pythoc.std.vector import Vector

from pythoc.bindings.c_ast import (
    Span, span_empty, span_is_empty,
    CType, QualType, PtrType, ArrayType, FuncType,
    StructType, EnumType, EnumValue, FieldInfo, ParamInfo,
    Decl, DeclKind,
    CTypeRef, QualTypeRef, StructTypeRef, EnumTypeRef,
    DeclRef, decl_nonnull, decl_free,
    QUAL_NONE, QUAL_CONST, QUAL_VOLATILE,
    STORAGE_NONE, STORAGE_EXTERN, STORAGE_STATIC,
)


# =============================================================================
# StringBuffer - Dynamic string builder using Vector
# =============================================================================

_CharVec = Vector(i8, inline_capacity=256)
StringBuffer = _CharVec.type

# Export Vector API as module-level functions
strbuf_init = _CharVec.init
strbuf_destroy = _CharVec.destroy
strbuf_size = _CharVec.size
_strbuf_push_back = _CharVec.push_back
_strbuf_get = _CharVec.get


@compile
def strbuf_data(buf: ptr[StringBuffer]) -> ptr[i8]:
    """Get pointer to string data (not null-terminated)"""
    if buf.size < 256:
        return ptr(buf.storage.inline_buffer[0])
    else:
        return buf.storage.heap_data.heap_buf


@compile
def strbuf_push_char(buf: ptr[StringBuffer], c: i8) -> void:
    """Append a single character"""
    _strbuf_push_back(buf, c)


@compile
def strbuf_push_cstr(buf: ptr[StringBuffer], s: ptr[i8]) -> void:
    """Append a null-terminated C string"""
    i: i64 = 0
    while s[i] != 0:
        _strbuf_push_back(buf, s[i])
        i = i + 1


@compile
def strbuf_push_span(buf: ptr[StringBuffer], s: Span) -> void:
    """Append a Span"""
    i: i32 = 0
    while i < s.len:
        _strbuf_push_back(buf, s.start[i])
        i = i + 1


@compile
def strbuf_push_i32(buf: ptr[StringBuffer], val: i32) -> void:
    """Append an i32 as decimal string"""
    if val < 0:
        _strbuf_push_back(buf, 45)  # '-'
        val = -val
    if val == 0:
        _strbuf_push_back(buf, 48)  # '0'
        return
    # Reverse digits into temp buffer
    digits: array[i8, 12]
    count: i32 = 0
    while val > 0:
        digits[count] = i8(48 + (val % 10))
        val = val / 10
        count = count + 1
    # Push in reverse order
    while count > 0:
        count = count - 1
        _strbuf_push_back(buf, digits[count])


@compile
def strbuf_push_i64(buf: ptr[StringBuffer], val: i64) -> void:
    """Append an i64 as decimal string"""
    if val < 0:
        _strbuf_push_back(buf, 45)  # '-'
        val = -val
    if val == 0:
        _strbuf_push_back(buf, 48)  # '0'
        return
    digits: array[i8, 24]
    count: i32 = 0
    while val > 0:
        digits[count] = i8(48 + i32(val % 10))
        val = val / 10
        count = count + 1
    while count > 0:
        count = count - 1
        _strbuf_push_back(buf, digits[count])


@compile
def strbuf_push_newline(buf: ptr[StringBuffer]) -> void:
    """Append a newline"""
    _strbuf_push_back(buf, 10)  # '\n'


@compile
def strbuf_push_indent(buf: ptr[StringBuffer], level: i32) -> void:
    """Append indentation (4 spaces per level)"""
    i: i32 = 0
    while i < level * 4:
        _strbuf_push_back(buf, 32)  # ' '
        i = i + 1


@compile
def strbuf_null_terminate(buf: ptr[StringBuffer]) -> void:
    """Add null terminator (for C string compatibility)"""
    _strbuf_push_back(buf, 0)


@compile
def strbuf_to_cstr(buf: ptr[StringBuffer]) -> ptr[i8]:
    """Get null-terminated C string (adds terminator if needed)"""
    sz: i64 = strbuf_size(buf)
    if sz == 0:
        _strbuf_push_back(buf, 0)
        return strbuf_data(buf)
    last: i8 = _strbuf_get(buf, sz - 1)
    if last != 0:
        _strbuf_push_back(buf, 0)
    return strbuf_data(buf)


# =============================================================================
# Type emission - recursive type to string
# =============================================================================

@compile
def emit_qualtype(buf: ptr[StringBuffer], qt: ptr[QualType]) -> void:
    """Emit a QualType to the buffer"""
    if qt == nullptr:
        strbuf_push_cstr(buf, "void")
        return
    emit_ctype(buf, qt.type)


@compile
def emit_ctype(buf: ptr[StringBuffer], ty: ptr[CType]) -> void:
    """Emit a CType to the buffer"""
    if ty == nullptr:
        strbuf_push_cstr(buf, "void")
        return
    
    match ty[0]:
        # Primitive types
        case CType.Void:
            strbuf_push_cstr(buf, "void")
        case CType.Char:
            strbuf_push_cstr(buf, "char")
        case CType.SChar:
            strbuf_push_cstr(buf, "i8")
        case CType.UChar:
            strbuf_push_cstr(buf, "u8")
        case CType.Short:
            strbuf_push_cstr(buf, "i16")
        case CType.UShort:
            strbuf_push_cstr(buf, "u16")
        case CType.Int:
            strbuf_push_cstr(buf, "i32")
        case CType.UInt:
            strbuf_push_cstr(buf, "u32")
        case CType.Long:
            strbuf_push_cstr(buf, "i64")
        case CType.ULong:
            strbuf_push_cstr(buf, "u64")
        case CType.LongLong:
            strbuf_push_cstr(buf, "i64")
        case CType.ULongLong:
            strbuf_push_cstr(buf, "u64")
        case CType.Float:
            strbuf_push_cstr(buf, "f32")
        case CType.Double:
            strbuf_push_cstr(buf, "f64")
        case CType.LongDouble:
            strbuf_push_cstr(buf, "f64")
        
        # Pointer type: ptr[T]
        case (CType.Ptr, pt):
            strbuf_push_cstr(buf, "ptr[")
            if pt != nullptr and pt.pointee != nullptr:
                emit_qualtype(buf, pt.pointee)
            else:
                strbuf_push_cstr(buf, "void")
            strbuf_push_char(buf, 93)  # ']'
        
        # Array type: array[T, N] or ptr[T] for unsized
        case (CType.Array, at):
            if at != nullptr:
                if at.size < 0:
                    # Unsized array -> ptr
                    strbuf_push_cstr(buf, "ptr[")
                    if at.elem != nullptr:
                        emit_qualtype(buf, at.elem)
                    else:
                        strbuf_push_cstr(buf, "void")
                    strbuf_push_char(buf, 93)  # ']'
                else:
                    strbuf_push_cstr(buf, "array[")
                    if at.elem != nullptr:
                        emit_qualtype(buf, at.elem)
                    else:
                        strbuf_push_cstr(buf, "void")
                    strbuf_push_cstr(buf, ", ")
                    strbuf_push_i32(buf, at.size)
                    strbuf_push_char(buf, 93)  # ']'
            else:
                strbuf_push_cstr(buf, "ptr[void]")
        
        # Function type - emit return type only for now
        case (CType.Func, ft):
            if ft != nullptr and ft.ret != nullptr:
                emit_qualtype(buf, ft.ret)
            else:
                strbuf_push_cstr(buf, "void")
        
        # Struct type
        case (CType.Struct, st):
            if st != nullptr and not span_is_empty(st.name):
                strbuf_push_span(buf, st.name)
            else:
                strbuf_push_cstr(buf, "_AnonymousStruct")
        
        # Union type
        case (CType.Union, st):
            if st != nullptr and not span_is_empty(st.name):
                strbuf_push_span(buf, st.name)
            else:
                strbuf_push_cstr(buf, "_AnonymousUnion")
        
        # Enum type
        case (CType.Enum, et):
            if et != nullptr and not span_is_empty(et.name):
                strbuf_push_span(buf, et.name)
            else:
                strbuf_push_cstr(buf, "i32")
        
        # Typedef reference
        case (CType.Typedef, name):
            if not span_is_empty(name):
                strbuf_push_span(buf, name)
            else:
                strbuf_push_cstr(buf, "i32")
        
        case _:
            strbuf_push_cstr(buf, "i32")


# =============================================================================
# Declaration emission
# =============================================================================

@compile
def emit_struct_decl(buf: ptr[StringBuffer], decl: ptr[Decl]) -> void:
    """Emit a struct declaration as @compile class"""
    if decl == nullptr or decl.type == nullptr:
        return
    
    ty: ptr[CType] = decl.type.type
    if ty == nullptr:
        return
    
    # Extract StructType payload
    st: ptr[StructType] = nullptr
    match ty[0]:
        case (CType.Struct, s):
            st = s
        case _:
            return
    
    if st == nullptr:
        return
    
    # @compile
    strbuf_push_cstr(buf, "@compile\n")
    
    # class Name:
    strbuf_push_cstr(buf, "class ")
    strbuf_push_span(buf, decl.name)
    strbuf_push_cstr(buf, ":\n")
    
    # Fields
    if st.field_count == 0:
        strbuf_push_indent(buf, 1)
        strbuf_push_cstr(buf, "pass\n")
    else:
        i: i32 = 0
        while i < st.field_count:
            field: ptr[FieldInfo] = ptr(st.fields[i])
            strbuf_push_indent(buf, 1)
            if not span_is_empty(field.name):
                strbuf_push_span(buf, field.name)
            else:
                strbuf_push_cstr(buf, "_field")
                strbuf_push_i32(buf, i)
            strbuf_push_cstr(buf, ": ")
            emit_qualtype(buf, field.type)
            strbuf_push_newline(buf)
            i = i + 1
    
    strbuf_push_newline(buf)


@compile
def emit_union_decl(buf: ptr[StringBuffer], decl: ptr[Decl]) -> void:
    """Emit a union declaration as pythoc union type"""
    if decl == nullptr or decl.type == nullptr:
        return
    
    ty: ptr[CType] = decl.type.type
    if ty == nullptr:
        return
    
    st: ptr[StructType] = nullptr
    match ty[0]:
        case (CType.Union, s):
            st = s
        case _:
            return
    
    if st == nullptr:
        return
    
    # Union as type alias: Name = union[field1: T1, field2: T2, ...]
    strbuf_push_span(buf, decl.name)
    strbuf_push_cstr(buf, " = union[")
    
    i: i32 = 0
    while i < st.field_count:
        if i > 0:
            strbuf_push_cstr(buf, ", ")
        field: ptr[FieldInfo] = ptr(st.fields[i])
        if not span_is_empty(field.name):
            strbuf_push_span(buf, field.name)
        else:
            strbuf_push_cstr(buf, "_field")
            strbuf_push_i32(buf, i)
        strbuf_push_cstr(buf, ": ")
        emit_qualtype(buf, field.type)
        i = i + 1
    
    strbuf_push_cstr(buf, "]\n\n")


@compile
def emit_enum_decl(buf: ptr[StringBuffer], decl: ptr[Decl]) -> void:
    """Emit an enum declaration as @enum class"""
    if decl == nullptr or decl.type == nullptr:
        return
    
    ty: ptr[CType] = decl.type.type
    if ty == nullptr:
        return
    
    et: ptr[EnumType] = nullptr
    match ty[0]:
        case (CType.Enum, e):
            et = e
        case _:
            return
    
    if et == nullptr:
        return
    
    # @enum(i32)
    strbuf_push_cstr(buf, "@enum(i32)\n")
    
    # class Name:
    strbuf_push_cstr(buf, "class ")
    strbuf_push_span(buf, decl.name)
    strbuf_push_cstr(buf, ":\n")
    
    # Enum values
    if et.value_count == 0:
        strbuf_push_indent(buf, 1)
        strbuf_push_cstr(buf, "pass\n")
    else:
        i: i32 = 0
        while i < et.value_count:
            ev: ptr[EnumValue] = ptr(et.values[i])
            strbuf_push_indent(buf, 1)
            strbuf_push_span(buf, ev.name)
            if ev.has_explicit_value != 0:
                strbuf_push_cstr(buf, " = ")
                strbuf_push_i64(buf, ev.value)
            else:
                strbuf_push_cstr(buf, ": None")
            strbuf_push_newline(buf)
            i = i + 1
    
    strbuf_push_newline(buf)


@compile
def emit_func_decl(buf: ptr[StringBuffer], decl: ptr[Decl]) -> void:
    """Emit a function declaration as @extern def"""
    if decl == nullptr or decl.type == nullptr:
        return
    
    ty: ptr[CType] = decl.type.type
    if ty == nullptr:
        return
    
    ft: ptr[FuncType] = nullptr
    match ty[0]:
        case (CType.Func, f):
            ft = f
        case _:
            return
    
    if ft == nullptr:
        return
    
    # @extern(lib='c')
    strbuf_push_cstr(buf, "@extern(lib='c')\n")
    
    # def name(params) -> ret:
    strbuf_push_cstr(buf, "def ")
    strbuf_push_span(buf, decl.name)
    strbuf_push_char(buf, 40)  # '('
    
    # Parameters
    i: i32 = 0
    while i < ft.param_count:
        if i > 0:
            strbuf_push_cstr(buf, ", ")
        param: ptr[ParamInfo] = ptr(ft.params[i])
        if not span_is_empty(param.name):
            strbuf_push_span(buf, param.name)
        else:
            strbuf_push_cstr(buf, "arg")
            strbuf_push_i32(buf, i)
        strbuf_push_cstr(buf, ": ")
        emit_qualtype(buf, param.type)
        i = i + 1
    
    # Variadic
    if ft.is_variadic != 0:
        if ft.param_count > 0:
            strbuf_push_cstr(buf, ", ")
        strbuf_push_cstr(buf, "*args")
    
    strbuf_push_cstr(buf, ") -> ")
    
    # Return type
    emit_qualtype(buf, ft.ret)
    
    strbuf_push_cstr(buf, ":\n")
    strbuf_push_indent(buf, 1)
    strbuf_push_cstr(buf, "pass\n\n")


@compile
def emit_decl(buf: ptr[StringBuffer], decl: ptr[Decl]) -> void:
    """Emit any declaration to the buffer"""
    if decl == nullptr:
        return
    
    match decl.kind:
        case DeclKind.Struct:
            emit_struct_decl(buf, decl)
        case DeclKind.Union:
            emit_union_decl(buf, decl)
        case DeclKind.Enum:
            emit_enum_decl(buf, decl)
        case DeclKind.Func:
            emit_func_decl(buf, decl)
        case _:
            pass


# =============================================================================
# Module header emission
# =============================================================================

@compile
def emit_module_header(buf: ptr[StringBuffer]) -> void:
    """Emit standard pythoc module header with imports"""
    strbuf_push_cstr(buf, '"""Auto-generated pythoc bindings"""\n\n')
    strbuf_push_cstr(buf, "from pythoc import (\n")
    strbuf_push_indent(buf, 1)
    strbuf_push_cstr(buf, "compile, extern, enum, i8, i16, i32, i64,\n")
    strbuf_push_indent(buf, 1)
    strbuf_push_cstr(buf, "u8, u16, u32, u64, f32, f64, ptr, array,\n")
    strbuf_push_indent(buf, 1)
    strbuf_push_cstr(buf, "void, char, nullptr, sizeof, struct, union\n")
    strbuf_push_cstr(buf, ")\n\n")


# =============================================================================
# High-level API
# =============================================================================

# Note: generate_bindings is not provided as a compiled function because
# inlining parse_declarations from a different module causes linear type
# tracking issues. Instead, users should compose the functions directly:
#
# Usage:
#     from pythoc.bindings.pythoc_backend import (
#         StringBuffer, strbuf_init, strbuf_destroy, strbuf_to_cstr,
#         emit_module_header, emit_decl
#     )
#     from pythoc.bindings.c_parser import parse_declarations
#     from pythoc.bindings.c_ast import decl_free
#
#     @compile
#     def my_generate_bindings(source: ptr[i8], buf: ptr[StringBuffer]) -> void:
#         emit_module_header(buf)
#         for decl_prf, decl in parse_declarations(source):
#             emit_decl(buf, decl)
#             decl_free(decl_prf, decl)
