"""
C Header Parser (pythoc compiled)

Parses C header files using the compiled lexer.
Builds AST nodes using the type-centric c_ast module.

Design:
- All parsing functions are @compile decorated
- Uses Token stream from lexer (zero-copy)
- Builds CType (tagged union), QualType, StructType, etc.
- Uses Span for zero-copy string references
- Uses linear types for memory ownership tracking
- Uses Python metaprogramming for token matching
"""

from pythoc import (
    compile, inline, i32, i64, i8, bool, ptr, array, nullptr, sizeof, void,
    char, refine, assume, struct, consume, linear
)
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import memcpy
from pythoc.std.refine_wrapper import nonnull_wrap

from pythoc.bindings.c_token import Token, TokenType, TokenRef, token_nonnull
from pythoc.bindings.lexer import (
    Lexer, LexerRef, lexer_nonnull, lexer_create, lexer_destroy,
    lexer_next_token_impl
)
from pythoc.bindings.c_ast import (
    # Core types
    Span, span_empty, span_is_empty,
    CType, QualType, PtrType, ArrayType, FuncType,
    StructType, EnumType, EnumValue, FieldInfo, ParamInfo,
    Decl, DeclKind,
    # Refined types
    CTypeRef, QualTypeRef, StructTypeRef, EnumTypeRef,
    ParamInfoRef, FieldInfoRef, EnumValueRef,
    ctype_nonnull, qualtype_nonnull, structtype_nonnull, enumtype_nonnull,
    paraminfo_nonnull, fieldinfo_nonnull, enumvalue_nonnull,
    # Proof types
    CTypeProof, QualTypeProof, StructTypeProof, EnumTypeProof, DeclProof,
    # Allocation
    ctype_alloc, qualtype_alloc, structtype_alloc, enumtype_alloc, decl_alloc,
    paraminfo_alloc, fieldinfo_alloc, enumvalue_alloc,
    # Type constructors
    prim, make_qualtype, make_ptr_type, make_array_type,
    make_func_type, make_struct_type, make_union_type, make_enum_type,
    make_typedef_type,
    # Free functions
    ctype_free, qualtype_free, decl_free,
    # Constants
    QUAL_NONE, QUAL_CONST, QUAL_VOLATILE,
    STORAGE_NONE, STORAGE_EXTERN, STORAGE_STATIC,
)


# =============================================================================
# Parser State
# =============================================================================

MAX_PARAMS = 32
MAX_FIELDS = 64
MAX_ENUM_VALUES = 256


@compile
class Parser:
    """Parser state"""
    lex: ptr[Lexer]
    current: Token              # Current token
    # Scratch buffers for building AST
    params: array[ParamInfo, MAX_PARAMS]
    fields: array[FieldInfo, MAX_FIELDS]
    enum_vals: array[EnumValue, MAX_ENUM_VALUES]


parser_nonnull, ParserRef = nonnull_wrap(ptr[Parser])


# =============================================================================
# Span helper - create from token
# =============================================================================

@compile
def span_from_token(tok: Token) -> Span:
    """Create a Span from current token (zero-copy)"""
    s: Span
    s.start = tok.start
    s.len = tok.length
    return s


# =============================================================================
# Parser helpers
# =============================================================================

@compile
def parser_advance(p: ParserRef) -> void:
    """Advance to next token"""
    lex_ref: LexerRef = assume(p.lex, lexer_nonnull)
    p.current = lexer_next_token_impl(lex_ref)


@compile
def parser_match(p: ParserRef, tok_type: i32) -> bool:
    """Check if current token matches type"""
    return p.current.type == tok_type


@compile
def parser_expect(p: ParserRef, tok_type: i32) -> bool:
    """Expect and consume token, return false if mismatch"""
    if p.current.type != tok_type:
        return False
    parser_advance(p)
    return True


@compile
def parser_skip_until_semicolon(p: ParserRef) -> void:
    """Skip tokens until semicolon or EOF"""
    while p.current.type != TokenType.SEMICOLON and p.current.type != TokenType.EOF:
        parser_advance(p)


@compile
def parser_skip_balanced(p: ParserRef, open_tok: i32, close_tok: i32) -> void:
    """Skip balanced brackets/braces/parens"""
    if p.current.type != open_tok:
        return
    depth: i32 = 1
    parser_advance(p)
    while depth > 0 and p.current.type != TokenType.EOF:
        if p.current.type == open_tok:
            depth = depth + 1
        elif p.current.type == close_tok:
            depth = depth - 1
        parser_advance(p)


# =============================================================================
# Type specifier tokens (for metaprogramming)
# =============================================================================

# Token types that are type specifiers
_type_specifier_tokens = [
    TokenType.VOID, TokenType.CHAR, TokenType.SHORT, TokenType.INT,
    TokenType.LONG, TokenType.FLOAT, TokenType.DOUBLE,
    TokenType.SIGNED, TokenType.UNSIGNED,
    TokenType.STRUCT, TokenType.UNION, TokenType.ENUM,
    TokenType.CONST, TokenType.VOLATILE,
]


@inline
def is_type_specifier(tok_type: i32) -> bool:
    """Check if token is a type specifier (compile-time unrolled)"""
    for spec_type in _type_specifier_tokens:
        if tok_type == spec_type:
            return True
    return False



# =============================================================================
# Type parsing state
# =============================================================================

@compile
class TypeParseState:
    """Intermediate state during type parsing"""
    base_token: i32         # TokenType of base type (INT, CHAR, etc.)
    is_signed: i8           # 1 = signed, 0 = default, -1 = unsigned
    is_const: i8            # 1 if const
    is_volatile: i8         # 1 if volatile
    long_count: i8          # Number of 'long' keywords (0, 1, or 2)
    ptr_depth: i8           # Number of pointer indirections
    name: Span              # For struct/union/enum/typedef names


typeparse_nonnull, TypeParseStateRef = nonnull_wrap(ptr[TypeParseState])


@compile
def typeparse_init(ts: TypeParseStateRef) -> void:
    """Initialize type parse state"""
    ts.base_token = 0
    ts.is_signed = 0
    ts.is_const = 0
    ts.is_volatile = 0
    ts.long_count = 0
    ts.ptr_depth = 0
    ts.name = span_empty()


# =============================================================================
# Type parsing - build CType from tokens
# =============================================================================

@compile
def parse_type_specifiers(p: ParserRef, ts: TypeParseStateRef) -> void:
    """
    Parse C type specifiers into TypeParseState.
    Handles: const, volatile, signed/unsigned, base types, struct/union/enum names
    """
    typeparse_init(ts)
    
    while True:
        tok_type: i32 = p.current.type
        
        # const
        if tok_type == TokenType.CONST:
            ts.is_const = 1
            parser_advance(p)
        # volatile
        elif tok_type == TokenType.VOLATILE:
            ts.is_volatile = 1
            parser_advance(p)
        # signed
        elif tok_type == TokenType.SIGNED:
            ts.is_signed = 1
            parser_advance(p)
        # unsigned
        elif tok_type == TokenType.UNSIGNED:
            ts.is_signed = -1
            parser_advance(p)
        # void
        elif tok_type == TokenType.VOID:
            ts.base_token = TokenType.VOID
            parser_advance(p)
        # char
        elif tok_type == TokenType.CHAR:
            ts.base_token = TokenType.CHAR
            parser_advance(p)
        # short
        elif tok_type == TokenType.SHORT:
            ts.base_token = TokenType.SHORT
            parser_advance(p)
        # int
        elif tok_type == TokenType.INT:
            ts.base_token = TokenType.INT
            parser_advance(p)
        # long
        elif tok_type == TokenType.LONG:
            ts.long_count = ts.long_count + 1
            if ts.base_token == 0:
                ts.base_token = TokenType.LONG
            parser_advance(p)
        # float
        elif tok_type == TokenType.FLOAT:
            ts.base_token = TokenType.FLOAT
            parser_advance(p)
        # double
        elif tok_type == TokenType.DOUBLE:
            ts.base_token = TokenType.DOUBLE
            parser_advance(p)
        # struct
        elif tok_type == TokenType.STRUCT:
            ts.base_token = TokenType.STRUCT
            parser_advance(p)
            if parser_match(p, TokenType.IDENTIFIER):
                ts.name = span_from_token(p.current)
                parser_advance(p)
            # Skip struct body if present
            if parser_match(p, TokenType.LBRACE):
                parser_skip_balanced(p, TokenType.LBRACE, TokenType.RBRACE)
            break
        # union
        elif tok_type == TokenType.UNION:
            ts.base_token = TokenType.UNION
            parser_advance(p)
            if parser_match(p, TokenType.IDENTIFIER):
                ts.name = span_from_token(p.current)
                parser_advance(p)
            if parser_match(p, TokenType.LBRACE):
                parser_skip_balanced(p, TokenType.LBRACE, TokenType.RBRACE)
            break
        # enum
        elif tok_type == TokenType.ENUM:
            ts.base_token = TokenType.ENUM
            parser_advance(p)
            if parser_match(p, TokenType.IDENTIFIER):
                ts.name = span_from_token(p.current)
                parser_advance(p)
            if parser_match(p, TokenType.LBRACE):
                parser_skip_balanced(p, TokenType.LBRACE, TokenType.RBRACE)
            break
        # identifier (typedef name) - only if no base type yet
        elif tok_type == TokenType.IDENTIFIER:
            if ts.base_token == 0:
                ts.base_token = TokenType.IDENTIFIER
                ts.name = span_from_token(p.current)
                parser_advance(p)
            break
        else:
            break
    
    # Parse pointer indirections
    while parser_match(p, TokenType.STAR):
        ts.ptr_depth = ts.ptr_depth + 1
        parser_advance(p)
        # Skip pointer qualifiers
        while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE):
            parser_advance(p)


@compile
def build_base_ctype(ts: TypeParseStateRef) -> struct[CTypeProof, ptr[CType]]:
    """
    Build base CType from TypeParseState.
    Returns (proof, ptr) for linear ownership tracking.
    """
    # Default to int if no base type specified but signed/unsigned present
    base: i32 = ts.base_token
    if base == 0:
        if ts.is_signed != 0 or ts.long_count > 0:
            base = TokenType.INT
        else:
            # No type at all - default to int
            base = TokenType.INT
    
    # Handle long long
    is_longlong: bool = ts.long_count >= 2
    is_unsigned: bool = ts.is_signed == -1
    
    # void
    if base == TokenType.VOID:
        return prim.void()
    
    # char
    if base == TokenType.CHAR:
        if is_unsigned:
            return prim.uchar()
        elif ts.is_signed == 1:
            return prim.schar()
        return prim.char()
    
    # short
    if base == TokenType.SHORT:
        if is_unsigned:
            return prim.ushort()
        return prim.short()
    
    # int
    if base == TokenType.INT:
        if is_longlong:
            if is_unsigned:
                return prim.ulonglong()
            return prim.longlong()
        if is_unsigned:
            return prim.uint()
        return prim.int()
    
    # long
    if base == TokenType.LONG:
        if is_longlong:
            if is_unsigned:
                return prim.ulonglong()
            return prim.longlong()
        if is_unsigned:
            return prim.ulong()
        return prim.long()
    
    # float
    if base == TokenType.FLOAT:
        return prim.float()
    
    # double
    if base == TokenType.DOUBLE:
        if ts.long_count > 0:
            return prim.longdouble()
        return prim.double()
    
    # struct/union/enum/typedef - create appropriate type
    if base == TokenType.STRUCT:
        return make_struct_type(ts.name, nullptr, 0, 0)
    if base == TokenType.UNION:
        return make_union_type(ts.name, nullptr, 0, 0)
    if base == TokenType.ENUM:
        return make_enum_type(ts.name, nullptr, 0, 0)
    if base == TokenType.IDENTIFIER:
        return make_typedef_type(ts.name)
    
    # Fallback to int
    return prim.int()


@compile
def wrap_in_pointer(qt_prf: QualTypeProof, qt: ptr[QualType]) -> struct[QualTypeProof, ptr[QualType]]:
    """Wrap a QualType in a pointer type. Consumes input proof."""
    ptr_prf, ptr_ty = make_ptr_type(qt_prf, qt, QUAL_NONE)
    return make_qualtype(ptr_prf, ptr_ty, QUAL_NONE)


@compile
def build_qualtype_from_state(ts: TypeParseStateRef) -> struct[QualTypeProof, ptr[QualType]]:
    """
    Build complete QualType from TypeParseState, including pointers.
    """
    # Build base type
    ty_prf, ty = build_base_ctype(ts)
    
    # Compute qualifiers
    quals: i8 = QUAL_NONE
    if ts.is_const != 0:
        quals = quals | QUAL_CONST
    if ts.is_volatile != 0:
        quals = quals | QUAL_VOLATILE
    
    # Wrap in QualType
    qt_prf, qt = make_qualtype(ty_prf, ty, quals)
    
    # Add pointer indirections (unroll up to 8 levels)
    if ts.ptr_depth >= 1:
        qt_prf, qt = wrap_in_pointer(qt_prf, qt)
    if ts.ptr_depth >= 2:
        qt_prf, qt = wrap_in_pointer(qt_prf, qt)
    if ts.ptr_depth >= 3:
        qt_prf, qt = wrap_in_pointer(qt_prf, qt)
    if ts.ptr_depth >= 4:
        qt_prf, qt = wrap_in_pointer(qt_prf, qt)
    if ts.ptr_depth >= 5:
        qt_prf, qt = wrap_in_pointer(qt_prf, qt)
    if ts.ptr_depth >= 6:
        qt_prf, qt = wrap_in_pointer(qt_prf, qt)
    if ts.ptr_depth >= 7:
        qt_prf, qt = wrap_in_pointer(qt_prf, qt)
    if ts.ptr_depth >= 8:
        qt_prf, qt = wrap_in_pointer(qt_prf, qt)
    
    return qt_prf, qt


# =============================================================================
# Declarator parsing
# =============================================================================

@compile
def parse_declarator_name(p: ParserRef) -> Span:
    """
    Parse declarator and return name as Span.
    Handles additional pointer stars and array brackets.
    Returns empty span if no name found.
    """
    # Handle additional pointer stars in declarator
    while parser_match(p, TokenType.STAR):
        parser_advance(p)
        while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE):
            parser_advance(p)
    
    # Get name
    name: Span = span_empty()
    if parser_match(p, TokenType.IDENTIFIER):
        name = span_from_token(p.current)
        parser_advance(p)
    elif parser_match(p, TokenType.LPAREN):
        # Function pointer or grouped declarator - skip for now
        parser_skip_balanced(p, TokenType.LPAREN, TokenType.RPAREN)
    
    # Skip array dimensions
    while parser_match(p, TokenType.LBRACKET):
        parser_skip_balanced(p, TokenType.LBRACKET, TokenType.RBRACKET)
    
    return name


# =============================================================================
# Function parsing
# =============================================================================

@compile
def parse_func_params(p: ParserRef, param_count: ptr[i32], is_variadic: ptr[i8]) -> ptr[ParamInfo]:
    """
    Parse function parameters.
    Returns heap-allocated ParamInfo array, sets param_count and is_variadic.
    Caller takes ownership of returned array.
    """
    if not parser_expect(p, TokenType.LPAREN):
        param_count[0] = 0
        is_variadic[0] = 0
        return nullptr
    
    param_count[0] = 0
    is_variadic[0] = 0
    
    # Empty params or (void)
    if parser_match(p, TokenType.RPAREN):
        parser_advance(p)
        return nullptr
    
    if parser_match(p, TokenType.VOID):
        parser_advance(p)
        if parser_match(p, TokenType.RPAREN):
            parser_advance(p)
            return nullptr
    
    # Parse parameters into scratch buffer
    while True:
        # Check for ...
        if parser_match(p, TokenType.ELLIPSIS):
            is_variadic[0] = 1
            parser_advance(p)
            break
        
        if param_count[0] >= MAX_PARAMS:
            break
        
        # Parse parameter type
        ts: TypeParseState
        ts_ref: TypeParseStateRef = assume(ptr(ts), typeparse_nonnull)
        parse_type_specifiers(p, ts_ref)
        qt_prf, qt = build_qualtype_from_state(ts_ref)
        
        # Parse parameter name
        name: Span = parse_declarator_name(p)
        
        # Store in scratch buffer
        p.params[param_count[0]].name = name
        p.params[param_count[0]].type = qt
        consume(qt_prf)  # Transfer ownership to params array
        
        param_count[0] = param_count[0] + 1
        
        if parser_match(p, TokenType.COMMA):
            parser_advance(p)
        else:
            break
    
    parser_expect(p, TokenType.RPAREN)
    
    # Copy params to heap
    if param_count[0] > 0:
        params: ptr[ParamInfo] = paraminfo_alloc(param_count[0])
        memcpy(params, ptr(p.params[0]), param_count[0] * sizeof(ParamInfo))
        return params
    
    return nullptr


@compile
def parse_function_type(p: ParserRef, ret_qt_prf: QualTypeProof, ret_qt: ptr[QualType]) -> struct[CTypeProof, ptr[CType]]:
    """
    Parse function type given return type.
    Takes ownership of ret_qt.
    """
    param_count: i32 = 0
    is_variadic: i8 = 0
    params: ptr[ParamInfo] = parse_func_params(p, ptr(param_count), ptr(is_variadic))
    
    return make_func_type(ret_qt_prf, ret_qt, params, param_count, is_variadic)


# =============================================================================
# Struct/Union parsing
# =============================================================================

@compile
def parse_struct_fields(p: ParserRef, field_count: ptr[i32]) -> ptr[FieldInfo]:
    """
    Parse struct/union fields.
    Returns heap-allocated FieldInfo array, sets field_count.
    Caller takes ownership of returned array.
    """
    if not parser_expect(p, TokenType.LBRACE):
        field_count[0] = 0
        return nullptr
    
    field_count[0] = 0
    
    while not parser_match(p, TokenType.RBRACE) and not parser_match(p, TokenType.EOF):
        if field_count[0] >= MAX_FIELDS:
            parser_skip_until_semicolon(p)
            parser_advance(p)
            continue
        
        # Parse field type
        ts: TypeParseState
        ts_ref: TypeParseStateRef = assume(ptr(ts), typeparse_nonnull)
        parse_type_specifiers(p, ts_ref)
        qt_prf, qt = build_qualtype_from_state(ts_ref)
        
        # Parse field name
        name: Span = parse_declarator_name(p)
        
        # Check for bitfield
        bit_width: i32 = -1
        if parser_match(p, TokenType.COLON):
            parser_advance(p)
            if parser_match(p, TokenType.NUMBER):
                # TODO: parse actual number value
                bit_width = 0
                parser_advance(p)
        
        # Store in scratch buffer
        p.fields[field_count[0]].name = name
        p.fields[field_count[0]].type = qt
        p.fields[field_count[0]].bit_width = bit_width
        consume(qt_prf)  # Transfer ownership
        
        field_count[0] = field_count[0] + 1
        
        # Handle multiple declarators: int a, b, c;
        while parser_match(p, TokenType.COMMA):
            parser_advance(p)
            if field_count[0] >= MAX_FIELDS:
                break
            
            # Copy type from previous field (need to allocate new QualType)
            prev_qt: ptr[QualType] = p.fields[field_count[0] - 1].type
            new_qt_prf, new_qt = qualtype_alloc()
            new_qt.type = prev_qt.type  # Share CType (shallow copy)
            new_qt.quals = prev_qt.quals
            
            name = parse_declarator_name(p)
            p.fields[field_count[0]].name = name
            p.fields[field_count[0]].type = new_qt
            p.fields[field_count[0]].bit_width = -1
            consume(new_qt_prf)
            
            field_count[0] = field_count[0] + 1
        
        parser_expect(p, TokenType.SEMICOLON)
    
    parser_expect(p, TokenType.RBRACE)
    
    # Copy fields to heap
    if field_count[0] > 0:
        fields: ptr[FieldInfo] = fieldinfo_alloc(field_count[0])
        memcpy(fields, ptr(p.fields[0]), field_count[0] * sizeof(FieldInfo))
        return fields
    
    return nullptr


@compile
def parse_struct_or_union(p: ParserRef, is_union: i8) -> struct[CTypeProof, ptr[CType]]:
    """Parse struct or union definition, return CType with ownership proof"""
    # Get name if present
    name: Span = span_empty()
    if parser_match(p, TokenType.IDENTIFIER):
        name = span_from_token(p.current)
        parser_advance(p)
    
    # Parse fields if body present
    fields: ptr[FieldInfo] = nullptr
    field_count: i32 = 0
    is_complete: i8 = 0
    
    if parser_match(p, TokenType.LBRACE):
        fields = parse_struct_fields(p, ptr(field_count))
        is_complete = 1
    
    if is_union != 0:
        return make_union_type(name, fields, field_count, is_complete)
    else:
        return make_struct_type(name, fields, field_count, is_complete)


# =============================================================================
# Enum parsing
# =============================================================================

@compile
def parse_enum_values(p: ParserRef, value_count: ptr[i32]) -> ptr[EnumValue]:
    """
    Parse enum values.
    Returns heap-allocated EnumValue array, sets value_count.
    """
    if not parser_expect(p, TokenType.LBRACE):
        value_count[0] = 0
        return nullptr
    
    value_count[0] = 0
    current_value: i64 = 0
    
    while not parser_match(p, TokenType.RBRACE) and not parser_match(p, TokenType.EOF):
        if value_count[0] >= MAX_ENUM_VALUES:
            break
        
        if parser_match(p, TokenType.IDENTIFIER):
            p.enum_vals[value_count[0]].name = span_from_token(p.current)
            p.enum_vals[value_count[0]].value = current_value
            p.enum_vals[value_count[0]].has_explicit_value = 0
            parser_advance(p)
            
            # Check for explicit value
            if parser_match(p, TokenType.ASSIGN):
                parser_advance(p)
                p.enum_vals[value_count[0]].has_explicit_value = 1
                # Skip value expression (simplified - just skip tokens)
                while not parser_match(p, TokenType.COMMA) and not parser_match(p, TokenType.RBRACE) and not parser_match(p, TokenType.EOF):
                    parser_advance(p)
            
            value_count[0] = value_count[0] + 1
            current_value = current_value + 1
            
            if parser_match(p, TokenType.COMMA):
                parser_advance(p)
        else:
            break
    
    parser_expect(p, TokenType.RBRACE)
    
    # Copy values to heap
    if value_count[0] > 0:
        values: ptr[EnumValue] = enumvalue_alloc(value_count[0])
        memcpy(values, ptr(p.enum_vals[0]), value_count[0] * sizeof(EnumValue))
        return values
    
    return nullptr


@compile
def parse_enum(p: ParserRef) -> struct[CTypeProof, ptr[CType]]:
    """Parse enum definition, return CType with ownership proof"""
    # Get name if present
    name: Span = span_empty()
    if parser_match(p, TokenType.IDENTIFIER):
        name = span_from_token(p.current)
        parser_advance(p)
    
    # Parse values if body present
    values: ptr[EnumValue] = nullptr
    value_count: i32 = 0
    is_complete: i8 = 0
    
    if parser_match(p, TokenType.LBRACE):
        values = parse_enum_values(p, ptr(value_count))
        is_complete = 1
    
    return make_enum_type(name, values, value_count, is_complete)


# =============================================================================
# Top-level declaration parsing
# =============================================================================

@compile
def parse_function_decl(p: ParserRef, ret_qt_prf: QualTypeProof, ret_qt: ptr[QualType], name: Span) -> struct[DeclProof, ptr[Decl]]:
    """Parse function declaration given return type and name"""
    # Parse function type (takes ownership of ret_qt)
    func_ty_prf, func_ty = parse_function_type(p, ret_qt_prf, ret_qt)
    
    # Wrap in QualType (no qualifiers for function type)
    func_qt_prf, func_qt = make_qualtype(func_ty_prf, func_ty, QUAL_NONE)
    
    # Create declaration
    decl_prf, decl = decl_alloc()
    decl.kind = DeclKind(DeclKind.Func)
    decl.name = name
    decl.type = func_qt
    decl.storage = STORAGE_NONE
    consume(func_qt_prf)  # Transfer ownership to Decl
    
    # Skip function body if present
    if parser_match(p, TokenType.LBRACE):
        parser_skip_balanced(p, TokenType.LBRACE, TokenType.RBRACE)
    elif parser_match(p, TokenType.SEMICOLON):
        parser_advance(p)
    
    return decl_prf, decl


@compile
def try_make_struct_decl(ty_prf: CTypeProof, ty: ptr[CType], storage: i8) -> struct[i8, DeclProof, ptr[Decl]]:
    """
    Try to create a struct declaration from CType.
    Returns (success, decl_prf, decl).
    If success=0, ty_prf is consumed, and caller must free returned decl.
    """
    name: Span = span_empty()
    has_name: i8 = 0
    
    match ty[0]:
        case (CType.Struct, st):
            if not span_is_empty(st.name):
                name = st.name
                has_name = 1
        case _:
            pass
    
    if has_name != 0:
        qt_prf, qt = make_qualtype(ty_prf, ty, QUAL_NONE)
        decl_prf, decl = decl_alloc()
        decl.kind = DeclKind(DeclKind.Struct)
        decl.name = name
        decl.type = qt
        decl.storage = storage
        consume(qt_prf)
        return 1, decl_prf, decl
    else:
        ctype_free(ty_prf, ty)
        # Return dummy decl - caller must free it
        dummy_prf, dummy_decl = decl_alloc()
        dummy_decl.type = nullptr
        return 0, dummy_prf, dummy_decl


@compile
def try_make_union_decl(ty_prf: CTypeProof, ty: ptr[CType], storage: i8) -> struct[i8, DeclProof, ptr[Decl]]:
    """Try to create a union declaration from CType."""
    name: Span = span_empty()
    has_name: i8 = 0
    
    match ty[0]:
        case (CType.Union, st):
            if not span_is_empty(st.name):
                name = st.name
                has_name = 1
        case _:
            pass
    
    if has_name != 0:
        qt_prf, qt = make_qualtype(ty_prf, ty, QUAL_NONE)
        decl_prf, decl = decl_alloc()
        decl.kind = DeclKind(DeclKind.Union)
        decl.name = name
        decl.type = qt
        decl.storage = storage
        consume(qt_prf)
        return 1, decl_prf, decl
    else:
        ctype_free(ty_prf, ty)
        dummy_prf, dummy_decl = decl_alloc()
        dummy_decl.type = nullptr
        return 0, dummy_prf, dummy_decl


@compile
def try_make_enum_decl(ty_prf: CTypeProof, ty: ptr[CType], storage: i8) -> struct[i8, DeclProof, ptr[Decl]]:
    """Try to create an enum declaration from CType."""
    name: Span = span_empty()
    has_name: i8 = 0
    
    match ty[0]:
        case (CType.Enum, et):
            if not span_is_empty(et.name):
                name = et.name
                has_name = 1
        case _:
            pass
    
    if has_name != 0:
        qt_prf, qt = make_qualtype(ty_prf, ty, QUAL_NONE)
        decl_prf, decl = decl_alloc()
        decl.kind = DeclKind(DeclKind.Enum)
        decl.name = name
        decl.type = qt
        decl.storage = storage
        consume(qt_prf)
        return 1, decl_prf, decl
    else:
        ctype_free(ty_prf, ty)
        dummy_prf, dummy_decl = decl_alloc()
        dummy_decl.type = nullptr
        return 0, dummy_prf, dummy_decl


# =============================================================================
# Yield-based declaration iterator
# =============================================================================

@compile
def parse_declarations(source: ptr[i8]) -> struct[DeclProof, ptr[Decl]]:
    """
    Yield declarations from source.
    Returns (DeclProof, ptr[Decl]) for each declaration.
    Caller takes ownership of each yielded Decl.
    
    Usage:
        for decl_prf, decl in parse_declarations(source):
            match decl.kind:
                case (DeclKind.Func):
                    # handle function
                    pass
            decl_free(decl_prf, decl)
    """
    prf, lex_raw = lexer_create(source)
    
    for lex in refine(lex_raw, lexer_nonnull):
        # Create parser
        parser: Parser = Parser()
        parser.lex = lex_raw
        parser_advance(assume(ptr(parser), parser_nonnull))
        p: ParserRef = assume(ptr(parser), parser_nonnull)
        
        while p.current.type != TokenType.EOF:
            # Skip storage class specifiers
            storage: i8 = STORAGE_NONE
            while parser_match(p, TokenType.EXTERN) or parser_match(p, TokenType.STATIC):
                if parser_match(p, TokenType.EXTERN):
                    storage = STORAGE_EXTERN
                else:
                    storage = STORAGE_STATIC
                parser_advance(p)
            
            # typedef
            if parser_match(p, TokenType.TYPEDEF):
                parser_advance(p)
                # Skip typedef for now
                parser_skip_until_semicolon(p)
                if parser_match(p, TokenType.SEMICOLON):
                    parser_advance(p)
                continue
            
            # struct
            if parser_match(p, TokenType.STRUCT):
                parser_advance(p)
                ty_prf, ty = parse_struct_or_union(p, 0)
                success, decl_prf, decl = try_make_struct_decl(ty_prf, ty, storage)
                if success != 0:
                    yield decl_prf, decl
                else:
                    decl_free(decl_prf, decl)
                parser_skip_until_semicolon(p)
                if parser_match(p, TokenType.SEMICOLON):
                    parser_advance(p)
                continue
            
            # union
            if parser_match(p, TokenType.UNION):
                parser_advance(p)
                ty_prf, ty = parse_struct_or_union(p, 1)
                success, decl_prf, decl = try_make_union_decl(ty_prf, ty, storage)
                if success != 0:
                    yield decl_prf, decl
                else:
                    decl_free(decl_prf, decl)
                parser_skip_until_semicolon(p)
                if parser_match(p, TokenType.SEMICOLON):
                    parser_advance(p)
                continue
            
            # enum
            if parser_match(p, TokenType.ENUM):
                parser_advance(p)
                ty_prf, ty = parse_enum(p)
                success, decl_prf, decl = try_make_enum_decl(ty_prf, ty, storage)
                if success != 0:
                    yield decl_prf, decl
                else:
                    decl_free(decl_prf, decl)
                parser_skip_until_semicolon(p)
                if parser_match(p, TokenType.SEMICOLON):
                    parser_advance(p)
                continue
            
            # Parse type and declarator
            ts: TypeParseState
            ts_ref: TypeParseStateRef = assume(ptr(ts), typeparse_nonnull)
            parse_type_specifiers(p, ts_ref)
            qt_prf, qt = build_qualtype_from_state(ts_ref)
            
            name: Span = parse_declarator_name(p)
            
            if span_is_empty(name):
                qualtype_free(qt_prf, qt)
                parser_skip_until_semicolon(p)
                if parser_match(p, TokenType.SEMICOLON):
                    parser_advance(p)
                continue
            
            # Function declaration
            if parser_match(p, TokenType.LPAREN):
                decl_prf, decl = parse_function_decl(p, qt_prf, qt, name)
                decl.storage = storage
                yield decl_prf, decl
            else:
                # Variable declaration - skip for now
                qualtype_free(qt_prf, qt)
                parser_skip_until_semicolon(p)
                if parser_match(p, TokenType.SEMICOLON):
                    parser_advance(p)
        
        lexer_destroy(prf, lex_raw)

