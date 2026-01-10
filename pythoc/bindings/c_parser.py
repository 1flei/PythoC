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
- Uses Python metaprogramming for token matching and code generation
- Uses match-case for type dispatch
- Uses refine for safe null checks
"""

from pythoc import (
    compile, inline, i32, i64, i8, bool, ptr, array, nullptr, sizeof, void,
    char, refine, assume, struct, consume, linear, defer
)
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import memcpy
from pythoc.std.refine_wrapper import nonnull_wrap

from pythoc.bindings.c_token import Token, TokenType, TokenRef, token_nonnull
from pythoc.bindings.lexer import (
    Lexer, LexerRef, lexer_nonnull, lexer_create, lexer_destroy,
    lexer_next_token, token_release, LexerProof, TokenProof
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
# Parse Error Types
# =============================================================================

@compile
class ParseError:
    """Parse error information"""
    line: i32
    col: i32
    error_code: i32  # Error code for programmatic handling


# Error codes
ERR_NONE: i32 = 0
ERR_UNEXPECTED_TOKEN: i32 = 1
ERR_EXPECTED_IDENTIFIER: i32 = 2
ERR_EXPECTED_SEMICOLON: i32 = 3
ERR_EXPECTED_RBRACE: i32 = 4
ERR_EXPECTED_RPAREN: i32 = 5
ERR_MAX_PARAMS_EXCEEDED: i32 = 6
ERR_MAX_FIELDS_EXCEEDED: i32 = 7
ERR_MAX_ENUM_VALUES_EXCEEDED: i32 = 8
ERR_NULL_LEXER: i32 = 9


# =============================================================================
# Parser State
# =============================================================================

MAX_PARAMS = 32
MAX_FIELDS = 64
MAX_ENUM_VALUES = 256
MAX_ERRORS = 16


@compile
class Parser:
    """Parser state - no linear fields, proofs passed separately"""
    lex: ptr[Lexer]
    current: Token               # Current token
    has_token: i8                # Whether current token is valid
    # Scratch buffers for building AST
    params: array[ParamInfo, MAX_PARAMS]
    fields: array[FieldInfo, MAX_FIELDS]
    enum_vals: array[EnumValue, MAX_ENUM_VALUES]
    # Error tracking
    errors: array[ParseError, MAX_ERRORS]
    error_count: i32


parser_nonnull, ParserRef = nonnull_wrap(ptr[Parser])


@compile
class ParserProofs:
    """Linear proofs for parser - passed separately from Parser state"""
    lex_prf: LexerProof
    current_prf: TokenProof


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
# Error handling
# =============================================================================

@compile
def parser_add_error(p: ParserRef, error_code: i32) -> void:
    """Record a parse error"""
    if p.error_count < MAX_ERRORS:
        p.errors[p.error_count].line = p.current.line
        p.errors[p.error_count].col = p.current.col
        p.errors[p.error_count].error_code = error_code
        p.error_count = p.error_count + 1


@compile
def parser_has_errors(p: ParserRef) -> bool:
    """Check if parser has recorded any errors"""
    return p.error_count > 0


# =============================================================================
# Parser helpers with proof tracking
# =============================================================================

@compile
def parser_advance(p: ParserRef, prfs: ParserProofs) -> ParserProofs:
    """Advance to next token, managing token proofs.
    
    Precondition: p.lex must be non-null (initialized parser).
    Returns updated proofs.
    """
    # Get lexer ref - p.lex must be non-null (precondition)
    lex: LexerRef = assume(p.lex, lexer_nonnull)
    
    # Release previous token if we have one, then get next
    if p.has_token != 0:
        prfs.lex_prf = token_release(p.current, prfs.current_prf, prfs.lex_prf)
    else:
        # First call - current_prf is dummy, just consume it
        consume(prfs.current_prf)
    
    p.current, prfs.current_prf, prfs.lex_prf = lexer_next_token(lex, prfs.lex_prf)
    p.has_token = 1
    return prfs


@compile
def parser_match(p: ParserRef, tok_type: i32) -> bool:
    """Check if current token matches type"""
    return p.current.type == tok_type


@compile
def parser_expect(p: ParserRef, prfs: ParserProofs, tok_type: i32) -> struct[bool, ParserProofs]:
    """Expect and consume token, return (success, updated_proofs)"""
    if p.current.type != tok_type:
        match tok_type:
            case TokenType.SEMICOLON:
                parser_add_error(p, ERR_EXPECTED_SEMICOLON)
            case TokenType.RBRACE:
                parser_add_error(p, ERR_EXPECTED_RBRACE)
            case TokenType.RPAREN:
                parser_add_error(p, ERR_EXPECTED_RPAREN)
            case TokenType.IDENTIFIER:
                parser_add_error(p, ERR_EXPECTED_IDENTIFIER)
            case _:
                parser_add_error(p, ERR_UNEXPECTED_TOKEN)
        return False, prfs
    prfs = parser_advance(p, prfs)
    return True, prfs


@compile
def parser_skip_until_semicolon(p: ParserRef, prfs: ParserProofs) -> ParserProofs:
    """Skip tokens until semicolon or EOF"""
    while p.current.type != TokenType.SEMICOLON and p.current.type != TokenType.EOF:
        prfs = parser_advance(p, prfs)
    return prfs


@compile
def parser_skip_balanced(p: ParserRef, prfs: ParserProofs, open_tok: i32, close_tok: i32) -> ParserProofs:
    """Skip balanced brackets/braces/parens"""
    if p.current.type != open_tok:
        return prfs
    depth: i32 = 1
    prfs = parser_advance(p, prfs)
    while depth > 0 and p.current.type != TokenType.EOF:
        match p.current.type:
            case _ if p.current.type == open_tok:
                depth = depth + 1
            case _ if p.current.type == close_tok:
                depth = depth - 1
            case _:
                pass
        prfs = parser_advance(p, prfs)
    return prfs


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
# Type parsing - build CType from tokens using match-case
# =============================================================================

@compile
def parse_type_specifiers(p: ParserRef, prfs: ParserProofs, ts: TypeParseStateRef) -> ParserProofs:
    """
    Parse C type specifiers into TypeParseState.
    Handles: const, volatile, signed/unsigned, base types, struct/union/enum names
    Uses match-case for cleaner dispatch.
    Returns updated proofs.
    """
    typeparse_init(ts)
    
    while True:
        tok_type: i32 = p.current.type
        
        match tok_type:
            # Qualifiers
            case TokenType.CONST:
                ts.is_const = 1
                prfs = parser_advance(p, prfs)
            case TokenType.VOLATILE:
                ts.is_volatile = 1
                prfs = parser_advance(p, prfs)
            # Sign specifiers
            case TokenType.SIGNED:
                ts.is_signed = 1
                prfs = parser_advance(p, prfs)
            case TokenType.UNSIGNED:
                ts.is_signed = -1
                prfs = parser_advance(p, prfs)
            # Primitive types
            case TokenType.VOID:
                ts.base_token = TokenType.VOID
                prfs = parser_advance(p, prfs)
            case TokenType.CHAR:
                ts.base_token = TokenType.CHAR
                prfs = parser_advance(p, prfs)
            case TokenType.SHORT:
                ts.base_token = TokenType.SHORT
                prfs = parser_advance(p, prfs)
            case TokenType.INT:
                ts.base_token = TokenType.INT
                prfs = parser_advance(p, prfs)
            case TokenType.LONG:
                ts.long_count = ts.long_count + 1
                if ts.base_token == 0:
                    ts.base_token = TokenType.LONG
                prfs = parser_advance(p, prfs)
            case TokenType.FLOAT:
                ts.base_token = TokenType.FLOAT
                prfs = parser_advance(p, prfs)
            case TokenType.DOUBLE:
                ts.base_token = TokenType.DOUBLE
                prfs = parser_advance(p, prfs)
            # Compound types - these break out of the loop
            case TokenType.STRUCT:
                ts.base_token = TokenType.STRUCT
                prfs = parser_advance(p, prfs)
                if parser_match(p, TokenType.IDENTIFIER):
                    ts.name = span_from_token(p.current)
                    prfs = parser_advance(p, prfs)
                if parser_match(p, TokenType.LBRACE):
                    prfs = parser_skip_balanced(p, prfs, TokenType.LBRACE, TokenType.RBRACE)
                break
            case TokenType.UNION:
                ts.base_token = TokenType.UNION
                prfs = parser_advance(p, prfs)
                if parser_match(p, TokenType.IDENTIFIER):
                    ts.name = span_from_token(p.current)
                    prfs = parser_advance(p, prfs)
                if parser_match(p, TokenType.LBRACE):
                    prfs = parser_skip_balanced(p, prfs, TokenType.LBRACE, TokenType.RBRACE)
                break
            case TokenType.ENUM:
                ts.base_token = TokenType.ENUM
                prfs = parser_advance(p, prfs)
                if parser_match(p, TokenType.IDENTIFIER):
                    ts.name = span_from_token(p.current)
                    prfs = parser_advance(p, prfs)
                if parser_match(p, TokenType.LBRACE):
                    prfs = parser_skip_balanced(p, prfs, TokenType.LBRACE, TokenType.RBRACE)
                break
            # Identifier (typedef name) - only if no base type yet AND no sign specifier
            case TokenType.IDENTIFIER:
                if ts.base_token == 0 and ts.is_signed == 0:
                    ts.base_token = TokenType.IDENTIFIER
                    ts.name = span_from_token(p.current)
                    prfs = parser_advance(p, prfs)
                break
            case _:
                break
    
    # Parse pointer indirections
    while parser_match(p, TokenType.STAR):
        ts.ptr_depth = ts.ptr_depth + 1
        prfs = parser_advance(p, prfs)
        # Skip pointer qualifiers
        while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE):
            prfs = parser_advance(p, prfs)
    
    return prfs


@compile
def build_base_ctype(ts: TypeParseStateRef) -> struct[CTypeProof, ptr[CType]]:
    """
    Build base CType from TypeParseState using match-case.
    Returns (proof, ptr) for linear ownership tracking.
    """
    # Default to int if no base type specified but signed/unsigned present
    base: i32 = ts.base_token
    if base == 0:
        base = TokenType.INT
    
    # Handle long long
    is_longlong: bool = ts.long_count >= 2
    is_unsigned: bool = ts.is_signed == -1
    
    match base:
        case TokenType.VOID:
            return prim.void()
        case TokenType.CHAR:
            if is_unsigned:
                return prim.uchar()
            elif ts.is_signed == 1:
                return prim.schar()
            return prim.char()
        case TokenType.SHORT:
            if is_unsigned:
                return prim.ushort()
            return prim.short()
        case TokenType.INT:
            if is_longlong:
                if is_unsigned:
                    return prim.ulonglong()
                return prim.longlong()
            if is_unsigned:
                return prim.uint()
            return prim.int()
        case TokenType.LONG:
            if is_longlong:
                if is_unsigned:
                    return prim.ulonglong()
                return prim.longlong()
            if is_unsigned:
                return prim.ulong()
            return prim.long()
        case TokenType.FLOAT:
            return prim.float()
        case TokenType.DOUBLE:
            if ts.long_count > 0:
                return prim.longdouble()
            return prim.double()
        case TokenType.STRUCT:
            return make_struct_type(ts.name, nullptr, 0, 0)
        case TokenType.UNION:
            return make_union_type(ts.name, nullptr, 0, 0)
        case TokenType.ENUM:
            return make_enum_type(ts.name, nullptr, 0, 0)
        case TokenType.IDENTIFIER:
            return make_typedef_type(ts.name)
        case _:
            # Fallback to int
            return prim.int()


@compile
def wrap_in_pointer(qt_prf: QualTypeProof, qt: ptr[QualType]) -> struct[QualTypeProof, ptr[QualType]]:
    """Wrap a QualType in a pointer type. Consumes input proof."""
    ptr_prf, ptr_ty = make_ptr_type(qt_prf, qt, QUAL_NONE)
    return make_qualtype(ptr_prf, ptr_ty, QUAL_NONE)


# =============================================================================
# Pointer depth wrapping - metaprogrammed
# =============================================================================

# Maximum supported pointer depth
MAX_PTR_DEPTH = 16


# Generate wrap_ptr_N functions for each depth using metaprogramming
def _make_wrap_ptr_func(depth: int):
    """Factory to generate wrap_ptr_N functions via metaprogramming."""
    if depth == 0:
        @compile(suffix=f"_0")
        def wrap_ptr(qt_prf: QualTypeProof, qt: ptr[QualType]) -> struct[QualTypeProof, ptr[QualType]]:
            return qt_prf, qt
        return wrap_ptr
    else:
        # Build function body that calls wrap_in_pointer N times
        prev_func = _make_wrap_ptr_func(depth - 1)
        
        @compile(suffix=f"_{depth}")
        def wrap_ptr(qt_prf: QualTypeProof, qt: ptr[QualType]) -> struct[QualTypeProof, ptr[QualType]]:
            qt_prf, qt = prev_func(qt_prf, qt)
            qt_prf, qt = wrap_in_pointer(qt_prf, qt)
            return qt_prf, qt
        return wrap_ptr


# Generate all wrap_ptr_N functions
_wrap_ptr_funcs = [_make_wrap_ptr_func(i) for i in range(MAX_PTR_DEPTH + 1)]


@compile
def apply_ptr_depth(qt_prf: QualTypeProof, qt: ptr[QualType], depth: i8) -> struct[QualTypeProof, ptr[QualType]]:
    """Apply pointer indirections using compile-time unrolling."""
    for i in range(MAX_PTR_DEPTH + 1):
        if depth == i:
            return _wrap_ptr_funcs[i](qt_prf, qt)
    return _wrap_ptr_funcs[MAX_PTR_DEPTH](qt_prf, qt)


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
    
    # Add pointer indirections using metaprogrammed unrolling
    qt_prf, qt = apply_ptr_depth(qt_prf, qt, ts.ptr_depth)
    
    return qt_prf, qt


# =============================================================================
# Declarator parsing
# =============================================================================

@compile
def parse_declarator_name(p: ParserRef, prfs: ParserProofs) -> struct[Span, ParserProofs]:
    """
    Parse declarator and return (name, updated_proofs).
    Handles additional pointer stars and array brackets.
    Returns empty span if no name found.
    """
    # Handle additional pointer stars in declarator
    while parser_match(p, TokenType.STAR):
        prfs = parser_advance(p, prfs)
        while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE):
            prfs = parser_advance(p, prfs)
    
    # Get name
    name: Span = span_empty()
    match p.current.type:
        case TokenType.IDENTIFIER:
            name = span_from_token(p.current)
            prfs = parser_advance(p, prfs)
        case TokenType.LPAREN:
            # Function pointer or grouped declarator - skip for now
            prfs = parser_skip_balanced(p, prfs, TokenType.LPAREN, TokenType.RPAREN)
        case _:
            pass
    
    # Skip array dimensions
    while parser_match(p, TokenType.LBRACKET):
        prfs = parser_skip_balanced(p, prfs, TokenType.LBRACKET, TokenType.RBRACKET)
    
    return name, prfs


# =============================================================================
# Function parsing
# =============================================================================

@compile
def parse_func_params(p: ParserRef, prfs: ParserProofs, param_count: ptr[i32], is_variadic: ptr[i8]) -> struct[ptr[ParamInfo], ParserProofs]:
    """
    Parse function parameters.
    Returns (heap-allocated ParamInfo array, updated_proofs), sets param_count and is_variadic.
    Caller takes ownership of returned array.
    """
    ok: bool
    ok, prfs = parser_expect(p, prfs, TokenType.LPAREN)
    if not ok:
        param_count[0] = 0
        is_variadic[0] = 0
        return nullptr, prfs
    
    param_count[0] = 0
    is_variadic[0] = 0
    
    # Empty params or (void)
    if parser_match(p, TokenType.RPAREN):
        prfs = parser_advance(p, prfs)
        return nullptr, prfs
    
    if parser_match(p, TokenType.VOID):
        prfs = parser_advance(p, prfs)
        if parser_match(p, TokenType.RPAREN):
            prfs = parser_advance(p, prfs)
            return nullptr, prfs
    
    # Parse parameters into scratch buffer
    while True:
        # Check for ...
        if parser_match(p, TokenType.ELLIPSIS):
            is_variadic[0] = 1
            prfs = parser_advance(p, prfs)
            break
        
        if param_count[0] >= MAX_PARAMS:
            parser_add_error(p, ERR_MAX_PARAMS_EXCEEDED)
            break
        
        # Parse parameter type - ptr(ts) for stack variable is always non-null
        ts: TypeParseState
        ts_ref: TypeParseStateRef = assume(ptr(ts), typeparse_nonnull)
        prfs = parse_type_specifiers(p, prfs, ts_ref)
        qt_prf, qt = build_qualtype_from_state(ts_ref)
        
        # Parse parameter name
        name: Span
        name, prfs = parse_declarator_name(p, prfs)
        
        # Store in scratch buffer
        p.params[param_count[0]].name = name
        p.params[param_count[0]].type = qt
        consume(qt_prf)  # Transfer ownership to params array
        
        param_count[0] = param_count[0] + 1
        
        match p.current.type:
            case TokenType.COMMA:
                prfs = parser_advance(p, prfs)
            case _:
                break
    
    _, prfs = parser_expect(p, prfs, TokenType.RPAREN)
    
    # Copy params to heap
    if param_count[0] > 0:
        params: ptr[ParamInfo] = paraminfo_alloc(param_count[0])
        memcpy(params, ptr(p.params[0]), param_count[0] * sizeof(ParamInfo))
        return params, prfs
    
    return nullptr, prfs


@compile
def parse_function_type(p: ParserRef, prfs: ParserProofs, ret_qt_prf: QualTypeProof, ret_qt: ptr[QualType]) -> struct[CTypeProof, ptr[CType], ParserProofs]:
    """
    Parse function type given return type.
    Takes ownership of ret_qt.
    Returns (proof, ctype, updated_proofs).
    """
    param_count: i32 = 0
    is_variadic: i8 = 0
    params: ptr[ParamInfo]
    params, prfs = parse_func_params(p, prfs, ptr(param_count), ptr(is_variadic))
    
    ty_prf, ty = make_func_type(ret_qt_prf, ret_qt, params, param_count, is_variadic)
    return ty_prf, ty, prfs


# =============================================================================
# Struct/Union parsing
# =============================================================================

@compile
def parse_struct_fields(p: ParserRef, prfs: ParserProofs, field_count: ptr[i32]) -> struct[ptr[FieldInfo], ParserProofs]:
    """
    Parse struct/union fields.
    Returns (heap-allocated FieldInfo array, updated_proofs), sets field_count.
    Caller takes ownership of returned array.
    """
    ok: bool
    ok, prfs = parser_expect(p, prfs, TokenType.LBRACE)
    if not ok:
        field_count[0] = 0
        return nullptr, prfs
    
    field_count[0] = 0
    
    while not parser_match(p, TokenType.RBRACE) and not parser_match(p, TokenType.EOF):
        if field_count[0] >= MAX_FIELDS:
            parser_add_error(p, ERR_MAX_FIELDS_EXCEEDED)
            prfs = parser_skip_until_semicolon(p, prfs)
            prfs = parser_advance(p, prfs)
            continue
        
        # Parse field type - ptr(ts) for stack variable is always non-null
        ts: TypeParseState
        ts_ref: TypeParseStateRef = assume(ptr(ts), typeparse_nonnull)
        prfs = parse_type_specifiers(p, prfs, ts_ref)
        qt_prf, qt = build_qualtype_from_state(ts_ref)
        
        # Parse field name
        name: Span
        name, prfs = parse_declarator_name(p, prfs)
        
        # Check for bitfield
        bit_width: i32 = -1
        if parser_match(p, TokenType.COLON):
            prfs = parser_advance(p, prfs)
            if parser_match(p, TokenType.NUMBER):
                # TODO: parse actual number value
                bit_width = 0
                prfs = parser_advance(p, prfs)
        
        # Store in scratch buffer
        p.fields[field_count[0]].name = name
        p.fields[field_count[0]].type = qt
        p.fields[field_count[0]].bit_width = bit_width
        consume(qt_prf)  # Transfer ownership
        
        field_count[0] = field_count[0] + 1
        
        # Handle multiple declarators: int a, b, c;
        while parser_match(p, TokenType.COMMA):
            prfs = parser_advance(p, prfs)
            if field_count[0] >= MAX_FIELDS:
                parser_add_error(p, ERR_MAX_FIELDS_EXCEEDED)
                break
            
            # Copy type from previous field (need to allocate new QualType)
            prev_qt: ptr[QualType] = p.fields[field_count[0] - 1].type
            new_qt_prf, new_qt = qualtype_alloc()
            new_qt.type = prev_qt.type  # Share CType (shallow copy)
            new_qt.quals = prev_qt.quals
            
            name, prfs = parse_declarator_name(p, prfs)
            p.fields[field_count[0]].name = name
            p.fields[field_count[0]].type = new_qt
            p.fields[field_count[0]].bit_width = -1
            consume(new_qt_prf)
            
            field_count[0] = field_count[0] + 1
        
        _, prfs = parser_expect(p, prfs, TokenType.SEMICOLON)
    
    _, prfs = parser_expect(p, prfs, TokenType.RBRACE)
    
    # Copy fields to heap
    if field_count[0] > 0:
        fields: ptr[FieldInfo] = fieldinfo_alloc(field_count[0])
        memcpy(fields, ptr(p.fields[0]), field_count[0] * sizeof(FieldInfo))
        return fields, prfs
    
    return nullptr, prfs


@compile
def parse_struct_or_union(p: ParserRef, prfs: ParserProofs, is_union: i8) -> struct[CTypeProof, ptr[CType], ParserProofs]:
    """Parse struct or union definition, return (CTypeProof, CType, updated_proofs)"""
    # Get name if present
    name: Span = span_empty()
    if parser_match(p, TokenType.IDENTIFIER):
        name = span_from_token(p.current)
        prfs = parser_advance(p, prfs)
    
    # Parse fields if body present
    fields: ptr[FieldInfo] = nullptr
    field_count: i32 = 0
    is_complete: i8 = 0
    
    if parser_match(p, TokenType.LBRACE):
        fields, prfs = parse_struct_fields(p, prfs, ptr(field_count))
        is_complete = 1
    
    if is_union != 0:
        ty_prf, ty = make_union_type(name, fields, field_count, is_complete)
        return ty_prf, ty, prfs
    else:
        ty_prf, ty = make_struct_type(name, fields, field_count, is_complete)
        return ty_prf, ty, prfs


# =============================================================================
# Enum parsing
# =============================================================================

@compile
def parse_enum_values(p: ParserRef, prfs: ParserProofs, value_count: ptr[i32]) -> struct[ptr[EnumValue], ParserProofs]:
    """
    Parse enum values.
    Returns (heap-allocated EnumValue array, updated_proofs), sets value_count.
    """
    ok: bool
    ok, prfs = parser_expect(p, prfs, TokenType.LBRACE)
    if not ok:
        value_count[0] = 0
        return nullptr, prfs
    
    value_count[0] = 0
    current_value: i64 = 0
    
    while not parser_match(p, TokenType.RBRACE) and not parser_match(p, TokenType.EOF):
        if value_count[0] >= MAX_ENUM_VALUES:
            parser_add_error(p, ERR_MAX_ENUM_VALUES_EXCEEDED)
            break
        
        match p.current.type:
            case TokenType.IDENTIFIER:
                p.enum_vals[value_count[0]].name = span_from_token(p.current)
                p.enum_vals[value_count[0]].value = current_value
                p.enum_vals[value_count[0]].has_explicit_value = 0
                prfs = parser_advance(p, prfs)
                
                # Check for explicit value
                if parser_match(p, TokenType.ASSIGN):
                    prfs = parser_advance(p, prfs)
                    p.enum_vals[value_count[0]].has_explicit_value = 1
                    # Skip value expression (simplified - just skip tokens)
                    while not parser_match(p, TokenType.COMMA) and not parser_match(p, TokenType.RBRACE) and not parser_match(p, TokenType.EOF):
                        prfs = parser_advance(p, prfs)
                
                value_count[0] = value_count[0] + 1
                current_value = current_value + 1
                
                if parser_match(p, TokenType.COMMA):
                    prfs = parser_advance(p, prfs)
            case _:
                break
    
    _, prfs = parser_expect(p, prfs, TokenType.RBRACE)
    
    # Copy values to heap
    if value_count[0] > 0:
        values: ptr[EnumValue] = enumvalue_alloc(value_count[0])
        memcpy(values, ptr(p.enum_vals[0]), value_count[0] * sizeof(EnumValue))
        return values, prfs
    
    return nullptr, prfs


@compile
def parse_enum(p: ParserRef, prfs: ParserProofs) -> struct[CTypeProof, ptr[CType], ParserProofs]:
    """Parse enum definition, return (CTypeProof, CType, updated_proofs)"""
    # Get name if present
    name: Span = span_empty()
    if parser_match(p, TokenType.IDENTIFIER):
        name = span_from_token(p.current)
        prfs = parser_advance(p, prfs)
    
    # Parse values if body present
    values: ptr[EnumValue] = nullptr
    value_count: i32 = 0
    is_complete: i8 = 0
    
    if parser_match(p, TokenType.LBRACE):
        values, prfs = parse_enum_values(p, prfs, ptr(value_count))
        is_complete = 1
    
    ty_prf, ty = make_enum_type(name, values, value_count, is_complete)
    return ty_prf, ty, prfs


# =============================================================================
# Top-level declaration parsing
# =============================================================================

@compile
def parse_function_decl(p: ParserRef, prfs: ParserProofs, ret_qt_prf: QualTypeProof, ret_qt: ptr[QualType], name: Span) -> struct[DeclProof, ptr[Decl], ParserProofs]:
    """Parse function declaration given return type and name"""
    # Parse function type (takes ownership of ret_qt)
    func_ty_prf, func_ty, prfs = parse_function_type(p, prfs, ret_qt_prf, ret_qt)
    
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
    match p.current.type:
        case TokenType.LBRACE:
            prfs = parser_skip_balanced(p, prfs, TokenType.LBRACE, TokenType.RBRACE)
        case TokenType.SEMICOLON:
            prfs = parser_advance(p, prfs)
        case _:
            pass
    
    return decl_prf, decl, prfs


# =============================================================================
# try_make_*_decl functions
# =============================================================================

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
    """
    Try to create a union declaration from CType.
    Returns (success, decl_prf, decl).
    If success=0, ty_prf is consumed, and caller must free returned decl.
    """
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
        # Return dummy decl - caller must free it
        dummy_prf, dummy_decl = decl_alloc()
        dummy_decl.type = nullptr
        return 0, dummy_prf, dummy_decl


@compile
def try_make_enum_decl(ty_prf: CTypeProof, ty: ptr[CType], storage: i8) -> struct[i8, DeclProof, ptr[Decl]]:
    """
    Try to create an enum declaration from CType.
    Returns (success, decl_prf, decl).
    If success=0, ty_prf is consumed, and caller must free returned decl.
    """
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
        # Return dummy decl - caller must free it
        dummy_prf, dummy_decl = decl_alloc()
        dummy_decl.type = nullptr
        return 0, dummy_prf, dummy_decl


@compile
def parse_typedef_decl(p: ParserRef, prfs: ParserProofs, qt_prf: QualTypeProof, qt: ptr[QualType]) -> struct[i8, DeclProof, ptr[Decl], ParserProofs]:
    """Parse typedef name and create declaration.
    
    Returns (success, decl_prf, decl, updated_prfs).
    Takes ownership of qt_prf/qt on success, frees them on failure.
    """
    # Parse pointer indirections
    while parser_match(p, TokenType.STAR):
        prfs = parser_advance(p, prfs)
        while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE):
            prfs = parser_advance(p, prfs)
        qt_prf, qt = wrap_in_pointer(qt_prf, qt)
    
    # Get typedef name
    name: Span
    name, prfs = parse_declarator_name(p, prfs)
    
    if not span_is_empty(name):
        decl_prf, decl = decl_alloc()
        decl.kind = DeclKind(DeclKind.Typedef)
        decl.name = name
        decl.type = qt
        decl.storage = STORAGE_NONE
        consume(qt_prf)
        prfs = parser_skip_until_semicolon(p, prfs)
        if parser_match(p, TokenType.SEMICOLON):
            prfs = parser_advance(p, prfs)
        return 1, decl_prf, decl, prfs
    else:
        qualtype_free(qt_prf, qt)
        prfs = parser_skip_until_semicolon(p, prfs)
        if parser_match(p, TokenType.SEMICOLON):
            prfs = parser_advance(p, prfs)
        dummy_prf, dummy_decl = decl_alloc()
        dummy_decl.type = nullptr
        return 0, dummy_prf, dummy_decl, prfs


@compile
def parse_regular_typedef(p: ParserRef, prfs: ParserProofs) -> struct[i8, DeclProof, ptr[Decl], ParserProofs]:
    """Parse a regular typedef (typedef int myint;).
    
    Returns (success, decl_prf, decl, updated_prfs).
    """
    ts: TypeParseState
    ts_ref: TypeParseStateRef = assume(ptr(ts), typeparse_nonnull)
    prfs = parse_type_specifiers(p, prfs, ts_ref)
    qt_prf, qt = build_qualtype_from_state(ts_ref)
    return parse_typedef_decl(p, prfs, qt_prf, qt)


@compile
def parse_regular_decl(p: ParserRef, prfs: ParserProofs, storage: i8) -> struct[i8, DeclProof, ptr[Decl], ParserProofs]:
    """Parse a regular declaration (function or variable).
    
    Returns (success, decl_prf, decl, updated_prfs).
    """
    ts: TypeParseState
    ts_ref: TypeParseStateRef = assume(ptr(ts), typeparse_nonnull)
    prfs = parse_type_specifiers(p, prfs, ts_ref)
    qt_prf, qt = build_qualtype_from_state(ts_ref)
    
    name: Span
    name, prfs = parse_declarator_name(p, prfs)
    
    if span_is_empty(name):
        qualtype_free(qt_prf, qt)
        prfs = parser_skip_until_semicolon(p, prfs)
        if parser_match(p, TokenType.SEMICOLON):
            prfs = parser_advance(p, prfs)
        dummy_prf, dummy_decl = decl_alloc()
        dummy_decl.type = nullptr
        return 0, dummy_prf, dummy_decl, prfs
    
    # Function declaration
    if parser_match(p, TokenType.LPAREN):
        decl_prf, decl, prfs = parse_function_decl(p, prfs, qt_prf, qt, name)
        decl.storage = storage
        return 1, decl_prf, decl, prfs
    else:
        # Variable declaration - skip for now
        qualtype_free(qt_prf, qt)
        prfs = parser_skip_until_semicolon(p, prfs)
        if parser_match(p, TokenType.SEMICOLON):
            prfs = parser_advance(p, prfs)
        dummy_prf, dummy_decl = decl_alloc()
        dummy_decl.type = nullptr
        return 0, dummy_prf, dummy_decl, prfs


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
                case DeclKind.Func:
                    # handle function
                    pass
            decl_free(decl_prf, decl)
    """
    lex_prf, lex_raw = lexer_create(source)
    defer(lexer_destroy, lex_prf, lex_raw)
    
    for lex in refine(lex_raw, lexer_nonnull):
        # Create parser state (no linear fields)
        parser: Parser = Parser()
        parser.lex = lex_raw
        parser.has_token = 0
        parser.error_count = 0
        
        # Create proofs struct (linear fields passed separately)
        # current_prf is dummy initially - will be set by first parser_advance
        prfs: ParserProofs
        prfs.lex_prf = move(lex_prf)
        prfs.current_prf = assume(linear(), "TokenProof")
        
        # Get first token - ptr(parser) for stack variable is always non-null
        p: ParserRef = assume(ptr(parser), parser_nonnull)
        prfs = parser_advance(p, prfs)

        def token_release_fn():
            lex_prf = token_release(p.current, prfs.current_prf, prfs.lex_prf)
        defer(token_release_fn)
        
        while p.current.type != TokenType.EOF:
            # Skip storage class specifiers
            storage: i8 = STORAGE_NONE
            while parser_match(p, TokenType.EXTERN) or parser_match(p, TokenType.STATIC):
                match p.current.type:
                    case TokenType.EXTERN:
                        storage = STORAGE_EXTERN
                    case TokenType.STATIC:
                        storage = STORAGE_STATIC
                    case _:
                        pass
                prfs = parser_advance(p, prfs)
            
            # Match on current token type for declaration dispatch
            match p.current.type:
                case TokenType.TYPEDEF:
                    prfs = parser_advance(p, prfs)
                    # Parse the underlying type
                    # Handle typedef struct/union/enum specially
                    match p.current.type:
                        case TokenType.STRUCT:
                            prfs = parser_advance(p, prfs)
                            ty_prf, ty, prfs = parse_struct_or_union(p, prfs, 0)
                            qt_prf, qt = make_qualtype(ty_prf, ty, QUAL_NONE)
                            success, decl_prf, decl, prfs = parse_typedef_decl(p, prfs, qt_prf, qt)
                            if success != 0:
                                yield decl_prf, decl
                            else:
                                decl_free(decl_prf, decl)
                        case TokenType.UNION:
                            prfs = parser_advance(p, prfs)
                            ty_prf, ty, prfs = parse_struct_or_union(p, prfs, 1)
                            qt_prf, qt = make_qualtype(ty_prf, ty, QUAL_NONE)
                            success, decl_prf, decl, prfs = parse_typedef_decl(p, prfs, qt_prf, qt)
                            if success != 0:
                                yield decl_prf, decl
                            else:
                                decl_free(decl_prf, decl)
                        case TokenType.ENUM:
                            prfs = parser_advance(p, prfs)
                            ty_prf, ty, prfs = parse_enum(p, prfs)
                            qt_prf, qt = make_qualtype(ty_prf, ty, QUAL_NONE)
                            success, decl_prf, decl, prfs = parse_typedef_decl(p, prfs, qt_prf, qt)
                            if success != 0:
                                yield decl_prf, decl
                            else:
                                decl_free(decl_prf, decl)
                        case _:
                            # Regular typedef: typedef int myint;
                            success, decl_prf, decl, prfs = parse_regular_typedef(p, prfs)
                            if success != 0:
                                yield decl_prf, decl
                            else:
                                decl_free(decl_prf, decl)
                
                case TokenType.STRUCT:
                    # Look ahead to determine if this is:
                    # 1. A struct definition: struct Name { ... };
                    # 2. A function/var with struct return type: struct Name* func();
                    # We parse the struct type first, then check what follows
                    prfs = parser_advance(p, prfs)
                    ty_prf, ty, prfs = parse_struct_or_union(p, prfs, 0)
                    
                    # Check if there's a pointer or identifier after the struct
                    # If so, this is a function/variable declaration, not just a struct
                    if parser_match(p, TokenType.STAR) or parser_match(p, TokenType.IDENTIFIER):
                        # This is a function/variable with struct return type
                        # Build QualType from the struct type
                        qt_prf, qt = make_qualtype(ty_prf, ty, QUAL_NONE)
                        
                        # Parse pointer indirections
                        while parser_match(p, TokenType.STAR):
                            prfs = parser_advance(p, prfs)
                            # Skip pointer qualifiers
                            while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE):
                                prfs = parser_advance(p, prfs)
                            qt_prf, qt = wrap_in_pointer(qt_prf, qt)
                        
                        # Parse declarator name
                        name: Span
                        name, prfs = parse_declarator_name(p, prfs)
                        
                        if not span_is_empty(name):
                            match p.current.type:
                                case TokenType.LPAREN:
                                    # Function declaration
                                    decl_prf, decl, prfs = parse_function_decl(p, prfs, qt_prf, qt, name)
                                    decl.storage = storage
                                    yield decl_prf, decl
                                case _:
                                    # Variable declaration - skip for now
                                    qualtype_free(qt_prf, qt)
                                    prfs = parser_skip_until_semicolon(p, prfs)
                                    if parser_match(p, TokenType.SEMICOLON):
                                        prfs = parser_advance(p, prfs)
                        else:
                            qualtype_free(qt_prf, qt)
                            prfs = parser_skip_until_semicolon(p, prfs)
                            if parser_match(p, TokenType.SEMICOLON):
                                prfs = parser_advance(p, prfs)
                    else:
                        # This is a struct definition/forward declaration
                        success, decl_prf, decl = try_make_struct_decl(ty_prf, ty, storage)
                        if success != 0:
                            yield decl_prf, decl
                        else:
                            decl_free(decl_prf, decl)
                        prfs = parser_skip_until_semicolon(p, prfs)
                        if parser_match(p, TokenType.SEMICOLON):
                            prfs = parser_advance(p, prfs)
                
                case TokenType.UNION:
                    # Same logic as STRUCT - check if this is a union type in a declaration
                    prfs = parser_advance(p, prfs)
                    ty_prf, ty, prfs = parse_struct_or_union(p, prfs, 1)
                    
                    if parser_match(p, TokenType.STAR) or parser_match(p, TokenType.IDENTIFIER):
                        # Function/variable with union return type
                        qt_prf, qt = make_qualtype(ty_prf, ty, QUAL_NONE)
                        
                        while parser_match(p, TokenType.STAR):
                            prfs = parser_advance(p, prfs)
                            while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE):
                                prfs = parser_advance(p, prfs)
                            qt_prf, qt = wrap_in_pointer(qt_prf, qt)
                        
                        name: Span
                        name, prfs = parse_declarator_name(p, prfs)
                        
                        if not span_is_empty(name):
                            match p.current.type:
                                case TokenType.LPAREN:
                                    decl_prf, decl, prfs = parse_function_decl(p, prfs, qt_prf, qt, name)
                                    decl.storage = storage
                                    yield decl_prf, decl
                                case _:
                                    qualtype_free(qt_prf, qt)
                                    prfs = parser_skip_until_semicolon(p, prfs)
                                    if parser_match(p, TokenType.SEMICOLON):
                                        prfs = parser_advance(p, prfs)
                        else:
                            qualtype_free(qt_prf, qt)
                            prfs = parser_skip_until_semicolon(p, prfs)
                            if parser_match(p, TokenType.SEMICOLON):
                                prfs = parser_advance(p, prfs)
                    else:
                        success, decl_prf, decl = try_make_union_decl(ty_prf, ty, storage)
                        if success != 0:
                            yield decl_prf, decl
                        else:
                            decl_free(decl_prf, decl)
                        prfs = parser_skip_until_semicolon(p, prfs)
                        if parser_match(p, TokenType.SEMICOLON):
                            prfs = parser_advance(p, prfs)
                
                case TokenType.ENUM:
                    prfs = parser_advance(p, prfs)
                    ty_prf, ty, prfs = parse_enum(p, prfs)
                    success, decl_prf, decl = try_make_enum_decl(ty_prf, ty, storage)
                    if success != 0:
                        yield decl_prf, decl
                    else:
                        decl_free(decl_prf, decl)
                    prfs = parser_skip_until_semicolon(p, prfs)
                    if parser_match(p, TokenType.SEMICOLON):
                        prfs = parser_advance(p, prfs)
                
                case _:
                    # Parse type and declarator
                    success, decl_prf, decl, prfs = parse_regular_decl(p, prfs, storage)
                    if success != 0:
                        yield decl_prf, decl
                    else:
                        decl_free(decl_prf, decl)
        