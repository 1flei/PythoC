"""
C Header Parser (pythoc compiled)

Parses C header files using the compiled lexer.
Builds AST nodes for: functions, structs, unions, enums, typedefs.

Design:
- All parsing functions are @compile decorated
- Uses Token stream from lexer (zero-copy)
- Builds CType, CFunc, CStruct, etc. AST nodes
- Uses Python metaprogramming for token matching
"""

from pythoc import compile, inline, i32, i8, bool, ptr, array, nullptr, sizeof, void, char, refine, assume, struct
from pythoc.libc.stdlib import malloc, realloc, free
from pythoc.libc.string import memcpy
from pythoc.std.refine_wrapper import nonnull_wrap

from pythoc.bindings.c_token import Token, TokenType, TokenRef, token_nonnull
from pythoc.bindings.lexer import (
    Lexer, LexerRef, lexer_nonnull, lexer_create, lexer_destroy,
    lexer_next_token_impl
)
from pythoc.bindings.c_ast import (
    BaseType, Signedness, DeclKind,
    CType, CTypeRef, ctype_nonnull, ctype_init,
    CParam, CParamRef, cparam_nonnull,
    CFunc, CFuncRef, cfunc_nonnull,
    CField, CFieldRef, cfield_nonnull,
    CStruct, CStructRef, cstruct_nonnull,
    CEnum, CEnumRef, cenum_nonnull,
    CEnumVal, CEnumValRef, cenumval_nonnull,
    CTypedef, CTypedefRef, ctypedef_nonnull,
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
    params: array[CParam, MAX_PARAMS]
    fields: array[CField, MAX_FIELDS]
    enum_vals: array[CEnumVal, MAX_ENUM_VALUES]


parser_nonnull, ParserRef = nonnull_wrap(ptr[Parser])


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
# Type parsing
# =============================================================================

@compile
def parse_type_spec(p: ParserRef, t: CTypeRef) -> void:
    """
    Parse C type specifiers into CType.
    Handles: const, signed/unsigned, base types, struct/union/enum names
    """
    ctype_init(t)
    has_long: i32 = 0
    
    while True:
        tok_type: i32 = p.current.type
        
        # const
        if tok_type == TokenType.CONST:
            t.is_const = 1
            parser_advance(p)
        # volatile (skip)
        elif tok_type == TokenType.VOLATILE:
            parser_advance(p)
        # signed
        elif tok_type == TokenType.SIGNED:
            t.sign = Signedness.SIGNED
            parser_advance(p)
        # unsigned
        elif tok_type == TokenType.UNSIGNED:
            t.sign = Signedness.UNSIGNED
            parser_advance(p)
        # void
        elif tok_type == TokenType.VOID:
            t.base = BaseType.VOID
            parser_advance(p)
        # char
        elif tok_type == TokenType.CHAR:
            t.base = BaseType.CHAR
            parser_advance(p)
        # short
        elif tok_type == TokenType.SHORT:
            t.base = BaseType.SHORT
            parser_advance(p)
        # int
        elif tok_type == TokenType.INT:
            t.base = BaseType.INT
            parser_advance(p)
        # long
        elif tok_type == TokenType.LONG:
            parser_advance(p)
            if has_long:
                t.base = BaseType.LONG_LONG
            else:
                has_long = 1
                t.base = BaseType.LONG
        # float
        elif tok_type == TokenType.FLOAT:
            t.base = BaseType.FLOAT
            parser_advance(p)
        # double
        elif tok_type == TokenType.DOUBLE:
            t.base = BaseType.DOUBLE
            parser_advance(p)
        # struct
        elif tok_type == TokenType.STRUCT:
            t.base = BaseType.STRUCT
            parser_advance(p)
            if parser_match(p, TokenType.IDENTIFIER):
                t.name_ptr = p.current.start
                t.name_len = p.current.length
                parser_advance(p)
            # Skip struct body if present
            if parser_match(p, TokenType.LBRACE):
                parser_skip_balanced(p, TokenType.LBRACE, TokenType.RBRACE)
        # union
        elif tok_type == TokenType.UNION:
            t.base = BaseType.UNION
            parser_advance(p)
            if parser_match(p, TokenType.IDENTIFIER):
                t.name_ptr = p.current.start
                t.name_len = p.current.length
                parser_advance(p)
            if parser_match(p, TokenType.LBRACE):
                parser_skip_balanced(p, TokenType.LBRACE, TokenType.RBRACE)
        # enum
        elif tok_type == TokenType.ENUM:
            t.base = BaseType.ENUM
            parser_advance(p)
            if parser_match(p, TokenType.IDENTIFIER):
                t.name_ptr = p.current.start
                t.name_len = p.current.length
                parser_advance(p)
            if parser_match(p, TokenType.LBRACE):
                parser_skip_balanced(p, TokenType.LBRACE, TokenType.RBRACE)
        # identifier (could be typedef name)
        elif tok_type == TokenType.IDENTIFIER:
            # Treat as typedef name
            t.base = BaseType.TYPEDEF_NAME
            t.name_ptr = p.current.start
            t.name_len = p.current.length
            parser_advance(p)
            break
        else:
            break
    
    # Parse pointer indirections
    while parser_match(p, TokenType.STAR):
        t.ptr_depth = t.ptr_depth + 1
        parser_advance(p)
        # Skip pointer qualifiers
        while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE):
            parser_advance(p)


@compile
def parse_declarator_name(p: ParserRef) -> struct[ptr[i8], i32]:
    """
    Parse declarator and return name pointer and length.
    Handles additional pointer stars and array brackets.
    Returns (nullptr, 0) if no name found.
    """
    name_ptr: ptr[i8] = ptr[i8](0)
    name_len: i32 = 0
    
    # Handle additional pointer stars in declarator
    while parser_match(p, TokenType.STAR):
        parser_advance(p)
        while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE):
            parser_advance(p)
    
    # Get name
    if parser_match(p, TokenType.IDENTIFIER):
        name_ptr = p.current.start
        name_len = p.current.length
        parser_advance(p)
    elif parser_match(p, TokenType.LPAREN):
        # Function pointer or grouped declarator - skip for now
        parser_skip_balanced(p, TokenType.LPAREN, TokenType.RPAREN)
    
    # Skip array dimensions
    while parser_match(p, TokenType.LBRACKET):
        parser_skip_balanced(p, TokenType.LBRACKET, TokenType.RBRACKET)
    
    return name_ptr, name_len


# =============================================================================
# Function parsing
# =============================================================================

@compile
def parse_func_params(p: ParserRef, func: CFuncRef) -> void:
    """Parse function parameters into func.params"""
    if not parser_expect(p, TokenType.LPAREN):
        return
    
    func.param_count = 0
    func.is_variadic = 0
    
    # Empty params or (void)
    if parser_match(p, TokenType.RPAREN):
        parser_advance(p)
        return
    
    if parser_match(p, TokenType.VOID):
        parser_advance(p)
        if parser_match(p, TokenType.RPAREN):
            parser_advance(p)
            return
    
    while True:
        # Check for ...
        if parser_match(p, TokenType.ELLIPSIS):
            func.is_variadic = 1
            parser_advance(p)
            break
        
        if func.param_count >= MAX_PARAMS:
            break
        
        # Parse parameter type
        param_ref: CParamRef = assume(ptr(p.params[func.param_count]), cparam_nonnull)
        type_ref: CTypeRef = assume(ptr(param_ref.type), ctype_nonnull)
        parse_type_spec(p, type_ref)
        
        # Parse parameter name
        name_ptr: ptr[i8]
        name_len: i32
        name_ptr, name_len = parse_declarator_name(p)
        param_ref.name_ptr = name_ptr
        param_ref.name_len = name_len
        
        func.param_count = func.param_count + 1
        
        if parser_match(p, TokenType.COMMA):
            parser_advance(p)
        else:
            break
    
    parser_expect(p, TokenType.RPAREN)
    
    # Copy params to heap
    if func.param_count > 0:
        size: i32 = func.param_count * sizeof(CParam)
        func.params = ptr[CParam](malloc(size))
        memcpy(func.params, ptr(p.params[0]), size)


# =============================================================================
# Struct/Union parsing
# =============================================================================

@compile
def parse_struct_fields(p: ParserRef, s: CStructRef) -> void:
    """Parse struct/union fields"""
    if not parser_expect(p, TokenType.LBRACE):
        return
    
    s.field_count = 0
    
    while not parser_match(p, TokenType.RBRACE) and not parser_match(p, TokenType.EOF):
        if s.field_count >= MAX_FIELDS:
            parser_skip_until_semicolon(p)
            parser_advance(p)
            continue
        
        # Parse field type
        field_ref: CFieldRef = assume(ptr(p.fields[s.field_count]), cfield_nonnull)
        type_ref: CTypeRef = assume(ptr(field_ref.type), ctype_nonnull)
        parse_type_spec(p, type_ref)
        
        # Parse field name
        name_ptr: ptr[i8]
        name_len: i32
        name_ptr, name_len = parse_declarator_name(p)
        field_ref.name_ptr = name_ptr
        field_ref.name_len = name_len
        field_ref.bit_width = -1
        
        # Check for bitfield
        if parser_match(p, TokenType.COLON):
            parser_advance(p)
            if parser_match(p, TokenType.NUMBER):
                # TODO: parse number value
                field_ref.bit_width = 0
                parser_advance(p)
        
        s.field_count = s.field_count + 1
        
        # Handle multiple declarators: int a, b, c;
        while parser_match(p, TokenType.COMMA):
            parser_advance(p)
            if s.field_count >= MAX_FIELDS:
                break
            # Copy type from previous field
            prev_field: CFieldRef = assume(ptr(p.fields[s.field_count - 1]), cfield_nonnull)
            field_ref = assume(ptr(p.fields[s.field_count]), cfield_nonnull)
            field_ref.type = prev_field.type
            
            name_ptr, name_len = parse_declarator_name(p)
            field_ref.name_ptr = name_ptr
            field_ref.name_len = name_len
            field_ref.bit_width = -1
            s.field_count = s.field_count + 1
        
        parser_expect(p, TokenType.SEMICOLON)
    
    parser_expect(p, TokenType.RBRACE)
    
    # Copy fields to heap
    if s.field_count > 0:
        size: i32 = s.field_count * sizeof(CField)
        s.fields = ptr[CField](malloc(size))
        memcpy(s.fields, ptr(p.fields[0]), size)


@compile
def parse_struct_or_union(p: ParserRef, is_union: i32) -> ptr[CStruct]:
    """Parse struct or union definition, return heap-allocated CStruct"""
    # Allocate struct
    s: ptr[CStruct] = ptr[CStruct](malloc(sizeof(CStruct)))
    s.is_union = is_union
    s.name_ptr = ptr[i8](0)
    s.name_len = 0
    s.fields = ptr[CField](0)
    s.field_count = 0
    
    # Get name if present
    if parser_match(p, TokenType.IDENTIFIER):
        s.name_ptr = p.current.start
        s.name_len = p.current.length
        parser_advance(p)
    
    # Parse fields if body present
    if parser_match(p, TokenType.LBRACE):
        s_ref: CStructRef = assume(s, cstruct_nonnull)
        parse_struct_fields(p, s_ref)
    
    return s


# =============================================================================
# Enum parsing
# =============================================================================

@compile
def parse_enum_values(p: ParserRef, e: CEnumRef) -> void:
    """Parse enum values"""
    if not parser_expect(p, TokenType.LBRACE):
        return
    
    e.value_count = 0
    
    while not parser_match(p, TokenType.RBRACE) and not parser_match(p, TokenType.EOF):
        if e.value_count >= MAX_ENUM_VALUES:
            break
        
        if parser_match(p, TokenType.IDENTIFIER):
            val_ref: CEnumValRef = assume(ptr(p.enum_vals[e.value_count]), cenumval_nonnull)
            val_ref.name_ptr = p.current.start
            val_ref.name_len = p.current.length
            val_ref.has_value = 0
            val_ref.value = 0
            parser_advance(p)
            
            # Check for explicit value
            if parser_match(p, TokenType.ASSIGN):
                parser_advance(p)
                val_ref.has_value = 1
                # Skip value expression (simplified)
                while not parser_match(p, TokenType.COMMA) and not parser_match(p, TokenType.RBRACE) and not parser_match(p, TokenType.EOF):
                    parser_advance(p)
            
            e.value_count = e.value_count + 1
            
            if parser_match(p, TokenType.COMMA):
                parser_advance(p)
        else:
            break
    
    parser_expect(p, TokenType.RBRACE)
    
    # Copy values to heap
    if e.value_count > 0:
        size: i32 = e.value_count * sizeof(CEnumVal)
        e.values = ptr[CEnumVal](malloc(size))
        memcpy(e.values, ptr(p.enum_vals[0]), size)


@compile
def parse_enum(p: ParserRef) -> ptr[CEnum]:
    """Parse enum definition"""
    e: ptr[CEnum] = ptr[CEnum](malloc(sizeof(CEnum)))
    e.name_ptr = ptr[i8](0)
    e.name_len = 0
    e.values = ptr[CEnumVal](0)
    e.value_count = 0
    
    if parser_match(p, TokenType.IDENTIFIER):
        e.name_ptr = p.current.start
        e.name_len = p.current.length
        parser_advance(p)
    
    if parser_match(p, TokenType.LBRACE):
        e_ref: CEnumRef = assume(e, cenum_nonnull)
        parse_enum_values(p, e_ref)
    
    return e


# =============================================================================
# Top-level declaration parsing
# =============================================================================

@compile
def parse_function(p: ParserRef, ret_type: CTypeRef, name_ptr: ptr[i8], name_len: i32) -> ptr[CFunc]:
    """Parse function declaration given return type and name"""
    func: ptr[CFunc] = ptr[CFunc](malloc(sizeof(CFunc)))
    func.name_ptr = name_ptr
    func.name_len = name_len
    func.ret_type = ret_type[0]  # Copy type
    func.params = ptr[CParam](0)
    func.param_count = 0
    func.is_variadic = 0
    
    func_ref: CFuncRef = assume(func, cfunc_nonnull)
    parse_func_params(p, func_ref)
    
    # Skip function body if present
    if parser_match(p, TokenType.LBRACE):
        parser_skip_balanced(p, TokenType.LBRACE, TokenType.RBRACE)
    elif parser_match(p, TokenType.SEMICOLON):
        parser_advance(p)
    
    return func


# =============================================================================
# Yield-based declaration iterator
# =============================================================================

@compile
def parse_declarations(source: ptr[i8]) -> struct[i32, ptr[void]]:
    """
    Yield declarations from source.
    Returns (DeclKind, ptr to declaration struct)
    
    Usage:
        for kind, decl_ptr in parse_declarations(source):
            if kind == DeclKind.FUNC:
                func = ptr[CFunc](decl_ptr)
                ...
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
            while parser_match(p, TokenType.EXTERN) or parser_match(p, TokenType.STATIC):
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
                s: ptr[CStruct] = parse_struct_or_union(p, 0)
                if s.name_len > 0:
                    yield DeclKind.STRUCT, ptr[void](s)
                parser_skip_until_semicolon(p)
                if parser_match(p, TokenType.SEMICOLON):
                    parser_advance(p)
                continue
            
            # union
            if parser_match(p, TokenType.UNION):
                parser_advance(p)
                u: ptr[CStruct] = parse_struct_or_union(p, 1)
                if u.name_len > 0:
                    yield DeclKind.UNION, ptr[void](u)
                parser_skip_until_semicolon(p)
                if parser_match(p, TokenType.SEMICOLON):
                    parser_advance(p)
                continue
            
            # enum
            if parser_match(p, TokenType.ENUM):
                parser_advance(p)
                e: ptr[CEnum] = parse_enum(p)
                if e.name_len > 0:
                    yield DeclKind.ENUM, ptr[void](e)
                parser_skip_until_semicolon(p)
                if parser_match(p, TokenType.SEMICOLON):
                    parser_advance(p)
                continue
            
            # Parse type and declarator
            t: CType = CType()
            t_ref: CTypeRef = assume(ptr(t), ctype_nonnull)
            parse_type_spec(p, t_ref)
            
            name_ptr: ptr[i8]
            name_len: i32
            name_ptr, name_len = parse_declarator_name(p)
            
            if name_len == 0:
                parser_skip_until_semicolon(p)
                if parser_match(p, TokenType.SEMICOLON):
                    parser_advance(p)
                continue
            
            # Function declaration
            if parser_match(p, TokenType.LPAREN):
                func: ptr[CFunc] = parse_function(p, t_ref, name_ptr, name_len)
                yield DeclKind.FUNC, ptr[void](func)
            else:
                # Variable declaration - skip
                parser_skip_until_semicolon(p)
                if parser_match(p, TokenType.SEMICOLON):
                    parser_advance(p)
        
        lexer_destroy(prf, lex_raw)
