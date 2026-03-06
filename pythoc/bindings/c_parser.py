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
    char, refine, assume, struct, consume, linear, defer, effect, enum, move
)
from pythoc.std import mem  # Sets up default mem effect
from pythoc.libc.string import memcpy
from pythoc.std.refinement import nonnull

from pythoc.bindings.c_token import Token, TokenType, TokenRef, token_nonnull
from pythoc.bindings.lexer import (
    Lexer, LexerRef, lexer_nonnull, lexer_create, lexer_destroy,
    lexer_next_token, token_release, LexerProof, TokenProof
)
from pythoc.bindings.c_ast import (
    # Core types
    Span, span_empty, span_is_empty, span_eq_cstr,
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
    _ctype_free_deep,
    # Clone functions
    qualtype_clone_deep,
    # Constants
    QUAL_NONE, QUAL_CONST, QUAL_VOLATILE,
    STORAGE_NONE, STORAGE_EXTERN, STORAGE_STATIC,
    # Expression AST
    ExprKind, Expr, expr_alloc, expr_alloc_array, expr_free_deep, expr_eval_const,
    # Statement AST
    StmtKind, Stmt, stmt_alloc, stmt_alloc_array, stmt_free_deep,
)


# =============================================================================
# Parse Error Types
# =============================================================================

@enum(i32)
class ParseErrorCode:
    """Error codes for programmatic handling"""
    NONE: None
    UNEXPECTED_TOKEN: None
    EXPECTED_IDENTIFIER: None
    EXPECTED_SEMICOLON: None
    EXPECTED_RBRACE: None
    EXPECTED_RPAREN: None
    MAX_PARAMS_EXCEEDED: None
    MAX_FIELDS_EXCEEDED: None
    MAX_ENUM_VALUES_EXCEEDED: None
    NULL_LEXER: None


@compile
class ParseError:
    """Parse error information"""
    line: i32
    col: i32
    error_code: ParseErrorCode


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


parser_nonnull, ParserRef = nonnull(ptr[Parser])


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
def parser_add_error(p: ParserRef, error_code: ParseErrorCode) -> void:
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
    """Check if current token matches type (tok_type is enum tag constant)"""
    return p.current.type == tok_type


@compile
def parser_expect(p: ParserRef, prfs: ParserProofs, tok_type: i32) -> struct[bool, ParserProofs]:
    """Expect and consume token, return (success, updated_proofs)"""
    if p.current.type != tok_type:
        match tok_type:
            case TokenType.SEMICOLON:
                parser_add_error(p, ParseErrorCode.EXPECTED_SEMICOLON)
            case TokenType.RBRACE:
                parser_add_error(p, ParseErrorCode.EXPECTED_RBRACE)
            case TokenType.RPAREN:
                parser_add_error(p, ParseErrorCode.EXPECTED_RPAREN)
            case TokenType.IDENTIFIER:
                parser_add_error(p, ParseErrorCode.EXPECTED_IDENTIFIER)
            case _:
                parser_add_error(p, ParseErrorCode.UNEXPECTED_TOKEN)
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
        if p.current.type == open_tok:
            depth = depth + 1
        elif p.current.type == close_tok:
            depth = depth - 1
        prfs = parser_advance(p, prfs)
    return prfs


@compile
def parser_skip_gcc_extensions(p: ParserRef, prfs: ParserProofs) -> ParserProofs:
    """Skip GCC __attribute__((...)), __extension__, __asm__(...) sequences."""
    while True:
        match p.current.type:
            case TokenType.ATTRIBUTE:
                # __attribute__((...)) - skip attribute keyword and double-parens
                prfs = parser_advance(p, prfs)
                if parser_match(p, TokenType.LPAREN):
                    prfs = parser_skip_balanced(p, prfs, TokenType.LPAREN, TokenType.RPAREN)
            case TokenType.EXTENSION:
                # __extension__ - just skip it
                prfs = parser_advance(p, prfs)
            case TokenType.ASM:
                # __asm__(...) - skip asm keyword and parens
                prfs = parser_advance(p, prfs)
                if parser_match(p, TokenType.LPAREN):
                    prfs = parser_skip_balanced(p, prfs, TokenType.LPAREN, TokenType.RPAREN)
            case _:
                break
    return prfs


# =============================================================================
# Expression Parser (Pratt / precedence climbing)
# =============================================================================

# Binding power table: (left_bp, right_bp) for infix operators
# Right-associative ops have right_bp < left_bp
_infix_bp = {
    # Assignment (right-assoc): bp 20
    TokenType.ASSIGN: (20, 19),
    TokenType.PLUS_ASSIGN: (20, 19),
    TokenType.MINUS_ASSIGN: (20, 19),
    TokenType.STAR_ASSIGN: (20, 19),
    TokenType.SLASH_ASSIGN: (20, 19),
    TokenType.PERCENT_ASSIGN: (20, 19),
    TokenType.LSHIFT_ASSIGN: (20, 19),
    TokenType.RSHIFT_ASSIGN: (20, 19),
    TokenType.AND_ASSIGN: (20, 19),
    TokenType.OR_ASSIGN: (20, 19),
    TokenType.XOR_ASSIGN: (20, 19),
    # Ternary handled specially
    # Logical OR
    TokenType.LOR: (40, 41),
    # Logical AND
    TokenType.LAND: (50, 51),
    # Bitwise OR
    TokenType.PIPE: (60, 61),
    # Bitwise XOR
    TokenType.CARET: (70, 71),
    # Bitwise AND
    TokenType.AMP: (80, 81),
    # Equality
    TokenType.EQ: (90, 91),
    TokenType.NE: (90, 91),
    # Comparison
    TokenType.LT: (100, 101),
    TokenType.GT: (100, 101),
    TokenType.LE: (100, 101),
    TokenType.GE: (100, 101),
    # Shift
    TokenType.LSHIFT: (110, 111),
    TokenType.RSHIFT: (110, 111),
    # Additive
    TokenType.PLUS: (120, 121),
    TokenType.MINUS: (120, 121),
    # Multiplicative
    TokenType.STAR: (130, 131),
    TokenType.SLASH: (130, 131),
    TokenType.PERCENT: (130, 131),
}

# Prefix binding powers
_PREFIX_BP = 150
_POSTFIX_BP = 160


@compile
def parse_number_value(start: ptr[i8], length: i32) -> i64:
    """Parse integer literal value from token text. Handles hex (0x) and decimal."""
    if length <= 0:
        return 0

    # Check for hex prefix
    if length >= 2 and start[0] == char("0") and (start[1] == char("x") or start[1] == char("X")):
        val: i64 = 0
        i: i32 = 2
        while i < length:
            c: i8 = start[i]
            if c >= char("0") and c <= char("9"):
                val = val * 16 + i64(c - char("0"))
            elif c >= char("a") and c <= char("f"):
                val = val * 16 + i64(c - char("a") + 10)
            elif c >= char("A") and c <= char("F"):
                val = val * 16 + i64(c - char("A") + 10)
            else:
                break
            i = i + 1
        return val

    # Decimal
    val: i64 = 0
    i: i32 = 0
    while i < length:
        c: i8 = start[i]
        if c >= char("0") and c <= char("9"):
            val = val * 10 + i64(c - char("0"))
        else:
            break  # Stop at non-digit (handles suffixes like 'L', 'U', etc.)
        i = i + 1
    return val


@compile
def parse_char_value(start: ptr[i8], length: i32) -> i64:
    """Parse character literal value from token text (e.g. 'a')."""
    if length >= 3 and start[0] == 39:  # '\''
        if start[1] == 92 and length >= 4:  # '\\'
            c2: i8 = start[2]
            if c2 == 110:   # 'n'
                return 10
            if c2 == 116:   # 't'
                return 9
            if c2 == 114:   # 'r'
                return 13
            if c2 == 48:    # '0'
                return 0
            if c2 == 92:    # '\\'
                return 92
            if c2 == 39:    # '\''
                return 39
            return i64(c2)
        return i64(start[1])
    return 0


@compile
def parse_expr_prefix(p: ParserRef, prfs: ParserProofs) -> struct[ptr[Expr], ParserProofs]:
    """Parse a prefix expression (atom or unary operator)."""
    e: ptr[Expr] = nullptr
    match p.current.type:
        # Integer literal
        case TokenType.NUMBER:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.IntLit)
            e.int_val = parse_number_value(p.current.start, p.current.length)
            e.span = span_from_token(p.current)
            prfs = parser_advance(p, prfs)
            return e, prfs

        # Character literal
        case TokenType.CHAR_LITERAL:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.CharLit)
            e.int_val = parse_char_value(p.current.start, p.current.length)
            e.span = span_from_token(p.current)
            prfs = parser_advance(p, prfs)
            return e, prfs

        # String literal
        case TokenType.STRING:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.StringLit)
            e.span = span_from_token(p.current)
            prfs = parser_advance(p, prfs)
            return e, prfs

        # Identifier
        case TokenType.IDENTIFIER:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.Ident)
            e.span = span_from_token(p.current)
            prfs = parser_advance(p, prfs)
            return e, prfs

        # sizeof
        case TokenType.SIZEOF:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.SizeofExpr)
            prfs = parser_advance(p, prfs)
            if parser_match(p, TokenType.LPAREN):
                prfs = parser_advance(p, prfs)
                # Check if next token is a type specifier -> sizeof(type)
                if is_type_specifier(p.current.type):
                    ts_sizeof: TypeParseState
                    ts_sizeof_ref: TypeParseStateRef = assume(ptr(ts_sizeof), typeparse_nonnull)
                    prfs = parse_type_specifiers(p, prfs, ts_sizeof_ref)
                    # Store type name in span for downstream use
                    e.span = ts_sizeof.name
                    # lhs stays nullptr to distinguish sizeof(type) from sizeof(expr)
                    # Clean up prebuilt type if inline body was parsed
                    if ts_sizeof.has_prebuilt != 0 and ts_sizeof.prebuilt_type != nullptr:
                        _ctype_free_deep(ts_sizeof.prebuilt_type)
                    _, prfs = parser_expect(p, prfs, TokenType.RPAREN)
                else:
                    e.lhs, prfs = parse_expr_bp(p, prfs, 0)
                    _, prfs = parser_expect(p, prfs, TokenType.RPAREN)
            else:
                e.lhs, prfs = parse_expr_bp(p, prfs, _PREFIX_BP)
            return e, prfs

        # Parenthesized expression (or cast)
        case TokenType.LPAREN:
            prfs = parser_advance(p, prfs)
            e, prfs = parse_expr_bp(p, prfs, 0)
            _, prfs = parser_expect(p, prfs, TokenType.RPAREN)
            return e, prfs

        # Unary prefix operators: -, +, ~, !, *, &, ++, --
        case TokenType.MINUS:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.UnaryOp)
            e.op = TokenType.MINUS
            prfs = parser_advance(p, prfs)
            e.lhs, prfs = parse_expr_bp(p, prfs, _PREFIX_BP)
            return e, prfs

        case TokenType.PLUS:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.UnaryOp)
            e.op = TokenType.PLUS
            prfs = parser_advance(p, prfs)
            e.lhs, prfs = parse_expr_bp(p, prfs, _PREFIX_BP)
            return e, prfs

        case TokenType.TILDE:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.UnaryOp)
            e.op = TokenType.TILDE
            prfs = parser_advance(p, prfs)
            e.lhs, prfs = parse_expr_bp(p, prfs, _PREFIX_BP)
            return e, prfs

        case TokenType.EXCLAIM:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.UnaryOp)
            e.op = TokenType.EXCLAIM
            prfs = parser_advance(p, prfs)
            e.lhs, prfs = parse_expr_bp(p, prfs, _PREFIX_BP)
            return e, prfs

        case TokenType.STAR:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.UnaryOp)
            e.op = TokenType.STAR
            prfs = parser_advance(p, prfs)
            e.lhs, prfs = parse_expr_bp(p, prfs, _PREFIX_BP)
            return e, prfs

        case TokenType.AMP:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.UnaryOp)
            e.op = TokenType.AMP
            prfs = parser_advance(p, prfs)
            e.lhs, prfs = parse_expr_bp(p, prfs, _PREFIX_BP)
            return e, prfs

        case TokenType.INC:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.UnaryOp)
            e.op = TokenType.INC
            prfs = parser_advance(p, prfs)
            e.lhs, prfs = parse_expr_bp(p, prfs, _PREFIX_BP)
            return e, prfs

        case TokenType.DEC:
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.UnaryOp)
            e.op = TokenType.DEC
            prfs = parser_advance(p, prfs)
            e.lhs, prfs = parse_expr_bp(p, prfs, _PREFIX_BP)
            return e, prfs

        case _:
            # Unknown prefix - record error and return 0
            parser_add_error(p, ParseErrorCode.UNEXPECTED_TOKEN)
            e = expr_alloc()
            e.kind = ExprKind(ExprKind.IntLit)
            e.int_val = 0
            return e, prfs


MAX_CALL_ARGS = 32


@compile
def parse_call_args(p: ParserRef, prfs: ParserProofs, arg_count: ptr[i32]) -> struct[ptr[Expr], ParserProofs]:
    """Parse function call arguments after '('."""
    arg_count[0] = 0

    if parser_match(p, TokenType.RPAREN):
        prfs = parser_advance(p, prfs)
        return nullptr, prfs

    # Use a scratch buffer of MAX_CALL_ARGS
    scratch: array[Expr, MAX_CALL_ARGS]
    while True:
        if arg_count[0] >= MAX_CALL_ARGS:
            break
        arg: ptr[Expr]
        arg, prfs = parse_expr_bp(p, prfs, 20)  # Above assignment level
        if arg != nullptr:
            scratch[arg_count[0]] = arg[0]
            # Don't free arg itself - we copied the value
            effect.mem.free(arg)
        arg_count[0] = arg_count[0] + 1

        if parser_match(p, TokenType.COMMA):
            prfs = parser_advance(p, prfs)
        else:
            break

    _, prfs = parser_expect(p, prfs, TokenType.RPAREN)

    if arg_count[0] > 0:
        args: ptr[Expr] = expr_alloc_array(arg_count[0])
        memcpy(args, ptr(scratch[0]), arg_count[0] * sizeof(Expr))
        return args, prfs

    return nullptr, prfs


@compile
def parse_expr_bp(p: ParserRef, prfs: ParserProofs, min_bp: i32) -> struct[ptr[Expr], ParserProofs]:
    """Pratt parser main loop. Parse expression with minimum binding power."""
    lhs: ptr[Expr]
    lhs, prfs = parse_expr_prefix(p, prfs)

    while True:
        tok: i32 = p.current.type

        # Check for postfix operators first
        if tok == TokenType.LPAREN:
            # Function call
            if _POSTFIX_BP < min_bp:
                break
            prfs = parser_advance(p, prfs)
            call_expr: ptr[Expr] = expr_alloc()
            call_expr.kind = ExprKind(ExprKind.Call)
            call_expr.lhs = lhs
            arg_count: i32 = 0
            call_expr.args, prfs = parse_call_args(p, prfs, ptr(arg_count))
            call_expr.arg_count = arg_count
            lhs = call_expr
            continue

        if tok == TokenType.LBRACKET:
            # Array index
            if _POSTFIX_BP < min_bp:
                break
            prfs = parser_advance(p, prfs)
            idx_expr: ptr[Expr] = expr_alloc()
            idx_expr.kind = ExprKind(ExprKind.Index)
            idx_expr.lhs = lhs
            idx_expr.rhs, prfs = parse_expr_bp(p, prfs, 0)
            _, prfs = parser_expect(p, prfs, TokenType.RBRACKET)
            lhs = idx_expr
            continue

        if tok == TokenType.DOT:
            # Member access
            if _POSTFIX_BP < min_bp:
                break
            prfs = parser_advance(p, prfs)
            mem_expr: ptr[Expr] = expr_alloc()
            mem_expr.kind = ExprKind(ExprKind.Member)
            mem_expr.lhs = lhs
            mem_expr.span = span_from_token(p.current)
            prfs = parser_advance(p, prfs)
            lhs = mem_expr
            continue

        if tok == TokenType.ARROW:
            # Arrow member access
            if _POSTFIX_BP < min_bp:
                break
            prfs = parser_advance(p, prfs)
            arrow_expr: ptr[Expr] = expr_alloc()
            arrow_expr.kind = ExprKind(ExprKind.Arrow)
            arrow_expr.lhs = lhs
            arrow_expr.span = span_from_token(p.current)
            prfs = parser_advance(p, prfs)
            lhs = arrow_expr
            continue

        if tok == TokenType.INC:
            # Postfix ++
            if _POSTFIX_BP < min_bp:
                break
            prfs = parser_advance(p, prfs)
            post_expr: ptr[Expr] = expr_alloc()
            post_expr.kind = ExprKind(ExprKind.PostfixOp)
            post_expr.op = TokenType.INC
            post_expr.lhs = lhs
            lhs = post_expr
            continue

        if tok == TokenType.DEC:
            # Postfix --
            if _POSTFIX_BP < min_bp:
                break
            prfs = parser_advance(p, prfs)
            post_expr2: ptr[Expr] = expr_alloc()
            post_expr2.kind = ExprKind(ExprKind.PostfixOp)
            post_expr2.op = TokenType.DEC
            post_expr2.lhs = lhs
            lhs = post_expr2
            continue

        # Ternary operator
        if tok == TokenType.QUESTION:
            ternary_bp: i32 = 30
            if ternary_bp < min_bp:
                break
            prfs = parser_advance(p, prfs)
            tern_expr: ptr[Expr] = expr_alloc()
            tern_expr.kind = ExprKind(ExprKind.Ternary)
            tern_expr.lhs = lhs
            tern_expr.rhs, prfs = parse_expr_bp(p, prfs, 0)
            _, prfs = parser_expect(p, prfs, TokenType.COLON)
            tern_expr.extra, prfs = parse_expr_bp(p, prfs, 30)  # Right-assoc
            lhs = tern_expr
            continue

        # Comma operator
        if tok == TokenType.COMMA:
            comma_bp: i32 = 10
            if comma_bp < min_bp:
                break
            prfs = parser_advance(p, prfs)
            comma_expr: ptr[Expr] = expr_alloc()
            comma_expr.kind = ExprKind(ExprKind.Comma)
            comma_expr.lhs = lhs
            comma_expr.rhs, prfs = parse_expr_bp(p, prfs, 11)
            lhs = comma_expr
            continue

        # Check infix binding power
        # Use metaprogramming to generate if-elif chain
        found: bool = False
        for op_tok, bp_pair in _infix_bp.items():
            l_bp = bp_pair[0]
            r_bp = bp_pair[1]
            if tok == op_tok:
                if l_bp < min_bp:
                    found = True
                    break
                prfs = parser_advance(p, prfs)
                bin_expr: ptr[Expr] = expr_alloc()
                # Determine if assignment or binary
                if l_bp == 20:
                    bin_expr.kind = ExprKind(ExprKind.Assign)
                else:
                    bin_expr.kind = ExprKind(ExprKind.BinaryOp)
                bin_expr.op = op_tok
                bin_expr.lhs = lhs
                bin_expr.rhs, prfs = parse_expr_bp(p, prfs, r_bp)
                lhs = bin_expr
                found = True
                break

        if not found:
            break
        # If found but bp was too low, we already broke

    return lhs, prfs


@compile
def parse_expression(p: ParserRef, prfs: ParserProofs) -> struct[ptr[Expr], ParserProofs]:
    """Parse a full expression (entry point)."""
    return parse_expr_bp(p, prfs, 0)


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
    TokenType.INLINE, TokenType.RESTRICT, TokenType.BUILTIN_VA_LIST,
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
    base_token: i32             # TokenType tag of base type (INT, CHAR, etc.)
    is_signed: i8           # 1 = signed, 0 = default, -1 = unsigned
    is_const: i8            # 1 if const
    is_volatile: i8         # 1 if volatile
    long_count: i8          # Number of 'long' keywords (0, 1, or 2)
    ptr_depth: i8           # Number of pointer indirections
    name: Span              # For struct/union/enum/typedef names
    prebuilt_type: ptr[CType]   # Pre-built CType for inline struct/union/enum bodies
    has_prebuilt: i8            # 1 if prebuilt_type is set (owns the CType)


typeparse_nonnull, TypeParseStateRef = nonnull(ptr[TypeParseState])


@compile
def typeparse_init(ts: TypeParseStateRef) -> void:
    """Initialize type parse state"""
    ts.base_token = TokenType.ERROR
    ts.is_signed = 0
    ts.is_const = 0
    ts.is_volatile = 0
    ts.long_count = 0
    ts.ptr_depth = 0
    ts.name = span_empty()
    ts.prebuilt_type = nullptr
    ts.has_prebuilt = 0


# =============================================================================
# Type parsing - build CType from tokens using match-case
# =============================================================================

@compile
def _typeparse_store_prebuilt(ts: TypeParseStateRef, ty_prf: CTypeProof, ty: ptr[CType]) -> void:
    """Store a pre-built CType (from parse_struct_or_union / parse_enum) into
    TypeParseState and extract its name.

    Consumes ty_prf.  Between this consume and the later assume(linear()) in
    build_base_ctype, the CType is held only by ts.prebuilt_type.  This is safe
    because parse_type_specifiers always returns cleanly and callers always call
    build_qualtype_from_state (which drains the prebuilt type) before any path
    that could free the parser.
    """
    ts.prebuilt_type = ty
    ts.has_prebuilt = 1

    # Extract name from the parsed compound type
    match ty[0]:
        case (CType.Struct, s):
            if s != nullptr:
                ts.name = s.name
        case (CType.Union, s):
            if s != nullptr:
                ts.name = s.name
        case (CType.Enum, e):
            if e != nullptr:
                ts.name = e.name
        case _:
            pass

    consume(ty_prf)


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
                if ts.base_token == TokenType.ERROR:
                    ts.base_token = TokenType.LONG
                prfs = parser_advance(p, prfs)
            case TokenType.FLOAT:
                ts.base_token = TokenType.FLOAT
                prfs = parser_advance(p, prfs)
            case TokenType.DOUBLE:
                ts.base_token = TokenType.DOUBLE
                prfs = parser_advance(p, prfs)
            # GCC extension specifiers - skip or handle
            case TokenType.INLINE:
                prfs = parser_advance(p, prfs)
            case TokenType.RESTRICT:
                prfs = parser_advance(p, prfs)
            case TokenType.BUILTIN_VA_LIST:
                ts.base_token = TokenType.BUILTIN_VA_LIST
                prfs = parser_advance(p, prfs)
            case TokenType.ATTRIBUTE:
                prfs = parser_skip_gcc_extensions(p, prfs)
            case TokenType.EXTENSION:
                prfs = parser_skip_gcc_extensions(p, prfs)
            # Compound types - parse body if present, then break
            case TokenType.STRUCT:
                ts.base_token = TokenType.STRUCT
                prfs = parser_advance(p, prfs)
                ty_prf: CTypeProof
                ty_ptr: ptr[CType]
                ty_prf, ty_ptr, prfs = parse_struct_or_union(p, prfs, 0)
                _typeparse_store_prebuilt(ts, ty_prf, ty_ptr)
                break
            case TokenType.UNION:
                ts.base_token = TokenType.UNION
                prfs = parser_advance(p, prfs)
                ty_prf: CTypeProof
                ty_ptr: ptr[CType]
                ty_prf, ty_ptr, prfs = parse_struct_or_union(p, prfs, 1)
                _typeparse_store_prebuilt(ts, ty_prf, ty_ptr)
                break
            case TokenType.ENUM:
                ts.base_token = TokenType.ENUM
                prfs = parser_advance(p, prfs)
                ty_prf: CTypeProof
                ty_ptr: ptr[CType]
                ty_prf, ty_ptr, prfs = parse_enum(p, prfs)
                _typeparse_store_prebuilt(ts, ty_prf, ty_ptr)
                break
            # Identifier (typedef name) - only if no base type yet AND no sign specifier
            case TokenType.IDENTIFIER:
                if ts.base_token == TokenType.ERROR and ts.is_signed == 0:
                    ts.base_token = TokenType.IDENTIFIER
                    ts.name = span_from_token(p.current)
                    prfs = parser_advance(p, prfs)
                break
            case _:
                break

    # Skip trailing GCC extensions after type specifiers
    prfs = parser_skip_gcc_extensions(p, prfs)

    # Parse pointer indirections
    while parser_match(p, TokenType.STAR):
        ts.ptr_depth = ts.ptr_depth + 1
        prfs = parser_advance(p, prfs)
        # Skip pointer qualifiers
        while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE) or parser_match(p, TokenType.RESTRICT):
            prfs = parser_advance(p, prfs)
        # Skip GCC extensions after pointer qualifiers
        prfs = parser_skip_gcc_extensions(p, prfs)

    return prfs


@compile
def build_base_ctype(ts: TypeParseStateRef) -> struct[CTypeProof, ptr[CType]]:
    """
    Build base CType from TypeParseState using match-case.
    Returns (proof, ptr) for linear ownership tracking.
    """
    # If a prebuilt type exists (inline struct/union/enum body was parsed),
    # use it directly instead of creating a stub
    if ts.has_prebuilt != 0 and ts.prebuilt_type != nullptr:
        ty: ptr[CType] = ts.prebuilt_type
        ts.prebuilt_type = nullptr
        ts.has_prebuilt = 0
        # Create a new proof for the already-allocated CType
        prf: CTypeProof = assume(linear(), "CTypeProof")
        return prf, ty

    # Default to int if no base type specified but signed/unsigned present
    # Extract tag as i32 for comparison with enum constants
    base: i32 = ts.base_token
    if base == TokenType.ERROR:
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
        case TokenType.BUILTIN_VA_LIST:
            name: Span = ts.name
            if span_is_empty(name):
                # Create a span for "__builtin_va_list" - use typedef reference
                return make_typedef_type(span_empty())
            return make_typedef_type(name)
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

# Declarator modifier kinds
DECL_MOD_NONE: i8 = 0
DECL_MOD_FUNC_PTR: i8 = 1   # Function pointer: (*name)(params)
DECL_MOD_ARRAY: i8 = 2       # Array: name[size]


@compile
class DeclaratorInfo:
    """Information about a parsed declarator"""
    name: Span
    modifier: i8             # DECL_MOD_NONE, DECL_MOD_FUNC_PTR, DECL_MOD_ARRAY
    array_size: i32          # Array size (-1 for unsized)
    extra_ptr_depth: i8      # Additional pointer stars in declarator
    # For function pointer: params stored in Parser.params scratch buffer
    func_ptr_param_count: i32
    func_ptr_is_variadic: i8
    func_ptr_params: ptr[ParamInfo]  # Heap-allocated params (caller takes ownership)


declinfo_nonnull, DeclaratorInfoRef = nonnull(ptr[DeclaratorInfo])


@compile
def declinfo_init(di: DeclaratorInfoRef) -> void:
    """Initialize a DeclaratorInfo"""
    di.name = span_empty()
    di.modifier = DECL_MOD_NONE
    di.array_size = -1
    di.extra_ptr_depth = 0
    di.func_ptr_param_count = 0
    di.func_ptr_is_variadic = 0
    di.func_ptr_params = nullptr


@compile
def parse_declarator(p: ParserRef, prfs: ParserProofs, di: DeclaratorInfoRef) -> ParserProofs:
    """
    Parse a declarator into DeclaratorInfo.
    Handles:
    - Simple: name
    - Pointer: *name, **name
    - Function pointer: (*name)(params)
    - Array: name[expr], name[]
    - GCC extensions between tokens
    """
    declinfo_init(di)

    # Handle additional pointer stars in declarator
    while parser_match(p, TokenType.STAR):
        di.extra_ptr_depth = di.extra_ptr_depth + 1
        prfs = parser_advance(p, prfs)
        while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE) or parser_match(p, TokenType.RESTRICT):
            prfs = parser_advance(p, prfs)

    # Skip GCC extensions before name
    prfs = parser_skip_gcc_extensions(p, prfs)

    # Check for function pointer pattern: ( * name )
    if parser_match(p, TokenType.LPAREN):
        # Peek: is this (*name) pattern?
        # Save state - we can look at token to check
        # Advance past '('
        prfs = parser_advance(p, prfs)

        # Count pointer stars inside parens
        inner_ptr_count: i8 = 0
        while parser_match(p, TokenType.STAR):
            inner_ptr_count = inner_ptr_count + 1
            prfs = parser_advance(p, prfs)
            # Skip qualifiers
            while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE) or parser_match(p, TokenType.RESTRICT):
                prfs = parser_advance(p, prfs)

        # Skip GCC extensions
        prfs = parser_skip_gcc_extensions(p, prfs)

        if inner_ptr_count > 0 and parser_match(p, TokenType.IDENTIFIER):
            # This is a function pointer: (*name)
            di.name = span_from_token(p.current)
            prfs = parser_advance(p, prfs)

            # Skip array brackets inside the grouped declarator (e.g. (*name[]))
            while parser_match(p, TokenType.LBRACKET):
                prfs = parser_skip_balanced(p, prfs, TokenType.LBRACKET, TokenType.RBRACKET)

            _, prfs = parser_expect(p, prfs, TokenType.RPAREN)

            # Now parse the parameter list
            if parser_match(p, TokenType.LPAREN):
                di.modifier = DECL_MOD_FUNC_PTR
                param_count: i32 = 0
                is_variadic: i8 = 0
                params: ptr[ParamInfo]
                params, prfs = parse_func_params(p, prfs, ptr(param_count), ptr(is_variadic))
                di.func_ptr_param_count = param_count
                di.func_ptr_is_variadic = is_variadic
                di.func_ptr_params = params
            else:
                # Just a grouped pointer declarator, add inner stars to ptr_depth
                di.extra_ptr_depth = di.extra_ptr_depth + inner_ptr_count
        elif inner_ptr_count > 0 and parser_match(p, TokenType.RPAREN):
            # Anonymous function pointer: (*)(params)
            prfs = parser_advance(p, prfs)
            if parser_match(p, TokenType.LPAREN):
                di.modifier = DECL_MOD_FUNC_PTR
                param_count: i32 = 0
                is_variadic: i8 = 0
                params: ptr[ParamInfo]
                params, prfs = parse_func_params(p, prfs, ptr(param_count), ptr(is_variadic))
                di.func_ptr_param_count = param_count
                di.func_ptr_is_variadic = is_variadic
                di.func_ptr_params = params
            else:
                di.extra_ptr_depth = di.extra_ptr_depth + inner_ptr_count
        else:
            # Not a function pointer pattern - this is a grouped expression
            # or something else. Skip the rest of the parens.
            # Parse name inside parens if any
            if parser_match(p, TokenType.IDENTIFIER):
                di.name = span_from_token(p.current)
                prfs = parser_advance(p, prfs)
            # Skip to closing paren
            depth: i32 = 1
            while depth > 0 and p.current.type != TokenType.EOF:
                if p.current.type == TokenType.LPAREN:
                    depth = depth + 1
                elif p.current.type == TokenType.RPAREN:
                    depth = depth - 1
                    if depth == 0:
                        prfs = parser_advance(p, prfs)
                        break
                prfs = parser_advance(p, prfs)
    elif parser_match(p, TokenType.IDENTIFIER):
        # Simple name
        di.name = span_from_token(p.current)
        prfs = parser_advance(p, prfs)

    # Parse array dimensions: name[expr] or name[]
    while parser_match(p, TokenType.LBRACKET):
        prfs = parser_advance(p, prfs)
        if parser_match(p, TokenType.RBRACKET):
            # Unsized array []
            di.modifier = DECL_MOD_ARRAY
            di.array_size = -1
            prfs = parser_advance(p, prfs)
        else:
            # Sized array [expr]
            di.modifier = DECL_MOD_ARRAY
            size_expr: ptr[Expr]
            size_expr, prfs = parse_expr_bp(p, prfs, 0)
            di.array_size = i32(expr_eval_const(size_expr))
            expr_free_deep(size_expr)
            _, prfs = parser_expect(p, prfs, TokenType.RBRACKET)

    # Skip GCC extensions after declarator
    prfs = parser_skip_gcc_extensions(p, prfs)

    return prfs


@compile
def parse_declarator_name(p: ParserRef, prfs: ParserProofs) -> struct[Span, ParserProofs]:
    """
    Parse declarator and return (name, updated_proofs).
    Compatibility wrapper around parse_declarator.
    Handles additional pointer stars and array brackets.
    Returns empty span if no name found.
    """
    di: DeclaratorInfo
    di_ref: DeclaratorInfoRef = assume(ptr(di), declinfo_nonnull)
    prfs = parse_declarator(p, prfs, di_ref)

    # Free any function pointer params we allocated
    if di.func_ptr_params != nullptr:
        effect.mem.free(di.func_ptr_params)

    return di.name, prfs


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
            parser_add_error(p, ParseErrorCode.MAX_PARAMS_EXCEEDED)
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

    # Skip GCC extensions after parameter list (e.g. __attribute__)
    prfs = parser_skip_gcc_extensions(p, prfs)

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
            parser_add_error(p, ParseErrorCode.MAX_FIELDS_EXCEEDED)
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
                bit_width = i32(parse_number_value(p.current.start, p.current.length))
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
                parser_add_error(p, ParseErrorCode.MAX_FIELDS_EXCEEDED)
                break

            # Deep clone type from previous field to avoid double-free
            prev_qt: ptr[QualType] = p.fields[field_count[0] - 1].type
            new_qt_prf, new_qt = qualtype_clone_deep(prev_qt)

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

    # Skip GCC extensions after struct/union closing brace
    prfs = parser_skip_gcc_extensions(p, prfs)

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
            parser_add_error(p, ParseErrorCode.MAX_ENUM_VALUES_EXCEEDED)
            break

        match p.current.type:
            case TokenType.IDENTIFIER:
                enum_name: Span = span_from_token(p.current)
                p.enum_vals[value_count[0]].name = enum_name
                p.enum_vals[value_count[0]].value = current_value
                p.enum_vals[value_count[0]].has_explicit_value = 0
                prfs = parser_advance(p, prfs)

                # Check for explicit value
                if parser_match(p, TokenType.ASSIGN):
                    prfs = parser_advance(p, prfs)
                    p.enum_vals[value_count[0]].has_explicit_value = 1
                    # Parse value expression and evaluate as constant
                    val_expr: ptr[Expr]
                    val_expr, prfs = parse_expr_bp(p, prfs, 20)  # Above assignment, below comma
                    explicit_val: i64 = expr_eval_const(val_expr)
                    p.enum_vals[value_count[0]].value = explicit_val
                    current_value = explicit_val
                    expr_free_deep(val_expr)

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

    # Skip GCC extensions after enum closing brace
    prfs = parser_skip_gcc_extensions(p, prfs)

    ty_prf, ty = make_enum_type(name, values, value_count, is_complete)
    return ty_prf, ty, prfs


# =============================================================================
# Statement Parser
# =============================================================================

MAX_BLOCK_STMTS = 256


@compile
def parse_statement(p: ParserRef, prfs: ParserProofs) -> struct[ptr[Stmt], ParserProofs]:
    """Parse a single statement. Returns (stmt_ptr, updated_proofs)."""
    s: ptr[Stmt] = nullptr
    match p.current.type:
        # Empty statement
        case TokenType.SEMICOLON:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.Empty)
            prfs = parser_advance(p, prfs)
            return s, prfs

        # Block statement
        case TokenType.LBRACE:
            return parse_block(p, prfs)

        # Return statement
        case TokenType.RETURN:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.Return)
            prfs = parser_advance(p, prfs)
            if not parser_match(p, TokenType.SEMICOLON):
                s.expr, prfs = parse_expression(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.SEMICOLON)
            return s, prfs

        # If statement
        case TokenType.IF:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.If)
            prfs = parser_advance(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.LPAREN)
            s.expr, prfs = parse_expression(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.RPAREN)
            s.body, prfs = parse_statement(p, prfs)
            if parser_match(p, TokenType.ELSE):
                prfs = parser_advance(p, prfs)
                s.else_body, prfs = parse_statement(p, prfs)
            return s, prfs

        # While statement
        case TokenType.WHILE:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.While)
            prfs = parser_advance(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.LPAREN)
            s.expr, prfs = parse_expression(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.RPAREN)
            s.body, prfs = parse_statement(p, prfs)
            return s, prfs

        # Do-while statement
        case TokenType.DO:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.DoWhile)
            prfs = parser_advance(p, prfs)
            s.body, prfs = parse_statement(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.WHILE)
            _, prfs = parser_expect(p, prfs, TokenType.LPAREN)
            s.expr, prfs = parse_expression(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.RPAREN)
            _, prfs = parser_expect(p, prfs, TokenType.SEMICOLON)
            return s, prfs

        # For statement
        case TokenType.FOR:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.For)
            prfs = parser_advance(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.LPAREN)
            # Init
            if not parser_match(p, TokenType.SEMICOLON):
                s.init_expr, prfs = parse_expression(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.SEMICOLON)
            # Condition
            if not parser_match(p, TokenType.SEMICOLON):
                s.expr, prfs = parse_expression(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.SEMICOLON)
            # Increment
            if not parser_match(p, TokenType.RPAREN):
                s.incr_expr, prfs = parse_expression(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.RPAREN)
            s.body, prfs = parse_statement(p, prfs)
            return s, prfs

        # Break
        case TokenType.BREAK:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.Break)
            prfs = parser_advance(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.SEMICOLON)
            return s, prfs

        # Continue
        case TokenType.CONTINUE:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.Continue)
            prfs = parser_advance(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.SEMICOLON)
            return s, prfs

        # Goto
        case TokenType.GOTO:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.Goto)
            prfs = parser_advance(p, prfs)
            if parser_match(p, TokenType.IDENTIFIER):
                s.label = span_from_token(p.current)
                prfs = parser_advance(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.SEMICOLON)
            return s, prfs

        # Switch
        case TokenType.SWITCH:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.Switch)
            prfs = parser_advance(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.LPAREN)
            s.expr, prfs = parse_expression(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.RPAREN)
            s.body, prfs = parse_statement(p, prfs)
            return s, prfs

        # Case
        case TokenType.CASE:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.Case)
            prfs = parser_advance(p, prfs)
            s.expr, prfs = parse_expression(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.COLON)
            # Parse the body statement after case label
            if not parser_match(p, TokenType.CASE) and not parser_match(p, TokenType.DEFAULT) and not parser_match(p, TokenType.RBRACE):
                s.body, prfs = parse_statement(p, prfs)
            return s, prfs

        # Default
        case TokenType.DEFAULT:
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.Default)
            prfs = parser_advance(p, prfs)
            _, prfs = parser_expect(p, prfs, TokenType.COLON)
            if not parser_match(p, TokenType.CASE) and not parser_match(p, TokenType.DEFAULT) and not parser_match(p, TokenType.RBRACE):
                s.body, prfs = parse_statement(p, prfs)
            return s, prfs

        case _:
            # Check if this looks like a local variable declaration (type specifier at stmt position)
            # Skip it since we only care about top-level declarations
            if parser_match(p, TokenType.INT) or parser_match(p, TokenType.CHAR) or parser_match(p, TokenType.VOID) or parser_match(p, TokenType.SHORT) or parser_match(p, TokenType.LONG) or parser_match(p, TokenType.FLOAT) or parser_match(p, TokenType.DOUBLE) or parser_match(p, TokenType.UNSIGNED) or parser_match(p, TokenType.SIGNED) or parser_match(p, TokenType.STRUCT) or parser_match(p, TokenType.UNION) or parser_match(p, TokenType.ENUM) or parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE) or parser_match(p, TokenType.STATIC) or parser_match(p, TokenType.EXTERN) or parser_match(p, TokenType.BUILTIN_VA_LIST):
                s = stmt_alloc()
                s.kind = StmtKind(StmtKind.Decl)
                prfs = parser_skip_until_semicolon(p, prfs)
                if parser_match(p, TokenType.SEMICOLON):
                    prfs = parser_advance(p, prfs)
                return s, prfs

            # Expression statement (or label)
            s = stmt_alloc()
            s.kind = StmtKind(StmtKind.Expr)
            s.expr, prfs = parse_expression(p, prfs)
            # Check if this is a label (identifier followed by ':')
            if parser_match(p, TokenType.COLON) and s.expr != nullptr:
                match s.expr.kind[0]:
                    case ExprKind.Ident:
                        s.kind = StmtKind(StmtKind.Label)
                        s.label = s.expr.span
                        expr_free_deep(s.expr)
                        s.expr = nullptr
                        prfs = parser_advance(p, prfs)
                        # Parse statement after label
                        if not parser_match(p, TokenType.RBRACE) and not parser_match(p, TokenType.EOF):
                            s.body, prfs = parse_statement(p, prfs)
                        return s, prfs
                    case _:
                        pass
            _, prfs = parser_expect(p, prfs, TokenType.SEMICOLON)
            return s, prfs


@compile
def parse_block(p: ParserRef, prfs: ParserProofs) -> struct[ptr[Stmt], ParserProofs]:
    """Parse a block statement { ... }."""
    s: ptr[Stmt] = stmt_alloc()
    s.kind = StmtKind(StmtKind.Block)

    ok: bool
    ok, prfs = parser_expect(p, prfs, TokenType.LBRACE)
    if not ok:
        return s, prfs

    # Parse statements into scratch (using a simple array)
    scratch: array[Stmt, MAX_BLOCK_STMTS]
    count: i32 = 0

    while not parser_match(p, TokenType.RBRACE) and not parser_match(p, TokenType.EOF):
        if count >= MAX_BLOCK_STMTS:
            # Too many statements - skip remaining
            prfs = parser_skip_balanced(p, prfs, TokenType.LBRACE, TokenType.RBRACE)
            break

        # Skip GCC extensions inside blocks
        prfs = parser_skip_gcc_extensions(p, prfs)
        if parser_match(p, TokenType.RBRACE) or parser_match(p, TokenType.EOF):
            break

        stmt_node: ptr[Stmt]
        stmt_node, prfs = parse_statement(p, prfs)
        if stmt_node != nullptr:
            scratch[count] = stmt_node[0]
            effect.mem.free(stmt_node)  # Free the node shell, we copied the value
            count = count + 1

    _, prfs = parser_expect(p, prfs, TokenType.RBRACE)

    if count > 0:
        s.stmts = stmt_alloc_array(count)
        memcpy(s.stmts, ptr(scratch[0]), count * sizeof(Stmt))
        s.stmt_count = count

    return s, prfs


# =============================================================================
# Top-level declaration parsing
# =============================================================================

@compile
def parse_function_decl(p: ParserRef, prfs: ParserProofs, ret_qt_prf: QualTypeProof, ret_qt: ptr[QualType], name: Span) -> struct[DeclProof, ptr[Decl], ParserProofs]:
    """Parse function declaration given return type and name.

    Takes ownership of ret_qt (via ret_qt_prf).
    """
    # Parse function type (takes ownership of ret_qt)
    func_ty_prf, func_ty, prfs = parse_function_type(p, prfs, ret_qt_prf, ret_qt)

    # Wrap in QualType (no qualifiers for function type)
    func_qt_prf, func_qt = make_qualtype(func_ty_prf, func_ty, QUAL_NONE)

    # Create declaration
    decl, decl_prf = decl_alloc()
    decl.kind = DeclKind(DeclKind.Func)
    decl.name = name
    decl.type = func_qt
    decl.storage = STORAGE_NONE
    consume(func_qt_prf)  # Transfer ownership to Decl

    # Skip GCC extensions before function body (e.g. __attribute__)
    prfs = parser_skip_gcc_extensions(p, prfs)

    # Parse function body if present, then immediately free it
    # (only declaration signatures matter for bindgen)
    # For Phase 1, use skip_balanced for speed — the statement parser is
    # available but compiles too slowly for complex function bodies.
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
        decl, decl_prf = decl_alloc()
        decl.kind = DeclKind(DeclKind.Struct)
        decl.name = name
        decl.type = qt
        decl.storage = storage
        consume(qt_prf)
        return 1, decl_prf, decl
    else:
        ctype_free(ty_prf, ty)
        # Return dummy decl - caller must free it
        dummy_decl, dummy_prf = decl_alloc()
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
        decl, decl_prf = decl_alloc()
        decl.kind = DeclKind(DeclKind.Union)
        decl.name = name
        decl.type = qt
        decl.storage = storage
        consume(qt_prf)
        return 1, decl_prf, decl
    else:
        ctype_free(ty_prf, ty)
        # Return dummy decl - caller must free it
        dummy_decl, dummy_prf = decl_alloc()
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
        decl, decl_prf = decl_alloc()
        decl.kind = DeclKind(DeclKind.Enum)
        decl.name = name
        decl.type = qt
        decl.storage = storage
        consume(qt_prf)
        return 1, decl_prf, decl
    else:
        ctype_free(ty_prf, ty)
        # Return dummy decl - caller must free it
        dummy_decl, dummy_prf = decl_alloc()
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
        while parser_match(p, TokenType.CONST) or parser_match(p, TokenType.VOLATILE) or parser_match(p, TokenType.RESTRICT):
            prfs = parser_advance(p, prfs)
        qt_prf, qt = wrap_in_pointer(qt_prf, qt)

    # Get typedef name using full declarator parser
    di: DeclaratorInfo
    di_ref: DeclaratorInfoRef = assume(ptr(di), declinfo_nonnull)
    prfs = parse_declarator(p, prfs, di_ref)

    # Apply extra pointer stars from declarator
    depth: i8 = 0
    while depth < di.extra_ptr_depth:
        qt_prf, qt = wrap_in_pointer(qt_prf, qt)
        depth = depth + 1

    # Handle function pointer modifier: typedef ret_type (*name)(params)
    if di.modifier == DECL_MOD_FUNC_PTR:
        # Build function type: Func(params) -> ret_type
        # qt is the return type
        func_ty_prf, func_ty = make_func_type(
            qt_prf, qt, di.func_ptr_params,
            di.func_ptr_param_count, di.func_ptr_is_variadic
        )
        # Wrap in pointer (function pointer)
        func_qt_prf, func_qt = make_qualtype(func_ty_prf, func_ty, QUAL_NONE)
        func_qt_prf, func_qt = wrap_in_pointer(func_qt_prf, func_qt)
        qt_prf = move(func_qt_prf)
        qt = func_qt
    elif di.modifier == DECL_MOD_ARRAY:
        # Build array type: Array(elem_type, size)
        arr_ty_prf, arr_ty = make_array_type(qt_prf, qt, di.array_size)
        qt_prf, qt = make_qualtype(arr_ty_prf, arr_ty, QUAL_NONE)
    else:
        # Free any allocated params that we won't use
        if di.func_ptr_params != nullptr:
            effect.mem.free(di.func_ptr_params)

    name: Span = di.name

    if not span_is_empty(name):
        decl, decl_prf = decl_alloc()
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
        dummy_decl, dummy_prf = decl_alloc()
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
        dummy_decl, dummy_prf = decl_alloc()
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
        dummy_decl, dummy_prf = decl_alloc()
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

    IMPORTANT: The source buffer must remain valid for the lifetime of the
    returned AST nodes, as all Span fields contain pointers into the source.
    Callers must ensure source outlives all returned Decl nodes.

    Usage:
        for decl_prf, decl in parse_declarations(source):
            match decl.kind:
                case DeclKind.Func:
                    # handle function
                    pass
            decl_free(decl_prf, decl)
        # After iteration completes, AST nodes still reference source buffer
        # Caller must keep source alive while using the AST
    """
    lex_raw, lex_prf = lexer_create(source)
    defer(lexer_destroy, lex_raw, lex_prf)

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

        # defer to clean up proofs at end of for-body
        def cleanup_prfs():
            lex_prf = token_release(p.current, prfs.current_prf, prfs.lex_prf)
        defer(cleanup_prfs)

        # Get first token - ptr(parser) for stack variable is always non-null
        p: ParserRef = assume(ptr(parser), parser_nonnull)
        prfs = parser_advance(p, prfs)

        while p.current.type != TokenType.EOF:
            # Skip GCC extensions at top level
            prfs = parser_skip_gcc_extensions(p, prfs)
            if p.current.type == TokenType.EOF:
                break

            # Skip storage class specifiers
            storage: i8 = STORAGE_NONE
            while parser_match(p, TokenType.EXTERN) or parser_match(p, TokenType.STATIC) or parser_match(p, TokenType.INLINE):
                match p.current.type:
                    case TokenType.EXTERN:
                        storage = STORAGE_EXTERN
                    case TokenType.STATIC:
                        storage = STORAGE_STATIC
                    case _:
                        pass
                prfs = parser_advance(p, prfs)
                # Skip GCC extensions after storage class
                prfs = parser_skip_gcc_extensions(p, prfs)

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