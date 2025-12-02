"""
Lexer for C header files
Converts source text into a stream of tokens

Design: Uses linear types to ensure token lifetime is within lexer lifetime.
- lexer_create() returns (lexer_prf, lexer)
- lexer_next_token(lex, lex_prf) -> (token, tk_prf, lex_prf) - lexer can produce multiple tokens
- token_release(token, tk_prf, lex_prf) -> lex_prf - token lifetime must be within lexer lifetime
- lexer_destroy(lex_prf, lex) - consumes lexer_prf
"""

from pythoc import compile, inline, i32, i8, bool, ptr, array, nullptr, sizeof, void, char, refine, refined, linear, struct, consume, assume
from pythoc.std.linear_wrapper import linear_wrap
from pythoc.std.refine_wrapper import nonnull_wrap
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import strlen
from pythoc.libc.ctype import isalpha, isdigit, isspace, isalnum

from pythoc.bindings.c_token import Token, TokenType, g_token_id_to_string, TokenRef, token_nonnull, _operator_to_token


@compile
class Lexer:
    """Lexer state"""
    source: ptr[i8]              # Input source code
    pos: i32                     # Current position in source
    line: i32                    # Current line number (1-based)
    col: i32                     # Current column number (1-based)
    length: i32                  # Total source length


@compile
def lexer_create_raw(source: ptr[i8]) -> ptr[Lexer]:
    """Create and initialize a new lexer"""
    lex: ptr[Lexer] = ptr[Lexer](malloc(sizeof(Lexer)))
    lex.source = source
    lex.pos = 0
    lex.line = 1
    lex.col = 1
    lex.length = strlen(source)
    return lex


@compile
def lexer_destroy_raw(lex: ptr[Lexer]) -> void:
    """Free lexer memory"""
    free(lex)


# Define proof types using linear_wrap
LexerProof, lexer_create, lexer_destroy = linear_wrap(
    lexer_create_raw, lexer_destroy_raw, struct_name="LexerProof")

# TokenProof is also a refined linear type with tag
TokenProof = refined[linear, "TokenProof"]

lexer_nonnull, LexerRef = nonnull_wrap(ptr[Lexer])


@compile
def token_release(token: Token, tk_prf: TokenProof, lex_prf: LexerProof) -> LexerProof:
    """
    Release a token, verifying its lifetime is within lexer lifetime.
    
    The lex_prf parameter ensures that token cannot outlive the lexer:
    - To release a token, you must have the lexer proof
    - This proves the lexer is still alive when the token is released
    
    Args:
        token: The token to release (value is dropped)
        tk_prf: Token proof to consume
        lex_prf: Lexer proof (passed through to verify lifetime)
    
    Returns:
        lex_prf: LexerProof passed through unchanged
    """
    consume(tk_prf)
    # Pass through lex_prf - this enforces that lexer outlives token
    return lex_prf


@compile
def lexer_peek(lex: LexerRef, offset: i32) -> i8:
    """Peek ahead at character without advancing"""
    pos: i32 = lex.pos + offset
    if pos >= lex.length:
        return 0  # EOF
    return lex.source[pos]


@compile
def lexer_current(lex: LexerRef) -> i8:
    """Get current character"""
    return lexer_peek(lex, 0)


@compile
def lexer_advance(lex: LexerRef) -> void:
    """Advance to next character, tracking line and column"""
    if lex.pos >= lex.length:
        return
    
    c: i8 = lex.source[lex.pos]
    lex.pos = lex.pos + 1
    
    if c == char("\n"):
        lex.line = lex.line + 1
        lex.col = 1
    else:
        lex.col = lex.col + 1


@inline
def next_token_is_keyword(lex: LexerRef, keyword) -> bool:
    """Check if current position matches keyword string"""
    for i, ch in enumerate(keyword):
        if lexer_peek(lex, i) != char(ch):
            return False
    return True


@compile
def lexer_skip_whitespace(lex: LexerRef) -> void:
    """Skip whitespace and comments"""
    while lex.pos < lex.length:
        c: i8 = lexer_current(lex)
        
        # Skip whitespace
        if isspace(c):
            lexer_advance(lex)
            continue
        
        # Skip // line comments
        if next_token_is_keyword(lex, "//"):
            lexer_advance(lex)
            lexer_advance(lex)
            while lex.pos < lex.length and lexer_current(lex) != char("\n"):
                lexer_advance(lex)
            continue
        
        # Skip /* block comments */
        if next_token_is_keyword(lex, "/*"):
            lexer_advance(lex)
            lexer_advance(lex)
            while lex.pos < lex.length:
                if next_token_is_keyword(lex, "*/"):
                    lexer_advance(lex)
                    lexer_advance(lex)
                    break
                lexer_advance(lex)
            continue
        
        # Not whitespace or comment
        break


@compile
def is_keyword(start: ptr[i8], length: i32) -> TokenType:
    """Check if token text is a C keyword, return token type or ERROR"""
    # Use Python metaprogramming to generate keyword checks at compile time
    for token_id, token_str in g_token_id_to_string.items():
        kw_len = len(token_str)
        if length == kw_len:
            # Check each character
            matches: bool = True
            for i in range(kw_len):
                if start[i] != char(token_str[i]):
                    matches = False
                    break
            if matches:
                return token_id
    return TokenType.ERROR


@compile
def lexer_read_identifier(lex: LexerRef, token: TokenRef) -> void:
    """Read identifier or keyword (zero-copy)"""
    token.start = lex.source + lex.pos
    start_pos: i32 = lex.pos
    
    c: i8 = lexer_current(lex)
    while lex.pos < lex.length:
        c = lexer_current(lex)
        if not (isalnum(c) or c == char("_")):
            break
        lexer_advance(lex)
    
    token.length = lex.pos - start_pos
    
    # Check if it's a keyword
    kw_type: i32 = is_keyword(token.start, token.length)
    if kw_type != TokenType.ERROR:
        token.type = kw_type
    else:
        token.type = TokenType.IDENTIFIER


@compile
def lexer_read_number(lex: LexerRef, token: TokenRef) -> void:
    """Read numeric literal (zero-copy, handles decimal and hex)"""
    token.start = lex.source + lex.pos
    start_pos: i32 = lex.pos
    
    # Check for hex prefix 0x or 0X
    if next_token_is_keyword(lex, "0x") or next_token_is_keyword(lex, "0X"):
        lexer_advance(lex)
        lexer_advance(lex)
    
    # Read digits, dots, and hex letters
    while lex.pos < lex.length:
        c: i8 = lexer_current(lex)
        # Check if valid number character
        if isdigit(c) or c == char(".") or c == char("x") or c == char("X"):
            lexer_advance(lex)
        elif (c >= char("A") and c <= char("F")) or (c >= char("a") and c <= char("f")):
            lexer_advance(lex)
        else:
            break
    
    token.length = lex.pos - start_pos
    token.type = TokenType.NUMBER


@compile
def try_match_operator(lex: LexerRef, token: TokenRef) -> bool:
    """
    Try to match an operator at current position.
    Uses Python metaprogramming to generate checks for all operators (longest match first).
    
    Returns:
        True if an operator was matched, False otherwise
    """
    token.start = lex.source + lex.pos
    
    # Use Python metaprogramming to generate operator checks at compile time
    # Operators are already sorted by length (descending) in _operator_to_token
    for op_str, op_type in _operator_to_token:
        op_len = len(op_str)
        # Check if operator matches at current position
        matches: bool = True
        for i in range(op_len):
            if lexer_peek(lex, i) != char(op_str[i]):
                matches = False
                break
        
        if matches:
            token.type = op_type
            token.length = op_len
            # Advance lexer position
            for _ in range(op_len):
                lexer_advance(lex)
            return True
    
    return False


@compile
def lexer_next_token_impl(lex: LexerRef) -> Token:
    """
    Get next token from source, consuming lexer_prf and producing tk_prf.
    
    The returned tk_prf must be released via token_release() to get lexer_prf back,
    ensuring token lifetime is within lexer lifetime.
    """
    # Consume lexer_prf upfront to avoid linear type issues in branches
    token: Token = Token()
    lexer_skip_whitespace(lex)
    
    # Check for EOF
    if lex.pos >= lex.length:
        token.type = TokenType.EOF
        token.start = lex.source + lex.pos
        token.length = 0
        token.line = lex.line
        token.col = lex.col
    else:
        # Record token position
        token.line = lex.line
        token.col = lex.col
        
        c: i8 = lexer_current(lex)
        
        # Identifier or keyword (starts with letter or underscore)
        if isalpha(c) or c == char("_"):
            token_ref = assume(ptr(token), token_nonnull)
            lexer_read_identifier(lex, token_ref)
        # Number (starts with digit)
        elif isdigit(c):
            token_ref = assume(ptr(token), token_nonnull)
            lexer_read_number(lex, token_ref)
        else:
            # Operators and punctuation - use unified operator matching
            token_ref = assume(ptr(token), token_nonnull)
            if not try_match_operator(lex, token_ref):
                # Unknown character - treat as error
                token.type = TokenType.ERROR
                token.start = lex.source + lex.pos
                token.length = 1
                lexer_advance(lex)
    return token


@compile
def lexer_next_token(lex: LexerRef, lexer_prf: LexerProof) -> struct[Token, TokenProof, LexerProof]:
    """
    Get next token with proof, preserving lexer proof.
    
    This design allows:
    - Multiple tokens to exist simultaneously (each with its own tk_prf)
    - Lexer to continue producing tokens (lex_prf is returned)
    - Token lifetime to be bounded by lexer lifetime (enforced at token_release)
    
    Args:
        lex: Lexer reference
        lexer_prf: Lexer proof (passed through)
    
    Returns:
        token: The parsed token
        tk_prf: Token proof (must be released via token_release)
        lex_prf: Lexer proof (passed through for continued use)
    """
    token: Token = lexer_next_token_impl(lex)
    
    # Create token proof - represents a live token
    tk_prf: TokenProof = assume(linear(), "TokenProof")
    # Pass through lexer proof - lexer is still valid
    return token, tk_prf, lexer_prf


@compile
def lex_tokens(lex: LexerRef) -> Token:
    """
    Yield-based iterator for lexing tokens from lex.
    """
    while lex.pos < lex.length:
        token: Token = lexer_next_token_impl(lex)
        yield token


@compile
def tokens_from_source(source: ptr[i8]) -> Token:
    """
    Yield-based iterator for lexing tokens from lex.
    """
    prf, lex_raw = lexer_create(source)
    for lex in refine(lex_raw, lexer_nonnull):
        for tk in lex_tokens(lex):
            yield tk
    lexer_destroy(prf, lex_raw)