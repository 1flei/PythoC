"""
Lexer for C header files
Converts source text into a stream of tokens

Design: Uses linear types to ensure token lifetime is within lexer lifetime.
- lexer_create() returns (lexer, lexer_prf)
- lexer_next_token() consumes lexer_prf, returns (token, tk_prf)
- token_release() consumes tk_prf, returns lexer_prf
- lexer_destroy() consumes lexer_prf
"""

from pythoc import compile, inline, i32, i8, bool, ptr, array, nullptr, sizeof, void, char, refined, linear, struct, consume, assume
from pythoc.std.linear_wrapper import linear_wrap
from pythoc.std.refine_wrapper import nonnull_wrap
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import strlen
from pythoc.libc.ctype import isalpha, isdigit, isspace, isalnum

from pythoc.bindings.c_token import Token, TokenType, g_token_id_to_string, TokenRef, token_nonnull


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
def token_release(token: Token, tk_prf: TokenProof) -> LexerProof:
    """
    Release a token, returning the lexer proof.
    This allows getting the next token from the lexer.
    
    Args:
        token: The token to release (can be dropped)
        tk_prf: Token proof to consume
    
    Returns:
        lexer_prf: LexerProof for lexer operations
    """
    consume(tk_prf)
    return assume(linear(), "LexerProof")


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
        if c == char("/") and lexer_peek(lex, 1) == char("/"):
            lexer_advance(lex)
            lexer_advance(lex)
            while lex.pos < lex.length and lexer_current(lex) != char("\n"):
                lexer_advance(lex)
            continue
        
        # Skip /* block comments */
        if c == char("/") and lexer_peek(lex, 1) == char("*"):
            lexer_advance(lex)
            lexer_advance(lex)
            while lex.pos < lex.length:
                if lexer_current(lex) == char("*") and lexer_peek(lex, 1) == char("/"):
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
    
    c: i8 = lexer_current(lex)
    
    # Check for hex prefix 0x or 0X
    if c == char("0") and (lexer_peek(lex, 1) == char("x") or lexer_peek(lex, 1) == char("X")):
        lexer_advance(lex)
        lexer_advance(lex)
    
    # Read digits, dots, and hex letters
    while lex.pos < lex.length:
        c = lexer_current(lex)
        # Check if valid number character
        if isdigit(c) or c == char(".") or c == char("x") or c == char("X"):
            lexer_advance(lex)
        elif (c >= char("A") and c <= char("F")) or (c >= char("a") and c <= char("f")):
            lexer_advance(lex)
        else:
            break
    
    token.length = lex.pos - start_pos
    token.type = TokenType.NUMBER


# Generate single-char token mapping using Python metaprogramming
_single_char_tokens = [
    ('*', TokenType.STAR),
    ('(', TokenType.LPAREN),
    (')', TokenType.RPAREN),
    ('[', TokenType.LBRACKET),
    (']', TokenType.RBRACKET),
    ('{', TokenType.LBRACE),
    ('}', TokenType.RBRACE),
    (';', TokenType.SEMICOLON),
    (',', TokenType.COMMA),
    (':', TokenType.COLON),
    ('=', TokenType.EQUALS),
]


@compile
def lexer_next_token(lex: LexerRef, lexer_prf: LexerProof) -> struct[Token, TokenProof]:
    """
    Get next token from source, consuming lexer_prf and producing tk_prf.
    
    The returned tk_prf must be released via token_release() to get lexer_prf back,
    ensuring token lifetime is within lexer lifetime.
    
    Args:
        lex: Lexer reference
        lexer_prf: LexerProof (refined[linear, "LexerProof"]) proof of lexer ownership
    
    Returns:
        (token, tk_prf): Token and TokenProof (refined[linear, "TokenProof"]) bundle
    """
    # Consume lexer_prf upfront to avoid linear type issues in branches
    consume(lexer_prf)
    
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
            # Single character tokens and multi-char tokens
            token.start = lex.source + lex.pos
            token.length = 1
            token_found: bool = False
            
            for ch, tok_type in _single_char_tokens:
                if c == char(ch):
                    token.type = tok_type
                    lexer_advance(lex)
                    token_found = True
                    break
            
            if not token_found:
                # Multi-character tokens
                if c == char("."):
                    # Check for ellipsis '...'
                    if lexer_peek(lex, 1) == char(".") and lexer_peek(lex, 2) == char("."):
                        token.type = TokenType.ELLIPSIS
                        token.length = 3
                        lexer_advance(lex)
                        lexer_advance(lex)
                        lexer_advance(lex)
                        token_found = True
                
                if not token_found:
                    # Unknown character - treat as error
                    token.type = TokenType.ERROR
                    lexer_advance(lex)
    
    # Create and return token proof using refined type
    tk_prf: TokenProof = assume(linear(), "TokenProof")
    return token, tk_prf
