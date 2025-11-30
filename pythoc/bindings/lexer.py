"""
Lexer for C header files
Converts source text into a stream of tokens
"""

from pythoc import compile, inline, i32, i8, bool, ptr, array, nullptr, sizeof, void, char
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import strlen
from pythoc.libc.ctype import isalpha, isdigit, isspace, isalnum

from pythoc.bindings.c_token import Token, TokenType, g_token_id_to_string


@compile
class Lexer:
    """Lexer state"""
    source: ptr[i8]              # Input source code
    pos: i32                     # Current position in source
    line: i32                    # Current line number (1-based)
    col: i32                     # Current column number (1-based)
    length: i32                  # Total source length


@compile
def lexer_create(source: ptr[i8]) -> ptr[Lexer]:
    """Create and initialize a new lexer"""
    lex: ptr[Lexer] = ptr[Lexer](malloc(sizeof(Lexer)))
    lex.source = source
    lex.pos = 0
    lex.line = 1
    lex.col = 1
    lex.length = strlen(source)
    return lex


@compile
def lexer_destroy(lex: ptr[Lexer]) -> void:
    """Free lexer memory"""
    free(lex)


@compile
def lexer_peek(lex: ptr[Lexer], offset: i32) -> i8:
    """Peek ahead at character without advancing"""
    pos: i32 = lex.pos + offset
    if pos >= lex.length:
        return 0  # EOF
    return lex.source[pos]


@compile
def lexer_current(lex: ptr[Lexer]) -> i8:
    """Get current character"""
    return lexer_peek(lex, 0)


@compile
def lexer_advance(lex: ptr[Lexer]) -> void:
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
def lexer_skip_whitespace(lex: ptr[Lexer]) -> void:
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
def is_keyword(word: ptr[i8]) -> TokenType:
    """Check if identifier is a C keyword, return token type or 0"""
    def word_equal_to(key) -> bool:
        for i in range(len(key)):
            if word[i] != char(key[i]):
                return False
        return True
    for token_id, token_str in g_token_id_to_string.items():
        if word_equal_to(token_str):
            return token_id
    return TokenType.ERROR  # Not a keyword


@compile
def lexer_read_identifier(lex: ptr[Lexer], token: ptr[Token]) -> void:
    """Read identifier or keyword"""
    i: i32 = 0
    
    # First character must be alpha or underscore
    c: i8 = lexer_current(lex)
    while lex.pos < lex.length and (isalnum(c) or c == char("_")):
        if i < 255:
            token.text[i] = c
            i = i + 1
        lexer_advance(lex)
        c = lexer_current(lex)
    
    token.text[i] = 0  # Null terminator
    
    # Check if it's a keyword
    kw_type: i32 = is_keyword(token.text)
    if kw_type != 0:
        token.type = kw_type
    else:
        token.type = TokenType.IDENTIFIER


@compile
def lexer_read_number(lex: ptr[Lexer], token: ptr[Token]) -> void:
    """Read numeric literal (simplified, handles decimal and hex)"""
    i: i32 = 0
    c: i8 = lexer_current(lex)
    
    # Check for hex prefix 0x or 0X
    if c == char("0") and (lexer_peek(lex, 1) == char("x") or lexer_peek(lex, 1) == char("X")):
        token.text[i] = c
        i = i + 1
        lexer_advance(lex)
        c = lexer_current(lex)
        token.text[i] = c
        i = i + 1
        lexer_advance(lex)
        c = lexer_current(lex)
    
    # Read digits, dots, and hex letters
    while lex.pos < lex.length:
        c = lexer_current(lex)
        # Check if valid number character
        if isdigit(c) or c == char(".") or c == char("x") or c == char("X"):
            if i < 255:
                token.text[i] = c
                i = i + 1
            lexer_advance(lex)
        elif (c >= char("A") and c <= char("F")) or (c >= char("a") and c <= char("f")):
            if i < 255:
                token.text[i] = c
                i = i + 1
            lexer_advance(lex)
        else:
            break
    
    token.text[i] = 0
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
def lexer_next_token(lex: ptr[Lexer], token: ptr[Token]) -> i32:
    """
    Get next token from source
    Returns 1 on success, 0 on EOF
    """
    lexer_skip_whitespace(lex)
    
    # Check for EOF
    if lex.pos >= lex.length:
        token.type = TokenType.EOF
        token.text[0] = 0
        token.line = lex.line
        token.col = lex.col
        return 0
    
    # Record token position
    token.line = lex.line
    token.col = lex.col
    
    c: i8 = lexer_current(lex)
    
    # Identifier or keyword (starts with letter or underscore)
    if isalpha(c) or c == char("_"):
        lexer_read_identifier(lex, token)
        return 1
    
    # Number (starts with digit)
    if isdigit(c):
        lexer_read_number(lex, token)
        return 1
    
    # Single character tokens - generated by metaprogramming
    token.text[0] = c
    token.text[1] = 0
    
    for ch, tok_type in _single_char_tokens:
        if c == char(ch):
            token.type = tok_type
            lexer_advance(lex)
            return 1
    
    # Multi-character tokens
    if c == char("."):
        # Check for ellipsis '...'
        if lexer_peek(lex, 1) == char(".") and lexer_peek(lex, 2) == char("."):
            token.type = TokenType.ELLIPSIS
            token.text[0] = char(".")
            token.text[1] = char(".")
            token.text[2] = char(".")
            token.text[3] = 0
            lexer_advance(lex)
            lexer_advance(lex)
            lexer_advance(lex)
            return 1
    
    # Unknown character - treat as error
    token.type = TokenType.ERROR
    lexer_advance(lex)
    return 1
