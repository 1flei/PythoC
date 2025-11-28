"""
Lexer for C header files
Converts source text into a stream of tokens
"""

from pythoc import compile, i32, i8, ptr, array, nullptr, sizeof, void, char
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import strlen
from pythoc.libc.ctype import isalpha, isdigit, isspace, isalnum

from .c_token import Token, TokenType


@compile
def str_equal(s1: ptr[i8], s2: ptr[i8]) -> i32:
    """Compare two strings for equality, return 1 if equal, 0 if not"""
    i: i32 = 0
    while s1[i] != 0 and s2[i] != 0:
        if s1[i] != s2[i]:
            return 0
        i = i + 1
    return s1[i] == s2[i]


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
def is_keyword(word: ptr[i8]) -> i32:
    """Check if identifier is a C keyword, return token type or 0"""
    # int
    if word[0] == char("i") and word[1] == char("n") and word[2] == char("t") and word[3] == 0:
        return TokenType.INT
    # char
    if word[0] == char("c") and word[1] == char("h") and word[2] == char("a") and word[3] == char("r") and word[4] == 0:
        return TokenType.CHAR
    # void
    if word[0] == char("v") and word[1] == char("o") and word[2] == char("i") and word[3] == char("d") and word[4] == 0:
        return TokenType.VOID
    # short
    if word[0] == char("s") and word[1] == char("h") and word[2] == char("o") and word[3] == char("r") and word[4] == char("t") and word[5] == 0:
        return TokenType.SHORT
    # long
    if word[0] == char("l") and word[1] == char("o") and word[2] == char("n") and word[3] == char("g") and word[4] == 0:
        return TokenType.LONG
    # float
    if word[0] == char("f") and word[1] == char("l") and word[2] == char("o") and word[3] == char("a") and word[4] == char("t") and word[5] == 0:
        return TokenType.FLOAT
    # double
    if word[0] == char("d") and word[1] == char("o") and word[2] == char("u") and word[3] == char("b") and word[4] == char("l") and word[5] == char("e") and word[6] == 0:
        return TokenType.DOUBLE
    # signed
    if word[0] == char("s") and word[1] == char("i") and word[2] == char("g") and word[3] == char("n") and word[4] == char("e") and word[5] == char("d") and word[6] == 0:
        return TokenType.SIGNED
    # unsigned
    if word[0] == char("u") and word[1] == char("n") and word[2] == char("s") and word[3] == char("i") and word[4] == char("g") and word[5] == char("n") and word[6] == char("e") and word[7] == char("d") and word[8] == 0:
        return TokenType.UNSIGNED
    # struct
    if word[0] == char("s") and word[1] == char("t") and word[2] == char("r") and word[3] == char("u") and word[4] == char("c") and word[5] == char("t") and word[6] == 0:
        return TokenType.STRUCT
    # union
    if word[0] == char("u") and word[1] == char("n") and word[2] == char("i") and word[3] == char("o") and word[4] == char("n") and word[5] == 0:
        return TokenType.UNION
    # enum
    if word[0] == char("e") and word[1] == char("n") and word[2] == char("u") and word[3] == char("m") and word[4] == 0:
        return TokenType.ENUM
    # typedef
    if word[0] == char("t") and word[1] == char("y") and word[2] == char("p") and word[3] == char("e") and word[4] == char("d") and word[5] == char("e") and word[6] == char("f") and word[7] == 0:
        return TokenType.TYPEDEF
    # const
    if word[0] == char("c") and word[1] == char("o") and word[2] == char("n") and word[3] == char("s") and word[4] == char("t") and word[5] == 0:
        return TokenType.CONST
    # volatile
    if word[0] == char("v") and word[1] == char("o") and word[2] == char("l") and word[3] == char("a") and word[4] == char("t") and word[5] == char("i") and word[6] == char("l") and word[7] == char("e") and word[8] == 0:
        return TokenType.VOLATILE
    # static
    if word[0] == char("s") and word[1] == char("t") and word[2] == char("a") and word[3] == char("t") and word[4] == char("i") and word[5] == char("c") and word[6] == 0:
        return TokenType.STATIC
    # extern
    if word[0] == char("e") and word[1] == char("x") and word[2] == char("t") and word[3] == char("e") and word[4] == char("r") and word[5] == char("n") and word[6] == 0:
        return TokenType.EXTERN
    
    return 0  # Not a keyword


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
    
    # Single character tokens
    token.text[0] = c
    token.text[1] = 0
    
    if c == char("*"):
        token.type = TokenType.STAR
        lexer_advance(lex)
        return 1
    
    if c == char("("):
        token.type = TokenType.LPAREN
        lexer_advance(lex)
        return 1
    
    if c == char(")"):
        token.type = TokenType.RPAREN
        lexer_advance(lex)
        return 1
    
    if c == char("["):
        token.type = TokenType.LBRACKET
        lexer_advance(lex)
        return 1
    
    if c == char("]"):
        token.type = TokenType.RBRACKET
        lexer_advance(lex)
        return 1
    
    if c == char("{"):
        token.type = TokenType.LBRACE
        lexer_advance(lex)
        return 1
    
    if c == char("}"):
        token.type = TokenType.RBRACE
        lexer_advance(lex)
        return 1
    
    if c == char(";"):
        token.type = TokenType.SEMICOLON
        lexer_advance(lex)
        return 1
    
    if c == char(","):
        token.type = TokenType.COMMA
        lexer_advance(lex)
        return 1
    
    if c == char(":"):
        token.type = TokenType.COLON
        lexer_advance(lex)
        return 1
    
    if c == char("="):
        token.type = TokenType.EQUALS
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
