"""
Lexer for C header files
Converts source text into a stream of tokens
"""

from pythoc import compile, i32, i8, ptr, array, nullptr, sizeof, void
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import strlen
from pythoc.libc.ctype import isalpha, isdigit, isspace, isalnum

from .c_token import (
    Token,
    TOK_EOF, TOK_ERROR, TOK_IDENTIFIER, TOK_NUMBER,
    TOK_INT, TOK_CHAR, TOK_SHORT, TOK_LONG, TOK_FLOAT, TOK_DOUBLE, TOK_VOID,
    TOK_SIGNED, TOK_UNSIGNED, TOK_STRUCT, TOK_UNION, TOK_ENUM, TOK_TYPEDEF,
    TOK_CONST, TOK_VOLATILE, TOK_STATIC, TOK_EXTERN,
    TOK_STAR, TOK_LPAREN, TOK_RPAREN, TOK_LBRACKET, TOK_RBRACKET,
    TOK_LBRACE, TOK_RBRACE, TOK_SEMICOLON, TOK_COMMA, TOK_COLON,
    TOK_EQUALS, TOK_ELLIPSIS
)


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
    
    if c == 10:  # '\n'
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
        if c == 47 and lexer_peek(lex, 1) == 47:  # '//'
            lexer_advance(lex)
            lexer_advance(lex)
            while lex.pos < lex.length and lexer_current(lex) != 10:  # '\n'
                lexer_advance(lex)
            continue
        
        # Skip /* block comments */
        if c == 47 and lexer_peek(lex, 1) == 42:  # '/*'
            lexer_advance(lex)
            lexer_advance(lex)
            while lex.pos < lex.length:
                if lexer_current(lex) == 42 and lexer_peek(lex, 1) == 47:  # '*/'
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
    if word[0] == 105 and word[1] == 110 and word[2] == 116 and word[3] == 0:  # 'int'
        return TOK_INT
    # char
    if word[0] == 99 and word[1] == 104 and word[2] == 97 and word[3] == 114 and word[4] == 0:  # 'char'
        return TOK_CHAR
    # void
    if word[0] == 118 and word[1] == 111 and word[2] == 105 and word[3] == 100 and word[4] == 0:  # 'void'
        return TOK_VOID
    # short
    if word[0] == 115 and word[1] == 104 and word[2] == 111 and word[3] == 114 and word[4] == 116 and word[5] == 0:  # 'short'
        return TOK_SHORT
    # long
    if word[0] == 108 and word[1] == 111 and word[2] == 110 and word[3] == 103 and word[4] == 0:  # 'long'
        return TOK_LONG
    # float
    if word[0] == 102 and word[1] == 108 and word[2] == 111 and word[3] == 97 and word[4] == 116 and word[5] == 0:  # 'float'
        return TOK_FLOAT
    # double
    if word[0] == 100 and word[1] == 111 and word[2] == 117 and word[3] == 98 and word[4] == 108 and word[5] == 101 and word[6] == 0:  # 'double'
        return TOK_DOUBLE
    # signed
    if word[0] == 115 and word[1] == 105 and word[2] == 103 and word[3] == 110 and word[4] == 101 and word[5] == 100 and word[6] == 0:  # 'signed'
        return TOK_SIGNED
    # unsigned
    if word[0] == 117 and word[1] == 110 and word[2] == 115 and word[3] == 105 and word[4] == 103 and word[5] == 110 and word[6] == 101 and word[7] == 100 and word[8] == 0:  # 'unsigned'
        return TOK_UNSIGNED
    # struct
    if word[0] == 115 and word[1] == 116 and word[2] == 114 and word[3] == 117 and word[4] == 99 and word[5] == 116 and word[6] == 0:  # 'struct'
        return TOK_STRUCT
    # union
    if word[0] == 117 and word[1] == 110 and word[2] == 105 and word[3] == 111 and word[4] == 110 and word[5] == 0:  # 'union'
        return TOK_UNION
    # enum
    if word[0] == 101 and word[1] == 110 and word[2] == 117 and word[3] == 109 and word[4] == 0:  # 'enum'
        return TOK_ENUM
    # typedef
    if word[0] == 116 and word[1] == 121 and word[2] == 112 and word[3] == 101 and word[4] == 100 and word[5] == 101 and word[6] == 102 and word[7] == 0:  # 'typedef'
        return TOK_TYPEDEF
    # const
    if word[0] == 99 and word[1] == 111 and word[2] == 110 and word[3] == 115 and word[4] == 116 and word[5] == 0:  # 'const'
        return TOK_CONST
    # volatile
    if word[0] == 118 and word[1] == 111 and word[2] == 108 and word[3] == 97 and word[4] == 116 and word[5] == 105 and word[6] == 108 and word[7] == 101 and word[8] == 0:  # 'volatile'
        return TOK_VOLATILE
    # static
    if word[0] == 115 and word[1] == 116 and word[2] == 97 and word[3] == 116 and word[4] == 105 and word[5] == 99 and word[6] == 0:  # 'static'
        return TOK_STATIC
    # extern
    if word[0] == 101 and word[1] == 120 and word[2] == 116 and word[3] == 101 and word[4] == 114 and word[5] == 110 and word[6] == 0:  # 'extern'
        return TOK_EXTERN
    
    return 0  # Not a keyword


@compile
def lexer_read_identifier(lex: ptr[Lexer], token: ptr[Token]) -> void:
    """Read identifier or keyword"""
    i: i32 = 0
    
    # First character must be alpha or underscore
    c: i8 = lexer_current(lex)
    while lex.pos < lex.length and (isalnum(c) or c == 95):  # '_' = 95
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
        token.type = TOK_IDENTIFIER


@compile
def lexer_read_number(lex: ptr[Lexer], token: ptr[Token]) -> void:
    """Read numeric literal (simplified, handles decimal and hex)"""
    i: i32 = 0
    c: i8 = lexer_current(lex)
    
    # Check for hex prefix 0x or 0X
    if c == 48 and (lexer_peek(lex, 1) == 120 or lexer_peek(lex, 1) == 88):  # '0x' or '0X'
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
        if isdigit(c) or c == 46 or c == 120 or c == 88:  # '.', 'x', 'X'
            if i < 255:
                token.text[i] = c
                i = i + 1
            lexer_advance(lex)
        elif (c >= 65 and c <= 70) or (c >= 97 and c <= 102):  # 'A-F' or 'a-f' for hex
            if i < 255:
                token.text[i] = c
                i = i + 1
            lexer_advance(lex)
        else:
            break
    
    token.text[i] = 0
    token.type = TOK_NUMBER


@compile
def lexer_next_token(lex: ptr[Lexer], token: ptr[Token]) -> i32:
    """
    Get next token from source
    Returns 1 on success, 0 on EOF
    """
    lexer_skip_whitespace(lex)
    
    # Check for EOF
    if lex.pos >= lex.length:
        token.type = TOK_EOF
        token.text[0] = 0
        token.line = lex.line
        token.col = lex.col
        return 0
    
    # Record token position
    token.line = lex.line
    token.col = lex.col
    
    c: i8 = lexer_current(lex)
    
    # Identifier or keyword (starts with letter or underscore)
    if isalpha(c) or c == 95:  # '_' = 95
        lexer_read_identifier(lex, token)
        return 1
    
    # Number (starts with digit)
    if isdigit(c):
        lexer_read_number(lex, token)
        return 1
    
    # Single character tokens
    token.text[0] = c
    token.text[1] = 0
    
    if c == 42:  # '*'
        token.type = TOK_STAR
        lexer_advance(lex)
        return 1
    
    if c == 40:  # '('
        token.type = TOK_LPAREN
        lexer_advance(lex)
        return 1
    
    if c == 41:  # ')'
        token.type = TOK_RPAREN
        lexer_advance(lex)
        return 1
    
    if c == 91:  # '['
        token.type = TOK_LBRACKET
        lexer_advance(lex)
        return 1
    
    if c == 93:  # ']'
        token.type = TOK_RBRACKET
        lexer_advance(lex)
        return 1
    
    if c == 123:  # '{'
        token.type = TOK_LBRACE
        lexer_advance(lex)
        return 1
    
    if c == 125:  # '}'
        token.type = TOK_RBRACE
        lexer_advance(lex)
        return 1
    
    if c == 59:  # ';'
        token.type = TOK_SEMICOLON
        lexer_advance(lex)
        return 1
    
    if c == 44:  # ','
        token.type = TOK_COMMA
        lexer_advance(lex)
        return 1
    
    if c == 58:  # ':'
        token.type = TOK_COLON
        lexer_advance(lex)
        return 1
    
    if c == 61:  # '='
        token.type = TOK_EQUALS
        lexer_advance(lex)
        return 1
    
    # Multi-character tokens
    if c == 46:  # '.'
        # Check for ellipsis '...'
        if lexer_peek(lex, 1) == 46 and lexer_peek(lex, 2) == 46:
            token.type = TOK_ELLIPSIS
            token.text[0] = 46
            token.text[1] = 46
            token.text[2] = 46
            token.text[3] = 0
            lexer_advance(lex)
            lexer_advance(lex)
            lexer_advance(lex)
            return 1
    
    # Unknown character - treat as error
    token.type = TOK_ERROR
    lexer_advance(lex)
    return 1
