"""
Token definitions for C header parser
"""

from pythoc import compile, i32, i8, array, enum

@enum
class TokenType:
    """Token type enumeration"""
    # Special tokens
    ERROR = 0
    EOF = 1
    
    # Keywords
    INT = 10
    CHAR = 11
    SHORT = 12
    LONG = 13
    FLOAT = 14
    DOUBLE = 15
    VOID = 16
    SIGNED = 17
    UNSIGNED = 18
    STRUCT = 19
    UNION = 20
    ENUM = 21
    TYPEDEF = 22
    CONST = 23
    VOLATILE = 24
    STATIC = 25
    EXTERN = 26
    
    # Identifiers and literals
    IDENTIFIER = 30
    NUMBER = 31
    STRING = 32
    CHAR_LITERAL = 33
    
    # Operators and punctuation
    STAR = 40        # *
    LPAREN = 41      # (
    RPAREN = 42      # )
    LBRACKET = 43    # [
    RBRACKET = 44    # ]
    LBRACE = 45      # {
    RBRACE = 46      # }
    SEMICOLON = 47   # ;
    COMMA = 48       # ,
    COLON = 49       # :
    EQUALS = 50      # =
    ELLIPSIS = 51    # ...
    
    # Preprocessor
    HASH = 60        # #
    DEFINE = 61      # #define
    INCLUDE = 62     # #include
    IFDEF = 63       # #ifdef
    IFNDEF = 64      # #ifndef
    ENDIF = 65       # #endif


@compile
class Token:
    """Represents a single token from the lexer"""
    type: i32                    # Token type (one of TokenType enum values)
    text: array[i8, 256]         # Token text content
    line: i32                    # Source line number (1-based)
    col: i32                     # Source column number (1-based)


# Map token type to C keyword string (lowercase)
_token_to_keyword = {
    TokenType.INT: "int",
    TokenType.CHAR: "char",
    TokenType.SHORT: "short",
    TokenType.LONG: "long",
    TokenType.FLOAT: "float",
    TokenType.DOUBLE: "double",
    TokenType.VOID: "void",
    TokenType.SIGNED: "signed",
    TokenType.UNSIGNED: "unsigned",
    TokenType.STRUCT: "struct",
    TokenType.UNION: "union",
    TokenType.ENUM: "enum",
    TokenType.TYPEDEF: "typedef",
    TokenType.CONST: "const",
    TokenType.VOLATILE: "volatile",
    TokenType.STATIC: "static",
    TokenType.EXTERN: "extern",
}

g_token_id_to_string = {}
g_token_string_to_id = {}

# Then, override with actual C keywords for keyword tokens
for token_type, keyword in _token_to_keyword.items():
    g_token_id_to_string[token_type] = keyword
    g_token_string_to_id[keyword] = token_type
