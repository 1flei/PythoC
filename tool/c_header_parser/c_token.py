"""
Token definitions for C header parser
"""

from pythoc import compile, i32, i8, array

# Token type constants
TOK_EOF: i32 = 0
TOK_ERROR: i32 = 1

# Keywords
TOK_INT: i32 = 10
TOK_CHAR: i32 = 11
TOK_SHORT: i32 = 12
TOK_LONG: i32 = 13
TOK_FLOAT: i32 = 14
TOK_DOUBLE: i32 = 15
TOK_VOID: i32 = 16
TOK_SIGNED: i32 = 17
TOK_UNSIGNED: i32 = 18
TOK_STRUCT: i32 = 19
TOK_UNION: i32 = 20
TOK_ENUM: i32 = 21
TOK_TYPEDEF: i32 = 22
TOK_CONST: i32 = 23
TOK_VOLATILE: i32 = 24
TOK_STATIC: i32 = 25
TOK_EXTERN: i32 = 26

# Identifiers and literals
TOK_IDENTIFIER: i32 = 30
TOK_NUMBER: i32 = 31
TOK_STRING: i32 = 32
TOK_CHAR_LITERAL: i32 = 33

# Operators and punctuation
TOK_STAR: i32 = 40        # *
TOK_LPAREN: i32 = 41      # (
TOK_RPAREN: i32 = 42      # )
TOK_LBRACKET: i32 = 43    # [
TOK_RBRACKET: i32 = 44    # ]
TOK_LBRACE: i32 = 45      # {
TOK_RBRACE: i32 = 46      # }
TOK_SEMICOLON: i32 = 47   # ;
TOK_COMMA: i32 = 48       # ,
TOK_COLON: i32 = 49       # :
TOK_EQUALS: i32 = 50      # =
TOK_ELLIPSIS: i32 = 51    # ...

# Preprocessor
TOK_HASH: i32 = 60        # #
TOK_DEFINE: i32 = 61      # #define
TOK_INCLUDE: i32 = 62     # #include
TOK_IFDEF: i32 = 63       # #ifdef
TOK_IFNDEF: i32 = 64      # #ifndef
TOK_ENDIF: i32 = 65       # #endif


@compile
class Token:
    """Represents a single token from the lexer"""
    type: i32                    # Token type (one of TOK_* constants)
    text: array[i8, 256]         # Token text content
    line: i32                    # Source line number (1-based)
    col: i32                     # Source column number (1-based)
