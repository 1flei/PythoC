"""
Token definitions for C header parser
"""

from pythoc import compile, i32, i8, array, enum, ptr
from pythoc.std.refine_wrapper import nonnull_wrap

@enum(i32)
class TokenType:
    """Token type enumeration"""
    # Special tokens
    ERROR: None
    EOF: None
    
    # Keywords
    INT: None
    CHAR: None
    SHORT: None
    LONG: None
    FLOAT: None
    DOUBLE: None
    VOID: None
    SIGNED: None
    UNSIGNED: None
    STRUCT: None
    UNION: None
    ENUM: None
    TYPEDEF: None
    CONST: None
    VOLATILE: None
    STATIC: None
    EXTERN: None
    SIZEOF: None
    RETURN: None
    IF: None
    ELSE: None
    WHILE: None
    FOR: None
    DO: None
    BREAK: None
    CONTINUE: None
    SWITCH: None
    CASE: None
    DEFAULT: None
    GOTO: None
    
    # Identifiers and literals
    IDENTIFIER: None
    NUMBER: None
    STRING: None
    CHAR_LITERAL: None
    
    # Single-character operators and punctuation
    LPAREN: None      # (
    RPAREN: None      # )
    LBRACKET: None    # [
    RBRACKET: None    # ]
    LBRACE: None      # {
    RBRACE: None      # }
    SEMICOLON: None   # ;
    COMMA: None       # ,
    COLON: None       # :
    DOT: None         # .
    PLUS: None        # +
    MINUS: None       # -
    STAR: None        # *
    SLASH: None       # /
    PERCENT: None     # %
    LT: None          # <
    GT: None          # >
    AMP: None         # &
    PIPE: None        # |
    CARET: None       # ^
    TILDE: None       # ~
    EXCLAIM: None     # !
    QUESTION: None    # ?
    ASSIGN: None      # =
    
    # Multi-character operators
    ELLIPSIS: None    # ...
    ARROW: None       # ->
    INC: None         # ++
    DEC: None         # --
    LSHIFT: None      # <<
    RSHIFT: None      # >>
    LE: None          # <=
    GE: None          # >=
    EQ: None          # ==
    NE: None          # !=
    LAND: None        # &&
    LOR: None         # ||
    PLUS_ASSIGN: None   # +=
    MINUS_ASSIGN: None  # -=
    STAR_ASSIGN: None   # *=
    SLASH_ASSIGN: None  # /=
    PERCENT_ASSIGN: None # %=
    LSHIFT_ASSIGN: None  # <<=
    RSHIFT_ASSIGN: None  # >>=
    AND_ASSIGN: None     # &=
    OR_ASSIGN: None      # |=
    XOR_ASSIGN: None     # ^=
    
    # Preprocessor
    HASH: None        # #
    DEFINE: None      # #define
    INCLUDE: None     # #include
    IFDEF: None       # #ifdef
    IFNDEF: None      # #ifndef
    ENDIF: None       # #endif


@compile
class Token:
    """Represents a single token from the lexer"""
    type: i32                    # Token type (one of TokenType enum values)
    start: ptr[i8]               # Pointer to token text in source (zero-copy)
    length: i32                  # Length of token text
    line: i32                    # Source line number (1-based)
    col: i32                     # Source column number (1-based)


token_nonnull, TokenRef = nonnull_wrap(ptr[Token])


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
    TokenType.SIZEOF: "sizeof",
    TokenType.RETURN: "return",
    TokenType.IF: "if",
    TokenType.ELSE: "else",
    TokenType.WHILE: "while",
    TokenType.FOR: "for",
    TokenType.DO: "do",
    TokenType.BREAK: "break",
    TokenType.CONTINUE: "continue",
    TokenType.SWITCH: "switch",
    TokenType.CASE: "case",
    TokenType.DEFAULT: "default",
    TokenType.GOTO: "goto",
}

# Map operator string to token type (sorted by length descending for longest match first)
_operator_to_token = [
    ("...", TokenType.ELLIPSIS),
    ("<<=", TokenType.LSHIFT_ASSIGN),
    (">>=", TokenType.RSHIFT_ASSIGN),
    ("->", TokenType.ARROW),
    ("++", TokenType.INC),
    ("--", TokenType.DEC),
    ("<<", TokenType.LSHIFT),
    (">>", TokenType.RSHIFT),
    ("<=", TokenType.LE),
    (">=", TokenType.GE),
    ("==", TokenType.EQ),
    ("!=", TokenType.NE),
    ("&&", TokenType.LAND),
    ("||", TokenType.LOR),
    ("+=", TokenType.PLUS_ASSIGN),
    ("-=", TokenType.MINUS_ASSIGN),
    ("*=", TokenType.STAR_ASSIGN),
    ("/=", TokenType.SLASH_ASSIGN),
    ("%=", TokenType.PERCENT_ASSIGN),
    ("&=", TokenType.AND_ASSIGN),
    ("|=", TokenType.OR_ASSIGN),
    ("^=", TokenType.XOR_ASSIGN),
    (".", TokenType.DOT),
    ("+", TokenType.PLUS),
    ("-", TokenType.MINUS),
    ("*", TokenType.STAR),
    ("/", TokenType.SLASH),
    ("%", TokenType.PERCENT),
    ("<", TokenType.LT),
    (">", TokenType.GT),
    ("&", TokenType.AMP),
    ("|", TokenType.PIPE),
    ("^", TokenType.CARET),
    ("!", TokenType.EXCLAIM),
    ("=", TokenType.ASSIGN),
    ("(", TokenType.LPAREN),
    (")", TokenType.RPAREN),
    ("[", TokenType.LBRACKET),
    ("]", TokenType.RBRACKET),
    ("{", TokenType.LBRACE),
    ("}", TokenType.RBRACE),
    (";", TokenType.SEMICOLON),
    (",", TokenType.COMMA),
    (":", TokenType.COLON),
    ("~", TokenType.TILDE),
    ("?", TokenType.QUESTION),
]

g_token_id_to_string = {}
g_token_string_to_id = {}

# Then, override with actual C keywords for keyword tokens
for token_type, keyword in _token_to_keyword.items():
    g_token_id_to_string[token_type] = keyword
    g_token_string_to_id[keyword] = token_type
