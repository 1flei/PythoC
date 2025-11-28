"""
Runner to test lexer from Python environment
"""
import sys
sys.path.insert(0, '/data/home/yifanlei/pythoc')

from pythoc import compile, i32, i8, ptr, array, sizeof
from pythoc.libc.stdio import printf
from pythoc.libc.stdlib import malloc, free

# Import tokens
from tool.c_header_parser.c_token import Token, TOK_INT, TOK_STAR, TOK_SEMICOLON
from tool.c_header_parser.lexer import Lexer, lexer_create, lexer_destroy, lexer_next_token


@compile
def test_basic() -> i32:
    """Test very basic lexer functionality"""
    # Create a simple source: "int *"
    source: ptr[i8] = ptr[i8](malloc(10))
    source[0] = 105  # 'i'
    source[1] = 110  # 'n'
    source[2] = 116  # 't'
    source[3] = 32   # ' '
    source[4] = 42   # '*'
    source[5] = 0    # null terminator
    
    lex: ptr[Lexer] = lexer_create(source)
    token: ptr[Token] = ptr[Token](malloc(sizeof(Token)))
    
    # Get first token - should be 'int'
    result1: i32 = lexer_next_token(lex, token)
    printf("Token 1: type=%d (expected %d for TOK_INT), result=%d\n", token.type, TOK_INT, result1)
    
    if token.type != TOK_INT:
        printf("FAIL: Expected TOK_INT\n")
        return 1
    
    # Get second token - should be '*'
    result2: i32 = lexer_next_token(lex, token)
    printf("Token 2: type=%d (expected %d for TOK_STAR), result=%d\n", token.type, TOK_STAR, result2)
    
    if token.type != TOK_STAR:
        printf("FAIL: Expected TOK_STAR\n")
        return 1
    
    printf("OK: Basic test passed!\n")
    
    free(token)
    lexer_destroy(lex)
    free(source)
    
    return 0


if __name__ == "__main__":
    result = test_basic()
    print(f"\nTest result: {result}")
