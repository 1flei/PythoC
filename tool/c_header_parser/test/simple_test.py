"""
Simple standalone test for lexer basic functionality
Tests without using string literals in PC code
"""

from pythoc import compile, i32, i8, ptr, array, sizeof
from pythoc.libc.stdio import printf
from pythoc.libc.stdlib import malloc, free

from c_token import Token, TOK_INT, TOK_STAR
from lexer import Lexer, lexer_create, lexer_destroy, lexer_next_token


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
    lexer_next_token(lex, token)
    printf("Token 1 type: %d (expected %d for TOK_INT)\n", token.type, TOK_INT)
    
    # Get second token - should be '*'
    lexer_next_token(lex, token)
    printf("Token 2 type: %d (expected %d for TOK_STAR)\n", token.type, TOK_STAR)
    
    free(token)
    lexer_destroy(lex)
    free(source)
    
    return 0


# Call from Python for testing
if __name__ == "__main__":
    test_basic()
