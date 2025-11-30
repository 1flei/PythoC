#!/usr/bin/env python3
"""
Simple standalone test for lexer basic functionality
Tests without using string literals in PC code
"""

from pythoc import compile, i32, i8, ptr, array, sizeof
from pythoc.libc.stdio import printf
from pythoc.libc.stdlib import malloc, free

from pythoc.bindings.c_token import Token, TokenType
from pythoc.bindings.lexer import Lexer, lexer_create, lexer_destroy, lexer_next_token


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
    printf("Token 1 type: %d (expected %d for TokenType.INT)\n", token.type, TokenType.INT)
    
    if token.type != TokenType.INT:
        printf("FAIL: Expected TokenType.INT\n")
        return 1
    
    # Get second token - should be '*'
    lexer_next_token(lex, token)
    printf("Token 2 type: %d (expected %d for TokenType.STAR)\n", token.type, TokenType.STAR)
    
    if token.type != TokenType.STAR:
        printf("FAIL: Expected TokenType.STAR\n")
        return 1
    
    free(token)
    lexer_destroy(lex)
    free(source)
    
    printf("OK: test_basic passed\n")
    return 0


if __name__ == "__main__":
    result = test_basic()
    exit(result)

