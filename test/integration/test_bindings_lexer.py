#!/usr/bin/env python3
"""
Test cases for the lexer with linear type safety
"""

from pythoc import compile, i32, i8, ptr, assume, bool
from pythoc.libc.stdio import printf

from pythoc.bindings.c_token import Token, TokenType
from pythoc.bindings.lexer import (
    lexer_create, lexer_destroy, lexer_next_token,
    lexer_nonnull, token_release
)


@compile
def check_token_type(actual: i32, expected: i32) -> bool:
    """Check if token type matches expected"""
    return actual == expected


@compile
def test_simple_tokens() -> i32:
    """Test basic token recognition"""
    source: ptr[i8] = "int * ; ( ) ,"

    prf, lex = lexer_create(source)
    lex_ref = assume(lex, lexer_nonnull)

    result: i32 = 0

    # Token 1: int
    token1, tk_prf1, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token1.type, TokenType.INT):
        printf("FAIL: Expected TokenType.INT, got %d\n", token1.type)
        result = 1
    prf = token_release(token1, tk_prf1, prf)

    # Token 2: *
    token2, tk_prf2, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token2.type, TokenType.STAR):
        printf("FAIL: Expected TokenType.STAR, got %d\n", token2.type)
        result = 1
    prf = token_release(token2, tk_prf2, prf)

    # Token 3: ;
    token3, tk_prf3, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token3.type, TokenType.SEMICOLON):
        printf("FAIL: Expected TokenType.SEMICOLON, got %d\n", token3.type)
        result = 1
    prf = token_release(token3, tk_prf3, prf)

    # Token 4: (
    token4, tk_prf4, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token4.type, TokenType.LPAREN):
        printf("FAIL: Expected TokenType.LPAREN, got %d\n", token4.type)
        result = 1
    prf = token_release(token4, tk_prf4, prf)

    # Token 5: )
    token5, tk_prf5, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token5.type, TokenType.RPAREN):
        printf("FAIL: Expected TokenType.RPAREN, got %d\n", token5.type)
        result = 1
    prf = token_release(token5, tk_prf5, prf)

    # Token 6: ,
    token6, tk_prf6, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token6.type, TokenType.COMMA):
        printf("FAIL: Expected TokenType.COMMA, got %d\n", token6.type)
        result = 1
    prf = token_release(token6, tk_prf6, prf)

    # Token 7: EOF
    token7, tk_prf7, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token7.type, TokenType.EOF):
        printf("FAIL: Expected EOF\n")
        result = 1
    prf = token_release(token7, tk_prf7, prf)

    lexer_destroy(prf, lex)
    if result == 0:
        printf("OK: test_simple_tokens passed\n")
    return result


@compile
def test_identifiers_and_keywords() -> i32:
    """Test identifier and keyword recognition"""
    source: ptr[i8] = "int foo char bar123"

    prf, lex = lexer_create(source)
    lex_ref = assume(lex, lexer_nonnull)

    result: i32 = 0

    # Token 1: int (keyword)
    token1, tk_prf1, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token1.type, TokenType.INT):
        printf("FAIL: Expected TokenType.INT\n")
        result = 1
    prf = token_release(token1, tk_prf1, prf)

    # Token 2: foo (identifier)
    token2, tk_prf2, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token2.type, TokenType.IDENTIFIER):
        printf("FAIL: Expected TokenType.IDENTIFIER\n")
        result = 1
    prf = token_release(token2, tk_prf2, prf)

    # Token 3: char (keyword)
    token3, tk_prf3, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token3.type, TokenType.CHAR):
        printf("FAIL: Expected TokenType.CHAR\n")
        result = 1
    prf = token_release(token3, tk_prf3, prf)

    # Token 4: bar123 (identifier with numbers)
    token4, tk_prf4, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token4.type, TokenType.IDENTIFIER):
        printf("FAIL: Expected TokenType.IDENTIFIER\n")
        result = 1
    prf = token_release(token4, tk_prf4, prf)

    lexer_destroy(prf, lex)
    if result == 0:
        printf("OK: test_identifiers_and_keywords passed\n")
    return result


@compile
def test_comments() -> i32:
    """Test comment skipping"""
    source: ptr[i8] = "int /* comment */ x"

    prf, lex = lexer_create(source)
    lex_ref = assume(lex, lexer_nonnull)

    result: i32 = 0

    # Token 1: int
    token1, tk_prf1, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token1.type, TokenType.INT):
        printf("FAIL: Expected TokenType.INT\n")
        result = 1
    prf = token_release(token1, tk_prf1, prf)

    # Token 2: x (comment should be skipped)
    token2, tk_prf2, prf = lexer_next_token(lex_ref, prf)
    if not check_token_type(token2.type, TokenType.IDENTIFIER):
        printf("FAIL: Expected TokenType.IDENTIFIER\n")
        result = 1
    prf = token_release(token2, tk_prf2, prf)

    lexer_destroy(prf, lex)
    if result == 0:
        printf("OK: test_comments passed\n")
    return result


def main():
    print("Compiling tests...")

    # Compile all test functions
    test_simple_tokens()
    test_identifiers_and_keywords()
    test_comments()


if __name__ == "__main__":
    import sys
    sys.exit(main())
