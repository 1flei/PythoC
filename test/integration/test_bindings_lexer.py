#!/usr/bin/env python3
"""
Test cases for the lexer
"""

from pythoc import compile, i32, i8, ptr, array, nullptr, sizeof, void, assume
from pythoc.libc.stdio import printf
from pythoc.libc.stdlib import malloc, free

from pythoc.bindings.c_token import Token, TokenType
from pythoc.bindings.lexer import (
    Lexer, lexer_create, lexer_destroy, lexer_next_token,
    lexer_nonnull, token_nonnull
)


@compile
def str_equal(s1: ptr[i8], s2: ptr[i8]) -> i32:
    """Compare two C strings for equality"""
    i: i32 = 0
    while s1[i] != 0 and s2[i] != 0:
        if s1[i] != s2[i]:
            return 0
        i = i + 1
    if s1[i] == s2[i]:
        return 1
    return 0


@compile
def test_simple_tokens() -> i32:
    """Test basic token recognition"""
    source: ptr[i8] = "int * ; ( ) ,"
    
    lex, prf = lexer_create(source)
    token: ptr[Token] = ptr[Token](malloc(sizeof(Token)))
    
    # Convert to refined types using assume
    lex_ref = assume(lex, lexer_nonnull)
    token_ref = assume(token, token_nonnull)
    
    # Token 1: int
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.INT:
        printf("FAIL: Expected TokenType.INT, got %d\n", token.type)
        return 1
    
    # Token 2: *
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.STAR:
        printf("FAIL: Expected TokenType.STAR, got %d\n", token.type)
        return 1
    
    # Token 3: ;
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.SEMICOLON:
        printf("FAIL: Expected TokenType.SEMICOLON, got %d\n", token.type)
        return 1
    
    # Token 4: (
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.LPAREN:
        printf("FAIL: Expected TokenType.LPAREN, got %d\n", token.type)
        return 1
    
    # Token 5: )
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.RPAREN:
        printf("FAIL: Expected TokenType.RPAREN, got %d\n", token.type)
        return 1
    
    # Token 6: ,
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.COMMA:
        printf("FAIL: Expected TokenType.COMMA, got %d\n", token.type)
        return 1
    
    # Token 7: EOF
    result: i32 = lexer_next_token(lex_ref, token_ref)
    if result != 0 or token.type != TokenType.EOF:
        printf("FAIL: Expected EOF\n")
        return 1
    
    lexer_destroy(lex, prf)
    free(token)
    printf("OK: test_simple_tokens passed\n")
    return 0


@compile
def test_identifiers_and_keywords() -> i32:
    """Test identifier and keyword recognition"""
    source: ptr[i8] = "int foo char bar123"
    
    lex, prf = lexer_create(source)
    token: ptr[Token] = ptr[Token](malloc(sizeof(Token)))
    
    # Convert to refined types using assume
    lex_ref = assume(lex, lexer_nonnull)
    token_ref = assume(token, token_nonnull)
    
    # Token 1: int (keyword)
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.INT:
        printf("FAIL: Expected TokenType.INT\n")
        return 1
    
    # Token 2: foo (identifier)
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.IDENTIFIER:
        printf("FAIL: Expected TokenType.IDENTIFIER\n")
        return 1
    if str_equal(token.text, "foo") == 0:
        printf("FAIL: Expected 'foo', got '%s'\n", token.text)
        return 1
    
    # Token 3: char (keyword)
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.CHAR:
        printf("FAIL: Expected TokenType.CHAR\n")
        return 1
    
    # Token 4: bar123 (identifier with numbers)
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.IDENTIFIER:
        printf("FAIL: Expected TokenType.IDENTIFIER\n")
        return 1
    if str_equal(token.text, "bar123") == 0:
        printf("FAIL: Expected 'bar123', got '%s'\n", token.text)
        return 1
    
    lexer_destroy(lex, prf)
    free(token)
    printf("OK: test_identifiers_and_keywords passed\n")
    return 0


@compile
def test_simple_function() -> i32:
    """Test lexing a simple function declaration"""
    source: ptr[i8] = "int add(int a, int b);"
    
    lex, prf = lexer_create(source)
    token: ptr[Token] = ptr[Token](malloc(sizeof(Token)))
    
    # Convert to refined types using assume
    lex_ref = assume(lex, lexer_nonnull)
    token_ref = assume(token, token_nonnull)
    
    expected_types: array[i32, 10]
    expected_types[0] = TokenType.INT
    expected_types[1] = TokenType.IDENTIFIER  # add
    expected_types[2] = TokenType.LPAREN
    expected_types[3] = TokenType.INT
    expected_types[4] = TokenType.IDENTIFIER  # a
    expected_types[5] = TokenType.COMMA
    expected_types[6] = TokenType.INT
    expected_types[7] = TokenType.IDENTIFIER  # b
    expected_types[8] = TokenType.RPAREN
    expected_types[9] = TokenType.SEMICOLON
    
    i: i32 = 0
    while i < 10:
        lexer_next_token(lex_ref, token_ref)
        if token.type != expected_types[i]:
            printf("FAIL: Token %d - expected %d, got %d\n", i, expected_types[i], token.type)
            return 1
        i = i + 1
    
    lexer_destroy(lex, prf)
    free(token)
    printf("OK: test_simple_function passed\n")
    return 0


@compile
def test_comments() -> i32:
    """Test comment skipping"""
    source: ptr[i8] = "int /* comment */ x // line comment\n;"
    
    lex, prf = lexer_create(source)
    token: ptr[Token] = ptr[Token](malloc(sizeof(Token)))
    
    # Convert to refined types using assume
    lex_ref = assume(lex, lexer_nonnull)
    token_ref = assume(token, token_nonnull)
    
    # Should get: int, x, ;
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.INT:
        printf("FAIL: Expected TokenType.INT after comment\n")
        return 1
    
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.IDENTIFIER:
        printf("FAIL: Expected identifier 'x'\n")
        return 1
    
    lexer_next_token(lex_ref, token_ref)
    if token.type != TokenType.SEMICOLON:
        printf("FAIL: Expected semicolon after line comment\n")
        return 1
    
    lexer_destroy(lex, prf)
    free(token)
    printf("OK: test_comments passed\n")
    return 0


@compile
def test_basic() -> i32:
    """Test very basic lexer functionality"""
    # Create a simple source: "int *"
    source: ptr[i8] = "int *"
    lex, prf = lexer_create(source)
    token: ptr[Token] = ptr[Token](malloc(sizeof(Token)))
    
    # Convert to refined types using assume
    lex_ref = assume(lex, lexer_nonnull)
    token_ref = assume(token, token_nonnull)
    
    # Get first token - should be 'int'
    lexer_next_token(lex_ref, token_ref)
    printf("Token 1 type: %d (expected %d for TokenType.INT)\n", token.type, TokenType.INT)
    
    if token.type != TokenType.INT:
        printf("FAIL: Expected TokenType.INT\n")
        return 1
    
    # Get second token - should be '*'
    lexer_next_token(lex_ref, token_ref)
    printf("Token 2 type: %d (expected %d for TokenType.STAR)\n", token.type, TokenType.STAR)
    
    if token.type != TokenType.STAR:
        printf("FAIL: Expected TokenType.STAR\n")
        return 1
    
    free(token)
    lexer_destroy(lex, prf)
    
    printf("OK: test_basic passed\n")
    return 0


@compile
def run_lexer_tests() -> i32:
    """Run all lexer tests"""
    printf("Running lexer tests...\n\n")
    
    failures: i32 = 0
    
    if test_simple_tokens() != 0:
        failures = failures + 1
    
    if test_identifiers_and_keywords() != 0:
        failures = failures + 1
    
    if test_simple_function() != 0:
        failures = failures + 1
    
    if test_comments() != 0:
        failures = failures + 1

    if test_basic() != 0:
        failures = failures + 1
    
    printf("\n")
    if failures == 0:
        printf("All lexer tests passed!\n")
    else:
        printf("%d lexer test(s) failed\n", failures)
    
    return failures


if __name__ == "__main__":
    result = run_lexer_tests()
    exit(result)

