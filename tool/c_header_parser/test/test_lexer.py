"""
Test cases for the lexer
"""

from pythoc import compile, i32, i8, ptr, array, nullptr, sizeof, void
from pythoc.libc.stdio import printf
from pythoc.libc.stdlib import malloc, free

from .c_token import Token, TOK_INT, TOK_IDENTIFIER, TOK_STAR, TOK_SEMICOLON, TOK_LPAREN, TOK_RPAREN, TOK_COMMA, TOK_EOF, TOK_CHAR
from .lexer import Lexer, lexer_create, lexer_destroy, lexer_next_token, str_equal


@compile
def test_simple_tokens() -> i32:
    """Test basic token recognition"""
    source: ptr[i8] = "int * ; ( ) ,"
    
    lex: ptr[Lexer] = lexer_create(source)
    token: ptr[Token] = ptr[Token](malloc(sizeof(Token)))
    
    # Token 1: int
    lexer_next_token(lex, token)
    if token.type != TOK_INT:
        printf("FAIL: Expected TOK_INT, got %d\n", token.type)
        return 1
    
    # Token 2: *
    lexer_next_token(lex, token)
    if token.type != TOK_STAR:
        printf("FAIL: Expected TOK_STAR, got %d\n", token.type)
        return 1
    
    # Token 3: ;
    lexer_next_token(lex, token)
    if token.type != TOK_SEMICOLON:
        printf("FAIL: Expected TOK_SEMICOLON, got %d\n", token.type)
        return 1
    
    # Token 4: (
    lexer_next_token(lex, token)
    if token.type != TOK_LPAREN:
        printf("FAIL: Expected TOK_LPAREN, got %d\n", token.type)
        return 1
    
    # Token 5: )
    lexer_next_token(lex, token)
    if token.type != TOK_RPAREN:
        printf("FAIL: Expected TOK_RPAREN, got %d\n", token.type)
        return 1
    
    # Token 6: ,
    lexer_next_token(lex, token)
    if token.type != TOK_COMMA:
        printf("FAIL: Expected TOK_COMMA, got %d\n", token.type)
        return 1
    
    # Token 7: EOF
    result: i32 = lexer_next_token(lex, token)
    if result != 0 or token.type != TOK_EOF:
        printf("FAIL: Expected EOF\n")
        return 1
    
    lexer_destroy(lex)
    free(token)
    printf("OK: test_simple_tokens passed\n")
    return 0


@compile
def test_identifiers_and_keywords() -> i32:
    """Test identifier and keyword recognition"""
    source: ptr[i8] = "int foo char bar123"
    
    lex: ptr[Lexer] = lexer_create(source)
    token: ptr[Token] = ptr[Token](malloc(sizeof(Token)))
    
    # Token 1: int (keyword)
    lexer_next_token(lex, token)
    if token.type != TOK_INT:
        printf("FAIL: Expected TOK_INT\n")
        return 1
    
    # Token 2: foo (identifier)
    lexer_next_token(lex, token)
    if token.type != TOK_IDENTIFIER:
        printf("FAIL: Expected TOK_IDENTIFIER\n")
        return 1
    if str_equal(token.text, "foo") == 0:
        printf("FAIL: Expected 'foo', got '%s'\n", token.text)
        return 1
    
    # Token 3: char (keyword)
    lexer_next_token(lex, token)
    if token.type != TOK_CHAR:
        printf("FAIL: Expected TOK_CHAR\n")
        return 1
    
    # Token 4: bar123 (identifier with numbers)
    lexer_next_token(lex, token)
    if token.type != TOK_IDENTIFIER:
        printf("FAIL: Expected TOK_IDENTIFIER\n")
        return 1
    if str_equal(token.text, "bar123") == 0:
        printf("FAIL: Expected 'bar123', got '%s'\n", token.text)
        return 1
    
    lexer_destroy(lex)
    free(token)
    printf("OK: test_identifiers_and_keywords passed\n")
    return 0


@compile
def test_simple_function() -> i32:
    """Test lexing a simple function declaration"""
    source: ptr[i8] = "int add(int a, int b);"
    
    lex: ptr[Lexer] = lexer_create(source)
    token: ptr[Token] = ptr[Token](malloc(sizeof(Token)))
    
    expected_types: array[i32, 10]
    expected_types[0] = TOK_INT
    expected_types[1] = TOK_IDENTIFIER  # add
    expected_types[2] = TOK_LPAREN
    expected_types[3] = TOK_INT
    expected_types[4] = TOK_IDENTIFIER  # a
    expected_types[5] = TOK_COMMA
    expected_types[6] = TOK_INT
    expected_types[7] = TOK_IDENTIFIER  # b
    expected_types[8] = TOK_RPAREN
    expected_types[9] = TOK_SEMICOLON
    
    i: i32 = 0
    while i < 10:
        lexer_next_token(lex, token)
        if token.type != expected_types[i]:
            printf("FAIL: Token %d - expected %d, got %d\n", i, expected_types[i], token.type)
            return 1
        i = i + 1
    
    lexer_destroy(lex)
    free(token)
    printf("OK: test_simple_function passed\n")
    return 0


@compile
def test_comments() -> i32:
    """Test comment skipping"""
    source: ptr[i8] = "int /* comment */ x // line comment\n;"
    
    lex: ptr[Lexer] = lexer_create(source)
    token: ptr[Token] = ptr[Token](malloc(sizeof(Token)))
    
    # Should get: int, x, ;
    lexer_next_token(lex, token)
    if token.type != TOK_INT:
        printf("FAIL: Expected TOK_INT after comment\n")
        return 1
    
    lexer_next_token(lex, token)
    if token.type != TOK_IDENTIFIER:
        printf("FAIL: Expected identifier 'x'\n")
        return 1
    
    lexer_next_token(lex, token)
    if token.type != TOK_SEMICOLON:
        printf("FAIL: Expected semicolon after line comment\n")
        return 1
    
    lexer_destroy(lex)
    free(token)
    printf("OK: test_comments passed\n")
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
    
    printf("\n")
    if failures == 0:
        printf("All lexer tests passed!\n")
    else:
        printf("%d lexer test(s) failed\n", failures)
    
    return failures
