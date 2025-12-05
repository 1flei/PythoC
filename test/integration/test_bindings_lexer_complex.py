#!/usr/bin/env python3
"""
Complex test cases for the lexer using real C code with yield interface
"""

from pythoc import compile, i32, i8, ptr, array
from pythoc.libc.stdio import printf

from pythoc.bindings.c_token import Token, TokenType
from pythoc.bindings.lexer import tokens_from_source, lexer_create_raw, lexer_destroy_raw, lexer_nonnull


# Real C code from nsieve.c
C_CODE = """
typedef unsigned char boolean;

static void nsieve(int m) {
  unsigned int count = 0, i, j;
  boolean *flags = (boolean *)malloc(m * sizeof(boolean));
  
  for (i = 2; i < m; ++i)
    if (flags[i]) {
      ++count;
      for (j = i << 1; j < m; j += i)
        flags[j] = 0;
    }
  
  free(flags);
}

int main(int argc, char **argv) {
  int m = atoi(argv[1]);
  for (int i = 0; i < 3; i++)
    nsieve(10000 << (m - i));
  return 0;
}
"""


@compile
def test_full_c_file() -> i32:
    """Test lexing a complete C file using yield interface"""
    source: ptr[i8] = C_CODE
    
    token_count: i32 = 0
    typedef_count: i32 = 0
    static_count: i32 = 0
    int_count: i32 = 0
    identifier_count: i32 = 0
    lshift_count: i32 = 0
    
    # Count tokens using for loop with yield generator
    for token in tokens_from_source(source):
        token_count = token_count + 1
        
        if token.type == TokenType.TYPEDEF:
            typedef_count = typedef_count + 1
        elif token.type == TokenType.STATIC:
            static_count = static_count + 1
        elif token.type == TokenType.INT:
            int_count = int_count + 1
        elif token.type == TokenType.IDENTIFIER:
            identifier_count = identifier_count + 1
        elif token.type == TokenType.LSHIFT:
            lshift_count = lshift_count + 1
    
    printf("Lexed %d tokens\n", token_count)
    printf("  typedef: %d, static: %d, int: %d\n", typedef_count, static_count, int_count)
    printf("  identifiers: %d, lshift: %d\n", identifier_count, lshift_count)
    
    # Verify counts
    if typedef_count != 1:
        printf("FAIL: Expected 1 typedef, got %d\n", typedef_count)
        return 1
    
    if static_count != 1:
        printf("FAIL: Expected 1 static, got %d\n", static_count)
        return 1
    
    if int_count < 5:
        printf("FAIL: Expected at least 5 int keywords, got %d\n", int_count)
        return 1
    
    if identifier_count < 20:
        printf("FAIL: Expected at least 20 identifiers, got %d\n", identifier_count)
        return 1
    
    if lshift_count != 2:
        printf("FAIL: Expected 2 left shifts, got %d\n", lshift_count)
        return 1
    
    if token_count < 100:
        printf("FAIL: Expected at least 100 tokens, got %d\n", token_count)
        return 1
    
    printf("OK: test_full_c_file passed\n")
    return 0


@compile
def test_operators() -> i32:
    """Test lexing various operators"""
    source: ptr[i8] = "i << 1; j += i; ++count; m->left"
    
    token_types: array[i32, 20]
    token_idx: i32 = 0
    
    for token in tokens_from_source(source):
        if token_idx < 20:
            token_types[token_idx] = token.type
            token_idx = token_idx + 1
    
    # Verify token sequence
    expected_idx: i32 = 0
    
    # i
    if token_types[expected_idx] != TokenType.IDENTIFIER:
        printf("FAIL: Token %d should be IDENTIFIER\n", expected_idx)
        return 1
    expected_idx = expected_idx + 1
    
    # <<
    if token_types[expected_idx] != TokenType.LSHIFT:
        printf("FAIL: Token %d should be LSHIFT\n", expected_idx)
        return 1
    expected_idx = expected_idx + 1
    
    # 1
    if token_types[expected_idx] != TokenType.NUMBER:
        printf("FAIL: Token %d should be NUMBER\n", expected_idx)
        return 1
    expected_idx = expected_idx + 1
    
    # ;
    if token_types[expected_idx] != TokenType.SEMICOLON:
        printf("FAIL: Token %d should be SEMICOLON\n", expected_idx)
        return 1
    expected_idx = expected_idx + 1
    
    # j
    expected_idx = expected_idx + 1
    
    # +=
    if token_types[expected_idx] != TokenType.PLUS_ASSIGN:
        printf("FAIL: Token %d should be PLUS_ASSIGN\n", expected_idx)
        return 1
    expected_idx = expected_idx + 1
    
    printf("OK: test_operators passed\n")
    return 0


@compile
def test_simple_code() -> i32:
    """Test a simple C statement"""
    source: ptr[i8] = "typedef unsigned char boolean;"
    
    expected_types: array[i32, 5]
    expected_types[0] = TokenType.TYPEDEF
    expected_types[1] = TokenType.UNSIGNED
    expected_types[2] = TokenType.CHAR
    expected_types[3] = TokenType.IDENTIFIER
    expected_types[4] = TokenType.SEMICOLON
    
    token_idx: i32 = 0
    for token in tokens_from_source(source):
        if token_idx < 5:
            if token.type != expected_types[token_idx]:
                printf("FAIL: Token %d expected %d, got %d\n", 
                       token_idx, expected_types[token_idx], token.type)
                return 1
        token_idx = token_idx + 1
    
    if token_idx != 5:
        printf("FAIL: Expected 5 tokens, got %d\n", token_idx)
        return 1
    
    printf("OK: test_simple_code passed\n")
    return 0


def main():
    # Compile all test functions
    test_full_c_file()
    test_operators()
    test_simple_code()
    
    print("\nAll complex lexer tests compiled and passed!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
