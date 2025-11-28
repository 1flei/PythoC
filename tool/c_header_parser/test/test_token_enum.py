"""
Test TokenType enum
Demonstrates proper use of enum for token constants
"""
import sys
sys.path.insert(0, '/data/home/yifanlei/pythoc')

from pythoc import compile, i32
from pythoc.libc.stdio import printf
from tool.c_header_parser.c_token import TokenType


@compile
def test_token_enum() -> i32:
    """Test using TokenType enum"""
    
    # Use enum values directly
    tok_int: i32 = TokenType.INT
    tok_star: i32 = TokenType.STAR
    tok_eof: i32 = TokenType.EOF
    
    printf("TokenType enum values:\n")
    printf("  INT = %d\n", tok_int)
    printf("  STAR = %d\n", tok_star)
    printf("  EOF = %d\n", tok_eof)
    
    # Verify values
    if tok_int != 10:
        printf("ERROR: INT should be 10\n")
        return 1
    
    if tok_star != 40:
        printf("ERROR: STAR should be 40\n")
        return 1
    
    if tok_eof != 0:
        printf("ERROR: EOF should be 0\n")
        return 1
    
    printf("\nAll enum values correct!\n")
    return 0


if __name__ == "__main__":
    print("=== Testing TokenType enum ===\n")
    
    result = test_token_enum()
    if result != 0:
        print("\ntest_token_enum FAILED")
        sys.exit(1)
    
    print("\n=== All enum tests passed! ===")
