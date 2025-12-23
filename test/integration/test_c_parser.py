"""
Test C parser module - basic type parsing

Tests the c_parser module's type parsing functionality.
"""
from __future__ import annotations
import unittest

from pythoc import compile, i32, i8, ptr, void, struct, consume, assume, nullptr
from pythoc.libc.stdio import printf

from pythoc.bindings.c_ast import (
    Span, CType, QualType, span_is_empty, span_empty,
    qualtype_free, ctype_free,
    QUAL_NONE, QUAL_CONST, QUAL_VOLATILE,
)
from pythoc.bindings.c_parser import (
    TypeParseState, TypeParseStateRef, typeparse_nonnull, typeparse_init,
    parse_type_specifiers, build_qualtype_from_state, build_base_ctype,
    Parser, ParserRef, parser_nonnull, parser_advance,
)
from pythoc.bindings.lexer import lexer_create, lexer_destroy, lexer_nonnull, LexerRef
from pythoc.bindings.c_token import TokenType


# =============================================================================
# Helper to parse a type from source string
# =============================================================================

@compile
def parse_type_from_source(source: ptr[i8]) -> struct[i8, i8, i8]:
    """
    Parse type from source and return (ctype_tag, quals, ptr_depth).
    Returns (0, 0, 0) on failure.
    """
    lex_prf, lex = lexer_create(source)
    
    # Check if lexer creation succeeded
    if lex == nullptr:
        return 0, 0, 0
    
    lex_ref: LexerRef = assume(lex, lexer_nonnull)
    
    # Create parser
    parser: Parser = Parser()
    parser.lex = lex
    parser_advance(assume(ptr(parser), parser_nonnull))
    p: ParserRef = assume(ptr(parser), parser_nonnull)
    
    # Parse type specifiers
    ts: TypeParseState
    ts_ref: TypeParseStateRef = assume(ptr(ts), typeparse_nonnull)
    parse_type_specifiers(p, ts_ref)
    
    # Build QualType
    qt_prf, qt = build_qualtype_from_state(ts_ref)
    
    # Extract info - qt.type is ptr[CType], qt.type[0] is CType enum
    # For enum, [0] gets the tag
    ctype_tag: i8 = qt.type[0][0]  # Get tag from CType enum
    quals: i8 = qt.quals
    ptr_depth: i8 = ts.ptr_depth
    
    # Cleanup
    qualtype_free(qt_prf, qt)
    lexer_destroy(lex_prf, lex)
    
    return ctype_tag, quals, ptr_depth


# =============================================================================
# Test functions
# =============================================================================

@compile
def test_parse_int() -> i32:
    """Test parsing 'int' type"""
    tag, quals, ptr_depth = parse_type_from_source("int x")
    
    # CType.Int tag value
    if tag == CType.Int and quals == QUAL_NONE and ptr_depth == 0:
        return 1
    return 0


@compile
def test_parse_const_int() -> i32:
    """Test parsing 'const int' type"""
    tag, quals, ptr_depth = parse_type_from_source("const int x")
    
    if tag == CType.Int and quals == QUAL_CONST and ptr_depth == 0:
        return 1
    return 0


@compile
def test_parse_int_ptr() -> i32:
    """Test parsing 'int *' type"""
    tag, quals, ptr_depth = parse_type_from_source("int * x")
    
    # After pointer wrapping, the outer type is Ptr
    if tag == CType.Ptr and ptr_depth == 1:
        return 1
    return 0


@compile
def test_parse_void() -> i32:
    """Test parsing 'void' type"""
    tag, quals, ptr_depth = parse_type_from_source("void")
    
    if tag == CType.Void:
        return 1
    return 0


@compile
def test_parse_unsigned_int() -> i32:
    """Test parsing 'unsigned int' type"""
    tag, quals, ptr_depth = parse_type_from_source("unsigned int x")
    
    if tag == CType.UInt:
        return 1
    return 0


@compile
def test_parse_long_long() -> i32:
    """Test parsing 'long long' type"""
    tag, quals, ptr_depth = parse_type_from_source("long long x")
    
    if tag == CType.LongLong:
        return 1
    return 0


@compile
def test_parse_double() -> i32:
    """Test parsing 'double' type"""
    tag, quals, ptr_depth = parse_type_from_source("double x")
    
    if tag == CType.Double:
        return 1
    return 0


@compile
def test_parse_char() -> i32:
    """Test parsing 'char' type"""
    tag, quals, ptr_depth = parse_type_from_source("char x")
    
    if tag == CType.Char:
        return 1
    return 0


# =============================================================================
# Main test runner
# =============================================================================

@compile
def main() -> i32:
    printf("=== C Parser Type Tests ===\n\n")
    
    result: i32 = test_parse_int()
    printf("parse_int: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_parse_const_int()
    printf("parse_const_int: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_parse_int_ptr()
    printf("parse_int_ptr: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_parse_void()
    printf("parse_void: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_parse_unsigned_int()
    printf("parse_unsigned_int: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_parse_long_long()
    printf("parse_long_long: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_parse_double()
    printf("parse_double: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    result = test_parse_char()
    printf("parse_char: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    printf("\n=== All Tests Passed ===\n")
    return 0


class TestCParser(unittest.TestCase):
    """Test C parser module"""
    
    def test_type_parsing(self):
        """Run main test"""
        result = main()
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
