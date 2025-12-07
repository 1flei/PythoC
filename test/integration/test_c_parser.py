"""
Test C parser AST types
"""
from __future__ import annotations
import unittest

from pythoc import compile, enum, i32, i8, ptr, void, struct
from pythoc.libc.stdio import printf


# =============================================================================
# C AST Types using enum with payload
# =============================================================================

# Base type specifiers
@enum(i8)
class BaseType:
    VOID: None
    CHAR: None
    SHORT: None
    INT: None
    LONG: None
    LONG_LONG: None
    FLOAT: None
    DOUBLE: None


# Signedness
@enum(i8)
class Signedness:
    DEFAULT: None
    SIGNED: None
    UNSIGNED: None


# C Type representation using enum with payload
# This allows representing different type categories with their associated data
@compile
class CType:
    """C type specification (zero-copy)"""
    base: i8               # BaseType enum
    sign: i8               # Signedness enum
    is_const: i8           # 1 if const, 0 otherwise
    ptr_depth: i8          # Number of * indirections
    # For struct/union/enum/typedef: name pointer and length
    name_ptr: ptr[i8]
    name_len: i32


# Function parameter
@compile
class CParam:
    """Function parameter"""
    type: CType
    name_ptr: ptr[i8]
    name_len: i32


# Function declaration
@compile
class CFunc:
    """Function declaration"""
    name_ptr: ptr[i8]
    name_len: i32
    ret_type: CType
    params: ptr[CParam]
    param_count: i32
    is_variadic: i8


# Struct/union field
@compile
class CField:
    """Struct/union field"""
    type: CType
    name_ptr: ptr[i8]
    name_len: i32
    bit_width: i32  # -1 if not bitfield


# Struct declaration
@compile
class CStruct:
    """Struct or union declaration"""
    name_ptr: ptr[i8]
    name_len: i32
    fields: ptr[CField]
    field_count: i32
    is_union: i8


# Enum value
@compile
class CEnumVal:
    """Enum value"""
    name_ptr: ptr[i8]
    name_len: i32
    has_value: i8
    value: i32


# Enum declaration
@compile
class CEnum:
    """Enum declaration"""
    name_ptr: ptr[i8]
    name_len: i32
    values: ptr[CEnumVal]
    value_count: i32


# Typedef declaration
@compile
class CTypedef:
    """Typedef declaration"""
    type: CType
    name_ptr: ptr[i8]
    name_len: i32


# Declaration kind enum with payload - the key design!
# Each variant carries its specific declaration type
@enum(i8)
class CDecl:
    """C declaration with payload"""
    FUNC: ptr[CFunc]
    STRUCT: ptr[CStruct]
    UNION: ptr[CStruct]  # Reuse CStruct for unions
    ENUM: ptr[CEnum]
    TYPEDEF: ptr[CTypedef]


# =============================================================================
# Test functions
# =============================================================================

@compile
def test_base_types() -> i32:
    """Test BaseType enum"""
    printf("Testing BaseType enum...\n")
    
    if BaseType.VOID != 0:
        printf("ERROR: VOID should be 0\n")
        return 1
    # INT is at index 3 (VOID=0, CHAR=1, SHORT=2, INT=3)
    if BaseType.INT != 3:
        printf("ERROR: INT should be 3\n")
        return 1
    
    printf("PASS: BaseType enum\n")
    return 0


@compile
def test_ctype_struct() -> i32:
    """Test CType struct"""
    printf("Testing CType struct...\n")
    
    t: CType = CType()
    t.base = BaseType.INT
    t.sign = Signedness.SIGNED
    t.is_const = 0
    t.ptr_depth = 0
    t.name_ptr = ptr[i8](0)
    t.name_len = 0
    
    if t.base != BaseType.INT:
        printf("ERROR: base should be INT\n")
        return 1
    
    printf("PASS: CType struct\n")
    return 0


@compile
def test_cdecl_enum() -> i32:
    """Test CDecl enum with payload"""
    printf("Testing CDecl enum with payload...\n")
    
    # Create a function declaration
    func: CFunc = CFunc()
    func.name_ptr = "test_func"
    func.name_len = 9
    func.param_count = 0
    func.is_variadic = 0
    
    # Wrap in CDecl enum
    decl: CDecl = CDecl(CDecl.FUNC, ptr(func))
    
    # Check tag - for enum value, [0] gets the tag
    if decl[0] != CDecl.FUNC:
        printf("ERROR: tag should be FUNC\n")
        return 1
    
    printf("PASS: CDecl enum with payload\n")
    return 0


@compile
def main() -> i32:
    printf("=== C Parser AST Tests ===\n\n")
    
    if test_base_types() != 0:
        return 1
    
    if test_ctype_struct() != 0:
        return 1
    
    if test_cdecl_enum() != 0:
        return 1
    
    printf("\n=== All Tests Passed ===\n")
    return 0


class TestCParserAST(unittest.TestCase):
    """Test C parser AST types"""
    
    def test_ast_types(self):
        """Run main test"""
        result = main()
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
