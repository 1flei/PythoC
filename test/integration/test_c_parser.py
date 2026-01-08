"""
Test C parser module - basic type parsing

Tests the c_parser module's type parsing functionality using parse_declarations API.
"""
from __future__ import annotations
import unittest

from pythoc import compile, i32, i8, ptr, void, struct, consume, refine
from pythoc.libc.stdio import printf

from pythoc.bindings.c_ast import (
    CType, QualType, Decl, DeclKind,
    qualtype_nonnull, QualTypeRef, ctype_nonnull, CTypeRef,
    QUAL_NONE, QUAL_CONST, QUAL_VOLATILE,
)
from pythoc.bindings.c_parser import parse_declarations, decl_free


# =============================================================================
# Helper to get base type and pointer depth from QualType
# =============================================================================

@compile
def get_base_type_tag(qt: QualTypeRef) -> i8:
    """
    Get the base type tag from a QualType, unwrapping pointer chain.
    Returns the innermost non-pointer CType tag.
    """
    ty: ptr[CType] = qt.type
    
    # Unwrap pointer chain to get base type
    while True:
        match ty[0]:
            case (CType.Ptr, pt):
                ty = pt.pointee.type
            case _:
                break
    
    # Return the tag of the base type
    return ty[0][0]


@compile
def get_ptr_depth(qt: QualTypeRef) -> i8:
    """
    Get the pointer depth from a QualType.
    Counts the number of CType.Ptr wrappers.
    """
    depth: i8 = 0
    ty: ptr[CType] = qt.type
    
    while True:
        match ty[0]:
            case (CType.Ptr, pt):
                depth = depth + 1
                ty = pt.pointee.type
            case _:
                break
    
    return depth


@compile
def get_outer_type_tag(qt: QualTypeRef) -> i8:
    """Get the outermost type tag (without unwrapping pointers)"""
    return qt.type[0][0]


# =============================================================================
# Test functions using parse_declarations
# =============================================================================

@compile
def test_parse_int() -> i32:
    """Test parsing 'int f();' function declaration with int return type"""
    for decl_prf, decl in parse_declarations("int f();"):
        for qt in refine(decl.type, qualtype_nonnull):
            # decl.type is the function type, get return type from it
            match qt.type[0]:
                case (CType.Func, ft):
                    for ret_qt in refine(ft.ret, qualtype_nonnull):
                        base_tag: i8 = get_base_type_tag(ret_qt)
                        quals: i8 = ret_qt.quals
                        ptr_depth: i8 = get_ptr_depth(ret_qt)
                        decl_free(decl_prf, decl)
                        if base_tag == CType.Int and quals == QUAL_NONE and ptr_depth == 0:
                            return 1
                        return 0
                case _:
                    pass
            decl_free(decl_prf, decl)
            return 0
        else:
            decl_free(decl_prf, decl)
    return 0


@compile
def test_parse_const_int() -> i32:
    """Test parsing 'const int f();' function declaration"""
    for decl_prf, decl in parse_declarations("const int f();"):
        for qt in refine(decl.type, qualtype_nonnull):
            match qt.type[0]:
                case (CType.Func, ft):
                    for ret_qt in refine(ft.ret, qualtype_nonnull):
                        base_tag: i8 = get_base_type_tag(ret_qt)
                        quals: i8 = ret_qt.quals
                        ptr_depth: i8 = get_ptr_depth(ret_qt)
                        decl_free(decl_prf, decl)
                        if base_tag == CType.Int and quals == QUAL_CONST and ptr_depth == 0:
                            return 1
                        return 0
                case _:
                    pass
            decl_free(decl_prf, decl)
            return 0
        else:
            decl_free(decl_prf, decl)
    return 0


@compile
def test_parse_int_ptr() -> i32:
    """Test parsing 'int *f();' function declaration returning int pointer"""
    for decl_prf, decl in parse_declarations("int *f();"):
        for qt in refine(decl.type, qualtype_nonnull):
            match qt.type[0]:
                case (CType.Func, ft):
                    for ret_qt in refine(ft.ret, qualtype_nonnull):
                        outer_tag: i8 = get_outer_type_tag(ret_qt)
                        base_tag: i8 = get_base_type_tag(ret_qt)
                        ptr_depth: i8 = get_ptr_depth(ret_qt)
                        decl_free(decl_prf, decl)
                        if outer_tag == CType.Ptr and base_tag == CType.Int and ptr_depth == 1:
                            return 1
                        return 0
                case _:
                    pass
            decl_free(decl_prf, decl)
            return 0
        else:
            decl_free(decl_prf, decl)
    return 0


@compile
def test_parse_int_ptr_ptr() -> i32:
    """Test parsing 'int **f();' function declaration (double pointer return)"""
    for decl_prf, decl in parse_declarations("int **f();"):
        for qt in refine(decl.type, qualtype_nonnull):
            match qt.type[0]:
                case (CType.Func, ft):
                    for ret_qt in refine(ft.ret, qualtype_nonnull):
                        outer_tag: i8 = get_outer_type_tag(ret_qt)
                        base_tag: i8 = get_base_type_tag(ret_qt)
                        ptr_depth: i8 = get_ptr_depth(ret_qt)
                        decl_free(decl_prf, decl)
                        if outer_tag == CType.Ptr and base_tag == CType.Int and ptr_depth == 2:
                            return 1
                        return 0
                case _:
                    pass
            decl_free(decl_prf, decl)
            return 0
        else:
            decl_free(decl_prf, decl)
    return 0


@compile
def test_parse_void() -> i32:
    """Test parsing 'void f();' function declaration"""
    for decl_prf, decl in parse_declarations("void f();"):
        for qt in refine(decl.type, qualtype_nonnull):
            match qt.type[0]:
                case (CType.Func, ft):
                    for ret_qt in refine(ft.ret, qualtype_nonnull):
                        base_tag: i8 = get_base_type_tag(ret_qt)
                        decl_free(decl_prf, decl)
                        if base_tag == CType.Void:
                            return 1
                        return 0
                case _:
                    pass
            decl_free(decl_prf, decl)
            return 0
        else:
            decl_free(decl_prf, decl)
    return 0


@compile
def test_parse_unsigned_int() -> i32:
    """Test parsing 'unsigned int f();' function declaration"""
    for decl_prf, decl in parse_declarations("unsigned int f();"):
        for qt in refine(decl.type, qualtype_nonnull):
            match qt.type[0]:
                case (CType.Func, ft):
                    for ret_qt in refine(ft.ret, qualtype_nonnull):
                        base_tag: i8 = get_base_type_tag(ret_qt)
                        decl_free(decl_prf, decl)
                        if base_tag == CType.UInt:
                            return 1
                        return 0
                case _:
                    pass
            decl_free(decl_prf, decl)
            return 0
        else:
            decl_free(decl_prf, decl)
    return 0


@compile
def test_parse_long_long() -> i32:
    """Test parsing 'long long f();' function declaration"""
    for decl_prf, decl in parse_declarations("long long f();"):
        for qt in refine(decl.type, qualtype_nonnull):
            match qt.type[0]:
                case (CType.Func, ft):
                    for ret_qt in refine(ft.ret, qualtype_nonnull):
                        base_tag: i8 = get_base_type_tag(ret_qt)
                        decl_free(decl_prf, decl)
                        if base_tag == CType.LongLong:
                            return 1
                        return 0
                case _:
                    pass
            decl_free(decl_prf, decl)
            return 0
        else:
            decl_free(decl_prf, decl)
    return 0


@compile
def test_parse_double() -> i32:
    """Test parsing 'double f();' function declaration"""
    for decl_prf, decl in parse_declarations("double f();"):
        for qt in refine(decl.type, qualtype_nonnull):
            match qt.type[0]:
                case (CType.Func, ft):
                    for ret_qt in refine(ft.ret, qualtype_nonnull):
                        base_tag: i8 = get_base_type_tag(ret_qt)
                        decl_free(decl_prf, decl)
                        if base_tag == CType.Double:
                            return 1
                        return 0
                case _:
                    pass
            decl_free(decl_prf, decl)
            return 0
        else:
            decl_free(decl_prf, decl)
    return 0


@compile
def test_parse_char() -> i32:
    """Test parsing 'char f();' function declaration"""
    for decl_prf, decl in parse_declarations("char f();"):
        for qt in refine(decl.type, qualtype_nonnull):
            match qt.type[0]:
                case (CType.Func, ft):
                    for ret_qt in refine(ft.ret, qualtype_nonnull):
                        base_tag: i8 = get_base_type_tag(ret_qt)
                        decl_free(decl_prf, decl)
                        if base_tag == CType.Char:
                            return 1
                        return 0
                case _:
                    pass
            decl_free(decl_prf, decl)
            return 0
        else:
            decl_free(decl_prf, decl)
    return 0


@compile
def test_parse_const_char_ptr() -> i32:
    """Test parsing 'const char *f();' function declaration"""
    for decl_prf, decl in parse_declarations("const char *f();"):
        for qt in refine(decl.type, qualtype_nonnull):
            match qt.type[0]:
                case (CType.Func, ft):
                    for ret_qt in refine(ft.ret, qualtype_nonnull):
                        outer_tag: i8 = get_outer_type_tag(ret_qt)
                        base_tag: i8 = get_base_type_tag(ret_qt)
                        ptr_depth: i8 = get_ptr_depth(ret_qt)
                        decl_free(decl_prf, decl)
                        if outer_tag == CType.Ptr and base_tag == CType.Char and ptr_depth == 1:
                            return 1
                        return 0
                case _:
                    pass
            decl_free(decl_prf, decl)
            return 0
        else:
            decl_free(decl_prf, decl)
    return 0


# =============================================================================
# Complex file parsing test
# =============================================================================

# Read nsieve.c content at compile time
import os as _os
_nsieve_path = _os.path.join(_os.path.dirname(__file__), '..', 'example', 'nsieve.c')
with open(_nsieve_path, 'r') as _f:
    NSIEVE_SOURCE = _f.read()


@compile
def test_parse_nsieve_file() -> i32:
    """
    Test parsing nsieve.c file content.
    Expected declarations:
    - nsieve function: static void nsieve(int m)
    - main function: int main(int argc, char **argv)
    
    Returns 1 if all expected declarations found, 0 otherwise.
    """
    found_nsieve: i8 = 0
    found_main: i8 = 0
    decl_count: i32 = 0
    
    for decl_prf, decl in parse_declarations(NSIEVE_SOURCE):
        decl_count = decl_count + 1
        
        # Check declaration kind and name
        match decl.kind[0]:
            case DeclKind.Func:
                # Check if it's nsieve or main function
                for qt in refine(decl.type, qualtype_nonnull):
                    match qt.type[0]:
                        case (CType.Func, ft):
                            # Check return type and validate function structure
                            for ret_qt in refine(ft.ret, qualtype_nonnull):
                                ret_tag: i8 = get_base_type_tag(ret_qt)
                                
                                # nsieve: void return, 1 param
                                if ret_tag == CType.Void and ft.param_count == 1:
                                    found_nsieve = 1
                                
                                # main: int return, 2 params
                                if ret_tag == CType.Int and ft.param_count == 2:
                                    found_main = 1
                        case _:
                            pass
            case _:
                pass
        
        decl_free(decl_prf, decl)
    
    printf("  Declarations found: %d\n", decl_count)
    printf("  nsieve function: %d\n", found_nsieve)
    printf("  main function: %d\n", found_main)
    
    # Success if we found both functions
    if found_nsieve != 0 and found_main != 0:
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
    
    result = test_parse_int_ptr_ptr()
    printf("parse_int_ptr_ptr: %d (expected 1)\n", result)
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
    
    result = test_parse_const_char_ptr()
    printf("parse_const_char_ptr: %d (expected 1)\n", result)
    if result != 1:
        return 1
    
    printf("\n--- Complex File Parsing Test ---\n")
    printf("Parsing nsieve.c...\n")
    result = test_parse_nsieve_file()
    printf("parse_nsieve_file: %d (expected 1)\n", result)
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
