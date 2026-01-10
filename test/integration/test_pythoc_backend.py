"""
Integration test for pythoc_backend - C to pythoc code generation

Tests:
1. Basic function declarations with configurable lib parameter
2. Struct declarations
3. Enum declarations
4. Complex C headers (similar to base_binary_tree_test.c)
5. Validation that generated pythoc code matches expected output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pythoc import compile, i8, i32, i64, ptr, void, array
from pythoc.libc.stdio import printf
from pythoc.libc.string import strcmp, strstr
from pythoc.bindings.c_parser import parse_declarations
from pythoc.bindings.c_ast import decl_free
from pythoc.bindings.pythoc_backend import (
    StringBuffer, strbuf_init, strbuf_destroy, strbuf_to_cstr,
    emit_module_header, emit_decl
)


@compile
def test_simple_function() -> i32:
    """Test simple function declaration with lib='c'"""
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    emit_module_header(ptr(buf))
    
    for decl_prf, decl in parse_declarations("int add(int a, int b);"):
        emit_decl(ptr(buf), decl, "c")
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Simple function ===\n%s\n", result)
    
    # Verify lib='c' is in the output
    if strstr(result, "@extern(lib='c')") == ptr[i8](0):
        printf("FAIL: Expected @extern(lib='c') in output\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    strbuf_destroy(ptr(buf))
    printf("PASS: test_simple_function\n\n")
    return 0


@compile
def test_math_lib() -> i32:
    """Test function declaration with lib='m' for math functions"""
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    emit_module_header(ptr(buf))
    
    for decl_prf, decl in parse_declarations("double sin(double x);"):
        emit_decl(ptr(buf), decl, "m")
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Math lib function ===\n%s\n", result)
    
    # Verify lib='m' is in the output
    if strstr(result, "@extern(lib='m')") == ptr[i8](0):
        printf("FAIL: Expected @extern(lib='m') in output\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    strbuf_destroy(ptr(buf))
    printf("PASS: test_math_lib\n\n")
    return 0


@compile
def test_struct() -> i32:
    """Test struct declaration"""
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    emit_module_header(ptr(buf))
    
    header: ptr[i8] = """
struct Point {
    int x;
    int y;
};
"""
    
    for decl_prf, decl in parse_declarations(header):
        emit_decl(ptr(buf), decl, "c")
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Struct ===\n%s\n", result)
    
    # Verify struct is generated correctly
    if strstr(result, "@compile") == ptr[i8](0):
        printf("FAIL: Expected @compile decorator\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "class Point") == ptr[i8](0):
        printf("FAIL: Expected 'class Point'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "x: i32") == ptr[i8](0):
        printf("FAIL: Expected 'x: i32'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "y: i32") == ptr[i8](0):
        printf("FAIL: Expected 'y: i32'\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    strbuf_destroy(ptr(buf))
    printf("PASS: test_struct\n\n")
    return 0


@compile
def test_enum() -> i32:
    """Test enum declaration"""
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    emit_module_header(ptr(buf))
    
    header: ptr[i8] = """
enum Color {
    RED,
    GREEN = 5,
    BLUE
};
"""
    
    for decl_prf, decl in parse_declarations(header):
        emit_decl(ptr(buf), decl, "c")
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Enum ===\n%s\n", result)
    
    # Verify enum is generated correctly
    if strstr(result, "@enum(i32)") == ptr[i8](0):
        printf("FAIL: Expected @enum(i32) decorator\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "class Color") == ptr[i8](0):
        printf("FAIL: Expected 'class Color'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "RED") == ptr[i8](0):
        printf("FAIL: Expected 'RED'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "GREEN") == ptr[i8](0):
        printf("FAIL: Expected 'GREEN'\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    strbuf_destroy(ptr(buf))
    printf("PASS: test_enum\n\n")
    return 0


@compile
def test_binary_tree_style() -> i32:
    """Test complex C code similar to base_binary_tree_test.c
    
    This tests:
    - typedef struct with self-referential pointers
    - Multiple function declarations
    - Various return types and parameter types
    """
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    emit_module_header(ptr(buf))
    
    # C code similar to base_binary_tree_test.c
    header: ptr[i8] = """
struct tn {
    struct tn* left;
    struct tn* right;
};

struct tn* NewTreeNode(struct tn* left, struct tn* right);
long ItemCheck(struct tn* tree);
struct tn* BottomUpTree(unsigned depth);
void DeleteTree(struct tn* tree);
"""
    
    for decl_prf, decl in parse_declarations(header):
        emit_decl(ptr(buf), decl, "c")
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Binary Tree Style ===\n%s\n", result)
    
    # Verify struct tn is generated
    if strstr(result, "class tn") == ptr[i8](0):
        printf("FAIL: Expected 'class tn'\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    # Verify self-referential pointer fields
    if strstr(result, "left: ptr[tn]") == ptr[i8](0):
        printf("FAIL: Expected 'left: ptr[tn]'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "right: ptr[tn]") == ptr[i8](0):
        printf("FAIL: Expected 'right: ptr[tn]'\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    # Verify function declarations
    if strstr(result, "def NewTreeNode") == ptr[i8](0):
        printf("FAIL: Expected 'def NewTreeNode'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "def ItemCheck") == ptr[i8](0):
        printf("FAIL: Expected 'def ItemCheck'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "def BottomUpTree") == ptr[i8](0):
        printf("FAIL: Expected 'def BottomUpTree'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "def DeleteTree") == ptr[i8](0):
        printf("FAIL: Expected 'def DeleteTree'\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    # Verify return types
    if strstr(result, "-> ptr[tn]") == ptr[i8](0):
        printf("FAIL: Expected '-> ptr[tn]' return type\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "-> i64") == ptr[i8](0):
        printf("FAIL: Expected '-> i64' return type for ItemCheck\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "-> void") == ptr[i8](0):
        printf("FAIL: Expected '-> void' return type for DeleteTree\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    strbuf_destroy(ptr(buf))
    printf("PASS: test_binary_tree_style\n\n")
    return 0


@compile
def test_complex_header() -> i32:
    """Test complex C header with multiple declarations"""
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    emit_module_header(ptr(buf))
    
    header: ptr[i8] = """
struct Point {
    int x;
    int y;
};

struct Rectangle {
    int width;
    int height;
};

enum Color {
    RED,
    GREEN,
    BLUE
};

int add(int a, int b);
void* malloc(unsigned long size);
int printf(const char* format, ...);
"""
    
    for decl_prf, decl in parse_declarations(header):
        emit_decl(ptr(buf), decl, "c")
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Complex header ===\n%s\n", result)
    
    # Verify all declarations are present
    if strstr(result, "class Point") == ptr[i8](0):
        printf("FAIL: Expected 'class Point'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "class Rectangle") == ptr[i8](0):
        printf("FAIL: Expected 'class Rectangle'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "class Color") == ptr[i8](0):
        printf("FAIL: Expected 'class Color'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "def add") == ptr[i8](0):
        printf("FAIL: Expected 'def add'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "def malloc") == ptr[i8](0):
        printf("FAIL: Expected 'def malloc'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "def printf") == ptr[i8](0):
        printf("FAIL: Expected 'def printf'\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    # Verify variadic function
    if strstr(result, "*args") == ptr[i8](0):
        printf("FAIL: Expected '*args' for variadic printf\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    strbuf_destroy(ptr(buf))
    printf("PASS: test_complex_header\n\n")
    return 0


@compile
def test_pointer_types() -> i32:
    """Test various pointer type conversions"""
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    emit_module_header(ptr(buf))
    
    header: ptr[i8] = """
void* get_ptr(void);
char* get_string(void);
int** get_ptr_ptr(void);
const char* get_const_string(void);
"""
    
    for decl_prf, decl in parse_declarations(header):
        emit_decl(ptr(buf), decl, "c")
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Pointer types ===\n%s\n", result)
    
    # Verify pointer types
    if strstr(result, "-> ptr[void]") == ptr[i8](0):
        printf("FAIL: Expected '-> ptr[void]'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "-> ptr[char]") == ptr[i8](0):
        printf("FAIL: Expected '-> ptr[char]'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "-> ptr[ptr[i32]]") == ptr[i8](0):
        printf("FAIL: Expected '-> ptr[ptr[i32]]'\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    strbuf_destroy(ptr(buf))
    printf("PASS: test_pointer_types\n\n")
    return 0


@compile
def test_c_source_file() -> i32:
    """Test parsing a full C source file with function definitions
    
    This tests:
    - typedef struct parsing
    - Function definitions (with bodies that get skipped)
    - Functions using typedef names as return/parameter types
    """
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    emit_module_header(ptr(buf))
    
    # C source code similar to base_binary_tree_test.c
    source: ptr[i8] = """
typedef struct tn {
    struct tn*    left;
    struct tn*    right;
} treeNode;

treeNode* NewTreeNode(treeNode* left, treeNode* right)
{
    treeNode*    new;
    new = (treeNode*)malloc(sizeof(treeNode));
    new->left = left;
    new->right = right;
    return new;
}

long ItemCheck(treeNode* tree)
{
    if (tree->left == NULL)
        return 1;
    else
        return 1 + ItemCheck(tree->left) + ItemCheck(tree->right);
}

treeNode* BottomUpTree(unsigned depth)
{
    if (depth > 0)
        return NewTreeNode(BottomUpTree(depth - 1), BottomUpTree(depth - 1));
    else
        return NewTreeNode(NULL, NULL);
}

void DeleteTree(treeNode* tree)
{
    if (tree->left != NULL)
    {
        DeleteTree(tree->left);
        DeleteTree(tree->right);
    }
    free(tree);
}

int main(int argc, char* argv[])
{
    return 0;
}
"""
    
    for decl_prf, decl in parse_declarations(source):
        emit_decl(ptr(buf), decl, "c")
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== C Source File ===\n%s\n", result)
    
    # Verify typedef is generated
    if strstr(result, "treeNode = tn") == ptr[i8](0):
        printf("FAIL: Expected 'treeNode = tn' typedef\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    # Verify function declarations are generated
    if strstr(result, "def NewTreeNode") == ptr[i8](0):
        printf("FAIL: Expected 'def NewTreeNode'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "def ItemCheck") == ptr[i8](0):
        printf("FAIL: Expected 'def ItemCheck'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "def BottomUpTree") == ptr[i8](0):
        printf("FAIL: Expected 'def BottomUpTree'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "def DeleteTree") == ptr[i8](0):
        printf("FAIL: Expected 'def DeleteTree'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "def main") == ptr[i8](0):
        printf("FAIL: Expected 'def main'\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    # Verify typedef usage in function signatures
    # Functions should use treeNode (typedef name) in their signatures
    if strstr(result, "ptr[treeNode]") == ptr[i8](0):
        printf("FAIL: Expected 'ptr[treeNode]' in function signatures\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    strbuf_destroy(ptr(buf))
    printf("PASS: test_c_source_file\n\n")
    return 0


@compile
def test_typedef_variants() -> i32:
    """Test various typedef patterns"""
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    emit_module_header(ptr(buf))
    
    source: ptr[i8] = """
typedef int myint;
typedef unsigned long size_t;
typedef char* string;
typedef void (*callback)(int);
typedef struct Point { int x; int y; } Point;
typedef enum Color { RED, GREEN, BLUE } Color;
"""
    
    for decl_prf, decl in parse_declarations(source):
        emit_decl(ptr(buf), decl, "c")
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Typedef Variants ===\n%s\n", result)
    
    # Verify basic typedefs
    if strstr(result, "myint = i32") == ptr[i8](0):
        printf("FAIL: Expected 'myint = i32'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "size_t = u64") == ptr[i8](0):
        printf("FAIL: Expected 'size_t = u64'\n")
        strbuf_destroy(ptr(buf))
        return 1
    if strstr(result, "string = ptr[char]") == ptr[i8](0):
        printf("FAIL: Expected 'string = ptr[char]'\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    # Verify struct typedef
    if strstr(result, "Point = Point") == ptr[i8](0):
        printf("FAIL: Expected 'Point = Point' typedef\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    # Verify enum typedef
    if strstr(result, "Color = Color") == ptr[i8](0):
        printf("FAIL: Expected 'Color = Color' typedef\n")
        strbuf_destroy(ptr(buf))
        return 1
    
    strbuf_destroy(ptr(buf))
    printf("PASS: test_typedef_variants\n\n")
    return 0


@compile
def main() -> i32:
    printf("=== Pythoc Backend Tests ===\n\n")
    
    failed: i32 = 0
    
    failed = failed + test_simple_function()
    failed = failed + test_math_lib()
    failed = failed + test_struct()
    failed = failed + test_enum()
    failed = failed + test_binary_tree_style()
    failed = failed + test_complex_header()
    failed = failed + test_pointer_types()
    failed = failed + test_c_source_file()
    failed = failed + test_typedef_variants()
    
    if failed > 0:
        printf("\n%d test(s) FAILED!\n", failed)
        return 1
    
    printf("\nAll tests PASSED!\n")
    return 0


if __name__ == "__main__":
    main()
