"""
Integration test for pythoc_backend - C to pythoc code generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pythoc import compile, i8, i32, ptr, void
from pythoc.libc.stdio import printf
from pythoc.bindings.c_parser import parse_declarations
from pythoc.bindings.c_ast import decl_free
from pythoc.bindings.pythoc_backend import (
    StringBuffer, strbuf_init, strbuf_destroy, strbuf_to_cstr,
    emit_module_header, emit_decl
)


@compile
def test_simple_function() -> i32:
    """Test simple function declaration"""
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    emit_module_header(ptr(buf))
    
    for decl_prf, decl in parse_declarations("int add(int a, int b);"):
        emit_decl(ptr(buf), decl)
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Simple function ===\n%s\n", result)
    
    strbuf_destroy(ptr(buf))
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
        emit_decl(ptr(buf), decl)
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Struct ===\n%s\n", result)
    
    strbuf_destroy(ptr(buf))
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
        emit_decl(ptr(buf), decl)
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Enum ===\n%s\n", result)
    
    strbuf_destroy(ptr(buf))
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
        emit_decl(ptr(buf), decl)
        decl_free(decl_prf, decl)
    
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    printf("=== Complex header ===\n%s\n", result)
    
    strbuf_destroy(ptr(buf))
    return 0


@compile
def main() -> i32:
    printf("=== Pythoc Backend Tests ===\n\n")
    
    test_simple_function()
    test_struct()
    test_enum()
    test_complex_header()
    
    printf("All tests completed!\n")
    return 0


if __name__ == "__main__":
    main()
