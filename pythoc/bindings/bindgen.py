"""
Bindings Generator - Generate pythoc bindings from C source

This module provides compiled functions to parse C headers/sources
and generate pythoc binding code. It wraps the c_parser and pythoc_backend
modules to provide an end-to-end bindgen pipeline.

Usage:
    The main entry point is generate_bindings_to_file() which:
    1. Reads C source text
    2. Parses declarations using c_parser
    3. Emits pythoc code using pythoc_backend
    4. Writes output to a file

Note: These functions must be called from @compile context because
parse_declarations is a yield-based @compile function.
"""

from pythoc import (
    compile, i32, i64, i8, ptr, void, nullptr
)
from pythoc.libc.stdio import fopen, fclose, fwrite, fprintf
from pythoc.libc.string import strlen

from pythoc.bindings.c_parser import parse_declarations
from pythoc.bindings.c_ast import decl_free
from pythoc.bindings.pythoc_backend import (
    StringBuffer, strbuf_init, strbuf_destroy, strbuf_to_cstr,
    emit_module_header, emit_decl, strbuf_size
)


@compile
def generate_bindings(source: ptr[i8], lib: ptr[i8]) -> ptr[i8]:
    """Generate pythoc bindings from C source text.
    
    Args:
        source: C source text (null-terminated)
        lib: Library name for @extern decorators (null-terminated)
    
    Returns:
        Pointer to generated pythoc source code (null-terminated).
        Caller is responsible for freeing this memory.
    """
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    # Emit module header with imports
    emit_module_header(ptr(buf))
    
    # Parse and emit each declaration
    for decl_prf, decl in parse_declarations(source):
        emit_decl(ptr(buf), decl, lib)
        decl_free(decl_prf, decl)
    
    # Get result as C string
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    
    # Note: We return pointer to buffer's internal storage.
    # The caller must copy this before the buffer is destroyed.
    # For file writing, we write directly before destroying.
    
    return result


@compile
def generate_bindings_to_file(source: ptr[i8], lib: ptr[i8], output_path: ptr[i8]) -> i32:
    """Generate pythoc bindings and write to file.
    
    Args:
        source: C source text (null-terminated)
        lib: Library name for @extern decorators (null-terminated)
        output_path: Path to output .py file (null-terminated)
    
    Returns:
        0 on success, non-zero on error
    """
    buf: StringBuffer
    strbuf_init(ptr(buf))
    
    # Emit module header with imports
    emit_module_header(ptr(buf))
    
    # Parse and emit each declaration
    for decl_prf, decl in parse_declarations(source):
        emit_decl(ptr(buf), decl, lib)
        decl_free(decl_prf, decl)
    
    # Get result as C string
    result: ptr[i8] = strbuf_to_cstr(ptr(buf))
    size: i64 = strbuf_size(ptr(buf))
    
    # Write to file
    fp: ptr[i8] = fopen(output_path, "w")
    if fp == nullptr:
        strbuf_destroy(ptr(buf))
        return 1
    
    # Don't include null terminator in write
    written: i64 = fwrite(result, 1, size - 1, fp)
    fclose(fp)
    strbuf_destroy(ptr(buf))
    
    if written != size - 1:
        return 2
    
    return 0
