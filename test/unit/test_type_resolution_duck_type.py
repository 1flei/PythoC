#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test unified call protocol for type resolution (duck type dispatch)

Verify correctness of handle_as_type method
"""

import ast
from pythoc.type_resolver import TypeResolver
from pythoc.builtin_entities import i32, i64, f32, f64, ptr, u8, u16


def test_simple_type_resolution():
    """Test simple type resolution"""
    print("Testing simple type resolution...")
    
    resolver = TypeResolver()
    
    # Test AST Name node
    node = ast.Name(id='i32')
    result = resolver.parse_annotation(node)
    assert result == i32, f"Expected i32, got {result}"
    print("  OK ast.Name('i32') -> i32")
    
    # Test string
    result = resolver.parse_annotation('f64')
    assert result == f64, f"Expected f64, got {result}"
    print("  OK 'f64' -> f64")
    
    # Test direct class reference
    result = resolver.parse_annotation(u8)
    assert result == u8, f"Expected u8, got {result}"
    print("  OK u8 -> u8")
    
    print("OK Simple type resolution tests passed\n")


def test_ptr_type_resolution():
    """Test pointer type resolution (using handle_as_type)"""
    print("Testing pointer type resolution...")
    
    resolver = TypeResolver()
    
    # Test simple ptr
    node = ast.Name(id='ptr')
    result = resolver.parse_annotation(node)
    assert result == ptr, f"Expected ptr, got {result}"
    print("  OK ast.Name('ptr') -> ptr")
    
    # Test ptr[i32]
    # Construct ast.Subscript node: ptr[i32]
    subscript_node = ast.Subscript(
        value=ast.Name(id='ptr'),
        slice=ast.Name(id='i32')
    )
    result = resolver.parse_annotation(subscript_node)
    
    # Verify result is a ptr subclass
    assert isinstance(result, type), f"Expected type, got {type(result)}"
    assert issubclass(result, ptr), f"Expected ptr subclass, got {result}"
    assert hasattr(result, 'pointee_type'), "Expected pointee_type attribute"
    assert result.pointee_type == i32, f"Expected pointee_type=i32, got {result.pointee_type}"
    print(f"  OK ptr[i32] -> {result.get_name()}")
    
    # Test ptr[ptr[i64]]
    nested_subscript = ast.Subscript(
        value=ast.Name(id='ptr'),
        slice=ast.Subscript(
            value=ast.Name(id='ptr'),
            slice=ast.Name(id='i64')
        )
    )
    result = resolver.parse_annotation(nested_subscript)
    assert issubclass(result, ptr), f"Expected ptr subclass, got {result}"
    assert hasattr(result, 'pointee_type'), "Expected pointee_type attribute"
    # pointee_type should also be a ptr subclass
    assert isinstance(result.pointee_type, type), "Expected pointee_type to be a type"
    assert issubclass(result.pointee_type, ptr), "Expected pointee_type to be ptr subclass"
    print(f"  OK ptr[ptr[i64]] -> {result.get_name()}")
    
    print("OK Pointer type resolution tests passed\n")


def test_duck_type_dispatch():
    """Test duck type dispatch mechanism"""
    print("Testing duck type dispatch mechanism...")
    
    # Use ptr type to verify duck type dispatch
    # ptr class has implemented handle_as_type method
    resolver = TypeResolver()
    
    # Test ptr[i32] - should dispatch through handle_as_type
    subscript_node = ast.Subscript(
        value=ast.Name(id='ptr'),
        slice=ast.Name(id='i32')
    )
    result = resolver.parse_annotation(subscript_node)
    
    # Verify result is created through handle_as_type
    assert isinstance(result, type), f"Expected type, got {type(result)}"
    assert issubclass(result, ptr), f"Expected ptr subclass, got {result}"
    assert hasattr(result, 'pointee_type'), "Expected pointee_type attribute"
    assert result.pointee_type == i32, f"Expected pointee_type=i32, got {result.pointee_type}"
    print(f"  OK Duck type dispatch created {result.get_name()} via ptr.handle_as_type")
    
    # Verify simple types also call handle_as_type (though returning itself)
    node = ast.Name(id='i32')
    result = resolver.parse_annotation(node)
    assert result == i32, f"Expected i32, got {result}"
    print("  OK Duck type dispatch works correctly for simple types")
    
    print("OK Duck type dispatch tests passed\n")


def test_array_type_resolution():
    """Test array type resolution (using handle_as_type)"""
    print("Testing array type resolution...")
    
    from pythoc.builtin_entities import array
    resolver = TypeResolver()
    
    # Test array[i32, 5] - should use handle_as_type
    subscript_node = ast.Subscript(
        value=ast.Name(id='array'),
        slice=ast.Tuple(elts=[
            ast.Name(id='i32'),
            ast.Constant(value=5)
        ])
    )
    result = resolver.parse_annotation(subscript_node)
    
    # Verify result
    assert result is not None, "Expected array type, got None"
    assert isinstance(result, type), f"Expected type, got {type(result)}"
    assert issubclass(result, array), f"Expected array subclass, got {result}"
    assert hasattr(result, 'element_type'), "Expected element_type attribute"
    assert result.element_type == i32, f"Expected element_type=i32, got {result.element_type}"
    assert hasattr(result, 'dimensions'), "Expected dimensions attribute"
    assert result.dimensions == (5,), f"Expected dimensions=(5,), got {result.dimensions}"
    print(f"  OK array[i32, 5] -> {result.get_name()} (via handle_as_type)")
    
    # Test multi-dimensional array array[f32, 3, 4]
    multi_dim_node = ast.Subscript(
        value=ast.Name(id='array'),
        slice=ast.Tuple(elts=[
            ast.Name(id='f32'),
            ast.Constant(value=3),
            ast.Constant(value=4)
        ])
    )
    result = resolver.parse_annotation(multi_dim_node)
    
    assert issubclass(result, array), f"Expected array subclass, got {result}"
    assert result.element_type == f32, f"Expected element_type=f32, got {result.element_type}"
    assert result.dimensions == (3, 4), f"Expected dimensions=(3, 4), got {result.dimensions}"
    print(f"  OK array[f32, 3, 4] -> {result.get_name()} (via handle_as_type)")
    
    print("OK Array type resolution tests passed\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Type Resolution Unified Call Protocol Tests")
    print("=" * 60)
    print()
    
    test_simple_type_resolution()
    test_ptr_type_resolution()
    test_duck_type_dispatch()
    test_array_type_resolution()
    
    print("=" * 60)
    print("OK All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
