# -*- coding: utf-8 -*-
"""
Meta-programming support for PC compiler
Enables Python-as-preprocessor style generic programming
"""

import ast
import inspect
import textwrap
from typing import Any, Type, Callable
from .logger import logger


class MetaContext:
    """Context for tracking meta-programming state"""
    def __init__(self):
        self.type_counter = {}  # Track instantiation count per type
        self.generated_names = set()  # Track all generated names
    
    def get_unique_name(self, base_name: str, type_obj: Any) -> str:
        """Generate a unique name for a generic instantiation"""
        type_name = getattr(type_obj, '__name__', str(type_obj))
        key = f"{base_name}_{type_name}"
        
        if key not in self.type_counter:
            self.type_counter[key] = 0
        
        count = self.type_counter[key]
        self.type_counter[key] += 1
        
        if count == 0:
            unique_name = key
        else:
            unique_name = f"{key}_{count}"
        
        self.generated_names.add(unique_name)
        return unique_name


# Global meta-programming context
_meta_context = MetaContext()


def get_meta_context():
    """Get the global meta-programming context"""
    return _meta_context


def build_generic_class_ast(class_name: str, base_class_ast: ast.ClassDef, type_substitutions: dict) -> ast.ClassDef:
    """
    Build a new class AST with type substitutions
    
    Args:
        class_name: Name for the new class
        base_class_ast: Original class AST to clone
        type_substitutions: Dict mapping type parameter names to actual types
    
    Returns:
        New ClassDef AST node with substituted types
    """
    import copy
    
    # Deep copy the AST
    new_ast = copy.deepcopy(base_class_ast)
    new_ast.name = class_name
    
    # Substitute types in annotations
    class TypeSubstitutor(ast.NodeTransformer):
        def __init__(self, substitutions):
            self.substitutions = substitutions
        
        def visit_Name(self, node):
            # Check if this name should be substituted
            if node.id in self.substitutions:
                type_obj = self.substitutions[node.id]
                # Replace with the actual type name
                type_name = getattr(type_obj, '__name__', str(type_obj))
                return ast.Name(id=type_name, ctx=node.ctx)
            return node
        
        def visit_Subscript(self, node):
            # Handle ptr[T] -> ptr[ActualType]
            self.generic_visit(node)
            return node
    
    substitutor = TypeSubstitutor(type_substitutions)
    new_ast = substitutor.visit(new_ast)
    
    # Fix missing locations
    ast.fix_missing_locations(new_ast)
    
    return new_ast


def build_generic_function_ast(func_name: str, base_func_ast: ast.FunctionDef, type_substitutions: dict) -> ast.FunctionDef:
    """
    Build a new function AST with type substitutions
    
    Args:
        func_name: Name for the new function
        base_func_ast: Original function AST to clone
        type_substitutions: Dict mapping type parameter names to actual types
    
    Returns:
        New FunctionDef AST node with substituted types
    """
    import copy
    
    # Deep copy the AST
    new_ast = copy.deepcopy(base_func_ast)
    new_ast.name = func_name
    
    # Substitute types in annotations
    class TypeSubstitutor(ast.NodeTransformer):
        def __init__(self, substitutions):
            self.substitutions = substitutions
        
        def visit_Name(self, node):
            # Check if this name should be substituted
            if node.id in self.substitutions:
                type_obj = self.substitutions[node.id]
                # Replace with the actual type name
                type_name = getattr(type_obj, '__name__', str(type_obj))
                return ast.Name(id=type_name, ctx=node.ctx)
            return node
        
        def visit_Subscript(self, node):
            # Handle generic subscripts
            self.generic_visit(node)
            return node
    
    substitutor = TypeSubstitutor(type_substitutions)
    new_ast = substitutor.visit(new_ast)
    
    # Fix missing locations
    ast.fix_missing_locations(new_ast)
    
    return new_ast


def extract_type_parameters_from_closure(func: Callable) -> dict:
    """
    Extract type parameters from a function's closure
    
    This is used to find what types were captured when a generic
    function/class was created inside another function.
    """
    if not hasattr(func, '__closure__') or func.__closure__ is None:
        return {}
    
    # Get the variable names from the code object
    if hasattr(func, '__code__'):
        freevars = func.__code__.co_freevars
        closure_values = [cell.cell_contents for cell in func.__closure__]
        
        # Build a mapping of variable names to values
        closure_dict = dict(zip(freevars, closure_values))
        
        # Filter to only include type-like objects
        # Only include BuiltinEntity types, NOT arbitrary classes
        from .builtin_entities import BuiltinEntity
        type_params = {}
        for name, value in closure_dict.items():
            # Only include PC builtin types (i32, f64, etc.)
            # Do NOT include struct classes or other arbitrary classes
            if isinstance(value, type) and issubclass(value, BuiltinEntity):
                if value.can_be_type():
                    type_params[name] = value
        
        return type_params
    
    return {}


def get_enclosing_function_source(func: Callable) -> tuple[str, str]:
    """
    Get the source code of the enclosing function that defines this nested function/class
    
    Returns:
        (source_code, enclosing_function_name)
    """
    try:
        # Try to get the source of the function
        source = inspect.getsource(func)
        
        # Get the function's qualified name to find the enclosing function
        qualname = func.__qualname__
        if '.<locals>.' in qualname:
            # This is a nested function/class
            parts = qualname.split('.<locals>.')
            enclosing_name = parts[0]
            return source, enclosing_name
        
        return source, func.__name__
    
    except (OSError, TypeError):
        return None, None


def create_generic_instantiation(
    original_obj: Any,
    type_params: dict,
    base_name: str = None
) -> tuple[Any, ast.AST]:
    """
    Create a concrete instantiation of a generic function or class
    
    Args:
        original_obj: The original function or class
        type_params: Dictionary mapping type parameter names to concrete types
        base_name: Base name for the instantiation (auto-detected if None)
    
    Returns:
        (instantiated_object, ast_node)
    """
    is_class = inspect.isclass(original_obj)
    
    # Determine base name
    if base_name is None:
        base_name = original_obj.__name__
    
    # Generate unique name for this instantiation
    # Combine all type parameters into the name
    if type_params:
        # Sort type parameters for consistent naming
        type_names = []
        for param_name in sorted(type_params.keys()):
            type_obj = type_params[param_name]
            type_name = getattr(type_obj, '__name__', str(type_obj))
            type_names.append(type_name)
        
        # Create name like: add_i32 or ListNode_i32_i64
        unique_name = base_name + '_' + '_'.join(type_names)
    else:
        unique_name = base_name
    
    # Get the source code
    try:
        source = inspect.getsource(original_obj)
        source = textwrap.dedent(source)
        
        # Parse to AST
        tree = ast.parse(source)
        
        if is_class:
            # Find the ClassDef node
            class_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == original_obj.__name__:
                    class_node = node
                    break
            
            if class_node:
                # Build new class AST with type substitutions
                new_ast = build_generic_class_ast(unique_name, class_node, type_params)
                return original_obj, new_ast
        else:
            # Find the FunctionDef node
            func_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == original_obj.__name__:
                    func_node = node
                    break
            
            if func_node:
                # Build new function AST with type substitutions
                new_ast = build_generic_function_ast(unique_name, func_node, type_params)
                return original_obj, new_ast
    
    except (OSError, TypeError) as e:
        logger.warning("Could not get source for function", func=original_obj.__name__, error=str(e))
    
    return original_obj, None
