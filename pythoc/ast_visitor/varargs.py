"""
Simple varargs support (MVP implementation)

This is a minimal implementation focusing on:
1. *args: struct[...] - compile-time expansion (zero-cost)
2. *struct - struct field unpacking at call site
3. *args: union[...] - LLVM varargs (runtime)

Full implementation (enum varargs) will be added later.
"""

import ast
from typing import List, Tuple, Optional, Any


def detect_varargs(func_node: ast.FunctionDef, type_resolver) -> Tuple[str, Optional[List[Any]], Optional[str]]:
    """Detect if function has *args with type annotation
    
    Args:
        func_node: Function AST node
        type_resolver: Optional TypeResolver for resolving type annotations
    
    Returns:
        Tuple of (kind, element_types, param_name):
        - kind: 'struct', 'union', 'none'
        - element_types: List of type annotations (ast nodes) for elements
        - param_name: Name of the *args parameter (e.g., 'args')
    
    Examples:
        def f(*args: struct[i32, f64]) -> void
        Returns: ('struct', [ast.Name('i32'), ast.Name('f64')], 'args')
        
        def f(*args: union[i32, f64]) -> void
        Returns: ('union', [ast.Name('i32'), ast.Name('f64')], 'args')
        
        def f(*args) -> void
        Returns: ('none', None, 'args')
    """
    if not func_node.args.vararg:
        return ('none', None, None)
    
    vararg = func_node.args.vararg
    if not vararg.annotation:
        # No annotation: *args without type (C-style varargs)
        return ('none', None, vararg.arg)
    
    annotation = vararg.annotation
    
    # Parse annotation using type_resolver to determine the kind
    parsed_type = type_resolver.parse_annotation(annotation)
    
    # Check if parsed type is struct, union, or enum
    from ..builtin_entities import struct, union, enum
    from ..builtin_entities.struct import StructType
    from ..builtin_entities.union import UnionType
    from ..builtin_entities.enum import EnumType
    
    # Get base type class
    if hasattr(parsed_type, '__origin__'):
        base_type = parsed_type.__origin__
    else:
        base_type = type(parsed_type) if not isinstance(parsed_type, type) else parsed_type
    
    # Determine kind
    kind = 'union'  # Default
    if isinstance(base_type, type):
        # Check for struct type (both StructType subclass and @compile decorated class)
        try:
            if issubclass(base_type, StructType) or (hasattr(base_type, '_is_struct') and base_type._is_struct):
                kind = 'struct'
            elif issubclass(base_type, UnionType) or (hasattr(base_type, '_is_union') and base_type._is_union):
                kind = 'union'
            elif issubclass(base_type, EnumType) or (hasattr(base_type, '_is_enum') and base_type._is_enum):
                kind = 'enum'
        except TypeError:
            # issubclass raised TypeError, not a class
            pass
    
    # Extract element types as AST nodes from annotation
    if isinstance(annotation, ast.Subscript):
        # annotation is like struct[i32, i32] or union[i32, f64]
        slice_node = annotation.slice
        if isinstance(slice_node, ast.Tuple):
            element_types = list(slice_node.elts)
        else:
            element_types = [slice_node]
    elif kind == 'struct' and (hasattr(parsed_type, '_struct_fields') or hasattr(parsed_type, '_field_types')):
        # annotation is like *args: Data where Data is a @compile class or struct[...]
        # We can't easily convert Python types back to AST, so we'll return empty list
        # and let the compiler extract field types from parsed_type
        element_types = []
    else:
        # Simple annotation or no subscript (e.g., *args: i32)
        element_types = [annotation]
    
    return (kind, element_types, vararg.arg)


def detect_struct_varargs(func_node: ast.FunctionDef, type_resolver=None) -> Tuple[bool, Optional[List[Any]], Optional[str]]:
    """Detect if function has *args: struct[...] parameter (legacy interface)
    
    This is kept for backward compatibility.
    
    Args:
        func_node: Function AST node
        type_resolver: Optional TypeResolver for resolving type annotations
    
    Returns:
        Tuple of (has_struct_varargs, element_types, param_name):
        - has_struct_varargs: True if function has *args: struct[...]
        - element_types: List of type annotations (ast nodes) for struct elements
        - param_name: Name of the *args parameter (e.g., 'args')
    
    Example:
        def f(*args: struct[i32, f64]) -> void
        Returns: (True, [ast.Name('i32'), ast.Name('f64')], 'args')
    """
    kind, element_types, param_name = detect_varargs(func_node, type_resolver)
    if kind == 'struct':
        return (True, element_types, param_name)
    return (False, None, None)


def expand_struct_varargs_in_ast(func_node: ast.FunctionDef, 
                                  element_types: List[ast.expr],
                                  varargs_name: str,
                                  type_resolver=None) -> ast.FunctionDef:
    """Expand *args: struct[...] into individual parameters
    
    Transforms the AST in-place:
        def f(*args: struct[i32, f64]) -> void
    Into:
        def f(arg0: i32, arg1: f64) -> void
    
    Also transforms args[0] -> arg0, args[1] -> arg1 in function body.
    For named fields: args.field_name -> argN where N is the field index.
    
    Args:
        func_node: Function AST node
        element_types: List of type AST nodes
        varargs_name: Name of the varargs parameter
        type_resolver: Optional TypeResolver for resolving field names
    
    Returns:
        Modified function AST node
    """
    # Get field names and types if available (for *args: StructClass where StructClass is @compile decorated)
    field_names = None
    if type_resolver and func_node.args.vararg and func_node.args.vararg.annotation:
        annotation = func_node.args.vararg.annotation
        parsed_type = type_resolver.parse_annotation(annotation)
        from ..builtin_entities.struct import StructType
        if isinstance(parsed_type, type) and (issubclass(parsed_type, StructType) or (hasattr(parsed_type, '_is_struct') and parsed_type._is_struct)):
            if hasattr(parsed_type, '_field_names'):
                field_names = parsed_type._field_names
            
            # If element_types is empty, extract from _struct_fields
            if not element_types and hasattr(parsed_type, '_struct_fields'):
                # We need to create AST nodes for the field types
                fields = parsed_type._struct_fields
                # Create AST Name nodes for each field type
                import builtins as __builtins__
                for field_name, field_type in fields:
                    # Try to get the type name
                    if hasattr(field_type, 'get_name'):
                        type_name = field_type.get_name()
                    elif hasattr(field_type, '__name__'):
                        type_name = field_type.__name__
                    else:
                        type_name = str(field_type)
                    # Create AST Name node
                    element_types.append(ast.Name(id=type_name, ctx=ast.Load()))
    
    # Create expanded parameters
    expanded_params = []
    for i, type_ann in enumerate(element_types):
        param_name = f'arg{i}'
        new_param = ast.arg(
            arg=param_name,
            annotation=type_ann,
            type_comment=None
        )
        expanded_params.append(new_param)
    
    # Replace vararg with expanded params
    func_node.args.vararg = None
    func_node.args.args.extend(expanded_params)
    
    # Transform args[i] -> argN and args.field_name -> argN in function body
    transformer = VarargsSubscriptTransformer(varargs_name, len(element_types), field_names)
    func_node = transformer.visit(func_node)
    
    return func_node


class VarargsSubscriptTransformer(ast.NodeTransformer):
    """Transform args[i] and args.field_name to argN for struct varargs"""
    
    def __init__(self, varargs_name: str, num_args: int, field_names: Optional[List[str]] = None):
        self.varargs_name = varargs_name
        self.num_args = num_args
        self.field_names = field_names  # Optional list of field names for named struct
    
    def visit_Subscript(self, node: ast.Subscript):
        """Transform args[0] -> arg0, args[1] -> arg1, etc."""
        # Check if this is accessing the varargs parameter
        if isinstance(node.value, ast.Name) and node.value.id == self.varargs_name:
            # Check if index is a constant
            if isinstance(node.slice, ast.Constant):
                index = node.slice.value
                if isinstance(index, int) and 0 <= index < self.num_args:
                    # Replace with parameter name
                    return ast.Name(id=f'arg{index}', ctx=ast.Load())
        
        # Not a varargs access, continue traversal
        return self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute):
        """Transform args.field_name -> argN for named struct fields"""
        # Check if this is accessing the varargs parameter
        if isinstance(node.value, ast.Name) and node.value.id == self.varargs_name:
            if self.field_names:
                # Try to find the field name in the field_names list
                field_name = node.attr
                try:
                    index = self.field_names.index(field_name)
                    if 0 <= index < self.num_args:
                        # Replace with parameter name
                        return ast.Name(id=f'arg{index}', ctx=ast.Load())
                except ValueError:
                    # Field not found, let it fall through
                    pass
        
        # Not a varargs access or field not found, continue traversal
        return self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Transform len(args) -> constant"""
        if isinstance(node.func, ast.Name) and node.func.id == 'len':
            if len(node.args) == 1:
                arg = node.args[0]
                if isinstance(arg, ast.Name) and arg.id == self.varargs_name:
                    # Replace len(args) with constant
                    return ast.Constant(value=self.num_args)
        
        # Not len(args), continue traversal
        return self.generic_visit(node)


def detect_starred_args(call_node: ast.Call) -> List[Tuple[int, ast.expr]]:
    """Detect starred arguments in function call
    
    Args:
        call_node: Call AST node
    
    Returns:
        List of (index, expr) tuples where starred args are found
    
    Example:
        f(a, *s, b) -> [(1, s)]
    """
    starred_positions = []
    for i, arg in enumerate(call_node.args):
        if isinstance(arg, ast.Starred):
            starred_positions.append((i, arg.value))
    return starred_positions


def expand_starred_struct_in_call(call_node: ast.Call, 
                                   starred_positions: List[Tuple[int, ast.expr]],
                                   struct_types: dict) -> ast.Call:
    """Expand *struct into individual arguments at call site
    
    Transforms:
        f(*my_struct, extra)
    Into:
        f(my_struct.field0, my_struct.field1, ..., extra)
    
    Args:
        call_node: Call AST node
        starred_positions: List of (index, expr) for starred args
        struct_types: Dict mapping struct names to field info
    
    Returns:
        Modified call AST node
    """
    # This will be implemented when we have struct type information available
    # For now, we'll handle it in the visitor where we have runtime type info
    return call_node
