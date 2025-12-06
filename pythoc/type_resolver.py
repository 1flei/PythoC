"""
Unified Type Annotation Resolver

This module provides a centralized system for parsing and resolving type annotations
in the PC compiler. It converts AST type annotations into BuiltinType intermediate
representations, which can then be converted to LLVM types.

Architecture (Unified with ValueRef):
    AST Annotation -> ValueRef(kind='python') -> handle_subscript(index=None) -> BuiltinType -> LLVM Type

Key Insight:
    Type annotation parsing REUSES the same ValueRef + handle_subscript mechanism as
    runtime value operations! This avoids code duplication and ensures consistency.

Key Features:
- Parse simple types (i32, f64, bool, etc.)
- Parse pointer types (ptr[T]) via handle_subscript(index=None)
- Parse array types (array[T, N, M, ...]) via handle_subscript(index=None)
- Parse tuple types (tuple[T1, T2, ...])
- Support struct types via handle_subscript(index=None)
- Support dict subscripts (type_map[T]) for runtime type selection
- Unified intermediate representation using BuiltinType classes
"""

import ast
import builtins
from typing import Optional, Any
from llvmlite import ir
from .logger import logger

from .builtin_entities import (
    get_builtin_entity,
    ptr,
    array as ArrayType,
    struct as StructType,
    union as UnionType,
)
from .registry import get_unified_registry
from .valueref import ValueRef, wrap_value


# TODO: TypeResolver reuses the same ValueRef + handle_xxx mechanism as in LLVMIRVisitor
class TypeResolver:
    """
    Unified type annotation resolver.

    This class provides methods to parse AST type annotations and convert them
    into BuiltinType intermediate representations.

    Key design: Reuses ValueRef + handle_subscript(index=None) mechanism!

    Usage:
        resolver = TypeResolver(module_context)
        builtin_type = resolver.parse_annotation(ast_node)
        llvm_type = resolver.annotation_to_llvm_type(ast_node)
    """

    def __init__(self, module_context=None, user_globals=None, visitor=None):
        """
        Initialize the type resolver.

        Args:
            module_context: Optional LLVM module context for struct types
            user_globals: Optional user global namespace for type alias resolution
            visitor: Optional AST visitor instance for duck type dispatch
        """
        self.module_context = module_context
        self.user_globals = user_globals or {}
        self.visitor = visitor
        from .registry import _unified_registry

        self.struct_registry = _unified_registry

    def _get_visitor_context(self):
        """Get visitor context for duck type dispatch

        Returns a minimal visitor-like object with type_resolver attribute.
        """
        if self.visitor:
            return self.visitor

        # Create a minimal context object
        class MinimalVisitorContext:
            def __init__(self, type_resolver):
                self.type_resolver = type_resolver

        return MinimalVisitorContext(self)

    def parse_annotation(self, annotation) -> Optional[Any]:
        """
        Parse type annotation and return BuiltinType class.

        Architecture (unified with ValueRef):
        1. Evaluate annotation AST at compile time -> ValueRef(kind='python')
        2. Use handle_subscript(index=None) for type subscripts (reuse mechanism!)
        3. Extract type from ValueRef

        This approach:
        - Reuses ValueRef + handle_subscript mechanism (no duplication!)
        - Avoids string-based type matching
        - Supports complex expressions (subscript, attribute, dict[key])
        - Uses same protocol as visit_Subscript (index=None for types)

        Args:
            annotation: Can be:
                - AST node (ast.Name, ast.Subscript, ast.Attribute, etc.)
                - String type name ("i32", "f64", etc.)
                - BuiltinType class directly (i32, f64, etc.)

        Returns:
            - BuiltinType subclass for builtin types
            - Struct/enum Python class
            - None for missing annotations or invalid types
        """
        # Handle None as missing annotation
        if annotation is None:
            return None

        try:
            # Step 1: Evaluate annotation to get ValueRef(kind='python')
            # This reuses the same mechanism as visit_expression, but for compile-time evaluation
            value_ref = self._evaluate_type_expression(annotation)

            # Step 2: Extract type from ValueRef
            return self._extract_type_from_valueref(value_ref)
        except (NameError, TypeError, AttributeError):
            # Invalid type name or unsupported type expression
            # Return None to let caller decide how to handle
            return None

    def _evaluate_type_expression(self, node):
        """
        Evaluate type annotation AST at compile time, returning ValueRef(kind='python').

        Key insight: This is like visit_expression, but for compile-time type evaluation.
        Reuses the same ValueRef + handle_subscript/handle_attribute mechanism!

        Comparison with visit_expression:
        - visit_expression: runtime evaluation -> ValueRef(kind='value'/'address', value=ir.Value)
        - _evaluate_type_expression: compile-time evaluation -> ValueRef(kind='python', value=<type>)

        Supports:
        - Name: lookup in user_globals or builtins -> wrap as ValueRef
        - Subscript: call handle_subscript(index=None) for type subscripts (ptr[i32])
        - Subscript: dict[key] for runtime dict subscripts (type_map[T])
        - Attribute: call handle_attribute for attribute access (obj.TypeAlias)
        - Constant: literal values
        - Tuple: for struct fields, array dims, etc.
        - String: parse and recurse (for "from __future__ import annotations")

        Returns:
            ValueRef(kind='python', value=<type/class/entity>)
        """
        # Handle string annotations (from __future__ import annotations)
        if isinstance(node, str):
            try:
                parsed = ast.parse(node, mode="eval")
                return self._evaluate_type_expression(parsed.body)
            except:
                # Fallback: string might be a type name
                return self._wrap_as_python_value(node)

        # Already a Python value (not AST, not string) - wrap it
        if not isinstance(node, ast.AST):
            return self._wrap_as_python_value(node)

        # None constant -> void type (special case, return directly)
        if isinstance(node, ast.Constant) and node.value is None:
            return ir.VoidType()

        # String constant (from __future__ annotations) -> parse recursively
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                parsed = ast.parse(node.value, mode="eval")
                return self._evaluate_type_expression(parsed.body)
            except:
                return node.value  # Fallback to string

        # Other constants - wrap as ValueRef
        if isinstance(node, ast.Constant):
            return self._wrap_as_python_value(node.value)

        # Name lookup
        if isinstance(node, ast.Name):
            value = self._lookup_name(node.id)
            return wrap_value(value, kind="python", type_hint=type(value))

        # Subscript: Use normalize_subscript_items + handle_type_subscript
        if isinstance(node, ast.Subscript):
            # Step 1: Evaluate base to ValueRef
            base = self._evaluate_type_expression(node.value)

            # Step 2: Get the base value (type class like ptr, struct, etc.)
            base_value = self._unwrap_value(base)

            # Step 3: Check if base supports type subscript
            if hasattr(base_value, "handle_type_subscript"):
                # Parse subscript items from AST
                items = self._parse_subscript_items(node.slice)
                # Normalize and call handle_type_subscript
                normalized = base_value.normalize_subscript_items(items)
                result_value = base_value.handle_type_subscript(normalized)
                return wrap_value(
                    result_value, kind="python", type_hint=type(result_value)
                )
            elif hasattr(base_value, "__class_getitem__"):
                # Fallback to __class_getitem__ for types that don't have handle_type_subscript
                # (e.g., func type which uses __class_getitem__ directly)
                # We need to convert ast.Slice nodes to Python slice objects
                slice_arg = self._convert_slice_to_runtime(node.slice)
                result_value = base_value[slice_arg]
                return wrap_value(
                    result_value, kind="python", type_hint=type(result_value)
                )
            else:
                # Not a type syntax - runtime dict subscript (dict[key])
                # e.g., type_map = {i32: Struct_i32}, type_map[T]
                # For dict subscript, we DO need to evaluate the key
                key = self._evaluate_type_expression(node.slice)
                key_value = self._unwrap_value(key)

                try:
                    result_value = base_value[key_value]
                    return wrap_value(
                        result_value, kind="python", type_hint=type(result_value)
                    )
                except (KeyError, TypeError) as e:
                    raise TypeError(
                        f"Cannot subscript {base_value} with {key_value}: {e}"
                    )

        # Attribute access: reuse handle_attribute if available
        if isinstance(node, ast.Attribute):
            obj_ref = self._evaluate_type_expression(node.value)
            obj_value = self._unwrap_value(obj_ref)

            # Check if object has handle_attribute
            if hasattr(obj_value, "handle_attribute"):
                return obj_value.handle_attribute(
                    self._get_visitor_context(), obj_ref, node.attr, node
                )
            else:
                # Ordinary Python attribute access
                attr_value = getattr(obj_value, node.attr)
                return wrap_value(attr_value, kind="python", type_hint=type(attr_value))

        # Tuple (for array dimensions, struct fields, etc.)
        if isinstance(node, ast.Tuple):
            elts = [self._evaluate_type_expression(e) for e in node.elts]
            values = [self._unwrap_value(e) for e in elts]
            return wrap_value(tuple(values), kind="python", type_hint=tuple)

        # List (for type lists like [i32, i32])
        if isinstance(node, ast.List):
            elts = [self._evaluate_type_expression(e) for e in node.elts]
            values = [self._unwrap_value(e) for e in elts]
            return wrap_value(list(values), kind="python", type_hint=list)

        # Unsupported AST node type
        raise TypeError(f"Unsupported type expression AST node: {ast.dump(node)}")

    def _lookup_name(self, name):
        """Lookup name in various scopes

        Search order:
        1. visitor's symbol table (for local Python type variables like MyType = i32)
        2. user_globals (for module-level types)

        Returns:
            Python value (type, class, entity, etc.)

        Raises:
            NameError: If name not found
        """
        # 1. Check visitor's symbol table first (for local variables like MyType = i32)
        if self.visitor and hasattr(self.visitor, "lookup_variable"):
            var_info = self.visitor.lookup_variable(name)
            if var_info and var_info.value_ref:
                # Only use Python types (kind='python')
                if var_info.value_ref.kind == "python":
                    return var_info.value_ref.value

        # 2. Fallback to user_globals
        if name in self.user_globals:
            return self.user_globals[name]

        raise NameError(f"Type annotation '{name}' not found in scope")

    def _wrap_as_python_value(self, value):
        """Wrap Python value as ValueRef(kind='python')

        Args:
            value: Python object (type, class, entity, etc.)

        Returns:
            ValueRef(kind='python') or value itself if already ValueRef
        """
        if isinstance(value, ValueRef):
            return value  # Already wrapped
        return wrap_value(value, kind="python", type_hint=type(value))

    def _unwrap_value(self, obj):
        """Extract Python value from ValueRef or return as-is

        Args:
            obj: ValueRef or Python object

        Returns:
            Python value
        """
        if isinstance(obj, ValueRef):
            return obj.value
        return obj

    def _convert_slice_to_runtime(self, slice_node):
        """Convert AST slice node to Python runtime objects

        This converts AST nodes to Python objects suitable for __class_getitem__.
        Handles:
        - ast.Slice -> Python slice object
        - ast.Tuple -> Python tuple
        - ast.List -> Python list
        - Other nodes -> evaluate and unwrap

        Args:
            slice_node: AST node from Subscript.slice

        Returns:
            Python object (slice, tuple, list, or value)
        """
        import builtins

        # Handle ast.Slice: convert to Python slice object
        if isinstance(slice_node, ast.Slice):
            # ast.Slice(lower, upper, step) -> slice(start, stop, step)
            # For named fields: a: i32 becomes Slice(lower=a, upper=i32)
            # The lower part is a field name (string or identifier), not a type
            # Convert to Python slice object

            # Lower: field name (ast.Name or ast.Constant with string)
            if slice_node.lower:
                if isinstance(slice_node.lower, ast.Name):
                    start_val = slice_node.lower.id  # Extract identifier as string
                elif isinstance(slice_node.lower, ast.Constant) and isinstance(
                    slice_node.lower.value, str
                ):
                    start_val = slice_node.lower.value  # String literal
                else:
                    # Fallback: evaluate it
                    start = self._evaluate_type_expression(slice_node.lower)
                    start_val = self._unwrap_value(start)
            else:
                start_val = None

            # Upper: type expression
            if slice_node.upper:
                stop = self._evaluate_type_expression(slice_node.upper)
                stop_val = self._unwrap_value(stop)
            else:
                stop_val = None

            # Step: usually None for named fields
            if slice_node.step:
                step = self._evaluate_type_expression(slice_node.step)
                step_val = self._unwrap_value(step)
            else:
                step_val = None

            return builtins.slice(start_val, stop_val, step_val)

        # Handle ast.Tuple: convert to Python tuple
        if isinstance(slice_node, ast.Tuple):
            elements = []
            for elt in slice_node.elts:
                # Recursively convert each element (might contain Slice nodes)
                converted = self._convert_slice_to_runtime(elt)
                elements.append(converted)
            return builtins.tuple(elements)

        # Handle ast.List: convert to Python list
        if isinstance(slice_node, ast.List):
            elements = []
            for elt in slice_node.elts:
                # Recursively convert each element
                converted = self._convert_slice_to_runtime(elt)
                elements.append(converted)
            return builtins.list(elements)

        # Other nodes: evaluate and unwrap
        value_ref = self._evaluate_type_expression(slice_node)
        return self._unwrap_value(value_ref)

    def _parse_subscript_items(self, slice_node):
        """Parse subscript items from AST for type subscript.
        
        Converts AST slice node to items suitable for normalize_subscript_items.
        
        Handles:
        - ast.Slice (a: i32) -> ("a", i32)
        - ast.Tuple -> tuple of items
        - ast.Name/ast.Subscript -> type expression
        
        Args:
            slice_node: AST node from Subscript.slice
            
        Returns:
            Single item or tuple of items
        """
        import builtins
        
        # Handle ast.Slice: named field (a: i32)
        if isinstance(slice_node, ast.Slice):
            # Extract field name
            if slice_node.lower is None:
                raise TypeError("Slice must have lower bound (field name)")
            
            if isinstance(slice_node.lower, ast.Name):
                field_name = slice_node.lower.id
            elif isinstance(slice_node.lower, ast.Constant) and isinstance(slice_node.lower.value, str):
                field_name = slice_node.lower.value
            else:
                raise TypeError(f"Invalid field name in slice: {ast.dump(slice_node.lower)}")
            
            # Extract field type
            if slice_node.upper is None:
                raise TypeError("Slice must have upper bound (field type)")
            
            field_type_ref = self._evaluate_type_expression(slice_node.upper)
            field_type = self._unwrap_value(field_type_ref)
            
            return (field_name, field_type)
        
        # Handle ast.Tuple: multiple items
        if isinstance(slice_node, ast.Tuple):
            items = []
            for elt in slice_node.elts:
                item = self._parse_subscript_items(elt)
                items.append(item)
            return builtins.tuple(items)
        
        # Other nodes: evaluate as type expression
        value_ref = self._evaluate_type_expression(slice_node)
        return self._unwrap_value(value_ref)

    def _extract_type_from_valueref(self, value_ref):
        """Extract type from ValueRef(kind='python') or raw value

        Args:
            value_ref: ValueRef(kind='python') or raw Python value

        Returns:
            Type representation (BuiltinEntity, struct class, etc.)
        """
        # None
        if value_ref is None:
            return None

        # Extract value from ValueRef
        if isinstance(value_ref, ValueRef):
            if value_ref.kind != "python":
                raise TypeError(
                    f"Expected Python type ValueRef, got kind='{value_ref.kind}'"
                )
            value = value_ref.value
        else:
            # Backward compat: raw value
            value = value_ref

        # VoidType
        if isinstance(value, ir.VoidType):
            return value

        # BuiltinEntity (class or instance)
        from .builtin_entities import BuiltinEntity

        if isinstance(value, type):
            try:
                if issubclass(value, BuiltinEntity) and value.can_be_type():
                    return value  # i32, f64, ptr (unspecialized)
            except TypeError:
                pass

            # @struct/@enum/@union decorated class
            if getattr(value, "_is_struct", False) or getattr(value, "_is_enum", False):
                return value

        # BuiltinEntity instance (ptr[i32], array[i32, 10])
        if isinstance(value, BuiltinEntity):
            return value

        # String forward reference
        if isinstance(value, str):
            entity = get_builtin_entity(value)
            if entity and entity.can_be_type():
                return entity
            if self.struct_registry.has_struct(value):
                return value  # Forward reference

        raise TypeError(f"Cannot use as type: {value} (type: {type(value)})")

    def annotation_to_llvm_type(self, annotation) -> ir.Type:
        """
        Parse type annotation and return LLVM type.

        This is a convenience method that combines parse_annotation and
        extraction of LLVM type. Provides backward compatibility.

        Args:
            annotation: AST node representing a type annotation

        Returns:
            LLVM IR type
        """
        builtin_type = self.parse_annotation(annotation)

        if builtin_type is None:
            raise TypeError(
                f"TypeResolver: Cannot resolve type annotation: {annotation}"
            )

        # Check if it's a Python struct class (from @compile decorator)
        if (
            isinstance(builtin_type, type)
            and hasattr(builtin_type, "_is_struct")
            and builtin_type._is_struct
        ):
            # Use the struct's get_llvm_type method
            if hasattr(builtin_type, "get_llvm_type"):
                return builtin_type.get_llvm_type(self.module_context)
            # Fallback: get from module context
            struct_name = builtin_type.__name__
            if self.module_context:
                return self.module_context.get_identified_type(struct_name)
            else:
                raise TypeError(
                    "TypeResolver: module_context required for struct type resolution"
                )

        # Extract LLVM type from builtin type
        if hasattr(builtin_type, "get_llvm_type"):
            # All PC types now accept module_context parameter uniformly
            return builtin_type.get_llvm_type(self.module_context)
        elif isinstance(builtin_type, ir.Type):
            # ANTI-PATTERN: LLVM types should never appear in Python type system
            # This indicates a bug where llvm types leaked into annotations/type hints
            raise TypeError(
                f"TypeResolver: received raw LLVM type {builtin_type} instead of BuiltinEntity. "
                f"This is a bug - LLVM types should not be used in Python type annotations. "
                f"Use BuiltinEntity types (i32, f64, ptr[T], etc.) instead."
            )
        else:
            raise TypeError(
                f"TypeResolver: cannot extract LLVM type from {builtin_type} (type: {type(builtin_type)})"
            )
