"""
Unified Type Annotation Resolver

This module provides a centralized system for parsing and resolving type annotations
in the PC compiler. It converts AST type annotations into BuiltinType intermediate
representations, which can then be converted to LLVM types.

Architecture:
    AST Annotation -> visitor.visit_expression() -> ValueRef(kind='python') -> BuiltinType -> LLVM Type

Key Insight:
    Type annotation parsing directly REUSES visitor.visit_expression()!
    This eliminates code duplication and ensures consistency between
    compile-time type evaluation and runtime expression evaluation.

Key Features:
- Parse simple types (i32, f64, bool, etc.)
- Parse pointer types (ptr[T]) via handle_subscript
- Parse array types (array[T, N, M, ...]) via handle_subscript
- Parse tuple types (tuple[T1, T2, ...])
- Support struct types via handle_subscript
- Support dict subscripts (type_map[T]) for runtime type selection
- Unified intermediate representation using BuiltinType classes
"""

import ast
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
from .builder import NullBuilder
from .builder.null_builder import CompileTimeOnlyError


class TypeResolver:
    """
    Unified type annotation resolver.

    This class provides methods to parse AST type annotations and convert them
    into BuiltinType intermediate representations.

    Key design: Directly uses visitor.visit_expression() for type evaluation!
    This eliminates the need for duplicate _evaluate_type_expression logic.

    Usage:
        resolver = TypeResolver(module_context, visitor=visitor)
        builtin_type = resolver.parse_annotation(ast_node)
        llvm_type = resolver.annotation_to_llvm_type(ast_node)
    """

    def __init__(self, module_context=None, user_globals=None, visitor=None):
        """
        Initialize the type resolver.

        Args:
            module_context: Optional LLVM module context for struct types
            user_globals: Optional user global namespace for type alias resolution
            visitor: AST visitor instance for expression evaluation (required for full functionality)
        """
        self.module_context = module_context
        self.user_globals = user_globals or {}
        self.visitor = visitor
        from .registry import _unified_registry
        self.struct_registry = _unified_registry

    def parse_annotation(self, annotation) -> Optional[Any]:
        """
        Parse type annotation and return BuiltinType class.

        Architecture:
        1. Use visitor.visit_expression() to evaluate annotation AST
        2. Extract type from resulting ValueRef(kind='python')

        This approach:
        - Reuses visit_expression (no code duplication!)
        - Ensures consistency between compile-time and runtime evaluation
        - Supports all expression types that visit_expression handles

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
        if annotation is None:
            return None

        try:
            # Use visitor.visit_expression() for type evaluation
            value_ref = self._evaluate_type_expression(annotation)
            return self._extract_type_from_valueref(value_ref)
        except (NameError, TypeError, AttributeError, CompileTimeOnlyError) as e:
            logger.debug(f"Type resolution failed for {annotation}: {e}")
            return None

    def _evaluate_type_expression(self, node):
        """
        Evaluate type annotation AST, returning ValueRef(kind='python').

        Uses visitor.visit_expression() when available, with fallback to
        manual evaluation for cases where visitor is not available.

        Args:
            node: AST node, string, or Python value

        Returns:
            ValueRef(kind='python', value=<type>)
        """
        # Handle string annotations (from __future__ import annotations)
        if isinstance(node, str):
            try:
                parsed = ast.parse(node, mode="eval")
                return self._evaluate_type_expression(parsed.body)
            except:
                return self._wrap_as_python_value(node)

        # Already a Python value (not AST, not string) - wrap it
        if not isinstance(node, ast.AST):
            return self._wrap_as_python_value(node)

        # None constant -> void type
        if isinstance(node, ast.Constant) and node.value is None:
            return ir.VoidType()

        # String constant (from __future__ annotations) -> parse recursively
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                parsed = ast.parse(node.value, mode="eval")
                return self._evaluate_type_expression(parsed.body)
            except:
                return node.value

        # Use visitor.visit_expression() if available
        if self.visitor is not None:
            return self.visitor.visit_expression(node)

        # Fallback: manual evaluation when visitor is not available
        return self._manual_evaluate(node)

    def _manual_evaluate(self, node):
        """
        Manual type expression evaluation (fallback when visitor unavailable).

        This is a simplified version that handles common cases.
        """
        # Other constants - wrap as ValueRef
        if isinstance(node, ast.Constant):
            return self._wrap_as_python_value(node.value)

        # Name lookup
        if isinstance(node, ast.Name):
            value = self._lookup_name(node.id)
            return wrap_value(value, kind="python", type_hint=type(value))

        # Subscript
        if isinstance(node, ast.Subscript):
            base = self._manual_evaluate(node.value)
            base_value = self._unwrap_value(base)

            if hasattr(base_value, "handle_type_subscript"):
                items = self._parse_subscript_items(node.slice)
                normalized = base_value.normalize_subscript_items(items)
                result_value = base_value.handle_type_subscript(normalized)
                return wrap_value(result_value, kind="python", type_hint=type(result_value))
            elif hasattr(base_value, "__class_getitem__"):
                slice_arg = self._convert_slice_to_runtime(node.slice)
                result_value = base_value[slice_arg]
                return wrap_value(result_value, kind="python", type_hint=type(result_value))
            else:
                key = self._manual_evaluate(node.slice)
                key_value = self._unwrap_value(key)
                result_value = base_value[key_value]
                return wrap_value(result_value, kind="python", type_hint=type(result_value))

        # Attribute access
        if isinstance(node, ast.Attribute):
            obj_ref = self._manual_evaluate(node.value)
            obj_value = self._unwrap_value(obj_ref)
            attr_value = getattr(obj_value, node.attr)
            return wrap_value(attr_value, kind="python", type_hint=type(attr_value))

        # Tuple
        if isinstance(node, ast.Tuple):
            elts = [self._manual_evaluate(e) for e in node.elts]
            values = [self._unwrap_value(e) for e in elts]
            return wrap_value(tuple(values), kind="python", type_hint=tuple)

        # List
        if isinstance(node, ast.List):
            elts = [self._manual_evaluate(e) for e in node.elts]
            values = [self._unwrap_value(e) for e in elts]
            return wrap_value(list(values), kind="python", type_hint=list)

        raise TypeError(f"Unsupported type expression AST node: {ast.dump(node)}")

    def _lookup_name(self, name):
        """Lookup name in various scopes.

        Search order:
        1. visitor's symbol table (for local Python type variables)
        2. user_globals (for module-level types)
        3. builtin entities (i32, f64, ptr, etc.)

        Returns:
            Python value (type, class, entity, etc.)

        Raises:
            NameError: If name not found
        """
        # Check visitor's symbol table first
        if self.visitor and hasattr(self.visitor, "lookup_variable"):
            var_info = self.visitor.lookup_variable(name)
            if var_info and var_info.value_ref:
                if var_info.value_ref.kind == "python":
                    return var_info.value_ref.value

        # Fallback to user_globals
        if name in self.user_globals:
            return self.user_globals[name]

        # Check builtin entities
        entity = get_builtin_entity(name)
        if entity is not None:
            return entity

        raise NameError(f"Type annotation '{name}' not found in scope")

    def _wrap_as_python_value(self, value):
        """Wrap Python value as ValueRef(kind='python')."""
        if isinstance(value, ValueRef):
            return value
        return wrap_value(value, kind="python", type_hint=type(value))

    def _unwrap_value(self, obj):
        """Extract Python value from ValueRef or return as-is."""
        if isinstance(obj, ValueRef):
            return obj.value
        return obj

    def _convert_slice_to_runtime(self, slice_node):
        """Convert AST slice node to Python runtime objects."""
        import builtins

        if isinstance(slice_node, ast.Slice):
            if slice_node.lower:
                if isinstance(slice_node.lower, ast.Name):
                    start_val = slice_node.lower.id
                elif isinstance(slice_node.lower, ast.Constant) and isinstance(slice_node.lower.value, str):
                    start_val = slice_node.lower.value
                else:
                    start = self._manual_evaluate(slice_node.lower)
                    start_val = self._unwrap_value(start)
            else:
                start_val = None

            if slice_node.upper:
                stop = self._manual_evaluate(slice_node.upper)
                stop_val = self._unwrap_value(stop)
            else:
                stop_val = None

            if slice_node.step:
                step = self._manual_evaluate(slice_node.step)
                step_val = self._unwrap_value(step)
            else:
                step_val = None

            return builtins.slice(start_val, stop_val, step_val)

        if isinstance(slice_node, ast.Tuple):
            elements = [self._convert_slice_to_runtime(elt) for elt in slice_node.elts]
            return builtins.tuple(elements)

        if isinstance(slice_node, ast.List):
            elements = [self._convert_slice_to_runtime(elt) for elt in slice_node.elts]
            return builtins.list(elements)

        value_ref = self._manual_evaluate(slice_node)
        return self._unwrap_value(value_ref)

    def _parse_subscript_items(self, slice_node):
        """Parse subscript items from AST for type subscript."""
        import builtins

        if isinstance(slice_node, ast.Slice):
            if slice_node.lower is None:
                raise TypeError("Slice must have lower bound (field name)")

            if isinstance(slice_node.lower, ast.Name):
                field_name = slice_node.lower.id
            elif isinstance(slice_node.lower, ast.Constant) and isinstance(slice_node.lower.value, str):
                field_name = slice_node.lower.value
            else:
                raise TypeError(f"Invalid field name in slice: {ast.dump(slice_node.lower)}")

            if slice_node.upper is None:
                raise TypeError("Slice must have upper bound (field type)")

            field_type_ref = self._manual_evaluate(slice_node.upper)
            field_type = self._unwrap_value(field_type_ref)

            return (field_name, field_type)

        if isinstance(slice_node, ast.Tuple):
            items = [self._parse_subscript_items(elt) for elt in slice_node.elts]
            return builtins.tuple(items)

        value_ref = self._manual_evaluate(slice_node)
        return self._unwrap_value(value_ref)

    def _extract_type_from_valueref(self, value_ref):
        """Extract type from ValueRef(kind='python') or raw value."""
        if value_ref is None:
            return None

        if isinstance(value_ref, ValueRef):
            if value_ref.kind != "python":
                raise TypeError(f"Expected Python type ValueRef, got kind='{value_ref.kind}'")
            value = value_ref.value
        else:
            value = value_ref

        if isinstance(value, ir.VoidType):
            return value

        from .builtin_entities import BuiltinEntity

        if isinstance(value, type):
            try:
                if issubclass(value, BuiltinEntity) and value.can_be_type():
                    return value
            except TypeError:
                pass

            if getattr(value, "_is_struct", False) or getattr(value, "_is_enum", False):
                return value

        if isinstance(value, BuiltinEntity):
            return value

        if isinstance(value, str):
            entity = get_builtin_entity(value)
            if entity and entity.can_be_type():
                return entity
            if self.struct_registry.has_struct(value):
                return value

        raise TypeError(f"Cannot use as type: {value} (type: {type(value)})")

    def annotation_to_llvm_type(self, annotation) -> ir.Type:
        """
        Parse type annotation and return LLVM type.

        This is a convenience method that combines parse_annotation and
        extraction of LLVM type.

        Args:
            annotation: AST node representing a type annotation

        Returns:
            LLVM IR type
        """
        builtin_type = self.parse_annotation(annotation)

        if builtin_type is None:
            raise TypeError(f"TypeResolver: Cannot resolve type annotation: {annotation}")

        if (isinstance(builtin_type, type) and
            hasattr(builtin_type, "_is_struct") and
            builtin_type._is_struct):
            if hasattr(builtin_type, "get_llvm_type"):
                return builtin_type.get_llvm_type(self.module_context)
            struct_name = builtin_type.__name__
            if self.module_context:
                return self.module_context.get_identified_type(struct_name)
            else:
                raise TypeError("TypeResolver: module_context required for struct type resolution")

        if hasattr(builtin_type, "get_llvm_type"):
            return builtin_type.get_llvm_type(self.module_context)
        elif isinstance(builtin_type, ir.Type):
            raise TypeError(
                f"TypeResolver: received raw LLVM type {builtin_type} instead of BuiltinEntity. "
                f"This is a bug - LLVM types should not be used in Python type annotations. "
                f"Use BuiltinEntity types (i32, f64, ptr[T], etc.) instead."
            )
        else:
            raise TypeError(
                f"TypeResolver: cannot extract LLVM type from {builtin_type} (type: {type(builtin_type)})"
            )
