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
from .type_check import is_python_value


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
            visitor: AST visitor instance for expression evaluation (created lazily if not provided)
        """
        self.module_context = module_context
        self.user_globals = user_globals or {}
        self.visitor = visitor
        self._constexpr_visitor = None  # Lazy-created constexpr visitor
        from .registry import _unified_registry
        self.struct_registry = _unified_registry
    
    def _get_constexpr_visitor(self):
        """Get or create a constexpr visitor for type evaluation.
        
        This visitor uses ConstexprBackend and can evaluate type expressions
        without LLVM context.
        """
        if self._constexpr_visitor is None:
            from .backend import ConstexprBackend
            from .ast_visitor import LLVMIRVisitor
            
            backend = ConstexprBackend(user_globals=self.user_globals)
            self._constexpr_visitor = LLVMIRVisitor(
                backend=backend,
                user_globals=self.user_globals
            )
        return self._constexpr_visitor

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

        # Use visitor.visit_expression() for type evaluation
        value_ref = self._evaluate_type_expression(annotation)
        return self._extract_type_from_valueref(value_ref)

    def _evaluate_type_expression(self, node):
        """
        Evaluate type annotation AST, returning ValueRef(kind='python').

        Uses visitor.visit_expression() for evaluation. If no visitor was
        provided, uses a constexpr visitor with ConstexprBackend.

        Args:
            node: AST node, string, or Python value

        Returns:
            ValueRef(kind='python', value=<type>)
        """
        # Handle string annotations (from __future__ import annotations)
        if isinstance(node, str):
            parsed = ast.parse(node, mode="eval")
            return self._evaluate_type_expression(parsed.body)

        # Already a Python value (not AST, not string) - wrap it
        if not isinstance(node, ast.AST):
            return self._wrap_as_python_value(node)

        # String constant (from __future__ annotations) -> parse recursively
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parsed = ast.parse(node.value, mode="eval")
            return self._evaluate_type_expression(parsed.body)

        # Use provided visitor or constexpr visitor
        visitor = self.visitor if self.visitor is not None else self._get_constexpr_visitor()
        return visitor.visit_expression(node)

    def _wrap_as_python_value(self, value):
        """Wrap Python value as ValueRef(kind='python')."""
        if isinstance(value, ValueRef):
            return value
        return wrap_value(value, kind="python", type_hint=type(value))

    def _extract_type_from_valueref(self, value_ref):
        """Extract type from ValueRef(kind='python') or raw value."""
        if value_ref is None:
            return None

        if isinstance(value_ref, ValueRef):
            if not is_python_value(value_ref):
                raise TypeError(f"Expected Python type ValueRef, got kind='{value_ref.kind}'")
            value = value_ref.value
        else:
            value = value_ref

        # None value -> void type (for return type annotation like -> None)
        if value is None:
            from .builtin_entities import void
            return void

        if isinstance(value, ir.VoidType):
            return value

        from .builtin_entities import BuiltinEntity

        if isinstance(value, type):
            if issubclass(value, BuiltinEntity) and value.can_be_type():
                return value

            if getattr(value, "_is_struct", False) or getattr(value, "_is_enum", False) or getattr(value, "_is_union", False):
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
