"""
Refined Type Implementation

Refinement types with multiple forms:
- refined[T, "tag"] - Type with string tag
- refined[T, pred] - Type with predicate constraint  
- refined[T, pred, "tag"] - Type with both predicate and tag
- refined[T, "tag1", "tag2"] - Type with multiple tags
- refined[pred] - Multi-param predicate (backward compat)

Runtime validation via refine() function (yield-based)
Unchecked construction via assume() function or explicit cast
"""

import ast
import inspect
from typing import Optional, List, Any
from llvmlite import ir

from .composite_base import CompositeType
from .struct import struct, create_struct_type
from ..logger import logger


# =============================================================================
# Internal unified helpers for refined type construction
# =============================================================================

def _sanitize_class_name(name: str) -> str:
    """Sanitize a name for use in class name generation"""
    s = str(name).replace('[', '_').replace(']', '_').replace(',', '_')
    s = s.replace(' ', '').replace('"', '').replace("'", '')
    return s


def _generate_refined_class_name(base_type, predicates: List, tags: List[str]) -> str:
    """Generate a sanitized class name for a refined type"""
    parts = []
    if base_type is not None:
        base_name = base_type.get_name() if hasattr(base_type, 'get_name') else str(base_type)
        parts.append(_sanitize_class_name(base_name))
    for pred in predicates:
        pred_name = pred.__name__ if hasattr(pred, '__name__') else str(pred)
        parts.append(_sanitize_class_name(pred_name))
    for tag in tags:
        parts.append('tag_' + _sanitize_class_name(tag))
    return "RefinedType_" + '_'.join(parts) if parts else "RefinedType"


def _create_refined_type_class(
    base_type,
    predicates: List,
    tags: List[str],
    struct_type=None,
    param_types: Optional[List] = None,
    param_names: Optional[List[str]] = None,
    is_single_param: bool = True,
) -> type:
    """Create a new RefinedType subclass with the given attributes

    This is the unified factory that creates the actual type class.
    Both Python-level and AST visitor paths should use this.
    """
    class_name = _generate_refined_class_name(base_type, predicates, tags)

    # Determine field info
    if param_types is None:
        if base_type is not None:
            param_types = [base_type]
            param_names = ['value']
        else:
            param_types = []
            param_names = []

    if param_names is None:
        param_names = [f'field_{i}' for i in range(len(param_types))]

    new_refined_type = type(class_name, (RefinedType,), {
        '_base_type': base_type,
        '_predicates': predicates if predicates else [],
        '_tags': tags if tags else [],
        '_struct_type': struct_type,
        '_field_types': param_types,
        '_field_names': param_names,
        '_param_types': param_types,
        '_param_names': param_names,
        '_is_single_param': is_single_param,
    })

    return new_refined_type


def _parse_predicate_signature(predicate, type_resolver=None) -> tuple:
    """Parse predicate function signature to extract parameter types and names

    Args:
        predicate: Callable predicate function with type annotations
        type_resolver: Optional TypeResolver for parsing string annotations

    Returns:
        (param_names, param_types, is_single_param) tuple
    """
    try:
        sig = inspect.signature(predicate)
    except (ValueError, TypeError) as e:
        logger.error(f"Cannot inspect predicate function signature: {e}",
                    node=None, exc_type=TypeError)

    param_names = []
    param_types = []

    for param_name, param in sig.parameters.items():
        param_names.append(param_name)

        if param.annotation == inspect.Parameter.empty:
            # At Python level, we may not have type info yet
            param_types.append(None)
        elif isinstance(param.annotation, str):
            # String annotation - needs type resolver
            if type_resolver:
                pc_type = type_resolver.parse_annotation(param.annotation)
            else:
                pc_type = param.annotation  # Keep as string, resolve later
            param_types.append(pc_type)
        elif isinstance(param.annotation, type):
            param_types.append(param.annotation)
        else:
            # Try to parse via type resolver if available
            if type_resolver:
                try:
                    pc_type = type_resolver.parse_annotation(param.annotation)
                    param_types.append(pc_type)
                except Exception:
                    param_types.append(param.annotation)
            else:
                param_types.append(param.annotation)

    if len(param_names) == 0:
        logger.error(f"Predicate function '{predicate.__name__}' must have at least one parameter",
                    node=None, exc_type=TypeError)

    is_single_param = (len(param_names) == 1)
    return param_names, param_types, is_single_param


def _create_from_single_predicate_internal(predicate, type_resolver=None):
    """Create refined type from single predicate (backward compat: refined[pred])

    This handles the case where the predicate signature determines the base type.
    For single-param predicates, the first param type becomes the base type.
    For multi-param predicates, a struct is created.
    """
    param_names, param_types, is_single_param = _parse_predicate_signature(predicate, type_resolver)

    # Validate that we have resolved types for multi-param case
    if not is_single_param and any(t is None for t in param_types):
        logger.error(
            f"Predicate '{predicate.__name__}' parameters must have type annotations",
            node=None, exc_type=TypeError
        )

    if is_single_param:
        base_type = param_types[0] if param_types[0] else None
        struct_type = None
    else:
        base_type = None
        # Create struct for multi-param predicate
        if all(t is not None for t in param_types):
            struct_type = create_struct_type(
                field_types=param_types,
                field_names=param_names
            )
        else:
            struct_type = None

    return _create_refined_type_class(
        base_type=base_type,
        predicates=[predicate],
        tags=[],
        struct_type=struct_type,
        param_types=param_types,
        param_names=param_names,
        is_single_param=is_single_param,
    )


def _create_from_base_and_constraints_internal(base_type, predicates: List, tags: List[str]):
    """Create refined type from base type + predicates + tags

    This handles:
    - refined[T, "tag"]
    - refined[T, pred]
    - refined[T, pred, "tag"]
    - refined[T, "tag1", "tag2"]
    """
    # Validate predicates have single parameter
    for pred in predicates:
        try:
            sig = inspect.signature(pred)
            params = list(sig.parameters.values())
            if len(params) != 1:
                base_name = base_type.get_name() if hasattr(base_type, 'get_name') else str(base_type)
                logger.error(
                    f"Predicate '{pred.__name__}' for refined[{base_name}, ...] "
                    f"must have exactly one parameter, got {len(params)}",
                    node=None, exc_type=TypeError
                )
        except (ValueError, TypeError) as e:
            logger.error(f"Cannot inspect predicate function signature: {e}",
                        node=None, exc_type=TypeError)

    return _create_refined_type_class(
        base_type=base_type,
        predicates=predicates,
        tags=tags,
        struct_type=None,
        param_types=[base_type],
        param_names=['value'],
        is_single_param=True,
    )


# =============================================================================
# RefinedType class
# =============================================================================

class RefinedType(CompositeType):
    """Refinement type with predicates and/or tags
    
    Supports multiple forms:
    - refined[T, "tag"] - Type with tag (no predicate)
    - refined[T, pred] - Type with predicate constraint
    - refined[T, pred, "tag"] - Type with both predicate and tag
    - refined[T, "tag1", "tag2"] - Type with multiple tags
    - refined[T, pred1, pred2, "tag"] - Multiple predicates and tags
    
    For base type + constraints, uses the base type (zero-overhead).
    For multi-parameter predicates, it's a struct.
    
    Conversion rules:
    - Explicit cast (T2(t1) or assume(t, ...)): always allowed
    - Implicit cast: refined with MORE terms -> refined with FEWER terms
      Example: refined[T, pred, "tag"] -> refined[T, pred] (implicit OK)
               refined[T, pred] -> refined[T, pred, "tag"] (explicit only)
    - To base type: all refined types can implicitly cast to T
    
    Attributes:
        _base_type: The underlying base type (T in refined[T, ...])
        _predicates: List of predicate functions
        _tags: List of string tags
        _struct_type: Underlying struct type (None for single-param)
        _param_types: List of parameter types (for multi-param predicates)
        _param_names: List of parameter names (for multi-param predicates)
    """
    
    _is_refined = True
    _base_type: Optional[Any] = None
    _predicates: Optional[List[Any]] = None
    _tags: Optional[List[str]] = None
    _struct_type: Optional[type] = None
    _param_types: Optional[List[Any]] = None
    _param_names: Optional[List[str]] = None
    
    @classmethod
    def get_name(cls) -> str:
        """Return refined type name"""
        parts = []
        if cls._base_type:
            base_name = cls._base_type.get_name() if hasattr(cls._base_type, 'get_name') else str(cls._base_type)
            parts.append(base_name)
        if cls._predicates:
            for pred in cls._predicates:
                pred_name = pred.__name__ if hasattr(pred, '__name__') else str(pred)
                parts.append(pred_name)
        if cls._tags:
            for tag in cls._tags:
                parts.append(f'"{tag}"')
        if parts:
            return f"refined[{', '.join(parts)}]"
        return "refined"
    
    @classmethod
    def get_ctypes_type(cls):
        """Get ctypes type for refined type.
        
        Delegates to the base type or struct type.
        """
        # If we have a base type, use it
        if cls._base_type is not None:
            if hasattr(cls._base_type, 'get_ctypes_type'):
                return cls._base_type.get_ctypes_type()
        
        # Multi-parameter predicate: use struct type
        if cls._struct_type is not None:
            if hasattr(cls._struct_type, 'get_ctypes_type'):
                return cls._struct_type.get_ctypes_type()
        
        return None
    
    @classmethod
    def get_size_bytes(cls) -> int:
        """Get size in bytes for refined type.
        
        Delegates to the base type or struct type.
        """
        # If we have a base type, use it
        if cls._base_type is not None:
            if hasattr(cls._base_type, 'get_size_bytes'):
                return cls._base_type.get_size_bytes()
        
        # Multi-parameter predicate: use struct type
        if cls._struct_type is not None:
            if hasattr(cls._struct_type, 'get_size_bytes'):
                return cls._struct_type.get_size_bytes()
        
        return 0
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Returns underlying LLVM type
        
        For base type + tags/predicates, returns base type's LLVM type.
        For multi-parameter predicates, returns the struct LLVM type.
        """
        # If we have a base type, use it
        if cls._base_type is not None:
            if hasattr(cls._base_type, 'get_llvm_type'):
                return cls._base_type.get_llvm_type(module_context)
            else:
                logger.error(f"{cls.get_name()} base type has no get_llvm_type method", node=None, exc_type=TypeError)
        
        # Multi-parameter predicate: use struct type
        if cls._struct_type is not None:
            return cls._struct_type.get_llvm_type(module_context)
        
        # Fallback to param_types for backward compatibility
        if cls._param_types and len(cls._param_types) > 0:
            param_type = cls._param_types[0]
            if hasattr(param_type, 'get_llvm_type'):
                return param_type.get_llvm_type(module_context)
        
        logger.error(f"{cls.get_name()} has no type information", node=None, exc_type=TypeError)
    
    @classmethod
    def handle_subscript(cls, visitor, base, index, node):
        """Handle refined type subscript access
        
        Two cases:
        1. Type subscript: refined[T, pred, "tag", ...] -> create RefinedType
        2. Value subscript: refined_value[i] -> delegate to struct (multi-param only)
        """
        from ..valueref import ValueRef
        
        # Case 1: Type subscript (refined[...])
        if isinstance(base, type) and issubclass(base, RefinedType):
            return cls._create_refined_type_from_args(index, node, visitor)
        
        # Case 2: Value subscript (refined_value[i])
        if cls._base_type is not None:
            if cls._struct_type is None:
                logger.error(
                    f"{cls.get_name()} subscript depends on base type {cls._base_type}",
                    node=node, exc_type=TypeError
                )
        
        # Multi-parameter: delegate to underlying struct
        if cls._struct_type is None:
            logger.error(f"{cls.get_name()} has no underlying struct for subscript access", node=node, exc_type=TypeError)
        
        return cls._struct_type.handle_subscript(visitor, base, index, node)
    
    @classmethod
    def _create_refined_type_from_args(cls, args, node, visitor):
        """Create a new RefinedType from mixed arguments (AST visitor entry point)

        Supports:
        - refined[ptr[T], "owned"] - base type + tag
        - refined[i32, is_positive] - base type + predicate
        - refined[i32, is_positive, "positive"] - base type + predicate + tag
        - refined[i32, "tag1", "tag2"] - base type + multiple tags
        - refined[is_valid_range] - multi-param predicate only (backward compat)

        Args:
            args: Single arg or tuple of args
            node: AST node for error reporting
            visitor: AST visitor instance

        Returns:
            New RefinedType class wrapped in ValueRef
        """
        from ..valueref import ValueRef, wrap_value
        from ..type_resolver import TypeResolver
        from .python_type import PythonType

        # Normalize args to list
        if isinstance(args, tuple):
            args_list = list(args)
        else:
            args_list = [args]

        if len(args_list) == 0:
            logger.error("refined requires at least one argument", node=node, exc_type=TypeError)

        # Parse arguments into: base_type, predicates, tags
        base_type = None
        predicates = []
        tags = []

        for i, arg in enumerate(args_list):
            # Unwrap ValueRef to get actual value
            if isinstance(arg, ValueRef):
                if arg.kind == 'python' and arg.value is not None:
                    arg_value = arg.value
                else:
                    logger.error(f"refined argument must be a type, predicate, or string tag", node=node, exc_type=TypeError)
            else:
                arg_value = arg

            # Check if it's a string tag
            if isinstance(arg_value, str):
                tags.append(arg_value)
            # Check if it's a callable (predicate)
            elif callable(arg_value):
                predicates.append(arg_value)
            # Check if it's a type
            elif isinstance(arg_value, type):
                # First type becomes base_type
                if base_type is None and i == 0:
                    base_type = arg_value
                else:
                    logger.error(f"refined can only have one base type (position 0), got type at position {i}", node=node, exc_type=TypeError)
            else:
                logger.error(f"refined argument must be a type, callable predicate, or string tag, got {type(arg_value)}", node=node, exc_type=TypeError)

        # Validate combinations
        if len(predicates) == 0 and len(tags) == 0:
            logger.error("refined requires at least one predicate or tag", node=node, exc_type=TypeError)

        # Case 1: Only predicates, no base type (backward compat: refined[pred])
        if base_type is None and len(predicates) == 1 and len(tags) == 0:
            type_resolver = TypeResolver(user_globals=visitor.user_globals if hasattr(visitor, 'user_globals') else {})
            new_type = _create_from_single_predicate_internal(predicates[0], type_resolver)
            return wrap_value(new_type, kind='python', type_hint=PythonType(new_type))

        # Case 2: Base type with tags/predicates
        if base_type is not None:
            new_type = _create_from_base_and_constraints_internal(base_type, predicates, tags)
            return wrap_value(new_type, kind='python', type_hint=PythonType(new_type))

        # Case 3: Multiple predicates without base type
        if base_type is None and len(predicates) > 0:
            logger.error(
                "refined with multiple predicates requires explicit base type: "
                "refined[T, pred1, pred2, ...]",
                node=node, exc_type=TypeError
            )

        logger.error(f"Invalid refined type specification: {args_list}", node=node, exc_type=TypeError)

    @classmethod
    def handle_call(cls, visitor, func_ref, args, node):
        """Handle refined type constructor call: refined[...](value)
        
        Creates refined type instance without checking predicates.
        Equivalent to assume(value, ...).
        """
        from ..valueref import wrap_value, ensure_ir, ValueRef
        
        # For base + constraints
        if cls._base_type is not None:
            # Refinement types cannot be constructed from nothing
            # Must use assume(base_value, ...) instead
            if len(args) == 0:
                logger.error(
                    f"{cls.get_name()} cannot be constructed without arguments. "
                    f"Refinement types must be created from a base value using assume(base_value, ...)",
                    node=node, exc_type=TypeError
                )
            elif len(args) == 1:
                arg = args[0]
                # If arg is Python value, promote it first
                if isinstance(arg, ValueRef) and arg.is_python_value():
                    arg = visitor.type_converter._promote_python_to_pc(arg.get_python_value(), cls._base_type)
                elif cls._base_type:
                    arg = visitor.type_converter.convert(arg, cls._base_type)
                
                return wrap_value(ensure_ir(arg), kind='value', type_hint=cls)
            else:
                logger.error(f"{cls.get_name()} takes 0 or 1 argument ({len(args)} given)", node=node, exc_type=TypeError)
        
        # For multi-param predicates: expect N args
        expected_count = len(cls._param_types) if cls._param_types else 0
        if len(args) != expected_count:
            logger.error(f"{cls.get_name()} takes {expected_count} argument(s) ({len(args)} given)", node=node, exc_type=TypeError)
        
        if cls._struct_type is None:
            logger.error(f"{cls.get_name()} cannot be called (no struct type)", node=node, exc_type=TypeError)
        
        struct_llvm_type = cls._struct_type.get_llvm_type(visitor.module.context)
        struct_value = ir.Constant(struct_llvm_type, ir.Undefined)
        
        for i, arg in enumerate(args):
            field_type = cls._param_types[i]
            if field_type:
                arg = visitor.type_converter.convert(arg, field_type)
            arg_ir = ensure_ir(arg)
            struct_value = visitor.builder.insert_value(struct_value, arg_ir, i)
        
        return wrap_value(struct_value, kind='value', type_hint=cls)
    
    @classmethod
    def handle_attribute(cls, visitor, base, attr_name, node):
        """Handle attribute access on refined type value"""
        from ..valueref import wrap_value, ensure_ir
        
        # For base + constraints: delegate to base type
        if cls._base_type is not None:
            if attr_name == 'value':
                return base
            
            underlying_type = cls._base_type
            if underlying_type and hasattr(underlying_type, 'handle_attribute'):
                base_ir = ensure_ir(base)
                base_with_underlying_type = wrap_value(
                    base_ir,
                    kind=base.kind,
                    type_hint=underlying_type,
                    address=base.address if hasattr(base, 'address') else None
                )
                return underlying_type.handle_attribute(visitor, base_with_underlying_type, attr_name, node)
            else:
                logger.error(f"{cls.get_name()} (refined {underlying_type}) has no attribute '{attr_name}'", node=node, exc_type=AttributeError)
        
        # For multi-param: delegate to struct
        if cls._struct_type is None:
            logger.error(f"{cls.get_name()} has no fields", node=node, exc_type=TypeError)
        
        return cls._struct_type.handle_attribute(visitor, base, attr_name, node)


class refined(metaclass=type):
    """Factory class for creating refined types

    Usage:
        refined[pred] -> RefinedType (backward compat)
        refined[T, "tag"] -> RefinedType
        refined[T, pred, "tag"] -> RefinedType

    Example:
        def is_positive(x: i32) -> bool:
            return x > 0

        PositiveInt = refined[is_positive]
        OwnedPtr = refined[ptr[T], "owned"]
    """

    def __class_getitem__(cls, args):
        """Create refined type: refined[...]

        Python-level operation that creates a RefinedType class.
        Uses unified internal helpers.
        """
        # Handle both single arg and tuple
        if not isinstance(args, tuple):
            args = (args,)

        # Check for simple single predicate case (backward compat at Python level)
        if len(args) == 1 and callable(args[0]) and not isinstance(args[0], type):
            # refined[pred] - use internal helper (no type resolver at Python level)
            return _create_from_single_predicate_internal(args[0], type_resolver=None)

        # For other cases: base + tags/predicates
        # Parse arguments: base_type, predicates, tags
        base_type = None
        predicates = []
        tags = []

        for i, arg in enumerate(args):
            if isinstance(arg, str):
                tags.append(arg)
            elif callable(arg) and not isinstance(arg, type):
                predicates.append(arg)
            elif isinstance(arg, type):
                if base_type is None and i == 0:
                    base_type = arg
                else:
                    logger.error(f"refined can only have one base type at position 0",
                                node=None, exc_type=TypeError)
            else:
                logger.error(f"refined argument must be a type, callable predicate, or string tag",
                            node=None, exc_type=TypeError)

        if base_type is None:
            logger.error("refined[...] requires a base type as first argument",
                        node=None, exc_type=TypeError)

        # Use internal helper for base type + constraints
        return _create_from_base_and_constraints_internal(base_type, predicates, tags)


__all__ = ['refined', 'RefinedType']
