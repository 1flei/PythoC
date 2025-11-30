"""
Refined Type Implementation

Refinement types with predicate validation:
- refined[Pred] creates a refined type with predicate constraint
- Internal representation: struct containing predicate parameters
- Runtime validation via refine() function (yield-based)
- Unchecked construction via assume() function
"""

import ast
import inspect
from typing import Optional, List, Any
from llvmlite import ir

from .composite_base import CompositeType
from .struct import struct
from ..logger import logger


class RefinedType(CompositeType):
    """Refinement type based on predicate function
    
    For single-parameter predicates, the refined type directly uses the
    underlying type (zero-overhead abstraction).
    
    For multi-parameter predicates, it's represented as a struct containing
    the parameters of the predicate function.
    
    Example:
        # Single parameter - direct type
        def is_positive(x: i32) -> bool:
            return x > 0
        PositiveInt = refined[is_positive]
        # Type: i32 (not struct)
        
        # Multiple parameters - struct
        def is_valid_range(start: i32, end: i32) -> bool:
            return start <= end
        ValidRange = refined[is_valid_range]
        # Type: struct[start: i32, end: i32]
    
    Attributes:
        _predicate_func: The predicate function object
        _param_types: List of parameter PC types from predicate
        _param_names: List of parameter names from predicate  
        _struct_type: Underlying struct type (None for single-param)
        _is_single_param: True if single parameter refinement
    """
    
    _is_refined = True
    _predicate_func: Optional[Any] = None
    _param_types: Optional[List[Any]] = None
    _param_names: Optional[List[str]] = None
    _struct_type: Optional[type] = None
    _is_single_param: bool = False
    
    @classmethod
    def get_name(cls) -> str:
        """Return refined type name"""
        if cls._predicate_func and hasattr(cls._predicate_func, '__name__'):
            return f"refined[{cls._predicate_func.__name__}]"
        return "refined"
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Returns underlying LLVM type
        
        For single-parameter refinements, returns the parameter's LLVM type.
        For multi-parameter refinements, returns the struct LLVM type.
        """
        if cls._is_single_param:
            # Single parameter: use the parameter type directly
            if cls._param_types and len(cls._param_types) > 0:
                param_type = cls._param_types[0]
                if hasattr(param_type, 'get_llvm_type'):
                    return param_type.get_llvm_type(module_context)
                else:
                    raise TypeError(f"{cls.get_name()} parameter type has no get_llvm_type method")
            else:
                raise TypeError(f"{cls.get_name()} has no parameter type")
        else:
            # Multi-parameter: use struct type
            if cls._struct_type is None:
                raise TypeError(f"{cls.get_name()} has no underlying struct type")
            return cls._struct_type.get_llvm_type(module_context)
    
    @classmethod
    def handle_subscript(cls, visitor, base, index, node):
        """Handle refined type subscript access
        
        Two cases:
        1. Type subscript: refined[pred_func] -> create RefinedType
        2. Value subscript: refined_value[i] -> delegate to struct (multi-param only)
        """
        from ..valueref import ValueRef
        
        # Case 1: Type subscript (refined[pred_func])
        if isinstance(base, type) and issubclass(base, RefinedType):
            # This is refined[pred_func] - create new refined type
            return cls._create_refined_type_from_predicate(index, node, visitor)
        
        # Case 2: Value subscript (refined_value[i])
        if cls._is_single_param:
            raise TypeError(
                f"{cls.get_name()} is a single-parameter refinement and does not support subscript access. "
                f"Use the value directly or access by field name: .{cls._param_names[0]}"
            )
        
        # Multi-parameter: delegate to underlying struct
        if cls._struct_type is None:
            raise TypeError(f"{cls.get_name()} has no underlying struct for subscript access")
        
        return cls._struct_type.handle_subscript(visitor, base, index, node)
    
    @classmethod
    def _create_refined_type_from_predicate(cls, predicate, node, visitor):
        """Create a new RefinedType from a predicate function
        
        Args:
            predicate: The predicate function (can be ValueRef or Python function)
            node: AST node for error reporting
            visitor: AST visitor instance
            
        Returns:
            New RefinedType class with predicate information extracted
        """
        from ..valueref import ValueRef
        from ..type_resolver import TypeResolver
        
        # Extract Python function from ValueRef if needed
        if isinstance(predicate, ValueRef):
            if predicate.kind == 'python' and callable(predicate.value):
                pred_func = predicate.value
            else:
                raise TypeError(f"refined type subscript must be a function, got {predicate}")
        elif callable(predicate):
            pred_func = predicate
        else:
            raise TypeError(f"refined type subscript must be a function, got {type(predicate)}")
        
        # Extract parameter information from predicate function
        try:
            sig = inspect.signature(pred_func)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Cannot inspect predicate function signature: {e}")
        
        param_names = []
        param_types = []
        
        # Get type resolver for annotation parsing
        type_resolver = TypeResolver(user_globals=visitor.user_globals if hasattr(visitor, 'user_globals') else {})
        
        for param_name, param in sig.parameters.items():
            param_names.append(param_name)
            
            # Extract type annotation
            if param.annotation == inspect.Parameter.empty:
                raise TypeError(
                    f"Predicate function '{pred_func.__name__}' parameter '{param_name}' "
                    f"must have type annotation"
                )
            
            # Parse annotation to PC type
            if isinstance(param.annotation, str):
                # String annotation - parse it
                pc_type = type_resolver.parse_annotation(param.annotation)
            elif isinstance(param.annotation, type):
                # Already a type - check if it's PC type or convert
                pc_type = param.annotation
            else:
                # Try to parse as AST node
                try:
                    pc_type = type_resolver.parse_annotation(param.annotation)
                except Exception as e:
                    raise TypeError(
                        f"Cannot parse type annotation for parameter '{param_name}': {e}"
                    )
            
            if pc_type is None:
                raise TypeError(
                    f"Predicate function '{pred_func.__name__}' parameter '{param_name}' "
                    f"has invalid type annotation: {param.annotation}"
                )
            
            param_types.append(pc_type)
        
        if len(param_types) == 0:
            raise TypeError(f"Predicate function '{pred_func.__name__}' must have at least one parameter")
        
        # Check if single parameter
        is_single_param = (len(param_types) == 1)
        
        # Create underlying struct type (only for multi-parameter)
        struct_type = None
        if not is_single_param:
            struct_type = struct._create_struct_type_from_fields(
                field_types=param_types,
                field_names=param_names
            )
        
        # Create new RefinedType subclass
        class_name = f"RefinedType_{pred_func.__name__}"
        new_refined_type = type(class_name, (RefinedType,), {
            '_predicate_func': pred_func,
            '_param_types': param_types,
            '_param_names': param_names,
            '_struct_type': struct_type,
            '_field_types': param_types,  # For CompositeType compatibility
            '_field_names': param_names,   # For CompositeType compatibility
            '_is_single_param': is_single_param,
        })
        
        logger.debug(
            f"Created refined type: {class_name} ({'single' if is_single_param else 'multi'}-param) "
            f"with {len(param_names)} parameter(s): {list(zip(param_names, param_types))}"
        )
        
        # Return as Python type wrapped in ValueRef
        from ..valueref import wrap_value
        from .python_type import PythonType
        return wrap_value(new_refined_type, kind='python', type_hint=PythonType(new_refined_type))
    
    @classmethod
    def handle_call(cls, visitor, args, node):
        """Handle refined type constructor call: refined[Pred](a, b, ...)
        
        This is equivalent to assume(a, b, ..., Pred) - creates the refined
        type instance without checking the predicate.
        
        For single-parameter refinements, returns the value directly.
        For multi-parameter refinements, returns a struct.
        
        Args:
            visitor: AST visitor instance
            args: Pre-evaluated argument ValueRefs
            node: ast.Call node
            
        Returns:
            ValueRef containing the refined type value
        """
        from ..valueref import wrap_value, ensure_ir
        
        # Verify argument count matches parameter count
        expected_count = len(cls._param_types) if cls._param_types else 0
        if len(args) != expected_count:
            raise TypeError(
                f"{cls.get_name()} takes {expected_count} argument(s) ({len(args)} given)"
            )
        
        if cls._is_single_param:
            # Single parameter: return the argument directly with refined type hint
            arg = args[0]
            field_type = cls._param_types[0]
            if field_type:
                arg = visitor.type_converter.convert(arg, field_type)
            
            # Return the value directly, but mark it with the refined type
            return wrap_value(
                ensure_ir(arg),
                kind='value',
                type_hint=cls
            )
        else:
            # Multi-parameter: build struct value
            if cls._struct_type is None:
                raise TypeError(f"{cls.get_name()} cannot be called (no struct type)")
            
            # Get LLVM struct type
            struct_llvm_type = cls._struct_type.get_llvm_type(visitor.module.context)
            
            # Create struct value by inserting each argument
            # Start with undef
            struct_value = ir.Constant(struct_llvm_type, ir.Undefined)
            
            # Insert each field value
            for i, arg in enumerate(args):
                # Convert arg to expected field type
                field_type = cls._param_types[i]
                if field_type:
                    arg = visitor.type_converter.convert(arg, field_type)
                arg_ir = ensure_ir(arg)
                struct_value = visitor.builder.insert_value(struct_value, arg_ir, i)
            
            # Wrap as refined type (same LLVM value, different PC type)
            return wrap_value(
                struct_value,
                kind='value',
                type_hint=cls
            )
    
    @classmethod
    def handle_attribute(cls, visitor, base, attr_name, node):
        """Handle attribute access on refined type value: refined_val.field
        
        For single-parameter refinements, delegate to the underlying type.
        For multi-parameter refinements, delegate to underlying struct.
        """
        from ..valueref import wrap_value, ensure_ir
        
        if cls._is_single_param:
            # Single parameter: delegate to the underlying type
            # First check if accessing by parameter name (return value itself)
            if attr_name == cls._param_names[0]:
                return base
            
            # Otherwise, delegate to the underlying type's handle_attribute
            # Need to change type_hint from refined type to underlying type
            underlying_type = cls._param_types[0]
            if underlying_type and hasattr(underlying_type, 'handle_attribute'):
                # Unwrap base and rewrap with underlying type hint
                base_ir = ensure_ir(base)
                base_with_underlying_type = wrap_value(
                    base_ir,
                    kind=base.kind,
                    type_hint=underlying_type,
                    address=base.address if hasattr(base, 'address') else None
                )
                return underlying_type.handle_attribute(visitor, base_with_underlying_type, attr_name, node)
            else:
                raise AttributeError(
                    f"{cls.get_name()} (refined {underlying_type}) has no attribute '{attr_name}'"
                )
        else:
            # Multi-parameter: delegate to struct
            if cls._struct_type is None:
                raise TypeError(f"{cls.get_name()} has no fields")
            
            return cls._struct_type.handle_attribute(visitor, base, attr_name, node)


class refined(metaclass=type):
    """Factory class for creating refined types
    
    Usage:
        refined[pred_func] -> RefinedType
    
    Example:
        def is_positive(x: i32) -> bool:
            return x > 0
        
        PositiveInt = refined[is_positive]
        x = PositiveInt(5)  # Create without checking
    """
    
    def __class_getitem__(cls, predicate):
        """Create refined type from predicate: refined[pred]
        
        This is a Python-level operation that creates a RefinedType class
        with the predicate information extracted.
        
        Args:
            predicate: Predicate function
            
        Returns:
            RefinedType class
        """
        if not callable(predicate):
            raise TypeError(f"refined subscript must be a callable, got {type(predicate)}")
        
        # Extract parameter information from predicate function
        try:
            sig = inspect.signature(predicate)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Cannot inspect predicate function signature: {e}")
        
        param_names = []
        param_types = []
        
        for param_name, param in sig.parameters.items():
            param_names.append(param_name)
            
            # Extract type annotation
            if param.annotation == inspect.Parameter.empty:
                # No annotation - will need to be resolved later during compilation
                param_types.append(None)
            else:
                # Store annotation as-is, will be resolved during compilation
                param_types.append(param.annotation)
        
        if len(param_names) == 0:
            raise TypeError(f"Predicate function must have at least one parameter")
        
        # Check if single parameter
        is_single_param = (len(param_names) == 1)
        
        # Create underlying struct type (only for multi-parameter)
        # Actual struct will be created with resolved types
        from .struct import create_struct_type
        
        struct_type = None
        if not is_single_param:
            # Try to create struct if all types are known and are PC types
            if all(t is not None for t in param_types):
                try:
                    struct_type = create_struct_type(
                        field_types=param_types,
                        field_names=param_names
                    )
                except Exception as e:
                    # If creation fails (e.g., types not yet available), leave as None
                    # Will be created during compilation
                    pass
        
        # Create new RefinedType subclass
        class_name = f"RefinedType_{predicate.__name__}"
        new_refined_type = type(class_name, (RefinedType,), {
            '_predicate_func': predicate,
            '_param_types': param_types,
            '_param_names': param_names,
            '_struct_type': struct_type,
            '_field_types': param_types,  # For CompositeType compatibility
            '_field_names': param_names,   # For CompositeType compatibility
            '_is_single_param': is_single_param,
        })
        
        return new_refined_type


__all__ = ['refined', 'RefinedType']
