"""
Base classes for builtin entities

This module contains the core base classes and metaclass for the builtin entity system.
"""

from abc import ABC, ABCMeta, abstractmethod
from llvmlite import ir
from typing import Any, Optional
import ast
import ctypes
from ..logger import logger


def _get_unified_registry():
    """Lazy import to avoid circular dependency"""
    from ..registry import _unified_registry
    return _unified_registry


class BuiltinEntityMeta(ABCMeta):
    """Metaclass for automatic registration of built-in entities"""
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Only register concrete classes
        if (not name.startswith('_') and 
            name not in ['BuiltinEntity', 'BuiltinType', 'BuiltinFunction', 'PythonType'] and
            not getattr(cls, '__abstractmethods__', None)):
            try:
                # Get the entity name from the class
                # For types like i32, the class name IS the entity name
                entity_name = name.lower()
                
                # Register to unified registry
                # Store for later registration to avoid circular import
                if not hasattr(mcs, '_pending_registrations'):
                    mcs._pending_registrations = []
                mcs._pending_registrations.append((entity_name, cls))
            except Exception as e:
                logger.error(f"Failed to register builtin entity {name}: {e}", node=None, exc_type=RuntimeError)
        
        return cls
    
    @classmethod
    def _register_pending(mcs):
        """Register any pending entities to unified registry"""
        if hasattr(mcs, '_pending_registrations'):
            registry = _get_unified_registry()
            for entity_name, cls in mcs._pending_registrations:
                registry.register_builtin_entity(entity_name, cls)
            mcs._pending_registrations.clear()





class BuiltinEntity(ABC, metaclass=BuiltinEntityMeta):
    """Base class for all built-in entities (types, functions, etc.)"""
    
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the entity name (e.g., 'i32', 'sizeof')"""
        pass
    
    @classmethod
    def can_be_type(cls) -> bool:
        """Can this entity be used as a type annotation?"""
        return False
    
    @classmethod
    def can_be_called(cls) -> bool:
        """Can this entity be called as a function?"""
        return False
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> Optional[ir.Type]:
        """Get LLVM type (for type entities)
        
        Args:
            module_context: Optional IR module context for types that need it (e.g., IdentifiedStructType)
        """
        return None
    
    @classmethod
    def get_size_bytes(cls) -> Optional[int]:
        """Get size in bytes (for type entities)"""
        return None
    
    @classmethod
    def is_signed(cls) -> bool:
        """Is this a signed type? (for integer types)"""
        return True
    
    @classmethod
    def is_struct_type(cls) -> bool:
        """Is this a struct type? Override in struct to return True."""
        return False
    
    @classmethod
    def is_enum_type(cls) -> bool:
        """Is this an enum type? Override in enum to return True."""
        return False


class BuiltinType(BuiltinEntity):
    """Base class for built-in type entities"""
    
    # Subclasses should define these
    _llvm_type: ir.Type = None
    _size_bytes: int = None
    _is_signed: bool = True
    _is_integer: bool = False
    _is_float: bool = False
    _is_bool: bool = False
    _is_pointer: bool = False
    
    @classmethod
    def get_ctypes_type(cls) -> Any:
        """Get the corresponding ctypes type for FFI.
        
        This method provides the correct ctypes type for function signatures,
        properly handling signed/unsigned distinction that LLVM IR doesn't preserve.
        
        Returns:
            ctypes type (e.g., ctypes.c_int32, ctypes.c_uint32, etc.)
        """
        # Bool type
        if cls._is_bool:
            return ctypes.c_bool
        
        # Integer types - use signedness info
        if cls._is_integer:
            size = cls._size_bytes
            if cls._is_signed:
                if size == 1:
                    return ctypes.c_int8
                elif size == 2:
                    return ctypes.c_int16
                elif size <= 4:
                    return ctypes.c_int32
                else:
                    return ctypes.c_int64
            else:
                if size == 1:
                    return ctypes.c_uint8
                elif size == 2:
                    return ctypes.c_uint16
                elif size <= 4:
                    return ctypes.c_uint32
                else:
                    return ctypes.c_uint64
        
        # Float types
        if cls._is_float:
            if cls._size_bytes == 4:
                return ctypes.c_float
            else:
                return ctypes.c_double
        
        # Pointer types
        if cls._is_pointer:
            return ctypes.c_void_p
        
        # Void type
        if isinstance(cls._llvm_type, ir.VoidType):
            return None
        
        # Default fallback
        return ctypes.c_void_p
    
    @classmethod
    def normalize_subscript_items(cls, items):
        """Normalize subscript items to standard format: Tuple[Tuple[Optional[str], type], ...]
        
        This method converts various input formats to a completely unified format where
        EVERY item is a tuple of (Optional[str], type).
        
        Args:
            items: Input from __class_getitem__ or TypeResolver, can be:
                - Single item: i32
                - Tuple of types: (i32, f64)
                - Tuple with slices: (slice("x", i32), slice("y", f64))
                - Tuple with named tuples: (("x", i32), ("y", f64))
                - Mixed: (i32, slice("y", f64), ("z", i32))
                - refined[struct[...], "slice"]: from visit_Slice (AST parsing)
                - refined[struct[...], "tuple"]: from visit_Tuple (AST parsing)
        
        Returns:
            Tuple[Tuple[Optional[str], type], ...] where:
                - Unnamed field: (None, type)
                - Named field: ("name", type)
        
        Examples:
            i32 -> ((None, i32),)
            (i32, f64) -> ((None, i32), (None, f64))
            (slice("x", i32), slice("y", f64)) -> (("x", i32), ("y", f64))
            (i32, slice("y", f64)) -> ((None, i32), ("y", f64))
            (("x", i32), i32) -> (("x", i32), (None, i32))
        """
        import builtins
        from .refined import RefinedType
        
        # First, unwrap refined[..., "tuple"] if present (from visit_Tuple in AST parsing)
        # This handles the case where AST parsing produces a single refined type
        # containing multiple elements
        if not isinstance(items, builtins.tuple):
            if isinstance(items, type) and issubclass(items, RefinedType):
                tags = getattr(items, '_tags', [])
                if "tuple" in tags:
                    # Recursively extract items from refined tuple
                    items = cls._extract_tuple_from_refined(items)
            if not isinstance(items, builtins.tuple):
                items = (items,)
        
        normalized = []
        for item in items:
            normalized.append(cls._normalize_single_item(item))
        
        return tuple(normalized)
    
    @classmethod
    def _normalize_single_item(cls, item):
        """Normalize a single subscript item to (Optional[str], type) format.
        
        Handles:
        - slice("name", type, None): Python runtime named field
        - ("name", type): already normalized tuple
        - refined[struct[...], "slice"]: from visit_Slice (AST parsing)
        - refined[struct[...], "tuple"]: recursively extract items
        - other: unnamed field -> (None, item)
        """
        import builtins
        from .refined import RefinedType
        
        # Case 1: Python runtime slice object
        if isinstance(item, builtins.slice):
            if item.start is None or item.stop is None:
                logger.error("Named field requires both name and type", node=None, exc_type=TypeError)
            field_name = item.start
            if not isinstance(field_name, str):
                logger.error(f"Field name must be a string, got {type(field_name)}", node=None, exc_type=TypeError)
            return (field_name, item.stop)
        
        # Case 2: Already normalized ("name", type) tuple
        if isinstance(item, builtins.tuple) and len(item) == 2 and isinstance(item[0], str):
            return item
        
        # Case 3: refined[struct[...], "slice"] from visit_Slice
        if isinstance(item, type) and issubclass(item, RefinedType):
            tags = getattr(item, '_tags', [])
            
            if "slice" in tags:
                # Extract ("name", type) from refined[struct[pyconst["name"], pyconst[type]], "slice"]
                return cls._extract_slice_from_refined(item)
            
            # "tuple" tag should be handled at normalize_subscript_items level, not here
            # If we get here with "tuple" tag, something is wrong
            if "tuple" in tags:
                logger.error("refined[..., 'tuple'] should be unwrapped at normalize_subscript_items level",
                            node=None, exc_type=TypeError)
        
        # Case 4: Unnamed item
        return (None, item)
    
    @classmethod
    def _extract_slice_from_refined(cls, refined_type):
        """Extract (name, type) from refined[struct[pyconst["name"], pyconst[type]], "slice"]
        
        This handles the output of visit_Slice when parsing AST type annotations.
        """
        base_type = getattr(refined_type, '_base_type', None)
        if base_type is None:
            logger.error(f"Invalid slice type: {refined_type}", node=None, exc_type=TypeError)
        
        # base_type is struct[pyconst["name"], pyconst[type]]
        field_types = getattr(base_type, '_field_types', [])
        if len(field_types) != 2:
            logger.error(f"Slice must have exactly 2 fields (name, type), got {len(field_types)}",
                        node=None, exc_type=TypeError)
        
        # Extract name from pyconst["name"]
        name_type = field_types[0]
        if hasattr(name_type, '_python_object'):
            name = name_type._python_object
        elif hasattr(name_type, 'get_python_object'):
            name = name_type.get_python_object()
        else:
            logger.error(f"Cannot extract name from {name_type}", node=None, exc_type=TypeError)
        
        # Extract type from pyconst[type]
        type_type = field_types[1]
        if hasattr(type_type, '_python_object'):
            field_type = type_type._python_object
        elif hasattr(type_type, 'get_python_object'):
            field_type = type_type.get_python_object()
        else:
            logger.error(f"Cannot extract type from {type_type}", node=None, exc_type=TypeError)
        
        return (name, field_type)
    
    @classmethod
    def _extract_tuple_from_refined(cls, refined_type):
        """Extract items from refined[struct[...], "tuple"]
        
        This handles the output of visit_Tuple when parsing AST type annotations.
        The refined type wraps a struct where each field is a pyconst containing
        either a type or another refined type (for slices).
        
        Returns:
            Tuple of items (types or (name, type) tuples)
        """
        from .refined import RefinedType
        
        base_type = getattr(refined_type, '_base_type', None)
        if base_type is None:
            logger.error(f"Invalid tuple type: {refined_type}", node=None, exc_type=TypeError)
        
        field_types = getattr(base_type, '_field_types', [])
        items = []
        
        for field_type in field_types:
            # Extract actual value from pyconst wrapper
            if hasattr(field_type, '_python_object'):
                actual_value = field_type._python_object
            elif hasattr(field_type, 'get_python_object'):
                actual_value = field_type.get_python_object()
            else:
                actual_value = field_type
            
            # Check if it's a nested refined type (slice or tuple)
            if isinstance(actual_value, type) and issubclass(actual_value, RefinedType):
                tags = getattr(actual_value, '_tags', [])
                if "slice" in tags:
                    items.append(cls._extract_slice_from_refined(actual_value))
                    continue
                if "tuple" in tags:
                    # Nested tuple - recursively extract and flatten
                    nested = cls._extract_tuple_from_refined(actual_value)
                    items.extend(nested)
                    continue
            
            # Plain value
            items.append(actual_value)
        
        return tuple(items)
    
    @classmethod
    def handle_type_subscript(cls, items):
        """Handle type subscript with normalized items
        
        This method should be overridden by subclasses that support subscript syntax.
        The items are already normalized by normalize_subscript_items.
        
        Args:
            items: Normalized tuple of items (from normalize_subscript_items or TypeResolver)
        
        Returns:
            Specialized type class
        
        Raises:
            TypeError: If the type doesn't support subscript syntax
        """
        logger.error(f"{cls.get_name()} does not support subscript syntax", node=None, exc_type=TypeError)
    
    def __class_getitem__(cls, item):
        """Python runtime entry point for type subscript syntax
        
        This is the unified entry point for all types. It:
        1. Normalizes the input (slice objects -> tuples)
        2. Delegates to handle_type_subscript
        
        Examples:
            Type[item] -> normalize_subscript_items(item) -> handle_type_subscript(normalized)
        """
        # Normalize slice objects to standard format
        normalized = cls.normalize_subscript_items(item)
        # Delegate to type-specific handler
        return cls.handle_type_subscript(normalized)
    
    @classmethod
    def can_be_type(cls) -> bool:
        return True
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True  # Types can be called for conversion
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get LLVM type for this builtin type
        
        Args:
            module_context: Optional IR module context (not used by basic types, but accepted for interface uniformity)
        """
        return cls._llvm_type
    
    @classmethod
    def get_size_bytes(cls) -> int:
        return cls._size_bytes
    
    @classmethod
    def is_signed(cls) -> bool:
        return cls._is_signed
    
    @classmethod
    def is_integer(cls) -> bool:
        return cls._is_integer
    
    @classmethod
    def is_float(cls) -> bool:
        return cls._is_float
    
    @classmethod
    def is_bool(cls) -> bool:
        return cls._is_bool
    
    @classmethod
    def is_pointer(cls) -> bool:
        return cls._is_pointer
    
    @classmethod
    def is_array(cls) -> bool:
        return False
    
    @classmethod
    def get_type_id(cls) -> str:
        """
        Generate unique type ID for this type.
        Default implementation for primitive types based on their structural properties.
        """
        return cls.get_name()
    
    @classmethod
    def is_compatible_with(cls, other) -> bool:
        """Check if this type is compatible with another type"""
        if cls == other:
            return True
        
        # Integer promotion rules
        if cls.is_integer() and hasattr(other, 'is_integer') and other.is_integer():
            # Can promote smaller to larger
            return cls.get_size_bytes() <= other.get_size_bytes()
        
        # Float promotion rules
        if cls.is_float() and hasattr(other, 'is_float') and other.is_float():
            return cls.get_size_bytes() <= other.get_size_bytes()
        
        # Integer to float promotion
        if cls.is_integer() and hasattr(other, 'is_float') and other.is_float():
            return True
        
        # Bool to integer promotion
        if cls.is_bool() and hasattr(other, 'is_integer') and other.is_integer():
            return True
        
        return False
    
    @classmethod
    def pointer_to(cls):
        """Create a pointer type to this type"""
        # Import ptr here to avoid circular dependency
        from .types import ptr
        
        # Create a specialized pointer class
        pointee_name = cls.get_name()
        new_name = f"ptr[{pointee_name}]"
        
        return type(
            new_name,
            (ptr,),
            {
                'pointee_type': cls,
                '_name': new_name,
            }
        )
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call) -> ir.Value:
        """Handle type conversion: i32(x)
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the callable (the type itself for type conversions)
            args: Pre-evaluated arguments (list of ValueRef)
            node: ast.Call node
        """
        return cls.handle_type_conversion(visitor, node)
    
    @classmethod
    def handle_type_conversion(cls, visitor, node: ast.Call) -> ir.Value:
        """Convert value to this type using TypeConverter"""
        if len(node.args) != 1:
            logger.error(f"{cls.get_name()}() takes exactly 1 argument ({len(node.args)} given)",
                        node=node, exc_type=TypeError)
        
        arg = visitor.visit_expression(node.args[0])
        # Note: TypeConverter will extract LLVM type from pythoc type with module_context
        # So we don't need to call get_llvm_type here - just pass the PC type class
        
        # Determine if target is unsigned
        target_is_unsigned = not cls.is_signed() if hasattr(cls, 'is_signed') else False
        
        # Use TypeConverter for the conversion (pass PC type directly)
        try:
            result = visitor.type_converter.convert(
                arg, 
                cls
            )
            return result
        except TypeError as e:
            logger.error(f"Cannot convert to {cls.get_name()}: {e}", node=node, exc_type=TypeError)
    
    # Binary operation handlers for numeric types (unified call protocol)
    @classmethod
    def handle_add(cls, visitor, left, right, node: ast.BinOp):
        """Handle addition for numeric types"""
        from ..valueref import wrap_value, ensure_ir
        left, right, is_float = visitor.type_converter.unify_binop_types(left, right)
        
        if is_float:
            result = visitor.builder.fadd(ensure_ir(left), ensure_ir(right))
        else:
            result = visitor.builder.add(ensure_ir(left), ensure_ir(right))
        return wrap_value(result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_sub(cls, visitor, left, right, node: ast.BinOp):
        """Handle subtraction for numeric types"""
        from ..valueref import wrap_value, ensure_ir
        left, right, is_float = visitor.type_converter.unify_binop_types(left, right)
        
        if is_float:
            result = visitor.builder.fsub(ensure_ir(left), ensure_ir(right))
        else:
            result = visitor.builder.sub(ensure_ir(left), ensure_ir(right))
        return wrap_value(result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_mul(cls, visitor, left, right, node: ast.BinOp):
        """Handle multiplication for numeric types"""
        from ..valueref import wrap_value, ensure_ir
        left = visitor.visit_expression(node.left)
        right = visitor.visit_expression(node.right)
        left, right, is_float = visitor.type_converter.unify_binop_types(left, right)
        
        if is_float:
            result = visitor.builder.fmul(ensure_ir(left), ensure_ir(right))
        else:
            result = visitor.builder.mul(ensure_ir(left), ensure_ir(right))
        return wrap_value(result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_div(cls, visitor, left, right, node: ast.BinOp):
        """Handle division for numeric types (always returns float)"""
        from ..valueref import wrap_value, ensure_ir, get_type
        from llvmlite import ir
        from ..builtin_entities import f64
        
        # Division always promotes to float
        if not (hasattr(left.type_hint, '_is_float') and left.type_hint._is_float):
            left = visitor._promote_to_float(left, f64)
        if not (hasattr(right.type_hint, '_is_float') and right.type_hint._is_float):
            right = visitor._promote_to_float(right, f64)
        
        result = visitor.builder.fdiv(ensure_ir(left), ensure_ir(right))
        return wrap_value(result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_floordiv(cls, visitor, left, right, node: ast.BinOp):
        """Handle floor division for numeric types"""
        from ..valueref import wrap_value, ensure_ir, get_type
        from llvmlite import ir
        left, right, is_float = visitor.type_converter.unify_binop_types(left, right)
        
        if is_float:
            result = visitor.builder.fdiv(ensure_ir(left), ensure_ir(right))
            # Use f64 intrinsic for floor (since we promote to f64)
            result = visitor.builder.call(visitor._get_floor_intrinsic(ir.DoubleType()), [result])
        else:
            result = visitor.builder.sdiv(ensure_ir(left), ensure_ir(right))
        return wrap_value(result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_mod(cls, visitor, left, right, node: ast.BinOp):
        """Handle modulo for numeric types"""
        from ..valueref import wrap_value, ensure_ir
        left, right, is_float = visitor.type_converter.unify_binop_types(left, right)
        
        if is_float:
            result = visitor.builder.frem(ensure_ir(left), ensure_ir(right))
        else:
            result = visitor.builder.srem(ensure_ir(left), ensure_ir(right))
        return wrap_value(result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_pow(cls, visitor, left, right, node: ast.BinOp):
        """Handle power for numeric types"""
        from ..valueref import wrap_value, ensure_ir, get_type
        left, right, is_float = visitor.type_converter.unify_binop_types(left, right)
        
        # Choose pow intrinsic by PC hint: floats use f64 pow, integers promoted already
        float_llvm = ir.DoubleType()
        result = visitor.builder.call(
            visitor._get_pow_intrinsic(float_llvm),
            [ensure_ir(left), ensure_ir(right)]
        )
        return wrap_value(result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_lshift(cls, visitor, left, right, node: ast.BinOp):
        """Handle left shift for integer types"""
        from ..valueref import wrap_value, ensure_ir
        left, right, _ = visitor.type_converter.unify_binop_types(left, right)
        
        result = visitor.builder.shl(ensure_ir(left), ensure_ir(right))
        return wrap_value(result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_rshift(cls, visitor, left, right, node: ast.BinOp):
        """Handle right shift for integer types.
        
        Uses arithmetic shift (ashr) for signed types (preserves sign bit),
        and logical shift (lshr) for unsigned types (fills with zeros).
        """
        from ..valueref import wrap_value, ensure_ir
        left, right, _ = visitor.type_converter.unify_binop_types(left, right)
        
        # Check if the type is signed or unsigned
        type_hint = left.type_hint
        is_signed = True  # default to signed
        if type_hint is not None and hasattr(type_hint, '_is_signed'):
            is_signed = type_hint._is_signed
        
        if is_signed:
            result = visitor.builder.ashr(ensure_ir(left), ensure_ir(right))
        else:
            result = visitor.builder.lshr(ensure_ir(left), ensure_ir(right))
        return wrap_value(result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_bitor(cls, visitor, left, right, node: ast.BinOp):
        """Handle bitwise or for integer types"""
        from ..valueref import wrap_value, ensure_ir
        left, right, _ = visitor.type_converter.unify_binop_types(left, right)
        
        result = visitor.builder.or_(ensure_ir(left), ensure_ir(right))
        return wrap_value(result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_bitxor(cls, visitor, left, right, node: ast.BinOp):
        """Handle bitwise xor for integer types"""
        from ..valueref import wrap_value, ensure_ir
        left, right, _ = visitor.type_converter.unify_binop_types(left, right)
        
        result = visitor.builder.xor(ensure_ir(left), ensure_ir(right))
        return wrap_value(result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_bitand(cls, visitor, left, right, node: ast.BinOp):
        """Handle bitwise and for integer types"""
        from ..valueref import wrap_value, ensure_ir
        left, right, _ = visitor.type_converter.unify_binop_types(left, right)
        
        result = visitor.builder.and_(ensure_ir(left), ensure_ir(right))
        return wrap_value(result, kind="value", type_hint=left.type_hint)


class BuiltinFunction(BuiltinEntity):
    """Base class for built-in function entities"""
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True
