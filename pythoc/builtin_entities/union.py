from llvmlite import ir
from typing import List, Optional, Any
from .composite_base import CompositeType
from ..valueref import extract_constant_index


class UnionType(CompositeType):
    """Union type base class - all fields share the same memory location
    
    Supports subscript syntax: union[i32, f64] or union[x: i32, y: f64]
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'union'
    
    @classmethod
    def get_type_id(cls) -> str:
        """Generate unique type ID for union types"""
        suffix = cls.get_type_id_suffix()
        return f'U{suffix}'
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get LLVM type for union (array of bytes)
        
        All fields share the same memory, so union is represented
        as a byte array with size equal to the largest field.
        
        Args:
            module_context: Optional IR module context (not used)
        """
        if cls._field_types is None:
            raise TypeError("union type requires field types")
        
        # Ensure field types are resolved
        cls._ensure_field_types_resolved()
        
        max_size = 0
        max_align = 1
        for field_type in cls._field_types:
            if hasattr(field_type, 'get_size_bytes'):
                field_size = field_type.get_size_bytes()
                # Skip void types (size 0)
                if field_size == 0:
                    continue
                field_align = min(field_size, 8)
            else:
                raise TypeError(f"union.get_llvm_type: unknown field type {field_type}")
            max_size = max(max_size, field_size)
            max_align = max(max_align, field_align)
        
        # Align size to max alignment
        if max_size == 0:
            max_size = 0
        elif max_size % max_align != 0:
            max_size += max_align - (max_size % max_align)
        
        # Return array directly, not wrapped in a struct
        return ir.ArrayType(ir.IntType(8), max_size)
    
    @classmethod
    def get_size_bytes(cls) -> int:
        """Get size in bytes (with alignment)"""
        if cls._field_types is None:
            return 0
        
        # Ensure field types are resolved
        cls._ensure_field_types_resolved()
        
        max_size = 0
        max_align = 1
        for field_type in cls._field_types:
            if hasattr(field_type, 'get_size_bytes'):
                field_size = field_type.get_size_bytes()
                if field_size == 0:
                    continue
                field_align = min(field_size, 8)
            else:
                raise TypeError(f"union.get_size_bytes: unknown field type {field_type}")
            max_size = max(max_size, field_size)
            max_align = max(max_align, field_align)
        
        # Align size
        if max_size == 0:
            return 0
        if max_size % max_align != 0:
            max_size += max_align - (max_size % max_align)
        return max_size
    
    @classmethod
    def handle_call(cls, visitor, args, node):
        """Handle union construction: union[i32, f64]()
        
        Creates an uninitialized union value (ir.Undefined).
        """
        return cls._create_undef_constructor(visitor, args, node)
    
    @classmethod
    def handle_subscript(cls, visitor, base, index, node):
        """Handle value subscript access on union instances: u[0]
        
        Note: Type subscripts (union[i32, f64]) are now handled by PythonType.handle_subscript
        which extracts items and calls union.handle_type_subscript directly.
        This method only handles value subscripts.
        """
        import ast
        from ..valueref import wrap_value, ensure_ir, get_type
        
        # VALUE SUBSCRIPT: u[0] - field access by index
        # Extract constant index from pre-evaluated index ValueRef
        index_val = extract_constant_index(index, "union subscript")
        
        if index_val < 0 or index_val >= len(cls._field_types):
            raise IndexError(f"union index {index_val} out of range (0-{len(cls._field_types)-1})")
        
        field_type = cls._field_types[index_val]
        
        # Get the address of the union
        union_ptr = cls._get_value_address(visitor, base)
        
        # For union, all fields share the same memory location
        # Bitcast the union pointer to the field type pointer
        if hasattr(field_type, 'get_llvm_type'):
            try:
                module_context = visitor.module.context if hasattr(visitor, 'module') else None
                field_llvm_type = field_type.get_llvm_type(module_context)
            except TypeError:
                field_llvm_type = field_type.get_llvm_type()
        else:
            raise TypeError(f"union field type {field_type} has no get_llvm_type method")
        
        # Bitcast union pointer to field type pointer
        field_ptr = visitor.builder.bitcast(union_ptr, ir.PointerType(field_llvm_type))
        
        # Load value and return with address for lvalue support
        loaded_value = visitor.builder.load(field_ptr)
        return wrap_value(loaded_value, kind="address", type_hint=field_type, address=field_ptr)
    
    @classmethod
    def handle_attribute(cls, visitor, base, attr_name, node):
        """Handle attribute access on union instances: u.x
        
        All fields share the same memory location, so we bitcast to the field type.
        """
        from ..valueref import wrap_value, ensure_ir, get_type
        from ..ir_helpers import propagate_qualifiers
        
        # Ensure field types are resolved
        cls._ensure_field_types_resolved()
        
        # Check if union has this field
        if not cls.has_field(attr_name):
            raise AttributeError(f"union has no field named '{attr_name}'")
        
        field_index = cls.get_field_index(attr_name)
        field_type = cls._field_types[field_index]
        
        # Propagate qualifiers from union type to field type
        # If we have const[union], accessing field should give const[field_type]
        if base.type_hint:
            field_type = propagate_qualifiers(base.type_hint, field_type)
        
        # Get the address of the union
        union_ptr = cls._get_value_address(visitor, base)
        
        # Bitcast to field type pointer
        if hasattr(field_type, 'get_llvm_type'):
            try:
                module_context = visitor.module.context if hasattr(visitor, 'module') else None
                field_llvm_type = field_type.get_llvm_type(module_context)
            except TypeError:
                field_llvm_type = field_type.get_llvm_type()
        else:
            raise TypeError(f"union field type {field_type} has no get_llvm_type method")
        
        field_ptr = visitor.builder.bitcast(union_ptr, ir.PointerType(field_llvm_type))
        
        # Load value and return with address
        loaded_value = visitor.builder.load(field_ptr)
        return wrap_value(loaded_value, kind="address", type_hint=field_type, address=field_ptr)


def create_union_type(field_types: List[Any], field_names: Optional[List[str]] = None) -> type:
    """Create a union type without global caching
    
    Args:
        field_types: List of field types
        field_names: Optional list of field names
    
    Returns:
        UnionType subclass
    """
    # Build type name
    from ..type_id import get_type_id
    type_names = []
    for t in field_types:
        if hasattr(t, 'get_name'):
            type_names.append(t.get_name())
        elif hasattr(t, '__name__'):
            type_names.append(t.__name__)
        else:
            type_names.append(str(t))
    
    type_name = f'union[{", ".join(type_names)}]'
    
    # Create new union type
    union_type = type(
        type_name,
        (UnionType,),
        {
            '_field_types': field_types,
            '_field_names': field_names,
        }
    )
    
    return union_type


class union(UnionType):
    """Union type factory - supports both decorator and subscript syntax
    
    This is the user-facing union type that supports:
    - union[i32, f64] - unnamed fields (subscript syntax)
    - union[x: i32, y: f64] - named fields (subscript syntax)
    - @union - decorator for classes (delegates to @compile)
    - @union(suffix=...) - decorator with suffix parameter
    - @union(anonymous=True) - decorator with anonymous naming
    """
    
    def __init__(self, target=None, suffix=None, anonymous=False):
        """Initialize union decorator with optional parameters"""
        self.target = target
        self.suffix = suffix
        self.anonymous = anonymous
    
    def __call__(self, cls):
        """Decorator application"""
        return self._apply_decorator(cls, self.suffix, self.anonymous)
    
    def __new__(cls, target=None, suffix=None, anonymous=False):
        """Support @union decorator syntax with parameters
        
        Uses the common decorator pattern from CompositeType base class.
        """
        # Delegate to base class factory method
        return cls.create_decorator_instance(target, suffix, anonymous)
    
    @classmethod
    def get_name(cls) -> str:
        return 'union'
    
    @classmethod
    def __class_getitem__(cls, item):
        """Support union[...] subscript syntax"""
        normalized = cls.normalize_subscript_items(item)
        return cls.handle_type_subscript(normalized)
    
    @classmethod
    def handle_type_subscript(cls, items):
        """Handle type subscript with normalized items
        
        Items are normalized to Tuple[Tuple[Optional[str], type], ...]
        
        Args:
            items: Normalized tuple of (Optional[str], type)
        
        Returns:
            UnionType subclass
        """
        import builtins
        
        if not isinstance(items, builtins.tuple):
            items = (items,)
        
        if len(items) == 0:
            raise TypeError("union requires at least one field")
        
        field_types = []
        field_names = []
        has_any_name = False
        
        for name, field_type in items:
            field_types.append(field_type)
            field_names.append(name)
            if name is not None:
                has_any_name = True
        
        if not has_any_name:
            field_names = None
        
        return create_union_type(field_types, field_names)


__all__ = ['union', 'UnionType', 'create_union_type']
