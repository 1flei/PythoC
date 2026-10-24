from llvmlite import ir
from typing import List, Optional, Any
from .composite_base import CompositeType
from ..valueref import extract_constant_index
from ..logger import logger


class UnionType(CompositeType):
    """Union type base class - all fields share the same memory location
    
    Supports subscript syntax: union[i32, f64] or union[x: i32, y: f64]
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'union'
    
    @classmethod
    def get_type_id(cls, _visited=None) -> str:
        """Generate unique type ID for union types"""
        suffix = cls.get_type_id_suffix(_visited)
        return f'U{suffix}'
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get backing storage matching the semantic union layout."""
        if cls._field_types is None:
            logger.error("union type requires field types", node=None, exc_type=TypeError)

        cls._ensure_field_types_resolved()

        from ..target_layout import get_target_layout
        layout = cls.get_layout()
        target_layout = get_target_layout()
        carrier = target_layout.alignment_carrier(layout.alignment)
        carrier_layout = target_layout.primitive(carrier)
        return ir.ArrayType(
            carrier,
            layout.allocation_size // carrier_layout.allocation_size,
        )
    
    @classmethod
    def get_ctypes_type(cls):
        """Get a ctypes union with the same member-driven alignment."""
        import ctypes

        cached = getattr(cls, '_ctypes_union_cache', None)
        if cached is not None:
            return cached

        from ..target_layout import NATURAL_LAYOUT
        if not getattr(cls, '_layout_config', NATURAL_LAYOUT).is_natural:
            from .bitfield_access import get_bit_storage_ctypes
            storage = get_bit_storage_ctypes(cls.get_layout(), 'CBitUnion')
            cls._ctypes_union_cache = storage
            return storage

        cls._ensure_field_types_resolved()
        fields = []
        for index, field_type in enumerate(cls._field_types):
            field_ctype = field_type.get_ctypes_type()
            if field_ctype is None:
                continue
            field_name = (
                cls._field_names[index] if cls._field_names is not None
                else None
            )
            if not field_name:
                field_name = f"field_{index}"
            fields.append((field_name, field_ctype))

        if not fields:
            return None

        union_class = type(
            f"CUnion_{cls.get_name()}",
            (ctypes.Union,),
            {'_fields_': fields},
        )
        from ..target_layout import validate_ctypes_layout
        validate_ctypes_layout(cls, union_class)
        cls._ctypes_union_cache = union_class
        return union_class
    
    @classmethod
    def get_layout(cls):
        if cls._field_types is None:
            from ..target_layout import TypeLayout
            return TypeLayout.empty()

        cls._ensure_field_types_resolved()
        from ..target_layout import get_target_layout, get_type_layout
        from ..target_layout import NATURAL_LAYOUT, bit_aggregate_layout
        layout_config = getattr(cls, '_layout_config', NATURAL_LAYOUT)
        if not layout_config.is_natural:
            return bit_aggregate_layout(
                cls._field_names, cls._field_types, layout_config,
                is_union=True,
            )
        fields = (get_type_layout(field) for field in cls._field_types)
        return get_target_layout().union(fields)

    @classmethod
    def get_size_bytes(cls) -> int:
        return cls.get_layout().allocation_size

    @classmethod
    def get_alignment(cls) -> int:
        return cls.get_layout().alignment
    
    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
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
            logger.error(f"union index {index_val} out of range (0-{len(cls._field_types)-1})",
                        node=node, exc_type=IndexError)
        
        field_type = cls._field_types[index_val]

        from ..ir_helpers import propagate_qualifiers
        field_type = propagate_qualifiers(base.type_hint, field_type)

        from ..target_layout import NATURAL_LAYOUT
        layout_config = getattr(cls, "_layout_config", NATURAL_LAYOUT)
        if not layout_config.is_natural:
            from .bitfield_access import access_bit_field
            return access_bit_field(
                visitor,
                base,
                cls,
                index_val,
                field_type,
            )
        
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
            logger.error(f"union field type {field_type} has no get_llvm_type method",
                        node=node, exc_type=TypeError)
        
        # A union field may be ``void`` (e.g. the payload of an enum variant with
        # no data). LLVM cannot represent a pointer-to-void, so return a dummy
        # address value instead of loading.
        from llvmlite.ir.types import VoidType
        if isinstance(field_llvm_type, VoidType):
            return wrap_value(union_ptr, kind="address", type_hint=field_type,
                              address=union_ptr)

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
            # Class-level access may resolve to a class static member before
            # falling back to compiled class methods.
            is_class_level = (
                base.is_python_value()
                and getattr(base.value, '_union_type', None) is cls
            )
            if is_class_level:
                from ..decorators.class_statics import (
                    lookup_class_static, get_or_create_static_global,
                )
                if lookup_class_static(cls, attr_name) is not None:
                    return get_or_create_static_global(visitor, cls, attr_name, node)
            # Fall back to class-level method lookup (compiled wrappers
            # attached by @compile/class-method support).
            from ..decorators.class_methods import (
                lookup_class_method, wrap_class_method_as_python_value,
            )
            method = lookup_class_method(cls, attr_name)
            if method is not None:
                return wrap_class_method_as_python_value(method, node)
            logger.error(f"union has no field named '{attr_name}'", node=node, exc_type=AttributeError)
        
        field_index = cls.get_field_index(attr_name)
        field_type = cls._field_types[field_index]
        
        # Propagate qualifiers from union type to field type
        # If we have const[union], accessing field should give const[field_type]
        if base.type_hint:
            field_type = propagate_qualifiers(base.type_hint, field_type)

        from ..target_layout import NATURAL_LAYOUT
        layout_config = getattr(cls, "_layout_config", NATURAL_LAYOUT)
        if not layout_config.is_natural:
            from .bitfield_access import access_bit_field
            return access_bit_field(
                visitor,
                base,
                cls,
                field_index,
                field_type,
            )
        
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
            logger.error(f"union field type {field_type} has no get_llvm_type method",
                        node=node, exc_type=TypeError)
        
        field_ptr = visitor.builder.bitcast(union_ptr, ir.PointerType(field_llvm_type))
        
        # Load value and return with address
        loaded_value = visitor.builder.load(field_ptr)
        return wrap_value(loaded_value, kind="address", type_hint=field_type, address=field_ptr)


def create_union_type(
    field_types: List[Any],
    field_names: Optional[List[str]] = None,
    python_class: Optional[type] = None,
    layout_config=None,
) -> type:
    """Create a union type without global caching
    
    Args:
        field_types: List of field types
        field_names: Optional list of field names
        python_class: Optional Python class (for @union decorated classes)
    
    Returns:
        UnionType subclass
    """
    from ..target_layout import NATURAL_LAYOUT
    layout_config = layout_config or NATURAL_LAYOUT

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
            '_pc_specialized': True,
            '_canonical_name': type_name,
            '_field_types': field_types,
            '_field_names': field_names,
            '_python_class': python_class,
            '_layout_config': layout_config,
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
    
    def __new__(cls, target=None, suffix=None, **kwargs):
        """Support @union decorator syntax with parameters
        
        Uses the common decorator pattern from CompositeType base class.
        """
        # Delegate to base class factory method
        return cls.create_decorator_instance(target, suffix, **kwargs)
    
    @classmethod
    def _apply_decorator(cls, target_cls, suffix=None, **kwargs):
        """Apply union decorator to target class
        
        Uses common field parsing from CompositeType base class.
        
        Args:
            target_cls: The class being decorated
            suffix: Optional suffix for compilation
            **kwargs: Additional parameters
        
        Returns:
            Decorated class with union behavior
        """
        # Capture user-frame symbols before doing anything pythoc-internal so
        # class-body methods can resolve closure names (factory locals etc.).
        from ..decorators.visible import capture_user_caller_symbols
        captured_symbols = capture_user_caller_symbols()

        # Mark as union
        target_cls._is_union = True
        target_cls._is_struct = False
        
        # Use common field parsing logic
        parsed = cls._parse_class_fields(target_cls)

        from ..target_layout import make_aggregate_layout_config
        layout_config = make_aggregate_layout_config(**kwargs)
        
        # Create union type
        unified_type = create_union_type(
            parsed['field_types'],
            parsed['field_names'],
            python_class=target_cls,
            layout_config=layout_config,
        )
        
        # Setup forward reference handling if needed
        if parsed['needs_type_resolution']:
            unified_type._needs_type_resolution = True
            unified_type._type_namespace = parsed['type_namespace']
            cls._setup_forward_ref_callbacks(
                unified_type, target_cls,
                parsed['field_types'], parsed['type_namespace']
            )
        
        # Link class to type and register
        cls._link_class_to_type(target_cls, unified_type, suffix)

        # Compile and attach plain-def methods declared in the union body.
        from ..decorators.class_methods import attach_class_methods
        attach_class_methods(
            target_cls,
            class_compile_suffix=suffix,
            captured_symbols=captured_symbols,
        )

        return target_cls
    
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
            logger.error("union requires at least one field", node=None, exc_type=TypeError)
        
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
