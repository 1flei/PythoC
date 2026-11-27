import sys

from llvmlite import ir
from typing import List, Optional, Any
from .composite_base import CompositeType
from ..valueref import extract_constant_index
from ..logger import logger
from ..source import SourceLocation


class UnionType(CompositeType):
    """Union type base class - all fields share the same memory location
    
    Supports subscript syntax: union[i32, f64] or union[x: i32, y: f64]
    """
    _is_union = True
    
    @classmethod
    def map_type_components(cls, mapper):
        if getattr(cls, '_python_class', None) is not None or cls._field_types is None:
            return cls
        fields = [mapper(field) for field in cls._field_types]
        if all(a is b for a, b in zip(fields, cls._field_types)):
            return cls
        return create_union_type(fields, cls._field_names,
                                 layout_config=cls._layout_config)

    @classmethod
    def get_name(cls) -> str:
        return 'union'
    
    @classmethod
    def get_type_id(cls, _visited=None) -> str:
        from ..type_id import get_type_id
        return get_type_id(cls)
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get backing storage matching the semantic union layout."""
        from ..type_graph import prepare_type
        prepare_type(cls)
        if cls._field_types is None:
            logger.error("union type requires field types", node=None, exc_type=TypeError)

        cls._ensure_field_types_resolved()

        from ..builder.record_types import union_storage
        from ..target_layout import uses_bit_storage

        node = cls._layout_diagnostic_node(sys._getframe(1))
        return union_storage(cls.get_layout(), uses_bit_storage(cls), node)
    
    @classmethod
    def get_ctypes_type(cls):
        """Get a ctypes union with the same member-driven alignment."""
        import ctypes
        from ..type_graph import prepare_type
        prepare_type(cls)

        from ..session import CompileSession
        cache = CompileSession.current().type_graph.ctypes_types
        if cls in cache:
            return cache[cls]

        from ..target_layout import align_ctypes_record, uses_bit_storage, validate_ctypes_layout
        from .bitfield_access import get_bit_storage_ctypes

        layout = cls.get_layout()
        node = cls._layout_diagnostic_node(sys._getframe(1))
        if uses_bit_storage(cls):
            storage = get_bit_storage_ctypes(layout, 'CBitUnion', node)
            cache[cls] = storage
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

        class_name = f'CUnion_{cls.get_name()}'
        attributes = {'_fields_': fields}
        if cls._layout_config.alignment is not None:
            attributes['_layout_'] = 'ms'
            attributes['_pack_'] = cls._layout_config.alignment // 8
        if cls._layout_config.abi_alignment is not None:
            attributes['_align_'] = layout.alignment
        union_class = type(class_name, (ctypes.Union,), attributes)
        union_class = align_ctypes_record(union_class, layout.alignment, node)
        validate_ctypes_layout(cls, union_class, node)
        cache[cls] = union_class
        return union_class
    
    @classmethod
    def get_layout(cls):
        if cls._field_types is None:
            from ..target_layout import TypeLayout
            return TypeLayout.empty()

        from ..type_graph import prepare_type
        prepare_type(cls)
        cls._ensure_field_types_resolved()
        from ..target_layout import aggregate_layout
        return aggregate_layout(
            cls._field_names, cls._field_types, cls._layout_config,
            cls._layout_diagnostic_node(sys._getframe(1)), is_union=True,
        )

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
        """Project a union alternative by declaration index."""
        cls._ensure_field_types_resolved()
        index = extract_constant_index(index, node, 'union subscript')
        if not 0 <= index < len(cls._field_types):
            logger.error(f'union index {index} out of range (0-{len(cls._field_types)-1})',
                         node=node, exc_type=IndexError)
        return cls._project_field(visitor, base, index)

    @classmethod
    def handle_attribute(cls, visitor, base, attr_name, node):
        """Handle attribute access on union instances: u.x
        
        All fields share the same memory location, so we bitcast to the field type.
        """
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
        
        return cls._project_field(visitor, base, cls._get_field_index(attr_name, node))

    @classmethod
    def _project_field(cls, visitor, base, index):
        from .types import void
        from ..ir_helpers import propagate_qualifiers
        from ..target_layout import uses_bit_storage
        from ..type_converter import strip_qualifiers
        from ..valueref import wrap_value

        field_type = propagate_qualifiers(base.type_hint, cls._field_types[index])
        if strip_qualifiers(field_type) is void:
            return wrap_value(None, kind='python', type_hint=void)
        if uses_bit_storage(cls):
            from .bitfield_access import access_bit_field
            return access_bit_field(visitor, base, cls, index, field_type)

        address = cls._get_value_address(visitor, base)
        field_llvm = field_type.get_llvm_type(visitor.module.context)
        address = visitor.builder.bitcast(address, ir.PointerType(field_llvm))
        return wrap_value(address, kind='address', type_hint=field_type, address=address)


def create_union_type(
    field_types: List[Any],
    field_names: Optional[List[str]] = None,
    python_class: Optional[type] = None,
    layout_config=None,
) -> type:
    """Build a union from Python type metadata."""
    node = logger.current_location() or SourceLocation.from_frame(sys._getframe(1))
    return _create_union_type(field_types, node, field_names, python_class, layout_config)


def _create_union_type(
    field_types: List[Any],
    node,
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
    from ..type_descriptor import validate_member_names, validate_type_argument
    validate_member_names(field_names, 'union', node=node)
    for field in field_types:
        validate_type_argument(field, 'union field', allow_incomplete=True, node=node)
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
    
    def __new__(cls, target=None, suffix=None, **kwargs):
        """Support @union decorator syntax with parameters
        
        Uses the common decorator pattern from CompositeType base class.
        """
        return cls.create_decorator_instance(target, suffix, kwargs)
    
    @classmethod
    def _apply_decorator(cls, target_cls, suffix, layout_config):
        """Apply union decorator to target class
        
        Uses common field parsing from CompositeType base class.
        
        Args:
            target_cls: The class being decorated
            suffix: Optional suffix for compilation
            layout_config: Normalized layout options
        
        Returns:
            Decorated class with union behavior
        """
        # Capture user-frame symbols before doing anything pythoc-internal so
        # class-body methods can resolve closure names (factory locals etc.).
        from ..decorators.visible import capture_user_caller_symbols
        captured_symbols = capture_user_caller_symbols()
        from ..decorators.class_methods import _read_class_source, attach_class_methods
        from ..source import class_annotation_nodes
        class_ast, source = _read_class_source(target_cls)

        # Mark as union
        target_cls._is_union = True
        target_cls._is_struct = False
        
        # Use common field parsing logic
        parsed = cls._parse_class_fields(target_cls, class_annotation_nodes(class_ast), source)

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
        
        # Link class to type and register
        cls._link_class_to_type(target_cls, unified_type, suffix)

        # Compile and attach plain-def methods declared in the union body.
        attach_class_methods(
            target_cls,
            class_compile_suffix=suffix,
            captured_symbols=captured_symbols,
            class_body=class_ast.body if class_ast is not None else [],
        )

        return target_cls
    
    @classmethod
    def get_name(cls) -> str:
        return 'union'
    
    @classmethod
    def __class_getitem__(cls, item):
        """Support union[...] subscript syntax"""
        normalized = cls.normalize_subscript_items(item, node=None)
        return cls.handle_type_subscript(normalized, node=None)
    
    @classmethod
    def handle_type_subscript(cls, items, node):
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
            logger.error("union requires at least one field", node=node, exc_type=TypeError)
        
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
        
        return _create_union_type(field_types, node, field_names)


__all__ = ['union', 'UnionType', 'create_union_type']
