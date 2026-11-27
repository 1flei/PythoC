"""
Unified Struct Type Implementation

This module provides a unified struct type that merges:
1. Anonymous struct: struct[i32, i32] or struct[x: i32, y: i32]
2. @compile class decorated structs

Design principles:
- Structural typing: same field types + field names -> same type
- No global type cache (each struct[...] creates new type for simplicity)
- Forward reference support via lazy type resolution
- No methods for anonymous structs (methods only for @compile classes)
"""

import sys

from llvmlite import ir
from ..logger import logger
from ..source import SourceLocation
from typing import List, Tuple, Optional, Any
from .composite_base import CompositeType
from .base import BuiltinEntityMeta
from ..valueref import extract_constant_index


class StructTypeMeta(BuiltinEntityMeta):
    """Metaclass for StructType that enables Python-compatible interfaces.
    
    This allows struct types to support:
    - len(struct_type) -> number of fields
    - for x in struct_type -> iterate over stored elements (if available)
    - iter(struct_type) -> iterator over elements
    
    These work at compile-time for type-level operations.
    """
    
    def __len__(cls):
        """Return number of fields in the struct.
        
        Enables len(struct_type) at compile-time.
        """
        cls._ensure_field_types_resolved()
        return len(cls._field_types) if cls._field_types else 0
    
    def __iter__(cls):
        """Iterate over stored elements if available.
        
        For struct types created from tuple conversion (with _elements),
        this iterates over the actual ValueRef elements.
        For plain struct types, this iterates over field types.
        
        This enables:
        - for x in struct_from_tuple: ... (iterate over values)
        - for t in struct_type: ... (iterate over field types)
        """
        # If struct has stored elements (from tuple conversion), iterate over them
        if hasattr(cls, '_elements') and cls._elements is not None:
            return iter(cls._elements)
        
        # Otherwise iterate over field types
        cls._ensure_field_types_resolved()
        return iter(cls._field_types) if cls._field_types else iter([])


class StructType(CompositeType, metaclass=StructTypeMeta):
    """Unified struct type with structural typing semantics
    
    This class represents both anonymous structs and @compile class structs.
    Type equality is determined by field structure (types + names), not by
    the original class name.
    
    Attributes:
        _canonical_name: Unique name based on field structure
        _field_types: List of field types (can be strings for lazy resolution)
        _field_names: List of field names (None for unnamed fields)
        _python_class: Original Python class (for @compile decorated classes)
        _struct_info: Reference to StructInfo in registry
        _needs_type_resolution: Whether field types need lazy resolution
        _field_types_resolved: Whether lazy resolution has been performed
    """
    
    _canonical_name: str = None
    _python_class: Optional[type] = None
    _struct_info: Optional[Any] = None  # StructInfo from registry
    # Force the IdentifiedStructType path even without a backing @compile class.
    # A self-referential struct (a field that forward-references the struct's
    # own name, e.g. ptr["Node"]) cannot be expressed as a literal (structural)
    # LLVM type, so a named identified type is required to break the cycle.
    # Programmatically built struct types set this together with _canonical_name
    # to name the identified type.
    _pc_record = True
    _force_identified: bool = False
    
    @classmethod
    def map_type_components(cls, mapper):
        if cls._python_class is not None or cls._force_identified or cls._field_types is None:
            return cls
        fields = [mapper(field) for field in cls._field_types]
        if all(a is b for a, b in zip(fields, cls._field_types)):
            return cls
        return create_struct_type(fields, cls._field_names, elements=cls._elements,
                                  layout_config=cls._layout_config)

    @classmethod
    def get_name(cls) -> str:
        """Return struct name"""
        if cls._canonical_name:
            return cls._canonical_name
        return 'struct'
    
    @classmethod
    def is_struct_type(cls) -> bool:
        """Struct types return True."""
        return True
    
    @classmethod
    def get_ctypes_type(cls):
        """Get ctypes Structure type for FFI.
        
        Creates a ctypes.Structure class dynamically based on field types.
        Uses caching to avoid recreating the same struct type.
        """
        import ctypes
        from ..type_graph import prepare_type
        prepare_type(cls)
        from ..type_graph import validate_type_graph
        validate_type_graph(cls)
        
        from ..session import CompileSession
        cache = CompileSession.current().type_graph.ctypes_types
        if cls in cache:
            return cache[cls]
        
        cls._ensure_field_types_resolved()

        from ..target_layout import align_ctypes_record, uses_bit_storage, validate_ctypes_layout
        from .bitfield_access import get_bit_storage_ctypes

        layout = cls.get_layout()
        node = cls._layout_diagnostic_node(sys._getframe(1))
        if uses_bit_storage(cls):
            storage = get_bit_storage_ctypes(
                layout,
                f"CStruct_{cls.get_name()}",
                node,
            )
            cache[cls] = storage
            return storage
        
        # Build fields list
        fields = []
        field_indices = []
        for i, field_type in enumerate(cls._field_types):
            if field_type is None:
                continue
            
            # Get ctypes type for field
            if hasattr(field_type, 'get_ctypes_type'):
                field_ctype = field_type.get_ctypes_type()
            else:
                # Skip types without ctypes support (e.g., zero-sized types)
                continue
            
            if field_ctype is None:
                continue
            
            # Use field name if available
            field_name = cls._field_names[i] if cls._field_names and i < len(cls._field_names) else None
            if field_name is None:
                field_name = f"field_{i}"
            
            fields.append((field_name, field_ctype))
            field_indices.append((i, field_name))
        
        # Create ctypes.Structure class dynamically
        class_name = f"CStruct_{cls.get_name()}"
        attributes = {'_fields_': fields}
        if cls._layout_config.member_alignments:
            attributes['_fields_'] = _explicit_ctypes_fields(fields, field_indices, layout)
            attributes['_layout_'] = 'ms'
            attributes['_pack_'] = 1
        elif cls._layout_config.alignment is not None:
            attributes['_layout_'] = 'ms'
            attributes['_pack_'] = cls._layout_config.alignment // 8
        if cls._layout_config.abi_alignment is not None:
            attributes['_align_'] = layout.alignment
        struct_class = type(class_name, (ctypes.Structure,), attributes)
        struct_class = align_ctypes_record(struct_class, layout.alignment, node)
        validate_ctypes_layout(cls, struct_class, node, field_indices)
        
        cache[cls] = struct_class
        return struct_class
    
    @classmethod
    def _is_compile_class(cls) -> bool:
        """Check if this struct is from @compile decorated class"""
        return cls._python_class is not None
    
    @classmethod
    def get_llvm_type(cls, module_context=None, node=None) -> ir.Type:
        """Get LLVM struct type
        
        New Strategy:
        - @compile class -> IdentifiedStructType (named type with identity)
        - Anonymous struct[...] -> LiteralStructType (structural type)
        - Field names affect PC type identity but not LLVM type structure
        
        Args:
            module_context: Optional IR module context for IdentifiedStructType
            node: Optional AST node for error reporting
        
        Returns:
            LiteralStructType or IdentifiedStructType
        """
        node = node or cls._layout_diagnostic_node(sys._getframe(1))
        if cls._field_types is None:
            logger.error("struct type requires field types", node=node, exc_type=TypeError)

        from ..target_layout import uses_bit_storage
        if uses_bit_storage(cls):
            from .bitfield_access import get_bit_storage_type
            return get_bit_storage_type(cls.get_layout(), node)

        from ..builder.record_types import materialize_record, record_storage
        from ..type_graph import prepare_type, record_definition, validate_type_graph

        if cls._is_compile_class() or cls._force_identified:
            definition = record_definition(cls)
            if module_context is None:
                logger.error(f"struct '{cls.get_name()}' requires module_context",
                             node=node, exc_type=TypeError)
            return materialize_record(definition, module_context, node)

        prepare_type(cls)
        validate_type_graph(cls)
        fields = tuple(field.get_llvm_type(module_context) for field in cls._field_types)
        storage = record_storage(fields, cls, node)
        record = ir.LiteralStructType(storage.elements, packed=storage.packed)
        record._pythoc_field_paths = storage.field_paths
        return record

    @classmethod
    def get_layout(cls):
        if cls._field_types is not None:
            from ..type_graph import prepare_type
            prepare_type(cls)
        from ..type_graph import validate_type_graph
        validate_type_graph(cls)
        if cls._field_types is None:
            from ..target_layout import TypeLayout
            return TypeLayout.empty()

        from ..target_layout import aggregate_layout
        return aggregate_layout(
            cls._field_names, cls._field_types, cls._layout_config,
            cls._layout_diagnostic_node(sys._getframe(1)),
        )

    @classmethod
    def get_size_bytes(cls) -> int:
        return cls.get_layout().allocation_size

    @classmethod
    def get_alignment(cls) -> int:
        return cls.get_layout().alignment

    
    @classmethod
    def get_type_id(cls, _visited=None) -> str:
        from ..type_id import get_type_id
        return get_type_id(cls)
    
    @classmethod
    def handle_type_call(cls, visitor, func_ref, args, node):
        """Handle struct construction: Point() or struct[i32, i32]()
        
        Creates an uninitialized struct value (ir.Undefined).
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the callable (the struct type itself)
            args: Pre-evaluated arguments (should be empty for default constructor)
            node: Original ast.Call node
        """
        if args:
            logger.error(f"{cls.get_name()}() takes no arguments ({len(args)} given). "
                         "Use a typed initializer to supply field values.",
                         node=node, exc_type=TypeError)
        from ..type_descriptor import contains_refinement_storage
        if contains_refinement_storage(cls):
            logger.error('Cannot default-construct an aggregate with proof-bearing fields; '
                         'provide an explicit initializer', node=node, exc_type=TypeError)
        from ..valueref import wrap_value
        return wrap_value(ir.Constant(cls.get_llvm_type(visitor.module.context), ir.Undefined),
                          kind='value', type_hint=cls)
    
    @classmethod
    def handle_attribute(cls, visitor, base, attr_name, node):
        """Handle attribute access on struct instances: point.x
        
        Supports both named field access and methods (for @compile classes).
        
        Args:
            visitor: AST visitor
            base: Pre-evaluated base object (ValueRef)
            attr_name: Attribute name (string)
            node: Original ast.Attribute node
        """
        # Ensure field types are resolved
        cls._ensure_field_types_resolved()

        from ..decorators.class_methods import (
            lookup_class_method, wrap_class_method_as_python_value,
        )

        # Instance-level and class-level attribute access have distinct
        # namespaces.  An instance access (``v.size``) always resolves to a
        # struct field; a class-level access (``Vec.size``) always resolves to
        # a class attribute such as a compiled method.  They are not mixed
        # through fallback rules.
        is_class_level = (
            base.is_python_value()
            and getattr(base.value, '_struct_type', None) is cls
        )

        if is_class_level:
            from ..decorators.class_statics import (
                lookup_class_static, get_or_create_static_global,
            )
            if lookup_class_static(cls, attr_name) is not None:
                return get_or_create_static_global(visitor, cls, attr_name, node)
            method = lookup_class_method(cls, attr_name)
            if method is not None:
                return wrap_class_method_as_python_value(method, node)
            logger.error(
                f"struct '{cls.get_name()}' has no class attribute '{attr_name}'",
                node=node, exc_type=AttributeError,
            )

        if not cls.has_field(attr_name):
            logger.error(f"struct '{cls.get_name()}' has no field named '{attr_name}'",
                        node=node, exc_type=AttributeError)

        return cls._project_field(visitor, base, cls._get_field_index(attr_name, node), node)

    @classmethod
    def handle_subscript(cls, visitor, base, index, node):
        """Project a field by its declaration index."""
        cls._ensure_field_types_resolved()
        index = extract_constant_index(index, node, 'struct subscript')
        if not 0 <= index < len(cls._field_types):
            logger.error(f'struct index {index} out of range (0-{len(cls._field_types)-1})',
                         node=node, exc_type=IndexError)
        return cls._project_field(visitor, base, index, node)

    @classmethod
    def _project_field(cls, visitor, base, index, node):
        from ..ir_helpers import propagate_qualifiers
        from ..literal_protocol import project_tracking_metadata
        from ..target_layout import uses_bit_storage
        from ..valueref import ensure_ir, wrap_value

        field_type = propagate_qualifiers(base.type_hint, cls._field_types[index])
        name, linear_path = project_tracking_metadata(base, index)
        if uses_bit_storage(cls):
            from .bitfield_access import access_bit_field
            return access_bit_field(visitor, base, cls, index, field_type,
                                    var_name=name, linear_path=linear_path)

        address = ensure_ir(base) if base.is_pointer_typed() else base.get_place()
        if address is None and not base.is_struct_value():
            logger.error(f'Cannot access field [{index}] on non-struct type',
                         node=node, exc_type=ValueError)
        record = address.type.pointee if address is not None else ensure_ir(base).type
        path = record._pythoc_field_paths[index]
        if path is None:
            logger.error(f'Field [{index}] has no LLVM representation',
                         node=node, exc_type=TypeError)

        metadata = dict(type_hint=field_type, var_name=name, linear_path=linear_path,
                        linear_owner_id=base.linear_owner_id)
        if address is None:
            value = visitor.builder.extract_value(ensure_ir(base), path)
            return wrap_value(value, kind='value', **metadata)

        indices = [ir.Constant(ir.IntType(32), i) for i in (0, *path)]
        pointer = visitor.builder.gep(address, indices)
        return wrap_value(pointer, kind='address', address=pointer, **metadata)

    @classmethod
    def get_all_fields(cls, visitor, base, node) -> list:
        """Get all fields of a struct as a list of ValueRefs.
        
        This is useful for unpacking struct values, e.g., for multi-dimensional
        subscript access where a tuple (i, j, k) becomes struct and needs to be
        unpacked into individual indices.
        
        Args:
            visitor: AST visitor
            base: Pre-evaluated struct ValueRef
            node: AST node for error reporting
        
        Returns:
            List of ValueRef for each field
        """
        from .python_type import pyconst
        from ..valueref import wrap_value
        
        cls._ensure_field_types_resolved()
        field_count = len(cls._field_types) if cls._field_types else 0
        
        fields = []
        for i in range(field_count):
            field_vref = cls.handle_subscript(visitor, base, 
                wrap_value(i, kind='python', type_hint=pyconst[i]), node)
            fields.append(visitor.value_dispatcher.read_rvalue(field_vref))
        
        return fields


def _explicit_ctypes_fields(fields, field_indices, layout):
    """Represent the already planned field offsets using packed ctypes storage."""
    import ctypes

    names = {name for name, _ in fields}
    output = []
    cursor = 0
    padding_index = 0
    for (name, ctype), (index, _) in zip(fields, field_indices):
        offset = layout.field_offsets[index]
        if offset > cursor:
            padding = f'_pc_padding_{padding_index}'
            while padding in names:
                padding_index += 1
                padding = f'_pc_padding_{padding_index}'
            output.append((padding, ctypes.c_ubyte * (offset - cursor)))
            padding_index += 1
        output.append((name, ctype))
        cursor = offset + ctypes.sizeof(ctype)
    return output


def create_struct_type(field_types: List[Any], field_names: Optional[List[str]] = None,
                       python_class: Optional[type] = None,
                       elements: Optional[List[Any]] = None,
                       layout_config=None) -> type:
    """Build a struct from Python type metadata."""
    node = logger.current_location() or SourceLocation.from_frame(sys._getframe(1))
    return _create_struct_type(
        field_types, node, field_names, python_class, elements, layout_config)


def _create_struct_type(field_types: List[Any], node, field_names: Optional[List[str]] = None,
                       python_class: Optional[type] = None,
                       elements: Optional[List[Any]] = None,
                       layout_config=None) -> type:
    """Create a struct type without global caching
    
    Args:
        field_types: List of field types
        field_names: Optional list of field names
        python_class: Optional Python class (for @compile decorated classes)
        elements: Optional list of ValueRef elements (for tuple conversion)
    
    Returns:
        StructType subclass
    """
    from ..type_descriptor import validate_member_names, validate_type_argument
    validate_member_names(field_names, 'struct', node=node)
    for field in field_types:
        validate_type_argument(field, 'struct field', allow_incomplete=True, allow_void=False, node=node)
    from ..target_layout import NATURAL_LAYOUT
    layout_config = layout_config or NATURAL_LAYOUT

    # Generate name for the struct type
    if python_class:
        canonical_name = python_class.__name__
    else:
        # Generate a descriptive name from field info.
        # Use get_name() / __name__ instead of get_type_id() to avoid
        # infinite recursion when a field type contains a self-referential
        # pointer (e.g. struct Expr { ptr['Expr'] lhs; }).
        # The full type identity is projected lazily by type_id.get_type_id,
        # after recursive fields have been bound.
        parts = ["struct"]
        for i, ftype in enumerate(field_types):
            if ftype is None:
                parts.append("void")
            elif hasattr(ftype, 'get_name'):
                parts.append(ftype.get_name())
            elif hasattr(ftype, '__name__'):
                parts.append(ftype.__name__)
            else:
                parts.append(str(ftype))
            if field_names and i < len(field_names) and field_names[i]:
                parts.append(field_names[i])
        canonical_name = "_".join(parts)
    
    # Create new struct type using StructTypeMeta (no caching)
    new_type = StructTypeMeta(
        canonical_name,
        (StructType,),
        {
            '_pc_specialized': True,
            '_canonical_name': canonical_name,
            '_field_types': field_types,
            '_field_names': field_names,
            '_python_class': python_class,
            '_struct_info': None,
            '_elements': elements,  # Store original ValueRef elements for iteration
            '_layout_config': layout_config,
        }
    )
    
    return new_type


class struct(StructType):
    """Struct type factory - supports both decorator and subscript syntax
    
    This is the user-facing struct type that supports:
    - struct[i32, i32] - unnamed fields (subscript syntax)
    - struct[x: i32, y: i32] - named fields (subscript syntax)
    - @struct - decorator for classes (delegates to @compile)
    - @struct(suffix=...) - decorator with suffix parameter
    - @struct(anonymous=True) - decorator with anonymous naming
    """
    
    def __new__(cls, target=None, suffix=None, **kwargs):
        """Support @struct decorator syntax with parameters
        
        Uses the common decorator pattern from CompositeType base class.
        """
        return cls.create_decorator_instance(target, suffix, kwargs)
    
    @classmethod
    def get_name(cls) -> str:
        return 'struct'
    
    @classmethod
    def __class_getitem__(cls, item):
        """Support struct[...] subscript syntax"""
        normalized = cls.normalize_subscript_items(item, node=None)
        return cls.handle_type_subscript(normalized, node=None)
    
    @classmethod
    def handle_type_subscript(cls, items, node):
        """Handle type subscript with normalized items
        
        Items are already normalized to Tuple[Tuple[Optional[str], type], ...] format.
        
        Args:
            items: Normalized tuple where each item is (Optional[str], type)
                   - (None, type) for unnamed fields
                   - ("name", type) for named fields
        
        Returns:
            StructType subclass
        """
        import builtins
        
        if not isinstance(items, builtins.tuple):
            logger.error("struct subscript must be a tuple", node=node, exc_type=TypeError)
        
        # Extract field types and names from normalized format
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
        
        return _create_struct_type(field_types, node, field_names)


__all__ = ['struct', 'StructType', 'create_struct_type']
