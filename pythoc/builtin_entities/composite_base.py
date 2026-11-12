"""
Composite Type Base Class

This module provides the base class for composite types (struct, union, enum)
with shared functionality for:
- Field subscript access (value[index])
- Type resolution (lazy and forward references)
- Common construction patterns
- Type ID generation
"""

from typing import Any, Dict, List, Optional, Tuple
from llvmlite import ir

from .base import BuiltinType
from ..logger import logger
from ..valueref import ValueRef
from ..target_layout import NATURAL_LAYOUT


class CompositeType(BuiltinType):
    """Base class for composite types (struct, union, enum)
    
    Provides common functionality:
    - Field access via subscript (value[index])
    - Forward reference resolution
    - Type ID generation with field structure hashing
    - Common construction patterns
    
    Subclasses should define:
    - _field_types: List of field types
    - _field_names: Optional list of field names
    - get_llvm_type(module_context): Return LLVM type representation
    """
    
    _field_types: Optional[List[Any]] = None
    _field_names: Optional[List[str]] = None
    _needs_type_resolution: bool = False
    _field_types_resolved: bool = False
    _type_namespace: Optional[dict] = None

    _layout_config = NATURAL_LAYOUT
    
    @classmethod
    def _get_value_address(cls, visitor, base: ValueRef):
        """Get address from a ValueRef or allocate and store
        
        This is a common pattern for composite types: we need a pointer to the value
        to perform GEP operations. If the value already has an address, use it.
        Otherwise, allocate stack space and store the value.
        
        Args:
            visitor: AST visitor instance
            base: ValueRef containing the composite value
            
        Returns:
            LLVM pointer to the value
        """
        from ..valueref import ensure_ir, get_type
        
        if base.has_place():
            return base.require_place()
        
        # Need to allocate and store the value
        value_ir = ensure_ir(base)
        value_ptr = visitor.builder.alloca(get_type(value_ir), name="aggregate.value")
        visitor.builder.store(value_ir, value_ptr)
        return value_ptr
    
    @classmethod
    def _gep_and_load(cls, visitor, ptr, index: int, field_type):
        """Common GEP + load pattern for struct-like field access
        
        Args:
            visitor: AST visitor instance
            ptr: Pointer to the composite value
            index: Field index (integer constant)
            field_type: PC type of the field
            
        Returns:
            ValueRef with loaded value and address for lvalue support
        """
        from ..valueref import wrap_value
        
        zero = ir.Constant(ir.IntType(32), 0)
        idx = ir.Constant(ir.IntType(32), index)
        field_ptr = visitor.builder.gep(ptr, [zero, idx], inbounds=True)
        
        return wrap_value(field_ptr, kind="address", type_hint=field_type, address=field_ptr)
    
    @classmethod
    def _create_undef_constructor(cls, visitor, args, node):
        """Common undefined value constructor: Type()
        
        This is the default constructor for composite types that creates
        an uninitialized value (LLVM undefined).
        
        Args:
            visitor: AST visitor instance
            args: Pre-evaluated arguments (should be empty)
            node: ast.Call node
            
        Returns:
            ValueRef with undefined value
            
        Raises:
            TypeError: If arguments are provided
        """
        from ..valueref import wrap_value
        
        if len(args) != 0:
            logger.error(f"{cls.get_name()}() takes no arguments ({len(args)} given)",
                        node=node, exc_type=TypeError)
        
        from ..type_descriptor import can_zero_initialize
        if not can_zero_initialize(cls):
            logger.error('Cannot default-construct an aggregate with proof-bearing fields; '
                         'provide an explicit initializer', node=node, exc_type=TypeError)
        llvm_type = cls.get_llvm_type(visitor.module.context)
        value = ir.Constant(llvm_type, ir.Undefined)
        
        # Use _python_class as type_hint if available (for @struct/@union decorated classes)
        # This ensures IntOrFloat() returns type_hint=IntOrFloat, not union[i32, f64]
        type_hint = getattr(cls, '_python_class', None) or cls
        return wrap_value(value, kind="value", type_hint=type_hint)
    
    @classmethod
    def get_layout(cls):
        from ..type_graph import prepare_type
        prepare_type(cls)
        cls._ensure_field_types_resolved()
        if cls._field_types is None:
            from ..target_layout import TypeLayout
            return TypeLayout.empty()

        from ..target_layout import get_target_layout, get_type_layout
        fields = (get_type_layout(field) for field in cls._field_types)
        return get_target_layout().struct(fields)

    @classmethod
    def get_alignment(cls) -> int:
        return cls.get_layout().alignment
    
    @classmethod
    def _ensure_field_types_resolved(cls):
        """Resolve scoped field references and refresh the storage projection."""
        if cls._field_types_resolved or not cls._needs_type_resolution:
            return

        from ..session import CompileSession
        with CompileSession.current().type_graph._lock:
            # Double-checked: another thread may have resolved while we waited.
            if cls._field_types_resolved or not cls._needs_type_resolution:
                return
            cls._resolve_field_types_locked()

    @classmethod
    def _resolve_field_types_locked(cls):
        """Resolve string field types to actual type objects.

        Caller must hold the session type graph lock and have checked
        _needs_type_resolution / _field_types_resolved.
        """
        from ..type_resolver import TypeResolver

        owner = getattr(cls, '_python_class', None)
        if owner is not None:
            from ..decorators.class_statics import split_static_members
            for _, _, field, _ in owner._type_declaration:
                TypeResolver.resolve_type(field, cls._type_namespace, required=False)
            fields = split_static_members(owner, cls._field_types)
            names = [name for name, _ in owner._struct_fields]
            if (names != cls._field_names or len(fields) != len(cls._field_types)
                    or any(a is not b for a, b in zip(fields, cls._field_types))):
                cls._field_types[:] = fields
                cls._field_names[:] = names
                cls._llvm_field_map = None
        else:
            for field in cls._field_types:
                TypeResolver.resolve_type(field, cls._type_namespace, required=False)

    @classmethod
    def get_field_index(cls, name: str) -> int:
        """Get field index by name
        
        Args:
            name: Field name
            
        Returns:
            Field index (0-based)
            
        Raises:
            AttributeError: If field name doesn't exist
        """
        if cls._field_names is None:
            logger.error(f"{cls.get_name()} has no named fields", node=None, exc_type=AttributeError)
        
        try:
            return cls._field_names.index(name)
        except ValueError:
            logger.error(f"{cls.get_name()} has no field named '{name}'", node=None, exc_type=AttributeError)
    
    @classmethod
    def has_field(cls, name: str) -> bool:
        """Check if composite type has a named field"""
        return cls._field_names is not None and name in cls._field_names
    
    @classmethod
    def get_field_count(cls) -> int:
        """Get total number of fields"""
        return len(cls._field_types) if cls._field_types else 0
    
    @classmethod
    def get_field_type(cls, index: int):
        """Get field type by index
        
        Args:
            index: Field index (0-based)
            
        Returns:
            PC type of the field
            
        Raises:
            IndexError: If index is out of range
        """
        if cls._field_types is None or index >= len(cls._field_types):
            logger.error(f"{cls.get_name()} field index {index} out of range", node=None, exc_type=IndexError)
        from ..type_resolver import TypeResolver
        return TypeResolver.resolve_type(cls._field_types[index], cls._type_namespace, freeze=True)
    
    @classmethod
    def can_be_called(cls) -> bool:
        """Composite types can be called to construct values"""
        # Base class cannot be called, only subclasses (created via subscript/decorator)
        return cls._field_types is not None
    
    @classmethod
    def _parse_class_fields(cls, target_cls):
        """Parse class fields from annotations - common logic for struct/union/enum
        
        This extracts field types and names from a decorated class, handling:
        - Annotation extraction
        - Type resolution (including forward references)
        - Type namespace building
        
        Args:
            target_cls: The class being decorated
            
        Returns:
            dict with keys:
                - field_types: List of resolved field types
                - field_names: List of field names
                - type_namespace: Dict for type resolution
                - needs_type_resolution: bool, True if forward refs exist
        """
        from ..type_resolver import TypeResolver
        from ..decorators.annotation_resolver import build_annotation_namespace
        from ..utils import normalize_suffix
        import sys
        
        # Extract struct fields from annotations
        if not hasattr(target_cls, '_struct_fields'):
            target_cls._struct_fields = []
            if hasattr(target_cls, '__annotations__'):
                for field_name, field_type in target_cls.__annotations__.items():
                    target_cls._struct_fields.append((field_name, field_type))
        
        # Get globals from the module where the class is defined
        user_globals = {}
        if hasattr(target_cls, '__module__'):
            if target_cls.__module__ in sys.modules:
                module = sys.modules[target_cls.__module__]
                if hasattr(module, '__dict__'):
                    user_globals = module.__dict__
        
        is_dynamic = '.<locals>.' in target_cls.__qualname__
        
        # Build comprehensive namespace for type resolution
        type_namespace = build_annotation_namespace(
            user_globals,
            is_dynamic=is_dynamic, definition=target_cls
        )
        
        # Add the class itself to namespace for self-referential types
        type_namespace[target_cls.__name__] = target_cls
        type_namespace.type_owner_name = target_cls.__name__
        type_namespace.type_owner_namespace = target_cls.__dict__
        
        type_resolver = TypeResolver(user_globals=type_namespace)
        
        # Resolve field types
        parsed_field_types = []
        for fname, ftype in target_cls._struct_fields:
            if isinstance(ftype, str):
                # Delay the whole expression until its dependencies exist.
                resolved_type = type_resolver.parse_annotation(ftype, defer_unresolved=True)
            else:
                resolved_type = ftype
            from ..type_resolver import TypeResolver
            parsed_field_types.append(TypeResolver.bind_type_references(
                resolved_type, type_namespace,
                self_types={target_cls.__name__: target_cls}))

        # Class-level static members (``name: static[T] = ...``) are not
        # instance fields; split them out before the unified type is built.
        from ..decorators.class_statics import split_static_members
        parsed_field_types = split_static_members(target_cls, parsed_field_types)

        field_names = [fname for fname, ftype in target_cls._struct_fields]
        needs_type_resolution = True
        
        return {
            'field_types': parsed_field_types,
            'field_names': field_names,
            'type_namespace': type_namespace,
            'needs_type_resolution': needs_type_resolution,
        }
    
    @classmethod
    def _link_class_to_type(cls, target_cls, unified_type, suffix=None):
        """Link Python class to unified type and setup common attributes
        
        Args:
            target_cls: Original decorated class
            unified_type: The created type (StructType/UnionType subclass)
            suffix: Optional suffix for naming
        """
        from ..registry import register_struct_from_class
        from ..forward_ref import mark_type_defined
        
        # Link Python class to unified type
        # Use appropriate attribute name based on type
        if getattr(target_cls, '_is_union', False):
            target_cls._union_type = unified_type
        else:
            target_cls._struct_type = unified_type
        target_cls._field_types = unified_type._field_types
        target_cls._field_names = unified_type._field_names
        
        # Copy _canonical_name if available
        if hasattr(unified_type, '_canonical_name'):
            target_cls._canonical_name = unified_type._canonical_name
        
        # Store suffix for deduplication and output file control
        if suffix:
            target_cls._anonymous_suffix = f'_{suffix}'
            target_cls._compile_suffix = suffix
        else:
            target_cls._anonymous_suffix = None
            target_cls._compile_suffix = None
        
        # Delegate common protocol methods to unified type
        target_cls.handle_type_call = unified_type.handle_type_call
        target_cls.handle_attribute = unified_type.handle_attribute
        target_cls.get_llvm_type = unified_type.get_llvm_type
        target_cls.get_name = unified_type.get_name
        target_cls.get_field_index = unified_type.get_field_index
        target_cls.has_field = unified_type.has_field
        target_cls.get_field_count = unified_type.get_field_count
        target_cls.get_layout = unified_type.get_layout
        target_cls.get_size_bytes = unified_type.get_size_bytes
        target_cls.get_alignment = unified_type.get_alignment
        target_cls.get_ctypes_type = unified_type.get_ctypes_type
        target_cls.handle_subscript = unified_type.handle_subscript
        target_cls._ensure_field_types_resolved = unified_type._ensure_field_types_resolved
        target_cls.get_type_id = unified_type.get_type_id
        target_cls._layout_config = unified_type._layout_config
        
        # Inject Python-level constructor that returns pc_literal
        _inject_pc_literal_constructor(target_cls, unified_type)

        # Register and mark as defined
        register_struct_from_class(target_cls)
        mark_type_defined(target_cls.__name__, target_cls)
    
    @classmethod
    def create_decorator_instance(cls, target=None, suffix=None, **kwargs):
        """Factory method to create decorator instance with common parameters
        
        This provides a unified interface for struct/union/enum decorators to accept
        common compilation parameters (suffix, etc.)
        
        Args:
            target: The class being decorated (if used without parens)
            suffix: Optional suffix for compilation output naming
            **kwargs: Additional decorator-specific parameters
        
        Returns:
            Either the decorated result (if target provided) or decorator instance
        """
        # Create a decorator instance that holds the parameters
        class DecoratorInstance:
            def __init__(self):
                self.suffix = suffix
                self.kwargs = kwargs
            
            def __call__(self, target_cls):
                return cls._apply_decorator(target_cls, suffix, **kwargs)
        
        # If target is provided, apply directly
        if target is not None and isinstance(target, type):
            return cls._apply_decorator(target, suffix, **kwargs)
        
        # Otherwise return decorator instance
        return DecoratorInstance()
    
    @classmethod
    def _apply_decorator(cls, target_cls, suffix=None, **kwargs):
        """Apply decorator to target class - to be overridden by subclasses
        
        Args:
            target_cls: The class being decorated
            suffix: Optional suffix for compilation
            **kwargs: Additional parameters
        
        Returns:
            Decorated class/type
        """
        # Default implementation: delegate to compile_dynamic_class
        from ..decorators.structs import compile_dynamic_class
        from ..target_layout import make_aggregate_layout_config

        layout_config = make_aggregate_layout_config(**kwargs)
        return compile_dynamic_class(
            target_cls,
            suffix=suffix,
            layout_config=layout_config,
        )


def _inject_pc_literal_constructor(target_cls, unified_type):
    """Inject ``__new__`` so Python-level construction returns a pc_literal.

    This is called from the decorator paths for @compile class (struct),
    @union, and @enum.  Inside @compile, construction goes through the AST
    visitor's handle_type_call -- never through Python's ``__new__`` -- so
    there is no interference with the compiled code path.
    """
    original_new = target_cls.__new__

    def __new__(cls, *args, **kwargs):
        from .pc_literal import pc_literal
        return pc_literal._from_composite_call(unified_type, *args, **kwargs)

    target_cls.__new__ = __new__


__all__ = ['CompositeType', '_inject_pc_literal_constructor']
