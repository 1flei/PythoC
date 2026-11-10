# -*- coding: utf-8 -*-
import inspect
import textwrap
import ast
from ..registry import register_struct_from_class
from ..builtin_entities.struct import create_struct_type
from ..logger import set_source_context, logger
from .class_methods import _INHERIT_EFFECT_SUFFIX


def compile_dynamic_class(
    cls,
    suffix=None,
    type_factory=None,
    captured_symbols=None,
    static_linkage='weak_odr',
    layout_config=None,
    effect_suffix=_INHERIT_EFFECT_SUFFIX,
):
    """Compile a @compile decorated class into a unified struct/union type
    
    This function now uses the unified struct type system with structural typing.
    Handles both overload (generic) and non-overload cases uniformly.
    
    Args:
        cls: The class to compile
        suffix: If provided, use this as custom suffix for output files (e.g., "int" for Vector(int))
        type_factory: Optional factory function to create the type (default: create_struct_type)
                      For union, pass create_union_type
        captured_symbols: Symbols captured at @compile decoration time. Forwarded to
                      attach_class_methods so that class-body ``def`` methods can resolve
                      names visible at the decoration site (closure variables, the class
                      itself for self-referential annotations, etc.).
        static_linkage: LLVM linkage for class-level static members (default
                      weak_odr: one program-wide entity, like a C++ inline
                      static member; 'internal' gives module-private storage)
        effect_suffix: Resolved effect binding for class-body methods. When
                      omitted, methods inherit the active definition context.
    """
    # Default to struct type factory
    if type_factory is None:
        type_factory = create_struct_type
    from ..utils import normalize_suffix
    
    # Normalize suffix (handle tuple/list of type parameters)
    suffix = normalize_suffix(suffix)
    
    # Save original class name for self-referential type resolution
    original_cls_name = cls.__name__
    cls_name = cls.__name__
    
    # Set source context for error messages
    try:
        from ..utils.inspect_utils import get_object_sourcelines
        _, start_line = get_object_sourcelines(cls)
        source_file = inspect.getfile(cls)
        set_source_context(source_file, start_line - 1)
    except (OSError, TypeError):
        pass
    
    # === UNIFIED STRUCT COMPILATION PATH ===
    # This handles ALL cases: overload/non-overload, dynamic/static, with/without source
    
    # Extract struct fields from annotations
    if not hasattr(cls, '_struct_fields'):
        cls._struct_fields = []
        if hasattr(cls, '__annotations__'):
            for field_name, field_type in cls.__annotations__.items():
                # Undo Python's private-name mangling: a field written as
                # __x in the class body is stored as _ClassName__x (with the
                # class name's leading underscores stripped), but C struct
                # fields keep their literal name (e.g. Darwin's
                # struct sigaction.__sigaction_u).
                mangled_prefix = f'_{original_cls_name.lstrip("_")}'
                if (field_name.startswith(mangled_prefix + '__')
                        and not field_name.endswith('__')):
                    field_name = field_name[len(mangled_prefix):]
                cls._struct_fields.append((field_name, field_type))
    
    # Mark as struct BEFORE parsing field types (critical for self-referential types)
    cls._is_struct = True
    
    # Add temporary get_type_id method EARLY for circular reference handling
    # This prevents errors when ptr[TreeNode] tries to call TreeNode.get_type_id()
    # before TreeNode is fully defined
    @classmethod
    def _temp_get_type_id(cls_arg, _visited=None):
        from ..type_id import get_type_id
        return get_type_id(cls_arg)
    cls.get_type_id = _temp_get_type_id
    
    # === BUILD TYPE RESOLUTION NAMESPACE ===
    from ..type_resolver import TypeResolver
    from .annotation_resolver import build_annotation_namespace
    
    # Get globals from the module where the class is defined
    user_globals = {}
    if hasattr(cls, '__module__'):
        import sys
        if cls.__module__ in sys.modules:
            module = sys.modules[cls.__module__]
            if hasattr(module, '__dict__'):
                user_globals = module.__dict__
    
    is_dynamic = '.<locals>.' in cls.__qualname__
    
    # Build comprehensive namespace for type resolution
    type_namespace = build_annotation_namespace(
        user_globals,
        is_dynamic=is_dynamic, definition=cls
    )
    
    # Add the class itself to namespace for self-referential types
    # Use BOTH original name and current name (after renaming)
    type_namespace[original_cls_name] = cls
    type_namespace[cls.__name__] = cls
    type_namespace.type_owner_name = original_cls_name
    type_namespace.type_owner_namespace = cls.__dict__
    
    type_resolver = TypeResolver(user_globals=type_namespace)
    
    # === RESOLVE FIELD TYPES ===

    parsed_field_types = []
    for fname, ftype in cls._struct_fields:
        if isinstance(ftype, str):
            # Delay the whole expression if a dependency is not defined yet.
            resolved_type = type_resolver.parse_annotation(ftype, defer_unresolved=True)
        else:
            resolved_type = ftype
        # Formed pointers can still carry a quoted self-reference. Bind it
        # to this declaration before a later factory call reuses its name.
        parsed_field_types.append(
            TypeResolver.bind_type_references(
                resolved_type, type_namespace,
                self_types={original_cls_name: cls, cls.__name__: cls}))

    # Class-level static members (``name: static[T] = ...``) are not instance
    # fields; split them out before the unified type is built.
    from .class_statics import split_static_members
    parsed_field_types = split_static_members(cls, parsed_field_types)

    # === CREATE UNIFIED STRUCT TYPE ===
    field_names = [fname for fname, ftype in cls._struct_fields]

    for fname, ftype in zip(field_names, parsed_field_types):
        if getattr(ftype, '_is_param', False) or (
            isinstance(ftype, type) and getattr(ftype, '_is_param', False)
        ):
            logger.error(
                f"'param' cannot be used as a struct field type ('{fname}')",
                exc_type=TypeError,
            )
    unified_type = type_factory(
        parsed_field_types,
        field_names,
        python_class=cls,
        layout_config=layout_config,
    )
    
    # Keep references scoped until their first actual lowering. Name
    # notifications must not freeze an intermediate registry placeholder.
    unified_type._needs_type_resolution = True

    # === LINK PYTHON CLASS TO UNIFIED TYPE ===
    cls._struct_type = unified_type
    cls._canonical_name = unified_type._canonical_name
    cls._field_types = unified_type._field_types
    cls._field_names = unified_type._field_names
    
    # Store suffix for deduplication and output file control
    if suffix:
        # Use suffix for deterministic deduplication
        cls._anonymous_suffix = f'_{suffix}'
        cls._compile_suffix = suffix
    else:
        cls._anonymous_suffix = None
        cls._compile_suffix = None
    cls._static_linkage = static_linkage
    
    # Delegate all protocol methods to unified type
    cls.handle_type_call = unified_type.handle_type_call
    cls.handle_attribute = unified_type.handle_attribute
    cls.get_llvm_type = unified_type.get_llvm_type
    cls.get_name = unified_type.get_name
    cls.get_field_index = unified_type.get_field_index
    cls.has_field = unified_type.has_field
    cls.get_field_count = unified_type.get_field_count
    cls.get_layout = unified_type.get_layout
    cls.get_size_bytes = unified_type.get_size_bytes
    cls.get_alignment = unified_type.get_alignment
    cls.get_ctypes_type = unified_type.get_ctypes_type
    cls.handle_subscript = unified_type.handle_subscript
    cls.is_struct_type = unified_type.is_struct_type
    cls.get_all_fields = unified_type.get_all_fields
    cls._ensure_field_types_resolved = unified_type._ensure_field_types_resolved
    cls.get_type_id = unified_type.get_type_id
    cls._layout_config = unified_type._layout_config
    
    # Delegate struct-specific methods if available
    
    # Add __iter__ method to support *struct unpacking in Python
    def __iter__(self):
        """Enable struct unpacking: f(*struct_instance)
        
        Yields field values in declaration order.
        This allows Python's * operator to work with struct instances.
        """
        for field_name in self._field_names:
            yield getattr(self, field_name)
    
    cls.__iter__ = __iter__
    
    # Inject Python-level constructor that returns pc_literal
    from ..builtin_entities.composite_base import _inject_pc_literal_constructor
    _inject_pc_literal_constructor(cls, unified_type)

    # === REGISTER AND MARK AS DEFINED ===
    register_struct_from_class(cls)
    
    # Mark type as defined (triggers forward reference callbacks)
    from ..forward_ref import mark_type_defined
    mark_type_defined(original_cls_name, cls)
    
    # Compile and attach plain-def methods declared in the class body.
    from .class_methods import attach_class_methods
    attach_class_methods(
        cls,
        class_compile_suffix=suffix,
        captured_symbols=captured_symbols,
        effect_suffix=effect_suffix,
    )

    return cls
