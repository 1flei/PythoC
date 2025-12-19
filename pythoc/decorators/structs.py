# -*- coding: utf-8 -*-
import inspect
import textwrap
import ast
from ..registry import register_struct_from_class
from ..builtin_entities.struct import create_struct_type
from ..logger import set_source_context


def add_struct_handle_call(cls):
    def handle_call(visitor, func_ref, args, node):
        """Handle struct construction with new protocol
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the callable (the struct type itself)
            args: Pre-evaluated arguments (should be empty)
            node: Original ast.Call node
        """
        from llvmlite import ir
        from ..valueref import wrap_value
        from ..registry import get_unified_registry
        
        if len(args) != 0:
            raise TypeError(f"{cls.__name__}() takes no arguments ({len(args)} given)")
        
        struct_name = cls.__name__
        struct_info = get_unified_registry().get_struct(struct_name)
        if not struct_info:
            raise NameError(f"Struct '{struct_name}' not found in registry")
        struct_type = visitor.module.context.get_identified_type(struct_name)
        struct_value = ir.Constant(struct_type, ir.Undefined)
        return wrap_value(struct_value, kind="value", type_hint=cls)
    
    def get_llvm_type(module_context=None):
        """Get LLVM type for this struct class
        
        Args:
            module_context: Optional IR module context. If provided, returns IdentifiedStructType.
                          If None, tries to get from struct_info.llvm_type or creates LiteralStructType.
        """
        from ..registry import get_unified_registry
        from llvmlite import ir
        
        struct_name = cls.__name__
        struct_info = get_unified_registry().get_struct(struct_name)
        if not struct_info:
            raise NameError(f"Struct '{struct_name}' not found in registry")
        
        # If module_context is provided, return identified struct type
        if module_context:
            return module_context.get_identified_type(struct_name)
        
        # Try to use the llvm_type from struct_info (which should be IdentifiedStructType)
        if struct_info.llvm_type:
            return struct_info.llvm_type
        
        # Otherwise, create a literal struct type
        field_llvm_types = []
        for field_name, field_type in struct_info.fields:
            if hasattr(field_type, 'get_llvm_type'):
                # All PC types now accept module_context parameter uniformly
                # Pass None since we don't have module_context here (old struct system)
                field_llvm_types.append(field_type.get_llvm_type(module_context))
            elif isinstance(field_type, ir.Type):
                # ANTI-PATTERN: LLVM types should not be used in struct field definitions
                raise TypeError(
                    f"Struct field '{field_name}' has raw LLVM type {field_type}. "
                    f"This is a bug - use BuiltinEntity types (i32, f64, ptr[T], etc.) instead."
                )
            else:
                # Fallback for unknown types - this should also be an error
                raise TypeError(
                    f"Struct field '{field_name}' has unknown type {field_type} (type: {type(field_type)}). "
                    f"Use BuiltinEntity types (i32, f64, ptr[T], etc.)."
                )
    
    def get_name():
        """Get type name"""
        return cls.__name__

    def handle_attribute(visitor, base, attr_name, node):
        """Handle attribute access on struct instances (unified duck typing protocol)
        
        Args:
            visitor: AST visitor
            base: Pre-evaluated base object (ValueRef)
            attr_name: Attribute name (string)
            node: Original ast.Attribute node
        """
        import ast
        from llvmlite import ir
        from ..valueref import wrap_value, ensure_ir, get_type
        from ..registry import get_unified_registry
        
        struct_name = cls.__name__
        struct_info = get_unified_registry().get_struct(struct_name)
        
        if not struct_info or not struct_info.has_field(attr_name):
            raise AttributeError(f"struct '{struct_name}' has no field named '{attr_name}'")
        
        field_index = struct_info.get_field_index(attr_name)
        field_type = struct_info.get_field_type_hint(attr_name, visitor.type_resolver)
        
        # Priority 1: Check if base has an address (for lvalue support)
        if base.kind == "pointer":
            # base is a pointer to struct
            struct_ptr = ensure_ir(base)
        elif hasattr(base, 'address') and base.address is not None:
            # base is a loaded value with address (common case for variables)
            struct_ptr = base.address
        else:
            # Priority 2: base is a pure struct value without address
            # Use extract_value (no lvalue support)
            base_ir_type = get_type(base)
            if isinstance(base_ir_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
                struct_value = ensure_ir(base)
                field_value = visitor.builder.extract_value(struct_value, field_index)
                return wrap_value(field_value, kind="value", type_hint=field_type)
            else:
                raise ValueError(f"Cannot access field '{attr_name}' on non-struct type {base_ir_type}")
        
        # Use GEP to get field address
        zero = ir.Constant(ir.IntType(32), 0)
        idx = ir.Constant(ir.IntType(32), field_index)
        field_ptr = visitor.builder.gep(struct_ptr, [zero, idx])
        
        # Load value and return with address for lvalue support
        loaded_value = visitor.builder.load(field_ptr)
        return wrap_value(loaded_value, kind="value", type_hint=field_type, address=field_ptr)
    
    cls.handle_call = staticmethod(handle_call)
    cls.handle_attribute = staticmethod(handle_attribute)
    cls.get_llvm_type = staticmethod(get_llvm_type)
    cls.get_name = staticmethod(get_name)
    return cls


def compile_dynamic_class(cls, anonymous=False, suffix=None, type_factory=None):
    """Compile a @compile decorated class into a unified struct/union type
    
    This function now uses the unified struct type system with structural typing.
    Handles both overload (generic) and non-overload cases uniformly.
    
    Args:
        cls: The class to compile
        anonymous: If True, append a unique suffix to the class name in LLVM IR
        suffix: If provided, use this as custom suffix for output files (e.g., "int" for Vector(int))
        type_factory: Optional factory function to create the type (default: create_struct_type)
                      For union, pass create_union_type
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
        _, start_line = inspect.getsourcelines(cls)
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
                cls._struct_fields.append((field_name, field_type))
    
    # Mark as struct BEFORE parsing field types (critical for self-referential types)
    cls._is_struct = True
    
    # Add temporary get_type_id method EARLY for circular reference handling
    # This prevents errors when ptr[TreeNode] tries to call TreeNode.get_type_id()
    # before TreeNode is fully defined
    @classmethod
    def _temp_get_type_id(cls_arg):
        return f'{len(cls_name)}{cls_name}'
    cls.get_type_id = _temp_get_type_id
    
    # === BUILD TYPE RESOLUTION NAMESPACE ===
    from ..type_resolver import TypeResolver
    from .annotation_resolver import build_annotation_namespace, resolve_string_annotation
    
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
        is_dynamic=is_dynamic
    )
    
    # Add the class itself to namespace for self-referential types
    # Use BOTH original name and current name (after renaming)
    type_namespace[original_cls_name] = cls
    type_namespace[cls.__name__] = cls
    
    type_resolver = TypeResolver(user_globals=type_namespace)
    
    # === RESOLVE FIELD TYPES ===
    parsed_field_types = []
    for fname, ftype in cls._struct_fields:
        if isinstance(ftype, str):
            resolved_type = resolve_string_annotation(ftype, type_namespace, type_resolver)
            parsed_field_types.append(resolved_type)
        else:
            parsed_field_types.append(ftype)
    
    # === CREATE UNIFIED STRUCT TYPE ===
    field_names = [fname for fname, ftype in cls._struct_fields]
    unified_type = type_factory(parsed_field_types, field_names, python_class=cls)
    
    # === HANDLE FORWARD/CIRCULAR REFERENCES ===
    # Check if any field types are still strings (unresolved forward references)
    unified_type._needs_type_resolution = any(isinstance(ft, str) for ft in parsed_field_types)
    if unified_type._needs_type_resolution:
        unified_type._type_namespace = type_namespace
        
        # Register callbacks for unresolved field types
        from ..forward_ref import register_forward_ref_callback, extract_type_names_from_annotation
        
        for field_index, (fname, ftype) in enumerate(cls._struct_fields):
            if isinstance(parsed_field_types[field_index], str):
                type_str = parsed_field_types[field_index]
                referenced_types = extract_type_names_from_annotation(type_str)
                
                for ref_type_name in referenced_types:
                    def make_callback(idx, type_str_copy, namespace_copy):
                        def callback(resolved_type_obj):
                            namespace_copy[ref_type_name] = resolved_type_obj
                            from ..type_resolver import TypeResolver
                            type_resolver_cb = TypeResolver(user_globals=namespace_copy)
                            try:
                                new_parsed = type_resolver_cb.parse_annotation(type_str_copy)
                                if new_parsed is not None:
                                    unified_type._field_types[idx] = new_parsed
                            except Exception:
                                pass
                        return callback
                    
                    register_forward_ref_callback(
                        ref_type_name,
                        make_callback(field_index, type_str, type_namespace.copy())
                    )
    
    # === LINK PYTHON CLASS TO UNIFIED TYPE ===
    cls._struct_type = unified_type
    cls._canonical_name = unified_type._canonical_name
    cls._field_types = unified_type._field_types
    cls._field_names = unified_type._field_names
    
    # Store suffix for deduplication and output file control
    # suffix: deterministic naming for deduplication (replaces anonymous in the future)
    # anonymous: auto-generated unique naming (legacy, will be deprecated)
    if suffix:
        # suffix takes priority - use it for deterministic deduplication
        cls._anonymous_suffix = f'_{suffix}'
        cls._compile_suffix = suffix
    elif anonymous:
        # Auto-generate unique suffix (legacy behavior)
        from ..utils import get_anonymous_suffix
        cls._anonymous_suffix = get_anonymous_suffix()
        cls._compile_suffix = None
    else:
        cls._anonymous_suffix = None
        cls._compile_suffix = None
    
    # Delegate all protocol methods to unified type
    cls.handle_call = unified_type.handle_call
    cls.handle_attribute = unified_type.handle_attribute
    cls.get_llvm_type = unified_type.get_llvm_type
    cls.get_name = unified_type.get_name
    cls.get_field_index = unified_type.get_field_index
    cls.has_field = unified_type.has_field
    cls.get_field_count = unified_type.get_field_count
    cls.get_size_bytes = unified_type.get_size_bytes
    cls.get_ctypes_type = unified_type.get_ctypes_type
    cls.handle_subscript = unified_type.handle_subscript
    cls._ensure_field_types_resolved = unified_type._ensure_field_types_resolved
    cls.get_type_id = unified_type.get_type_id
    
    # Delegate struct-specific methods if available
    if hasattr(unified_type, '_get_structure_hash'):
        cls._get_structure_hash = unified_type._get_structure_hash
    if hasattr(unified_type, '_compute_structure_hash'):
        cls._compute_structure_hash = unified_type._compute_structure_hash
    
    # Add __iter__ method to support *struct unpacking in Python
    def __iter__(self):
        """Enable struct unpacking: f(*struct_instance)
        
        Yields field values in declaration order.
        This allows Python's * operator to work with struct instances.
        """
        for field_name in self._field_names:
            yield getattr(self, field_name)
    
    cls.__iter__ = __iter__
    
    # === REGISTER AND MARK AS DEFINED ===
    register_struct_from_class(cls)
    
    # Mark type as defined (triggers forward reference callbacks)
    from ..forward_ref import mark_type_defined
    mark_type_defined(original_cls_name, cls)
    
    return cls
