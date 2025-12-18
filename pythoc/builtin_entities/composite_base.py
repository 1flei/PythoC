"""
Composite Type Base Class

This module provides the base class for composite types (struct, union, enum)
with shared functionality for:
- Field subscript access (value[index])
- Type resolution (lazy and forward references)
- Common construction patterns
- Type ID generation
"""

import ast
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from llvmlite import ir

from .base import BuiltinType
from ..logger import logger


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
    
    @classmethod
    def _extract_constant_index(cls, node: ast.Subscript, visitor=None, index_value=None) -> int:
        """Extract constant integer from subscript node or ValueRef
        
        NEW: This is a compatibility wrapper around extract_constant_index().
        Supports both old API (node only) and new API (visitor + index_value).
        
        Old API (deprecated but still supported):
            index = cls._extract_constant_index(node)
        
        New API (recommended):
            index_value = visitor.visit_expression(node.slice)
            index = cls._extract_constant_index(node, visitor, index_value)
        
        Args:
            node: ast.Subscript node (for error context and old API)
            visitor: Optional AST visitor (for new API)
            index_value: Optional pre-evaluated ValueRef (for new API)
            
        Returns:
            Integer index value
            
        Raises:
            TypeError: If index is not a constant integer
        """
        from ..valueref import extract_constant_index
        
        # New API: use pre-evaluated index_value
        if index_value is not None:
            return extract_constant_index(index_value, f"{cls.get_name()} subscript")
        
        # Old API: parse AST node directly (for backward compatibility)
        if isinstance(node.slice, ast.Constant):
            index_val = node.slice.value
        elif isinstance(node.slice, ast.Index):  # Python 3.8 compatibility
            if isinstance(node.slice.value, ast.Constant):
                index_val = node.slice.value.value
            else:
                logger.error(f"{cls.get_name()} subscript index must be a constant",
                            node=node, exc_type=TypeError)
        else:
            logger.error(f"{cls.get_name()} subscript index must be a constant",
                        node=node, exc_type=TypeError)
        
        if not isinstance(index_val, int):
            logger.error(f"{cls.get_name()} subscript index must be an integer, got {type(index_val)}",
                        node=node, exc_type=TypeError)
        
        return index_val
    
    @classmethod
    def _get_value_address(cls, visitor, base):
        """Get address from ValueRef or allocate and store
        
        This is a common pattern for composite types: we need a pointer to the value
        to perform GEP operations. If the value already has an address, use it.
        Otherwise, allocate stack space and store the value.
        
        Args:
            visitor: AST visitor instance
            base: ValueRef containing the composite value
            
        Returns:
            LLVM pointer to the value
        """
        from ..valueref import ValueRef, ensure_ir, get_type
        
        if isinstance(base, ValueRef) and base.address is not None:
            return base.address
        
        # Need to allocate and store the value
        value_ir = ensure_ir(base)
        value_ptr = visitor.builder.alloca(get_type(value_ir))
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
        
        loaded_value = visitor.builder.load(field_ptr)
        return wrap_value(loaded_value, kind="address", type_hint=field_type, address=field_ptr)
    
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
        
        llvm_type = cls.get_llvm_type(visitor.module.context)
        value = ir.Constant(llvm_type, ir.Undefined)
        
        # Use _python_class as type_hint if available (for @struct/@union decorated classes)
        # This ensures IntOrFloat() returns type_hint=IntOrFloat, not union[i32, f64]
        type_hint = getattr(cls, '_python_class', None) or cls
        return wrap_value(value, kind="value", type_hint=type_hint)
    
    @classmethod
    def get_alignment(cls) -> int:
        """Get alignment in bytes for this composite type.
        
        Default implementation: max alignment of all fields, capped at 8.
        Subclasses may override for specific alignment rules.
        
        Returns:
            Alignment in bytes (1, 2, 4, or 8)
        """
        cls._ensure_field_types_resolved()
        
        if cls._field_types is None:
            return 1
        
        max_align = 1
        for field_type in cls._field_types:
            if field_type is None:
                continue
            if hasattr(field_type, 'get_size_bytes'):
                field_size = field_type.get_size_bytes()
                if field_size == 0:
                    continue
                if hasattr(field_type, 'get_alignment'):
                    field_align = field_type.get_alignment()
                else:
                    field_align = min(field_size, 8)
                max_align = max(max_align, field_align)
        
        return max_align
    
    @classmethod
    def _ensure_field_types_resolved(cls):
        """Resolve field types lazily if needed (for forward references)
        
        This method handles forward references by resolving string type names
        to actual type objects. It's called automatically before field access.
        
        Forward references can occur when:
        - Struct A contains field of type B, and B is defined later
        - Recursive types (struct containing pointer to itself)
        
        Resolution strategy:
        1. Use saved type namespace if available
        2. Add all registered structs/types to namespace
        3. Add all defined types from forward_ref system
        4. Resolve string field types using TypeResolver
        """
        if cls._field_types_resolved or not cls._needs_type_resolution:
            return
        
        from ..type_resolver import TypeResolver
        from ..registry import get_unified_registry
        
        # Build namespace for type resolution
        type_namespace = cls._type_namespace or {}
        
        # Add all registered structs to namespace
        registry = get_unified_registry()
        for struct_name in registry.list_structs():
            struct_info = registry.get_struct(struct_name)
            if struct_info and struct_info.python_class:
                type_namespace[struct_name] = struct_info.python_class
        
        # Add all defined types from forward_ref system
        try:
            from ..forward_ref import _defined_types
            type_namespace.update(_defined_types)
        except ImportError:
            pass
        
        type_resolver = TypeResolver(user_globals=type_namespace)
        
        # Resolve string field types
        resolved_types = []
        for ftype in cls._field_types:
            if isinstance(ftype, str):
                try:
                    resolved = type_resolver.parse_annotation(ftype)
                    if resolved is None:
                        logger.error(f"Cannot resolve field type: {ftype}", node=None, exc_type=TypeError)
                    resolved_types.append(resolved)
                except Exception as e:
                    logger.error(f"Failed to resolve field type '{ftype}': {e}", node=None, exc_type=TypeError)
            else:
                resolved_types.append(ftype)
        
        cls._field_types = resolved_types
        cls._field_types_resolved = True
    
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
        return cls._field_types[index]
    
    @classmethod
    def get_type_id_suffix(cls) -> str:
        """Generate type ID suffix based on field structure
        
        This creates a hash-based suffix that uniquely identifies the composite type
        by its field types and names. Same structure = same hash = same type.
        
        Returns:
            8-character hex hash suffix
            
        Examples:
            struct[x: i32, y: i32] -> "a3b4c5d6"
            union[i32, f64] -> "f1e2d3c4"
        """
        from ..type_id import get_type_id
        
        if not cls._field_types:
            return "empty"
        
        # Build field structure string for hashing
        field_ids = []
        field_names = cls._field_names or [None] * len(cls._field_types)
        
        for fname, ftype in zip(field_names, cls._field_types):
            ftype_id = get_type_id(ftype) if ftype is not None else "void"
            if fname:
                field_ids.append(f"{fname}:{ftype_id}")
            else:
                field_ids.append(ftype_id)
        
        field_str = ','.join(field_ids)
        field_hash = hashlib.md5(field_str.encode()).hexdigest()[:8]
        return field_hash
    
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
        from ..decorators.annotation_resolver import build_annotation_namespace, resolve_string_annotation
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
            is_dynamic=is_dynamic
        )
        
        # Add the class itself to namespace for self-referential types
        type_namespace[target_cls.__name__] = target_cls
        
        type_resolver = TypeResolver(user_globals=type_namespace)
        
        # Resolve field types
        parsed_field_types = []
        for fname, ftype in target_cls._struct_fields:
            if isinstance(ftype, str):
                resolved_type = resolve_string_annotation(ftype, type_namespace, type_resolver)
                parsed_field_types.append(resolved_type)
            else:
                parsed_field_types.append(ftype)
        
        field_names = [fname for fname, ftype in target_cls._struct_fields]
        needs_type_resolution = any(isinstance(ft, str) for ft in parsed_field_types)
        
        return {
            'field_types': parsed_field_types,
            'field_names': field_names,
            'type_namespace': type_namespace,
            'needs_type_resolution': needs_type_resolution,
        }
    
    @classmethod
    def _setup_forward_ref_callbacks(cls, unified_type, target_cls, parsed_field_types, type_namespace):
        """Setup forward reference callbacks for unresolved types
        
        Args:
            unified_type: The created type (StructType/UnionType subclass)
            target_cls: Original decorated class
            parsed_field_types: List of parsed field types (may contain strings)
            type_namespace: Type resolution namespace
        """
        from ..forward_ref import register_forward_ref_callback, extract_type_names_from_annotation
        from ..type_resolver import TypeResolver
        
        for field_index, (fname, ftype) in enumerate(target_cls._struct_fields):
            if isinstance(parsed_field_types[field_index], str):
                type_str = parsed_field_types[field_index]
                referenced_types = extract_type_names_from_annotation(type_str)
                
                for ref_type_name in referenced_types:
                    def make_callback(idx, type_str_copy, namespace_copy):
                        def callback(resolved_type_obj):
                            namespace_copy[ref_type_name] = resolved_type_obj
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
    
    @classmethod
    def _link_class_to_type(cls, target_cls, unified_type, suffix=None, anonymous=False):
        """Link Python class to unified type and setup common attributes
        
        Args:
            target_cls: Original decorated class
            unified_type: The created type (StructType/UnionType subclass)
            suffix: Optional suffix for naming
            anonymous: Whether to use anonymous naming
        """
        from ..utils import get_anonymous_suffix
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
        elif anonymous:
            target_cls._anonymous_suffix = get_anonymous_suffix()
            target_cls._compile_suffix = None
        else:
            target_cls._anonymous_suffix = None
            target_cls._compile_suffix = None
        
        # Delegate common protocol methods to unified type
        target_cls.handle_call = unified_type.handle_call
        target_cls.handle_attribute = unified_type.handle_attribute
        target_cls.get_llvm_type = unified_type.get_llvm_type
        target_cls.get_name = unified_type.get_name
        target_cls.get_field_index = unified_type.get_field_index
        target_cls.has_field = unified_type.has_field
        target_cls.get_field_count = unified_type.get_field_count
        target_cls.get_size_bytes = unified_type.get_size_bytes
        target_cls.get_ctypes_type = unified_type.get_ctypes_type
        target_cls.handle_subscript = unified_type.handle_subscript
        target_cls._ensure_field_types_resolved = unified_type._ensure_field_types_resolved
        target_cls.get_type_id = unified_type.get_type_id
        
        # Register and mark as defined
        register_struct_from_class(target_cls)
        mark_type_defined(target_cls.__name__, target_cls)
    
    @classmethod
    def create_decorator_instance(cls, target=None, suffix=None, anonymous=False, **kwargs):
        """Factory method to create decorator instance with common parameters
        
        This provides a unified interface for struct/union/enum decorators to accept
        common compilation parameters (suffix, anonymous, etc.)
        
        Args:
            target: The class being decorated (if used without parens)
            suffix: Optional suffix for compilation output naming
            anonymous: Whether to generate anonymous/unique names
            **kwargs: Additional decorator-specific parameters
        
        Returns:
            Either the decorated result (if target provided) or decorator instance
        """
        # Create a decorator instance that holds the parameters
        class DecoratorInstance:
            def __init__(self):
                self.suffix = suffix
                self.anonymous = anonymous
                self.kwargs = kwargs
            
            def __call__(self, target_cls):
                return cls._apply_decorator(target_cls, suffix, anonymous, **kwargs)
        
        # If target is provided, apply directly
        if target is not None and isinstance(target, type):
            return cls._apply_decorator(target, suffix, anonymous, **kwargs)
        
        # Otherwise return decorator instance
        return DecoratorInstance()
    
    @classmethod
    def _apply_decorator(cls, target_cls, suffix=None, anonymous=False, **kwargs):
        """Apply decorator to target class - to be overridden by subclasses
        
        Args:
            target_cls: The class being decorated
            suffix: Optional suffix for compilation
            anonymous: Whether to use anonymous naming
            **kwargs: Additional parameters
        
        Returns:
            Decorated class/type
        """
        # Default implementation: delegate to compile_dynamic_class
        from ..decorators.structs import compile_dynamic_class
        return compile_dynamic_class(target_cls, suffix=suffix, anonymous=anonymous)


__all__ = ['CompositeType']
