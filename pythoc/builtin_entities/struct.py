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

from llvmlite import ir
from ..logger import logger
from typing import List, Tuple, Optional, Any, Dict
from .composite_base import CompositeType
from ..valueref import extract_constant_index


class StructType(CompositeType):
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
        _structure_hash: Hash of field types and names for fast compatibility check
    """
    
    _canonical_name: str = None
    _python_class: Optional[type] = None
    _struct_info: Optional[Any] = None  # StructInfo from registry
    _llvm_struct_type: Optional[Any] = None  # Cached IdentifiedStructType
    _setting_body: bool = False  # Flag to prevent recursive body setting
    _llvm_body_set: bool = False  # Whether LLVM body has been set
    _structure_hash: Optional[int] = None  # Hash for fast type compatibility check
    _llvm_field_map: Optional[Dict[int, int]] = None  # PC field idx -> LLVM field idx
    
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
        
        # Check cache first
        if hasattr(cls, '_ctypes_struct_cache') and cls._ctypes_struct_cache is not None:
            return cls._ctypes_struct_cache
        
        cls._ensure_field_types_resolved()
        
        # Build fields list
        fields = []
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
        
        # Create ctypes.Structure class dynamically
        class_name = f"CStruct_{cls.get_name()}"
        struct_class = type(class_name, (ctypes.Structure,), {
            '_fields_': fields
        })
        
        cls._ctypes_struct_cache = struct_class
        return struct_class
    
    @classmethod
    def _is_compile_class(cls) -> bool:
        """Check if this struct is from @compile decorated class"""
        return cls._python_class is not None
    
    @classmethod
    def _build_llvm_field_map(cls) -> Dict[int, int]:
        """Build mapping from PC field index to LLVM field index.
        
        This is needed because zero-sized fields (like pyconst) are skipped
        in LLVM struct layout but still exist in PC field list.
        
        Returns:
            Dict[int, int]: PC field index -> LLVM field index
                            -1 means zero-sized field (no LLVM representation)
        """
        cls._ensure_field_types_resolved()
        
        field_map = {}
        llvm_idx = 0
        
        for pc_idx, field_type in enumerate(cls._field_types):
            # Check if field is zero-sized
            # Criteria: get_size_bytes() == 0 AND get_llvm_type() returns None
            # (linear type is zero-sized but has LLVM representation as empty struct)
            is_zero_sized = False
            if hasattr(field_type, 'get_size_bytes'):
                field_size = field_type.get_size_bytes()
                if field_size == 0:
                    # Check if it has no LLVM representation (like pyconst)
                    # linear has empty struct representation, so it's NOT skipped
                    if hasattr(field_type, 'get_llvm_type'):
                        llvm_type = field_type.get_llvm_type()
                        if llvm_type is None:
                            is_zero_sized = True
            
            if is_zero_sized:
                # Zero-sized field (pyconst) - mark as -1
                field_map[pc_idx] = -1
            else:
                # Normal field - assign LLVM index
                field_map[pc_idx] = llvm_idx
                llvm_idx += 1
        
        return field_map
    
    @classmethod
    def _get_llvm_field_index(cls, pc_field_index: int) -> int:
        """Get LLVM field index from PC field index.
        
        Args:
            pc_field_index: PC field index (0-based)
        
        Returns:
            LLVM field index, or -1 for zero-sized fields
        """
        if cls._llvm_field_map is None:
            cls._llvm_field_map = cls._build_llvm_field_map()
        
        return cls._llvm_field_map.get(pc_field_index, -1)
    
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
        if cls._field_types is None:
            logger.error("struct type requires field types", node=node, exc_type=TypeError)
        
        # Ensure field types are resolved
        cls._ensure_field_types_resolved()
        
        # Rule: @compile class always uses IdentifiedStructType
        if cls._is_compile_class():
            if module_context is None:
                logger.error(
                    f"@compile class '{cls._canonical_name}' requires module_context for IdentifiedStructType",
                    node=node, exc_type=TypeError)
            # Get or create the IdentifiedStructType (opaque initially)
            # Use Python class name, not canonical name, for @compile classes
            # This ensures consistency with registry lookups
            llvm_type_name = cls._python_class.__name__ if cls._python_class else cls._canonical_name
            
            # Apply anonymous suffix if present (for @compile(anonymous=True))
            if cls._python_class and hasattr(cls._python_class, '_anonymous_suffix') and cls._python_class._anonymous_suffix:
                llvm_type_name = llvm_type_name + cls._python_class._anonymous_suffix
            
            # For anonymous types, don't cache at class level (each instance needs its own LLVM type)
            # For non-anonymous types, use cached type
            if cls._python_class and hasattr(cls._python_class, '_anonymous_suffix') and cls._python_class._anonymous_suffix:
                # Anonymous: always get fresh type from module_context
                llvm_struct_type = module_context.get_identified_type(llvm_type_name)
            else:
                # Non-anonymous: use cached type
                if cls._llvm_struct_type is None:
                    cls._llvm_struct_type = module_context.get_identified_type(llvm_type_name)
                llvm_struct_type = cls._llvm_struct_type
            
            # Set body if not already set (avoid infinite recursion for self-referential types)
            # Check if elements is None or empty (opaque type) AND not currently setting body
            if not llvm_struct_type.elements and not cls._setting_body:
                # Mark that we're setting body to prevent recursion
                cls._setting_body = True
                try:
                    # Build field types, filtering out zero-sized fields (pyconst)
                    llvm_field_types = []
                    for field_type in cls._field_types:
                        if hasattr(field_type, 'get_llvm_type'):
                            llvm_type = field_type.get_llvm_type(module_context)
                            # Skip None (zero-sized fields like pyconst)
                            if llvm_type is not None:
                                llvm_field_types.append(llvm_type)
                        else:
                            logger.error(f"Unknown struct field type {field_type}", node=node, exc_type=TypeError)
                    
                    # Set body (this will set elements attribute)
                    llvm_struct_type.set_body(*llvm_field_types)
                finally:
                    # Always clear the flag
                    cls._setting_body = False
            
            return llvm_struct_type
        
        # Rule: Anonymous struct[...] always uses LiteralStructType
        # Note: Field names affect PC type caching (via canonical_name) but not LLVM structure
        
        # For LiteralStructType, need all fields resolved
        cls._ensure_field_types_resolved()
        
        if any(isinstance(ft, str) for ft in cls._field_types):
            logger.error(f"Cannot create LiteralStructType with unresolved field types: {cls._field_types}",
                        node=node, exc_type=TypeError)
        
        # Build LLVM field types, filtering out zero-sized fields (pyconst)
        llvm_field_types = []
        for field_type in cls._field_types:
            if hasattr(field_type, 'get_llvm_type'):
                # Try to call with module_context, fallback to no args
                import inspect
                try:
                    sig = inspect.signature(field_type.get_llvm_type)
                    if len(sig.parameters) > 0:
                        llvm_type = field_type.get_llvm_type(module_context)
                    else:
                        llvm_type = field_type.get_llvm_type()
                    # Skip None (zero-sized fields like pyconst)
                    if llvm_type is not None:
                        llvm_field_types.append(llvm_type)
                except (ValueError, TypeError):
                    llvm_type = field_type.get_llvm_type()
                    if llvm_type is not None:
                        llvm_field_types.append(llvm_type)
            elif isinstance(field_type, ir.Type):
                # ANTI-PATTERN: field_type should be BuiltinEntity, not ir.Type
                logger.error(
                    f"struct field type is raw LLVM type {field_type}. "
                    f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead.",
                    node=node, exc_type=TypeError)
            else:
                logger.error(f"Unknown struct field type {field_type}", node=node, exc_type=TypeError)
        
        # Return LiteralStructType
        return ir.LiteralStructType(llvm_field_types)
    
    @classmethod
    def get_size_bytes(cls) -> int:
        """Get size in bytes (with alignment)
        
        Zero-sized fields (like pyconst) are skipped in size calculation.
        """
        # Ensure field types are resolved
        cls._ensure_field_types_resolved()
        
        if cls._field_types is None:
            return 0
        
        # Calculate size with proper alignment (simplified)
        total_size = 0
        max_align = 1
        
        for field_type in cls._field_types:
            if hasattr(field_type, 'get_size_bytes'):
                field_size = field_type.get_size_bytes()
                # Skip zero-sized fields (pyconst)
                if field_size == 0:
                    continue
                field_align = min(field_size, 8)
            else:
                field_size = 4
                field_align = 4
            
            max_align = max(max_align, field_align)
            # Align current offset
            if total_size % field_align != 0:
                total_size += field_align - (total_size % field_align)
            total_size += field_size
        
        # Align total size to max alignment
        if total_size % max_align != 0:
            total_size += max_align - (total_size % max_align)
        
        return total_size

    
    @classmethod
    def _compute_structure_hash(cls) -> int:
        """Compute hash based on field types and field names
        
        Hash includes:
        - Field types (order matters)
        - Field names (order matters)
        
        Hash excludes:
        - Type name (so @compile class and anonymous struct can be compatible)
        """
        cls._ensure_field_types_resolved()
        
        # Build tuple of (field_type_id, field_name) for hashing
        hash_components = []
        for i, field_type in enumerate(cls._field_types):
            # Use type identity for hashing
            type_id = id(field_type) if field_type is not None else 0
            field_name = cls._field_names[i] if cls._field_names and i < len(cls._field_names) else None
            hash_components.append((type_id, field_name))
        
        return hash(tuple(hash_components))
    
    @classmethod
    def _get_structure_hash(cls) -> int:
        """Get or compute structure hash (cached)"""
        if cls._structure_hash is None:
            cls._structure_hash = cls._compute_structure_hash()
        return cls._structure_hash
    
    @classmethod
    def get_type_id(cls, _visited=None) -> str:
        """Generate unique type ID for struct types"""
        struct_name = cls._canonical_name or getattr(cls, '__name__', 'S')
        suffix = cls.get_type_id_suffix()
        return f"{len(struct_name)}{struct_name}_{suffix}"
    
    @classmethod
    def is_compatible_with(cls, other) -> bool:
        """Check if this struct type is compatible with another type
        
        Strict Rule (Option 1):
        - struct types are compatible if they have identical field types AND field names
        - Field names matter: struct[a: i32, b: f64] != struct[i32, f64]
        - Field names matter: struct[a: i32, b: f64] != struct[x: i32, y: f64]
        - @compile class can be compatible with anonymous struct if structure matches
        
        Uses structure hash for fast comparison.
        """
        # Check if other is a struct type
        if not (hasattr(other, '_field_types') and hasattr(other, '_get_structure_hash')):
            return False
        
        # Fast path: compare structure hashes
        try:
            return cls._get_structure_hash() == other._get_structure_hash()
        except:
            # Fallback to slow path if hash computation fails
            pass
        
        # Slow path: direct comparison
        cls._ensure_field_types_resolved()
        if hasattr(other, '_ensure_field_types_resolved'):
            other._ensure_field_types_resolved()
        
        # Compare field types
        if cls._field_types != other._field_types:
            return False
        
        # Compare field names
        cls_names = cls._field_names if cls._field_names else [None] * len(cls._field_types)
        other_names = other._field_names if other._field_names else [None] * len(other._field_types)
        
        return cls_names == other_names
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node):
        """Handle struct construction: Point() or struct[i32, i32]()
        
        Creates an uninitialized struct value (ir.Undefined).
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the callable (the struct type itself)
            args: Pre-evaluated arguments (should be empty for default constructor)
            node: Original ast.Call node
        """
        from ..valueref import wrap_value
        
        if len(args) != 0:
            logger.error(f"{cls.get_name()}() takes no arguments ({len(args)} given)",
                        node=node, exc_type=TypeError)
        
        # Get LLVM struct type
        struct_type = cls.get_llvm_type(visitor.module.context)
        
        # Create undef value
        struct_value = ir.Constant(struct_type, ir.Undefined)
        
        return wrap_value(struct_value, kind="value", type_hint=cls)
    
    @classmethod
    def handle_attribute(cls, visitor, base, attr_name, node):
        """Handle attribute access on struct instances: point.x
        
        Supports both named field access and methods (for @compile classes).
        Special handling for zero-sized fields (pyconst).
        
        Args:
            visitor: AST visitor
            base: Pre-evaluated base object (ValueRef)
            attr_name: Attribute name (string)
            node: Original ast.Attribute node
        """
        import ast
        from ..valueref import wrap_value, ensure_ir, get_type
        from ..ir_helpers import propagate_qualifiers
        
        # Ensure field types are resolved
        cls._ensure_field_types_resolved()
        
        # Check if it's a field access
        if not cls.has_field(attr_name):
            logger.error(f"struct '{cls.get_name()}' has no field named '{attr_name}'",
                        node=node, exc_type=AttributeError)
        
        field_index = cls.get_field_index(attr_name)
        field_type = cls._field_types[field_index]

        # Propagate qualifiers from struct type to field type
        # If we have const[struct], accessing field should give const[field_type]
        if base.type_hint:
            field_type = propagate_qualifiers(base.type_hint, field_type)
        
        # Check if field is zero-sized (pyconst)
        if hasattr(field_type, 'handle_field_access'):
            # Delegate to field type's handler (for pyconst)
            return field_type.handle_field_access(visitor, base, field_index, attr_name, node)
        
        # Get LLVM field index (may differ from PC index due to zero-sized fields)
        llvm_field_index = cls._get_llvm_field_index(field_index)
        if llvm_field_index == -1:
            # This shouldn't happen if handle_field_access is implemented correctly
            logger.error(f"Zero-sized field '{attr_name}' has no LLVM representation",
                        node=node, exc_type=TypeError)
        
        # Priority 1: Check if base has an address (for lvalue support)
        if base.kind == "address":
            # base is a loaded value with address (common case for variables)
            struct_ptr = base.address
        else:
            # Priority 2: base is a pure struct value without address
            # Use extract_value (no lvalue support)
            base_ir_type = get_type(base)
            if isinstance(base_ir_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
                struct_value = ensure_ir(base)
                field_value = visitor.builder.extract_value(struct_value, llvm_field_index)
                # For struct values without address, we can't provide lvalue
                return wrap_value(field_value, kind="value", type_hint=field_type)
            else:
                logger.error(f"Cannot access field '{attr_name}' on non-struct type {base_ir_type}",
                            node=node, exc_type=ValueError)
        
        # Use GEP to get field address
        zero = ir.Constant(ir.IntType(32), 0)
        idx = ir.Constant(ir.IntType(32), llvm_field_index)
        field_ptr = visitor.builder.gep(struct_ptr, [zero, idx])
        
        # Load value and return with address for lvalue support
        loaded_value = visitor.builder.load(field_ptr)
        
        # Propagate linear tracking info
        base_var_name = getattr(base, 'var_name', None)
        base_linear_path = getattr(base, 'linear_path', None)
        
        if base_var_name and base_linear_path is not None:
            # Extend the path with this field index
            result_var_name = base_var_name
            result_linear_path = base_linear_path + (field_index,)
        else:
            result_var_name = None
            result_linear_path = None
        
        return wrap_value(loaded_value, kind="address", type_hint=field_type, address=field_ptr,
                         var_name=result_var_name, linear_path=result_linear_path)
    
    @classmethod
    def handle_subscript(cls, visitor, base, index, node):
        """Handle value subscript access on struct instances: s[0]
        
        Note: Type subscripts (struct[i32, f64]) are now handled by PythonType.handle_subscript
        which extracts items and calls struct.handle_type_subscript directly.
        This method only handles value subscripts.
        
        Args:
            visitor: AST visitor
            base: Pre-evaluated base object (ValueRef)
            index: Pre-evaluated index (ValueRef)
            node: Original ast.Subscript node
        """
        import ast
        from ..valueref import wrap_value, ensure_ir, get_type

        # VALUE SUBSCRIPT: s[0] - field access by index
        # Ensure field types are resolved
        cls._ensure_field_types_resolved()
        
        # Extract constant index from pre-evaluated index ValueRef
        index_val = extract_constant_index(index, "struct subscript")
        
        if index_val < 0 or index_val >= len(cls._field_types):
            logger.error(f"struct index {index_val} out of range (0-{len(cls._field_types)-1})",
                        node=node, exc_type=IndexError)
        
        field_type = cls._field_types[index_val]
        
        # Check if field is zero-sized (pyconst)
        if hasattr(field_type, 'handle_field_access'):
            # Delegate to field type's handler (for pyconst)
            field_name = cls._field_names[index_val] if cls._field_names and index_val < len(cls._field_names) else f"[{index_val}]"
            return field_type.handle_field_access(visitor, base, index_val, field_name, node)
        
        # Get LLVM field index
        llvm_index_val = cls._get_llvm_field_index(index_val)
        if llvm_index_val == -1:
            logger.error(f"Zero-sized field [{index_val}] has no LLVM representation",
                        node=node, exc_type=TypeError)
        
        # Priority: if base has address, use GEP for lvalue support
        # Otherwise, if base is a struct value, use extract_value
        base_ir_type = get_type(base)
        
        # Compute extended linear path for result
        base_var_name = getattr(base, 'var_name', None)
        base_linear_path = getattr(base, 'linear_path', None)
        if base_var_name and base_linear_path is not None:
            result_var_name = base_var_name
            result_linear_path = base_linear_path + (index_val,)
        else:
            result_var_name = None
            result_linear_path = None
        
        if base.kind == "pointer":
            struct_ptr = ensure_ir(base)
        elif base.address:
            # Base is a loaded struct value with address - use GEP on address
            struct_ptr = base.address
        elif isinstance(base_ir_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
            # Base is a struct value without address (e.g., function return value)
            # Use extract_value - this cannot be used as lvalue
            struct_value = ensure_ir(base)
            field_value = visitor.builder.extract_value(struct_value, llvm_index_val)
            return wrap_value(field_value, kind="value", type_hint=field_type)
        else:
            logger.error(f"Cannot access field [{index_val}] on non-struct type",
                        node=node, exc_type=ValueError)
        
        # Use GEP to get field address
        zero = ir.Constant(ir.IntType(32), 0)
        idx = ir.Constant(ir.IntType(32), llvm_index_val)
        field_ptr = visitor.builder.gep(struct_ptr, [zero, idx])
        
        # Load value and return with address for lvalue support
        loaded_value = visitor.builder.load(field_ptr)
        return wrap_value(loaded_value, kind="address", type_hint=field_type, address=field_ptr,
                        var_name=result_var_name, linear_path=result_linear_path)

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
            fields.append(field_vref)
        
        return fields


def create_struct_type(field_types: List[Any], field_names: Optional[List[str]] = None, 
                       python_class: Optional[type] = None) -> type:
    """Create a struct type without global caching
    
    Args:
        field_types: List of field types
        field_names: Optional list of field names
        python_class: Optional Python class (for @compile decorated classes)
    
    Returns:
        StructType subclass
    """
    # Generate name for the struct type
    if python_class:
        canonical_name = python_class.__name__
    else:
        # Generate a descriptive name from field info
        from ..type_id import get_type_id
        parts = ["struct"]
        for i, ftype in enumerate(field_types):
            type_id = get_type_id(ftype) if ftype else "void"
            parts.append(type_id)
            if field_names and i < len(field_names) and field_names[i]:
                parts.append(field_names[i])
        canonical_name = "_".join(parts)
    
    # Create new struct type (no caching)
    new_type = type(
        canonical_name,
        (StructType,),
        {
            '_canonical_name': canonical_name,
            '_field_types': field_types,
            '_field_names': field_names,
            '_python_class': python_class,
            '_struct_info': None,
            '_structure_hash': None,
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
    
    def __init__(self, target=None, suffix=None, anonymous=False):
        """Initialize struct decorator with optional parameters"""
        self.target = target
        self.suffix = suffix
        self.anonymous = anonymous
    
    def __call__(self, cls):
        """Decorator application"""
        return self._apply_decorator(cls, self.suffix, self.anonymous)
    
    def __new__(cls, target=None, suffix=None, anonymous=False):
        """Support @struct decorator syntax with parameters
        
        Uses the common decorator pattern from CompositeType base class.
        """
        # Delegate to base class factory method
        return cls.create_decorator_instance(target, suffix, anonymous)
    
    @classmethod
    def get_name(cls) -> str:
        return 'struct'
    
    @classmethod
    def __class_getitem__(cls, item):
        """Support struct[...] subscript syntax"""
        normalized = cls.normalize_subscript_items(item)
        return cls.handle_type_subscript(normalized)
    
    @classmethod
    def handle_type_subscript(cls, items):
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
            logger.error("struct subscript must be a tuple", node=None, exc_type=TypeError)
        
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
        
        return create_struct_type(field_types, field_names)


__all__ = ['struct', 'StructType', 'create_struct_type']
