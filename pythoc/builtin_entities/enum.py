"""
Enum builtin entity for PC

Provides Rust-style tagged unions with type annotation syntax:

@enum(i32)
class MyEnum:
    Variant1: i32 = 0      # explicit tag, has payload
    Variant2: f64          # auto tag (1), has payload
    Variant3: None         # no payload, auto tag (2)
    Variant4: ptr[i8] = 10 # explicit tag with jump

Memory layout: Always struct { tag_type, union[type...] }
- For variants with payload: union contains the actual types
- For variants without payload (None): union contains void (size 0)

Generates:
- Constants: MyEnum.Variant1, MyEnum.Variant2, etc. (integer tag values)
- Constructor: MyEnum(tag, payload) -> enum instance
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Tuple

from .composite_base import CompositeType
from .types import i8, i32, i64, void
from .union import union as union_type
from ..valueref import ValueRef, wrap_value, extract_constant_index
from ..logger import logger


def _create_enum_type(variants, tag_type, class_name=None):
    """Unified factory function to create enum types
    
    Args:
        variants: List of (name, payload_type, tag) tuples
        tag_type: Type for the tag field (e.g., i8, i32)
        class_name: Optional name for the enum class
    
    Returns:
        New EnumType subclass
    """
    # Build union for payloads (use variant names as field names)
    payload_items = []
    for var_name, ptype, _tag in variants:
        if ptype is not None:
            payload_items.append((var_name, ptype))
        else:
            payload_items.append((var_name, void))
    
    # Create union with all payload types using variant names as field names
    union_payload = union_type.handle_type_subscript(tuple(payload_items))
    
    # Build tag constants dict
    tag_values = {}
    for var_name, _ptype, tag in variants:
        tag_values[var_name] = tag
    
    # Create enum class name
    if class_name is None:
        class_name = f"enum[{', '.join(name for name, _, _ in variants)}]"
    
    # Create new enum class that inherits from EnumType
    enum_cls = type(
        class_name,
        (EnumType,),
        {
            '_tag_type': tag_type,
            '_union_payload': union_payload,
            '_variant_names': [name for name, _, _ in variants],
            '_variant_types': [ptype for _, ptype, _ in variants],
            '_tag_values': tag_values,
        }
    )
    
    # Setup field types for CompositeType base class
    enum_cls._setup_field_types()
    
    # Set tag constants as class attributes
    for var_name, tag in tag_values.items():
        setattr(enum_cls, var_name, tag)
    
    return enum_cls


# Note: enum decorator is defined as a class at the bottom of this file
# (see: class enum(metaclass=enum_meta))
# This allows it to support both decorator syntax and subscript syntax



class EnumType(CompositeType):
    """Base class for enum types created via subscript or decorator
    
    This is the type returned by enum[...] subscript syntax.
    
    Enum is represented as a composite type with 2 fields:
    - Field 0: tag (integer discriminant)
    - Field 1: payload (union of variant types)
    
    Additional enum-specific attributes:
    - _tag_type: Type of the tag field
    - _variant_names: List of variant names
    - _variant_types: List of variant payload types
    - _tag_values: Dict[str, int] mapping variant names to tag values
    - _union_payload: Union type for all variant payloads
    """
    _is_enum = True
    _tag_type = None
    _variant_names = None
    _variant_types = None
    _tag_values = None
    _union_payload = None
    
    @classmethod
    def is_enum_type(cls) -> bool:
        """Override base class to return True for enum types."""
        return True
    
    @classmethod
    def get_ctypes_type(cls):
        """Get ctypes type for enum (struct with tag + union payload).
        
        Returns a ctypes.Structure with tag field and payload byte array.
        """
        import ctypes
        
        if cls._tag_type is None or cls._union_payload is None:
            return ctypes.c_void_p
        
        # Get tag ctypes type
        if hasattr(cls._tag_type, 'get_ctypes_type'):
            tag_ctype = cls._tag_type.get_ctypes_type()
        else:
            tag_ctype = ctypes.c_int32
        
        # Get payload ctypes type (union as byte array)
        if hasattr(cls._union_payload, 'get_ctypes_type'):
            payload_ctype = cls._union_payload.get_ctypes_type()
        else:
            payload_ctype = ctypes.c_uint8 * 8  # default
        
        # Create struct with tag + payload
        class_name = f"CEnum_{cls.get_name()}"
        fields = [('tag', tag_ctype)]
        if payload_ctype is not None:
            fields.append(('payload', payload_ctype))
        
        return type(class_name, (ctypes.Structure,), {'_fields_': fields})
    
    @classmethod
    def get_name(cls) -> str:
        if hasattr(cls, '__name__'):
            return cls.__name__
        return 'enum'
    
    @classmethod
    def get_llvm_type(cls, module_context):
        """Get LLVM type for enum: struct { tag, union_payload }
        
        Enum is a composite type with exactly 2 fields:
        - Field 0: tag (discriminant)
        - Field 1: payload (union)
        """
        from llvmlite import ir
        
        if cls._tag_type is None or cls._union_payload is None:
            logger.error(f"{cls.get_name()} requires tag type and union payload", node=None, exc_type=TypeError)
        
        tag_llvm = cls._tag_type.get_llvm_type(module_context)
        payload_llvm = cls._union_payload.get_llvm_type(module_context)
        return ir.LiteralStructType([tag_llvm, payload_llvm])
    
    @classmethod
    def get_size_bytes(cls) -> int:
        """Get size in bytes for enum type
        
        Enum is struct { tag, union_payload }, so size is:
        tag_size + padding + union_size, aligned to max alignment
        """
        if cls._tag_type is None or cls._union_payload is None:
            return 0
        
        tag_size = cls._tag_type.get_size_bytes()
        union_size = cls._union_payload.get_size_bytes()
        
        # Calculate alignment
        tag_align = min(tag_size, 8)
        union_align = min(union_size, 8) if union_size > 0 else 1
        max_align = max(tag_align, union_align)
        
        # Calculate total size with padding
        # Start with tag
        total = tag_size
        
        # Align for union
        if total % union_align != 0 and union_size > 0:
            total += union_align - (total % union_align)
        
        # Add union size
        total += union_size
        
        # Align total to max alignment
        if total % max_align != 0:
            total += max_align - (total % max_align)
        
        return total
    
    @classmethod
    def _setup_field_types(cls):
        """Setup _field_types and _field_names for CompositeType base class
        
        Enum has exactly 2 fields: [tag, payload]
        This allows us to reuse CompositeType's subscript logic.
        """
        if cls._tag_type is None or cls._union_payload is None:
            return
        
        cls._field_types = [cls._tag_type, cls._union_payload]
        cls._field_names = ['tag', 'payload']
        cls._field_types_resolved = True
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node):
        """Handle enum construction: E(E.Tag, payload) or E(E.Tag)
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the callable (the enum type itself)
            args: [tag_value] or [tag_value, payload_value]
            node: ast.Call node
        """
        from ..valueref import wrap_value, ensure_ir, get_type_hint
        from ..type_converter import TypeConverter
        from llvmlite import ir
        from .types import void
        
        if len(args) < 1 or len(args) > 2:
            logger.error(f"{cls.get_name()}() requires 1-2 arguments: (tag) or (tag, payload)",
                        node=node, exc_type=TypeError)
        
        tag_val = args[0]
        payload_val = args[1] if len(args) == 2 else None
        
        # Convert tag to correct type
        converter = TypeConverter(visitor)
        tag_converted = converter.convert(tag_val, cls._tag_type)
        tag_ir = ensure_ir(tag_converted)
        
        # Try to infer which variant this is from constant tag value
        variant_idx = None
        variant_payload_type = None
        if isinstance(tag_ir, ir.Constant):
            tag_const_val = tag_ir.constant
            # Search through variants to find matching tag
            if hasattr(cls, '_variant_names') and hasattr(cls, '_variant_types'):
                variant_names = cls._variant_names or []
                variant_types = cls._variant_types or []
                
                # _tag_values can be either a dict (from @enum decorator) or a list (from enum[...])
                tag_values = cls._tag_values if hasattr(cls, '_tag_values') else None
                if isinstance(tag_values, dict):
                    # @enum decorator format: Dict[str, int]
                    for idx, (var_name, var_type) in enumerate(zip(variant_names, variant_types)):
                        if var_name in tag_values and tag_values[var_name] == tag_const_val:
                            variant_idx = idx
                            variant_payload_type = var_type
                            break
                elif tag_values:
                    # enum[...] subscript format: List[int]
                    for idx, (var_name, var_type, var_tag) in enumerate(zip(
                        variant_names, variant_types, tag_values
                    )):
                        if var_tag == tag_const_val:
                            variant_idx = idx
                            variant_payload_type = var_type
                            break
        
        # Build enum struct: { tag, payload }
        tag_llvm = cls._tag_type.get_llvm_type(visitor.module.context)
        payload_llvm = cls._union_payload.get_llvm_type(visitor.module.context)
        enum_struct_t = ir.LiteralStructType([tag_llvm, payload_llvm])
        enum_alloca = visitor.builder.alloca(enum_struct_t)
        
        # Store tag
        zero = ir.Constant(ir.IntType(32), 0)
        tag_idx = ir.Constant(ir.IntType(32), 0)
        tag_ptr = visitor.builder.gep(enum_alloca, [zero, tag_idx], inbounds=True)
        visitor.builder.store(tag_ir, tag_ptr)
        
        # Store payload if provided
        if payload_val is not None:
            payload_idx = ir.Constant(ir.IntType(32), 1)
            payload_ptr = visitor.builder.gep(enum_alloca, [zero, payload_idx], inbounds=True)
            
            # If we know the variant type from constant tag, use it for conversion
            if variant_payload_type is not None:
                if variant_payload_type == void:
                    # Void variant - skip payload storage
                    pass
                else:
                    # Convert payload to the correct variant payload type
                    payload_converted = converter.convert(payload_val, variant_payload_type)
                    val_ir = ensure_ir(payload_converted)
                    variant_llvm = variant_payload_type.get_llvm_type(visitor.module.context)
                    variant_ptr = visitor.builder.bitcast(payload_ptr, ir.PointerType(variant_llvm))
                    visitor.builder.store(val_ir, variant_ptr)
            else:
                # Unknown variant or runtime tag - infer type from payload
                val_type = get_type_hint(payload_val)
                if val_type is None:
                    logger.error(f"Enum payload requires PC type hint when tag is not compile-time constant",
                                node=node, exc_type=TypeError)
                
                # Skip void types
                if val_type == void:
                    pass
                else:
                    # Convert payload if it's a Python value
                    if isinstance(payload_val, ValueRef) and payload_val.is_python_value():
                        payload_converted = converter.convert(payload_val, val_type)
                        val_ir = ensure_ir(payload_converted)
                    else:
                        val_ir = ensure_ir(payload_val)
                    
                    variant_llvm = val_type.get_llvm_type(visitor.module.context)
                    variant_ptr = visitor.builder.bitcast(payload_ptr, ir.PointerType(variant_llvm))
                    visitor.builder.store(val_ir, variant_ptr)
        
        # Load and return constructed value
        result = visitor.builder.load(enum_alloca)
        return wrap_value(result, kind="address", type_hint=cls, address=enum_alloca)
    
    @classmethod
    def handle_attribute(cls, visitor, base, attr_name, node):
        """Handle attribute access on enum type (for constants like EnumType.VARIANT)
        
        Args:
            visitor: AST visitor
            base: Pre-evaluated base (should be the enum class itself)
            attr_name: Attribute name (variant name)
            node: ast.Attribute node
            
        Returns:
            ValueRef with tag constant value (as Python constant for consistency)
        """
        from ..valueref import wrap_value
        from ..builtin_entities.python_type import PythonType
        
        # Check if this is a variant name
        if hasattr(cls, attr_name):
            tag_value = getattr(cls, attr_name)
            if isinstance(tag_value, int):
                # Return tag as Python constant (follows PC convention)
                python_type_inst = PythonType.wrap(tag_value, is_constant=True)
                return wrap_value(tag_value, kind="python", type_hint=python_type_inst)
        
        logger.error(f"Enum type {cls.get_name()} has no attribute '{attr_name}'",
                    node=node, exc_type=AttributeError)
    
    @classmethod
    def handle_subscript(cls, visitor, base, index, node):
        """Handle enum[...] type subscript or value subscript
        
        Two modes:
        1. Type subscript (index=None): Create new enum type from variants
           Example: enum[A, B: i32]
        
        2. Value subscript (index!=None): Access enum fields by index
           Example: enum_val[0] -> tag, enum_val[1] -> payload
           This delegates to CompositeType's subscript logic.
        """
        import ast
        from ..valueref import wrap_value
        
        # Value subscript: delegate to CompositeType (access by field index)
        if index is not None:
            cls._setup_field_types()
            # Extract constant index from pre-evaluated index ValueRef
            index_val = extract_constant_index(index, "enum subscript")
            return cls._gep_and_load(
                visitor,
                cls._get_value_address(visitor, base),
                index_val,
                cls.get_field_type(index_val)
            )
        
        # Type subscript: enum[Variant1, Variant2: Type2, ...]
        if index is None:
            slice_node = node.slice
            
            # Parse variants - enum has different syntax than struct/union
            # Handle single item (not a tuple)
            if not isinstance(slice_node, ast.Tuple):
                items = [slice_node]
            else:
                items = slice_node.elts
            
            # Parse each variant
            variants = []  # List of (name, type)
            for item in items:
                if isinstance(item, ast.Name):
                    # Bare name - no payload (void)
                    var_name = item.id
                    var_type = void
                    variants.append((var_name, var_type))
                elif isinstance(item, ast.Slice):
                    # Name: Type syntax
                    # lower = variant name, upper = payload type
                    if isinstance(item.lower, ast.Name):
                        var_name = item.lower.id
                    elif isinstance(item.lower, ast.Constant) and isinstance(item.lower.value, str):
                        var_name = item.lower.value
                    else:
                        logger.error(f"Enum variant name must be an identifier or string, got {ast.dump(item.lower)}",
                                    node=node, exc_type=TypeError)
                    
                    # Parse payload type
                    var_type = visitor.type_resolver.parse_annotation(item.upper)
                    variants.append((var_name, var_type))
                else:
                    logger.error(f"Invalid enum variant syntax: {ast.dump(item)}",
                                node=node, exc_type=TypeError)
            
            # Assign auto tags and build variant list for factory
            resolved_variants = []
            next_tag = 0
            for var_name, var_type in variants:
                resolved_variants.append((var_name, var_type, next_tag))
                next_tag += 1
            
            # Use unified factory to create enum type
            enum_cls = _create_enum_type(resolved_variants, i8)
            
            # Return as ValueRef with type as value (for callable protocol)
            return wrap_value(enum_cls, kind="python", type_hint=enum_cls)
        else:
            # Value subscript: should not happen on enum types
            logger.error("Enum types do not support value subscripting", node=node, exc_type=TypeError)
    
    @classmethod
    def handle_type_subscript(cls, items):
        """Handle type subscript with normalized items (for type resolver)
        
        Items are normalized to Tuple[Tuple[Optional[str], type], ...].
        """
        import builtins
        
        if not isinstance(items, builtins.tuple):
            items = (items,)
        
        if len(items) == 0:
            logger.error("enum requires at least one variant", node=None, exc_type=TypeError)
        
        # Parse variants
        variants = []
        for name, var_type in items:
            if name is None:
                logger.error(f"Enum variants must have names, got unnamed type {var_type}",
                            node=None, exc_type=TypeError)
            variants.append((name, var_type))
        
        # Assign auto tags and build variant list for factory
        resolved_variants = []
        next_tag = 0
        for var_name, var_type in variants:
            resolved_variants.append((var_name, var_type, next_tag))
            next_tag += 1
        
        # Use unified factory to create enum type
        return _create_enum_type(resolved_variants, i8)
    
    @classmethod
    def __class_getitem__(cls, item):
        """Python runtime entry point using normalization -> handle_type_subscript"""
        normalized = cls.normalize_subscript_items(item)
        return cls.handle_type_subscript(normalized)


# Make enum support subscript syntax
class enum_meta(type):
    """Metaclass to enable enum[...] subscript syntax"""
    
    def __getitem__(cls, item):
        """Handle enum[Variant1: Type1, Variant2: Type2, ...] syntax
        
        Creates an anonymous enum type with default tag type (i8).
        """
        import builtins
        
        # Parse subscript items
        if not isinstance(item, builtins.tuple):
            items = (item,)
        else:
            items = item
        
        # Parse variants
        variants = []  # List of (name, type_or_None, tag)
        next_tag = 0
        for it in items:
            if isinstance(it, str):
                # Plain name - no payload (void)
                variants.append((it, None, next_tag))
                next_tag += 1
            elif isinstance(it, builtins.slice):
                # Name: Type syntax comes as a slice
                if it.step is not None:
                    logger.error(f"Invalid enum variant syntax: {it}", node=None, exc_type=TypeError)
                
                # Get variant name
                if isinstance(it.start, str):
                    var_name = it.start
                else:
                    var_name = str(it.start)
                
                # Get variant type (None means no payload)
                var_type = it.stop
                variants.append((var_name, var_type, next_tag))
                next_tag += 1
            else:
                # Unknown format
                logger.error(f"Invalid enum variant: {it}", node=None, exc_type=TypeError)
        
        # Use unified factory to create enum type
        return _create_enum_type(variants, i8)


# Replace the enum function with a class that has the metaclass
class enum(metaclass=enum_meta):
    """Enum type factory - supports decorator and subscript syntax
    
    Decorator usage (accepts explicit tag width and optional parameters):
        @enum           # default i8
        @enum(i32)      # explicit tag width
        @enum(i32, suffix=my_type)  # with suffix for generic instantiation
        @enum(i32, anonymous=True)  # with anonymous naming
        class MyEnum:
            Variant1: i32
            Variant2: None
    
    Subscript usage (uses default tag width i8):
        E = enum[Variant1: i32, Variant2: None]
    """
    
    @classmethod
    def handle_subscript(cls, visitor, base, index, node):
        """Handle enum[...] type subscript in PC code
        
        Delegates to EnumType.handle_subscript for implementation.
        """
        return EnumType.handle_subscript(visitor, base, index, node)
    
    def __init__(self, tag_type_or_class=None, suffix=None, anonymous=False):
        """Initialize with tag type for decorator usage or class for direct decoration"""
        # Handle @enum without parens: __init__ not called, __call__ gets the class directly
        # Handle @enum(i32): __init__ gets i32, __call__ gets the class
        # Handle @enum(i32, suffix=t): __init__ gets i32 and suffix
        # Handle @enum: __init__ gets None (default), __call__ gets the class
        
        self.suffix = suffix
        self.anonymous = anonymous
        
        if tag_type_or_class is None:
            # @enum() or used in enum[...] (won't reach here for subscript)
            self.tag_type = i8
        elif isinstance(tag_type_or_class, type) and not hasattr(tag_type_or_class, 'get_llvm_type'):
            # @enum without parens - got the class directly
            # This shouldn't happen with proper metaclass, but handle it anyway
            self.tag_type = i8
            self._cls = tag_type_or_class
        else:
            # @enum(i32) or @enum(i32, suffix=...) - got a tag type
            self.tag_type = tag_type_or_class
    
    def __call__(self, cls):
        """Decorator application"""
        return _create_enum_class(cls, self.tag_type, self.suffix, self.anonymous)
    
    def __new__(cls, tag_type_or_class=None, suffix=None, anonymous=False):
        """Handle @enum without parens and with parameters"""
        # If called with a class as the only argument, it's direct decoration
        if (tag_type_or_class is not None and 
            isinstance(tag_type_or_class, type) and 
            not hasattr(tag_type_or_class, 'get_llvm_type')):
            # @enum without parens - directly decorate
            return _create_enum_class(tag_type_or_class, i8, suffix, anonymous)
        
        # Otherwise, create instance for later __call__
        instance = object.__new__(cls)
        return instance


def _create_enum_class(cls, tag_type, suffix=None, anonymous=False):
    """Build enum class from type annotations
    
    Parses __annotations__ dict for variant definitions with syntax:
    - VarName: Type = explicit_tag
    - VarName: Type (auto tag)
    - VarName (no payload, auto tag)
    
    Args:
        cls: The class definition to convert to enum
        tag_type: Type for the tag field (e.g., i8, i32)
        suffix: Optional suffix for naming/compilation (currently unused for enums)
        anonymous: Optional anonymous naming flag (currently unused for enums)
    """
    from ..type_resolver import TypeResolver
    from ..registry import get_unified_registry
    
    # Get annotations
    annotations = getattr(cls, '__annotations__', {})
    
    # Get user globals from the class's module
    import sys
    module_name = cls.__module__
    user_globals = {}
    if module_name in sys.modules:
        user_globals = vars(sys.modules[module_name])
    
    type_resolver = TypeResolver(user_globals=user_globals)
    
    # Parse class body IN SOURCE ORDER to preserve variant order
    # Build variant list: (name, payload_type, explicit_tag_or_None)
    variants: List[Tuple[str, Optional[Any], Optional[int]]] = []
    
    import inspect
    try:
        src = inspect.getsource(cls)
        import textwrap
        src = textwrap.dedent(src)
        tree = ast.parse(src)
        if tree.body and isinstance(tree.body[0], ast.ClassDef):
            class_def = tree.body[0]
            for stmt in class_def.body:
                # AnnAssign: VarName: Type = explicit_tag
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    var_name = stmt.target.id
                    var_annotation = annotations.get(var_name)
                    
                    # Parse payload type
                    if var_annotation is None or var_annotation == 'None':
                        payload_type = None
                    else:
                        payload_type = type_resolver.parse_annotation(var_annotation)
                    
                    # Parse explicit tag
                    explicit_tag = None
                    if stmt.value is not None and isinstance(stmt.value, ast.Constant):
                        explicit_tag = stmt.value.value
                    
                    variants.append((var_name, payload_type, explicit_tag))
                
                # Assign without annotation: VarName = explicit_tag (no-payload variant)
                elif isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            if not var_name.startswith('_'):
                                # No payload, but has explicit tag
                                explicit_tag = None
                                if isinstance(stmt.value, ast.Constant):
                                    explicit_tag = stmt.value.value
                                variants.append((var_name, None, explicit_tag))
                
                # Bare name: VarName (no payload, no explicit tag)
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Name):
                    bare_name = stmt.value.id
                    if not bare_name.startswith('_'):
                        variants.append((bare_name, None, None))
    except Exception as e:
        # Fallback: use annotations dict (order not guaranteed)
        for var_name, var_annotation in annotations.items():
            if var_annotation is None or var_annotation == 'None':
                payload_type = None
            else:
                payload_type = type_resolver.parse_annotation(var_annotation)
            
            # Try to get explicit tag from class dict
            explicit_tag = None
            if var_name in cls.__dict__ and isinstance(cls.__dict__[var_name], int):
                explicit_tag = cls.__dict__[var_name]
            
            variants.append((var_name, payload_type, explicit_tag))
    
    # Assign auto tags
    next_tag = 0
    resolved_variants: List[Tuple[str, Optional[Any], int]] = []
    for var_name, payload_type, explicit_tag in variants:
        if explicit_tag is not None:
            tag = explicit_tag
            next_tag = tag + 1
        else:
            tag = next_tag
            next_tag += 1
        resolved_variants.append((var_name, payload_type, tag))
    
    # Use unified factory to create enum type
    enum_cls = _create_enum_type(resolved_variants, tag_type, cls.__name__)
    
    # Preserve original module for debugging
    enum_cls.__module__ = cls.__module__
    
    # Register in forward reference system so other types can reference this enum
    from ..forward_ref import mark_type_defined
    mark_type_defined(cls.__name__, enum_cls)
    
    # handle_attribute is inherited from EnumType base class (no need to override)
    
    return enum_cls
