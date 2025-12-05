"""
Type entities (i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, ptr, array, etc.)

This module contains all the builtin type entities.
"""

from llvmlite import ir
from typing import Any, Optional
import ast

from .base import BuiltinType, _get_unified_registry
from ..valueref import ensure_ir, wrap_value, get_type, get_type_hint
from ..logger import logger

# Import LLVM extensions to enable BFloatType and FP128Type
try:
    from .. import llvm_extensions
except ImportError:
    pass  # Extensions not available, will use fallbacks


# ============================================================================
# Integer Types
# ============================================================================

for _w in range(1, 65):
    _name = f'i{_w}'
    _attrs = {
        '_llvm_type': ir.IntType(_w),
        '_size_bytes': max(1, (_w + 7) // 8),
        '_is_signed': True,
        '_is_integer': True,
        'get_name': classmethod(lambda cls, n=_name: n),
    }
    globals()[_name] = type(_name, (BuiltinType,), _attrs)

    _name_u = f'u{_w}'
    _attrs_u = {
        '_llvm_type': ir.IntType(_w),
        '_size_bytes': max(1, (_w + 7) // 8),
        '_is_signed': False,
        '_is_integer': True,
        'get_name': classmethod(lambda cls, n=_name_u: n),
    }
    globals()[_name_u] = type(_name_u, (BuiltinType,), _attrs_u)


# ============================================================================
# Floating Point Types
# ============================================================================

class f32(BuiltinType):
    """32-bit floating point type (IEEE 754 single precision)"""
    _llvm_type = ir.FloatType()
    _size_bytes = 4
    _is_signed = True
    _is_float = True
    
    @classmethod
    def get_name(cls) -> str:
        return 'f32'


class f64(BuiltinType):
    """64-bit floating point type (IEEE 754 double precision)"""
    _llvm_type = ir.DoubleType()
    _size_bytes = 8
    _is_signed = True
    _is_float = True
    
    @classmethod
    def get_name(cls) -> str:
        return 'f64'


# ============================================================================
# Special Types
# ============================================================================

class bool(BuiltinType):
    """Boolean type (i1 in LLVM)"""
    _llvm_type = ir.IntType(1)
    _size_bytes = 1
    _is_signed = False
    _is_bool = True
    
    @classmethod
    def get_name(cls) -> str:
        return 'bool'


class void(BuiltinType):
    """Void type - represents no value (used for function return types)"""
    _llvm_type = ir.VoidType()
    _size_bytes = 0
    _is_signed = False
    
    @classmethod
    def get_name(cls) -> str:
        return 'void'
    
    @classmethod
    def can_be_called(cls) -> bool:
        return False  # void cannot be called or used as type constructor


# ============================================================================
# Pointer Type
# ============================================================================

class ptr(BuiltinType):
    """Pointer type - supports ptr[T] syntax"""
    _size_bytes = 8  # 64-bit pointer
    _is_signed = False
    _is_pointer = True
    pointee_type = None  # Will be set for specialized types
    
    @classmethod
    def get_name(cls) -> str:
        if cls.pointee_type is not None:
            pointee_name = cls.pointee_type.get_name() if hasattr(cls.pointee_type, 'get_name') else str(cls.pointee_type)
            return f'ptr[{pointee_name}]'
        return 'ptr'
    
    @classmethod
    def get_type_id(cls) -> str:
        """Generate unique type ID for pointer types."""
        if cls.pointee_type is not None:
            # Import here to avoid circular dependency
            from ..type_id import get_type_id
            pointee_id = get_type_id(cls.pointee_type)
            return f'P{pointee_id}'
        return 'Pv'  # void pointer
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get LLVM pointer type"""
        if cls.pointee_type is not None:
            # Determine pointee LLVM type from either PC type or direct LLVM type
            if hasattr(cls.pointee_type, 'get_llvm_type'):
                pointee_llvm = None
                if module_context is not None:
                    # Prefer passing module_context for identified/opaque structs
                    try:
                        pointee_llvm = cls.pointee_type.get_llvm_type(module_context)
                    except TypeError as e:
                        if "positional argument" in str(e) or "takes" in str(e):
                            pointee_llvm = cls.pointee_type.get_llvm_type()
                        else:
                            raise
                else:
                    # No context available: still prefer passing module_context when possible
                    try:
                        # Always prefer module_context for identified/opaque struct types
                        pointee_llvm = cls.pointee_type.get_llvm_type(module_context)
                    except TypeError:
                        # Fallback only if the type truly does not accept a context
                        pointee_llvm = cls.pointee_type.get_llvm_type()
                if pointee_llvm is None:
                    raise TypeError(f"ptr.get_llvm_type: failed to obtain pointee LLVM type for {cls.pointee_type}")
            elif isinstance(cls.pointee_type, ir.Type):
                # ANTI-PATTERN: pointee_type should be BuiltinEntity, not ir.Type
                raise TypeError(
                    f"ptr.get_llvm_type: pointee_type is raw LLVM type {cls.pointee_type}. "
                    f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead."
                )
            else:
                raise TypeError(f"ptr.get_llvm_type: unknown pointee type {cls.pointee_type}")
            return ir.PointerType(pointee_llvm)
        return ir.PointerType(ir.IntType(8))  # Default to i8*
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True  # ptr[T] can be called for pointer casting
    
    @classmethod
    def get_pointee_type(cls):
        """Get the type this pointer points to"""
        return cls.pointee_type
    
    @classmethod
    def handle_call(cls, visitor, args, node: ast.Call) -> ir.Value:
        """Handle ptr(value) or ptr[T](value) call
        
        Args:
            visitor: AST visitor
            args: Pre-evaluated arguments (list of ValueRef)
            node: Original ast.Call node
        """
        # If this is a specialized ptr[T], handle type conversion
        if cls.pointee_type is not None:
            return cls._handle_typed_ptr_call(visitor, args, node)
        
        # Otherwise, handle ptr(variable) - get address of a variable
        return cls._handle_getptr_call(visitor, args, node)
    
    @classmethod
    def _handle_getptr_call(cls, visitor, args, node: ast.Call) -> ir.Value:
        """Handle ptr(variable) call - get address of a variable (equivalent to getptr)
        
        Unified implementation: delegates to visit_lvalue to get the address,
        sharing the same logic as lvalue assignments.
        
        Args:
            visitor: AST visitor
            args: Pre-evaluated arguments (should be length 1) - NOT USED, we use node.args[0]
            node: Original ast.Call node
        """
        if len(node.args) != 1:
            raise ValueError("ptr() requires exactly one argument")
        
        arg_node = node.args[0]
        
        # Unified path for subscript: compute lvalue address via visitor and wrap
        if isinstance(arg_node, ast.Subscript):
            lvalue = visitor.visit_lvalue(arg_node)
            address_ptr = lvalue.ir_value
            if not lvalue.type_hint:
                raise TypeError("ptr(): missing PC type hint for lvalue; cannot infer pointee type")
            pointee_type = lvalue.type_hint
            ptr_type = cls[pointee_type]
            
            return wrap_value(address_ptr, kind='value', type_hint=ptr_type)
        
        # Fallback unified path: use visit_lvalue to get the address
        lvalue = visitor.visit_lvalue(arg_node)
        address_ptr = lvalue.ir_value
        
        # Infer the pointee type from the lvalue's type hint
        if lvalue.type_hint:
            pointee_type = lvalue.type_hint
        else:
            # No PC type hint available; do not infer from LLVM
            raise TypeError("ptr(): missing PC type hint for lvalue; cannot infer pointee type")
        
        # Create ptr[T] type hint
        ptr_type = cls[pointee_type]
        
        # Return as a pointer value
        return wrap_value(address_ptr, kind='value', type_hint=ptr_type)
    
    @classmethod
    def handle_type_conversion(cls, visitor, node: ast.Call) -> ir.Value:
        """Handle type conversion to pointer type using TypeConverter"""
        if len(node.args) != 1:
            raise TypeError(f"{cls.get_name()}() takes exactly 1 argument ({len(node.args)} given)")
        
        arg = visitor.visit_expression(node.args[0])
        # Note: TypeConverter will extract LLVM type from pythoc type with module_context
        # So we don't need to call get_llvm_type here
        
        # Use TypeConverter for the conversion (pass PC type directly)
        try:
            result = visitor.type_converter.convert(
                arg,
                cls
            )
            return result
        except TypeError as e:
            raise TypeError(f"Cannot convert to {cls.get_name()}: {e}")
    
    @classmethod
    def handle_add(cls, visitor, left, right, node: ast.BinOp):
        """Handle pointer addition: ptr + offset """
        from ..valueref import wrap_value, ensure_ir, get_type

        logger.debug("Pointer arithmetic add", left_is_ptr=left, right_is_int=right)
        # cast to pc type for python value
        if right.is_python_value():
            right = visitor.type_converter.convert(right, i64)
        
        # Decide by ValueRef.kind and PC type hints only
        gep_result = visitor.builder.gep(ensure_ir(left), [ensure_ir(right)])
        return wrap_value(gep_result, kind="value", type_hint=left.type_hint)
    
    @classmethod
    def handle_sub(cls, visitor, left, right, node: ast.BinOp):
        """Handle pointer addition: ptr - offset """
        # cast to pc type for python value
        if right.is_python_value():
            right = visitor.type_converter.convert(right, i64)
        
        # Decide by ValueRef.kind and PC type hints only
        neg_right = visitor.builder.neg(ensure_ir(right))
        gep_result = visitor.builder.gep(ensure_ir(left), [ensure_ir(neg_right)])
        return wrap_value(gep_result, kind="value", type_hint=left.type_hint)

    @classmethod
    def handle_deref(cls, visitor, base, node: ast.Subscript):
        """Handle pointer dereference operations (unified duck typing protocol)

        """

        logger.debug("Pointer dereference", base=base)

        from ..valueref import wrap_value, ensure_ir
        from ..ir_helpers import propagate_qualifiers
        
        pointee_type_hint = None
        if base.type_hint and hasattr(base.type_hint, 'pointee_type'):
            pointee_type_hint = base.type_hint.pointee_type
        
        if pointee_type_hint is None:
            raise TypeError(f"Cannot infer pointee type for pointer subscript {base}")

        # Propagate qualifiers from pointer type to pointee type
        # If we have const[ptr[i32]], accessing it should give const[i32]
        if base.type_hint:
            pointee_type_hint = propagate_qualifiers(base.type_hint, pointee_type_hint)

        # For array, let the result decay to pointer as in C
        if hasattr(pointee_type_hint, "is_array") and pointee_type_hint.is_array():
            base_ir = ensure_ir(base)
            return wrap_value(base_ir, kind="value", type_hint=pointee_type_hint)
        # Load the value from the pointer
        base_ir = ensure_ir(base)
        pointee_llvm = pointee_type_hint.get_llvm_type(visitor.module.context)
        typed_ptr = visitor.builder.bitcast(base_ir, ir.PointerType(pointee_llvm))
        loaded_value = visitor.builder.load(typed_ptr)
        return wrap_value(loaded_value, kind="address", type_hint=pointee_type_hint, address=typed_ptr)
    
    @classmethod
    def handle_subscript(cls, visitor, base, index, node: ast.Subscript):
        """Handle pointer subscript operations (unified duck typing protocol)
        
        Supports two modes:
        1. Type subscript (index=None): ptr[int_type] or ptr[int_type, dims...] - creates specialized pointer type
           - ptr[i32, 5] = ptr[i32] (first dim ignored)
           - ptr[i32, 3, 5] = ptr[array[i32, 5]] (first dim ignored)
           - ptr[i32, 3, 5, 7] = ptr[array[i32, 5, 7]] (first dim ignored)
        2. Value subscript (index=ValueRef): p[index] - pointer arithmetic (equivalent to *(p + index))
        
        Args:
            visitor: AST visitor instance
            base: Pre-evaluated base object (ValueRef)
            index: Pre-evaluated index (ValueRef) or None for type subscript
            node: Original ast.Subscript node
            
        Returns:
            For type subscript: specialized ptr type (ValueRef kind='python')
            For value subscript: ValueRef with dereferenced value
        """

        from ..valueref import wrap_value, ensure_ir
        
        # Check if this is a type subscript or value subscript
        # Use index=None as the marker (unified protocol with struct/union/enum!)
        if index is None:
            # Type subscript: ptr[T] or ptr[T, dims...]
            # Handle tuple index for multi-dimensional array decay
            # ptr[i32, 5] -> ptr[i32]
            # ptr[i32, 3, 5] -> ptr[array[i32, 5]]
            # ptr[i32, 3, 5, 7] -> ptr[array[i32, 5, 7]]
            
            if isinstance(node.slice, ast.Tuple):
                # Multi-dimensional: ptr[T, dim1, dim2, ...]
                # First element is the base type, rest are dimensions
                # First dimension is ignored (for array decay)
                elts = node.slice.elts
                if len(elts) < 2:
                    raise TypeError(f"ptr tuple subscript requires at least 2 elements")
                
                # Parse the base type (first element)
                base_type = visitor.type_resolver.parse_annotation(elts[0])
                if not base_type:
                    raise TypeError(f"Cannot parse base type from: {ast.dump(elts[0])}")
                
                # Ignore first dimension (elts[1]), use remaining dimensions
                remaining_dims = elts[2:]
                
                if len(remaining_dims) == 0:
                    # ptr[i32, 5] -> ptr[i32] (first dim ignored, no remaining dims)
                    inner_type = base_type
                else:
                    # ptr[i32, 3, 5] -> ptr[array[i32, 5]]
                    # ptr[i32, 3, 5, 7] -> ptr[array[i32, 5, 7]]
                    # Parse remaining dimensions
                    dims = []
                    for dim_node in remaining_dims:
                        if isinstance(dim_node, ast.Constant):
                            dims.append(dim_node.value)
                        else:
                            raise TypeError(f"Array dimension must be constant, got: {ast.dump(dim_node)}")
                    
                    # Create array type with remaining dimensions
                    from .array import array
                    inner_type = array[base_type, *dims]
            else:
                # Single type: ptr[T]
                inner_type = visitor.type_resolver.parse_annotation(node.slice)
                if not inner_type:
                    raise TypeError(f"Cannot parse type from subscript: {ast.dump(node.slice)}")
            
            # Create a specialized ptr[T] type
            class SpecializedPtr(ptr):
                pointee_type = inner_type
            
            # Wrap the type in a ValueRef so it can be used in expressions
            return wrap_value(SpecializedPtr, kind="python", type_hint=SpecializedPtr)
        
        # Value subscript: p[index] or p[i, j, k] - pointer arithmetic
        # Check if index is a struct (multi-dimensional access from tuple expression)
        if index.is_struct_value():
            # Multi-dimensional pointer access: p[i, j, k] where (i, j, k) is a struct
            indices = index.get_pc_type().get_all_fields(visitor, index, node)
            return cls._handle_multidim_subscript(visitor, base, indices, node)
        
        # Single index: p[i] - convert to *(ptr + index)
        base = cls.handle_add(visitor, base, index, node)
        return cls.handle_deref(visitor, base, node)

    @classmethod
    def _handle_multidim_subscript(cls, visitor, base, indices, node: ast.Subscript):
        """Handle multi-dimensional pointer subscript: p[i, j, k]
        
        For ptr[i32, N1, N2, N3] (which is ptr[array[i32, N2, N3]]), 
        p[i, j, k] is equivalent to: p[i][j][k]
        
        This mirrors array's behavior but for pointers to multi-dimensional arrays.
        
        Args:
            visitor: AST visitor instance
            base: Pre-evaluated base object (ValueRef)
            indices: List of pre-evaluated indices (list[ValueRef])
            node: Original ast.Subscript node (for error reporting)
            
        Returns:
            ValueRef with element value
        """
        # Apply each index sequentially
        current = base
        for index in indices:
            # Apply subscript (this handles pointer arithmetic and dereference)
            current = cls.handle_subscript(visitor, current, index, node)
            
            # After first dereference, we might have an array
            # If so, switch to using the array's type for subsequent subscripts
            if current.type_hint and hasattr(current.type_hint, 'get_decay_pointer_type'):
                # This is an array, get its decay pointer type for next iteration
                from .array import array
                if isinstance(current.type_hint, type) and issubclass(current.type_hint, array):
                    # Convert to pointer for next subscript
                    ptr_type = current.type_hint.get_decay_pointer_type()
                    current = visitor.type_converter.convert(current, ptr_type)
        
        return current

    
    @classmethod
    def handle_attribute(cls, visitor, base, attr_name, node: ast.Attribute):
        """Handle ptr[...].attr by penetrating pointer layers and delegating to struct
        
        Design Philosophy:
            ptr should NOT duplicate struct's field access logic.
            Instead, penetrate all ptr layers and delegate to the final struct type.
        
        Supports:
            - ptr[Struct].field -> delegate to Struct.handle_attribute
            - ptr[ptr[Struct]].field -> load once, then delegate
            - ptr[ptr[ptr[Struct]]].field -> load twice, then delegate
        
        Args:
            visitor: AST visitor instance
            base: Pre-evaluated base object (ValueRef)
            attr_name: Attribute name (string)
            node: Original ast.Attribute node
            
        Returns:
            ValueRef with field value
        """
        from ..valueref import wrap_value, ensure_ir, get_type
        
        current_ir = ensure_ir(base)
        current_type_hint = get_type_hint(base)

        if not current_type_hint or not hasattr(current_type_hint, 'pointee_type'):
            raise TypeError("Cannot access attribute on untyped pointer")
        
        # Penetrate ptr[ptr[...]] layers by loading
        # Stop when pointee is not a ptr type
        while (hasattr(current_type_hint, 'pointee_type') and
               isinstance(current_type_hint.pointee_type, type) and
               hasattr(current_type_hint.pointee_type, 'pointee_type')):
            # This is ptr[ptr[...]], load to get the inner pointer
            current_ir = visitor.builder.load(current_ir)
            current_type_hint = current_type_hint.pointee_type
        
        # Now current_type_hint should be ptr[Struct] where Struct is the final type
        struct_type = current_type_hint.pointee_type
        
        # Require PC struct type; do not accept raw LLVM struct types
        if not (isinstance(struct_type, type) and hasattr(struct_type, 'handle_attribute')):
            raise TypeError("ptr attribute access requires PC struct type; got non-PC type")
        
        # Delegate to struct's handle_attribute
        if not hasattr(struct_type, 'handle_attribute'):
            raise AttributeError(
                f"Cannot access attribute '{attr_name}' on ptr[{getattr(struct_type, '__name__', struct_type)}]: "
                f"pointee type does not support attribute access"
            )
        
        struct_value = visitor.builder.load(current_ir)
        struct_base = wrap_value(struct_value, kind="address", type_hint=struct_type, address=current_ir)

        return struct_type.handle_attribute(visitor, struct_base, attr_name, node)
        # Create a ValueRef that struct.handle_attribute expects
        # IMPORTANT: type_hint should be the struct type, not the ptr type!
        # ptr_base = wrap_value(current_ir, kind="value", type_hint=struct_type)
        
        # # Delegate to struct - let it handle GEP and load
        # return struct_type.handle_attribute(visitor, ptr_base, attr_name, node)
    
    @classmethod
    def handle_type_subscript(cls, items):
        """Handle type subscript with normalized items for ptr
        
        Args:
            items: Normalized tuple with exactly one item: (None or name, type)
        
        Returns:
            ptr subclass with pointee_type set
        """
        import builtins
        if not isinstance(items, builtins.tuple):
            items = (items,)
        if len(items) != 1:
            raise TypeError("ptr requires exactly one type parameter: ptr[T]")
        name, item = items[0]
        if item is None:
            raise TypeError("ptr requires a pointee type: ptr[T]")
        inner_name = getattr(item, 'get_name', lambda: str(item))()
        return type(
            f'ptr[{inner_name}]',
            (ptr,),
            {'pointee_type': item}
        )
    
    @classmethod
    def handle_as_type(cls, visitor, node: ast.AST):
        """Handle ptr type annotation resolution (unified call protocol)
        
        Args:
            visitor: AST visitor instance
            node: AST node
                - ast.Name('ptr'): returns ptr base class
                - ast.Subscript('ptr', T): returns ptr[T] specialized type
        
        Returns:
            ptr class or ptr[T] specialized type
        """
 # ptr ,
        if isinstance(node, ast.Name):
            return cls
        
 # ptr[T] ptr[T, dims...] ,
        if isinstance(node, ast.Subscript):
            # Handle tuple subscript: ptr[i32, 3, 4] -> ptr[array[i32, 4]]
            if isinstance(node.slice, ast.Tuple):
                elts = node.slice.elts
                if len(elts) < 2:
                    raise TypeError(f"ptr tuple subscript requires at least 2 elements")
                
                # Parse the base type (first element)
                base_type = visitor.type_resolver.parse_annotation(elts[0])
                if not base_type:
                    raise TypeError(f"Cannot parse base type from: {ast.dump(elts[0])}")
                
                # Ignore first dimension (elts[1]), use remaining dimensions
                remaining_dims = elts[2:]
                
                if len(remaining_dims) == 0:
                    # ptr[i32, 5] -> ptr[i32] (first dim ignored, no remaining dims)
                    inner_type = base_type
                else:
                    # ptr[i32, 3, 5] -> ptr[array[i32, 5]]
                    # ptr[i32, 3, 5, 7] -> ptr[array[i32, 5, 7]]
                    # Parse remaining dimensions
                    dims = []
                    for dim_node in remaining_dims:
                        if isinstance(dim_node, ast.Constant):
                            dims.append(dim_node.value)
                        else:
                            raise TypeError(f"Array dimension must be constant, got: {ast.dump(dim_node)}")
                    
                    # Create array type with remaining dimensions
                    from .array import array
                    inner_type = array[base_type, *dims]
            else:
                # Single type: ptr[T]
                inner_type = visitor.type_resolver.parse_annotation(node.slice)
                if inner_type is None:
                    raise TypeError(f"Cannot parse pointee type from: {ast.dump(node.slice)}")
            
            # Delegate via normalization -> handle_type_subscript
            normalized = cls.normalize_subscript_items(inner_type)
            return cls.handle_type_subscript(normalized)
        
        # Other cases return base class
        return cls
    
    @classmethod
    def _handle_typed_ptr_call(cls, visitor, args, node: ast.Call) -> ir.Value:
        """Handle ptr[T](value) call - cast value to ptr[T]
        
        Delegates to TypeConverter for unified type conversion logic.
        
        Args:
            visitor: AST visitor
            args: Pre-evaluated arguments (should be length 1)
            node: Original ast.Call node
        """
        if len(args) != 1:
            raise TypeError(f"ptr[T]() takes exactly 1 argument ({len(args)} given)")
        
        # Delegate to type converter, just like other types do
        value = args[0]
        
        try:
            result = visitor.type_converter.convert(value, cls)
            logger.debug("Typed pointer call conversion", value=value, result=result)
            return result
        except TypeError as e:
            raise TypeError(f"Cannot convert to {cls.get_name()}: {e}")
    
    def __init__(self, pointee_type=None, _runtime_data=None):
        """Support runtime pointer operations"""
        self.pointee_type = pointee_type
        self._runtime_data = _runtime_data
        self._fields = {}
        
        if pointee_type and hasattr(pointee_type, '__annotations__'):
            for field_name, field_type in pointee_type.__annotations__.items():
                self._fields[field_name] = None
    
    def __getattr__(self, name):
        """Support field access for struct pointers"""
        if name.startswith('_'):
            return super().__getattribute__(name)
        if name in self._fields:
            return self._fields[name]
        if (self.pointee_type and hasattr(self.pointee_type, '__annotations__') 
            and name in self.pointee_type.__annotations__):
            return None
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Support field assignment for struct pointers"""
        if name.startswith('_') or name in ['pointee_type']:
            super().__setattr__(name, value)
        else:
            if hasattr(self, '_fields') and name in self._fields:
                self._fields[name] = value
            elif (hasattr(self, 'pointee_type') and self.pointee_type 
                  and hasattr(self.pointee_type, '__annotations__') 
                  and name in self.pointee_type.__annotations__):
                if not hasattr(self, '_fields'):
                    self._fields = {}
                self._fields[name] = value
            else:
                super().__setattr__(name, value)
