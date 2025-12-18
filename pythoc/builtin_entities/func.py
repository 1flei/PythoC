from llvmlite import ir
import builtins
from .base import BuiltinType
from ..logger import logger

# Function type
class func(BuiltinType):
    """Function type - function pointer, supports func[[T1, T2], RetType] syntax"""
    _size_bytes = 8  # Function pointer is 8 bytes (64-bit)
    _is_signed = False
    _is_pointer = True
    param_types = None   # Tuple of parameter types (T1, T2, ...)
    return_type = None   # Return type
    
    @classmethod
    def get_name(cls) -> str:
        if cls.param_types is not None and cls.return_type is not None:
            # Format: func[[i32, i32], i32]
            param_names = []
            for t in cls.param_types:
                if hasattr(t, 'get_name'):
                    param_names.append(t.get_name())
                elif hasattr(t, '__name__'):
                    param_names.append(t.__name__)
                else:
                    param_names.append(str(t))
            
            if hasattr(cls.return_type, 'get_name'):
                ret_name = cls.return_type.get_name()
            elif hasattr(cls.return_type, '__name__'):
                ret_name = cls.return_type.__name__
            else:
                ret_name = str(cls.return_type)
            
            return f'func[[{", ".join(param_names)}], {ret_name}]'
        return 'func'
    
    @classmethod
    def get_llvm_type(cls, module_context=None) -> ir.Type:
        """Get LLVM function pointer type"""
        if cls.param_types is None or cls.return_type is None:
            # Default to void()*
            func_type = ir.FunctionType(ir.VoidType(), [])
            return ir.PointerType(func_type)
        
        # Get parameter LLVM types
        param_llvm_types = []
        for param_type in cls.param_types:
            if hasattr(param_type, 'get_llvm_type'):
                # All PC types now accept module_context parameter uniformly
                param_llvm_types.append(param_type.get_llvm_type(module_context))
            elif isinstance(param_type, ir.Type):
                # ANTI-PATTERN: param_type should be BuiltinEntity, not ir.Type
                logger.error(
                    f"function param type is raw LLVM type {param_type}. "
                    f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead.",
                    node=None, exc_type=TypeError)
            else:
                logger.error(f"Unknown function param type {param_type}", node=None, exc_type=TypeError)
        
        # Get return LLVM type
        if hasattr(cls.return_type, 'get_llvm_type'):
            # All PC types now accept module_context parameter uniformly
            return_llvm_type = cls.return_type.get_llvm_type(module_context)
        elif isinstance(cls.return_type, ir.Type):
            # ANTI-PATTERN: return_type should be BuiltinEntity, not ir.Type
            logger.error(
                f"function return type is raw LLVM type {cls.return_type}. "
                f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead.",
                node=None, exc_type=TypeError)
        else:
            logger.error(f"Unknown function return type {cls.return_type}", node=None, exc_type=TypeError)
        
        # Create function type and return pointer to it
        func_type = ir.FunctionType(return_llvm_type, param_llvm_types)
        return ir.PointerType(func_type)
    
    @classmethod
    def get_ctypes_type(cls):
        """Get ctypes function pointer type for FFI.
        
        Returns CFUNCTYPE for typed function pointers, c_void_p for untyped.
        """
        import ctypes
        
        if cls.param_types is None or cls.return_type is None:
            return ctypes.c_void_p
        
        # Get return ctypes type
        if hasattr(cls.return_type, 'get_ctypes_type'):
            ret_ctype = cls.return_type.get_ctypes_type()
        else:
            ret_ctype = None
        
        # Get parameter ctypes types
        param_ctypes = []
        for param_type in cls.param_types:
            if hasattr(param_type, 'get_ctypes_type'):
                param_ctypes.append(param_type.get_ctypes_type())
            else:
                param_ctypes.append(ctypes.c_void_p)
        
        return ctypes.CFUNCTYPE(ret_ctype, *param_ctypes)
    
    @classmethod
    def can_be_called(cls) -> bool:
        return True  # func type can be called (represents function pointers)
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node):
        """Handle function pointer call
        
        This is called when we have a func-typed value that needs to be called.
        The value can be:
        1. A function pointer variable (alloca) - need to load it first
        2. A direct function pointer value (from expression)
        
        Args:
            visitor: AST visitor
            func_ref: ValueRef of the function pointer
            args: Pre-evaluated arguments (actual call arguments)
            node: ast.Call node
        
        Returns:
            ValueRef with call result
        """
        from ..valueref import ensure_ir, wrap_value, get_type
        
        # Get the function pointer value from func_ref
        func_value = func_ref
        
        # Check if we need to load the function pointer
        # If func_value.value is an alloca, we need to load it
        func_ptr_ir = func_value.value
        if isinstance(func_ptr_ir, ir.AllocaInstr):
            # This is a function pointer variable, load it
            func_ptr = visitor.builder.load(func_ptr_ir)
        else:
            # This is already a function pointer value
            func_ptr = ensure_ir(func_value)
        
        # Get parameter and return types from this func type
        if cls.param_types is None or cls.return_type is None:
            logger.error(f"Function pointer has incomplete type", node=node, exc_type=TypeError)
        
        # Convert PC types to LLVM types
        param_llvm_types = []
        for pc_param_type in cls.param_types:
            if hasattr(pc_param_type, 'get_llvm_type'):
                # All PC types now accept module_context parameter uniformly
                param_llvm_types.append(pc_param_type.get_llvm_type(visitor.module.context))
            else:
                logger.error(f"Cannot get LLVM type from {pc_param_type}", node=node, exc_type=TypeError)
        
        # Type conversion for arguments if needed
        converted_args = []
        for idx, (arg, expected_type) in enumerate(zip(args, param_llvm_types)):
            target_pc_type = cls.param_types[idx]
            
            # Check if PC types match (including refined types)
            # Even if LLVM types match, PC types might be different (e.g., refined vs base)
            pc_types_match = (hasattr(arg, 'type_hint') and arg.type_hint == target_pc_type)
            
            if arg.is_python_value() or get_type(arg) != expected_type or not pc_types_match:
                # Convert using PC param type directly - this will enforce refined type checking
                converted = visitor.type_converter.convert(arg, target_pc_type, node)
                converted_args.append(ensure_ir(converted))
            else:
                converted_args.append(ensure_ir(arg))
        
        # Call the function pointer - pass return_type_hint and arg_type_hints for ABI coercion
        logger.debug(f"func.handle_call: calling {getattr(func_ptr, 'name', func_ptr)}, args={len(converted_args)}, return_type={cls.return_type}")
        logger.debug(f"func.handle_call: func_ptr.function_type={func_ptr.function_type}")
        logger.debug(f"func.handle_call: converted_args types={[str(a.type) for a in converted_args]}")
        result = visitor.builder.call(func_ptr, converted_args, return_type_hint=cls.return_type, arg_type_hints=cls.param_types)
        
        # Wrap result with return type hint (tracking happens in visit_expression)
        return wrap_value(result, kind="value", type_hint=cls.return_type)
    
    @classmethod
    def get_function_type(cls, module_context=None):
        """Get the underlying LLVM function type (not pointer)"""
        if cls.param_types is None or cls.return_type is None:
            return ir.FunctionType(ir.VoidType(), [])
        
        # Get parameter LLVM types
        param_llvm_types = []
        for param_type in cls.param_types:
            if hasattr(param_type, 'get_llvm_type'):
                # All PC types now accept module_context parameter uniformly
                param_llvm_types.append(param_type.get_llvm_type(module_context))
            elif isinstance(param_type, ir.Type):
                # ANTI-PATTERN: param_type should be BuiltinEntity, not ir.Type
                logger.error(
                    f"function param type is raw LLVM type {param_type}. "
                    f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead.",
                    node=None, exc_type=TypeError)
            else:
                logger.error(f"Unknown function param type {param_type}", node=None, exc_type=TypeError)
        
        # Get return LLVM type
        if hasattr(cls.return_type, 'get_llvm_type'):
            # All PC types now accept module_context parameter uniformly
            return_llvm_type = cls.return_type.get_llvm_type(module_context)
        elif isinstance(cls.return_type, ir.Type):
            # ANTI-PATTERN: return_type should be BuiltinEntity, not ir.Type
            logger.error(
                f"function return type is raw LLVM type {cls.return_type}. "
                f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead.",
                node=None, exc_type=TypeError)
        else:
            logger.error(f"Unknown function return type {cls.return_type}", node=None, exc_type=TypeError)
        
        return ir.FunctionType(return_llvm_type, param_llvm_types)
    
    @classmethod
    def normalize_subscript_items(cls, items):
        """Override normalization to flatten legacy func[[params], ret] -> func[params..., ret]
        
        Steps:
        1) Use base normalization to get pairs: ((None, [p1,p2]) , (None, ret))
        2) If first item is an unnamed list/tuple, expand into ((None, p1), (None, p2), ..., (None, ret))
        """
        import builtins
        base_normalized = super().normalize_subscript_items(items)
        if (isinstance(base_normalized, builtins.tuple) and len(base_normalized) >= 2 and
            base_normalized[0][0] is None and isinstance(base_normalized[0][1], (builtins.list, builtins.tuple))):
            legacy_params = base_normalized[0][1]
            expanded_params = tuple((None, p) for p in legacy_params)
            # keep the rest items (including return type) after expansion
            return expanded_params + base_normalized[1:]
        return base_normalized

    @classmethod
    def handle_type_subscript(cls, items):
        """Handle type subscript with normalized items for func
        
        Args:
            items: Normalized tuple of (Optional[str], type), with the LAST item as return type
        
        Returns:
            func subclass with param_types, param_names, and return_type set
        """
        import builtins
        if not isinstance(items, builtins.tuple):
            items = (items,)
        if len(items) == 0:
            logger.error("func requires at least a return type: func[return_type]", node=None, exc_type=TypeError)
        # Last item is return type
        *param_items, ret_item = items
        ret_name_opt, return_type = ret_item
        # Parse parameters
        param_types = []
        param_names = []
        has_any_name = False
        for name_opt, ptype in param_items:
            param_types.append(ptype)
            param_names.append(name_opt)
            if name_opt is not None:
                has_any_name = True
        if not has_any_name:
            param_names = None
        # Build type name
        param_strs = []
        for i, ptype in enumerate(param_types):
            pname = param_names[i] if param_names else None
            type_str = getattr(ptype, 'get_name', lambda: str(ptype))()
            if pname:
                param_strs.append(f"{pname}: {type_str}")
            else:
                param_strs.append(type_str)
        ret_str = getattr(return_type, 'get_name', lambda: str(return_type))()
        type_name = f'func[{", ".join(param_strs + [ret_str])}]' if param_strs else f'func[{ret_str}]'
        return type(
            type_name,
            (cls,),
            {
                'param_types': builtins.tuple(param_types),
                'param_names': param_names,
                'return_type': return_type,
            }
        )
    
    def __class_getitem__(cls, item):
        """Python runtime entry point using normalization -> handle_type_subscript"""
        normalized = cls.normalize_subscript_items(item)
        return cls.handle_type_subscript(normalized)