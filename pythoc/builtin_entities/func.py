from llvmlite import ir

from .base import BuiltinType
from ..logger import logger


# Function type
class func(BuiltinType):
    """Function type - function pointer, using func[param_types..., return_type]."""
    _size_bytes = 8  # Function pointer is 8 bytes (64-bit)
    _is_signed = False
    _is_pointer = True
    param_types = None   # Tuple of parameter types (T1, T2, ...)
    return_type = None   # Return type
    has_varargs = False  # Whether the function was declared with *args: T
    has_kwargs = False   # Whether the function was declared with **kwargs: T

    @classmethod
    def get_name(cls) -> str:
        if cls.param_types is not None and cls.return_type is not None:
            param_strs = []
            stored_param_names = cls.param_names or ()
            for index, param_type in enumerate(cls.param_types):
                if hasattr(param_type, 'get_name'):
                    type_name = param_type.get_name()
                elif hasattr(param_type, '__name__'):
                    type_name = param_type.__name__
                else:
                    type_name = str(param_type)

                param_name = stored_param_names[index] if index < len(stored_param_names) else None
                if param_name:
                    param_strs.append(f'{param_name}: {type_name}')
                else:
                    param_strs.append(type_name)

            if hasattr(cls.return_type, 'get_name'):
                ret_name = cls.return_type.get_name()
            elif hasattr(cls.return_type, '__name__'):
                ret_name = cls.return_type.__name__
            else:
                ret_name = str(cls.return_type)

            if param_strs:
                return f'func[{", ".join(param_strs)}, {ret_name}]'
            return f'func[{ret_name}]'
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
                    node=None,
                    exc_type=TypeError,
                )
            else:
                logger.error(
                    f"Unknown function param type {param_type}",
                    node=None,
                    exc_type=TypeError,
                )

        # Get return LLVM type
        if hasattr(cls.return_type, 'get_llvm_type'):
            # All PC types now accept module_context parameter uniformly
            return_llvm_type = cls.return_type.get_llvm_type(module_context)
        elif isinstance(cls.return_type, ir.Type):
            # ANTI-PATTERN: return_type should be BuiltinEntity, not ir.Type
            logger.error(
                f"function return type is raw LLVM type {cls.return_type}. "
                f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead.",
                node=None,
                exc_type=TypeError,
            )
        else:
            logger.error(
                f"Unknown function return type {cls.return_type}",
                node=None,
                exc_type=TypeError,
            )

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

        # Get the function pointer from func_ref
        # func_ref comes from visit_Name which already loads the alloca,
        # so func_ref.value is the loaded function pointer (not the alloca).
        # Use ensure_ir to get the underlying value (works with both ir.Value and VReg).
        func_ptr = ensure_ir(func_ref)

        # Get parameter and return types from this func type
        if cls.param_types is None or cls.return_type is None:
            logger.error("Function pointer has incomplete type", node=node, exc_type=TypeError)

        # Convert PC types to LLVM types
        param_llvm_types = []
        for pc_param_type in cls.param_types:
            if hasattr(pc_param_type, 'get_llvm_type'):
                # All PC types now accept module_context parameter uniformly
                param_llvm_types.append(pc_param_type.get_llvm_type(visitor.module.context))
            else:
                logger.error(
                    f"Cannot get LLVM type from {pc_param_type}",
                    node=node,
                    exc_type=TypeError,
                )

        # Pack positional args for *args: T functions.
        # The func type has a single varargs parameter of type T, but callers
        # may pass N individual args that should be packed into a pc_tuple
        # carrier and converted to T via the type conversion system.
        # This fires when args count doesn't match param count (excess, deficit,
        # or when the varargs slot isn't already a single carrier).
        if cls.has_varargs and len(args) != len(cls.param_types):
            from ..builtin_entities.pc_tuple import create_pc_tuple_type
            from ..builtin_entities.python_type import PythonType

            # Find varargs param index: last param before kwargs (if present)
            kwargs_count = 1 if cls.has_kwargs else 0
            varargs_idx = len(cls.param_types) - 1 - kwargs_count
            normal_count = varargs_idx

            normal_args = list(args[:normal_count])
            # Everything between normal args and kwargs (if any) is varargs material
            kwargs_args = list(args[len(args) - kwargs_count:]) if kwargs_count else []
            excess_args = list(args[normal_count:len(args) - kwargs_count if kwargs_count else len(args)])

            tuple_type = create_pc_tuple_type(excess_args)
            tuple_hint = PythonType.wrap(tuple_type, is_constant=True)
            tuple_ref = wrap_value(tuple_type, kind='python', type_hint=tuple_hint)
            args = normal_args + [tuple_ref] + kwargs_args

        # Type conversion for fixed arguments
        converted_args = []
        for idx, (arg, expected_type) in enumerate(zip(args, param_llvm_types)):
            target_pc_type = cls.param_types[idx]

            # Check if PC types match (including refined types)
            # Even if LLVM types match, PC types might be different (e.g., refined vs base)
            pc_types_match = hasattr(arg, 'type_hint') and arg.type_hint == target_pc_type

            if arg.is_python_value() or get_type(arg) != expected_type or not pc_types_match:
                # Convert using PC param type directly - this will enforce refined type checking
                converted = visitor.implicit_coercer.coerce(arg, target_pc_type, node)
                converted_args.append(ensure_ir(converted))
            else:
                converted_args.append(ensure_ir(arg))

        # For varargs functions, pass remaining arguments beyond the fixed
        # params. Apply C default argument promotions (i8/i16 -> i32, f32 -> f64).
        fn_type = getattr(func_ptr, 'function_type', None)
        is_varargs = fn_type and getattr(fn_type, 'var_arg', False)
        if is_varargs and len(args) > len(param_llvm_types):
            from . import i8 as pc_i8, i16 as pc_i16, i32 as pc_i32
            from . import f32 as pc_f32, f64 as pc_f64
            for extra_arg in args[len(param_llvm_types):]:
                hint = getattr(extra_arg, 'type_hint', None)
                # C default argument promotions for varargs
                if hint in (pc_i8, pc_i16):
                    promoted = visitor.type_converter.convert(extra_arg, pc_i32, node)
                    converted_args.append(ensure_ir(promoted))
                elif hint == pc_f32:
                    promoted = visitor.type_converter.convert(extra_arg, pc_f64, node)
                    converted_args.append(ensure_ir(promoted))
                else:
                    converted_args.append(ensure_ir(extra_arg))

        # Call the function pointer - pass return_type_hint and arg_type_hints for ABI coercion
        logger.debug(
            f"func.handle_call: calling {getattr(func_ptr, 'name', func_ptr)}, "
            f"args={len(converted_args)}, return_type={cls.return_type}"
        )
        logger.debug(
            f"func.handle_call: func_ptr.function_type={getattr(func_ptr, 'function_type', None)}"
        )
        logger.debug(
            f"func.handle_call: converted_args types="
            f"{[str(getattr(a, 'type', '?')) for a in converted_args]}"
        )
        result = visitor.builder.call(
            func_ptr,
            converted_args,
            return_type_hint=cls.return_type,
            arg_type_hints=cls.param_types,
        )

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
                    node=None,
                    exc_type=TypeError,
                )
            else:
                logger.error(
                    f"Unknown function param type {param_type}",
                    node=None,
                    exc_type=TypeError,
                )

        # Get return LLVM type
        if hasattr(cls.return_type, 'get_llvm_type'):
            # All PC types now accept module_context parameter uniformly
            return_llvm_type = cls.return_type.get_llvm_type(module_context)
        elif isinstance(cls.return_type, ir.Type):
            # ANTI-PATTERN: return_type should be BuiltinEntity, not ir.Type
            logger.error(
                f"function return type is raw LLVM type {cls.return_type}. "
                f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead.",
                node=None,
                exc_type=TypeError,
            )
        else:
            logger.error(
                f"Unknown function return type {cls.return_type}",
                node=None,
                exc_type=TypeError,
            )

        return ir.FunctionType(return_llvm_type, param_llvm_types)

    @classmethod
    def handle_type_subscript(cls, items):
        """Handle type subscript with normalized items for func

        Args:
            items: Normalized tuple of (Optional[str], type), with the LAST item as return type

        Returns:
            func subclass with param_types, param_names, and return_type set
        """
        import builtins
        from ..literal_protocol import is_sequence_carrier

        if not isinstance(items, builtins.tuple):
            items = (items,)
        if len(items) == 0:
            logger.error("func requires at least a return type: func[return_type]", node=None, exc_type=TypeError)

        # Last item is return type
        *param_items, ret_item = items
        _ret_name_opt, return_type = ret_item
        if is_sequence_carrier(return_type):
            logger.error(
                "Legacy func syntax is removed; use func[param_types..., return_type]",
                node=None,
                exc_type=TypeError,
            )

        # Parse parameters
        param_types = []
        param_names = []
        has_any_name = False
        for name_opt, ptype in param_items:
            if is_sequence_carrier(ptype):
                logger.error(
                    "Legacy func syntax is removed; use func[param_types..., return_type]",
                    node=None,
                    exc_type=TypeError,
                )
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
            },
        )

    def __class_getitem__(cls, item):
        """Python runtime entry point using normalization -> handle_type_subscript"""
        normalized = cls.normalize_subscript_items(item)
        return cls.handle_type_subscript(normalized)
