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
    has_llvm_varargs = False  # Whether the function uses bare LLVM/C varargs
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
        """Get C ABI lowered LLVM function pointer type."""
        return ir.PointerType(cls.get_function_type(module_context))

    @classmethod
    def _pc_type_to_llvm(cls, pc_type, module_context=None):
        if hasattr(pc_type, 'get_llvm_type'):
            return pc_type.get_llvm_type(module_context)
        if isinstance(pc_type, ir.Type):
            logger.error(
                f"function type contains raw LLVM type {pc_type}. "
                f"This is a bug - use BuiltinEntity (i32, f64, etc.) instead.",
                node=None,
                exc_type=TypeError,
            )
        logger.error(
            f"Unknown function type component {pc_type}",
            node=None,
            exc_type=TypeError,
        )

    @classmethod
    def _get_user_signature(cls, module_context=None):
        if cls.param_types is None or cls.return_type is None:
            return [], ir.VoidType()
        param_llvm_types = [
            cls._pc_type_to_llvm(param_type, module_context)
            for param_type in cls.param_types
        ]
        return_llvm_type = cls._pc_type_to_llvm(cls.return_type, module_context)
        return param_llvm_types, return_llvm_type

    @classmethod
    def _lower_c_abi_signature(cls, param_llvm_types, return_llvm_type):
        from ..builder.abi import get_target_abi

        abi = get_target_abi()
        actual_param_types = []

        for param_type in param_llvm_types:
            if abi.is_aggregate_type(param_type):
                coercion = abi.classify_argument_type(param_type)
                if coercion.is_indirect:
                    actual_param_types.append(ir.PointerType(param_type))
                elif coercion.needs_coercion and coercion.coerced_type is not None:
                    actual_param_types.append(coercion.coerced_type)
                else:
                    actual_param_types.append(param_type)
            else:
                actual_param_types.append(param_type)

        actual_return_type = return_llvm_type
        if abi.is_aggregate_type(return_llvm_type):
            coercion = abi.classify_return_type(return_llvm_type)
            if coercion.is_indirect:
                actual_return_type = ir.VoidType()
                actual_param_types.insert(0, ir.PointerType(return_llvm_type))
            elif coercion.needs_coercion and coercion.coerced_type is not None:
                actual_return_type = coercion.coerced_type

        return actual_param_types, actual_return_type

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

        fn_type = getattr(func_ptr, 'function_type', None)
        is_varargs = bool(fn_type and getattr(fn_type, 'var_arg', False))

        # --- Normalize typed *args / **kwargs carriers ---
        # This is the single convergence point for ALL call paths.
        # If the caller passed raw positional args to a function with typed
        # collectors, pack them into pc_tuple / pc_dict here.
        if not is_varargs and len(args) != len(cls.param_types):
            if cls.has_varargs or cls.has_kwargs:
                from ..call_normalization import normalize_typed_collectors
                args = normalize_typed_collectors(
                    args, cls.param_types,
                    has_varargs=cls.has_varargs,
                    has_kwargs=cls.has_kwargs,
                )

        if not is_varargs and len(args) != len(cls.param_types):
            logger.error(
                f"Function expects {len(cls.param_types)} arguments, got {len(args)}",
                node=node,
                exc_type=TypeError,
            )

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
        """Get the C ABI lowered LLVM function type (not pointer)."""
        param_llvm_types, return_llvm_type = cls._get_user_signature(module_context)
        actual_param_types, actual_return_type = cls._lower_c_abi_signature(
            param_llvm_types, return_llvm_type,
        )
        return ir.FunctionType(
            actual_return_type,
            actual_param_types,
            var_arg=cls.has_llvm_varargs,
        )

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
                "func return type must be a type, not a sequence literal",
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
                    "func parameter type must be a type, not a sequence literal",
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
