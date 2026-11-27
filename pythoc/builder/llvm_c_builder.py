"""
LLVM IR Builder with C ABI support - extends LLVMBuilder with ABI coercion.

This builder produces LLVM IR with C ABI-compatible function signatures.
It handles ABI coercion for struct returns and calls transparently,
making it suitable for interop with C code.

Use this builder when you need C-compatible calling conventions.
Use LLVMBuilder for pure LLVM IR without ABI coercion.
"""

from typing import Any, List, Optional
from llvmlite import ir

from .llvm_simple_builder import LLVMBuilder
from ..logger import logger


class FunctionWrapper:
    """Wrapper around ir.Function that hides ABI details (like sret offset).
    
    This allows upper layers to access user-defined parameters without
    knowing about sret pointers or other ABI-specific arguments.
    """
    
    def __init__(self, ir_function: ir.Function, sret_info: Optional[dict] = None,
                 param_coercion_info: Optional[dict] = None,
                 apply_c_abi: bool = True, return_coercion=None):
        """
        Args:
            ir_function: The underlying LLVM function
            sret_info: Optional sret info dict with:
                - 'uses_sret': bool
                - 'sret_type': ir.Type (original struct type)
            param_coercion_info: Optional dict mapping user param index to coercion info:
                - Each entry: {'original_type': ir.Type, 'coerced_type': ir.Type, 'is_byval': bool}
            apply_c_abi: Whether this function uses C ABI (default True)
        """
        self._ir_function = ir_function
        self._sret_info = sret_info
        self._param_coercion_info = param_coercion_info or {}
        self.return_coercion = return_coercion
        self._arg_offset = 1 if (sret_info and sret_info.get('uses_sret')) else 0
        self._apply_c_abi = apply_c_abi
    
    @property
    def ir_function(self) -> ir.Function:
        """Get the underlying ir.Function."""
        return self._ir_function
    
    @property
    def sret_info(self) -> Optional[dict]:
        """Get sret info for this function."""
        return self._sret_info
    
    @property
    def param_coercion_info(self) -> dict:
        """Get parameter coercion info for this function."""
        return self._param_coercion_info
    
    @property
    def apply_c_abi(self) -> bool:
        """Whether this function uses C ABI."""
        return self._apply_c_abi
    
    def get_user_arg(self, index: int) -> ir.Argument:
        """Get user-defined argument by index (hiding sret offset).
        
        Note: For coerced struct parameters, this returns the coerced value.
        Use get_user_arg_unpacked() to get the unpacked struct value.
        """
        return self._ir_function.args[index + self._arg_offset]
    
    def get_user_arg_unpacked(self, index: int, builder) -> Any:
        """Get user-defined argument, unpacking ABI coercion if needed.

        Args:
            index: User parameter index
            builder: Builder for generating unpack instructions (ir.IRBuilder or
                     ControlFlowBuilder - any object with load/alloca/store/bitcast)

        Returns:
            Tuple of (value, original_type) where value is the unpacked struct
            or the original argument if no coercion was applied.
        """
        arg = self.get_user_arg(index)
        coercion = self._param_coercion_info.get(index)

        if coercion:
            from .abi.coercion import unpack_coerced_parameter
            original_type = coercion['original_type']
            coerced_type = coercion['coerced_type']
            is_byval = coercion['is_byval']

            unpacked = unpack_coerced_parameter(
                builder, arg, original_type, coerced_type, is_byval
            )
            return unpacked, original_type
        else:
            return arg, arg.type
    
    def get_param_coercion(self, index: int) -> Optional[dict]:
        """Get coercion info for a user parameter, or None if not coerced."""
        return self._param_coercion_info.get(index)
    
    @property
    def user_arg_count(self) -> int:
        """Get the number of user-defined arguments (excluding sret)."""
        return len(self._ir_function.args) - self._arg_offset
    
    def append_basic_block(self, name: str = "") -> ir.Block:
        """Append a basic block to the function."""
        return self._ir_function.append_basic_block(name)
    
    # Delegate common attributes to ir_function
    @property
    def name(self) -> str:
        return self._ir_function.name
    
    @property
    def return_value(self):
        return self._ir_function.return_value
    
    @property
    def args(self):
        """Raw LLVM args - use get_user_arg() for user parameters."""
        return self._ir_function.args


class LLVMCBuilder(LLVMBuilder):
    """LLVM IR code generation backend with C ABI support.
    
    Extends LLVMBuilder with ABI coercion for struct returns and calls.
    
    ABI Handling:
    - ret(): Automatically coerces struct returns according to target ABI
    - call(): Automatically unpacks coerced struct returns
    - declare_function(): Creates ABI-compatible function signatures
    
    The frontend doesn't need to know about ABI details - just call ret(value)
    and call(fn, args) and the builder handles the rest.
    """
    
    def __init__(self, ir_builder: ir.IRBuilder = None):
        """Initialize with an optional ir.IRBuilder.
        
        Args:
            ir_builder: llvmlite IRBuilder instance. Can be set later.
        """
        super().__init__(ir_builder)
        # ABI context for current function (set by set_return_abi_context)
        self._return_coercion = None
        self._sret_info = None
        self._current_function = None
        self._apply_c_abi = True
    
    # ========== ABI Context Management ==========
    
    def set_return_abi_context(self, current_function: ir.Function,
                               sret_info: Optional[dict] = None,
                               apply_c_abi: bool = True, return_coercion=None):
        """Set ABI context for the current function.
        
        This should be called after creating the function entry block.
        The builder will use this context to handle struct returns correctly.
        
        Args:
            current_function: The current ir.Function being compiled
            sret_info: Optional sret info dict with:
                - 'uses_sret': bool
                - 'sret_ptr_arg_index': int (index of sret pointer in args)
                - 'original_return_type': ir.Type
            apply_c_abi: Whether this function uses C ABI (default True)
        """
        self._current_function = current_function
        self._sret_info = sret_info
        self._apply_c_abi = apply_c_abi
        self._return_coercion = return_coercion

    def clear_return_abi_context(self):
        """Clear the ABI context (call when leaving a function)."""
        self._current_function = None
        self._sret_info = None
        self._return_coercion = None
    
    # ========== Function Declaration ==========
    
    def declare_function(self, module: ir.Module, name: str,
                        param_types, return_type,
                        var_arg: bool = False,
                        existing_func: Optional[ir.Function] = None,
                        apply_c_abi: bool = True, *, is_extern: bool = False) -> FunctionWrapper:
        """Declare a function with ABI handling for struct returns and arguments.
        
        This method handles ABI coercion transparently:
        - If return_type is a struct that requires sret, the function signature
          is modified to add sret pointer as first parameter
        - If param_types contain structs, they are coerced according to ABI
        - Returns a FunctionWrapper that hides these details from upper layers
        
        Args:
            module: The LLVM module to declare the function in
            name: Function name
            param_types: Declared PythoC parameter types
            return_type: Declared PythoC return type
            var_arg: Whether the function is variadic
            existing_func: Optional existing forward declaration to reuse
            apply_c_abi: Set False to declare an explicit LLVM signature
            is_extern: Allow library semantics for a foreign symbol declaration
        
        Returns:
            FunctionWrapper that provides access to user parameters
        """
        return_coercion = None
        sret_info = None
        param_coercion_info = {}
        byval_indices = []
        if apply_c_abi:
            from .abi import get_target_abi

            abi = get_target_abi()
            signature = abi.classify_signature(param_types, return_type, module.context)
            func_type = signature.function_type(var_arg)
            return_coercion = signature.result
            if return_coercion.is_indirect:
                sret_info = {
                    'uses_sret': True,
                    'sret_type': return_coercion.original_type,
                    'sret_ptr_arg_index': 0,
                }
            for index, argument in enumerate(signature.arguments):
                if not argument.needs_coercion:
                    continue
                param_coercion_info[index] = {
                    'original_type': argument.original_type,
                    'coerced_type': argument.abi_type,
                    'is_byval': argument.is_indirect,
                }
                if argument.is_indirect and abi.uses_byval_for_indirect_args():
                    byval_indices.append(index + int(sret_info is not None))
        else:
            func_type = ir.FunctionType(return_type, param_types, var_arg=var_arg)

        if existing_func is not None and isinstance(existing_func, ir.Function):
            if existing_func.function_type != func_type:
                raise TypeError(
                    f"Conflicting function signature for '{name}': "
                    f"{existing_func.function_type} != {func_type}"
                )
            logger.debug(f"declare_function: using existing_func {name}, existing type={existing_func.type}, new type={func_type}")
            ir_function = existing_func
        else:
            logger.debug(f"declare_function: creating new function {name}, type={func_type}")
            ir_function = ir.Function(module, func_type, name)

        if not is_extern:
            ir_function.attributes.add('nobuiltin')
        if sret_info:
            ir_function.args[0].add_attribute('sret')
            ir_function.args[0].add_attribute('noalias')
        for index in byval_indices:
            ir_function.args[index].add_attribute('byval')

        return FunctionWrapper(
            ir_function, sret_info, param_coercion_info, apply_c_abi, return_coercion,
        )
    
    def setup_function_entry(self, func_wrapper: FunctionWrapper) -> ir.Block:
        """Set up function entry block and ABI context.
        
        Args:
            func_wrapper: FunctionWrapper from declare_function
        
        Returns:
            The entry basic block
        """
        entry_block = func_wrapper.append_basic_block('entry')
        ir_builder = ir.IRBuilder(entry_block)
        self._builder = ir_builder
        
        # Set ABI context
        self.set_return_abi_context(func_wrapper.ir_function, func_wrapper.sret_info,
                                    func_wrapper.apply_c_abi, func_wrapper.return_coercion)
        
        return entry_block
    
    # ========== Overridden Methods with ABI Coercion ==========
    
    def ret(self, value: Any = None) -> Any:
        """Return from function with automatic ABI coercion for aggregates.
        
        Handles:
        - sret (indirect): store to sret pointer and return void
        - coerced: pack aggregate into coerced type and return
        - direct: return value as-is
        
        Note: Aggregates include both struct types and array types (union uses array).
        """
        if value is None:
            return self._builder.ret_void()

        coercion = self._return_coercion
        if not self._apply_c_abi or coercion is None or not coercion.needs_coercion:
            return self._builder.ret(value)

        if coercion.is_indirect:
            sret_ptr = self._current_function.args[self._sret_info['sret_ptr_arg_index']]
            self._builder.store(value, sret_ptr)
            return self._builder.ret_void()

        from .abi.coercion import pack_struct_for_return
        coerced_value = pack_struct_for_return(self._builder, value, coercion)
        return self._builder.ret(coerced_value)
    
    def call(self, fn: Any, args: List[Any], name: str = "",
             return_type_hint: Any = None, arg_type_hints: List[Any] = None,
             cconv: Optional[str] = None) -> Any:
        """Call a function with automatic ABI coercion for struct returns and arguments.
        
        Handles:
        - sret (indirect): allocate buffer, pass as first arg, load result
        - coerced return: call and unpack coerced return value
        - coerced args: pack struct arguments according to ABI
        - direct: call and return as-is
        
        Args:
            fn: Function to call
            args: Arguments to pass
            name: Optional name for the result
            return_type_hint: Optional PC type hint for return value unpacking
            arg_type_hints: Optional list of PC type hints for argument coercion
        
        Returns:
            The call result, unpacked if struct coercion was applied
        """
        if not arg_type_hints and return_type_hint is None:
            return self._call(fn, args, name=name, cconv=cconv)

        from .abi import get_target_abi
        from .abi.coercion import (
            create_sret_alloca, pack_struct_for_argument, unpack_struct_from_return,
        )

        from ..builtin_entities.types import void

        module_context = self._builder.block.function.module.context
        return_type = return_type_hint if return_type_hint is not None else void
        abi = get_target_abi()
        signature = abi.classify_signature(
            arg_type_hints or (), return_type, module_context,
        )

        call_args = []
        arg_attrs = {}
        if signature.result.is_indirect:
            sret_buf = create_sret_alloca(self._builder, signature.result.original_type)
            call_args.append(sret_buf)
            arg_attrs[0] = ['sret', 'noalias']

        for arg, argument in zip(args, signature.arguments):
            if argument.is_indirect and abi.uses_byval_for_indirect_args():
                arg_attrs[len(call_args)] = ['byval']
            call_args.append(pack_struct_for_argument(self._builder, arg, argument))
        call_args.extend(args[len(signature.arguments):])
        result = self._call(
            fn, call_args, name=name, cconv=cconv, arg_attrs=arg_attrs or None,
        )
        if signature.result.is_indirect:
            return self._builder.load(sret_buf, name='sret.load')
        return unpack_struct_from_return(self._builder, result, signature.result)
