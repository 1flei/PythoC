"""
LLVM IR Builder - wraps llvmlite's ir.IRBuilder.

This is the default code generation backend that produces LLVM IR.
It handles ABI coercion for struct returns and calls transparently.
"""

from typing import Any, List, Optional, Callable
from llvmlite import ir

from .abstract import AbstractBuilder
from ..logger import logger


class FunctionWrapper:
    """Wrapper around ir.Function that hides ABI details (like sret offset).
    
    This allows upper layers to access user-defined parameters without
    knowing about sret pointers or other ABI-specific arguments.
    """
    
    def __init__(self, ir_function: ir.Function, sret_info: Optional[dict] = None,
                 param_coercion_info: Optional[dict] = None,
                 apply_c_abi: bool = True):
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
    
    def get_user_arg_unpacked(self, index: int, ir_builder: ir.IRBuilder) -> Any:
        """Get user-defined argument, unpacking ABI coercion if needed.
        
        Args:
            index: User parameter index
            ir_builder: IR builder for generating unpack instructions
            
        Returns:
            Tuple of (value, original_type) where value is the unpacked struct
            or the original argument if no coercion was applied.
        """
        arg = self.get_user_arg(index)
        coercion = self._param_coercion_info.get(index)
        
        if coercion:
            from ..abi.coercion import unpack_coerced_parameter
            original_type = coercion['original_type']
            coerced_type = coercion['coerced_type']
            is_byval = coercion['is_byval']
            
            unpacked = unpack_coerced_parameter(
                ir_builder, arg, original_type, coerced_type, is_byval
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


class LLVMBuilder(AbstractBuilder):
    """LLVM IR code generation backend.
    
    Wraps llvmlite's ir.IRBuilder to implement the AbstractBuilder interface.
    Handles ABI coercion for struct returns and calls transparently.
    
    ABI Handling:
    - ret(): Automatically coerces struct returns according to target ABI
    - call(): Automatically unpacks coerced struct returns
    
    The frontend doesn't need to know about ABI details - just call ret(value)
    and call(fn, args) and the builder handles the rest.
    """
    
    def __init__(self, ir_builder: ir.IRBuilder = None):
        """Initialize with an optional ir.IRBuilder.
        
        Args:
            ir_builder: llvmlite IRBuilder instance. Can be set later.
        """
        self._builder = ir_builder
        # ABI context for current function (set by set_return_abi_context)
        self._return_coercion = None
        self._sret_info = None
        self._current_function = None
        self._apply_c_abi = True
    
    @property
    def ir_builder(self) -> ir.IRBuilder:
        """Get the underlying ir.IRBuilder."""
        return self._builder
    
    @ir_builder.setter
    def ir_builder(self, builder: ir.IRBuilder):
        """Set the underlying ir.IRBuilder."""
        self._builder = builder
    
    # ========== ABI Context Management ==========
    
    def set_return_abi_context(self, current_function: ir.Function,
                               sret_info: Optional[dict] = None,
                               apply_c_abi: bool = True):
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
        self._return_coercion = None
        
        # Pre-compute return coercion if needed
        if current_function is not None:
            ret_type = current_function.return_value.type
            # If we have sret_info, the original type is stored there
            if sret_info and sret_info.get('uses_sret'):
                original_type = sret_info.get('original_return_type')
                if original_type:
                    from ..abi import get_target_abi
                    abi = get_target_abi()
                    if abi.is_aggregate_type(original_type):
                        self._return_coercion = abi.classify_return_type(original_type)
    
    def clear_return_abi_context(self):
        """Clear the ABI context (call when leaving a function)."""
        self._current_function = None
        self._sret_info = None
        self._return_coercion = None
    
    # ========== Function Declaration ==========
    
    def declare_function(self, module: ir.Module, name: str,
                        param_types: List[ir.Type], return_type: ir.Type,
                        var_arg: bool = False,
                        existing_func: Optional[ir.Function] = None,
                        apply_c_abi: bool = True) -> FunctionWrapper:
        """Declare a function with ABI handling for struct returns and arguments.
        
        This method handles ABI coercion transparently:
        - If return_type is a struct that requires sret, the function signature
          is modified to add sret pointer as first parameter
        - If param_types contain structs, they are coerced according to ABI
        - Returns a FunctionWrapper that hides these details from upper layers
        
        Args:
            module: The LLVM module to declare the function in
            name: Function name
            param_types: List of user-defined parameter LLVM types
            return_type: User-defined return LLVM type
            var_arg: Whether the function is variadic
            existing_func: Optional existing forward declaration to reuse
            apply_c_abi: Whether to apply C ABI coercion (default True)
        
        Returns:
            FunctionWrapper that provides access to user parameters
        """
        actual_return_type = return_type
        uses_sret = False
        sret_type = None
        actual_param_types = list(param_types)
        byval_params = []
        param_coercion_info = {}  # Track parameter coercion for unpacking
        
        if apply_c_abi:
            from ..abi import get_target_abi
            abi = get_target_abi()
            
            actual_param_types = []
            
            # Coerce aggregate parameters (struct or array)
            for i, pt in enumerate(param_types):
                if abi.is_aggregate_type(pt):
                    coercion = abi.classify_argument_type(pt)
                    if coercion.is_indirect:
                        # byval: pass as pointer with byval attribute
                        actual_param_types.append(ir.PointerType(pt))
                        byval_params.append((i, pt))
                        param_coercion_info[i] = {
                            'original_type': pt,
                            'coerced_type': ir.PointerType(pt),
                            'is_byval': True
                        }
                    elif coercion.needs_coercion and coercion.coerced_type is not None:
                        actual_param_types.append(coercion.coerced_type)
                        param_coercion_info[i] = {
                            'original_type': pt,
                            'coerced_type': coercion.coerced_type,
                            'is_byval': False
                        }
                    else:
                        actual_param_types.append(pt)
                else:
                    actual_param_types.append(pt)
            
            if abi.is_aggregate_type(return_type):
                coercion = abi.classify_return_type(return_type)
                if coercion.is_indirect:
                    # Use sret: add pointer parameter, return void
                    uses_sret = True
                    sret_type = return_type
                    actual_return_type = ir.VoidType()
                    # Insert sret pointer as first parameter
                    actual_param_types.insert(0, ir.PointerType(return_type))
                elif coercion.needs_coercion and coercion.coerced_type is not None:
                    actual_return_type = coercion.coerced_type
        
        # Create function type and declaration
        func_type = ir.FunctionType(actual_return_type, actual_param_types, var_arg=var_arg)
        
        if existing_func is not None and isinstance(existing_func, ir.Function):
            from ..logger import logger
            logger.debug(f"declare_function: using existing_func {name}, existing type={existing_func.type}, new type={func_type}")
            ir_function = existing_func
        else:
            from ..logger import logger
            logger.debug(f"declare_function: creating new function {name}, type={func_type}")
            ir_function = ir.Function(module, func_type, name)
        
        # Add sret attribute to first parameter if using sret
        if uses_sret:
            ir_function.args[0].add_attribute('sret')
            ir_function.args[0].add_attribute('noalias')
        
        # Add byval attribute to indirect struct parameters
        # Account for sret offset if present
        sret_offset = 1 if uses_sret else 0
        for orig_idx, byval_type in byval_params:
            arg_idx = orig_idx + sret_offset
            ir_function.args[arg_idx].add_attribute('byval')
        
        # Build sret_info
        sret_info = None
        if uses_sret:
            sret_info = {
                'uses_sret': True,
                'sret_type': sret_type,
                'sret_ptr_arg_index': 0
            }
        
        return FunctionWrapper(ir_function, sret_info, param_coercion_info, apply_c_abi)
    
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
                                    func_wrapper.apply_c_abi)
        
        return entry_block
    
    # ========== Arithmetic Operations ==========
    
    def add(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.add(lhs, rhs, name=name)
    
    def sub(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.sub(lhs, rhs, name=name)
    
    def mul(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.mul(lhs, rhs, name=name)
    
    def sdiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.sdiv(lhs, rhs, name=name)
    
    def udiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.udiv(lhs, rhs, name=name)
    
    def srem(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.srem(lhs, rhs, name=name)
    
    def urem(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.urem(lhs, rhs, name=name)
    
    def fadd(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fadd(lhs, rhs, name=name)
    
    def fsub(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fsub(lhs, rhs, name=name)
    
    def fmul(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fmul(lhs, rhs, name=name)
    
    def fdiv(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fdiv(lhs, rhs, name=name)
    
    # ========== Bitwise Operations ==========
    
    def and_(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.and_(lhs, rhs, name=name)
    
    def or_(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.or_(lhs, rhs, name=name)
    
    def xor(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.xor(lhs, rhs, name=name)
    
    def shl(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.shl(lhs, rhs, name=name)
    
    def ashr(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.ashr(lhs, rhs, name=name)
    
    def lshr(self, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.lshr(lhs, rhs, name=name)
    
    # ========== Comparison Operations ==========
    
    def icmp_signed(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.icmp_signed(op, lhs, rhs, name=name)
    
    def icmp_unsigned(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.icmp_unsigned(op, lhs, rhs, name=name)
    
    def fcmp_ordered(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fcmp_ordered(op, lhs, rhs, name=name)
    
    def fcmp_unordered(self, op: str, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.fcmp_unordered(op, lhs, rhs, name=name)
    
    # ========== Memory Operations ==========
    
    def alloca(self, typ: Any, size: Any = None, name: str = "") -> Any:
        return self._builder.alloca(typ, size=size, name=name)
    
    def load(self, ptr: Any, name: str = "", align: Any = None) -> Any:
        return self._builder.load(ptr, name=name, align=align)
    
    def store(self, value: Any, ptr: Any, align: Any = None) -> Any:
        return self._builder.store(value, ptr, align=align)
    
    def gep(self, ptr: Any, indices: List[Any], inbounds: bool = False, name: str = "") -> Any:
        return self._builder.gep(ptr, indices, inbounds=inbounds, name=name)
    
    # ========== Type Conversion ==========
    
    def trunc(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.trunc(value, typ, name=name)
    
    def zext(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.zext(value, typ, name=name)
    
    def sext(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.sext(value, typ, name=name)
    
    def fptrunc(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.fptrunc(value, typ, name=name)
    
    def fpext(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.fpext(value, typ, name=name)
    
    def fptosi(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.fptosi(value, typ, name=name)
    
    def fptoui(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.fptoui(value, typ, name=name)
    
    def sitofp(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.sitofp(value, typ, name=name)
    
    def uitofp(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.uitofp(value, typ, name=name)
    
    def ptrtoint(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.ptrtoint(value, typ, name=name)
    
    def inttoptr(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.inttoptr(value, typ, name=name)
    
    def bitcast(self, value: Any, typ: Any, name: str = "") -> Any:
        return self._builder.bitcast(value, typ, name=name)
    
    # ========== Control Flow ==========
    
    def branch(self, target: Any) -> Any:
        return self._builder.branch(target)
    
    def cbranch(self, cond: Any, truebr: Any, falsebr: Any) -> Any:
        return self._builder.cbranch(cond, truebr, falsebr)
    
    def switch(self, value: Any, default: Any) -> Any:
        return self._builder.switch(value, default)
    
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
        
        value_type = value.type
        
        # Apply ABI coercion for aggregate returns (struct or array)
        if not self._apply_c_abi:
            return self._builder.ret(value)
        
        # Check if this is an aggregate type that needs coercion
        from ..abi import get_target_abi
        from ..abi.coercion import pack_struct_for_return
        
        abi = get_target_abi()
        if not abi.is_aggregate_type(value_type):
            return self._builder.ret(value)
        
        # Get ABI coercion info
        coercion = abi.classify_return_type(value_type)
        
        if not coercion.needs_coercion:
            return self._builder.ret(value)
        
        if coercion.is_indirect:
            # sret: store to the sret pointer and return void
            if self._sret_info and self._sret_info.get('uses_sret'):
                sret_ptr = self._current_function.args[self._sret_info['sret_ptr_arg_index']]
                self._builder.store(value, sret_ptr)
                return self._builder.ret_void()
            else:
                logger.debug(f"Struct return requires sret but sret_info not found")
                return self._builder.ret(value)
        
        # Apply coercion: pack struct into coerced type
        coerced_value = pack_struct_for_return(self._builder, value, coercion)
        return self._builder.ret(coerced_value)
    
    def ret_void(self) -> Any:
        return self._builder.ret_void()
    
    def unreachable(self) -> Any:
        return self._builder.unreachable()
    
    # ========== PHI and Select ==========
    
    def phi(self, typ: Any, name: str = "") -> Any:
        return self._builder.phi(typ, name=name)
    
    def select(self, cond: Any, lhs: Any, rhs: Any, name: str = "") -> Any:
        return self._builder.select(cond, lhs, rhs, name=name)
    
    # ========== Function Calls ==========
    
    def call(self, fn: Any, args: List[Any], name: str = "",
             return_type_hint: Any = None, arg_type_hints: List[Any] = None) -> Any:
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
        logger.debug(f"LLVMBuilder.call: fn={getattr(fn, 'name', fn)}, return_type_hint={return_type_hint}, args={len(args)}")
        
        # Get module context
        if hasattr(fn, 'module') and fn.module:
            module_context = fn.module.context
        elif self._builder and self._builder.block:
            module_context = self._builder.block.function.module.context
        else:
            module_context = None
        
        # Apply ABI coercion to aggregate arguments (struct or array)
        coerced_args = list(args)
        if arg_type_hints and module_context:
            from ..abi import get_target_abi
            from ..abi.coercion import pack_struct_for_argument
            abi = get_target_abi()
            
            for i, (arg, hint) in enumerate(zip(args, arg_type_hints)):
                if hint is not None and hasattr(hint, 'get_llvm_type'):
                    arg_llvm_type = hint.get_llvm_type(module_context)
                    if abi.is_aggregate_type(arg_llvm_type):
                        coercion = abi.classify_argument_type(arg_llvm_type)
                        if coercion.needs_coercion:
                            # Coerce aggregate argument (both coerced and indirect)
                            coerced_args[i] = pack_struct_for_argument(self._builder, arg, coercion)
                            logger.debug(f"LLVMBuilder.call: coerced arg {i} from {arg.type} to {coerced_args[i].type}")
        
        # Check if we need to handle sret (indirect aggregate return)
        if return_type_hint is not None and hasattr(return_type_hint, 'get_llvm_type') and module_context:
            agg_type = return_type_hint.get_llvm_type(module_context)
            
            from ..abi import get_target_abi
            abi = get_target_abi()
            
            # Check if it's an aggregate type that needs ABI handling
            if abi.is_aggregate_type(agg_type):
                coercion = abi.classify_return_type(agg_type)
                
                if coercion.is_indirect:
                    # sret: allocate buffer, pass as first arg, load result
                    logger.debug(f"LLVMBuilder.call: using sret for {getattr(fn, 'name', fn)}, agg_type={agg_type}")
                    sret_buf = self._builder.alloca(agg_type, name="sret.buf")
                    call_args = [sret_buf] + coerced_args
                    logger.debug(f"LLVMBuilder.call: sret call_args={len(call_args)}, fn.function_type={fn.function_type}")
                    self._builder.call(fn, call_args, name=name)
                    # Load and return the aggregate value
                    return self._builder.load(sret_buf, name="sret.load")
                
                elif coercion.needs_coercion:
                    # Coerced return: call and unpack
                    call_result = self._builder.call(fn, coerced_args, name=name)
                    from ..abi.coercion import unpack_struct_from_return
                    return unpack_struct_from_return(self._builder, call_result, coercion)
        
        # Default: direct call
        call_result = self._builder.call(fn, coerced_args, name=name)
        return call_result
    
    # ========== Aggregate Operations ==========
    
    def extract_value(self, agg: Any, idx: Any, name: str = "") -> Any:
        return self._builder.extract_value(agg, idx, name=name)
    
    def insert_value(self, agg: Any, value: Any, idx: Any, name: str = "") -> Any:
        return self._builder.insert_value(agg, value, idx, name=name)
    
    # ========== Block Management ==========
    
    @property
    def block(self) -> Any:
        return self._builder.block
    
    def position_at_start(self, block: Any) -> None:
        self._builder.position_at_start(block)
    
    def position_at_end(self, block: Any) -> None:
        self._builder.position_at_end(block)
    
    def position_before(self, instr: Any) -> None:
        self._builder.position_before(instr)
    
    def position_after(self, instr: Any) -> None:
        self._builder.position_after(instr)
    
    # ========== Misc ==========
    
    def neg(self, value: Any, name: str = "") -> Any:
        return self._builder.neg(value, name=name)
    
    def fneg(self, value: Any, name: str = "") -> Any:
        return self._builder.fneg(value, name=name)
    
    def not_(self, value: Any, name: str = "") -> Any:
        return self._builder.not_(value, name=name)
