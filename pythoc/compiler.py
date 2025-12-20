"""
LLVM Compiler using llvmlite
Enhanced version with modular architecture and comprehensive functionality
"""

import ast
from typing import List
from llvmlite import ir, binding
from .ast_visitor import LLVMIRVisitor
from .registry import get_unified_registry, register_struct_from_class
from .type_resolver import TypeResolver
from .logger import logger

# Initialize LLVM
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

class LLVMCompiler:
    """Enhanced LLVM compiler using llvmlite"""
    
    def __init__(self, user_globals=None):
        self.module = None
        self.compiled_functions = []

        self.extern_functions = {}  # Registry for extern function declarations
        self.user_globals = user_globals or {}  # User code's global namespace
        self.create_module()

    def update_globals(self, user_globals):
        """Update user globals"""
        self.user_globals = user_globals
    
    def create_module(self, name: str = "main"):
        """Create a new LLVM module"""
        self.module = ir.Module(name=name)
        self.module.triple = binding.get_default_triple()
        # Set data layout for correct struct size calculation
        target = binding.Target.from_triple(self.module.triple)
        target_machine = target.create_target_machine()
        self.module.data_layout = target_machine.target_data
        self.extern_functions = {}  # Reset extern functions for new module
        self._declare_extern_functions()
        self._declare_imported_user_functions()
        return self.module
    
    
    def _declare_extern_functions(self):
        """Declare only the extern functions that are actually used in the code"""
        # This method is now called after compilation to only declare used functions
        # The actual declaration happens in _declare_extern_function when needed
        pass
    
    def _declare_imported_user_functions(self):
        """Declare user functions imported from other modules"""
        if not hasattr(self, 'imported_user_functions'):
            return
        
        from .registry import get_unified_registry
        
        for func_name, module_path in self.imported_user_functions.items():
            # Get function type hints from registry
            func_info = get_unified_registry().get_function_info(func_name)
            if not func_info:
                continue
            
            # Re-parse type annotations using current module's context
            # This ensures struct types use the current module's type IDs
            param_types = []
            if func_info.param_type_hints:
                for param_name, pc_type in func_info.param_type_hints.items():
                    llvm_type = self._recreate_type_in_context(pc_type)
                    param_types.append(llvm_type)
            
            # Get return type
            # return_type = ir.IntType(32)  # default
            if func_info.return_type_hint:
                return_type = self._recreate_type_in_context(func_info.return_type_hint)
            
            # Declare function with proper ABI handling
            if func_name not in self.module.globals:
                from .builder import LLVMBuilder
                temp_builder = LLVMBuilder()
                func_wrapper = temp_builder.declare_function(
                    self.module, func_name,
                    param_types, return_type
                )
    
    def _recreate_type_in_context(self, pc_type):
        """Recreate a PC type's LLVM representation in current module's context"""
        # Handle None
        if pc_type is None:
            raise ValueError("Cannot create LLVM type for None")
        
        # Handle LLVM types directly (e.g., struct types)
        if isinstance(pc_type, ir.Type):
            # If it's a struct type, get the corresponding type from current context
            if isinstance(pc_type, ir.IdentifiedStructType):
                type_name = pc_type.name
                return self.module.context.get_identified_type(type_name)
            return pc_type
        
        # Handle ptr types (check class attribute first)
        if hasattr(pc_type, 'pointee_type') and pc_type.pointee_type is not None:
            pointee = self._recreate_type_in_context(pc_type.pointee_type)
            return ir.PointerType(pointee)
        
        # Handle struct types by name
        if hasattr(pc_type, '__name__'):
            type_name = pc_type.__name__
            if get_unified_registry().has_struct(type_name):
                # Get struct type from current module's context
                return self.module.context.get_identified_type(type_name)
        
        # Handle basic types with get_llvm_type method
        return pc_type.get_llvm_type(self.module.context)
    
    def _declare_extern_function(self, func_name):
        """Declare a specific extern function when it's actually called"""
        if func_name in self.extern_functions:
            return self.extern_functions[func_name]
            
        # Import here to avoid circular import
        from .decorators import get_extern_function_info
        from .builder import LLVMBuilder
        
        func_info = get_extern_function_info(func_name)
        if not func_info:
            raise NameError(f"Extern function '{func_name}' not registered")
        
        # Check if this function was imported (if we have import tracking)
        if hasattr(self, 'imported_externs') and self.imported_externs:
            if func_name not in self.imported_externs:
                # Function not imported, but we'll still allow it if it's being called
                # This handles cases where functions are called without explicit import
                pass
        
        # Convert PC types to LLVM types
        param_types = []
        for param_name, param_type in func_info['param_types']:
            if param_name == 'args':  # Handle *args (varargs)
                continue  # Skip varargs in type list
            if param_type is None:
                raise TypeError(f"Extern function '{func_name}': parameter '{param_name}' has no type annotation")
            llvm_type = param_type.get_llvm_type(self.module.context)
            param_types.append(llvm_type)
        
        # Convert return type from pythoc type to LLVM type
        if func_info['return_type'] is None:
            return_type = ir.VoidType()
        else:
            return_type = func_info['return_type'].get_llvm_type(self.module.context)
        
        # Handle varargs (printf-style functions)
        has_varargs = any(param_name == 'args' for param_name, _ in func_info['param_types'])
        
        # Use builder to declare function with ABI handling
        temp_builder = LLVMBuilder()
        func_wrapper = temp_builder.declare_function(
            self.module, func_name, param_types, return_type, var_arg=has_varargs
        )
        extern_func = func_wrapper.ir_function
        
        # Set calling convention if specified
        if func_info.get('calling_convention') == 'stdcall':
            extern_func.calling_convention = 'x86_stdcallcc'
        else:
            extern_func.calling_convention = 'ccc'  # Default C calling convention
        
        self.extern_functions[func_name] = extern_func
        return extern_func
    
    def compile_function_from_ast(self, ast_node: ast.FunctionDef, source_code: str = None, reset_module: bool = False, param_type_hints: dict = None, return_type_hint = None, user_globals: dict = None) -> ir.Function:
        """
        Compile a function directly from an AST node (meta-programming support)
        
        This is used for runtime-generated functions where we have the AST
        but not necessarily the original source file.
        
        Args:
            ast_node: The AST FunctionDef node to compile
            source_code: Optional source code string (for debugging/context)
            reset_module: If True, create a fresh module; if False, add to existing module
            param_type_hints: Optional dict mapping parameter names to PC types (for meta-programming)
            return_type_hint: Optional return type hint (for meta-programming)
        
        Returns:
            The compiled LLVM function
        """
        logger.debug(f"compile_function_from_ast: {ast_node.name}")
        # Only create a fresh module if requested or if no module exists
        if reset_module or self.module is None:
            self.module = self.create_module()
            self.compiled_functions.clear()
        
        # Check if function already exists in module (e.g., from forward declaration)
        existing_func = None
        try:
            existing_func = self.module.get_global(ast_node.name) if ast_node.name else None
        except KeyError:
            # Function doesn't exist yet, which is fine
            pass

        user_globals = user_globals or self.user_globals
        type_resolver = TypeResolver(self.module.context, user_globals=user_globals)
        
        # First, create function declaration
        param_types = []
        for arg in ast_node.args.args:
            # Use pre-parsed type hints if available (for meta-programming)
            if param_type_hints and arg.arg in param_type_hints:
                pc_type = param_type_hints[arg.arg]
                if hasattr(pc_type, 'get_llvm_type'):
                    # All PC types now accept module_context parameter
                    param_type = pc_type.get_llvm_type(self.module.context)
                else:
                    raise TypeError(f"Invalid type for parameter '{arg.arg}'")
            elif arg.annotation:
                param_type = self._parse_type_annotation(arg.annotation)
            else:
                raise TypeError(f"Parameter '{arg}' has no type annotation")
            param_types.append(param_type)
        
        # Get return type (default to void if no annotation)
        if return_type_hint:
            # Use pre-parsed return type hint (for meta-programming)
            if hasattr(return_type_hint, 'get_llvm_type'):
                # All PC types now accept module_context parameter
                return_type = return_type_hint.get_llvm_type(self.module.context)
            else:
                return_type = ir.VoidType()
        elif ast_node.returns:
            return_type = self._parse_type_annotation(ast_node.returns)
        else:
            return_type = ir.VoidType()  # Default to void
        
        # Detect varargs type (only call once and save results)
        from .ast_visitor.varargs import detect_varargs
        varargs_kind, element_types, varargs_name = detect_varargs(ast_node, type_resolver)
        
        # For struct varargs with a @compile decorated class (e.g., *args: Data),
        # element_types will be empty. Extract field types from the struct class.
        if varargs_kind == 'struct' and not element_types and ast_node.args.vararg:
            annotation = ast_node.args.vararg.annotation
            parsed_type = type_resolver.parse_annotation(annotation)
            if hasattr(parsed_type, '_field_types'):
                # For struct[...] created types, use _field_types
                element_types = list(parsed_type._field_types)
            elif hasattr(parsed_type, '_struct_fields'):
                # Create AST nodes for each field type (for consistency)
                element_types = []
                for field_name, field_type in parsed_type._struct_fields:
                    # We'll store the PC type directly since we have it
                    # This is a bit of a hack but avoids AST node creation
                    element_types.append(field_type)
        
        # Determine LLVM varargs flag and actual parameter types
        # - struct varargs: expand into individual parameters, no LLVM varargs
        # - union/enum/none varargs: use LLVM varargs (va_list)
        has_llvm_varargs = varargs_kind in ('union', 'enum', 'none')
        
        # For struct varargs, expand parameter types
        if varargs_kind == 'struct':
            # Add each struct element as a separate parameter
            for elem_type in element_types:
                # elem_type might be either an AST node or a PC type directly
                if hasattr(elem_type, 'get_llvm_type'):
                    # It's a PC type directly
                    elem_pc_type = elem_type
                else:
                    # It's an AST node, parse it
                    elem_pc_type = type_resolver.parse_annotation(elem_type)
                
                if hasattr(elem_pc_type, 'get_llvm_type'):
                    param_types.append(elem_pc_type.get_llvm_type(self.module.context))
                else:
                    raise TypeError(f"Invalid varargs element type: {elem_type}")
        
        # Use builder to declare function with C ABI for interop with C code
        # pythoc functions must use C ABI so they can be called from C via function pointers
        from .builder import LLVMBuilder, FunctionWrapper
        temp_builder = LLVMBuilder()
        logger.debug(f"declare_function: {ast_node.name}, param_types={param_types}, return_type={return_type}, existing_func={existing_func}")
        func_wrapper = temp_builder.declare_function(
            self.module, ast_node.name,
            param_types, return_type,
            var_arg=has_llvm_varargs,
            existing_func=existing_func
        )
        logger.debug(f"After declare_function: ir_function.args types={[a.type for a in func_wrapper.ir_function.args]}")
        logger.debug(f"param_coercion_info={func_wrapper.param_coercion_info}")
        llvm_function = func_wrapper.ir_function
        sret_info = func_wrapper.sret_info
        
        # Set parameter names (user parameters only, sret is handled internally)
        param_names = [arg.arg for arg in ast_node.args.args]
        if varargs_kind == 'struct':
            # Add synthetic parameter names for expanded struct elements
            for i in range(len(element_types)):
                param_names.append(f'{varargs_name}_elem{i}')
        
        for i, param_name in enumerate(param_names):
            func_wrapper.get_user_arg(i).name = param_name
        
        # Now compile the function body
        # Build func_type_hints dict for the single function
        func_type_hints = {}
        # Store sret info for use in return statement
        if sret_info:
            func_type_hints['_sret_info'] = sret_info
        # Store param coercion info for parameter unpacking
        if func_wrapper.param_coercion_info:
            func_type_hints['_param_coercion_info'] = func_wrapper.param_coercion_info
        # Return type
        if return_type_hint is not None:
            func_type_hints[ast_node.name] = {'return': return_type_hint}
        elif ast_node.returns:
            rt = type_resolver.parse_annotation(ast_node.returns)
            if rt:
                func_type_hints[ast_node.name] = {'return': rt}
        # Param types
        param_hints = {}
        for arg in ast_node.args.args:
            if param_type_hints and arg.arg in param_type_hints:
                param_hints[arg.arg] = param_type_hints[arg.arg]
            elif arg.annotation:
                pt = type_resolver.parse_annotation(arg.annotation)
                if pt:
                    param_hints[arg.arg] = pt
        if func_type_hints.get(ast_node.name):
            func_type_hints[ast_node.name]['params'] = param_hints
        else:
            func_type_hints[ast_node.name] = {'params': param_hints}
        
        visitor = LLVMIRVisitor(self.module, None, func_type_hints, None, compiler=self, user_globals=user_globals)
        
        visitor.current_function = llvm_function
        
        # Store varargs information for this function (using results from earlier detection)
        visitor.current_varargs_info = None
        if varargs_kind != 'none':
            # Parse element types
            element_pc_types = []
            if element_types:
                for elem_type in element_types:
                    # elem_type might be either an AST node or a PC type directly
                    if hasattr(elem_type, 'get_llvm_type'):
                        # It's a PC type directly
                        pc_type = elem_type
                    else:
                        # It's an AST node, parse it
                        pc_type = visitor.type_resolver.parse_annotation(elem_type)
                    if pc_type:
                        element_pc_types.append(pc_type)
            
            # Store varargs info
            # - For struct varargs: used for len(args) constant folding
            # - For union/enum varargs: used for va_arg access
            visitor.current_varargs_info = {
                'kind': varargs_kind,
                'name': varargs_name,
                'element_types': element_pc_types,
                'num_normal_params': len(ast_node.args.args),
                'va_list': None  # Will be initialized on first access (union/enum only)
            }
        
        # Create entry block
        entry_block = llvm_function.append_basic_block('entry')
        from .builder import LLVMBuilder
        ir_builder = ir.IRBuilder(entry_block)
        visitor.builder = LLVMBuilder(ir_builder)
        
        # Set ABI context for struct returns
        sret_info = func_type_hints.get('_sret_info') if func_type_hints else None
        visitor.builder.set_return_abi_context(llvm_function, sret_info)
        
        # Initialize parameters - they will be registered in variable registry
        # For struct varargs, we also initialize the expanded parameters AND create a struct
        normal_param_count = len(ast_node.args.args)
        total_params = normal_param_count
        if varargs_kind == 'struct':
            total_params += len(element_types)
        
        # First, register all normal parameters
        # Use func_wrapper.get_user_arg_unpacked() to handle ABI coercion transparently
        for i in range(normal_param_count):
            arg = ast_node.args.args[i]
            param_name = arg.arg
            param_annotation = arg.annotation
            
            # Get unpacked parameter value (handles ABI coercion transparently)
            param_val, param_type = func_wrapper.get_user_arg_unpacked(i, visitor.builder.ir_builder)
            
            # Allocate and store parameter
            alloca = visitor._create_alloca_in_entry(param_type, f"{param_name}_addr")
            visitor.builder.store(param_val, alloca)
            
            # Register parameter in variable registry (always), with best-effort type hint
            type_hint = None
            # Use pre-parsed type hints if available (for meta-programming)
            if param_type_hints and param_name in param_type_hints:
                type_hint = param_type_hints[param_name]
            elif param_annotation:
                type_hint = visitor.get_pc_type_from_annotation(param_annotation)
            
            # Create ValueRef with proper wrapper for function pointers
            from .context import VariableInfo
            from .valueref import wrap_value
            from .builtin_entities.func import func
            
            # Check if this is a function pointer parameter
            if type_hint and isinstance(type_hint, type) and issubclass(type_hint, func):
                # Store alloca directly, func.handle_call will load it when needed
                value_ref = wrap_value(
                    alloca,
                    kind='address',
                    type_hint=type_hint,
                    address=alloca
                )
            else:
                value_ref = wrap_value(
                    alloca,
                    kind='address',
                    type_hint=type_hint,
                    address=alloca
                )
            
            var_info = VariableInfo(
                name=param_name,
                value_ref=value_ref,
                alloca=alloca,
                source="parameter",
                is_parameter=True,
                is_mutable=True
            )
            visitor.ctx.var_registry.declare(var_info, allow_shadow=True)
            
            # Initialize linear states for parameters (active = ownership transferred)
            if type_hint and visitor._is_linear_type(type_hint):
                visitor._init_linear_states(var_info, type_hint, initial_state='active')
        
        # For struct varargs, create a synthetic struct from the expanded parameters
        if varargs_kind == 'struct':
            # Get all expanded parameter values using func_wrapper
            # Use get_user_arg_unpacked() to handle ABI coercion transparently
            expanded_values = []
            expanded_types_llvm = []
            for elem_idx in range(len(element_types)):
                param_idx = normal_param_count + elem_idx
                param_val, param_type = func_wrapper.get_user_arg_unpacked(
                    param_idx, visitor.builder.ir_builder
                )
                expanded_values.append(param_val)
                expanded_types_llvm.append(param_type)
            
            # Create an anonymous struct type
            struct_type_llvm = ir.LiteralStructType(expanded_types_llvm)
            
            # Allocate space for the struct
            varargs_alloca = visitor._create_alloca_in_entry(struct_type_llvm, f"{varargs_name}_struct")
            
            # Store each expanded parameter into the struct
            for elem_idx, param_val in enumerate(expanded_values):
                # GEP to get pointer to field
                field_ptr = visitor.builder.gep(
                    varargs_alloca,
                    [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), elem_idx)],
                    inbounds=True
                )
                visitor.builder.store(param_val, field_ptr)
            
            # Determine the PC type hint for the varargs struct
            # If it's a named struct class, use that type
            # Otherwise, create an anonymous struct type hint
            varargs_type_hint = None
            if ast_node.args.vararg and ast_node.args.vararg.annotation:
                parsed_type = visitor.type_resolver.parse_annotation(ast_node.args.vararg.annotation)
                # Check if it's a struct class or a generic struct[...]
                if hasattr(parsed_type, '_is_struct') and parsed_type._is_struct:
                    varargs_type_hint = parsed_type
                elif hasattr(parsed_type, '__origin__'):
                    # It's a generic struct[...], use it directly
                    varargs_type_hint = parsed_type
                else:
                    # Fallback: use parsed_type directly if it looks like a struct type
                    varargs_type_hint = parsed_type
            
            # Register the varargs struct as a variable
            from .context import VariableInfo
            from .valueref import wrap_value
            
            varargs_value_ref = wrap_value(
                varargs_alloca,
                kind='address',
                type_hint=varargs_type_hint,
                address=varargs_alloca
            )
            
            varargs_var_info = VariableInfo(
                name=varargs_name,
                value_ref=varargs_value_ref,
                alloca=varargs_alloca,
                source="parameter",
                is_parameter=True,
                is_mutable=False  # varargs is read-only
            )
            
            visitor.ctx.var_registry.declare(varargs_var_info, allow_shadow=True)
        
        # For union/enum varargs, register a placeholder variable
        # The actual va_list will be initialized on first access in subscripts.py
        if varargs_kind in ('union', 'enum'):
            from .context import VariableInfo
            from .valueref import wrap_value
            
            # Parse the varargs type annotation to get the enum/union type
            varargs_type_hint = None
            if ast_node.args.vararg and ast_node.args.vararg.annotation:
                varargs_type_hint = visitor.type_resolver.parse_annotation(ast_node.args.vararg.annotation)
            
            # Create a placeholder ValueRef - the actual va_list is initialized lazily
            # We use kind='varargs' to indicate this is a special varargs placeholder
            varargs_placeholder = wrap_value(
                None,  # Will be initialized on first access
                kind='varargs',  # Special kind to indicate varargs placeholder
                type_hint=varargs_type_hint,  # The enum/union type
            )
            
            varargs_var_info = VariableInfo(
                name=varargs_name,
                value_ref=varargs_placeholder,
                alloca=None,  # No alloca yet - will be created on first access
                source="parameter",
                is_parameter=True,
                is_mutable=False  # varargs is read-only
            )
            
            visitor.ctx.var_registry.declare(varargs_var_info, allow_shadow=True)
        
        # Initialize list to accumulate all inlined statements
        visitor._all_inlined_stmts = []
        
        # Visit function body
        for stmt in ast_node.body:
            visitor.visit(stmt)
        
        # Debug hook - capture all inlined statements accumulated during compilation
        from .utils.ast_debug import ast_debugger
        if visitor._all_inlined_stmts:
            ast_debugger.capture(
                "function_all_inlines",
                visitor._all_inlined_stmts,
                func_name=ast_node.name,
                inline_count=len([s for s in visitor._all_inlined_stmts if isinstance(s, ast.While)]),
                total_stmts=len(visitor._all_inlined_stmts)
            )
        
        # Check that all linear tokens have been consumed before function exit
        visitor._check_linear_tokens_consumed()
        
        # Ensure function has a return
        if not visitor.builder.block.is_terminated:
            if return_type == ir.VoidType():
                visitor.builder.ret_void()
            elif isinstance(return_type, ir.PointerType):
                visitor.builder.ret(ir.Constant(return_type, None))
            elif isinstance(return_type, ir.IntType):
                visitor.builder.ret(ir.Constant(return_type, 0))
            elif isinstance(return_type, (ir.FloatType, ir.DoubleType)):
                visitor.builder.ret(ir.Constant(return_type, 0.0))
            elif isinstance(return_type, (ir.LiteralStructType, ir.IdentifiedStructType)):
                visitor.builder.ret(ir.Constant(return_type, ir.Undefined))
            elif isinstance(return_type, ir.ArrayType):
                visitor.builder.ret(ir.Constant(return_type, ir.Undefined))
            else:
                visitor.builder.ret(ir.Constant(return_type, ir.Undefined))
        
        # Clean up empty basic blocks by adding unreachable instructions
        # This is needed because constant loop unrolling may create empty blocks
        empty_blocks_cleaned = 0
        for block in llvm_function.blocks:
            instr_list = list(block.instructions)
            if not instr_list:
                # Empty block - add unreachable to make it valid LLVM IR
                visitor.builder.position_at_end(block)
                visitor.builder.unreachable()
                empty_blocks_cleaned += 1
        if empty_blocks_cleaned > 0:
            logger.info(f"Cleaned up {empty_blocks_cleaned} empty blocks in {llvm_function.name}")
        
        self.compiled_functions.append(llvm_function)
        return llvm_function
    
    def _parse_type_annotation(self, annotation):
        """Parse type annotation and return corresponding LLVM type
        
        Uses TypeResolver for unified type parsing.
        """
        type_resolver = TypeResolver(self.module.context, user_globals=self.user_globals)
        return type_resolver.annotation_to_llvm_type(annotation)
    
    def get_ir(self) -> str:
        """Get the LLVM IR as a string"""
        if self.module is None:
            return ""
        return str(self.module)
    
    def save_ir_to_file(self, filename: str):
        """Save the LLVM IR to a file (optimized version if available)"""
        if self.module is None:
            return
        
        with open(filename, 'w') as f:
            f.write(self.get_ir())
    
    def verify_module(self) -> bool:
        """Verify the LLVM module"""
        if self.module is None:
            return False
        
        # Parse the module to check for errors
        try:
            module_str = str(self.module)
            llvm_module = binding.parse_assembly(module_str)
            llvm_module.verify()
        except Exception as e:
            # Try to get module string for error reporting, but handle failures
            try:
                ir_str = str(self.module)
            except Exception:
                ir_str = "<failed to serialize module>"
            logger.error(f"Verification failed: {e}\nModule IR:\n{ir_str}")
            raise e
        return True
    
    def optimize_module(self, optimization_level: int = 2):
        """Optimize the LLVM module with O2 level optimizations by default"""
        if self.module is None:
            return
        
        try:
            # Parse the IR into an LLVM module
            llvm_module = binding.parse_assembly(str(self.module))
            
            # CRITICAL: Use function pass manager for mem2reg (alloca->SSA promotion)
            # This is essential for good code generation
            fpm = binding.create_function_pass_manager(llvm_module)
            
            # Add SROA pass (Scalar Replacement of Aggregates) - similar to mem2reg
            # This promotes allocas to registers and is the most important optimization
            fpm.add_sroa_pass()
            
            if optimization_level >= 1:
                # Basic scalar optimizations - order matters!
                try:
                    # Early constant propagation - critical for constant folding after SROA
                    fpm.add_constant_propagation_pass()
                    # Early CSE before more expensive opts
                    fpm.add_early_cse_pass()
                    # Instruction combining and simplification
                    fpm.add_instruction_combining_pass()
                    fpm.add_reassociate_expressions_pass()
                    fpm.add_cfg_simplification_pass()
                    # Loop structure normalization
                    fpm.add_loop_simplify_pass()
                except AttributeError:
                    pass
            
            if optimization_level >= 2:
                # Loop optimizations - important for this project
                try:
                    fpm.add_licm_pass()
                    fpm.add_indvars_pass()
                    fpm.add_loop_deletion_pass()
                    fpm.add_loop_rotate_pass()
                except AttributeError:
                    pass
                
                # More aggressive optimizations
                try:
                    fpm.add_gvn_pass()
                    # Run InstCombine again after GVN to clean up
                    fpm.add_instruction_combining_pass()
                    fpm.add_dead_code_elimination_pass()
                    fpm.add_aggressive_dce_pass()
                except AttributeError:
                    pass
            
            if optimization_level >= 3:
                # Aggressive loop optimizations
                try:
                    fpm.add_loop_unroll_pass()
                    fpm.add_tail_call_elimination_pass()
                    fpm.add_jump_threading_pass()
                except AttributeError:
                    pass
                
                # Vectorization passes
                try:
                    fpm.add_loop_vectorize_pass()
                    fpm.add_slp_vectorize_pass()
                except AttributeError:
                    pass
            
            # Initialize and run function passes on all functions
            fpm.initialize()
            for func in llvm_module.functions:
                if not func.is_declaration:
                    fpm.run(func)
            fpm.finalize()
            
            # Now run module-level passes
            pm = binding.create_module_pass_manager()
            
            if optimization_level >= 1:
                # Module-level basic optimizations
                try:
                    pm.add_constant_merge_pass()
                    pm.add_dead_arg_elimination_pass()
                except AttributeError:
                    pass
            
            if optimization_level >= 2:
                # Module-level aggressive optimizations
                try:
                    pm.add_function_attrs_pass()
                    pm.add_global_dce_pass()
                    pm.add_global_optimizer_pass()
                    # Inline small functions even at O2
                    pm.add_function_inlining_pass(225)
                except AttributeError:
                    pass
            
            if optimization_level >= 3:
                # Aggressive optimizations (O3 level)
                try:
                    # Higher inline threshold for O3
                    pm.add_function_inlining_pass(500)
                    pm.add_ipsccp_pass()
                    pm.add_dead_arg_elimination_pass()
                except AttributeError:
                    pass
            
            # Run module passes
            pm.run(llvm_module)
            
            # Store the optimized IR
            self._optimized_ir = str(llvm_module)
            
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")
    
    def get_ir(self) -> str:
        """Get the LLVM IR as a string, returning optimized version if available"""
        if hasattr(self, '_optimized_ir') and self._optimized_ir:
            return self._optimized_ir
        elif self.module is None:
            return ""
        else:
            return str(self.module)
    
    def compile_to_object(self, filename: str):
        """Compile the LLVM IR to an object file"""
        if self.module is None:
            raise RuntimeError("No module to compile")
        
        # Use optimized IR if available, otherwise use original module
        ir_to_compile = self.get_ir()
        
        # Parse the IR
        llvm_module = binding.parse_assembly(ir_to_compile)
        
        # Create a target machine with PIC relocation model
        # This is critical for shared libraries to support lazy symbol resolution
        # and circular dependencies between .so files
        target = binding.Target.from_triple(self.module.triple)
        target_machine = target.create_target_machine(reloc='pic', codemodel='default')
        
        # Compile to object code
        with open(filename, 'wb') as f:
            f.write(target_machine.emit_object(llvm_module))
    
    def compile_to_executable(self, output_name: str, obj_file: str):
        """
        Link object file to create an executable binary.
        
        Note: This method expects the object file to already exist (generated by @compile decorator).
        It does NOT compile the module itself - that should be done by the decorator.
        
        Args:
            output_name: Path to output executable
            obj_file: Path to existing object file to link
        """
        if self.module is None:
            raise RuntimeError("No module to compile")
        
        import os
        from .utils.build_utils import link_executable
        
        # Verify object file exists
        if not os.path.exists(obj_file):
            raise RuntimeError(
                f"Object file not found: {obj_file}\n"
                f"Make sure the @compile decorator has been executed before calling compile_to_executable()."
            )
        
        # Use the unified link_executable function
        link_executable([obj_file], output_name)
        return True