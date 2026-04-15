"""
LLVM Compiler using llvmlite
Enhanced version with modular architecture and comprehensive functionality
"""

import ast
import sys
from dataclasses import dataclass
from typing import Any, Dict, List
from llvmlite import ir, binding
from .ast_visitor import LLVMIRVisitor
from .registry import get_unified_registry, register_struct_from_class
from .type_resolver import TypeResolver
from .logger import logger

# Initialize LLVM
# llvmlite 0.45+ auto-initializes core; calling initialize() raises RuntimeError.
try:
    binding.initialize()
except RuntimeError:
    pass
binding.initialize_native_target()
binding.initialize_native_asmprinter()

# Detect pass manager API: llvmlite <0.45 uses legacy PM, >=0.45 uses new PM.
_USE_NEW_PM = not hasattr(binding, 'create_function_pass_manager')


@dataclass
class _ResolvedFunctionDeclaration:
    """Compiler-local callable declaration data."""
    param_names: List[str]
    param_type_hints: Dict[str, Any]
    return_type_hint: Any
    param_llvm_types: List[ir.Type]
    return_llvm_type: ir.Type
    varargs: Any

    @property
    def has_llvm_varargs(self) -> bool:
        return self.varargs.has_llvm_varargs

class LLVMCompiler:
    """Enhanced LLVM compiler using llvmlite"""
    
    def __init__(self, user_globals=None):
        self.module = None
        self.compiled_functions = []

        self.user_globals = user_globals or {}  # User code's global namespace
        self.create_module()

    def update_globals(self, user_globals):
        """Update user globals"""
        self.user_globals = user_globals
    
    def create_module(self, name: str = "main"):
        """Create a new LLVM module"""
        self.module = ir.Module(name=name)
        # On Windows, use windows-gnu triple so that generated .o files are
        # compatible with zig's GNU linker (the only supported linker on Windows).
        triple = binding.get_default_triple()
        if sys.platform == 'win32':
            triple = triple.replace('-windows-msvc', '-windows-gnu')
        self.module.triple = triple
        # Set data layout for correct struct size calculation
        target = binding.Target.from_triple(self.module.triple)
        target_machine = target.create_target_machine()
        self.module.data_layout = target_machine.target_data
        return self.module
    
    
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

        if hasattr(pc_type, 'get_llvm_type'):
            try:
                return pc_type.get_llvm_type(self.module.context)
            except TypeError as e:
                if "positional argument" in str(e) or "takes" in str(e):
                    return pc_type.get_llvm_type()
                raise
        
        # Handle struct types by name
        if hasattr(pc_type, '__name__'):
            type_name = pc_type.__name__
            if get_unified_registry().has_struct(type_name):
                # Get struct type from current module's context
                return self.module.context.get_identified_type(type_name)

        raise TypeError(f"Cannot recreate LLVM type for {pc_type}")
    
    def _resolve_function_declaration(
        self,
        ast_node: ast.FunctionDef,
        param_type_hints: dict = None,
        return_type_hint=None,
        user_globals: dict = None,
    ) -> _ResolvedFunctionDeclaration:
        """Resolve one function declaration into PC and LLVM types once."""
        from .ast_visitor.varargs import resolve_varargs
        from .builtin_entities.types import void

        user_globals = user_globals or self.user_globals
        type_resolver = TypeResolver(self.module.context, user_globals=user_globals)

        resolved_param_type_hints: Dict[str, Any] = {}
        param_names: List[str] = []

        for arg in ast_node.args.args:
            if param_type_hints and arg.arg in param_type_hints:
                pc_type = param_type_hints[arg.arg]
            elif arg.annotation:
                pc_type = type_resolver.parse_annotation(arg.annotation)
            else:
                raise TypeError(f"Parameter '{arg.arg}' has no type annotation")

            if pc_type is None:
                raise TypeError(f"Parameter '{arg.arg}' has invalid type annotation")

            resolved_param_type_hints[arg.arg] = pc_type
            param_names.append(arg.arg)

        if return_type_hint is not None:
            resolved_return_type = return_type_hint
        elif ast_node.returns:
            resolved_return_type = type_resolver.parse_annotation(ast_node.returns)
        else:
            resolved_return_type = void

        if resolved_return_type is None:
            resolved_return_type = void

        varargs = resolve_varargs(ast_node, type_resolver)
        if varargs.is_typed and varargs.param_name is not None:
            resolved_param_type_hints.pop(varargs.param_name, None)
            for index, elem_pc_type in enumerate(varargs.element_types):
                expanded_name = f"{varargs.param_name}_elem{index}"
                param_names.append(expanded_name)
                resolved_param_type_hints[expanded_name] = elem_pc_type

        param_llvm_types = [
            self._recreate_type_in_context(resolved_param_type_hints[name])
            for name in param_names
        ]
        return_llvm_type = self._recreate_type_in_context(resolved_return_type)

        return _ResolvedFunctionDeclaration(
            param_names=param_names,
            param_type_hints=resolved_param_type_hints,
            return_type_hint=resolved_return_type,
            param_llvm_types=param_llvm_types,
            return_llvm_type=return_llvm_type,
            varargs=varargs,
        )

    def compile_function_from_ast(self, ast_node: ast.FunctionDef, source_code: str = None, reset_module: bool = False, param_type_hints: dict = None, return_type_hint = None, user_globals: dict = None, group_key = None, func_state = None) -> ir.Function:
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

        # Any new body added to the module invalidates previously cached optimized IR.
        self._optimized_ir = None

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
        resolved_decl = self._resolve_function_declaration(
            ast_node,
            param_type_hints=param_type_hints,
            return_type_hint=return_type_hint,
            user_globals=user_globals,
        )
        varargs_kind = resolved_decl.varargs.kind
        varargs_name = resolved_decl.varargs.param_name
        
        # Use builder to declare function with C ABI for interop with C code
        # pythoc functions must use C ABI so they can be called from C via function pointers
        from .builder import LLVMBuilder, FunctionWrapper
        temp_builder = LLVMBuilder()
        logger.debug(
            f"declare_function: {ast_node.name}, "
            f"param_types={resolved_decl.param_llvm_types}, "
            f"return_type={resolved_decl.return_llvm_type}, existing_func={existing_func}"
        )
        func_wrapper = temp_builder.declare_function(
            self.module, ast_node.name,
            resolved_decl.param_llvm_types, resolved_decl.return_llvm_type,
            var_arg=resolved_decl.has_llvm_varargs,
            existing_func=existing_func
        )
        logger.debug(f"After declare_function: ir_function.args types={[a.type for a in func_wrapper.ir_function.args]}")
        logger.debug(f"param_coercion_info={func_wrapper.param_coercion_info}")
        llvm_function = func_wrapper.ir_function
        sret_info = func_wrapper.sret_info
        
        # Set parameter names (user parameters only, sret is handled internally)
        for i, param_name in enumerate(resolved_decl.param_names):
            func_wrapper.get_user_arg(i).name = param_name
        
        visitor = LLVMIRVisitor(
            module=self.module,
            builder=None,
            struct_types=None,
            compiler=self,
            user_globals=user_globals,
        )

        normal_param_hints = {}
        for arg in ast_node.args.args:
            if arg.arg in resolved_decl.param_type_hints:
                normal_param_hints[arg.arg] = resolved_decl.param_type_hints[arg.arg]

        varargs_info = None
        if varargs_name is not None:
            varargs_info = {
                'kind': varargs_kind,
                'name': varargs_name,
                'element_types': list(resolved_decl.varargs.element_types),
                'num_normal_params': len(ast_node.args.args),
                'va_list': None,
            }

        # Set up long-lived binding state and this compilation's active frame.
        if func_state is None:
            from .context import FunctionBindingState
            func_state = FunctionBindingState(group_key=group_key)
        from .context import ActiveCompileFrame
        compile_frame = ActiveCompileFrame(
            current_function=llvm_function,
            function_name=ast_node.name,
            return_type_hint=resolved_decl.return_type_hint,
            param_type_hints=normal_param_hints,
            sret_info=sret_info,
            param_coercion_info=func_wrapper.param_coercion_info or {},
            varargs_info=varargs_info,
        )
        visitor.func_state = compile_frame
        visitor.binding_state = func_state
        visitor.current_function = llvm_function
        visitor.current_group_key = group_key  # For dependency tracking at call time
        
        # Create entry block
        entry_block = llvm_function.append_basic_block('entry')
        from .builder import LLVMBuilder
        ir_builder = ir.IRBuilder(entry_block)
        real_builder = LLVMBuilder(ir_builder)

        # Reset scoped label tracking for this function
        visitor.scope_manager.reset_label_tracking()

        # Set ABI context for struct returns
        real_builder.set_return_abi_context(llvm_function, compile_frame.sret_info)

        # Create ControlFlowBuilder wrapping real_builder and set as visitor.builder
        # This must happen before parameter initialization which uses visitor.builder
        from .ast_visitor.control_flow_builder import ControlFlowBuilder
        visitor.builder = ControlFlowBuilder(real_builder, visitor, ast_node.name)
        
        # Initialize parameters - they will be registered in variable registry
        # For struct varargs, we also initialize the expanded parameters AND create a struct
        normal_param_count = len(ast_node.args.args)
        
        # First, register all normal parameters
        # Use func_wrapper.get_user_arg_unpacked() to handle ABI coercion transparently
        for i in range(normal_param_count):
            arg = ast_node.args.args[i]
            param_name = arg.arg
            
            # Get unpacked parameter value (handles ABI coercion transparently)
            param_val, param_type = func_wrapper.get_user_arg_unpacked(i, visitor.builder)
            
            # Allocate and store parameter
            alloca = visitor._create_alloca_in_entry(param_type, f"{param_name}_addr")
            visitor.builder.store(param_val, alloca)
            
            # Register parameter in variable registry (always), with best-effort type hint
            type_hint = None
            # Use pre-parsed type hints if available (for meta-programming)
            if param_name in resolved_decl.param_type_hints:
                type_hint = resolved_decl.param_type_hints[param_name]
            
            from .context import VariableInfo
            from .valueref import wrap_value
            
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
            visitor.scope_manager.declare_variable(var_info, allow_shadow=True)
            
            # Initialize linear states for parameters (active = ownership transferred)
            if type_hint and visitor._is_linear_type(type_hint):
                visitor._init_linear_states(var_info, type_hint, initial_state='active')
        
        # For typed varargs (*args: T), create a synthetic composite from
        # the expanded parameters so the function body can access them as
        # args[i] or args.field.
        if resolved_decl.varargs.is_typed:
            # Get all expanded parameter values using func_wrapper
            # Use get_user_arg_unpacked() to handle ABI coercion transparently
            expanded_values = []
            expanded_types_llvm = []
            for elem_idx in range(len(resolved_decl.varargs.element_types)):
                param_idx = normal_param_count + elem_idx
                param_val, param_type = func_wrapper.get_user_arg_unpacked(
                    param_idx, visitor.builder
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
            varargs_type_hint = resolved_decl.varargs.parsed_type
            
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
            
            visitor.scope_manager.declare_variable(varargs_var_info, allow_shadow=True)
        
        # Initialize list to accumulate all inlined statements (via func_state)
        visitor._all_inlined_stmts = compile_frame.all_inlined_stmts

        # Emit LinearRegister events for linear parameters
        # This must happen after ControlFlowBuilder is created (visitor.builder)
        for var_info in visitor.scope_manager.get_all_in_current_scope():
            if var_info.is_parameter and var_info.type_hint:
                # Check if parameter type is linear and get all linear paths
                if visitor._is_linear_type(var_info.type_hint):
                    paths = visitor._get_linear_paths(var_info.type_hint)
                    for path in paths:
                        # Parameters with linear types start as 'valid' (ownership passed in)
                        visitor.builder.record_linear_register(
                            var_info.var_id, var_info.name, path, 'valid',
                            line_number=var_info.line_number, node=ast_node
                        )
        
        # Enter function-level scope in ScopeManager
        # This is the root scope for all defers in this function
        from .scope_manager import ScopeType
        visitor.scope_manager.enter_scope(ScopeType.FUNCTION)
        
        # Visit function body
        # Skip statements after control flow termination (e.g., after infinite loops)
        # Exception: with label() statements are always processed because they create new reachable blocks
        for stmt in ast_node.body:
            # Check if current block is terminated (unreachable code)
            if visitor.builder.is_terminated():
                # Check if this is a with label() statement - these should always be processed
                is_label_stmt = False
                if isinstance(stmt, ast.With):
                    for item in stmt.items:
                        if isinstance(item.context_expr, ast.Call):
                            func = item.context_expr.func
                            if isinstance(func, ast.Name) and func.id == 'label':
                                is_label_stmt = True
                                break
                
                if not is_label_stmt:
                    logger.debug(f"Skipping unreachable statement at line {getattr(stmt, 'lineno', '?')}")
                    continue
            visitor.builder.add_stmt(stmt)
            visitor.visit(stmt)

        # Finalize CFG structure (no IR emission yet).
        # Connects fall-through CFG blocks to exit, computes loop headers.
        visitor.builder.finalize()
        visitor.builder.dump_cfg()  # Uses logger.debug by default

        # CFG merge checks (and loop invariants).
        # Must happen before emit_ir() since IR doesn't exist yet.
        visitor.builder.run_cfg_linear_check()

        # Exit function-level scope in ScopeManager
        # This enforces: when variables go out of scope, all linear states are inactive.
        # Note: defer execution during scope exit records PCIR (not real IR).
        visitor.scope_manager.exit_scope(visitor.builder, node=ast_node)

        # Check for unresolved scoped goto statements
        visitor.scope_manager.check_goto_consistency(ast_node)

        # Check for unexecuted deferred calls (should not happen if implementation is correct)
        from .builtin_entities.defer import check_defers_at_function_end
        check_defers_at_function_end(visitor, ast_node)

        # Replay PCIR -> actual LLVM IR
        # This must happen after all CFG analysis and scope checking.
        visitor.builder.emit_ir()

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

        if getattr(func_state, 'wrapper', None) is not None:
            func_info = getattr(func_state.wrapper, '_func_info', None)
            if func_info is not None:
                func_info.llvm_function = llvm_function
                func_info.is_compiled = True

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
            # Write IR to temp file for debugging
            import tempfile
            import os
            try:
                ir_str = str(self.module)
                # Create temp file in system temp directory
                fd, ir_path = tempfile.mkstemp(suffix='.ll', prefix='pythoc_error_')
                with os.fdopen(fd, 'w') as f:
                    f.write(ir_str)
                logger.error(f"Verification failed: {e}\nModule IR written to: {ir_path}")
            except Exception as write_err:
                logger.error(f"Verification failed: {e}\n(Failed to write IR: {write_err})")
            raise e
        return True
    
    def optimize_module(self, optimization_level: int = 2):
        """Optimize the LLVM module.

        The pipeline runs multiple FPM<->MPM rounds so that inlining exposes
        further simplification opportunities (SROA, InstCombine, etc.).

        Supports both the legacy pass manager (llvmlite <0.45) and the new
        pass manager (llvmlite >=0.45) transparently.
        """
        if self.module is None:
            return

        try:
            llvm_module = binding.parse_assembly(str(self.module))
            ir_text = str(llvm_module)

            rounds = 1 if optimization_level <= 1 else 2
            inline_threshold = 225 if optimization_level <= 2 else 500

            for _ in range(rounds):
                ir_text = self._run_function_passes(ir_text, optimization_level)
                ir_text = self._run_module_passes(
                    ir_text, optimization_level, inline_threshold)

            # Final FPM cleanup after the last inlining round
            ir_text = self._run_function_passes(ir_text, optimization_level)

            self._optimized_ir = ir_text

        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")

    @staticmethod
    def _run_function_passes(ir_text: str, opt_level: int) -> str:
        """Run function-level optimization passes."""
        if _USE_NEW_PM:
            return LLVMCompiler._run_function_passes_new(ir_text, opt_level)
        return LLVMCompiler._run_function_passes_legacy(ir_text, opt_level)

    @staticmethod
    def _run_module_passes(
        ir_text: str, opt_level: int, inline_threshold: int
    ) -> str:
        """Run module-level optimization passes."""
        if _USE_NEW_PM:
            return LLVMCompiler._run_module_passes_new(
                ir_text, opt_level, inline_threshold)
        return LLVMCompiler._run_module_passes_legacy(
            ir_text, opt_level, inline_threshold)

    # ---- Legacy pass manager (llvmlite <0.45, LLVM 14) ----

    @staticmethod
    def _run_function_passes_legacy(ir_text: str, opt_level: int) -> str:
        llvm_module = binding.parse_assembly(ir_text)
        fpm = binding.create_function_pass_manager(llvm_module)

        fpm.add_sroa_pass()
        fpm.add_instruction_combining_pass()
        fpm.add_cfg_simplification_pass()

        if opt_level >= 1:
            fpm.add_reassociate_expressions_pass()
            fpm.add_sroa_pass()
            fpm.add_instruction_combining_pass()

        if opt_level >= 2:
            fpm.add_gvn_pass()
            fpm.add_instruction_combining_pass()
            fpm.add_jump_threading_pass()
            fpm.add_cfg_simplification_pass()

            fpm.add_licm_pass()
            fpm.add_loop_rotate_pass()
            fpm.add_loop_deletion_pass()

            fpm.add_tail_call_elimination_pass()
            fpm.add_sroa_pass()
            fpm.add_instruction_combining_pass()
            fpm.add_gvn_pass()
            fpm.add_cfg_simplification_pass()
            fpm.add_dead_code_elimination_pass()

        if opt_level >= 3:
            fpm.add_loop_unroll_pass()
            fpm.add_instruction_combining_pass()
            fpm.add_cfg_simplification_pass()

        fpm.initialize()
        for func in llvm_module.functions:
            if not func.is_declaration:
                fpm.run(func)
        fpm.finalize()
        return str(llvm_module)

    @staticmethod
    def _run_module_passes_legacy(
        ir_text: str, opt_level: int, inline_threshold: int
    ) -> str:
        llvm_module = binding.parse_assembly(ir_text)
        pm = binding.create_module_pass_manager()

        if opt_level >= 1:
            pm.add_constant_merge_pass()
            pm.add_dead_arg_elimination_pass()

        if opt_level >= 2:
            pm.add_function_attrs_pass()
            pm.add_ipsccp_pass()
            pm.add_function_inlining_pass(inline_threshold)
            pm.add_global_optimizer_pass()
            pm.add_global_dce_pass()

        pm.run(llvm_module)
        return str(llvm_module)

    # ---- New pass manager (llvmlite >=0.45, LLVM 18+) ----

    @staticmethod
    def _make_pass_builder(llvm_module):
        """Create a PassBuilder for the new pass manager API."""
        target = binding.Target.from_triple(llvm_module.triple)
        tm = target.create_target_machine()
        pto = binding.create_pipeline_tuning_options()
        return binding.create_pass_builder(tm, pto)

    @staticmethod
    def _run_function_passes_new(ir_text: str, opt_level: int) -> str:
        llvm_module = binding.parse_assembly(ir_text)
        pb = LLVMCompiler._make_pass_builder(llvm_module)
        fpm = binding.create_new_function_pass_manager()

        fpm.add_sroa_pass()
        fpm.add_instruction_combine_pass()
        fpm.add_simplify_cfg_pass()

        if opt_level >= 1:
            fpm.add_reassociate_pass()
            fpm.add_sroa_pass()
            fpm.add_instruction_combine_pass()

        if opt_level >= 2:
            fpm.add_new_gvn_pass()
            fpm.add_instruction_combine_pass()
            fpm.add_jump_threading_pass()
            fpm.add_simplify_cfg_pass()

            # LICM is not exposed in the new PM; loop-rotate + delete suffice.
            fpm.add_loop_rotate_pass()
            fpm.add_loop_deletion_pass()

            fpm.add_tail_call_elimination_pass()
            fpm.add_sroa_pass()
            fpm.add_instruction_combine_pass()
            fpm.add_new_gvn_pass()
            fpm.add_simplify_cfg_pass()
            fpm.add_aggressive_dce_pass()

        if opt_level >= 3:
            fpm.add_loop_unroll_pass()
            fpm.add_instruction_combine_pass()
            fpm.add_simplify_cfg_pass()

        for func in llvm_module.functions:
            if not func.is_declaration:
                fpm.run(func, pb)
        return str(llvm_module)

    @staticmethod
    def _run_module_passes_new(
        ir_text: str, opt_level: int, inline_threshold: int
    ) -> str:
        llvm_module = binding.parse_assembly(ir_text)
        pb = LLVMCompiler._make_pass_builder(llvm_module)
        pm = binding.create_new_module_pass_manager()

        if opt_level >= 1:
            pm.add_constant_merge_pass()
            pm.add_dead_arg_elimination_pass()

        if opt_level >= 2:
            pm.add_post_order_function_attributes_pass()
            pm.add_ipsccp_pass()
            pm.add_always_inliner_pass()
            pm.add_global_opt_pass()
            pm.add_global_dead_code_eliminate_pass()

        pm.run(llvm_module, pb)
        return str(llvm_module)
    
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