import os
import atexit
from ..utils.link_utils import file_lock


class OutputManager:
    """
    Manages compilation groups and output file generation.
    
    A compilation group represents a set of functions compiled into the same
    .ll/.o/.so files. Functions are grouped by:
    - (source_file, None, None) for normal functions
    - (definition_file, scope, suffix) for suffix-specialized functions
    """
    
    def __init__(self):
        """Initialize output manager with empty group registry."""
        # Key: (source_file, scope_name, suffix) tuple
        # Value: dict with compiler, wrappers, file paths, etc.
        self._pending_groups = {}
        
        # Pending compilation callbacks: group_key -> [(callback, func_info), ...]
        # callback signature: (compiler) -> None
        self._pending_compilations = {}
        
        # Track if flush has been called (to avoid double compilation)
        self._flushed_groups = set()
    
    def get_or_create_group(self, group_key, compiler, ir_file, obj_file, so_file, 
                           source_file, skip_codegen=False):
        """
        Get existing group or create a new one.
        
        Args:
            group_key: (source_file, scope, suffix) tuple identifying the group
            compiler: LLVMCompiler instance for this group
            ir_file: Path to output .ll file
            obj_file: Path to output .o file
            so_file: Path to output .so file
            source_file: Original source file path
            skip_codegen: If True, skip code generation (files are up-to-date)
        
        Returns:
            dict: Group info with keys: compiler, wrappers, ir_file, obj_file, so_file, 
                  skip_codegen, source_file
        """
        if group_key not in self._pending_groups:
            # If skipping codegen, try to load existing IR
            if skip_codegen and os.path.exists(ir_file):
                try:
                    compiler.load_ir_from_file(ir_file)
                except Exception:
                    # If loading fails, force recompilation
                    skip_codegen = False
            
            self._pending_groups[group_key] = {
                'compiler': compiler,
                'wrappers': [],
                'source_file': source_file,
                'ir_file': ir_file,
                'obj_file': obj_file,
                'so_file': so_file,
                'skip_codegen': skip_codegen
            }
        
        return self._pending_groups[group_key]
    
    def add_wrapper_to_group(self, group_key, wrapper):
        """
        Add a compiled function wrapper to its group.
        
        Args:
            group_key: Group identifier
            wrapper: Function wrapper to add
        """
        if group_key in self._pending_groups:
            group = self._pending_groups[group_key]
            if not group['skip_codegen']:
                group['wrappers'].append(wrapper)
    
    def queue_compilation(self, group_key, callback, func_info):
        """
        Queue a function for deferred compilation.
        
        Args:
            group_key: Group identifier
            callback: Callable (compiler) -> None that compiles the function
            func_info: FunctionInfo for forward declaration
        """
        if group_key not in self._pending_compilations:
            self._pending_compilations[group_key] = []
        self._pending_compilations[group_key].append((callback, func_info))
    
    def _forward_declare_function(self, compiler, func_info):
        """
        Add forward declaration for a function in the module.
        
        Args:
            compiler: LLVMCompiler instance
            func_info: FunctionInfo with signature information
        """
        from llvmlite import ir
        
        func_name = func_info.mangled_name or func_info.name
        
        # Check if already declared
        try:
            compiler.module.get_global(func_name)
            return  # Already exists
        except KeyError:
            pass
        
        # Build LLVM function type from func_info
        module_context = compiler.module.context
        param_llvm_types = []
        for param_name in func_info.param_names:
            pc_type = func_info.param_type_hints.get(param_name)
            if pc_type and hasattr(pc_type, 'get_llvm_type'):
                param_llvm_types.append(pc_type.get_llvm_type(module_context))
            else:
                # Fallback to i32 if type unknown
                param_llvm_types.append(ir.IntType(32))
        
        if func_info.return_type_hint and hasattr(func_info.return_type_hint, 'get_llvm_type'):
            return_type = func_info.return_type_hint.get_llvm_type(module_context)
        else:
            return_type = ir.VoidType()
        
        # Use LLVMBuilder to declare function with proper ABI handling
        from ..builder import LLVMBuilder
        temp_builder = LLVMBuilder()
        func_wrapper = temp_builder.declare_function(
            compiler.module, func_name,
            param_llvm_types, return_type
        )
    
    def _compile_pending_for_group(self, group_key):
        """
        Compile all pending functions for a group using two-pass approach.
        
        Phase 1: Forward declare all functions
        Phase 2: Compile all function bodies
        
        Supports transitive effect propagation: if compiling a function body
        triggers generation of new suffix versions (e.g., b_get_value_mock),
        those new functions are also compiled in subsequent iterations.
        
        Args:
            group_key: Group identifier
            
        Returns:
            bool: True if compilation succeeded, False if failed
        """
        group = self._pending_groups.get(group_key)
        if not group:
            return True
        
        compiler = group['compiler']
        
        # Track all compiled func_infos to avoid re-compilation
        compiled_funcs = set()
        
        # Loop until no more pending compilations for this group
        # This handles transitive effect propagation where compiling one function
        # may trigger generation of new suffix versions
        while True:
            pending = self._pending_compilations.get(group_key, [])
            if not pending:
                break
            
            # Clear pending to avoid re-processing
            del self._pending_compilations[group_key]
            
            # Filter out already compiled functions
            new_pending = []
            for callback, func_info in pending:
                func_key = func_info.mangled_name or func_info.name
                if func_key not in compiled_funcs:
                    new_pending.append((callback, func_info))
                    compiled_funcs.add(func_key)
            
            if not new_pending:
                break
            
            # Phase 1: Forward declare all new functions
            for callback, func_info in new_pending:
                self._forward_declare_function(compiler, func_info)
            
            # Phase 2: Compile all new function bodies
            # Note: This may add more pending compilations to this group
            for callback, func_info in new_pending:
                callback(compiler)
        
        return True
    
    def flush_all(self):
        """
        Flush all pending output files to disk.
        
        This should be called before native execution to ensure
        all .ll and .o files have been written.
        
        Uses two-pass compilation to support mutual recursion:
        1. Forward declare all functions in each group
        2. Compile all function bodies
        3. Write .ll and .o files
        """
        # Check if any group has already loaded its library
        from ..native_executor import get_multi_so_executor
        executor = get_multi_so_executor()
        
        for group_key, group in self._pending_groups.items():
            source_file = group.get('source_file')
            if source_file and executor.has_loaded_library(source_file):
                # Check if this group has new pending compilations
                if group_key in self._pending_compilations and self._pending_compilations[group_key]:
                    raise RuntimeError(
                        f"Cannot compile new functions in '{source_file}' after native execution has started. "
                        f"All @compile decorators must be executed before calling any compiled functions."
                    )
        
        # Copy keys to avoid "dictionary changed size during iteration"
        group_keys = list(self._pending_groups.keys())
        for group_key in group_keys:
            group = self._pending_groups.get(group_key)
            if not group:
                continue
            if group.get('skip_codegen', False):
                # Already up-to-date
                continue
            
            # Skip if already flushed
            if group_key in self._flushed_groups:
                continue
            
            # Skip if marked as failed
            if group.get('compilation_failed', False):
                continue
            
            # Compile pending functions for this group (two-pass)
            try:
                self._compile_pending_for_group(group_key)
            except Exception:
                # Mark this group as failed and re-raise
                group['compilation_failed'] = True
                raise
            
            if not group.get('wrappers'):
                # No functions compiled
                continue
            
            compiler = group['compiler']
            
            # Use file lock to prevent concurrent compilation of the same module
            # This protects against parallel test runs compiling the same .o file
            # Note: We don't skip compilation even if .o exists, because each process
            # may generate different anonymous symbol names. The lock ensures only
            # one process writes at a time.
            obj_file = group['obj_file']
            lockfile_path = obj_file + '.lock'
            
            with file_lock(lockfile_path):
                # Verify module
                if not compiler.verify_module():
                    source_file, scope, suffix = group_key
                    raise RuntimeError(f"Module verification failed for group {group_key}")
                
                # Optimize
                opt_level = int(os.environ.get('PC_OPT_LEVEL', '2'))
                compiler.optimize_module(optimization_level=opt_level)
                
                # Write files
                compiler.save_ir_to_file(group['ir_file'])
                compiler.compile_to_object(group['obj_file'])
            
            # Mark this group as flushed
            self._flushed_groups.add(group_key)
            
            # Clear wrappers after flushing to mark this group as up-to-date
            group['wrappers'] = []
        
        # Don't clear pending groups - they serve as metadata cache for subsequent runs
    
    def get_group(self, group_key):
        """
        Get group info by key.
        
        Args:
            group_key: Group identifier
        
        Returns:
            dict or None: Group info if exists
        """
        return self._pending_groups.get(group_key)
    
    def clear_all(self):
        """Clear all pending groups (for testing/reset)."""
        self._pending_groups.clear()
        self._pending_compilations.clear()
        self._flushed_groups.clear()
    
    def clear_failed_group(self, group_key):
        """
        Clear a failed compilation group to allow retry or cleanup.
        
        Args:
            group_key: Group identifier to clear
        """
        if group_key in self._pending_groups:
            del self._pending_groups[group_key]
        if group_key in self._pending_compilations:
            del self._pending_compilations[group_key]
        if group_key in self._flushed_groups:
            self._flushed_groups.remove(group_key)


# Global singleton instance
_output_manager = OutputManager()

# Track if atexit handler is registered
_atexit_registered = False


def _atexit_flush():
    """
    Atexit handler to ensure all pending compilations are flushed.
    
    This guarantees that running a Python file with @compile decorators
    will always generate .ll and .o files, even if no compiled function
    is ever called.
    """
    try:
        _output_manager.flush_all()
    except Exception:
        # Silently ignore errors during atexit
        # (e.g., if program is terminating due to another error)
        pass


def _ensure_atexit_registered():
    """Register atexit handler if not already registered."""
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_atexit_flush)
        _atexit_registered = True


def get_output_manager():
    """Get the global OutputManager singleton."""
    _ensure_atexit_registered()
    return _output_manager


def flush_all_pending_outputs():
    """
    Convenience function to flush all pending outputs.
    
    This is the main entry point used by the runtime.
    """
    _output_manager.flush_all()


def clear_failed_group(group_key):
    """
    Clear a failed compilation group.
    
    This is useful for error testing where a group fails to compile
    and we want to clean up before the next test.
    
    Args:
        group_key: (source_file, scope, suffix) tuple
    """
    _output_manager.clear_failed_group(group_key)
