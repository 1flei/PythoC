import os
import atexit
from ..utils.link_utils import file_lock
from .deps import get_dependency_tracker, GroupKey


class OutputManager:
    """
    Manages compilation groups and output file generation.
    
    A compilation group represents a set of functions compiled into the same
    .ll/.o/.so files. Functions are grouped by a 4-tuple:
    - (source_file, scope, compile_suffix, effect_suffix)
    
    Supports both 3-tuple (legacy) and 4-tuple group keys for compatibility.
    """
    
    def __init__(self):
        """Initialize output manager with empty group registry."""
        # Key: group_key tuple (3 or 4 elements)
        # Value: dict with compiler, wrappers, file paths, etc.
        self._pending_groups = {}
        
        # All groups (including completed ones) for compile_to_executable lookup
        self._all_groups = {}
        
        # Pending compilation callbacks: group_key -> [(callback, func_info), ...]
        # callback signature: (compiler) -> None
        self._pending_compilations = {}
        
        # Track if flush has been called (to avoid double compilation)
        self._flushed_groups = set()
    
    def get_or_create_group(self, group_key, compiler, ir_file, obj_file, so_file, 
                           source_file):
        """
        Get existing group or create a new one.
        
        Args:
            group_key: Group key tuple (3 or 4 elements)
            compiler: LLVMCompiler instance for this group
            ir_file: Path to output .ll file
            obj_file: Path to output .o file
            so_file: Path to output .so file
            source_file: Original source file path
        
        Returns:
            dict: Group info with keys: compiler, wrappers, ir_file, obj_file, so_file, source_file
        """
        # Check if already in _all_groups (including completed groups)
        if group_key in self._all_groups:
            group = self._all_groups[group_key]
            in_flushed = group_key in self._flushed_groups

            # If the group was already flushed, it may still be safe to add more
            # functions as long as the corresponding shared library has NOT been
            # loaded yet. This is important for flows where internal native
            # execution (e.g. bindgen) triggers a flush mid-import, but the user
            # module continues to define more @compile functions afterwards.
            if in_flushed:
                from ..native_executor import get_multi_so_executor
                executor = get_multi_so_executor()

                existing_so_file = group.get('so_file')
                if existing_so_file and existing_so_file in executor.loaded_libs:
                    source_file_path = group_key[0] if group_key else 'unknown'
                    from ..logger import logger
                    logger.error(
                        f"Cannot define new compiled function after native execution has started. "
                        f"File '{source_file_path}' was already compiled and loaded. "
                        f"Move all @compile decorated functions before any code that triggers execution."
                    )
                    raise RuntimeError(
                        f"Cannot define new @compile function after module '{source_file_path}' "
                        f"has started native execution. All @compile functions must be defined "
                        f"before any compiled function is called."
                    )

                # Re-open the group for further compilation. We must also ensure
                # it is considered pending again, otherwise wrappers won't be
                # added. We also mark it as requiring recompilation, because the
                # on-disk .o might now be newer than the source file (due to an
                # earlier flush in the same process).
                if group_key in self._flushed_groups:
                    self._flushed_groups.remove(group_key)
                self._pending_groups[group_key] = group
                group['force_recompile'] = True

            return group
        
        if group_key not in self._pending_groups:
            group = {
                'compiler': compiler,
                'wrappers': [],
                'source_file': source_file,
                'ir_file': ir_file,
                'obj_file': obj_file,
                'so_file': so_file,
            }
            self._pending_groups[group_key] = group
            self._all_groups[group_key] = group
        
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
        from ..logger import logger
        logger.debug(f"queue_compilation: {func_info.name}, group_key={group_key}, total_pending={len(self._pending_compilations[group_key])}")
    
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
    
    def _compile_pending_for_group(self, group_key, group):
        """
        Compile all pending functions for a group using two-pass approach.
        
        Phase 1: Forward declare all functions
        Phase 2: Compile all function bodies
        
        Supports transitive effect propagation: if compiling a function body
        triggers generation of new suffix versions (e.g., b_get_value_mock),
        those new functions are also compiled in subsequent iterations.
        
        Args:
            group_key: Group identifier
            group: Group info dict
            
        Returns:
            bool: True if compilation succeeded, False if failed
        """
        if not group:
            return True
        
        compiler = group['compiler']
        from ..logger import logger
        logger.debug(f"_compile_pending_for_group: group_key={group_key}, pending={len(self._pending_compilations.get(group_key, []))}")
        
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
        
        Cache check is done here, at flush time:
        - If .o is up-to-date (newer than source), skip compilation
        - Otherwise, compile and regenerate .o
        """
        from ..logger import logger
        from .cache import BuildCache
        
        # Check if any group has already loaded its library
        from ..native_executor import get_multi_so_executor
        executor = get_multi_so_executor()
        
        for group_key, group in self._pending_groups.items():
            so_file = group.get('so_file')
            # Check if THIS SPECIFIC so_file is already loaded
            # (not just any library from the same source file)
            if so_file and so_file in executor.loaded_libs:
                # Check if this group has new pending compilations
                if group_key in self._pending_compilations and self._pending_compilations[group_key]:
                    source_file = group.get('source_file', so_file)
                    raise RuntimeError(
                        f"Cannot compile new functions in '{source_file}' after native execution has started. "
                        f"All @compile decorators must be executed before calling any compiled functions."
                    )
        
        # Process groups iteratively until stable
        # New groups may be added during compilation (transitive effect propagation)
        logger.debug(f"flush_all: _pending_groups={list(self._pending_groups.keys())}")
        logger.debug(f"flush_all: _pending_compilations={[(k, len(v)) for k,v in self._pending_compilations.items()]}")
        while self._pending_groups:
            group_key = next(iter(self._pending_groups))
            group = self._pending_groups.pop(group_key)
            
            # Skip if already flushed
            if group_key in self._flushed_groups:
                continue
            
            # Skip if marked as failed
            if group.get('compilation_failed', False):
                continue
            
            # Check cache: is .o up-to-date?
            # If the group was re-opened after an earlier flush in the same
            # process, we must force recompilation even if timestamps suggest a
            # cache hit.
            obj_file = group['obj_file']
            source_file = group.get('source_file')
            force_recompile = bool(group.get('force_recompile', False))
            if (not force_recompile) and source_file and BuildCache.check_obj_uptodate(obj_file, source_file):
                # Cache hit - .o is up-to-date, skip compilation
                # But we still need to restore dependencies from .deps file
                self._restore_deps_from_cache(group_key, group)
                self._flushed_groups.add(group_key)

                # Clear any newly-queued callbacks/wrappers for this group. The
                # on-disk .o already contains the compiled definitions.
                group['force_recompile'] = False
                if group_key in self._pending_compilations:
                    del self._pending_compilations[group_key]
                group['wrappers'] = []

                logger.debug(f"Cache hit for {group_key}, skipping compilation")
                continue
            
            # Cache miss - need to compile
            # Compile pending functions for this group (two-pass)
            try:
                self._compile_pending_for_group(group_key, group)
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
            lockfile_path = obj_file + '.lock'
            
            with file_lock(lockfile_path):
                # Verify module
                if not compiler.verify_module():
                    raise RuntimeError(f"Module verification failed for group {group_key}")
                
                # Save unoptimized IR if requested
                if os.environ.get('PC_SAVE_UNOPT_IR'):
                    unopt_ir_file = group['ir_file'].replace('.ll', '.unopt.ll')
                    with open(unopt_ir_file, 'w') as f:
                        f.write(str(compiler.module))
                
                # Optimize
                opt_level = int(os.environ.get('PC_OPT_LEVEL', '2'))
                compiler.optimize_module(optimization_level=opt_level)
                
                # Write files
                compiler.save_ir_to_file(group['ir_file'])
                compiler.compile_to_object(group['obj_file'])
                
                # Save dependency information
                self._save_group_deps(group_key, compiler, obj_file)
            
            # Mark this group as flushed
            self._flushed_groups.add(group_key)
            group['force_recompile'] = False
            
            # Clear wrappers after flushing to mark this group as up-to-date
            group['wrappers'] = []
        
        # Don't clear pending groups - they serve as metadata cache for subsequent runs
    
    def _restore_deps_from_cache(self, group_key, group):
        """
        Restore dependency information from .deps file on cache hit.
        
        This ensures that link libraries and objects are properly registered
        even when we skip compilation.
        
        Args:
            group_key: Group key tuple
            group: Group info dict
        """
        obj_file = group['obj_file']
        compiler = group['compiler']
        
        dep_tracker = get_dependency_tracker()
        deps = dep_tracker.load_deps(obj_file)
        if deps:
            # Restore imported_user_functions from group-level deps
            if not hasattr(compiler, 'imported_user_functions'):
                compiler.imported_user_functions = {}
            for group_dep in deps.group_dependencies:
                if group_dep.target_group:
                    # For group-level deps, we can't restore the exact function name
                    # but we can restore the file dependency
                    target_file = group_dep.target_group.file
                    # Use a placeholder name based on the group key
                    placeholder_name = f"group_{hash(group_dep.target_group.to_tuple()) & 0xFFFFFF:06x}"
                    compiler.imported_user_functions[placeholder_name] = target_file
            
            # Restore link libraries and objects to registry
            from ..registry import get_unified_registry
            registry = get_unified_registry()
            for lib in deps.link_libraries:
                registry.add_link_library(lib)
            for obj in deps.link_objects:
                registry.add_link_object(obj)
    
    def _save_group_deps(self, group_key, compiler, obj_file):
        """
        Save dependency information for a compiled group.
        
        Collects dependencies from compiler.imported_user_functions and
        link libraries/objects, then persists to .deps file.
        
        Args:
            group_key: Group key tuple
            compiler: LLVMCompiler with compilation results
            obj_file: Path to .o file
        """
        from .deps import get_dependency_tracker, GroupKey
        from ..registry import get_unified_registry
        
        dep_tracker = get_dependency_tracker()
        registry = get_unified_registry()
        
        # Get or create deps for this group
        group_deps = dep_tracker.get_or_create_group_deps(group_key)
        
        # Set source mtime
        group = self._pending_groups.get(group_key)
        if group and group.get('source_file'):
            source_file = group['source_file']
            if os.path.exists(source_file):
                group_deps.source_mtime = os.path.getmtime(source_file)
        
        # Collect dependencies from imported_user_functions
        if hasattr(compiler, 'imported_user_functions'):
            for dep_name, dep_source_file in compiler.imported_user_functions.items():
                func_info = registry.get_function_info(dep_name)
                if not func_info:
                    func_info = registry.get_function_info_by_mangled(dep_name)
                
                if func_info:
                    # Get the group key from wrapper if available
                    dep_group_key = None
                    if hasattr(func_info, 'wrapper') and func_info.wrapper:
                        wrapper = func_info.wrapper
                        if hasattr(wrapper, '_group_key'):
                            dep_group_key = GroupKey.from_tuple(wrapper._group_key)
                    
                    # Check if it's an extern function
                    is_extern = getattr(func_info, 'is_extern', False)
                    
                    if is_extern:
                        # Record external library dependency at group level
                        libraries = []
                        if hasattr(func_info, 'library') and func_info.library:
                            libraries = [func_info.library]
                        dep_tracker.record_extern_dependency(group_key, libraries)
                    else:
                        # Record group dependency
                        if dep_group_key:
                            dep_tracker.record_group_dependency(group_key, dep_group_key.to_tuple(), "function_call")
        
        # Add link libraries from registry
        for lib in registry.get_link_libraries():
            if lib not in group_deps.link_libraries:
                group_deps.link_libraries.append(lib)
        
        # Add link objects from registry
        for obj in registry.get_link_objects():
            if obj not in group_deps.link_objects:
                group_deps.link_objects.append(obj)
        
        # Save to file
        dep_tracker.save_deps(group_key, obj_file)
    
    def get_group(self, group_key):
        """
        Get group info by key.
        
        Args:
            group_key: Group identifier
        
        Returns:
            dict or None: Group info if exists
        """
        return self._all_groups.get(group_key)
    
    def get_all_groups(self):
        """
        Get all groups (including completed ones).
        
        Returns:
            dict: All groups mapping group_key -> group info
        """
        return self._all_groups
    
    def clear_all(self):
        """Clear all pending groups (for testing/reset)."""
        self._pending_groups.clear()
        self._all_groups.clear()
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
        if group_key in self._all_groups:
            del self._all_groups[group_key]
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
