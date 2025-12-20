# -*- coding: utf-8 -*-
"""
Native Executor V2 - Load multiple shared libraries for multi-file scenarios

This version creates one shared library per source file and loads them all with proper dependency order.
"""

import os
import sys
import ctypes
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from llvmlite import ir

from .utils.link_utils import get_shared_lib_extension


class MultiSOExecutor:
    """Execute compiled LLVM functions by loading multiple shared libraries"""
    
    def __init__(self):
        self.loaded_libs = {}  # source_file -> ctypes.CDLL
        self.function_cache = {}  # func_name -> ctypes function wrapper
        self.lib_dependencies = {}  # source_file -> [dependent_source_files]
        self.lib_mtimes = {}  # source_file -> mtime when loaded
        
    def compile_source_to_so(self, obj_file: str, so_file: str) -> str:
        """
        Compile object file(s) to shared library
        
        This method will link the specified object file along with any related
        object files (e.g., functions with suffix from the same source file).
        
        Args:
            obj_file: Path to main object file
            so_file: Path for output shared library
            
        Returns:
            Path to the compiled shared library
        """
        if not os.path.exists(obj_file):
            raise FileNotFoundError(f"Object file not found: {obj_file}")
        
        # Find all related object files
        # Pattern: base_name.*.o files in the same directory
        obj_dir = os.path.dirname(obj_file)
        base_name = os.path.basename(obj_file).replace('.o', '')
        
        # Collect all related .o files
        obj_files = [obj_file]
        if obj_dir:
            # Look for files matching: base_name.*.o (e.g., test.func.suffix.o)
            import glob
            pattern = os.path.join(obj_dir, f"{base_name}.*.o")
            related_files = glob.glob(pattern)
            for f in related_files:
                if f != obj_file and os.path.exists(f):
                    obj_files.append(f)
        
        # Use unified linker
        from .utils.link_utils import link_files
        return link_files(obj_files, so_file, shared=True)
    
    def load_library_with_dependencies(self, source_file: str, so_file: str, 
                                      dependencies: List[str]) -> ctypes.CDLL:
        """
        Load a shared library and its dependencies in correct order
        
        Handles circular dependencies by using topological sort and RTLD_LAZY | RTLD_GLOBAL.
        
        Args:
            source_file: Source file path (for caching)
            so_file: Path to shared library
            dependencies: List of (dep_source_file, dep_so_file) tuples
            
        Returns:
            Loaded library handle
        """
        # Flush all pending output files before loading
        from .decorators.compile import flush_all_pending_outputs
        flush_all_pending_outputs()
        
        # Filter out self-dependencies (a library shouldn't depend on itself)
        filtered_deps = [(dep_src, dep_so) for dep_src, dep_so in dependencies if dep_so != so_file]
        
        # Check if we need to reload the library (file was modified)
        need_reload = False
        if so_file in self.loaded_libs:
            if os.path.exists(so_file):
                current_mtime = os.path.getmtime(so_file)
                cached_mtime = self.lib_mtimes.get(so_file, 0)
                if current_mtime > cached_mtime:
                    need_reload = True
                    # Clear cached functions for this library
                    keys_to_remove = [k for k in self.function_cache.keys() if k.startswith(f"{so_file}:")]
                    for key in keys_to_remove:
                        del self.function_cache[key]
        
        # Recursively load dependencies with their own dependencies
        # Use a set to track what we're loading to detect circular dependencies
        loading_stack = set()
        self._load_with_recursive_deps(so_file, filtered_deps, loading_stack, need_reload)
        
        return self.loaded_libs[so_file]
    
    def _load_with_recursive_deps(self, so_file: str, dependencies: List[Tuple[str, str]], 
                                   loading_stack: Set[str], force_reload: bool = False):
        """
        Recursively load a library and all its dependencies
        
        For circular dependencies, we collect all libraries first, then load them in order.
        RTLD_LAZY | RTLD_GLOBAL allows symbols to be resolved across circular dependencies.
        
        Args:
            so_file: Path to shared library to load
            dependencies: Direct dependencies of this library
            loading_stack: Set of libraries currently being loaded (for cycle detection)
            force_reload: Whether to force reload even if already loaded
        """
        # Collect all libraries in dependency graph (including circular ones)
        all_libs_to_load = []
        visited = set()
        
        # First collect all dependencies (regardless of whether .so exists yet)
        for dep_source, dep_so in dependencies:
            if dep_so not in visited:
                dep_deps = self._get_library_dependencies(dep_source, dep_so)
                self._collect_all_libs(dep_so, dep_deps, visited, all_libs_to_load)
        
        # Then add the main library itself (after all dependencies)
        if so_file not in visited:
            all_libs_to_load.append(so_file)
        
        # For circular dependencies, we need to load all libraries even if they have undefined symbols
        # Strategy: Try loading in reverse order, if a library fails due to undefined symbols,
        # skip it and try loading other libraries first, then retry failed ones
        load_order = list(reversed(all_libs_to_load))
        # First pass: try to load all libraries
        failed_libs = []
        for lib_file in load_order:
            if lib_file not in self.loaded_libs or (lib_file == so_file and force_reload):
                if os.path.exists(lib_file):
                    result = self._load_single_library(lib_file, lib_file)
                    if result is None:
                        # Failed due to undefined symbols, will retry later
                        failed_libs.append(lib_file)
        
        # Second pass: retry failed libraries (their symbols might now be available)
        for lib_file in failed_libs:
            self._load_single_library(lib_file, lib_file)
        
        # Return the main library
        if so_file not in self.loaded_libs:
            raise RuntimeError(f"Failed to load main library {so_file}")
        return self.loaded_libs[so_file]
    
    def _collect_all_libs(self, so_file: str, dependencies: List[Tuple[str, str]], 
                          visited: Set[str], result: List[str]):
        """
        Collect all libraries in dependency graph using DFS
        
        Args:
            so_file: Current library to process
            dependencies: Direct dependencies of current library  
            visited: Set of already visited libraries
            result: List to append libraries to (in post-order)
        """
        if so_file in visited:
            return
        
        visited.add(so_file)
        
        # First recursively collect dependencies (regardless of whether .so exists yet)
        for dep_source, dep_so in dependencies:
            if dep_so not in visited:
                dep_deps = self._get_library_dependencies(dep_source, dep_so)
                self._collect_all_libs(dep_so, dep_deps, visited, result)
        
        # Add current library after its dependencies (post-order)
        result.append(so_file)
    
    def _get_library_dependencies(self, source_file: str, so_file: str) -> List[Tuple[str, str]]:
        """
        Get dependencies for a library by inspecting its source file's compiler
        
        Args:
            source_file: Source file path
            so_file: Shared library path
            
        Returns:
            List of (dep_source_file, dep_so_file) tuples
        """
        from .registry import _unified_registry
        registry = _unified_registry
        
        # Try to get compiler for this source file
        compiler = registry.get_compiler(source_file)
        if not compiler:
            return []
        
        dependencies = []
        if hasattr(compiler, 'imported_user_functions'):
            for dep_func_name, dep_source_file in compiler.imported_user_functions.items():
                # Get function info to find its so_file
                func_info = registry.get_function_info(dep_func_name)
                if not func_info:
                    func_info = registry.get_function_info_by_mangled(dep_func_name)
                
                if func_info and func_info.so_file:
                    dep_so_file = func_info.so_file
                    # Avoid self-dependency
                    if dep_so_file != so_file:
                        dependencies.append((dep_source_file, dep_so_file))
        
        return dependencies
    
    def _load_library_macos_lazy(self, so_file: str) -> ctypes.CDLL:
        """
        Load library on macOS using libc's dlopen with true RTLD_LAZY support.
        
        Python's ctypes.CDLL and _ctypes.dlopen on macOS force RTLD_NOW even when
        RTLD_LAZY is specified (mode becomes 0xB instead of 0x9), breaking circular
        dependencies. We bypass this by calling libc.dlopen directly.
        
        Args:
            so_file: Path to shared library
            
        Returns:
            ctypes.CDLL wrapper around the loaded library
            
        Raises:
            OSError: If dlopen fails
        """
        # Get libc (load current process to access system dlopen)
        libc = ctypes.CDLL(None)
        
        # Set up dlopen function signature
        libc.dlopen.argtypes = [ctypes.c_char_p, ctypes.c_int]
        libc.dlopen.restype = ctypes.c_void_p
        
        # Set up dlerror for error reporting
        libc.dlerror.argtypes = []
        libc.dlerror.restype = ctypes.c_char_p
        
        # RTLD constants for macOS
        RTLD_LAZY = 0x1     # Lazy symbol resolution
        RTLD_GLOBAL = 0x8   # Make symbols globally available
        
        # Convert path to absolute to avoid search path issues
        abs_path = os.path.abspath(so_file)
        
        # Call dlopen directly with true RTLD_LAZY
        handle = libc.dlopen(abs_path.encode('utf-8'), RTLD_LAZY | RTLD_GLOBAL)
        
        if not handle:
            # Get error message from dlerror
            error = libc.dlerror()
            error_msg = error.decode('utf-8') if error else 'unknown error'
            raise OSError(f"dlopen failed: {error_msg}")
        
        # Wrap the handle in a CDLL-like object
        # We create a custom wrapper since we can't use CDLL's handle parameter reliably
        class LibraryHandle:
            """Wrapper for dlopen handle that provides CDLL-like interface"""
            def __init__(self, handle, path):
                self._handle = handle
                self._name = path
                self._func_cache = {}
                self._libc = libc
            
            def __getattr__(self, name):
                # Avoid recursion for private attributes
                if name.startswith('_'):
                    raise AttributeError(name)
                
                # Check cache
                if name in self._func_cache:
                    return self._func_cache[name]
                
                # Look up symbol using dlsym
                self._libc.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
                self._libc.dlsym.restype = ctypes.c_void_p
                
                addr = self._libc.dlsym(self._handle, name.encode('utf-8'))
                if not addr:
                    error = self._libc.dlerror()
                    error_msg = error.decode('utf-8') if error else f'symbol {name} not found'
                    raise AttributeError(error_msg)
                
                # Create a ctypes function object from the address
                # Start with a generic function pointer, caller will set argtypes/restype
                func = ctypes.CFUNCTYPE(ctypes.c_int)(addr)
                
                # Store the raw address as an attribute for compatibility
                func._address = addr
                
                # Cache and return
                self._func_cache[name] = func
                return func
        
        return LibraryHandle(handle, abs_path)
    
    def _load_single_library(self, lib_key: str, so_file: str) -> ctypes.CDLL:
        """Load a single shared library"""
        if not os.path.exists(so_file):
            raise FileNotFoundError(f"Shared library not found: {so_file}")
        
        from .utils.link_utils import file_lock
        lockfile_path = so_file + '.lock'
        
        with file_lock(lockfile_path):
            try:
                # On macOS, ctypes.CDLL forces RTLD_NOW even when RTLD_LAZY is specified,
                # breaking circular dependencies. Use libc.dlopen directly.
                if sys.platform == 'darwin' and hasattr(os, 'RTLD_LAZY'):
                    lib = self._load_library_macos_lazy(so_file)
                elif hasattr(os, 'RTLD_LAZY') and hasattr(os, 'RTLD_GLOBAL'):
                    lib = ctypes.CDLL(so_file, mode=os.RTLD_LAZY | os.RTLD_GLOBAL)
                elif hasattr(ctypes, 'RTLD_GLOBAL'):
                    lib = ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                else:
                    lib = ctypes.CDLL(so_file)
                
                self.loaded_libs[lib_key] = lib
                self.lib_mtimes[lib_key] = os.path.getmtime(so_file)
                return lib
                
            except OSError as e:
                # For circular dependencies, undefined symbols might be resolved later
                # when other libraries are loaded. Don't fail immediately.
                error_msg = str(e)
                if "undefined symbol" in error_msg or "symbol not found" in error_msg:
                    # Don't add to loaded_libs yet - will try again later
                    return None
                raise RuntimeError(f"Failed to load library {so_file}: {e}")
            except Exception as e:
                raise RuntimeError(f"Failed to load library {so_file}: {e}")
    
    def get_function(self, func_name: str, compiler, so_file: str) -> Callable:
        """
        Get a function from loaded libraries
        
        Args:
            func_name: Function name
            compiler: LLVMCompiler instance (for signature info)
            so_file: SO file containing the function
            
        Returns:
            Python callable wrapper
        """
        # Check cache
        cache_key = f"{so_file}:{func_name}"
        if cache_key in self.function_cache:
            return self.function_cache[cache_key]
        
        # Find which library contains this function
        lib = self.loaded_libs.get(so_file)
        if lib is None:
            raise RuntimeError(f"Library {so_file} not loaded")
        
        # Get function signature from compiler
        signature = self._get_function_signature(func_name, compiler)
        if signature is None:
            raise RuntimeError(f"Function {func_name} not found in module")
        
        return_type, param_types = signature
        
        # Get function from library
        try:
            native_func = getattr(lib, func_name)
        except AttributeError:
            # Try to find in other loaded libraries
            for other_lib in self.loaded_libs.values():
                try:
                    native_func = getattr(other_lib, func_name)
                    break
                except AttributeError:
                    continue
            else:
                raise RuntimeError(f"Function {func_name} not found in any loaded library")
        
        # Set function signature
        native_func.restype = return_type
        native_func.argtypes = param_types
        
        # Create wrapper
        def wrapper(*args):
            c_args = []
            for arg, param_type in zip(args, param_types):
                if param_type == ctypes.c_void_p:
                    if isinstance(arg, int):
                        c_args.append(arg)
                    elif hasattr(arg, 'value'):
                        c_args.append(arg.value)
                    else:
                        c_args.append(ctypes.cast(arg, ctypes.c_void_p).value)
                elif isinstance(arg, param_type):
                    # Already correct type (e.g., struct), pass as-is
                    c_args.append(arg)
                else:
                    # Convert to target type (e.g., int -> c_int32)
                    c_args.append(param_type(arg))
            
            result = native_func(*c_args)
            
            if return_type is None:
                return None
            elif return_type == ctypes.c_bool:
                return bool(result)
            else:
                return result
        
        self.function_cache[cache_key] = wrapper
        return wrapper
    
    def _get_function_signature(self, func_name: str, compiler) -> Optional[Tuple]:
        """Get function signature from registry using pythoc types.
        
        Uses pythoc types from registry to get correct ctypes mapping,
        especially for signed/unsigned distinction that LLVM IR doesn't preserve.
        """
        from .registry import get_unified_registry
        registry = get_unified_registry()
        func_info = registry.get_function_info(func_name)
        if not func_info:
            func_info = registry.get_function_info_by_mangled(func_name)
        
        if func_info:
            # Use pythoc types for accurate ctypes mapping
            return_type = self._pc_type_to_ctypes(func_info.return_type_hint)
            param_types = [
                self._pc_type_to_ctypes(func_info.param_type_hints.get(name))
                for name in func_info.param_names
            ]
            return (return_type, param_types)
        
        return None
    
    def _pc_type_to_ctypes(self, pc_type) -> Any:
        """Convert pythoc type to ctypes type.
        
        All pythoc types should implement get_ctypes_type() method.
        """
        if pc_type is None:
            return None
        
        if hasattr(pc_type, 'get_ctypes_type'):
            return pc_type.get_ctypes_type()
        
        # Fallback for unknown types
        return ctypes.c_void_p
    
    def clear(self):
        """Clear all loaded libraries and caches"""
        self.loaded_libs.clear()
        self.function_cache.clear()
        self.lib_dependencies.clear()
        self.lib_mtimes.clear()
    
    def has_loaded_library(self, source_file: str) -> bool:
        """Check if a library for the given source file is already loaded"""
        # Check if any loaded library corresponds to this source file
        # Build the expected shared library path from source file
        cwd = os.getcwd()
        if source_file.startswith(cwd):
            rel_path = os.path.relpath(source_file, cwd)
        else:
            rel_path = source_file
        lib_ext = get_shared_lib_extension()
        so_file = os.path.join('build', os.path.dirname(rel_path), 
                              os.path.splitext(os.path.basename(source_file))[0] + lib_ext)
        return so_file in self.loaded_libs
    
    def execute_function(self, wrapper) -> Callable:
        """
        Execute a compiled function - handles compilation, loading, and caching
        
        Args:
            wrapper: The wrapper function object with all compilation metadata
            
        Returns:
            Python callable wrapper for the native function
        """
        # Extract metadata from wrapper
        if not (hasattr(wrapper, '_so_file') and hasattr(wrapper, '_compiler')):
            raise RuntimeError(f"Function was not properly compiled (missing metadata)")
        
        source_file = wrapper._source_file
        so_file = wrapper._so_file
        compiler = wrapper._compiler
        func_name = wrapper._original_name
        actual_func_name = getattr(wrapper, '_actual_func_name', func_name)
        
        # Check if we need to reload (e.g., after clear_registry())
        if source_file in self.loaded_libs:
            # Library is loaded, but check if wrapper's cache is invalidated
            if not hasattr(wrapper, '_native_func'):
                # Cache was cleared, need to reload
                pass
        elif hasattr(wrapper, '_native_func'):
            # Library not loaded but wrapper has cache - clear it
            delattr(wrapper, '_native_func')
        
        # Check function cache using source_file:actual_func_name as key
        cache_key = f"{source_file}:{actual_func_name}"
        if cache_key in self.function_cache:
            return self.function_cache[cache_key]
        
        # Flush pending outputs before checking files
        from .build import flush_all_pending_outputs
        flush_all_pending_outputs()
        
        # Check if we need to compile shared library from .o
        need_compile = False
        lib_ext = get_shared_lib_extension()
        obj_file = so_file.replace(lib_ext, '.o')
        
        if not os.path.exists(so_file):
            need_compile = True
        elif not os.path.exists(obj_file):
            # If shared library exists but .o doesn't, something is wrong
            need_compile = False
        else:
            # Check both .ll and .o timestamps
            ll_file = so_file.replace(lib_ext, '.ll')
            so_mtime = os.path.getmtime(so_file)
            
            # If .ll is newer than .so, need to recompile
            if os.path.exists(ll_file):
                ll_mtime = os.path.getmtime(ll_file)
                if ll_mtime > so_mtime:
                    need_compile = True
            
            # If .o is newer than .so, also need to recompile
            obj_mtime = os.path.getmtime(obj_file)
            if obj_mtime > so_mtime:
                need_compile = True
        
        if need_compile:
            if not os.path.exists(obj_file):
                raise RuntimeError(f"Object file {obj_file} not found for {func_name}")
            # If recompiling, unload the old library first
            if so_file in self.loaded_libs:
                del self.loaded_libs[so_file]
                # Also clear cached functions from this library
                keys_to_remove = [k for k in self.function_cache.keys() if k.startswith(f"{source_file}:")]
                for key in keys_to_remove:
                    del self.function_cache[key]
            self.compile_source_to_so(obj_file, so_file)
        
        # Collect dependencies from compiler
        dependencies = self._get_dependencies(wrapper._compiler, source_file)
        
        # Compile dependencies recursively if needed
        self._compile_dependencies_recursive(dependencies, set())
        
        # Load library with dependencies
        self.load_library_with_dependencies(source_file, so_file, dependencies)
        
        # Get and cache the native function
        native_func = self.get_function(actual_func_name, compiler, so_file)
        return native_func
    
    def _compile_dependencies_recursive(self, dependencies: List[Tuple[str, str]], visited: Set[str]):
        """
        Recursively compile all dependencies before loading
        
        This ensures all shared library files exist before we try to load them.
        
        Args:
            dependencies: List of (dep_source_file, dep_so_file) tuples
            visited: Set of already processed so_files to avoid infinite loops
        """
        lib_ext = get_shared_lib_extension()
        for dep_source_file, dep_so_file in dependencies:
            if dep_so_file in visited:
                continue
            visited.add(dep_so_file)
            
            # First, recursively compile this dependency's dependencies
            dep_deps = self._get_library_dependencies(dep_source_file, dep_so_file)
            if dep_deps:
                self._compile_dependencies_recursive(dep_deps, visited)
            
            # Now compile this dependency if needed
            dep_obj_file = dep_so_file.replace(lib_ext, '.o')
            if os.path.exists(dep_obj_file):
                need_compile = False
                if not os.path.exists(dep_so_file):
                    need_compile = True
                else:
                    # Check if .o is newer than shared library
                    if os.path.getmtime(dep_obj_file) > os.path.getmtime(dep_so_file):
                        need_compile = True
                    # Also check .ll timestamp
                    dep_ll_file = dep_so_file.replace(lib_ext, '.ll')
                    if os.path.exists(dep_ll_file):
                        if os.path.getmtime(dep_ll_file) > os.path.getmtime(dep_so_file):
                            need_compile = True
                
                if need_compile:
                    self.compile_source_to_so(dep_obj_file, dep_so_file)
    
    def _get_dependencies(self, compiler, source_file: str) -> List[Tuple[str, str]]:
        """
        Collect dependencies for a compiled function
        
        Args:
            compiler: LLVMCompiler instance
            source_file: Source file path
            
        Returns:
            List of (dep_source_file, dep_so_file) tuples
        """
        dependencies = []
        if hasattr(compiler, 'imported_user_functions'):
            from .registry import _unified_registry
            registry = _unified_registry
            for dep_func_name, _dep_module in compiler.imported_user_functions.items():
                # Try to get function info by original name first, then by mangled name
                func_info = registry.get_function_info(dep_func_name)
                if not func_info:
                    func_info = registry.get_function_info_by_mangled(dep_func_name)
                
                if not func_info or not func_info.source_file:
                    continue
                
                # Use the so_file from func_info if available, otherwise compute it
                dep_source_file = func_info.source_file
                if func_info.so_file:
                    dep_so_file = func_info.so_file
                else:
                    # Fallback to old behavior for functions without so_file
                    cwd = os.getcwd()
                    rel_path = os.path.relpath(dep_source_file, cwd) if dep_source_file.startswith(cwd) else dep_source_file
                    lib_ext = get_shared_lib_extension()
                    dep_so_file = os.path.join('build', os.path.dirname(rel_path), os.path.splitext(os.path.basename(dep_source_file))[0] + lib_ext)
                
                # Always add dependency, will be compiled if needed
                dependencies.append((dep_source_file, dep_so_file))
        return dependencies


# Global executor instance
_multi_so_executor = None


def get_multi_so_executor() -> MultiSOExecutor:
    """Get or create the global multi-SO executor"""
    global _multi_so_executor
    if _multi_so_executor is None:
        _multi_so_executor = MultiSOExecutor()
    return _multi_so_executor
