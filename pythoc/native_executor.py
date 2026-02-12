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
from .build import BuildCache, get_dependency_tracker


class MultiSOExecutor:
    """Execute compiled LLVM functions by loading multiple shared libraries"""
    
    def __init__(self):
        self.loaded_libs = {}  # source_file -> ctypes.CDLL
        self.function_cache = {}  # func_name -> ctypes function wrapper
        self.lib_dependencies = {}  # source_file -> [dependent_source_files]
        self.lib_mtimes = {}  # source_file -> mtime when loaded
        # Keep `os.add_dll_directory()` handles alive on Windows (Python 3.8+)
        self._dll_dir_handles = []
        # Cache for transitive dependent .o files: (source_file, so_file) -> [obj_files]
        # Invalidated when a relink actually happens (not on every call).
        self._transitive_obj_cache: Dict[Tuple[str, str], List[str]] = {}
        # Cache: frozenset of dependency so_files that have been verified
        # up-to-date by _compile_dependencies_recursive.  Invalidated when any
        # relink occurs within this process.
        self._verified_deps: Set[str] = set()
        
    def compile_source_to_so(
        self,
        obj_file: str,
        so_file: str,
        extra_obj_files: Optional[List[str]] = None,
        extra_link_libraries: Optional[List[str]] = None,
    ) -> str:
        """Compile one compilation group's object file into a shared library.

        Args:
            obj_file: Path to the main object file for this group.
            so_file: Output shared library path.
            extra_obj_files: Optional extra object files to link in.
            extra_link_libraries: Optional extra libraries to link against.

        Returns:
            Path to the compiled shared library.
        """
        if not os.path.exists(obj_file):
            raise FileNotFoundError(f"Object file not found: {obj_file}")

        from .utils.link_utils import link_files

        all_objs = [obj_file]
        if extra_obj_files:
            for p in extra_obj_files:
                if p and p not in all_objs:
                    all_objs.append(p)

        result = link_files(all_objs, so_file, shared=True, link_libraries=extra_link_libraries)
        return result
    
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
        
        # For circular dependencies, we need to load all libraries even if they have undefined symbols.
        #
        # On Windows, the loader requires dependent DLLs to be discoverable at load time, so we must
        # load dependencies first (topological order) to populate the DLL search path.
        if sys.platform == 'win32':
            load_order = list(all_libs_to_load)
        else:
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
    
    def _derive_so_file_from_group_key(self, group_key) -> Optional[str]:
        """
        Derive .so file path from GroupKey without relying on registry.
        
        This allows cache hit to work even when dependent functions are not
        registered in the current process.
        
        Args:
            group_key: GroupKey object with file, scope, compile_suffix, effect_suffix
            
        Returns:
            Path to .so file, or None if cannot derive
        """
        from .utils.link_utils import get_shared_lib_extension
        
        source_file = group_key.file
        if not source_file:
            return None
        
        cwd = os.getcwd()
        
        # Calculate relative path for build directory
        if source_file.startswith(cwd + os.sep) or source_file.startswith(cwd + '/'):
            rel_path = os.path.relpath(source_file, cwd)
        else:
            # For files outside cwd, use a safe relative path based on filename
            base_name = os.path.splitext(os.path.basename(source_file))[0]
            rel_path = f"external/{base_name}.py"
        
        build_dir = os.path.join('build', os.path.dirname(rel_path))
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        
        # Get file suffix from group_key (e.g., "scope.compile_suffix.effect_suffix")
        file_suffix = group_key.get_file_suffix()
        
        if file_suffix:
            file_base = f"{base_name}.{file_suffix}"
        else:
            file_base = base_name
        
        lib_ext = get_shared_lib_extension()
        so_file = os.path.join(build_dir, file_base + lib_ext)
        
        return so_file
    
    def _get_library_dependencies(self, source_file: str, so_file: str) -> List[Tuple[str, str]]:
        """
        Get dependencies for a library using the deps system.
        
        First tries to load from .deps file (persisted deps), then falls back
        to imported_user_functions from compiler for compatibility.
        
        Args:
            source_file: Source file path
            so_file: Shared library path
            
        Returns:
            List of (dep_source_file, dep_so_file) tuples
        """
        from .registry import _unified_registry
        registry = _unified_registry
        
        dependencies = []
        seen_so_files = set()
        
        # Try to load deps from .deps file first
        lib_ext = get_shared_lib_extension()
        obj_file = so_file.replace(lib_ext, '.o')
        dep_tracker = get_dependency_tracker()
        deps = dep_tracker.load_deps(obj_file)
        
        if deps:
            # Use group-level deps system
            for group_dep in deps.group_dependencies:
                if group_dep.target_group:
                    dep_source_file = group_dep.target_group.file
                    # Derive so_file from group_key directly (not from registry)
                    # This ensures cache hit works even when dep function is not registered
                    dep_so_file = self._derive_so_file_from_group_key(group_dep.target_group)
                    if dep_so_file and dep_so_file != so_file and dep_so_file not in seen_so_files:
                        seen_so_files.add(dep_so_file)
                        dependencies.append((dep_source_file, dep_so_file))
            return dependencies
        
        # No deps file found - return empty (deps system is the only source of truth)
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
                # On Windows, dependent DLLs are *not* searched relative to the target DLL
                # in many modern configurations (Safe DLL search / Python 3.8+ behavior).
                # Ensure the directory containing `so_file` is on the DLL search path.
                if sys.platform == 'win32':
                    abs_path = os.path.abspath(so_file)
                    dll_dir = os.path.dirname(abs_path)
                    try:
                        if hasattr(os, 'add_dll_directory') and dll_dir:
                            self._dll_dir_handles.append(os.add_dll_directory(dll_dir))
                    except Exception:
                        # Fallback: best-effort PATH prepend
                        pass
                    if dll_dir and dll_dir not in os.environ.get('PATH', ''):
                        os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

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
                # For circular dependencies, missing/undefined imports might be
                # resolved later when other libraries are loaded. Don't fail
                # immediately; return None so the caller can retry.
                error_msg = str(e)

                if "undefined symbol" in error_msg or "symbol not found" in error_msg:
                    return None

                if sys.platform == 'win32':
                    # Windows reports missing dependent DLLs as "Could not find module ..."
                    # (or a localized equivalent). This can be transient when our
                    # dependency load order hits a cycle; retry after other DLLs
                    # have been loaded and their directories added to the search path.
                    missing_markers = [
                        # Missing dependent DLL
                        "Could not find module",
                        "The specified module could not be found",
                        "找不到指定的模块",
                        # Missing imported procedure (often due to load-order cycles)
                        "The specified procedure could not be found",
                        "找不到指定的程序",
                    ]
                    if any(m in error_msg for m in missing_markers):
                        return None


                raise RuntimeError(f"Failed to load library {so_file}: {e}")

            except Exception as e:
                raise RuntimeError(f"Failed to load library {so_file}: {e}")
    
    def get_function(self, func_name: str, compiler, so_file: str, wrapper=None) -> Callable:
        """
        Get a function from loaded libraries
        
        Args:
            func_name: Function name
            compiler: LLVMCompiler instance (for signature info)
            so_file: SO file containing the function
            wrapper: Optional @compile wrapper with _func_info attribute
            
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
        
        # Get function signature - prefer wrapper's func_info over registry lookup
        signature = self._get_function_signature(func_name, compiler, wrapper=wrapper)
        if signature is None:
            raise RuntimeError(f"Function {func_name} not found in module")
        
        return_type, param_types = signature
        
        # Filter out None types (linear/zero-size types) from param_types
        # Keep track of which indices have real types
        real_param_indices = []
        real_param_types = []
        for i, pt in enumerate(param_types):
            if pt is not None:
                real_param_indices.append(i)
                real_param_types.append(pt)
        
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
        
        # Set function signature (only real types)
        native_func.restype = return_type
        native_func.argtypes = real_param_types
        
        # Create wrapper that filters out linear args
        def wrapper(*args):
            # Filter args to only include those at real_param_indices
            filtered_args = [args[i] for i in real_param_indices if i < len(args)]
            
            c_args = []
            for arg, param_type in zip(filtered_args, real_param_types):
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
    
    def _get_function_signature(self, func_name: str, compiler, wrapper=None) -> Optional[Tuple]:
        """Get function signature from wrapper or registry using pythoc types.
        
        Uses pythoc types for correct ctypes mapping,
        especially for signed/unsigned distinction that LLVM IR doesn't preserve.
        
        Args:
            func_name: Function name (for error messages)
            compiler: LLVMCompiler instance
            wrapper: Optional @compile wrapper with _func_info attribute
        """
        # Prefer getting func_info directly from wrapper
        func_info = None
        if wrapper is not None:
            func_info = getattr(wrapper, '_func_info', None)
        
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
        if source_file.startswith(cwd + os.sep) or source_file.startswith(cwd + '/'):
            rel_path = os.path.relpath(source_file, cwd)
        else:
            # For files outside cwd, use a safe relative path
            base_name = os.path.splitext(os.path.basename(source_file))[0]
            rel_path = f"external/{base_name}"
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
        lib_ext = get_shared_lib_extension()
        obj_file = so_file.replace(lib_ext, '.o')

        # Collect dependencies from persisted deps graph (source of truth)
        dependencies = self._get_dependencies(wrapper._compiler, source_file, so_file)

        # On Windows, linking dependent groups' raw `.o` files into every DLL can
        # duplicate global/static state across DLLs (e.g., function-local `static`),
        # which breaks tests like `test_cimport_effect`. Prefer linking against
        # dependency import libraries instead.
        extra_link_libraries: Optional[List[str]] = None
        if sys.platform == 'win32':
            # Ensure dependencies are built first so their import libs exist.
            self._compile_dependencies_recursive(dependencies, set())
            # Merge dependency DLLs with this group's persisted link libraries
            # (e.g., libraries from @extern(lib=...) registered via cimport).
            dep_libs = [dep_so for _, dep_so in dependencies if dep_so != so_file]
            persisted_libs = self._get_persisted_link_libraries(obj_file)
            extra_link_libraries = dep_libs + [l for l in persisted_libs if l not in dep_libs]

        # Use BuildCache to check if .so needs re-linking.
        #
        # IMPORTANT (Windows): even though we *link* against dependency import libraries
        # (not by embedding dependent `.o` files), the correct import table depends on
        # which dependency groups are being referenced. Those dependency `.o` files are
        # the upstream inputs that produce their `.dll`/import-libs, so include them in
        # the relink input set to avoid stale linkage.
        relink_inputs = [obj_file]
        deps_file = obj_file.replace('.o', '.deps')
        if os.path.exists(deps_file):
            relink_inputs.append(deps_file)

        if sys.platform == 'win32':
            dep_obj_files = self._collect_dependent_obj_files_transitive(source_file, so_file)
            relink_inputs += dep_obj_files
            # Also include dependency `.deps` so changes in the dependency graph
            # (which affect the import table) trigger a relink.
            for dep_obj in dep_obj_files:
                dep_deps = dep_obj.replace('.o', '.deps')
                if os.path.exists(dep_deps):
                    relink_inputs.append(dep_deps)

        need_compile = BuildCache.check_so_needs_relink(so_file, relink_inputs)

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
            self.compile_source_to_so(obj_file, so_file, extra_link_libraries=extra_link_libraries)

        # Compile dependencies recursively if needed (non-Windows)
        if sys.platform != 'win32':
            self._compile_dependencies_recursive(dependencies, set())
        
        # Load library with dependencies
        self.load_library_with_dependencies(source_file, so_file, dependencies)
        
        # Get and cache the native function - pass wrapper for direct func_info access
        native_func = self.get_function(actual_func_name, compiler, so_file, wrapper=wrapper)
        return native_func
    
    def _collect_dependent_obj_files(self, so_file: str, source_file: str, visited: Set[str]) -> List[str]:
        """
        Collect all object files that a .so depends on.
        
        This includes the main .o file and all transitive dependencies.
        Used by BuildCache.check_so_needs_relink() to determine if re-linking is needed.
        
        Args:
            so_file: Shared library path
            source_file: Source file path
            visited: Set of already visited so_files
            
        Returns:
            List of .o file paths
        """
        lib_ext = get_shared_lib_extension()
        obj_file = so_file.replace(lib_ext, '.o')
        obj_files = [obj_file]
        
        if so_file in visited:
            return obj_files
        visited.add(so_file)
        
        # Get dependencies and add their .o files
        dependencies = self._get_library_dependencies(source_file, so_file)
        for dep_source_file, dep_so_file in dependencies:
            dep_obj_file = dep_so_file.replace(lib_ext, '.o')
            if os.path.exists(dep_obj_file) and dep_obj_file not in obj_files:
                obj_files.append(dep_obj_file)
            # Note: We don't recurse here because check_so_needs_relink only needs
            # direct dependencies - transitive deps are handled by their own .so files
        
        return obj_files

    def _collect_dependent_obj_files_transitive(self, source_file: str, so_file: str) -> List[str]:
        """Collect dependent groups' object files recursively.

        This follows the persisted group-dependency graph (from `.deps` files)
        and returns a de-duplicated list of dependency `.o` files.

        NOTE: The returned list does NOT include this group's own `.o`.

        Results are cached per (source_file, so_file) for the duration of the
        current top-level execute_function() call to avoid redundant DFS
        traversals and .deps file I/O.
        """
        cache_key = (source_file, so_file)
        if cache_key in self._transitive_obj_cache:
            return self._transitive_obj_cache[cache_key]

        lib_ext = get_shared_lib_extension()
        visited_so: Set[str] = set()
        result: List[str] = []

        def dfs(cur_source: str, cur_so: str):
            deps = self._get_library_dependencies(cur_source, cur_so)
            for dep_source_file, dep_so_file in deps:
                if dep_so_file in visited_so:
                    continue
                visited_so.add(dep_so_file)

                dep_obj_file = dep_so_file.replace(lib_ext, '.o')
                if os.path.exists(dep_obj_file) and dep_obj_file not in result:
                    result.append(dep_obj_file)

                dfs(dep_source_file, dep_so_file)

        dfs(source_file, so_file)
        self._transitive_obj_cache[cache_key] = result
        return result
    
    def _compile_dependencies_recursive(self, dependencies: List[Tuple[str, str]], visited: Set[str]):
        """
        Recursively compile all dependencies before loading
        
        This ensures all shared library files exist before we try to load them.

        On Windows, linking is parallelised because each ``zig cc`` invocation
        has ~0.5 s of startup overhead, and a large dependency graph can contain
        50–100 DLLs.  Stub import libraries are pre-generated so that all link
        jobs can run concurrently without circular-dependency deadlocks.
        
        Args:
            dependencies: List of (dep_source_file, dep_so_file) tuples
            visited: Set of already processed so_files to avoid infinite loops
        """
        lib_ext = get_shared_lib_extension()

        # ── Fast path: if ALL direct dependency so_files have already been
        # verified up-to-date in this process, skip the entire DFS.
        dep_so_set = frozenset(dep_so for _, dep_so in dependencies)
        if dep_so_set and dep_so_set.issubset(self._verified_deps):
            return



        # Windows: pre-generate stub import libraries for all dependencies that
        # have an .o file but no .lib yet. This breaks circular dependency
        # deadlocks where DLL A needs B.lib to link and B needs A.lib.
        # Stub .libs are generated from .o symbol tables via dlltool, allowing
        # the linker to proceed even before the actual .dll is built.
        if sys.platform == 'win32':
            self._ensure_stub_implibs(dependencies, set())

        # Collect all link jobs first (DFS), then execute in parallel on Windows.
        link_jobs: List[Tuple[str, str, Optional[List[str]]]] = []  # (obj, so, libs)

        def _collect_jobs(deps):
            for dep_source_file, dep_so_file in deps:
                if dep_so_file in visited:
                    continue
                visited.add(dep_so_file)

                dep_deps = self._get_library_dependencies(dep_source_file, dep_so_file)
                dep_obj_file = dep_so_file.replace(lib_ext, '.o')

                # Recursively collect this dependency's dependencies first
                if dep_deps:
                    _collect_jobs(dep_deps)

                # Check if this dependency needs linking
                if os.path.exists(dep_obj_file):
                    dep_link_libraries: Optional[List[str]] = None
                    if sys.platform == 'win32':
                        dep_link_libraries = [d_so for _, d_so in dep_deps if d_so != dep_so_file]

                    relink_inputs = [dep_obj_file]
                    dep_deps_file = dep_obj_file.replace('.o', '.deps')
                    if os.path.exists(dep_deps_file):
                        relink_inputs.append(dep_deps_file)

                    if sys.platform == 'win32':
                        dep_obj_files = self._collect_dependent_obj_files_transitive(dep_source_file, dep_so_file)
                        relink_inputs += dep_obj_files
                        for trans_obj in dep_obj_files:
                            trans_deps = trans_obj.replace('.o', '.deps')
                            if os.path.exists(trans_deps):
                                relink_inputs.append(trans_deps)

                    if BuildCache.check_so_needs_relink(dep_so_file, relink_inputs):
                        link_jobs.append((dep_obj_file, dep_so_file, dep_link_libraries))

        _collect_jobs(dependencies)

        if not link_jobs:
            # Everything is up-to-date — remember all visited so_files so that
            # future calls with the same (or subset of) dependencies can skip
            # the DFS entirely.
            self._verified_deps.update(visited)
            return

        # On Windows, run link jobs in parallel to amortise zig startup overhead.
        if sys.platform == 'win32' and len(link_jobs) > 1:
            import concurrent.futures
            max_workers = min(len(link_jobs), os.cpu_count() or 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self.compile_source_to_so, obj, so, extra_link_libraries=libs): so
                    for obj, so, libs in link_jobs
                }
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()  # propagate exceptions
        else:
            for obj, so, libs in link_jobs:
                self.compile_source_to_so(obj, so, extra_link_libraries=libs)

        # Linking happened — invalidate caches for the so_files that were
        # actually relinked, but keep verified status for unaffected DLLs.
        relinked_sos = {so for _, so, _ in link_jobs}
        self._verified_deps -= relinked_sos
        # Clear transitive obj cache entries that may reference relinked DLLs.
        stale_keys = [k for k in self._transitive_obj_cache if k[1] in relinked_sos]
        for k in stale_keys:
            del self._transitive_obj_cache[k]
        # Mark newly-linked so_files as verified (they're now up-to-date).
        self._verified_deps.update(visited)

    def _ensure_stub_implibs(self, dependencies: List[Tuple[str, str]], visited: Set[str]):
        """Pre-generate stub import libraries for the entire dependency graph.

        On Windows, the PE/COFF linker requires all imported symbols to be
        resolvable at link time.  When two DLLs have circular dependencies
        (e.g. c_ast ↔ linear_wrapper), neither can be linked first because
        each needs the other's ``.lib``.

        We break this deadlock by scanning *all* ``.o`` files reachable from
        ``dependencies`` and generating stub ``.lib`` files via ``dlltool``
        before any actual linking happens.  The stubs are created from the
        ``.o`` symbol table and a ``.def`` file, so they don't need the real
        DLL to exist.

        Stub generation is parallelised when there are many jobs to reduce
        the impact of per-invocation subprocess startup overhead on Windows.
        """
        from .utils.link_utils import _generate_stub_implib

        lib_ext = get_shared_lib_extension()

        # Collect all stub jobs via DFS first, then execute.
        stub_jobs: List[Tuple[str, str, str]] = []  # (obj, dll_name, implib)

        def _collect(deps):
            for dep_source_file, dep_so_file in deps:
                if dep_so_file in visited:
                    continue
                visited.add(dep_so_file)

                dep_obj_file = dep_so_file.replace(lib_ext, '.o')
                implib = os.path.splitext(dep_so_file)[0] + '.lib'

                if os.path.exists(dep_obj_file) and not os.path.exists(implib):
                    stub_jobs.append((dep_obj_file, os.path.basename(dep_so_file), implib))

                dep_deps = self._get_library_dependencies(dep_source_file, dep_so_file)
                if dep_deps:
                    _collect(dep_deps)

        _collect(dependencies)

        if not stub_jobs:
            return

        if len(stub_jobs) > 1:
            import concurrent.futures
            max_workers = min(len(stub_jobs), os.cpu_count() or 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(_generate_stub_implib, obj, dll, imp)
                    for obj, dll, imp in stub_jobs
                ]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
        else:
            obj, dll, imp = stub_jobs[0]
            _generate_stub_implib(obj, dll, imp)
    
    def _get_persisted_link_libraries(self, obj_file: str) -> List[str]:
        """Read link_libraries from the persisted .deps file for a group.

        These are libraries registered via ``@extern(lib=...)`` or ``cimport``
        and stored in the group's ``.deps`` file.  On Windows we need them at
        link time in addition to the inter-group dependency DLLs.
        """
        dep_tracker = get_dependency_tracker()
        deps = dep_tracker.load_deps(obj_file)
        if deps and deps.link_libraries:
            return list(deps.link_libraries)
        return []

    def _get_dependencies(self, compiler, source_file: str, so_file: str) -> List[Tuple[str, str]]:
        """
        Collect dependencies for a compiled function.
        
        Uses the deps system if available, falls back to imported_user_functions.
        
        Args:
            compiler: LLVMCompiler instance
            source_file: Source file path
            so_file: Shared library path
            
        Returns:
            List of (dep_source_file, dep_so_file) tuples
        """
        # Use the unified _get_library_dependencies which handles deps system
        return self._get_library_dependencies(source_file, so_file)


# Global executor instance
_multi_so_executor = None


def get_multi_so_executor() -> MultiSOExecutor:
    """Get or create the global multi-SO executor"""
    global _multi_so_executor
    if _multi_so_executor is None:
        _multi_so_executor = MultiSOExecutor()
    return _multi_so_executor
