"""
C Import (cimport) - Import C headers/sources as pythoc modules

This module provides the cimport() function to:
1. Parse C header/source files
2. Generate pythoc bindings
3. Import the bindings as a Python module
4. Optionally compile C sources and register for linking

Architecture:
- Can use optional libclang to read C declarations and layout-aware types
- Falls back to the compiled native pythoc bindings parser when libclang is not available

Usage:
    from pythoc.cimport import cimport
    
    # Header-only import with library
    libc = cimport('stdio.h', lib='c')
    
    # Import with source compilation
    mylib = cimport('mylib.h', sources=['mylib.c'], compile_sources=True)
    
    # Direct C source import
    mod = cimport('helper.c', lib='helper', compile_sources=True)
"""

import os
import sys
import hashlib
import importlib.util
import subprocess
import warnings
import shlex
from typing import Optional, List, Any
from types import ModuleType

from .registry import get_unified_registry
from .utils.cc_utils import compile_c_to_object, compile_c_sources, find_available_cc

# Delay import to avoid circular dependency
def _get_bindgen():
    from .bindings.bindgen import generate_bindings_to_file
    return generate_bindings_to_file


_VALID_CIMPORT_BACKENDS = {"auto", "clang", "native"}


def _normalize_cimport_backend(backend: Optional[str]) -> str:
    """Resolve backend request from API/env into auto|clang|native."""
    from .config import config
    requested = backend or config.cimport_backend

    # Compatibility knob for experiments: enable_clang_cimport=True forces clang
    # unless an explicit backend is requested via API or config.
    if requested is None:
        if config.enable_clang_cimport:
            requested = "clang"

    requested = (requested or "auto").strip().lower()
    if requested not in _VALID_CIMPORT_BACKENDS:
        valid = ", ".join(sorted(_VALID_CIMPORT_BACKENDS))
        raise ValueError(f"Invalid cimport backend '{requested}'. Expected one of: {valid}")
    return requested


def _clang_backend_available() -> bool:
    try:
        from .cimport_clang import is_clang_backend_available
    except Exception:
        return False
    return is_clang_backend_available()


def _select_cimport_backend(requested: str) -> str:
    if requested == "auto":
        return "clang" if _clang_backend_available() else "native"
    return requested


def _bindings_path_for_backend(cache_dir: str, basename: str, backend_name: str) -> str:
    if backend_name == "native":
        # Preserve the historical filename for the existing compiled bindgen path.
        return os.path.join(cache_dir, f"bindings_{basename}.py")
    return os.path.join(cache_dir, f"bindings_{backend_name}_{basename}.py")


def _bindings_need_regen(input_path: str, bindings_path: str) -> bool:
    if not os.path.exists(bindings_path):
        return True
    return os.path.getmtime(input_path) > os.path.getmtime(bindings_path)


def _normalize_lib_for_generated_source(lib: str) -> str:
    # IMPORTANT (Windows): absolute `lib` paths may contain backslashes.
    # If those are embedded into generated Python source like
    # `@extern(lib='C:\\Users\\...')`, sequences like `\U` can be parsed as
    # unicode escapes and break import. Normalize to forward slashes.
    if os.name == 'nt' and lib:
        return os.path.abspath(lib).replace('\\', '/')
    return lib


def _generate_bindings_native(
    path: str,
    lib: str,
    bindings_path: str,
    *,
    cc: Optional[str],
    cflags: Optional[List[str]],
    include_dirs: Optional[List[str]],
    defines: Optional[List[str]],
) -> None:
    source_text = _preprocess_source(
        path, cc=cc, cflags=cflags,
        include_dirs=include_dirs, defines=defines
    )
    generate_bindings_to_file = _get_bindgen()
    lib_for_bindgen = _normalize_lib_for_generated_source(lib or '')
    result = generate_bindings_to_file(
        source_text.encode('utf-8') + b'\0',
        lib_for_bindgen.encode('utf-8') + b'\0',
        bindings_path.encode('utf-8') + b'\0'
    )
    if result != 0:
        raise RuntimeError(f"Failed to generate native bindings for {path}, error code: {result}")


def _env_list(value: Optional[str]) -> List[str]:
    """Shell-tokenise a whitespace-separated config string."""
    if not value:
        return []
    return shlex.split(value)


def _generate_bindings_clang(
    path: str,
    lib: str,
    bindings_path: str,
    *,
    cflags: Optional[List[str]],
    include_dirs: Optional[List[str]],
    defines: Optional[List[str]],
    target: Optional[str],
    sysroot: Optional[str],
    clang_args: Optional[List[str]],
) -> None:
    from .cimport_clang import generate_bindings_to_file
    from .config import config

    effective_clang_args = _env_list(config.cimport_clang_args)
    if clang_args:
        effective_clang_args.extend(clang_args)

    generate_bindings_to_file(
        path,
        _normalize_lib_for_generated_source(lib or ''),
        bindings_path,
        cflags=cflags,
        include_dirs=include_dirs,
        defines=defines,
        target=target or config.cimport_target,
        sysroot=sysroot or config.cimport_sysroot,
        clang_args=effective_clang_args,
    )


def _compute_cache_key(path: str, lib: str, sources: Optional[List[str]] = None,
                       objects: Optional[List[str]] = None) -> str:
    """Compute a cache key for the bindings module.
    
    Args:
        path: Path to C header/source file
        lib: Library name
        sources: Additional source files
        objects: Object files
    
    Returns:
        Hex hash string for caching
    """
    hasher = hashlib.sha256()
    
    # Include main file path and mtime
    hasher.update(path.encode())
    if os.path.exists(path):
        hasher.update(str(os.path.getmtime(path)).encode())
    
    # Include lib name
    hasher.update((lib or '').encode())
    
    # Include source files
    for src in sorted(sources or []):
        hasher.update(src.encode())
        if os.path.exists(src):
            hasher.update(str(os.path.getmtime(src)).encode())
    
    # Include object files
    for obj in sorted(objects or []):
        hasher.update(obj.encode())
        if os.path.exists(obj):
            hasher.update(str(os.path.getmtime(obj)).encode())
    
    return hasher.hexdigest()[:16]


def _get_cache_dir(cache_key: str) -> str:
    """Get the cache directory for a given cache key."""
    cache_dir = os.path.join('build', 'cimport', cache_key)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _import_module_from_file(module_name: str, file_path: str) -> ModuleType:
    """Import a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _preprocess_source(path: str, cc: Optional[str] = None,
                       cflags: Optional[List[str]] = None,
                       include_dirs: Optional[List[str]] = None,
                       defines: Optional[List[str]] = None) -> str:
    """Preprocess a C source file using the C compiler's preprocessor.

    Runs `cc -E -P` to expand macros, includes, and conditionals.
    Falls back to raw file read on failure.

    Args:
        path: Path to C header/source file
        cc: C compiler to use (auto-detect if None)
        cflags: Additional compiler flags
        include_dirs: Include directories
        defines: Preprocessor defines

    Returns:
        Preprocessed source text
    """
    try:
        if cc is None:
            cc = find_available_cc()

        cmd = cc.split()
        cmd.extend(['-E', '-P'])

        for inc in (include_dirs or []):
            cmd.extend(['-I', inc])

        for define in (defines or []):
            cmd.append(f'-D{define}')

        if cflags:
            cmd.extend(cflags)

        cmd.append(os.path.abspath(path))

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=60, stdin=subprocess.DEVNULL
        )

        if result.returncode == 0:
            return result.stdout

        warnings.warn(
            f"Preprocessor failed for {path} (exit code {result.returncode}): "
            f"{result.stderr.strip()[:200]}. Falling back to raw file read.",
            stacklevel=2
        )
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as e:
        warnings.warn(
            f"Preprocessor unavailable for {path}: {e}. Falling back to raw file read.",
            stacklevel=2
        )

    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def cimport(path: str, *,
            kind: str = 'auto',
            lib: Optional[str] = None,
            sources: Optional[List[str]] = None,
            objects: Optional[List[str]] = None,
            compile_sources: bool = False,
            cc: Optional[str] = None,
            cflags: Optional[List[str]] = None,
            include_dirs: Optional[List[str]] = None,
            defines: Optional[List[str]] = None,
            backend: Optional[str] = None,
            target: Optional[str] = None,
            sysroot: Optional[str] = None,
            clang_args: Optional[List[str]] = None,
            export: Optional[List[str]] = None,
            export_all: bool = False,
            prefix: Optional[str] = None) -> ModuleType:
    """Import C header/source and return a pythoc bindings module.
    
    Args:
        path: Path to .h or .c file
        kind: 'auto' (infer from extension), 'header', or 'source'
        lib: Library name for @extern(lib='...'). If contains '/' treated as path.
        sources: Additional .c sources to compile
        objects: Explicit .o files to register for linking
        compile_sources: If True, compile .c sources to .o
        cc: C compiler to use (auto-detect if None)
        cflags: Additional compiler flags
        include_dirs: Include directories for compilation
        defines: Preprocessor defines
        backend: 'auto', 'clang', or 'native'. Defaults to PC_CIMPORT_BACKEND or auto.
        target: Optional clang target triple for the clang backend.
        sysroot: Optional sysroot for the clang backend.
        clang_args: Additional raw clang parse arguments.
        export: Symbol names to export to caller globals (explicit opt-in)
        export_all: If True, export all symbols to caller globals
        prefix: Optional symbol prefix
    
    Returns:
        Module object containing the generated bindings
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If parsing or compilation fails
    """
    # Resolve path - first try as-is (handles relative paths with ..)
    if not os.path.isabs(path):
        # First check if the path exists relative to cwd
        if os.path.exists(path):
            path = os.path.abspath(path)
        else:
            # Try relative to caller's directory
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_file = frame.f_back.f_globals.get('__file__')
                if caller_file:
                    caller_dir = os.path.dirname(os.path.abspath(caller_file))
                    candidate = os.path.join(caller_dir, path)
                    if os.path.exists(candidate):
                        path = candidate
            path = os.path.abspath(path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"C file not found: {path}")
    
    # Determine kind
    if kind == 'auto':
        ext = os.path.splitext(path)[1].lower()
        if ext == '.h':
            kind = 'header'
        elif ext == '.c':
            kind = 'source'
        else:
            kind = 'header'  # Default to header for unknown extensions
    
    # Default lib name
    # When compile_sources=True and lib is not specified, use empty string
    # to indicate symbols come from directly linked object files
    if lib is None:
        if compile_sources:
            # Symbols will be resolved from .o files, no library needed
            lib = ''
        else:
            basename = os.path.splitext(os.path.basename(path))[0]
            lib = basename
    
    # Initialize lists
    sources = list(sources or [])
    objects = list(objects or [])
    
    # For source files, add to sources list for compilation
    if kind == 'source' and compile_sources:
        if path not in sources:
            sources.insert(0, path)
    
    # Create cache directory based on file path structure
    # This ensures same files always use same cache location
    base_cache_dir = os.path.join('build', 'cimport')
    
    # Convert absolute path to a cache-safe relative path
    path_abs = os.path.abspath(path)
    if os.name == 'nt' and ':' in path_abs:
        # Windows: remove drive letter
        path_rel = path_abs.split(':', 1)[1].lstrip(os.sep)
    else:
        # Unix: remove leading slash
        path_rel = path_abs.lstrip('/')
    
    # Create cache directory preserving directory structure
    cache_dir = os.path.join(base_cache_dir, os.path.dirname(path_rel))
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate bindings module path
    basename = os.path.splitext(os.path.basename(path))[0]
    if prefix:
        module_name = f"_cimport_{prefix}_{basename}"
    else:
        module_name = f"_cimport_{basename}"

    backend_request = _normalize_cimport_backend(backend)
    selected_backend = _select_cimport_backend(backend_request)
    bindings_path = _bindings_path_for_backend(cache_dir, basename, selected_backend)
    needs_regen = _bindings_need_regen(path, bindings_path)
    
    # Generate bindings if needed
    if needs_regen:
        try:
            if selected_backend == "clang":
                _generate_bindings_clang(
                    path,
                    lib or '',
                    bindings_path,
                    cflags=cflags,
                    include_dirs=include_dirs,
                    defines=defines,
                    target=target,
                    sysroot=sysroot,
                    clang_args=clang_args,
                )
            else:
                _generate_bindings_native(
                    path,
                    lib or '',
                    bindings_path,
                    cc=cc,
                    cflags=cflags,
                    include_dirs=include_dirs,
                    defines=defines,
                )
        except Exception as exc:
            if backend_request != "auto" or selected_backend != "clang":
                raise

            warnings.warn(
                f"clang cimport backend failed for {path}: {exc}. "
                "Falling back to native cimport backend.",
                stacklevel=2,
            )
            selected_backend = "native"
            bindings_path = _bindings_path_for_backend(cache_dir, basename, selected_backend)
            if _bindings_need_regen(path, bindings_path):
                _generate_bindings_native(
                    path,
                    lib or '',
                    bindings_path,
                    cc=cc,
                    cflags=cflags,
                    include_dirs=include_dirs,
                    defines=defines,
                )
    
    # Compile sources if requested
    if compile_sources and sources:
        compiled_objects = []
        for src in sources:
            # Create object file path in same cache directory structure
            src_abs = os.path.abspath(src)
            
            # Convert to cache-safe path
            if os.name == 'nt' and ':' in src_abs:
                src_rel = src_abs.split(':', 1)[1].lstrip(os.sep)
            else:
                src_rel = src_abs.lstrip('/')
            
            # Place object file in same directory structure under cache
            obj_cache_dir = os.path.join(base_cache_dir, os.path.dirname(src_rel))
            os.makedirs(obj_cache_dir, exist_ok=True)
            
            obj_name = os.path.splitext(os.path.basename(src_rel))[0] + '.o'
            obj_path = os.path.join(obj_cache_dir, obj_name)
            
            # Only compile if object doesn't exist or source is newer
            needs_compile = True
            if os.path.exists(obj_path) and os.path.exists(src):
                obj_mtime = os.path.getmtime(obj_path)
                src_mtime = os.path.getmtime(src)
                if obj_mtime >= src_mtime:
                    needs_compile = False
            
            if needs_compile:
                from .utils.cc_utils import compile_c_to_object
                compile_c_to_object(
                    src, obj_path, cc=cc, cflags=cflags,
                    include_dirs=include_dirs, defines=defines
                )
            
            compiled_objects.append(obj_path)
        
        objects.extend(compiled_objects)
    
    # Register objects for linking
    registry = get_unified_registry()
    for obj in objects:
        # Check if an object with the same content is already registered
        # This prevents duplicate symbols from different temporary files with same content
        should_register = True
        if os.path.exists(obj):
            obj_size = os.path.getsize(obj)
            existing_objects = registry.get_link_objects()
            
            for existing_obj in existing_objects:
                if os.path.exists(existing_obj):
                    # Quick size check first
                    if os.path.getsize(existing_obj) == obj_size:
                        # Same size - compare content to detect duplicates
                        try:
                            with open(obj, 'rb') as f1, open(existing_obj, 'rb') as f2:
                                if f1.read() == f2.read():
                                    # Same content - skip registration
                                    should_register = False
                                    break
                        except (IOError, OSError):
                            # If we can't read files, assume they're different
                            pass
        
        if should_register:
            registry.add_link_object(obj)
    
    # Import the bindings module
    module = _import_module_from_file(module_name, bindings_path)
    
    # Handle exports to caller globals
    if export or export_all:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_globals = frame.f_back.f_globals
            
            if export_all:
                # Export all public symbols
                for name in dir(module):
                    if not name.startswith('_'):
                        caller_globals[name] = getattr(module, name)
            elif export:
                # Export only specified symbols
                for name in export:
                    if hasattr(module, name):
                        caller_globals[name] = getattr(module, name)
                    else:
                        raise AttributeError(
                            f"Symbol '{name}' not found in generated bindings"
                        )
    
    return module


# Convenience alias
def cimport_header(path: str, lib: str, **kwargs) -> ModuleType:
    """Import a C header file.
    
    Convenience wrapper for cimport(..., kind='header').
    """
    return cimport(path, kind='header', lib=lib, **kwargs)


def cimport_source(path: str, lib: Optional[str] = None,
                   compile_sources: bool = True, **kwargs) -> ModuleType:
    """Import a C source file.
    
    Convenience wrapper for cimport(..., kind='source', compile_sources=True).
    """
    return cimport(path, kind='source', lib=lib, compile_sources=compile_sources, **kwargs)
