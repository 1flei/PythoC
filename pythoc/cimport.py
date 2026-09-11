"""
C Import (cimport) - Import C headers/sources as pythoc modules

This module provides the cimport() function to:
1. Parse C header/source files using libclang
2. Generate pythoc bindings
3. Import the bindings as a Python module
4. Optionally compile C sources and register for linking

Architecture:
- Uses libclang (required) to read C declarations and layout-aware types
- For a pure-pythoc backend (no libclang), use the separate `pcc` package

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
from typing import Optional, List, Any
from types import ModuleType

from .registry import get_unified_registry
from .utils.cc_utils import compile_c_to_object, compile_c_sources, find_available_cc
from .utils.link_utils import file_lock, linklibrary


_VALID_CIMPORT_BACKENDS = {"auto", "clang"}


def _normalize_cimport_backend(backend: Optional[str]) -> str:
    """Resolve backend request from API/env into auto|clang."""
    from .config import config
    requested = backend or config.cimport_backend
    requested = (requested or "auto").strip().lower()
    if requested not in _VALID_CIMPORT_BACKENDS:
        valid = ", ".join(sorted(_VALID_CIMPORT_BACKENDS))
        raise ValueError(f"Invalid cimport backend '{requested}'. Expected one of: {valid}")
    return requested


def _bindings_path_for_backend(cache_dir: str, basename: str, backend_name: str) -> str:
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
    # Bare library names ('c', 'm', 'mylib', ...) are NOT paths: never
    # abspath them, or 'c' would turn into '<cwd>/c' and end up as a bogus
    # file argument on the linker command line.
    if (os.name == 'nt' and lib
            and (os.path.isabs(lib) or '/' in lib or '\\' in lib)):
        return os.path.abspath(lib).replace('\\', '/')
    return lib


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
    enable_wrappers: bool = True,
    stub_path: Optional[str] = None,
    includes: bool = False,
) -> None:
    from .cimport_clang import generate_bindings_to_file

    # Write to a temp file and atomically rename so concurrent processes
    # never observe a partially written bindings module.
    tmp_path = bindings_path + '.tmp.' + str(os.getpid())
    try:
        generate_bindings_to_file(
            path,
            _normalize_lib_for_generated_source(lib or ''),
            tmp_path,
            cflags=cflags,
            include_dirs=include_dirs,
            defines=defines,
            target=target,
            sysroot=sysroot,
            clang_args=clang_args,
            enable_wrappers=enable_wrappers,
            stub_path=stub_path,
            includes=includes,
        )
        os.replace(tmp_path, bindings_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _hash_parse_options(backend: str, lib: str, target: Optional[str],
                        parse_args: List[str], cc: Optional[str],
                        includes: bool = False) -> str:
    """Short hash of every input that changes generated bindings output.

    Two cimport() calls for the same header with different flags must not
    share cached bindings, so the effective parse arguments (which already
    include defines/include_dirs/cflags/clang_args/sysroot, the env
    PC_CIMPORT_* knobs, and discovered system include dirs) go into the
    cache file name, along with the backend and ``lib`` (embedded in the
    generated @extern decorators).  ``cc`` is included because it drives
    compilation of the wrapper stub and compile_sources objects cached
    next to the bindings.  ``includes`` widens the emitted declaration set
    to the whole translation unit.  The pythoc compiler stamp is included
    so that editing pythoc itself (bindings generator or wrapper emission)
    invalidates all artifacts derived from it.
    """
    from .utils.compiler_stamp import get_compiler_mtime
    hasher = hashlib.sha256()
    hasher.update(backend.encode())
    hasher.update((lib or '').encode())
    hasher.update((target or '').encode())
    hasher.update((cc or '').encode())
    hasher.update(b'\x01' if includes else b'\x00')
    hasher.update(repr(get_compiler_mtime()).encode())
    for arg in parse_args:
        hasher.update(b'\x00')
        hasher.update(arg.encode())
    return hasher.hexdigest()[:8]


def _resolve_via_include_dirs(name: str, include_dirs: Optional[List[str]],
                              target: Optional[str],
                              sysroot: Optional[str]) -> Optional[str]:
    """Resolve a header name (e.g. 'stdio.h') through the include path.

    Searches explicit include_dirs first, then PC_CIMPORT_INCLUDE_PATH,
    then the host compiler's built-in system include dirs.
    """
    from .cimport_clang import default_include_search_dirs

    search_dirs = list(include_dirs or [])
    search_dirs.extend(default_include_search_dirs(target, sysroot))
    for directory in search_dirs:
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


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
            libraries: Optional[List[str]] = None,
            compile_sources: bool = False,
            cc: Optional[str] = None,
            cflags: Optional[List[str]] = None,
            include_dirs: Optional[List[str]] = None,
            defines: Optional[List[str]] = None,
            backend: Optional[str] = None,
            target: Optional[str] = None,
            sysroot: Optional[str] = None,
            clang_args: Optional[List[str]] = None,
            includes: bool = False,
            export: Optional[List[str]] = None,
            export_all: bool = False,
            prefix: Optional[str] = None) -> ModuleType:
    """Import C header/source and return a pythoc bindings module.

    Args:
        path: Path to .h or .c file, or a header name resolved through the
            include search path (include_dirs, PC_CIMPORT_INCLUDE_PATH,
            then the host compiler's system include dirs).
        kind: 'auto' (infer from extension), 'header', or 'source'
        lib: Library name for @extern(lib='...'). If contains '/' treated as path.
        sources: Additional .c sources to compile
        objects: Explicit .o files to register for linking
        libraries: Extra libraries to link, loaded via linklibrary():
            shared libraries are dlopen'ed RTLD_GLOBAL (JIT + Python-side
            extern resolution) and registered for AOT linking; static
            archives are registered as link objects.
        compile_sources: If True, compile .c sources to .o
        cc: C compiler to use (auto-detect if None)
        cflags: Additional compiler flags
        include_dirs: Include directories for compilation
        defines: Preprocessor defines
        backend: 'auto' or 'clang'. Defaults to PC_CIMPORT_BACKEND or auto.
        target: Optional clang target triple for the clang backend.
        sysroot: Optional sysroot for the clang backend.
        clang_args: Additional raw clang parse arguments.
        includes: If True, also emit declarations pulled in transitively
            from files the imported file includes (default: only
            declarations written in the imported file itself).  Needed for
            system headers that delegate to private sub-headers (glibc
            math.h -> bits/mathcalls.h, macOS string.h -> _string.h).
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
            if not os.path.exists(path):
                # Header given by name only (e.g. 'stdio.h' or
                # 'sys/stat.h'): resolve through the include search path.
                resolved = _resolve_via_include_dirs(
                    path, include_dirs, target, sysroot)
                if resolved is not None:
                    path = resolved
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

    # Extra libraries: load into the process now so JIT and Python-side
    # extern calls resolve their symbols, and register them for AOT links.
    for library in libraries or []:
        linklibrary(library)
    
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
    
    backend_request = _normalize_cimport_backend(backend)
    # clang is the only backend; 'auto' resolves to 'clang'
    selected_backend = "clang" if backend_request == "auto" else backend_request

    # The bindings cache is keyed on the effective parse options, not just
    # the file path: re-importing the same header with different
    # target/defines/cflags/include_dirs/clang_args (or different
    # PC_CIMPORT_* env configuration) must not reuse stale bindings.
    from .cimport_clang import resolve_parse_options
    effective_target, _effective_sysroot, parse_args = resolve_parse_options(
        cflags=cflags, include_dirs=include_dirs, defines=defines,
        target=target, sysroot=sysroot, clang_args=clang_args,
    )
    options_hash = _hash_parse_options(selected_backend, lib or '',
                                       effective_target, parse_args, cc,
                                       includes=includes)

    # Generate bindings module path
    basename = os.path.splitext(os.path.basename(path))[0]
    cached_name = f"{basename}_{options_hash}"
    if prefix:
        module_name = f"_cimport_{prefix}_{cached_name}"
    else:
        module_name = f"_cimport_{cached_name}"
    bindings_path = _bindings_path_for_backend(cache_dir, cached_name, selected_backend)

    # Header-defined functions without an external symbol (static / C99
    # inline) get forwarding wrappers compiled from a generated stub.
    # A .c imported with compile_sources=True is compiled as its own
    # translation unit, so wrapping it would duplicate its symbols.
    enable_wrappers = not (kind == 'source' and compile_sources)
    from .cimport_wrappers import stub_paths
    stub_c_path, stub_obj_path = stub_paths(cache_dir, cached_name)

    # Generate bindings if needed.  A lock file next to the bindings
    # serializes concurrent regeneration across processes; re-check
    # freshness inside the lock so waiters skip redundant work.
    with file_lock(bindings_path + '.lock'):
        if _bindings_need_regen(path, bindings_path):
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
                enable_wrappers=enable_wrappers,
                stub_path=stub_c_path,
                includes=includes,
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
            
            obj_name = os.path.splitext(os.path.basename(src_rel))[0] + f'_{options_hash}.o'
            obj_path = os.path.join(obj_cache_dir, obj_name)

            # Only compile if object doesn't exist or source is newer.
            # A per-object lock serializes concurrent compilation across
            # processes; re-check freshness inside the lock.
            with file_lock(obj_path + '.lock'):
                needs_compile = True
                if os.path.exists(obj_path) and os.path.exists(src):
                    obj_mtime = os.path.getmtime(obj_path)
                    src_mtime = os.path.getmtime(src)
                    if obj_mtime >= src_mtime:
                        needs_compile = False

                if needs_compile:
                    compile_c_to_object(
                        src, obj_path, cc=cc, cflags=cflags,
                        include_dirs=include_dirs, defines=defines
                    )

            compiled_objects.append(obj_path)
        
        objects.extend(compiled_objects)

    # Compile the inline-wrapper stub produced alongside the bindings, with
    # the same include_dirs/defines/cflags so the header resolves identically.
    # The object is stale when it is older than the stub or the header itself
    # (a header edit that leaves the stub text unchanged must still rebuild).
    if os.path.exists(stub_c_path):
        with file_lock(stub_obj_path + '.lock'):
            needs_compile = True
            if os.path.exists(stub_obj_path):
                obj_mtime = os.path.getmtime(stub_obj_path)
                if (obj_mtime >= os.path.getmtime(stub_c_path)
                        and obj_mtime >= os.path.getmtime(path)):
                    needs_compile = False
            if needs_compile:
                from .config import config
                effective_sysroot = sysroot or config.cimport_sysroot
                stub_cflags = list(cflags or [])
                if effective_sysroot:
                    stub_cflags.append(f'--sysroot={effective_sysroot}')
                compile_c_to_object(
                    stub_c_path, stub_obj_path, cc=cc, cflags=stub_cflags,
                    include_dirs=include_dirs, defines=defines
                )
        objects.append(stub_obj_path)
    
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
