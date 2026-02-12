"""
Linker utilities for PC compiler

Provides unified linker functionality for both executables and shared libraries.
Supports multiple linkers including gcc, clang, and zig for cross-platform compatibility.
"""

import os
import sys
import struct
import shutil
import subprocess
import time
from functools import lru_cache
from typing import List, Optional, Set
from contextlib import contextmanager


@lru_cache(maxsize=None)
def _which_cached(name: str) -> Optional[str]:
    """Cached shutil.which() — avoids repeated PATH scans (expensive on Windows)."""
    return shutil.which(name)


@lru_cache(maxsize=None)
def _resolve_zig_exe() -> Optional[str]:
    """Resolve the native zig executable path from the ziglang pip package.

    On Windows, ``python-zig`` is a thin Python wrapper that spawns
    ``zig.exe`` as a subprocess.  Each invocation pays the Python
    interpreter startup cost (~0.15 s).  By calling ``zig.exe`` directly
    we eliminate that overhead — critical when there are 50-100 link
    operations in a single build.

    Returns the absolute path to ``zig.exe`` (or ``zig`` on other
    platforms) if available, otherwise ``None``.
    """
    try:
        import ziglang
        exe_name = 'zig.exe' if sys.platform == 'win32' else 'zig'
        zig_path = os.path.join(os.path.dirname(ziglang.__file__), exe_name)
        if os.path.isfile(zig_path):
            return zig_path
    except ImportError:
        pass
    return None


def _zig_tool_cmd(subcommand: str) -> Optional[List[str]]:
    """Return the command list to invoke a zig sub-tool (``cc``, ``dlltool``, …).

    Prefers the native ``zig.exe <sub>`` when available (faster),
    falls back to ``python-zig <sub>``.
    """
    exe = _resolve_zig_exe()
    if exe:
        return [exe, subcommand]
    if _which_cached('python-zig'):
        return ['python-zig', subcommand]
    return None


def _read_coff_exports(obj_path: str) -> Set[str]:
    """Read exported (global defined) symbols from a COFF ``.o`` file.

    This is a pure-Python replacement for ``llvm-nm -g --defined-only``,
    avoiding a ~0.1-0.4 s subprocess overhead **per call** on Windows.
    The COFF object format is straightforward: a fixed header followed by
    section headers and a symbol table with an appended string table.

    Returns a set of exported symbol names (equivalent to symbols with
    types T, D, B, R, S, V, W in ``llvm-nm`` output).
    """
    try:
        with open(obj_path, 'rb') as f:
            data = f.read()
    except (OSError, IOError):
        return set()

    if len(data) < 20:
        return set()

    # COFF header: Machine(2) NumSections(2) TimeDateStamp(4)
    #              PointerToSymbolTable(4) NumberOfSymbols(4) ...
    sym_table_offset = struct.unpack_from('<I', data, 8)[0]
    num_symbols = struct.unpack_from('<I', data, 12)[0]

    if sym_table_offset == 0 or num_symbols == 0:
        return set()

    # String table starts immediately after the symbol table.
    # Each symbol entry is 18 bytes.
    string_table_offset = sym_table_offset + num_symbols * 18

    exports: Set[str] = set()
    i = 0
    while i < num_symbols:
        entry_offset = sym_table_offset + i * 18
        if entry_offset + 18 > len(data):
            break

        name_bytes = data[entry_offset:entry_offset + 8]

        # Decode symbol name
        if name_bytes[:4] == b'\x00\x00\x00\x00':
            # Long name: offset into string table
            str_offset = struct.unpack_from('<I', name_bytes, 4)[0]
            abs_offset = string_table_offset + str_offset
            if abs_offset < len(data):
                end = data.index(b'\x00', abs_offset) if b'\x00' in data[abs_offset:] else len(data)
                name = data[abs_offset:end].decode('utf-8', errors='replace')
            else:
                name = ''
        else:
            name = name_bytes.rstrip(b'\x00').decode('utf-8', errors='replace')

        # SectionNumber (2 bytes, signed), Type (2 bytes),
        # StorageClass (1 byte), NumberOfAuxSymbols (1 byte)
        section_num = struct.unpack_from('<h', data, entry_offset + 12)[0]
        storage_class = data[entry_offset + 16]
        num_aux = data[entry_offset + 17]

        # IMAGE_SYM_CLASS_EXTERNAL = 2; section > 0 means defined
        if storage_class == 2 and section_num > 0:
            if name and not name.startswith('.') and not name.startswith('$'):
                exports.add(name)

        i += 1 + num_aux

    return exports


# Platform-specific file locking
if sys.platform == 'win32':
    import msvcrt
    
    @contextmanager
    def file_lock(lockfile_path: str, timeout: float = 60.0):
        """Windows file locking using msvcrt."""
        lockfile = None
        start_time = time.time()
        
        try:
            lock_dir = os.path.dirname(lockfile_path)
            if lock_dir and not os.path.exists(lock_dir):
                os.makedirs(lock_dir, exist_ok=True)
            
            while True:
                try:
                    lockfile = open(lockfile_path, 'a+')
                    msvcrt.locking(lockfile.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except (IOError, OSError):
                    if lockfile:
                        lockfile.close()
                        lockfile = None
                    
                    if time.time() - start_time > timeout:
                        raise TimeoutError(
                            f"Failed to acquire lock on {lockfile_path} within {timeout}s"
                        )
                    
                    wait_time = min(0.01 * (2 ** min((time.time() - start_time) / 0.1, 5)), 0.5)
                    time.sleep(wait_time)
            
            yield
            
        finally:
            if lockfile:
                try:
                    msvcrt.locking(lockfile.fileno(), msvcrt.LK_UNLCK, 1)
                    lockfile.close()
                except Exception:
                    pass
else:
    import fcntl
    
    @contextmanager
    def file_lock(lockfile_path: str, timeout: float = 60.0):
        """Unix file locking using fcntl."""
        lockfile = None
        start_time = time.time()
        
        try:
            lock_dir = os.path.dirname(lockfile_path)
            if lock_dir and not os.path.exists(lock_dir):
                os.makedirs(lock_dir, exist_ok=True)
            
            while True:
                try:
                    lockfile = open(lockfile_path, 'a')
                    fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (IOError, OSError):
                    if lockfile:
                        lockfile.close()
                        lockfile = None
                    
                    if time.time() - start_time > timeout:
                        raise TimeoutError(
                            f"Failed to acquire lock on {lockfile_path} within {timeout}s"
                        )
                    
                    wait_time = min(0.01 * (2 ** min((time.time() - start_time) / 0.1, 5)), 0.5)
                    time.sleep(wait_time)
            
            yield
            
        finally:
            if lockfile:
                try:
                    fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
                    lockfile.close()
                except Exception:
                    pass


def get_shared_lib_extension() -> str:
    """Get platform-specific shared library extension.
    
    Returns:
        '.dll' on Windows, '.so' on Linux and mac
    """
    if sys.platform == 'win32':
        return '.dll'
    else:
        return '.so'


def get_executable_extension() -> str:
    """Get platform-specific executable extension.
    
    Returns:
        '.exe' on Windows, '' on Linux and mac
    """
    if sys.platform == 'win32':
        return '.exe'
    else:
        return ''


def _get_linker_candidates() -> List[str]:
    """Get linker candidates based on availability.
    
    On Windows, only zig is supported — it targets windows-gnu
    which behaves consistently with Linux.

    Prefers the native ``zig.exe cc`` (faster) over ``python-zig cc``.
    """
    candidates = []

    if sys.platform == 'win32':
        # Windows: only zig is supported.
        zig_cc = _zig_tool_cmd('cc')
        if zig_cc:
            candidates.append(' '.join(zig_cc))
    else:
        for linker in ['cc', 'clang', 'gcc']:
            if _which_cached(linker):
                candidates.append(linker)
        # zig as fallback on non-Windows
        zig_cc = _zig_tool_cmd('cc')
        if zig_cc:
            candidates.append(' '.join(zig_cc))
    
    return candidates


def find_available_linker() -> str:
    """Find an available linker on the system.
    
    On Windows only zig is supported. On other platforms: cc, clang, gcc, python-zig cc.
    
    Returns:
        Linker command string
        
    Raises:
        RuntimeError: If no linker is found
    """
    candidates = _get_linker_candidates()
    if candidates:
        return candidates[0]
    
    if sys.platform == 'win32':
        raise RuntimeError(
            "No linker found. On Windows, zig is required.\n"
            "Install via pip: pip install ziglang"
        )
    raise RuntimeError(
        "No linker found. Please install one of: cc, gcc, clang, or zig.\n"
        "Tip: Install zig via pip: pip install ziglang"
    )


def get_default_linkers() -> List[str]:
    """Get list of linkers to try in order of preference.
    
    Returns:
        List of available linker command strings
    """
    candidates = _get_linker_candidates()
    return candidates if candidates else ['cc', 'clang', 'gcc']


def get_link_flags(link_libraries: Optional[List[str]] = None) -> List[str]:
    """Get link flags.

    By default, this reads link libraries from the unified registry.
    Callers may optionally pass an explicit list to avoid relying on global
    registry state.

    Args:
        link_libraries: Optional list of library names/paths.

    Returns:
        List of linker flags including -l options and direct library paths
    """
    if link_libraries is None:
        from ..registry import get_unified_registry
        libs = get_unified_registry().get_link_libraries()
    else:
        libs = link_libraries

    lib_flags: List[str] = []

    for lib in libs:
        if sys.platform == 'win32' and lib in {'c', 'm'}:
            continue

        # On Windows many paths come in with backslashes and are often relative
        # (e.g. "build\\...\\foo.dll"). If we incorrectly treat these as bare
        # library names and add `-l`, zig/clang will try to search for a system
        # library literally named "build\\...\\foo.dll" and fail.
        is_path_like = os.path.isabs(lib) or ('/' in lib) or (os.sep in lib)
        has_lib_ext = os.path.splitext(lib)[1].lower() in {'.a', '.so', '.dll', '.lib'}

        if is_path_like or has_lib_ext:
            # Library file path (absolute or relative) - pass directly to linker.
            # On Windows, prefer the import library (`.lib`) when a `.dll` path
            # is provided, because the COFF linker resolves symbols via the
            # import library at link time.
            if sys.platform == 'win32' and lib.lower().endswith('.dll'):
                implib = os.path.splitext(lib)[0] + '.lib'
                if os.path.exists(implib):
                    lib_flags.append(implib)
                else:
                    lib_flags.append(lib)
            else:
                lib_flags.append(lib)
        else:
            # Library name - use -l flag
            lib_flags.append(f'-l{lib}')



    # Add --no-as-needed to ensure all libraries are linked
    # This is critical for libraries like libgcc_s that provide
    # soft-float support functions (e.g., for f16/bf16/f128)
    if lib_flags and sys.platform not in ('win32', 'darwin'):
        lib_flags = ['-Wl,--no-as-needed'] + lib_flags

    return lib_flags


def get_platform_link_flags(shared: bool = False, linker: str = 'gcc') -> List[str]:
    """Get platform-specific link flags
    
    Args:
        shared: True for shared library, False for executable
        linker: Linker being used (affects flag format)
    
    Returns:
        List of platform-specific flags
    """
    is_zig = 'zig' in linker
    
    if sys.platform == 'win32':
        # Windows: only zig is supported (targets x86_64-windows-gnu).
        # --export-all-symbols makes all functions visible to ctypes.
        flags = ['-target', 'x86_64-windows-gnu']
        if shared:
            flags += ['-shared', '-Wl,--export-all-symbols']
        return flags
    elif sys.platform == 'darwin':
        # Explicitly specify architecture to avoid x86_64/arm64 mismatch issues
        import platform
        arch = platform.machine()
        arch_flag = ['-arch', arch] if arch in ('arm64', 'x86_64') and not is_zig else []
        if shared:
            return arch_flag + ['-shared', '-undefined', 'dynamic_lookup']
        else:
            return arch_flag
    else:  # Linux
        if shared:
            if is_zig:
                return ['-shared', '-fPIC']
            else:
                return ['-shared', '-fPIC', '-Wl,--export-dynamic', 
                       '-Wl,--allow-shlib-undefined', '-Wl,--unresolved-symbols=ignore-all']
        else:
            return []






def _write_exports_def(def_file: str, dll_name: str, obj_files: List[str]) -> bool:
    """Write a Windows `.def` file exporting global symbols from `obj_files`.

    We pass the generated `.def` as a regular linker input file. This is
    accepted by zig on windows-gnu, and it ensures the DLL export table (and the
    resulting import library) contains the symbols needed by downstream links.

    Uses pure-Python COFF parsing instead of ``llvm-nm`` subprocess calls,
    which saves ~0.1-0.4 s per object file on Windows.

    Returns True if a file was written and should be passed to the linker.
    """
    if sys.platform != 'win32':
        return False

    # Fast path: if the .def file already exists and is newer than all .o files,
    # skip re-scanning the object files.
    if os.path.exists(def_file):
        def_mtime = os.path.getmtime(def_file)
        all_uptodate = True
        for obj in obj_files:
            if obj and os.path.exists(obj):
                if os.path.getmtime(obj) > def_mtime:
                    all_uptodate = False
                    break
        if all_uptodate:
            return True

    exports: Set[str] = set()
    for obj in obj_files:
        if not obj:
            continue
        low = obj.lower()
        if not (low.endswith('.o') or low.endswith('.obj')):
            continue
        exports |= _read_coff_exports(obj)

    if not exports:
        return False

    try:
        os.makedirs(os.path.dirname(def_file) or '.', exist_ok=True)
        with open(def_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(f'LIBRARY "{dll_name}"\n')
            f.write('EXPORTS\n')
            for sym in sorted(exports):
                f.write(f'    {sym}\n')
        return True
    except Exception:
        return False



def _generate_stub_implib(obj_file: str, dll_name: str, implib_path: str) -> bool:
    """Generate a stub import library (`.lib`) from an object file's exports.

    On Windows, DLL linking requires all referenced symbols to be resolved at
    link time. When two DLLs have circular dependencies (A imports from B and B
    imports from A), neither can be linked first. We break this deadlock by
    generating *stub* import libraries from `.o` files: the `.def` describes
    what the eventual DLL will export, and `dlltool` creates a minimal `.lib`
    that satisfies the linker without needing the actual `.dll`.

    The `.def` file is now generated via pure-Python COFF parsing (no
    ``llvm-nm`` subprocess), saving ~0.1-0.4 s per call.

    Returns True if a usable `.lib` was generated.
    """
    if sys.platform != 'win32':
        return False

    # Fast path: if .lib already exists and is newer than .o, skip.
    if os.path.exists(implib_path) and os.path.exists(obj_file):
        if os.path.getmtime(implib_path) >= os.path.getmtime(obj_file):
            return True

    # We need dlltool to create .lib (def generation is now pure Python)
    dlltool_cmd = _which_cached('llvm-dlltool')
    if dlltool_cmd:
        dlltool_cmd = [dlltool_cmd]
    else:
        dlltool_cmd = _zig_tool_cmd('dlltool')
    if not dlltool_cmd:
        return False

    # Generate .def file using pure-Python COFF parsing (no subprocess)
    def_file = os.path.splitext(implib_path)[0] + '.exports.def'
    if not _write_exports_def(def_file, dll_name, [obj_file]):
        return False

    try:
        cmd = dlltool_cmd + ['-d', def_file, '-l', implib_path, '-m', 'i386:x86-64']

        subprocess.run(
            cmd, check=True, capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=30,
        )
        return os.path.exists(implib_path)
    except Exception:
        return False


def build_link_command(


    obj_files: List[str],
    output_file: str,
    shared: bool = False,
    linker: str = 'gcc',
    link_objects: Optional[List[str]] = None,
    link_libraries: Optional[List[str]] = None,
) -> List[str]:
    """Build linker command.

    Args:
        obj_files: List of object file paths
        output_file: Output file path (.so/.dll or executable)
        shared: True for shared library, False for executable
        linker: Linker command (e.g., 'gcc', 'clang', 'python-zig cc')
        link_objects: Optional extra object files to link (defaults to registry)
        link_libraries: Optional link libraries to use (defaults to registry)

    Returns:
        Link command as list of arguments
    """
    from ..registry import get_unified_registry

    # Split linker command (handles 'python-zig cc' etc.)
    linker_cmd = linker.split()

    platform_flags = get_platform_link_flags(shared, linker=linker)
    lib_flags = get_link_flags(link_libraries)

    # Include link objects from registry (from cimport compiled sources)
    if link_objects is None:
        link_objects = get_unified_registry().get_link_objects()
    all_obj_files = list(obj_files) + list(link_objects)

    # python-zig changes its working directory internally, so relative paths
    # break.  Normalise everything to absolute paths for robustness.
    all_obj_files = [os.path.abspath(p) for p in all_obj_files]
    output_file = os.path.abspath(output_file)




    # Also normalise library paths (entries that are file paths, not -l flags)

    lib_flags = [
        os.path.abspath(f) if not f.startswith('-') and (os.path.isabs(f) or os.sep in f or '/' in f) else f
        for f in lib_flags
    ]

    return linker_cmd + platform_flags + all_obj_files + ['-o', output_file] + lib_flags


def try_link_with_linkers(
    obj_files: List[str],
    output_file: str,
    shared: bool = False,
    linkers: Optional[List[str]] = None,
    link_objects: Optional[List[str]] = None,
    link_libraries: Optional[List[str]] = None,
) -> str:
    """Try linking with multiple linkers.

    Returns:
        Path to linked file
    """
    if linkers is None:
        linkers = get_default_linkers()

    errors = []
    for linker in linkers:
        # Check if linker executable is available
        # For 'python-zig cc', check 'python-zig'; for 'gcc', check 'gcc'
        linker_exe = linker.split()[0]
        if not _which_cached(linker_exe):
            errors.append(f"{linker}: not found")
            continue


        link_cmd: List[str] = []
        try:
            link_cmd = build_link_command(

                obj_files,
                output_file,
                shared=shared,
                linker=linker,
                link_objects=link_objects,
                link_libraries=link_libraries,
            )




            # On Windows, force an explicit exports list via a `.def` file.
            # Passing the `.def` as an input file works with zig (windows-gnu)
            # and avoids MSVC-style `/DEF:` flags.
            if sys.platform == 'win32' and shared and output_file.lower().endswith('.dll'):
                try:
                    out_idx = link_cmd.index('-o')
                    obj_candidates = [a for a in link_cmd[:out_idx] if a.lower().endswith(('.o', '.obj'))]
                    def_file = os.path.splitext(os.path.abspath(output_file))[0] + '.exports.def'
                    if _write_exports_def(def_file, os.path.basename(output_file), obj_candidates):
                        if def_file not in link_cmd:
                            link_cmd.insert(out_idx, def_file)
                except Exception:
                    pass

            # Use stdin=DEVNULL to prevent subprocess from waiting for input
            # This is critical on Windows, especially in ProcessPoolExecutor workers
            subprocess.run(

                link_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
                stdin=subprocess.DEVNULL,
            )
            return output_file

        except subprocess.TimeoutExpired:
            errors.append(f"{linker}: timed out after 120s")
        except subprocess.CalledProcessError as e:
            msg = (e.stdout or '') + ("\n" if e.stdout else '') + (e.stderr or '')
            # Include the command for debugging; this is especially helpful on Windows
            # where library resolution can be sensitive to path/extension details.
            cmd_str = ' '.join(link_cmd) if link_cmd else '<link_cmd unavailable>'
            errors.append(f"{linker}: {msg.strip()}\nCMD: {cmd_str}")



    # All linkers failed
    file_type = "shared library" if shared else "executable"
    raise RuntimeError(
        f"Failed to link {file_type} with all linkers ({', '.join(linkers)}):\n" +
        "\n".join(errors)
    )


def link_files(
    obj_files: List[str],
    output_file: str,
    shared: bool = False,
    linker: Optional[str] = None,
    link_objects: Optional[List[str]] = None,
    link_libraries: Optional[List[str]] = None,
) -> str:
    """Link object files to executable or shared library
    
    Args:
        obj_files: List of object file paths
        output_file: Output file path
        shared: True for shared library, False for executable
        linker: Linker to use (default: auto-detect, tries multiple)
    
    Returns:
        Path to linked file
    
    Raises:
        RuntimeError: If linking fails
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Use file lock to prevent concurrent linking of the same output file
    # This is critical when running parallel tests that import the same modules
    lockfile_path = output_file + '.lock'
    
    with file_lock(lockfile_path):
        # Also acquire locks for all input .o files to ensure they are fully written
        # This prevents "file truncated" errors when another process is still writing
        obj_locks = []
        try:
            # Acquire .o locks in sorted order to prevent deadlocks when
            # multiple processes link different DLLs with overlapping deps.
            for obj_file in sorted(obj_files):
                obj_lockfile = obj_file + '.lock'
                lock = file_lock(obj_lockfile)
                lock.__enter__()
                obj_locks.append(lock)
            
            # Check if output file already exists and is up-to-date.
            #
            # IMPORTANT (Windows): for DLLs we also require the sidecar export
            # definition + import library to exist; otherwise we must relink to
            # get a usable export table and `.lib`.
            if os.path.exists(output_file):
                output_mtime = os.path.getmtime(output_file)
                obj_mtimes = [os.path.getmtime(obj) for obj in obj_files if os.path.exists(obj)]
                if obj_mtimes and all(output_mtime >= mtime for mtime in obj_mtimes):
                    if sys.platform == 'win32' and shared and output_file.lower().endswith('.dll'):
                        exports_def = os.path.splitext(output_file)[0] + '.exports.def'
                        implib = os.path.splitext(output_file)[0] + '.lib'
                        if os.path.exists(exports_def) and os.path.exists(implib):
                            return output_file
                    else:
                        # Output is up-to-date, skip linking
                        return output_file

            
            # If specific linker requested, try it first then fallback
            if linker:
                linkers = [linker] + [l for l in get_default_linkers() if l != linker]
            else:
                linkers = get_default_linkers()
            
            return try_link_with_linkers(
                obj_files,
                output_file,
                shared=shared,
                linkers=linkers,
                link_objects=link_objects,
                link_libraries=link_libraries,
            )
            
        finally:
            # Release all .o file locks in reverse order
            for lock in reversed(obj_locks):
                lock.__exit__(None, None, None)
