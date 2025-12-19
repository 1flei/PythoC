"""
Linker utilities for PC compiler

Provides unified linker functionality for both executables and shared libraries.
Supports multiple linkers including gcc, clang, and zig for cross-platform compatibility.
"""

import os
import sys
import shutil
import subprocess
import time
from typing import List, Optional
from contextlib import contextmanager

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


def _find_zig_executable() -> Optional[str]:
    """Find zig executable, including python-zig from ziglang package.
    
    Returns:
        Path to zig executable or None if not found
    """
    # First try system zig
    if shutil.which('zig'):
        return 'zig'
    
    # Try python-zig (installed via pip install ziglang)
    if shutil.which('python-zig'):
        return 'python-zig'
    
    # Try running via python -m ziglang
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'ziglang', 'version'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return 'ziglang-module'  # Special marker for module invocation
    except Exception:
        pass
    
    return None


def find_available_linker() -> str:
    """Find an available linker on the system.
    
    Priority order:
    - Unix: cc, gcc, clang, zig
    - Windows: zig, clang-cl, clang, gcc
    
    Returns:
        Name of available linker
        
    Raises:
        RuntimeError: If no linker is found
    """
    if sys.platform == 'win32':
        # On Windows, prefer zig as it provides a complete toolchain
        # Check for zig first (including python-zig)
        zig = _find_zig_executable()
        if zig:
            return zig
        
        for linker in ['clang-cl', 'clang', 'gcc']:
            if shutil.which(linker):
                return linker
    else:
        # On Unix, prefer system cc/gcc/clang, fallback to zig
        for linker in ['cc', 'gcc', 'clang']:
            if shutil.which(linker):
                return linker
        
        zig = _find_zig_executable()
        if zig:
            return zig
    
    raise RuntimeError(
        "No linker found. Please install one of: cc, gcc, clang, or zig.\n"
        "Tip: Install zig via pip: pip install ziglang\n"
        "Or download from: https://ziglang.org/download/"
    )


def get_default_linkers() -> List[str]:
    """Get list of linkers to try in order of preference.
    
    Returns:
        List of linker names (may include special markers like 'ziglang-module')
    """
    linkers = []
    
    if sys.platform == 'win32':
        # On Windows, try zig variants first
        zig = _find_zig_executable()
        if zig:
            linkers.append(zig)
        linkers.extend(['clang-cl', 'clang', 'gcc'])
    else:
        linkers.extend(['cc', 'gcc', 'clang'])
        zig = _find_zig_executable()
        if zig:
            linkers.append(zig)
    
    return linkers


def get_link_flags() -> List[str]:
    """Get link flags from registry
    
    Returns:
        List of linker flags including -l options and direct library paths
    """
    from ..registry import get_unified_registry
    libs = get_unified_registry().get_link_libraries()
    lib_flags = []
    
    for lib in libs:
        if os.path.isabs(lib) or '/' in lib:
            # Full path to library - pass directly to linker
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


def _is_zig_linker(linker: str) -> bool:
    """Check if linker is a zig variant."""
    return linker in ('zig', 'python-zig', 'ziglang-module')


def get_platform_link_flags(shared: bool = False, linker: str = 'gcc') -> List[str]:
    """Get platform-specific link flags
    
    Args:
        shared: True for shared library, False for executable
        linker: Linker being used (affects flag format)
    
    Returns:
        List of platform-specific flags
    """
    is_zig = _is_zig_linker(linker)
    
    if sys.platform == 'win32':
        if is_zig:
            # zig cc uses gcc-compatible flags
            return ['-shared'] if shared else []
        else:
            return ['-shared'] if shared else []
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
            # Allow undefined symbols in shared libraries (for circular dependencies)
            if is_zig:
                # zig cc supports most gcc flags
                return ['-shared', '-fPIC']
            else:
                return ['-shared', '-fPIC', '-Wl,--export-dynamic', 
                       '-Wl,--allow-shlib-undefined', '-Wl,--unresolved-symbols=ignore-all']
        else:
            return []


def build_link_command(obj_files: List[str], output_file: str,
                       shared: bool = False, linker: str = 'gcc') -> List[str]:
    """Build linker command
    
    Args:
        obj_files: List of object file paths
        output_file: Output file path (.so/.dll or executable)
        shared: True for shared library, False for executable
        linker: Linker to use (gcc, cc, clang, zig, python-zig, ziglang-module, etc.)
    
    Returns:
        Link command as list of arguments
    """
    # Handle different zig invocation methods
    if linker == 'zig':
        linker_cmd = ['zig', 'cc']
    elif linker == 'python-zig':
        linker_cmd = ['python-zig', 'cc']
    elif linker == 'ziglang-module':
        linker_cmd = [sys.executable, '-m', 'ziglang', 'cc']
    else:
        linker_cmd = [linker]
    
    platform_flags = get_platform_link_flags(shared, linker=linker)
    lib_flags = get_link_flags()
    
    return linker_cmd + platform_flags + obj_files + ['-o', output_file] + lib_flags


def try_link_with_linkers(obj_files: List[str], output_file: str, 
                         shared: bool = False,
                         linkers: Optional[List[str]] = None) -> str:
    """Try linking with multiple linkers
    
    Args:
        obj_files: List of object file paths
        output_file: Output file path
        shared: True for shared library, False for executable
        linkers: List of linkers to try (defaults to platform-specific list)
    
    Returns:
        Path to linked file
    
    Raises:
        RuntimeError: If all linkers fail
    """
    if linkers is None:
        linkers = get_default_linkers()
    
    errors = []
    for linker in linkers:
        # Check if linker is available (zig variants are pre-checked in get_default_linkers)
        if _is_zig_linker(linker):
            # Zig variants were already validated when added to linkers list
            pass
        elif not shutil.which(linker):
            errors.append(f"{linker}: not found")
            continue
        
        try:
            link_cmd = build_link_command(obj_files, output_file, shared=shared, linker=linker)
            subprocess.run(link_cmd, check=True, capture_output=True, text=True)
            return output_file
        except subprocess.CalledProcessError as e:
            errors.append(f"{linker}: {e.stderr}")
    
    # All linkers failed
    file_type = "shared library" if shared else "executable"
    raise RuntimeError(
        f"Failed to link {file_type} with all linkers ({', '.join(linkers)}):\n" + 
        "\n".join(errors) +
        "\n\nTip: Install 'zig' for cross-platform linking: https://ziglang.org/download/"
    )


def link_files(obj_files: List[str], output_file: str, 
               shared: bool = False, linker: Optional[str] = None) -> str:
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
            for obj_file in obj_files:
                obj_lockfile = obj_file + '.lock'
                lock = file_lock(obj_lockfile)
                lock.__enter__()
                obj_locks.append(lock)
            
            # Check if output file already exists and is up-to-date
            if os.path.exists(output_file):
                output_mtime = os.path.getmtime(output_file)
                obj_mtimes = [os.path.getmtime(obj) for obj in obj_files if os.path.exists(obj)]
                if obj_mtimes and all(output_mtime >= mtime for mtime in obj_mtimes):
                    # Output is up-to-date, skip linking
                    return output_file
            
            # If specific linker requested, try it first then fallback
            if linker:
                linkers = [linker] + [l for l in get_default_linkers() if l != linker]
            else:
                linkers = get_default_linkers()
            
            return try_link_with_linkers(obj_files, output_file, shared=shared, linkers=linkers)
            
        finally:
            # Release all .o file locks in reverse order
            for lock in reversed(obj_locks):
                lock.__exit__(None, None, None)
