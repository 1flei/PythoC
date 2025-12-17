"""
Linker utilities for PC compiler

Provides unified linker functionality for both executables and shared libraries.
"""

import os
import sys
import subprocess
import time
import fcntl
from typing import List, Optional
from contextlib import contextmanager


from contextlib import contextmanager


@contextmanager
def file_lock(lockfile_path: str, timeout: float = 60.0):
    """
    Context manager for file-based locking to prevent concurrent builds.
    
    Args:
        lockfile_path: Path to the lock file
        timeout: Maximum time to wait for lock (seconds)
    
    Yields:
        None when lock is acquired
    
    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """
    lockfile = None
    start_time = time.time()
    
    try:
        # Ensure lock directory exists
        lock_dir = os.path.dirname(lockfile_path)
        if lock_dir and not os.path.exists(lock_dir):
            os.makedirs(lock_dir, exist_ok=True)
        
        # Try to acquire lock with exponential backoff
        while True:
            try:
                lockfile = open(lockfile_path, 'a')
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # Lock acquired
            except (IOError, OSError):
                # Lock is held by another process
                if lockfile:
                    lockfile.close()
                    lockfile = None
                
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Failed to acquire lock on {lockfile_path} within {timeout}s")
                
                # Exponential backoff: 10ms, 20ms, 40ms, ..., max 500ms
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


def get_link_flags() -> List[str]:
    """Get link flags from registry
    
    Returns:
        List of linker flags including -l options
    """
    from ..registry import get_unified_registry
    libs = get_unified_registry().get_link_libraries()
    lib_flags = [f'-l{lib}' for lib in libs]
    
    # Add --no-as-needed to ensure all libraries are linked
    # This is critical for libraries like libgcc_s that provide
    # soft-float support functions (e.g., for f16/bf16/f128)
    if lib_flags and sys.platform not in ('win32', 'darwin'):
        lib_flags = ['-Wl,--no-as-needed'] + lib_flags
    
    return lib_flags


def get_platform_link_flags(shared: bool = False) -> List[str]:
    """Get platform-specific link flags
    
    Args:
        shared: True for shared library, False for executable
    
    Returns:
        List of platform-specific flags
    """
    if sys.platform == 'win32':
        return ['-shared'] if shared else []
    elif sys.platform == 'darwin':
        # Explicitly specify architecture to avoid x86_64/arm64 mismatch issues
        import platform
        arch = platform.machine()
        arch_flag = ['-arch', arch] if arch in ('arm64', 'x86_64') else []
        if shared:
            return arch_flag + ['-shared', '-undefined', 'dynamic_lookup']
        else:
            return arch_flag
    else:  # Linux
        if shared:
            # Allow undefined symbols in shared libraries (for circular dependencies)
            # --unresolved-symbols=ignore-all: don't error on undefined symbols
            # --allow-shlib-undefined: allow undefined symbols from other shared libs
            # --export-dynamic: export all symbols to dynamic symbol table
            return ['-shared', '-fPIC', '-Wl,--export-dynamic', 
                   '-Wl,--allow-shlib-undefined', '-Wl,--unresolved-symbols=ignore-all']
        else:
            return []


def build_link_command(obj_files: List[str], output_file: str, 
                       shared: bool = False, linker: str = 'gcc') -> List[str]:
    """Build linker command
    
    Args:
        obj_files: List of object file paths
        output_file: Output file path (.so or executable)
        shared: True for shared library, False for executable
        linker: Linker to use (gcc, cc, clang, etc.)
    
    Returns:
        Link command as list of arguments
    """
    platform_flags = get_platform_link_flags(shared)
    lib_flags = get_link_flags()
    
    return [linker] + platform_flags + obj_files + ['-o', output_file] + lib_flags


def try_link_with_linkers(obj_files: List[str], output_file: str, 
                         shared: bool = False,
                         linkers: Optional[List[str]] = None) -> str:
    """Try linking with multiple linkers
    
    Args:
        obj_files: List of object file paths
        output_file: Output file path
        shared: True for shared library, False for executable
        linkers: List of linkers to try (defaults to ['cc', 'gcc', 'clang'])
    
    Returns:
        Path to linked file
    
    Raises:
        RuntimeError: If all linkers fail
    """
    if linkers is None:
        linkers = ['cc', 'gcc', 'clang']
    
    errors = []
    for linker in linkers:
        try:
            link_cmd = build_link_command(obj_files, output_file, shared=shared, linker=linker)
            subprocess.run(link_cmd, check=True, capture_output=True, text=True)
            return output_file
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            if isinstance(e, subprocess.CalledProcessError):
                errors.append(f"{linker}: {e.stderr}")
            else:
                errors.append(f"{linker}: not found")
    
    # All linkers failed
    file_type = "shared library" if shared else "executable"
    raise RuntimeError(
        f"Failed to link {file_type} with all linkers ({', '.join(linkers)}):\n" + 
        "\n".join(errors)
    )


def link_files(obj_files: List[str], output_file: str, 
               shared: bool = False, linker: str = 'gcc') -> str:
    """Link object files to executable or shared library
    
    Args:
        obj_files: List of object file paths
        output_file: Output file path
        shared: True for shared library, False for executable
        linker: Linker to use (default: gcc)
    
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
            
            link_cmd = build_link_command(obj_files, output_file, shared=shared, linker=linker)
            
            try:
                subprocess.run(link_cmd, check=True, capture_output=True, text=True)
                return output_file
            except subprocess.CalledProcessError as e:
                file_type = "shared library" if shared else "executable"
                raise RuntimeError(f"Failed to link {file_type}: {e.stderr}")
        finally:
            # Release all .o file locks in reverse order
            for lock in reversed(obj_locks):
                lock.__exit__(None, None, None)
