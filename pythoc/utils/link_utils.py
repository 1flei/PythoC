"""
Linker utilities for PC compiler

Provides unified linker functionality for both executables and shared libraries.
"""

import os
import sys
import subprocess
from typing import List, Optional


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
        return ['-shared', '-undefined', 'dynamic_lookup'] if shared else []
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
    
    link_cmd = build_link_command(obj_files, output_file, shared=shared, linker=linker)
    
    try:
        subprocess.run(link_cmd, check=True, capture_output=True, text=True)
        return output_file
    except subprocess.CalledProcessError as e:
        file_type = "shared library" if shared else "executable"
        raise RuntimeError(f"Failed to link {file_type}: {e.stderr}")
