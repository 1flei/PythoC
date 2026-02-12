"""
C Compiler utilities for cimport

Provides functions to compile C source files to object files.
Used by cimport to compile .c sources when compile_sources=True.
"""

import os
import sys
import shutil
import hashlib
import subprocess
from typing import List, Optional

from .link_utils import _which_cached, _zig_tool_cmd


def get_cc_candidates() -> List[str]:
    """Get C compiler candidates based on availability.
    
    Returns list of compiler commands in priority order.
    On Windows, only zig is supported — it targets windows-gnu,
    which behaves consistently with Linux and avoids MSVC-specific quirks.

    Prefers the native ``zig.exe cc`` (faster) over ``python-zig cc``.
    """
    candidates = []

    if sys.platform == 'win32':
        # Windows: only zig is supported.
        zig_cc = _zig_tool_cmd('cc')
        if zig_cc:
            candidates.append(' '.join(zig_cc))
    else:
        for cc in ['cc', 'clang', 'gcc']:
            if _which_cached(cc):
                candidates.append(cc)
        # zig as fallback on non-Windows
        zig_cc = _zig_tool_cmd('cc')
        if zig_cc:
            candidates.append(' '.join(zig_cc))
    
    return candidates


def find_available_cc() -> str:
    """Find an available C compiler on the system.
    
    Returns:
        Compiler command string
        
    Raises:
        RuntimeError: If no compiler is found
    """
    candidates = get_cc_candidates()
    if candidates:
        return candidates[0]
    
    if sys.platform == 'win32':
        raise RuntimeError(
            "No C compiler found. On Windows, zig is required.\n"
            "Install via pip: pip install ziglang"
        )
    raise RuntimeError(
        "No C compiler found. Please install one of: cc, gcc, clang, or zig.\n"
        "Tip: Install zig via pip: pip install ziglang"
    )


def compute_source_hash(source_path: str, cc: str, cflags: List[str],
                        include_dirs: List[str], defines: List[str]) -> str:
    """Compute a hash for caching based on source file and compilation options.
    
    Args:
        source_path: Path to source file
        cc: Compiler command
        cflags: Additional compiler flags
        include_dirs: Include directories
        defines: Preprocessor defines
    
    Returns:
        Hex digest of hash
    """
    hasher = hashlib.sha256()
    
    # Include source file path and mtime
    hasher.update(source_path.encode())
    if os.path.exists(source_path):
        hasher.update(str(os.path.getmtime(source_path)).encode())
        # Also hash file content for more accurate caching
        with open(source_path, 'rb') as f:
            hasher.update(f.read())
    
    # Include compiler and flags
    hasher.update(cc.encode())
    for flag in sorted(cflags or []):
        hasher.update(flag.encode())
    for inc in sorted(include_dirs or []):
        hasher.update(inc.encode())
    for define in sorted(defines or []):
        hasher.update(define.encode())
    
    return hasher.hexdigest()[:16]


def compile_c_to_object(source_path: str, output_path: Optional[str] = None,
                        cc: Optional[str] = None, cflags: Optional[List[str]] = None,
                        include_dirs: Optional[List[str]] = None,
                        defines: Optional[List[str]] = None,
                        cache_dir: Optional[str] = None) -> str:
    """Compile a C source file to an object file.
    
    Args:
        source_path: Path to .c source file
        output_path: Path for output .o file (auto-generated if None)
        cc: C compiler to use (auto-detect if None)
        cflags: Additional compiler flags
        include_dirs: Include directories (-I)
        defines: Preprocessor defines (-D)
        cache_dir: Directory for cached objects (uses build/cimport if None)
    
    Returns:
        Path to compiled object file
        
    Raises:
        RuntimeError: If compilation fails
        FileNotFoundError: If source file doesn't exist
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Use default compiler if not specified
    if cc is None:
        cc = find_available_cc()
    
    # Default flags
    cflags = list(cflags or [])
    include_dirs = list(include_dirs or [])
    defines = list(defines or [])
    
    # Compute cache key
    cache_hash = compute_source_hash(source_path, cc, cflags, include_dirs, defines)
    
    # Determine output path
    if output_path is None:
        if cache_dir is None:
            cache_dir = os.path.join('build', 'cimport', cache_hash)
        os.makedirs(cache_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(source_path))[0]
        output_path = os.path.join(cache_dir, f'{basename}.o')
    else:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Check if cached object is up-to-date
    if os.path.exists(output_path):
        source_mtime = os.path.getmtime(source_path)
        obj_mtime = os.path.getmtime(output_path)
        if obj_mtime >= source_mtime:
            return output_path
    
    # Build compilation command
    cmd = cc.split()  # Handle 'python-zig cc'
    
    # Add compilation flags
    # -fPIC is required on ELF/Mach-O; on Windows we use zig (windows-gnu target)
    # which accepts -fPIC harmlessly, so we always pass it for simplicity.
    cmd.extend(['-c', '-fPIC', '-O2'])
    
    # On Windows, zig must target windows-gnu for consistency with the rest of
    # the toolchain (llvmlite generates GNU-format .o files).
    if sys.platform == 'win32' and 'zig' in cc:
        cmd.extend(['-target', 'x86_64-windows-gnu'])
    
    # Add include directories
    for inc in include_dirs:
        cmd.extend(['-I', inc])
    
    # Add defines
    for define in defines:
        cmd.append(f'-D{define}')
    
    # Add user flags
    cmd.extend(cflags)
    
    # Add source and output
    # python-zig changes its working directory internally, so relative paths
    # break. Normalise to absolute paths for robustness.
    cmd.extend([os.path.abspath(source_path), '-o', os.path.abspath(output_path)])
    
    # Run compilation
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True,
            timeout=120, stdin=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Compilation timed out: {source_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Compilation failed for {source_path}:\n{e.stderr}")
    
    return output_path


def compile_c_sources(sources: List[str], cc: Optional[str] = None,
                      cflags: Optional[List[str]] = None,
                      include_dirs: Optional[List[str]] = None,
                      defines: Optional[List[str]] = None,
                      cache_dir: Optional[str] = None) -> List[str]:
    """Compile multiple C source files to object files.
    
    Args:
        sources: List of .c source file paths
        cc: C compiler to use
        cflags: Additional compiler flags
        include_dirs: Include directories
        defines: Preprocessor defines
        cache_dir: Directory for cached objects
    
    Returns:
        List of compiled object file paths
    """
    objects = []
    for source in sources:
        obj = compile_c_to_object(
            source, cc=cc, cflags=cflags,
            include_dirs=include_dirs, defines=defines,
            cache_dir=cache_dir
        )
        objects.append(obj)
    return objects
