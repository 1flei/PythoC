"""Build utilities for compiling PC programs to executables.

This module provides high-level functions for building native executables
from @compile decorated Python functions.
"""

import os
import inspect
from typing import Optional, List


def get_source_file_from_caller(offset: int = 0) -> str:
    """Get the source file path from the calling frame.
    
    Args:
        offset: Additional offset in the call stack (default: 0)
    
    Returns:
        Absolute path to the source file
    
    Raises:
        RuntimeError: If source file cannot be detected
    """
    frame = inspect.currentframe()
    if frame and frame.f_back:
        # Walk up the stack: [0] this function, [1] caller, [2+offset] target
        target_frame = frame.f_back
        for _ in range(offset + 1):
            if target_frame and target_frame.f_back:
                target_frame = target_frame.f_back
            else:
                break
        
        if target_frame:
            return target_frame.f_code.co_filename
    
    raise RuntimeError("Cannot detect source file. Please provide source_file parameter.")


def determine_output_path(source_file: str, output_path: Optional[str] = None) -> str:
    """Determine the output executable path.
    
    Args:
        source_file: Absolute path to the source file
        output_path: Optional custom output path
    
    Returns:
        Absolute path for the output executable (with .exe on Windows)
    """
    import sys
    
    if output_path is not None:
        # Add .exe extension on Windows if not already present
        if sys.platform == 'win32' and not output_path.endswith('.exe'):
            return output_path + '.exe'
        return output_path
    
    cwd = os.getcwd()
    if source_file.startswith(cwd):
        rel_path = os.path.relpath(source_file, cwd)
    else:
        rel_path = source_file
    
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    output_dir = os.path.join('build', os.path.dirname(rel_path))
    exe_path = os.path.join(output_dir, base_name)
    
    # Add .exe extension on Windows
    if sys.platform == 'win32':
        exe_path += '.exe'
    
    return exe_path


def get_object_file_path(source_file: str) -> str:
    """Convert source file path to corresponding object file path.
    
    Args:
        source_file: Absolute path to the source file
    
    Returns:
        Path to the object file in build directory
    """
    cwd = os.getcwd()
    if source_file.startswith(cwd):
        rel_path = os.path.relpath(source_file, cwd)
    else:
        rel_path = source_file
    
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    return os.path.join('build', os.path.dirname(rel_path), base_name + '.o')


def collect_object_files(source_files: List[str]) -> List[str]:
    """Collect all object files from source files.
    
    Args:
        source_files: List of source file paths
    
    Returns:
        List of object file paths (skips yield-only source files)
    
    Note:
        Yield functions are inlined and don't generate .o files.
        Such source files are automatically skipped.
    """
    obj_files = []
    for src_file in source_files:
        obj_file = get_object_file_path(src_file)
        if os.path.exists(obj_file):
            obj_files.append(obj_file)
        # else: Skip - likely a yield-only source file that gets inlined
    
    if not obj_files:
        raise RuntimeError(
            "No object files found. "
            "Make sure you have @compile functions before calling compile_to_executable()."
        )
    
    return obj_files


def link_executable(obj_files: List[str], output_path: str) -> str:
    """Link object files into a native executable.
    
    Tries linkers in the following order:
    1. cc (system default compiler)
    2. gcc (GNU C compiler)
    3. clang (LLVM C compiler)
    
    Args:
        obj_files: List of object file paths
        output_path: Path for the output executable
    
    Returns:
        Path to the created executable
    
    Raises:
        RuntimeError: If linking fails with all available linkers
    """
    from .link_utils import try_link_with_linkers
    
    result = try_link_with_linkers(obj_files, output_path, shared=False)
    print(f"Successfully compiled to executable: {output_path}")
    print(f"Linked {len(obj_files)} object file(s)")
    return result


def compile_to_executable(output_path: Optional[str] = None, source_file: Optional[str] = None) -> str:
    """Compile all @compile decorated functions to a native executable.
    
    This function automatically:
    1. Flushes all pending compilations
    2. Collects all compiled modules and their dependencies
    3. Links object files together
    4. Generates a native executable binary
    
    Args:
        output_path: Optional output path for the executable.
                    If not provided, defaults to build/<source_file_name>
                    Example: 'build/test/example/pc_binary_tree_test'
        source_file: Optional source file path. If not provided, automatically
                    detected from the calling frame.
    
    Returns:
        Path to the created executable
    
    Raises:
        RuntimeError: If no functions are compiled or linking fails
    
    Example:
        from pythoc import compile_to_executable
        
        # Auto-detect paths
        compile_to_executable()
        
        # Specify output path
        compile_to_executable('build/myprogram')
    """
    from ..decorators.compile import flush_all_pending_outputs
    from ..decorators.compile import get_output_manager
    
    # Flush all pending output files before checking for object files
    flush_all_pending_outputs()
    
    output_manager = get_output_manager()
    
    # Auto-detect source file from caller
    if source_file is None:
        source_file = get_source_file_from_caller(offset=0)
    
    source_file = os.path.abspath(source_file)
    
    # Determine output path
    output_path = determine_output_path(source_file, output_path)
    
    # Get all source files that have been compiled from output manager
    all_source_files = []
    for group_key, group in output_manager._pending_groups.items():
        src_file = group.get('source_file')
        if src_file and src_file not in all_source_files:
            all_source_files.append(src_file)
    
    if not all_source_files:
        raise RuntimeError("No @compile decorated functions found. Nothing to compile.")
    
    # Collect all object files
    obj_files = collect_object_files(all_source_files)
    
    # Link to create executable
    return link_executable(obj_files, output_path)
