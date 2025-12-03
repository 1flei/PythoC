# -*- coding: utf-8 -*-
import inspect
import os
import tempfile
import textwrap


def get_function_source_with_inspect(func):
    """Get function source code.
    
    Returns:
        str: Dedented source code of the function
    """
    # Check if function has pre-stored source code (for yield-generated functions)
    if hasattr(func, '__pc_source__'):
        return func.__pc_source__
    
    source = inspect.getsource(func)
    dedented_source = textwrap.dedent(source)
    return dedented_source


def get_function_start_line(func) -> int:
    """Get the starting line number of a function in its source file.
    
    Returns:
        int: The line number where the function definition starts (1-indexed),
             or 1 if it cannot be determined.
    """
    # Check if function has pre-stored line number
    if hasattr(func, '__pc_start_line__'):
        return func.__pc_start_line__
    
    try:
        # getsourcelines returns (lines, start_line_number)
        _, start_line = inspect.getsourcelines(func)
        return start_line
    except (OSError, TypeError):
        return 1


def get_function_file_with_inspect(func):
    """Get the source file path of a function.
    
    Returns:
        str or None: Path to the source file, or None if not available.
    """
    try:
        source_file = inspect.getfile(func)
        return source_file
    except (OSError, TypeError):
        return None


def get_function_file_and_source(func):
    """Get both source file path and source code for a function.
    
    Returns:
        tuple[str, str]: (source_file_path, source_code)
        
    Raises:
        RuntimeError: If source code cannot be obtained.
    """
    source_file = get_function_file_with_inspect(func)
    source_code = get_function_source_with_inspect(func)
    
    if source_code is None:
        raise RuntimeError(f"Cannot get source code for function {func.__name__}")
    
    # Handle python -c case or other cases where file is not available
    if source_file is None or source_file == '<stdin>':
        # Write source to a temporary file
        fd, source_file = tempfile.mkstemp(suffix='.py', prefix='pc_tmp_')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(source_code)
        except:
            os.close(fd)
            raise
    
    return source_file, source_code


def get_function_file_source_and_line(func):
    """Get source file path, source code, and starting line number.
    
    This is the preferred function to use when compiling, as it provides
    all information needed for accurate error messages.
    
    Returns:
        tuple[str, str, int]: (source_file_path, source_code, start_line)
        
    Raises:
        RuntimeError: If source code cannot be obtained.
    """
    source_file, source_code = get_function_file_and_source(func)
    start_line = get_function_start_line(func)
    return source_file, source_code, start_line
