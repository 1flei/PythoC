# -*- coding: utf-8 -*-
import inspect
import os
import tempfile
import textwrap

# Memo for inspect.getsourcelines per object.  Compilation queries the same
# function/class object's source 2+ times (start line + source text, plus
# class method extraction); each call re-scans the whole source file in
# inspect.findsource.  Same object -> same source, so results are stable;
# entries are identity-validated and the strong reference prevents id reuse.
_OBJECT_SOURCELINES: dict = {}
_OBJECT_SOURCELINES_LIMIT = 100000


def get_object_sourcelines(obj):
    """inspect.getsourcelines() memoized per object identity."""
    key = id(obj)
    ent = _OBJECT_SOURCELINES.get(key)
    if ent is not None and ent[0] is obj:
        return ent[1]
    result = inspect.getsourcelines(obj)
    if len(_OBJECT_SOURCELINES) >= _OBJECT_SOURCELINES_LIMIT:
        _OBJECT_SOURCELINES.clear()
    _OBJECT_SOURCELINES[key] = (obj, result)
    return result


def get_object_source(obj) -> str:
    """inspect.getsource() built on the memoized getsourcelines."""
    lines, _ = get_object_sourcelines(obj)
    return ''.join(lines)


def get_function_source_with_inspect(func):
    """Get function source code.

    Returns:
        str: Dedented source code of the function
    """
    # Check if function has pre-stored source code (for yield-generated functions)
    if hasattr(func, '__pc_source__'):
        return func.__pc_source__

    source = get_object_source(func)
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
        _, start_line = get_object_sourcelines(func)
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
