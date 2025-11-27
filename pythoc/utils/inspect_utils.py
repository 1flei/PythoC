# -*- coding: utf-8 -*-
import inspect
import os
import tempfile
import textwrap


def get_function_source_with_inspect(func):
    # Check if function has pre-stored source code (for yield-generated functions)
    if hasattr(func, '__pc_source__'):
        return func.__pc_source__
    
    source = inspect.getsource(func)
    dedented_source = textwrap.dedent(source)
    return dedented_source

def get_function_file_with_inspect(func):
    try:
        source_file = inspect.getfile(func)
        return source_file
    except (OSError, TypeError):
        return None

def get_function_file_and_source(func):
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
