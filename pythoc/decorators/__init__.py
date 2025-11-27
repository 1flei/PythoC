# -*- coding: utf-8 -*-
"""
pythoc.decorators package facade
- Re-exports public API: compile, inline, jit, extern, ExternFunctionWrapper, helpers
- Internals live in sibling modules: runtime, structs, extern, inline, jit, mangling
"""
from functools import wraps
import inspect
import os
import ast
from typing import Any, List

from ..compiler import LLVMCompiler
from ..registry import register_struct_from_class, _unified_registry

from .structs import (
    add_struct_handle_call as _add_struct_handle_call,
    compile_dynamic_class as _compile_dynamic_class,
)
from .extern import (
    extern,
    ExternFunctionWrapper,
    get_extern_functions,
    is_extern_function,
    get_extern_function_info,
)
from .inline import inline
from .jit import jit
from .mangling import mangle_function_name as _mangle_function_name
from .compile import compile


def _get_registry():
    return _unified_registry


# registry helpers
def get_compiler(source_file):
    return _get_registry().get_compiler(source_file)


def list_compiled_functions():
    return _get_registry().list_compiled_functions()


def clear_registry():
    _get_registry().clear_all()
    # Also clear the native executor cache
    from ..native_executor import get_multi_so_executor
    executor = get_multi_so_executor()
    executor.clear()


def get_function_source(func_name, source_file=None):
    return _get_registry().get_function_source(func_name, source_file)


def list_function_sources():
    return _get_registry().list_function_sources()
