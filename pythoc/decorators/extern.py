# -*- coding: utf-8 -*-
import sys
from typing import Any, List
from ..valueref import ValueRef


class ExternFunctionWrapper:
    def __init__(self, func, lib, calling_convention, return_type, param_types, **kwargs):
        self.func = func
        self.func_name = func.__name__
        self.lib = lib
        self.calling_convention = calling_convention
        self.return_type = return_type
        self.param_types = param_types
        self.config = kwargs
        self._ctypes_func = None

    def handle_call(self, visitor, func_ref, args, node):
        from ..valueref import ensure_ir, wrap_value, get_type
        from llvmlite import ir
        try:
            func = visitor.module.get_global(self.func_name)
        except KeyError:
            if hasattr(visitor, 'compiler') and visitor.compiler:
                func = visitor.compiler._declare_extern_function(self.func_name)
            else:
                raise NameError(f"Extern function '{self.func_name}' not declared in module")
        # args are already pre-evaluated by visit_Call
        has_varargs = any(param_name == 'args' for param_name, _ in self.param_types)
        converted_args = []
        
        if has_varargs:
            fixed_param_count = len([p for p in self.param_types if p[0] != 'args'])
            # Convert fixed params strictly by PC type hints
            for i in range(min(fixed_param_count, len(args))):
                arg = args[i]
                target_pc_type = self.param_types[i][1] if i < len(self.param_types) else None
                if target_pc_type is None:
                    raise TypeError(f"Extern call '{self.func_name}': missing PC type hint for fixed parameter {i}")
                converted = visitor.type_converter.convert(arg, target_pc_type)
                converted_args.append(ensure_ir(converted))
            # Handle varargs: apply C default promotions
            for i in range(fixed_param_count, len(args)):
                arg = args[i]
                arg_hint = getattr(arg, 'type_hint', None)
                
                # Check if arg is a PythonType that needs promotion
                from ..builtin_entities.python_type import PythonType
                if arg_hint is not None and isinstance(arg_hint, PythonType):
                    # Promote to default PC type (int->i32, float->f64)
                    promoted = arg_hint.promote_to_default_pc_type()
                    converted_args.append(ensure_ir(promoted))
                    continue
                
                # If arg is a Python value without PythonType hint (shouldn't happen)
                if isinstance(arg, ValueRef) and arg.is_python_value():
                    python_val = arg.value
                    if isinstance(python_val, bool) or isinstance(python_val, int):
                        from ..builtin_entities import i32 as pc_i32
                        promoted = visitor.type_converter.convert(arg, pc_i32)
                        converted_args.append(ensure_ir(promoted))
                        continue
                    if isinstance(python_val, float):
                        from ..builtin_entities import f64 as pc_f64
                        promoted = visitor.type_converter.convert(arg, pc_f64)
                        converted_args.append(ensure_ir(promoted))
                        continue
                
                # Apply C default argument promotions for varargs
                # - Integer types smaller than int are promoted to int (i32)
                # - float is promoted to double (f64)
                if arg_hint is not None:
                    from ..builtin_entities import i8, i16, i32, f32, f64
                    # Check if it's a small integer type that needs promotion
                    if arg_hint in (i8, i16):
                        # Promote to i32
                        promoted = visitor.type_converter.convert(arg, i32)
                        converted_args.append(ensure_ir(promoted))
                        continue
                    elif arg_hint == f32:
                        # Promote to f64
                        promoted = visitor.type_converter.convert(arg, f64)
                        converted_args.append(ensure_ir(promoted))
                        continue
                    else:
                        # Already promoted type or other type, pass through
                        converted_args.append(ensure_ir(arg))
                else:
                    # No hint, pass as-is
                    converted_args.append(ensure_ir(arg))
        else:
            # Non-varargs: convert all args by PC type hints only
            for i, arg in enumerate(args):
                target_pc_type = self.param_types[i][1] if i < len(self.param_types) else None
                if target_pc_type is None:
                    raise TypeError(f"Extern call '{self.func_name}': missing PC type hint for parameter {i}")
                converted = visitor.type_converter.convert(arg, target_pc_type)
                converted_args.append(ensure_ir(converted))
        
        # Build arg_type_hints for ABI coercion
        arg_type_hints = [pt[1] for pt in self.param_types if pt[0] != 'args']
        
        call_result = visitor.builder.call(
            func, converted_args,
            return_type_hint=self.return_type,
            arg_type_hints=arg_type_hints
        )
        if self.return_type is None:
            from ..builtin_entities.types import void
            return wrap_value(call_result, kind="value", type_hint=void)
        else:
            return wrap_value(call_result, kind="value", type_hint=self.return_type)

    def __call__(self, *args, **kwargs):
        if self._ctypes_func is None:
            self._load_ctypes_function()
        return self._ctypes_func(*args, **kwargs)

    def _load_ctypes_function(self):
        import ctypes
        import platform
        
        # Map library names to platform-specific library paths
        system = platform.system()
        if system == 'Windows':
            lib_map = {
                'c': 'msvcrt.dll',
                'm': 'msvcrt.dll',
                'gcc_s': 'libgcc_s_seh-1.dll',
            }
        else:
            # On Unix, use standard library loading
            # None means current process (for libc functions)
            lib_map = {
                'c': None,
            }
        
        lib_path = lib_map.get(self.lib)
        if lib_path is None and self.lib not in lib_map:
            # Not in map, build default library name based on platform
            if sys.platform == 'win32':
                lib_path = f'{self.lib}.dll'
            else:
                lib_path = f'lib{self.lib}.so'
        
        try:
            lib_handle = ctypes.CDLL(lib_path)
        except OSError as e:
            # Try without 'lib' prefix if it's a custom path
            if not self.lib.startswith('/') and '/' not in self.lib:
                try:
                    lib_handle = ctypes.CDLL(self.lib)
                except OSError:
                    raise OSError(f"Cannot load library '{self.lib}': {e}")
            else:
                raise
        
        self._ctypes_func = getattr(lib_handle, self.func_name)
        # TODO: map argtypes/restype

    def __repr__(self):
        return f"ExternFunctionWrapper({self.func_name}, lib={self.lib})"


def extern(func=None, *, lib=None, calling_convention="cdecl", **kwargs):
    def decorator(f):
        import inspect
        sig = inspect.signature(f)
        return_type = sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None
        param_types = [(name, param.annotation) for name, param in sig.parameters.items()]
        from ..registry import register_extern_function
        register_extern_function(
            name=f.__name__,
            return_type=return_type,
            param_types=param_types,
            lib=lib or 'libc',
            calling_convention=calling_convention,
            signature=sig,
            function=f,
            **kwargs
        )
        wrapper = ExternFunctionWrapper(
            func=f,
            lib=lib or 'libc',
            calling_convention=calling_convention,
            return_type=return_type,
            param_types=param_types,
            **kwargs
        )
        wrapper._is_extern = True
        wrapper._extern_config = {
            'lib': lib or 'libc',
            'calling_convention': calling_convention,
            'signature': sig,
            'function': f,
            'return_type': return_type,
            'param_types': param_types,
            **kwargs
        }
        return wrapper
    return decorator(func) if func else decorator


def get_extern_functions():
    registry = __import__('pythoc.registry', fromlist=['get_unified_registry']).get_unified_registry()
    result = {}
    for func_name in registry.list_extern_functions():
        extern_info = registry.get_extern_function(func_name)
        if extern_info:
            result[func_name] = {
                'lib': extern_info.lib,
                'calling_convention': extern_info.calling_convention,
                'return_type': extern_info.return_type,
                'param_types': extern_info.param_types,
                'signature': extern_info.signature,
                'function': extern_info.function
            }
    return result


def is_extern_function(func_name):
    registry = __import__('pythoc.registry', fromlist=['get_unified_registry']).get_unified_registry()
    return registry.is_extern_function(func_name)


def get_extern_function_info(func_name):
    registry = __import__('pythoc.registry', fromlist=['get_unified_registry']).get_unified_registry()
    extern_info = registry.get_extern_function(func_name)
    if extern_info:
        return {
            'lib': extern_info.lib,
            'calling_convention': extern_info.calling_convention,
            'return_type': extern_info.return_type,
            'param_types': extern_info.param_types,
            'signature': extern_info.signature,
            'function': extern_info.function
        }
    return None
