# -*- coding: utf-8 -*-
import sys


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
        """Handle @extern function call by lowering to func type.

        This follows the same pattern as @compile wrappers: lower the
        wrapper to a func[...] ValueRef, then delegate to func.handle_call.
        This ensures varargs promotion, ABI coercion, and all other call
        mechanics are handled in one place.
        """
        from ..callable_lowering import lower_extern_wrapper

        caller_group_key = getattr(visitor, 'current_group_key', None)
        lowered = lower_extern_wrapper(
            self, visitor.module, caller_group_key, node=node,
        )
        func_type = lowered.type_hint
        return func_type.handle_call(visitor, lowered, args, node)

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
        # Note: No longer registering in registry - info is stored on wrapper
        wrapper = ExternFunctionWrapper(
            func=f,
            lib=lib or 'c',
            calling_convention=calling_convention,
            return_type=return_type,
            param_types=param_types,
            **kwargs
        )
        wrapper._is_extern = True
        wrapper._extern_config = {
            'lib': lib or 'c',
            'calling_convention': calling_convention,
            'signature': sig,
            'function': f,
            'return_type': return_type,
            'param_types': param_types,
            **kwargs
        }
        return wrapper
    return decorator(func) if func else decorator
