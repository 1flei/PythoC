# -*- coding: utf-8 -*-
import inspect
import sys


def _load_lib_handle(lib):
    """Resolve a lib spec to a ctypes library handle.

    Resolution rules (shared by extern functions and extern globals):
    - '' -> the current process (symbols from registered .o files; group
      .so objects are dlopened RTLD_GLOBAL after compilation)
    - 'c' -> the current process on Unix, msvcrt.dll on Windows
    - a bare name -> lib{name}.so / {name}.dll
    - anything containing '/' -> treated as a path
    """
    import ctypes
    import platform

    if not lib:
        # Symbols from registered object files (cimport compile_sources
        # products) live in the process-global extern-objects bundle; make
        # sure it is linked and dlopen'ed before resolving through the
        # global namespace so this works before any @compile call.
        from ..utils.link_utils import ensure_link_objects_loaded
        ensure_link_objects_loaded()
        return ctypes.CDLL(None)

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

    lib_path = lib_map.get(lib)
    if lib_path is None and lib not in lib_map:
        # Not in map, build default library name based on platform
        if sys.platform == 'win32':
            lib_path = f'{lib}.dll'
        else:
            lib_path = f'lib{lib}.so'

    try:
        return ctypes.CDLL(lib_path)
    except OSError as e:
        # Try without 'lib' prefix if it's a custom path
        if not lib.startswith('/') and '/' not in lib:
            try:
                return ctypes.CDLL(lib)
            except OSError:
                raise OSError(f"Cannot load library '{lib}': {e}")
        raise


class ExternFunctionWrapper:
    def __init__(self, func, lib, calling_convention, return_type, param_types, **kwargs):
        self.func = func
        self.func_name = func.__name__
        # The C symbol name used for linking and ctypes lookup.  Defaults to
        # the Python function name, but can be overridden when the Python
        # identifier must differ from the C symbol (e.g. ``raise_`` -> ``raise``,
        # or a function whose name collides with a type name).
        self.c_name = kwargs.pop('name', func.__name__)
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
        lib_handle = _load_lib_handle(self.lib)
        self._ctypes_func = getattr(lib_handle, self.c_name)
        # TODO: map argtypes/restype

    def __repr__(self):
        return f"ExternFunctionWrapper({self.func_name}, lib={self.lib})"


def _extern_class(cls):
    """Treat a decorated class as a declaration of its static members.

    ``@extern`` class statics lower to external-linkage LLVM globals with no
    initializer, matching a C ``extern`` object declaration.  The class is
    still processed as a ``@compile`` struct so ``Cls.member`` access works.
    """
    from .structs import compile_dynamic_class
    compile_dynamic_class(cls, static_linkage='external')
    cls._static_is_decl = True
    return cls


class ExternGlobal:
    """Declaration of an external global variable with a raw C symbol name.

    Unlike ``@extern`` class statics (whose LLVM symbol is mangled to
    ``Class.member``), an ``ExternGlobal`` binds an unmangled C symbol such
    as ``global_counter``, so it can reference globals defined by C libraries
    or by object files registered for linking.

    Inside ``@compile`` functions the binding resolves to an address ValueRef
    backed by a declaration-only LLVM global, so both reads and writes work.
    From plain Python, ``.value`` reads/writes the storage through
    ``ctypes.in_dll`` against the resolved library handle.

    Attributes:
        pc_type: PC type of the global (e.g. ``i32``); a ``thread_local[T]``
            qualified type marks the LLVM global as thread-local.
        c_name: Unmangled C symbol name.  When None, the name of the binding
            used inside compiled code is adopted on first lowering.
        lib: Library spec, resolved the same way as ``@extern(lib=...)``
            ('c', bare library name, or a path; '' means the symbol comes
            from registered object files).
    """

    _is_extern_global = True

    def __init__(self, pc_type, name=None, lib=None):
        self.pc_type = pc_type
        self.c_name = name
        self.lib = lib if lib is not None else 'c'
        self._ctypes_var = None

    def lower_to_module(self, visitor, binding_name=None, node=None):
        """Declare the global in the visitor's module and return its address.

        Returns a ValueRef(kind='address') referring to the extern symbol;
        loads/stores go through the same lvalue machinery as class statics.
        Also records the library dependency so JIT linking and AOT builds
        resolve the symbol, mirroring extern function lowering.
        """
        from llvmlite import ir
        from ..valueref import wrap_value
        from ..logger import logger
        from ..ir_helpers import is_thread_local

        module = visitor.module
        c_name = self.c_name or binding_name
        if not c_name:
            logger.error(
                "extern_global requires a C symbol name: pass name=... "
                "explicitly",
                node=node, exc_type=ValueError,
            )
        if module is None:
            logger.error(
                f"Extern global '{c_name}' cannot be used in compile-time "
                f"constant evaluation",
                node=node, exc_type=ValueError,
            )
        if self.c_name is None:
            self.c_name = c_name

        try:
            global_var = module.get_global(c_name)
        except KeyError:
            llvm_type = self.pc_type.get_llvm_type(module.context)
            global_var = ir.GlobalVariable(module, llvm_type, c_name)
            if is_thread_local(self.pc_type):
                global_var.storage_class = 'thread_local'

        # Record link library dependency (same as extern function lowering)
        lib = self.lib
        if lib:
            from ..registry import get_unified_registry
            get_unified_registry().add_link_library(lib)
            caller_group_key = getattr(visitor, 'current_group_key', None)
            if caller_group_key:
                from ..build.deps import get_dependency_tracker
                get_dependency_tracker().record_extern_dependency(
                    caller_group_key, [lib],
                )

        return wrap_value(global_var, kind='address', type_hint=self.pc_type,
                          address=global_var)

    @property
    def value(self):
        """Read the current value of the global from plain Python."""
        return self._ctypes_binding().value

    @value.setter
    def value(self, new_value):
        """Write the global from plain Python."""
        self._ctypes_binding().value = new_value

    def _ctypes_binding(self):
        if self._ctypes_var is None:
            if not self.c_name:
                raise RuntimeError(
                    "extern_global has no C symbol name: pass name=... "
                    "explicitly")
            ctypes_type = self.pc_type.get_ctypes_type()
            if ctypes_type is None:
                raise RuntimeError(
                    f"extern global '{self.c_name}' has no ctypes mapping "
                    f"for type {self.pc_type}")
            lib_handle = _load_lib_handle(self.lib)
            try:
                self._ctypes_var = ctypes_type.in_dll(lib_handle, self.c_name)
            except ValueError as e:
                raise RuntimeError(
                    f"extern global '{self.c_name}' is not available "
                    f"(lib={self.lib!r}): {e}. If the symbol comes from "
                    f"compiled sources, make sure the cimport(..., "
                    f"compile_sources=True) call that registers their "
                    f"object files ran in this process.")
        return self._ctypes_var

    def __repr__(self):
        type_name = (self.pc_type.get_name()
                     if hasattr(self.pc_type, 'get_name')
                     else repr(self.pc_type))
        return f"ExternGlobal({self.c_name!r}: {type_name}, lib={self.lib!r})"


def extern_global(pc_type, name=None, lib=None):
    """Declare a module-level extern global with a raw (unmangled) C symbol.

    Usage:
        global_counter = extern_global(i32, 'global_counter', lib='mylib')

    The binding is usable both inside @compile functions (read and write)
    and from plain Python via ``.value``.
    """
    return ExternGlobal(pc_type, name=name, lib=lib)


def extern(func=None, *, lib=None, calling_convention="cdecl", **kwargs):
    def decorator(f):
        if inspect.isclass(f):
            return _extern_class(f)
        sig = inspect.signature(f)
        resolved_annotations = {}
        if getattr(f, '__annotations__', None):
            from ..type_resolver import TypeResolver
            from .annotation_resolver import (
                build_annotation_namespace,
                resolve_annotations_dict,
            )

            is_dynamic = '.<locals>.' in f.__qualname__
            type_resolver = TypeResolver(user_globals=f.__globals__)
            eval_namespace = build_annotation_namespace(
                f.__globals__, is_dynamic=is_dynamic,
            )
            resolved_annotations = resolve_annotations_dict(
                f.__annotations__, eval_namespace, type_resolver,
            )

        return_type = resolved_annotations.get('return', sig.return_annotation)
        if return_type == inspect.Signature.empty:
            return_type = None

        param_types = []
        for name, param in sig.parameters.items():
            param_type = resolved_annotations.get(name, param.annotation)
            param_types.append((name, param_type))
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
