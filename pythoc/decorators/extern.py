# -*- coding: utf-8 -*-
import inspect
import os
import sys


def _load_lib_handle(lib):
    """Resolve a lib spec to a ctypes library handle.

    Resolution rules (shared by extern functions and extern globals):
    - '' -> the current process (symbols from registered .o files; group
      .so objects are dlopened RTLD_GLOBAL after compilation)
    - 'c' -> the current process on Unix, the Universal CRT on Windows
    - a bare name -> lib{name}.so / {name}.dll
    - anything containing '/' -> treated as a path
    """
    import ctypes
    from ctypes.util import find_library
    from ..utils.c_runtime import load_c_runtime

    if not lib:
        # Symbols from registered object files (cimport compile_sources
        # products) live in the process-global extern-objects bundle; make
        # sure it is linked and dlopen'ed before resolving through the
        # global namespace so this works before any @compile call.
        if sys.platform == 'win32':
            # Windows has no process-global dynamic symbol namespace that
            # ctypes can reach (ctypes.CDLL(None) is not the global symbol
            # table, and on Python 3.12+ it raises TypeError outright), and
            # registry link objects are statically linked into each group
            # DLL instead of a shared bundle.  Python-side access to such
            # symbols is therefore unsupported; @compile code is unaffected
            # (it links the objects directly).
            raise RuntimeError(
                "Python-side access to process-global symbols (lib='') is "
                "not supported on Windows; call the function from @compile "
                "code instead"
            )
        from ..utils.link_utils import ensure_link_objects_loaded
        ensure_link_objects_loaded()
        return ctypes.CDLL(None)

    # Map library names to platform-specific library paths
    if lib == 'c' or (sys.platform == 'win32' and lib == 'm'):
        return load_c_runtime()
    if os.path.isabs(lib) or '/' in lib or '\\' in lib:
        return ctypes.CDLL(lib)
    lib_path = find_library(lib)
    if lib_path is None:
        if sys.platform == 'win32':
            lib_path = 'libgcc_s_seh-1.dll' if lib == 'gcc_s' else f'{lib}.dll'
        else:
            extension = '.dylib' if sys.platform == 'darwin' else '.so'
            lib_path = f'lib{lib}{extension}'

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
    def __init__(self, func, lib, calling_convention, **kwargs):
        if calling_convention not in ('cdecl', 'stdcall'):
            raise TypeError(f'Unsupported calling convention: {calling_convention}')
        self.func = func
        self.func_name = func.__name__
        # The C symbol name used for linking and ctypes lookup.  Defaults to
        # the Python function name, but can be overridden when the Python
        # identifier must differ from the C symbol (e.g. ``raise_`` -> ``raise``,
        # or a function whose name collides with a type name).
        self.c_name = kwargs.pop('name', func.__name__)
        self.lib = lib
        self.calling_convention = calling_convention
        self.config = kwargs
        self._ctypes_func = None
        # Signature types are resolved lazily on first access instead of at
        # decoration time.  C's textual ``#include`` semantics permit two
        # headers to reference each other's types (a.h defines a typedef,
        # then includes b.h whose declarations use it); under Python module
        # semantics such mutual references are unavoidable import cycles, so
        # a generated ``@extern`` declaration may run while the name it
        # annotates is bound in neither its own globals nor any importable
        # module.  Deferring resolution to first use lets the name arrive
        # through the module globals (once the defining module finishes) or
        # the session forward-ref registry, mirroring how @compile already
        # defers annotation handling to flush time.
        self._raw_annotations = dict(getattr(func, '__annotations__', None) or {})
        self._signature = inspect.signature(func)
        self._resolved_types = None
        from ..type_resolver import TypeResolver
        names = set()
        for annotation in self._raw_annotations.values():
            if isinstance(annotation, str):
                names.update(TypeResolver.expression_names(annotation))
        self._annotation_namespace = TypeResolver.capture_type_namespace(names, definition=func)

    def _resolve_annotation(self, annotation, owner, func, func_name):
        """Resolve one string annotation to a type object.

        Plain names that are unbound in the evaluation namespace fall back
        to the session forward-ref registry (fail-loud with context when
        the registry cannot help either).
        """
        from .annotation_resolver import build_annotation_namespace

        is_dynamic = '.<locals>.' in func.__qualname__
        eval_namespace = build_annotation_namespace(
            self._annotation_namespace, is_dynamic=is_dynamic,
        )
        try:
            from ..type_resolver import TypeResolver
            return TypeResolver.resolve_type(annotation, eval_namespace, freeze=True)
        except NameError as e:
            raise NameError(
                f"@extern '{func_name}': {owner} annotation {annotation!r} "
                f"is not defined in module {func.__module__!r} nor in the "
                f"forward-ref registry (original error: {e})"
            ) from e

    def _ensure_resolved(self):
        if self._resolved_types is not None:
            return
        resolved = {}
        for name, annotation in self._raw_annotations.items():
            if isinstance(annotation, str):
                resolved[name] = self._resolve_annotation(
                    annotation, f"parameter '{name}'", self.func, self.func_name)
            else:
                resolved[name] = annotation

        sig = self._signature
        return_type = resolved.get('return', sig.return_annotation)
        if return_type is inspect.Signature.empty:
            return_type = None

        param_types = []
        for name, param in sig.parameters.items():
            if name in resolved:
                param_type = resolved[name]
            elif isinstance(param.annotation, str):
                param_type = self._resolve_annotation(
                    param.annotation, f"parameter '{name}'",
                    self.func, self.func_name)
            else:
                # Preserve the historical behavior for unannotated params:
                # the empty marker flows through and consumers report it.
                param_type = param.annotation
            param_types.append((name, param_type))
        self._resolved_types = (return_type, param_types)

    @property
    def return_type(self):
        self._ensure_resolved()
        return self._resolved_types[0]

    @property
    def param_types(self):
        self._ensure_resolved()
        return self._resolved_types[1]

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
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if self._ctypes_func is None:
            self._load_ctypes_function()
        values = [self._ctypes_argument(bound.arguments[name], ctype)
                  for name, ctype in self._ctypes_params]
        if self._ctypes_varargs is not None:
            values.extend(self._ctypes_vararg(value)
                          for value in bound.arguments[self._ctypes_varargs])
        return self._ctypes_func(*values)

    @staticmethod
    def _ctypes_argument(value, ctype):
        convert = getattr(value, '_to_ctypes', None)
        if convert is not None:
            return convert(ctype)
        return value.encode('utf-8') if isinstance(value, str) else value

    @classmethod
    def _ctypes_vararg(cls, value):
        import ctypes
        from ..type_converter import TypeConverter
        from ..type_descriptor import default_argument_promotion
        from ..builtin_entities.pc_literal import pc_literal

        if isinstance(value, (str, bytes)):
            return cls._ctypes_argument(value, ctypes.c_void_p)
        if value is None:
            return ctypes.c_void_p()
        if isinstance(value, ctypes.c_float):
            return ctypes.c_double(value.value)
        if isinstance(value, pc_literal):
            source = value._pc_type
        elif isinstance(value, (int, float)):
            source = TypeConverter.infer_default_pc_type_from_python(value)
        else:
            return value
        target = default_argument_promotion(source) or source
        ctype = cls._ctypes_type(target)
        converted = cls._ctypes_argument(value, ctype)
        return converted if isinstance(converted, ctype) else ctype(converted)

    @staticmethod
    def _ctypes_type(pc_type):
        """Require a faithful libffi carrier before entering native code."""
        import ctypes
        from ..type_graph import prepare_type
        from ..type_converter import strip_qualifiers

        if pc_type is None:
            return None
        prepare_type(pc_type)
        pc_type = strip_qualifiers(pc_type)
        from ..schema_protocol import get_field_layout_types
        from ..type_converter import get_base_type

        pending, seen = [pc_type], set()
        while pending:
            stored = get_base_type(pending.pop())
            if stored in seen:
                continue
            seen.add(stored)
            layout_config = getattr(stored, '_layout_config', None)
            if (getattr(stored, '_is_union', False)
                    or (layout_config is not None and not layout_config.is_natural)):
                raise TypeError(
                    f'{stored.get_name()} requires a native trampoline '
                    'for Python extern calls')
            if getattr(stored, '_is_integer', False):
                width = stored._bit_width
                carrier = stored.get_ctypes_type()
                if width > 64 or ctypes.sizeof(carrier) * 8 != width:
                    raise TypeError(
                        f'{stored.get_name()} requires a native trampoline '
                        'for Python extern calls')
            pending.extend(get_field_layout_types(stored))
            if getattr(stored, 'dimensions', None) is not None:
                pending.append(stored.element_type)
            if getattr(stored, 'param_types', None) is not None:
                pending.extend((*stored.param_types, stored.return_type))
        ctype = pc_type.get_ctypes_type()

        if ctype is not None and issubclass(ctype, ctypes.Array):
            raise TypeError('Python extern calls do not accept arrays by value')
        layout = pc_type.get_layout()
        if ctype is None:
            if layout.allocation_size:
                raise TypeError(f'{pc_type.get_name()} has no ctypes representation')
        elif (ctypes.sizeof(ctype) != layout.allocation_size
              or ctypes.alignment(ctype) != layout.alignment):
            raise TypeError(f'ctypes layout differs from {pc_type.get_name()}')
        return ctype

    def _load_ctypes_function(self):
        import ctypes
        from ..callable_lowering import extern_callable_type

        callable_type, params, varargs = extern_callable_type(self)
        param_ctypes = [(name, self._ctypes_type(pc_type)) for name, pc_type in params]
        if any(ctype is None for _, ctype in param_ctypes):
            raise TypeError('Python extern calls do not accept zero-sized parameters')
        restype = self._ctypes_type(callable_type.return_type)
        prototype_factory = ctypes.CFUNCTYPE
        if self.calling_convention == 'stdcall':
            if sys.platform != 'win32':
                raise TypeError('Python stdcall extern calls require Windows')
            prototype_factory = ctypes.WINFUNCTYPE
        prototype = prototype_factory(restype, *(ctype for _, ctype in param_ctypes))
        lib_handle = _load_lib_handle(self.lib)
        self._ctypes_params = param_ctypes
        self._ctypes_varargs = varargs
        self._ctypes_func = prototype((self.c_name, lib_handle))

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
        # Annotation resolution is deferred to first use of the wrapper's
        # return_type/param_types (see ExternFunctionWrapper): decoration
        # time may legitimately run before an annotated name is bound, e.g.
        # in generated bindings whose headers mutually reference types.
        # Note: No longer registering in registry - info is stored on wrapper
        # Preserve an explicit lib='' (process-global symbols from registered
        # object files); only a missing lib defaults to 'c'.
        wrapper = ExternFunctionWrapper(
            func=f,
            lib=lib if lib is not None else 'c',
            calling_convention=calling_convention,
            **kwargs
        )
        wrapper._is_extern = True
        wrapper._extern_config = {
            'lib': lib if lib is not None else 'c',
            'calling_convention': calling_convention,
            'signature': inspect.signature(f),
            'function': f,
            **kwargs
        }
        return wrapper
    return decorator(func) if func else decorator
