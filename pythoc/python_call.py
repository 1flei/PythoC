"""Python call endpoint for @compile functions.

PythoC -> PythoC lowering continues to use the compile descriptor attached to
the callable. This module only binds the Python vectorcall endpoint.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import sys
import sysconfig
import threading
from typing import Any, Dict, Optional

_extension_lock = threading.RLock()
_extension_module = None
_building_runtime = False
_loaded_extensions = []
_manifest_modules: Dict[str, Any] = {}


class _RuntimeAnchor:
    """Compile descriptor used while the callable runtime is itself compiling.

    The runtime object file does not need a Python vectorcall endpoint.
    """

    def is_fast_bound(self):
        return False


def _runtime_anchor(func):
    anchor = _RuntimeAnchor()
    name = getattr(func, '__name__', 'fn')
    anchor.__name__ = name
    anchor.__qualname__ = getattr(func, '__qualname__', name)
    anchor.__module__ = getattr(func, '__module__', None)
    anchor.__doc__ = getattr(func, '__doc__', None)
    return anchor


def ensure_callable_extension():
    """Import the PythoCCallable extension, building it locally when needed."""
    global _extension_module
    if _extension_module is not None:
        return _extension_module
    with _extension_lock:
        if _extension_module is not None:
            return _extension_module
        from .utils.link_utils import file_lock

        output = _extension_output()
        with file_lock(output + '.build.lock', timeout=120):
            built = _build_callable_extension()
            extension = _load_runtime_extension(output, fresh=built)
            if built:
                _write_runtime_stamp()
        _extension_module = extension
        return extension


def _cached_extension():
    """Load the already-built runtime extension; never builds it.

    Returns None when the extension is missing, stale, or the layout
    probe inputs are unavailable (e.g. no C API headers).  Building is
    deferred to the first native call (``ensure_callable_extension``),
    so decorating a @compile function never needs a C toolchain.
    """
    global _extension_module
    if _extension_module is not None:
        return _extension_module
    with _extension_lock:
        if _extension_module is not None:
            return _extension_module
        try:
            output = _extension_output()
            stamp_path = output + '.stamp'
            if not (os.path.isfile(output) and os.path.isfile(stamp_path)):
                return None
            with open(stamp_path, 'r', encoding='utf-8') as handle:
                if handle.read().strip() != _runtime_stamp():
                    return None
        except OSError:
            return None
        _extension_module = _load_runtime_extension(output, fresh=False)
        return _extension_module


class _DeferredCallable:
    """Python facade for a compiled function whose runtime is not built yet.

    The native wrapper type lives in the callable runtime extension, which
    needs a C toolchain to build.  AOT-only flows never call their wrappers
    and must not pay that; the facade defers the build to the first call,
    then shares its attribute dict with the native object so state written
    before or after the first call is visible on both.
    """

    def __init__(self, func):
        self.__dict__['_deferred_func'] = func
        self.__dict__['_native_self'] = None
        _set_callable_metadata(self, func)

    def _bootstrap(self):
        native = self.__dict__.get('_native_self')
        if native is not None:
            return native
        func = self.__dict__.pop('_deferred_func')
        self.__dict__.pop('_native_self')
        native = _native_callable(ensure_callable_extension(), func)
        native.__dict__.update(self.__dict__)
        self.__dict__ = native.__dict__
        self._native_self = native
        return native

    def __call__(self, *args, **kwargs):
        return self._bootstrap()(*args, **kwargs)

    def __getattr__(self, name):
        native = self.__dict__.get('_native_self')
        if native is not None:
            return getattr(native, name)
        if name == 'is_fast_bound':
            return lambda: False
        raise AttributeError(name)


def create_compiled_callable(func):
    """Return a vectorcall callable that preserves compile-time metadata."""
    if _building_runtime:
        return _runtime_anchor(func)
    extension = _cached_extension()
    if extension is None:
        return _DeferredCallable(func)
    return _native_callable(extension, func)


def _set_callable_metadata(wrapper, func):
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = getattr(func, '__qualname__', func.__name__)
    wrapper.__module__ = getattr(func, '__module__', None)
    wrapper.__doc__ = func.__doc__
    annotations = getattr(func, '__annotations__', None)
    if isinstance(annotations, dict):
        wrapper.__annotations__ = dict(annotations)
    try:
        wrapper.__signature__ = inspect.signature(func)
    except (TypeError, ValueError):
        pass


def _native_callable(extension, func):
    wrapper = extension.PythoCCallable(func)
    resolve_done = threading.Event()
    _set_callable_metadata(wrapper, func)

    def _pythoc_resolve(bound=wrapper):
        bound._pythoc_resolve_owner = threading.get_ident()
        try:
            resolve_compiled_callable(bound)
        except BaseException as error:
            bound._pythoc_resolve_error = error
            raise
        finally:
            bound._pythoc_resolve_owner = None
            resolve_done.set()

    def _pythoc_wait(bound=wrapper):
        if bound._pythoc_resolve_owner == threading.get_ident():
            raise RuntimeError('re-entrant PythoC callable resolve')
        resolve_done.wait()
        error = getattr(bound, '_pythoc_resolve_error', None)
        if error is not None:
            raise error

    def _pythoc_raise_error(bound=wrapper):
        error = getattr(bound, '_pythoc_resolve_error', None)
        if error is not None:
            raise error
        raise RuntimeError('PythoC callable resolve failed')

    wrapper._pythoc_resolve_owner = None
    wrapper._pythoc_resolve = _pythoc_resolve
    wrapper._pythoc_wait = _pythoc_wait
    wrapper._pythoc_raise_error = _pythoc_raise_error
    return wrapper


def resolve_compiled_callable(wrapper) -> None:
    """Bind one Python callable on its first call."""
    binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
    if binding is not None and binding.is_template:
        from .decorators.compile import (
            DEFAULT_EFFECT_KEY,
            materialize_specialization,
        )
        materialize_specialization(wrapper, DEFAULT_EFFECT_KEY, {})

    if try_bind_installed_adapter(wrapper):
        return

    from .python_adapter import bind_development_callable
    bind_development_callable(wrapper)


def try_bind_installed_adapter(wrapper) -> bool:
    """Bind a wheel/package adapter when its manifest is already installed."""
    if wrapper.is_fast_bound():
        return True
    manifest_path = _find_manifest(wrapper)
    if manifest_path is None:
        return False
    if isinstance(wrapper, _DeferredCallable):
        # An installed manifest means this wrapper must bind at decoration
        # time, which requires the native runtime; build it now.
        wrapper = wrapper._bootstrap()
    manifest = _read_manifest(manifest_path)
    export_id = python_export_id(wrapper)
    export = manifest.get('exports', {}).get(export_id)
    if not export:
        return False
    module = _load_manifest_extension(manifest)
    install_module(module, [wrapper])
    address = module.adapter_address(export_id)
    wrapper.bind_adapter(address)
    wrapper._pythoc_extension = module
    return True


def python_export_id(wrapper) -> str:
    """Stable identity shared by a source function and its installed adapter."""
    binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
    source = os.path.basename(getattr(binding, 'source_file', '') or '')
    original = getattr(binding, 'original_name', None) or wrapper.__name__
    symbol = getattr(binding, 'actual_func_name', None) or original
    return '{}:{}:{}'.format(source, original, symbol)


_ENTRY_TYPES = {}


def install_library(library, wrappers):
    """Publish process-local type objects into one loaded entry."""
    import ctypes

    from .builtin_entities.pc_literal import pc_literal

    _collect_entry_types(wrappers)
    function = library.pythoc_install_runtime
    function.argtypes = [ctypes.py_object, ctypes.py_object, ctypes.py_object]
    function.restype = None
    function(
        pc_literal,
        ensure_callable_extension().PythoCCallable,
        _ENTRY_TYPES,
    )


def install_module(module, wrappers):
    """Publish process-local type objects through the extension method."""
    from .builtin_entities.pc_literal import pc_literal

    _collect_entry_types(wrappers)
    module.install_runtime((
        pc_literal,
        ensure_callable_extension().PythoCCallable,
        _ENTRY_TYPES,
    ))


def _collect_entry_types(wrappers):
    for wrapper in wrappers:
        info = getattr(wrapper, '_func_info', None)
        if info is None:
            continue
        _collect_entry_type(info.return_type_hint)
        for pc_type in info.param_type_hints.values():
            _collect_entry_type(pc_type)


def _collect_entry_type(pc_type):
    if pc_type is None or not hasattr(pc_type, 'get_name'):
        return
    _ENTRY_TYPES[pc_type.get_name()] = pc_type
    peeled = pc_type
    if hasattr(pc_type, '_resolve_qualified_type'):
        inner = pc_type._resolve_qualified_type()
        if inner is not None:
            peeled = inner
            _ENTRY_TYPES[peeled.get_name()] = peeled
    for field_type in getattr(peeled, '_field_types', None) or []:
        _collect_entry_type(field_type)
    element = getattr(peeled, 'element_type', None)
    if element is not None and element is not peeled:
        _collect_entry_type(element)


def _find_manifest(wrapper) -> Optional[str]:
    override = os.environ.get('PYTHOC_NATIVE_MANIFEST')
    if override:
        return override if os.path.isfile(override) else None
    binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
    source = getattr(binding, 'source_file', None) if binding is not None else None
    if not source:
        return None
    path = os.path.join(os.path.dirname(os.path.abspath(source)),
                        '_pythoc_manifest.json')
    return path if os.path.isfile(path) else None


def _read_manifest(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('schema') != 1:
        raise RuntimeError('unsupported PythoC native manifest: {}'.format(path))
    return manifest


def _load_manifest_extension(manifest):
    path = os.path.abspath(manifest['extension'])
    cached = _manifest_modules.get(path)
    if cached is not None:
        return cached
    module_name = manifest.get('module', '_pythoc_native')
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load PythoC extension {}'.format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _manifest_modules[path] = module
    _loaded_extensions.append(module)
    return module


def _runtime_sources():
    here = os.path.dirname(os.path.abspath(__file__))
    include = sysconfig.get_config_var('INCLUDEPY') or sysconfig.get_path('include')
    return [
        os.path.join(here, 'callable_type.py'),
        os.path.join(here, 'callable_layout.py'),
        os.path.join(include, 'Python.h'),
    ]


def _runtime_stamp() -> str:
    digest = hashlib.sha256()
    digest.update(sys.version.encode('utf-8'))
    digest.update(str(sysconfig.get_config_var('EXT_SUFFIX')).encode('utf-8'))
    for path in _runtime_sources():
        stat = os.stat(path)
        digest.update(path.encode('utf-8'))
        digest.update(str(stat.st_mtime_ns).encode('utf-8'))
        digest.update(str(stat.st_size).encode('utf-8'))
    return digest.hexdigest()


def _runtime_object(source_file: str) -> str:
    from .build.output_manager import get_output_manager

    real = os.path.realpath(source_file)
    for group in get_output_manager()._all_groups.values():
        recorded = group.get('source_file') or ''
        if recorded and os.path.realpath(recorded) == real:
            obj = group.get('obj_file')
            if obj and os.path.isfile(obj):
                return obj
    raise RuntimeError(
        'PythoC callable runtime did not produce an object file'
    )


def _extension_output() -> str:
    """Per-interpreter extension. Built on first use, never shipped."""
    suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
    directory = os.path.join('build', 'python_runtime')
    os.makedirs(directory, exist_ok=True)
    return os.path.abspath(os.path.join(directory, '_callable' + suffix))


def _load_runtime_extension(path: str, fresh: bool):
    name = 'pythoc._callable'
    if fresh:
        sys.modules.pop(name, None)
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load PythoC callable runtime {}'.format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_runtime_stamp() -> None:
    path = _extension_output() + '.stamp'
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(_runtime_stamp() + '\n')


def _build_callable_extension() -> bool:
    """Compile callable_type.py. Return True when a new extension was linked."""
    global _building_runtime
    if _building_runtime:
        raise RuntimeError('PythoC callable runtime build re-entered')
    output = _extension_output()
    stamp_path = output + '.stamp'
    stamp = _runtime_stamp()
    if os.path.isfile(output) and os.path.isfile(stamp_path):
        with open(stamp_path, 'r', encoding='utf-8') as handle:
            if handle.read().strip() == stamp:
                return False

    _building_runtime = True
    try:
        from . import callable_type
        from .build.output_manager import (
            flush_all_pending_outputs,
            get_output_manager,
        )
        from .artifact import ArtifactRole
        from .logger import logger

        manager = get_output_manager()
        runtime_source = os.path.realpath(callable_type.__file__)
        for group_key, group in manager.get_all_groups().items():
            if os.path.realpath(group.get('source_file') or '') == runtime_source:
                manager.set_group_artifact_role(
                    group_key,
                    ArtifactRole.PYTHON_RUNTIME,
                )
        logger.debug('callable runtime layout: {}'.format(callable_type.LAYOUT))
        flush_all_pending_outputs()
        obj = _runtime_object(callable_type.__file__)
        if os.path.isfile(output):
            os.remove(output)
        from .python_adapter import _python_libraries
        from .utils.link_utils import link_files

        link_files(
            [obj],
            output,
            shared=True,
            link_objects=[],
            link_libraries=[],
            # Posix resolves the CPython symbols from the host process at
            # load time; Windows needs the pythonXY import library.
            extra_flags=_python_libraries(),
        )
    finally:
        _building_runtime = False
    return True

