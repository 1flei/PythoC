"""Bind CPython vectorcall adapters for C-ABI PythoC signatures.

The adapter itself is compiled by PythoC. This module decides whether a
signature can be lowered, then asks python_entry_bind for the entry object.
Development calls keep the executor image. The package extension target
links entries and kernel objects together. Neither target changes native
AOT output.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import sysconfig
from typing import Any, Dict, List, Optional, Sequence

from .python_call import python_export_id

_INT_WIDTHS = {
    'i8': ('signed', 8),
    'i16': ('signed', 16),
    'i32': ('signed', 32),
    'i64': ('signed', 64),
    'u8': ('unsigned', 8),
    'u16': ('unsigned', 16),
    'u32': ('unsigned', 32),
    'u64': ('unsigned', 64),
}
_UNSUPPORTED = object()
_KEEP_ALIVE = []


def bind_development_callable(wrapper) -> None:
    """Bind a first-call adapter without changing the kernel image."""
    spec = function_spec(wrapper)
    if spec is None:
        wrapper.bind_slow(_slow_implementation(wrapper))
        return

    from .native_executor import get_multi_so_executor

    executor = get_multi_so_executor()
    native = executor.execute_function(wrapper)
    symbol = spec['symbol']
    kernel = getattr(native, '_pythoc_kernel_lib', None)
    if kernel is None:
        kernel = executor.loaded_libs.get(wrapper._binding.so_file)
    if kernel is None:
        raise RuntimeError(
            'PythoC kernel was not loaded for {}'.format(symbol)
        )
    _publish_group_kernels(wrapper, kernel)
    entries = _development_group_entries(wrapper, spec)
    specs = [entry_spec for _entry, entry_spec in entries]
    adapter_path = _development_adapter_path(wrapper, specs)
    library = _load_adapter_library(adapter_path)
    from .python_call import install_library
    install_library(library, [wrapper])
    wrapper.bind_adapter(_address_of(getattr(library, spec['adapter'])))
    wrapper._pythoc_adapter_lib = library
    wrapper._pythoc_kernel_lib = kernel


def compile_python_extension(
    symbols: Sequence[Any],
    output_path: str,
    module_name: str,
) -> str:
    """Link selected kernels and Python adapters into one extension."""
    specs = []
    skipped = []
    for symbol in symbols:
        spec = function_spec(symbol)
        if spec is None:
            skipped.append(getattr(symbol, '__name__', repr(symbol)))
            continue
        specs.append(spec)
    if not specs:
        raise RuntimeError(
            'no C-ABI @compile symbols can be exported to Python'
        )

    from .artifact import (
        ArtifactKind,
        ArtifactPhase,
        ArtifactPlan,
        ArtifactStep,
        ExportSpec,
        LinkPlan,
        build_artifact,
    )

    link_plan = LinkPlan.from_compiled_symbols(symbols)
    output_path = _extension_output_path(output_path)
    init_symbol = _init_symbol(module_name)
    adapter_key = _adapter_object_path(
        output_path,
        specs,
        init_symbol=init_symbol,
        module_name=module_name.split('.')[-1],
    )
    adapter_group = _adapter_group_key(adapter_key)
    adapter_object = _adapter_group_object(adapter_group)
    version_script = (
        output_path + '.map' if sys.platform == 'linux' else None
    )
    extra = list(_python_libraries())
    if version_script is not None:
        extra.append('-Wl,--version-script={}'.format(version_script))
    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(output_path)),
        '_pythoc_manifest.json',
    )
    manifest = {
        'schema': 1,
        'module': module_name,
        'extension': os.path.abspath(output_path),
        'exports': {
            spec['export_id']: {'adapter': spec['adapter'], 'symbol': spec['symbol']}
            for spec in specs
        },
        'skipped': skipped,
    }
    steps = [
        ArtifactStep(
            id='compile-python-entry:{}'.format(
                os.path.abspath(adapter_object)
            ),
            kind='compile_generated_entry',
            phase=ArtifactPhase.PRE_LINK,
            outputs=(adapter_object,),
            run=lambda: _write_and_compile(
                specs,
                adapter_group,
                init_symbol=init_symbol,
                module_name=module_name.split('.')[-1],
            ),
            cache_check=lambda: _adapter_group_cache_hit(
                adapter_group,
                specs,
            ),
        ),
    ]
    if version_script is not None:
        steps.append(ArtifactStep(
            id='write-version-script:{}'.format(
                os.path.abspath(version_script)
            ),
            kind='write_version_script',
            phase=ArtifactPhase.PRE_LINK,
            outputs=(version_script,),
            run=lambda: _write_version_script(output_path, init_symbol),
        ))
    steps.append(ArtifactStep(
        id='publish-manifest:{}'.format(os.path.abspath(manifest_path)),
        kind='publish_artifact_manifest',
        phase=ArtifactPhase.POST_LINK,
        outputs=(manifest_path,),
        run=lambda: _write_manifest(manifest_path, manifest),
    ))
    exports = tuple(
        ExportSpec(
            export_id=spec['export_id'],
            native_symbol=spec['symbol'],
            adapter_symbol=spec['adapter'],
            python_name=spec['python_name'],
        )
        for spec in specs
    )
    plan = ArtifactPlan(
        kind=ArtifactKind.PYTHON_EXTENSION,
        link=link_plan,
        output_path=output_path,
        exports=exports,
        prefix_objects=(adapter_object, _entry_object_path()),
        extra_link_flags=tuple(extra),
        steps=tuple(steps),
        metadata={
            'module': module_name,
            'manifest': manifest,
        },
    )
    artifact = build_artifact(plan)
    print('Successfully compiled Python extension: {}'.format(output_path))
    print('Wrote native manifest: {}'.format(manifest_path))
    return artifact.path


def _write_manifest(path: str, manifest: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return path


def function_spec(wrapper) -> Optional[dict]:
    """Return a C-ABI export spec, or None when the signature cannot be lowered."""
    info = getattr(wrapper, '_func_info', None)
    binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
    if info is None or binding is None:
        return None
    if info.has_varargs or info.has_kwargs or info.has_llvm_varargs:
        return None
    if info.linkage in ('internal', 'linkonce_odr'):
        return None
    abi = _Abi()
    names = list(info.param_names)
    kinds = []
    from .schema_protocol import get_linear_schema_paths

    for name in names:
        param_type = info.param_type_hints.get(name)
        peeled_param = _peel_type(param_type)
        direct_linear = (
            getattr(peeled_param, '_is_linear', False)
            and not getattr(param_type, '_is_refined', False)
            and peeled_param is param_type
        )
        if (
            get_linear_schema_paths(param_type)
            and not direct_linear
        ):
            return None
        kind = _classify_type(param_type, abi, set())
        if kind is None:
            return None
        if kind != ('skip',):
            kind = _as_abi_value(kind, abi)
        kinds.append(kind)
    returns = _classify_return(info.return_type_hint, abi)
    if returns is None:
        return None
    if get_linear_schema_paths(info.return_type_hint):
        return None
    return_name = (
        info.return_type_hint.get_name()
        if hasattr(info.return_type_hint, 'get_name')
        else ''
    )
    if returns == ('void',) and return_name != 'void':
        return None
    defaults = _matching_defaults(wrapper, names)
    if defaults is _UNSUPPORTED:
        return None
    if not _defaults_fit(kinds, defaults):
        return None
    symbol = binding.actual_func_name or binding.original_name or info.name
    if not symbol:
        return None
    return {
        'export_id': python_export_id(wrapper),
        'python_name': binding.original_name or info.name,
        'symbol': symbol,
        'adapter': _adapter_symbol(symbol),
        'names': names,
        'defaults': defaults,
        'param_abi': kinds,
        'return_abi': returns,
        'abi_types': abi.types,
        'param_type_identity': [
            _boundary_type_identity(info.param_type_hints.get(name))
            for name in names
        ],
        'return_type_identity': _boundary_type_identity(
            info.return_type_hint
        ),
        'param_types': [info.param_type_hints.get(name) for name in names],
        'return_type': info.return_type_hint,
        'callee': wrapper,
    }


def _matching_defaults(wrapper, names: List[str]):
    wrapped = getattr(wrapper, '__wrapped__', None)
    target = wrapped if wrapped is not None else wrapper
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return [None] * len(names)
    parameters = list(signature.parameters.values())
    if [item.name for item in parameters] != names:
        return [None] * len(names)
    defaults = []
    allowed = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    for parameter in parameters:
        if parameter.kind not in allowed:
            return _UNSUPPORTED
        if parameter.default is inspect.Parameter.empty:
            defaults.append(None)
            continue
        literal = _literal_default(parameter.default)
        if literal is _UNSUPPORTED:
            return _UNSUPPORTED
        defaults.append(literal)
    return defaults


def _boundary_type_identity(pc_type, seen=None):
    if pc_type is None:
        return None
    seen = set() if seen is None else seen
    peeled = _peel_type(pc_type)
    marker = id(peeled)
    name = (
        pc_type.get_name()
        if hasattr(pc_type, 'get_name')
        else repr(pc_type)
    )
    base_name = (
        peeled.get_name()
        if hasattr(peeled, 'get_name')
        else repr(peeled)
    )
    identity = {
        'name': name,
        'base': base_name,
        'size': _size_of(peeled),
        'align': _align_of(peeled),
    }
    if marker in seen:
        identity['recursive'] = True
        return identity
    seen.add(marker)
    try:
        fields = getattr(peeled, '_field_types', None)
        if fields is not None:
            identity['fields'] = [
                _boundary_type_identity(field, seen)
                for field in fields
            ]
            identity['field_names'] = list(
                getattr(peeled, '_field_names', None) or []
            )
        element = getattr(peeled, 'element_type', None)
        if element is not None:
            identity['element'] = _boundary_type_identity(element, seen)
            dims = getattr(peeled, 'dimensions', None)
            if dims is not None:
                identity['dimensions'] = (
                    list(dims) if isinstance(dims, tuple) else dims
                )
        variants = list(getattr(peeled, '_variant_names', None) or [])
        if variants:
            variant_types = list(
                getattr(peeled, '_variant_types', None) or []
            )
            tags = getattr(peeled, '_tag_values', None)
            identity['variants'] = [
                {
                    'name': variant,
                    'tag': (
                        int(tags[variant])
                        if isinstance(tags, dict)
                        else int(tags[index])
                    ),
                    'payload': _boundary_type_identity(
                        (
                            variant_types[index]
                            if index < len(variant_types)
                            else None
                        ),
                        seen,
                    ),
                }
                for index, variant in enumerate(variants)
            ]
        return identity
    finally:
        seen.remove(marker)


def _literal_default(value):
    if value is None:
        return ('none',)
    if isinstance(value, bool):
        return ('bool', value)
    if isinstance(value, int):
        return ('int', value)
    if isinstance(value, float):
        return ('float', value)
    return _UNSUPPORTED


def _defaults_fit(kinds, defaults) -> bool:
    for kind, default in zip(kinds, defaults):
        if default is None or kind == ('skip',) or default[0] == 'none':
            continue
        tag = default[0]
        if kind[0] in ('f32', 'f64'):
            if tag not in ('float', 'int', 'bool'):
                return False
        elif kind[0] in ('bool', 'int', 'ptr'):
            if tag not in ('int', 'bool'):
                return False
        else:
            return False
    return True


class _Abi:
    """Typedefs accumulated while classifying one signature."""

    def __init__(self):
        self.types = []

    def add(self, form, **payload):
        item = {'form': form, 'id': len(self.types)}
        item.update(payload)
        self.types.append(item)
        return ('agg', item['id'])


def _classify_return(pc_type, abi):
    if pc_type is None:
        return None
    name = pc_type.get_name() if hasattr(pc_type, 'get_name') else ''
    if name == 'void':
        return ('void',)
    kind = _classify_type(pc_type, abi, set())
    if kind is None:
        return None
    if kind == ('skip',):
        return ('void',)
    return _as_abi_value(kind, abi)


def _as_abi_value(kind, abi):
    """Pass a bare array as a one-field struct so C does not decay it."""
    if kind[0] != 'agg':
        return kind
    item = abi.types[kind[1]]
    if item['form'] != 'array':
        return kind
    return abi.add('struct', fields=[('v', 'v', kind)], transparent=True)


def _classify_type(pc_type, abi, stack):
    pc_type = _peel_type(pc_type)
    if pc_type is None or not hasattr(pc_type, 'get_name'):
        return None
    if pc_type.get_name() == 'void' or getattr(pc_type, '_is_linear', False):
        return ('skip',)
    if getattr(pc_type, '_is_bool', False):
        return ('bool',)
    if getattr(pc_type, '_is_float', False):
        name = pc_type.get_name()
        if name in ('f32', 'f64'):
            return (name,)
        return None
    if getattr(pc_type, '_is_integer', False):
        spec = _INT_WIDTHS.get(pc_type.get_name())
        if spec is not None:
            return ('int', spec[0], spec[1])
        if pc_type.get_name() in ('i128', 'u128'):
            signed = 'unsigned' if pc_type.get_name().startswith('u') else 'signed'
            return ('int', signed, 128)
        return None
    if getattr(pc_type, '_is_pointer', False):
        return ('ptr',)
    marker = id(pc_type)
    if marker in stack:
        return None
    stack.add(marker)
    try:
        return _classify_aggregate(pc_type, abi, stack)
    finally:
        stack.discard(marker)


def _classify_aggregate(pc_type, abi, stack):
    if _type_flag(pc_type, 'is_enum_type') or getattr(pc_type, '_is_enum', False):
        if _size_of(pc_type) == 0:
            return ('skip',)
        return _classify_enum(pc_type, abi, stack)
    if _type_flag(pc_type, 'is_struct_type') or getattr(pc_type, '_is_struct', False):
        return _classify_struct(pc_type, abi, stack)
    if getattr(pc_type, '_is_union', False) or pc_type.get_name() == 'union':
        return _classify_blob(pc_type, abi)
    if _type_flag(pc_type, 'is_array') or pc_type.get_name() == 'array':
        return _classify_array(pc_type, abi, stack)
    return None


def _type_flag(pc_type, name: str) -> bool:
    checker = getattr(pc_type, name, None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except TypeError:
        return False


def _classify_struct(pc_type, abi, stack):
    field_types = getattr(pc_type, '_field_types', None)
    if field_types is None:
        return None
    field_names = getattr(pc_type, '_field_names', None) or []
    fields = []
    used = set()
    for index, field_type in enumerate(field_types):
        field_type = _resolve_named_type(field_type)
        kind = _classify_type(field_type, abi, stack)
        if kind is None:
            return None
        if kind == ('skip',):
            continue
        raw_name = field_names[index] if index < len(field_names) else None
        c_name = _field_ident(raw_name, index, used)
        fields.append((raw_name, c_name, kind))
    if not fields:
        return ('skip',)
    return abi.add('struct', fields=fields, transparent=False)


def _classify_enum(pc_type, abi, stack):
    tag = _classify_type(getattr(pc_type, '_tag_type', None), abi, stack)
    payload = getattr(pc_type, '_union_payload', None)
    if tag is None or tag == ('skip',) or payload is None:
        return None
    payload_size = _size_of(payload)
    fields = [('tag', 'tag', tag)]
    if payload_size:
        align = _align_of(payload)
        fields.append(('payload', 'payload', abi.add(
            'blob', size=payload_size, align=align,
        )))
    return abi.add('struct', fields=fields, transparent=False)


def _classify_array(pc_type, abi, stack):
    element = _resolve_named_type(getattr(pc_type, 'element_type', None))
    dims = getattr(pc_type, 'dimensions', None)
    if element is None or not dims:
        return None
    try:
        dims = tuple(int(dim) for dim in dims)
    except (TypeError, ValueError):
        return None
    if any(dim < 0 for dim in dims):
        return None
    if any(dim == 0 for dim in dims):
        return ('skip',)
    elem_kind = _classify_type(element, abi, stack)
    if elem_kind is None or elem_kind == ('skip',):
        return None
    return abi.add('array', elem=elem_kind, dims=dims)


def _classify_blob(pc_type, abi):
    size = _size_of(pc_type)
    if size <= 0:
        return ('skip',)
    return abi.add('blob', size=size, align=_align_of(pc_type))


def _peel_type(pc_type):
    seen = set()
    while pc_type is not None and id(pc_type) not in seen:
        seen.add(id(pc_type))
        if hasattr(pc_type, '_resolve_qualified_type'):
            inner = pc_type._resolve_qualified_type()
            if inner is not None and inner is not pc_type:
                pc_type = inner
                continue
        if getattr(pc_type, '_is_refined', False):
            inner = getattr(pc_type, '_base_type', None)
            if inner is None:
                inner = getattr(pc_type, '_struct_type', None)
            if inner is not None and inner is not pc_type:
                pc_type = inner
                continue
        break
    return _resolve_named_type(pc_type)


def _resolve_named_type(pc_type):
    if not isinstance(pc_type, str):
        return pc_type
    from .forward_ref import get_defined_type
    return get_defined_type(pc_type)


def _size_of(pc_type):
    if pc_type is None or not hasattr(pc_type, 'get_size_bytes'):
        return None
    return pc_type.get_size_bytes()


def _align_of(pc_type):
    if hasattr(pc_type, 'get_alignment'):
        align = pc_type.get_alignment()
        if align:
            return align
    size = _size_of(pc_type) or 1
    return min(size, 8) or 1


def _field_ident(name, index, used):
    ident = name if isinstance(name, str) and _is_identifier(name) else ''
    if not ident or ident in _C_RESERVED:
        ident = 'f{}'.format(index)
    while ident in used:
        ident = '{}_{}'.format(ident, index)
    used.add(ident)
    return ident


_C_RESERVED = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
    'inline', 'int', 'long', 'register', 'restrict', 'return', 'short',
    'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
    'unsigned', 'void', 'volatile', 'while', '_Bool', 'bool', 'bytes',
}


def _adapter_symbol(symbol: str) -> str:
    if _is_identifier(symbol):
        return 'pythoc_pyadapter_' + symbol
    digest = hashlib.sha256(symbol.encode('utf-8')).hexdigest()[:16]
    return 'pythoc_pyadapter_' + digest


def _is_identifier(text: str) -> bool:
    if not text or text[0].isdigit():
        return False
    return all(ch == '_' or ch.isalnum() for ch in text) and all(ord(ch) < 128 for ch in text)


_ADAPTER_SO = {}
_ADAPTER_LIB = {}


def _development_group_entries(wrapper, current_spec):
    from .build.output_manager import get_output_manager

    binding = wrapper._binding
    wrappers = get_output_manager().get_group_wrappers(binding.group_key)
    if wrapper not in wrappers:
        wrappers.append(wrapper)
    entries = []
    for entry in wrappers:
        spec = current_spec if entry is wrapper else function_spec(entry)
        if spec is not None:
            entries.append((entry, spec))
    entries.sort(key=lambda item: item[1]['export_id'])
    unique = {}
    for entry, spec in entries:
        unique.setdefault(spec['adapter'], (entry, spec))
    return list(unique.values())


def _kernel_link_input(lib: str) -> str:
    """COFF links resolve symbols through the import library, not the DLL."""
    if sys.platform == 'win32':
        implib = os.path.splitext(lib)[0] + '.lib'
        if os.path.exists(implib):
            return implib
    return lib


def _development_adapter_path(wrapper, specs: List[dict]) -> str:
    so_file = os.path.abspath(wrapper._binding.so_file)
    adapters = tuple(spec['adapter'] for spec in specs)
    key = (so_file, adapters)
    cached = _ADAPTER_SO.get(key)
    if cached and os.path.exists(cached):
        return cached
    root, _extension = os.path.splitext(so_file)
    identity = _adapter_identity(specs, None, None)
    base = '{}.pythoc_pyadapter_group.{}'.format(root, identity[:16])
    output = base + _shared_suffix()
    kernel_libs = []
    for spec in specs:
        callee = spec.get('callee')
        binding = getattr(callee, '_binding', getattr(callee, '_state', None))
        lib = getattr(binding, 'so_file', None) if binding is not None else None
        if lib:
            lib = _kernel_link_input(os.path.abspath(lib))
            if lib not in kernel_libs:
                kernel_libs.append(lib)
    _compile_adapter_library(
        specs,
        output,
        # Link against the kernel images so the dynamic loader binds
        # each kernel symbol to that specific image.  A flat namespace
        # lookup would resolve a bare name like 'store_i32' to any
        # already-loaded image that happens to define it.
        link_objects=kernel_libs,
        link_libraries=_python_libraries(),
    )
    _ADAPTER_SO[key] = output
    return output


def _load_adapter_library(path):
    absolute = os.path.abspath(path)
    library = _ADAPTER_LIB.get(absolute)
    if library is not None:
        return library
    library = _load_library(absolute)
    _ADAPTER_LIB[absolute] = library
    _KEEP_ALIVE.append(library)
    return library


def _compile_adapter_library(
    specs: List[dict],
    output_path: str,
    link_objects,
    link_libraries,
) -> None:
    from .artifact import (
        ArtifactKind,
        ArtifactPhase,
        ArtifactPlan,
        ArtifactStep,
        LinkPlan,
        build_artifact,
    )
    objects = []
    for spec in specs:
        entry_specs = [spec]
        adapter_key = _adapter_object_path(output_path, entry_specs)
        adapter_group = _adapter_group_key(adapter_key)
        object_path = _adapter_group_object(adapter_group)
        objects.append(object_path)
        # Registration is cheap Python work; the object is materialized by
        # the single flush step below.  A per-entry flush here would run
        # the global flush from concurrent build steps and can sweep a
        # group mid-registration, leaving its object unwritten.
        _write_and_compile(
            entry_specs,
            adapter_group,
            init_symbol=None,
            module_name=None,
            flush=False,
        )

    def _flush_entries():
        from .build.output_manager import flush_all_pending_outputs
        flush_all_pending_outputs()

    steps = [
        ArtifactStep(
            id='flush-python-entries:{}'.format(os.path.abspath(output_path)),
            kind='flush_pending_outputs',
            phase=ArtifactPhase.PRE_LINK,
            outputs=tuple(objects),
            run=_flush_entries,
        ),
    ]
    plan = ArtifactPlan(
        kind=ArtifactKind.SHARED_LIBRARY,
        link=LinkPlan(roots=(), obj_files=()),
        output_path=output_path,
        prefix_objects=tuple(objects) + (_entry_object_path(),),
        link_objects=tuple(link_objects or ()),
        extra_link_flags=tuple(link_libraries or ()),
        steps=tuple(steps),
    )
    build_artifact(plan)


def _write_and_compile(specs, group_key, init_symbol, module_name,
                       flush=True) -> str:
    from .python_entry_bind import compile_adapter_object
    return compile_adapter_object(
        specs,
        group_key,
        init_symbol=init_symbol,
        module_name=module_name,
        flush=flush,
    )



def _adapter_identity(specs, init_symbol, module_name) -> str:
    payload = {
        'runtime': {
            'cache_tag': getattr(sys.implementation, 'cache_tag', None),
            'soabi': sysconfig.get_config_var('SOABI'),
        },
        'init_symbol': init_symbol,
        'module_name': module_name,
        'entries': [
            {
                'export_id': spec['export_id'],
                'symbol': spec['symbol'],
                'adapter': spec['adapter'],
                'names': spec['names'],
                'defaults': spec['defaults'],
                'param_abi': spec['param_abi'],
                'return_abi': spec['return_abi'],
                'abi_types': spec['abi_types'],
                'param_type_identity': spec['param_type_identity'],
                'return_type_identity': spec['return_type_identity'],
            }
            for spec in specs
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _adapter_object_path(
    output_path: str,
    specs,
    init_symbol=None,
    module_name=None,
) -> str:
    root, _ext = os.path.splitext(output_path)
    identity = _adapter_identity(specs, init_symbol, module_name)
    return '{}.{}.adapter.o'.format(root, identity[:20])


def _adapter_group_key(adapter_key):
    source_file = os.path.realpath(
        os.path.join(os.path.dirname(__file__), 'python_entry_bind.py')
    )
    cache_key = os.path.abspath(adapter_key).encode('utf-8')
    digest = hashlib.sha256(cache_key).hexdigest()[:20]
    return (
        source_file,
        'meta',
        'pyadapter2_{}'.format(digest),
        None,
    )


def _adapter_group_object(group_key):
    from .build.deps import get_dependency_tracker

    return get_dependency_tracker().derive_obj_file_from_group_key(group_key)


def _entry_object_path():
    source_file = os.path.realpath(
        os.path.join(os.path.dirname(__file__), 'python_entry.py')
    )
    return _adapter_group_object((source_file, None, None, None))


def _adapter_group_cache_hit(group_key, specs):
    from .build.output_manager import get_output_manager

    return get_output_manager().group_object_cache_hit(
        group_key,
        expected_symbols=[spec['adapter'] for spec in specs],
    )


def _extension_output_path(output_path: str) -> str:
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if output_path.endswith(suffix):
        return output_path
    return output_path + suffix


def _shared_suffix() -> str:
    from .utils.link_utils import get_shared_lib_extension
    return get_shared_lib_extension()


def _init_symbol(module_name: str) -> str:
    leaf = module_name.split('.')[-1]
    if not _is_identifier(leaf):
        raise ValueError('Python extension module name is not a C identifier: {}'.format(leaf))
    return 'PyInit_' + leaf


def _write_version_script(output_path: str, init_symbol: str) -> Optional[str]:
    if sys.platform != 'linux':
        return None
    path = output_path + '.map'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join([
            '{',
            '  global:',
            '    {};'.format(init_symbol),
            '    pythoc_pyadapter_*;',
            '  local:',
            '    *;',
            '};',
            '',
        ]))
    return os.path.abspath(path)


def _python_libraries() -> List[str]:
    """Raw linker flags. Passed through extra_flags verbatim.

    The interpreter's symbols resolve from the host process at load time
    (``-undefined dynamic_lookup`` on macOS, the process-global symbol
    table on Linux), so libpython is only linked on Windows where
    undefined symbols are not allowed.  Linking it anywhere else can load
    a second, uninitialized copy of the runtime into the process when
    the interpreter itself is statically linked.

    On Windows, ``LIBDIR`` is typically None; the import library lives in
    ``<prefix>\\libs\\python<py_version_nodot>.lib``.
    """
    if sys.platform != 'win32':
        return []
    libdir = (
        sysconfig.get_config_var('LIBDIR')
        or os.path.join(sys.base_prefix, 'libs')
    )
    version = (
        sysconfig.get_config_var('py_version_nodot')
        or sysconfig.get_config_var('LDVERSION')
        or sysconfig.get_config_var('VERSION')
    )
    return ['-L{}'.format(libdir), '-lpython{}'.format(version)]



def _load_library(path: str):
    import ctypes
    mode = getattr(os, 'RTLD_NOW', 0) | getattr(os, 'RTLD_LOCAL', 0)
    if mode:
        return ctypes.CDLL(path, mode=mode)
    return ctypes.CDLL(path)


def _address_of(func) -> int:
    import ctypes
    return ctypes.cast(func, ctypes.c_void_p).value


def ctypes_function_address(library, name: str) -> int:
    func = getattr(library, name)
    return _address_of(func)


def _publish_group_kernels(wrapper, kernel_lib) -> None:
    """Stamp each sibling kernel address so function-pointer args stay native."""
    from .build.output_manager import get_output_manager

    binding = getattr(wrapper, '_binding', None)
    items = []
    if binding is not None:
        items.extend(get_output_manager().get_group_wrappers(binding.group_key))
    if wrapper not in items:
        items.append(wrapper)
    for item in items:
        state = getattr(item, '_binding', getattr(item, '_state', None))
        if state is None:
            continue
        symbol = state.actual_func_name or state.original_name
        if not symbol:
            continue
        try:
            item._pythoc_kernel = ctypes_function_address(kernel_lib, symbol)
        except AttributeError:
            continue


def _slow_implementation(wrapper):
    def slow(*args, **kwargs):
        from .call_normalization import pack_native_call_args
        from .native_executor import get_multi_so_executor

        native = get_multi_so_executor().execute_function(wrapper)
        return native(*pack_native_call_args(wrapper, args, kwargs))

    return slow
