"""Read the running interpreter's C API layout.

The callable runtime is compiled once per interpreter. Field offsets and
slot numbers come from that interpreter's headers, so one PythoC source
covers each CPython ABI and other runtimes that expose the same headers.
"""

from __future__ import annotations

import os
import sys
import sysconfig
import tempfile


_SNIPPET = r"""
#include <Python.h>

#ifdef Py_TPFLAGS_HAVE_VECTORCALL
typedef vectorcallfunc pc_vc_fn;
#else
typedef void *pc_vc_fn;
#endif

struct PcCallableProbe {
    PyObject_HEAD
    pc_vc_fn vectorcall;
    PyObject *dict;
    PyObject *wrapped;
    PyObject *slow;
    PyThread_type_lock lock;
    int state;
};

enum {
    pc_slot_dealloc = Py_tp_dealloc,
    pc_slot_traverse = Py_tp_traverse,
    pc_slot_clear = Py_tp_clear,
    pc_slot_repr = Py_tp_repr,
    pc_slot_call = Py_tp_call,
    pc_slot_methods = Py_tp_methods,
    pc_slot_getset = Py_tp_getset,
    pc_slot_init = Py_tp_init,
    pc_slot_new = Py_tp_new,
    pc_slot_doc = Py_tp_doc,
    pc_meth_o = METH_O,
    pc_meth_noargs = METH_NOARGS,
    pc_flag_default = Py_TPFLAGS_DEFAULT,
    pc_flag_gc = Py_TPFLAGS_HAVE_GC,
    pc_flag_vectorcall =
#ifdef Py_TPFLAGS_HAVE_VECTORCALL
        Py_TPFLAGS_HAVE_VECTORCALL
#else
        0
#endif
    ,
    pc_have_vectorcall =
#ifdef Py_TPFLAGS_HAVE_VECTORCALL
        1
#else
        0
#endif
    ,
    pc_api_version = PYTHON_API_VERSION,
    pc_size_pointer = sizeof(void *),
    pc_size_ssize = sizeof(Py_ssize_t),
    pc_size_int = sizeof(int)
};
"""


def python_abi_layout() -> dict:
    """Return sizes and offsets for the interpreter that is running now.

    Primary reader is libclang (no code execution, cross-friendly).  When
    libclang is unavailable -- an optional dependency -- fall back to a
    tiny C probe compiled and run with the same toolchain that links the
    runtime anyway.
    """
    include_dirs = _include_dirs()
    errors = []
    for reader in (_layout_via_libclang, _layout_via_cc_probe):
        try:
            return reader(include_dirs)
        except Exception as error:
            errors.append('{}: {}'.format(reader.__name__, error))
    raise RuntimeError(
        'cannot read the C API layout of {} {}:\n{}'.format(
            sys.implementation.name,
            sys.version.split()[0],
            '\n'.join(errors),
        )
    )


def _layout_via_libclang(include_dirs) -> dict:
    source = _write_snippet()
    try:
        tu = _parse(source, include_dirs)
        enums = _enum_values(tu.cursor)
        probe = _record(tu.cursor, 'PcCallableProbe')
        typeobj = _record(tu.cursor, '_typeobject')
        module = _record(tu.cursor, 'PyModuleDef')
        slot = _record(tu.cursor, 'PyType_Slot')
        spec = _record(tu.cursor, 'PyType_Spec')
        method = _record(tu.cursor, 'PyMethodDef')
        getset = _record(tu.cursor, 'PyGetSetDef')
    finally:
        os.remove(source)

    flags = _field(typeobj, 'tp_flags')
    layout = {
        'pointer': _need(enums, 'pc_size_pointer'),
        'ssize': _need(enums, 'pc_size_ssize'),
        'int': _need(enums, 'pc_size_int'),
        'have_vectorcall': _need(enums, 'pc_have_vectorcall'),
        'flag_default': _need(enums, 'pc_flag_default'),
        'flag_gc': _need(enums, 'pc_flag_gc'),
        'flag_vectorcall': _need(enums, 'pc_flag_vectorcall'),
        'api_version': _need(enums, 'pc_api_version'),
        'slot_dealloc': _need(enums, 'pc_slot_dealloc'),
        'slot_traverse': _need(enums, 'pc_slot_traverse'),
        'slot_clear': _need(enums, 'pc_slot_clear'),
        'slot_repr': _need(enums, 'pc_slot_repr'),
        'slot_call': _need(enums, 'pc_slot_call'),
        'slot_methods': _need(enums, 'pc_slot_methods'),
        'slot_getset': _need(enums, 'pc_slot_getset'),
        'slot_init': _need(enums, 'pc_slot_init'),
        'slot_new': _need(enums, 'pc_slot_new'),
        'slot_doc': _need(enums, 'pc_slot_doc'),
        'meth_o': _need(enums, 'pc_meth_o'),
        'meth_noargs': _need(enums, 'pc_meth_noargs'),
        'basicsize': probe['size'],
        'vectorcall': _field(probe, 'vectorcall')['offset'],
        'dict': _field(probe, 'dict')['offset'],
        'wrapped': _field(probe, 'wrapped')['offset'],
        'slow': _field(probe, 'slow')['offset'],
        'lock': _field(probe, 'lock')['offset'],
        'state': _field(probe, 'state')['offset'],
        'tp_dictoffset': _field(typeobj, 'tp_dictoffset')['offset'],
        'tp_flags': flags['offset'],
        'tp_flags_size': flags['size'],
        'module_size': module['size'],
        'm_name': _field(module, 'm_name')['offset'],
        'm_size': _field(module, 'm_size')['offset'],
        'type_slot_size': slot['size'],
        'type_spec_size': spec['size'],
        'method_size': method['size'],
        'getset_size': getset['size'],
    }
    if layout['have_vectorcall']:
        layout['tp_vectorcall_offset'] = _field(
            typeobj, 'tp_vectorcall_offset')['offset']
    _check(layout)
    return layout


_PROBE_MAIN = r"""
#include <stdio.h>
#include <stddef.h>

#define P(name, value) printf("%s %llu\n", name, (unsigned long long)(value))

int main(void) {
    P("pointer", sizeof(void *));
    P("ssize", sizeof(Py_ssize_t));
    P("int", sizeof(int));
    P("have_vectorcall", pc_have_vectorcall);
    P("flag_default", pc_flag_default);
    P("flag_gc", pc_flag_gc);
    P("flag_vectorcall", pc_flag_vectorcall);
    P("api_version", pc_api_version);
    P("slot_dealloc", pc_slot_dealloc);
    P("slot_traverse", pc_slot_traverse);
    P("slot_clear", pc_slot_clear);
    P("slot_repr", pc_slot_repr);
    P("slot_call", pc_slot_call);
    P("slot_methods", pc_slot_methods);
    P("slot_getset", pc_slot_getset);
    P("slot_init", pc_slot_init);
    P("slot_new", pc_slot_new);
    P("slot_doc", pc_slot_doc);
    P("meth_o", pc_meth_o);
    P("meth_noargs", pc_meth_noargs);
    P("basicsize", sizeof(struct PcCallableProbe));
    P("vectorcall", offsetof(struct PcCallableProbe, vectorcall));
    P("dict", offsetof(struct PcCallableProbe, dict));
    P("wrapped", offsetof(struct PcCallableProbe, wrapped));
    P("slow", offsetof(struct PcCallableProbe, slow));
    P("lock", offsetof(struct PcCallableProbe, lock));
    P("state", offsetof(struct PcCallableProbe, state));
    P("tp_dictoffset", offsetof(PyTypeObject, tp_dictoffset));
    P("tp_flags", offsetof(PyTypeObject, tp_flags));
    P("tp_flags_size", sizeof(((PyTypeObject *)0)->tp_flags));
    P("module_size", sizeof(PyModuleDef));
    P("m_name", offsetof(PyModuleDef, m_name));
    P("m_size", offsetof(PyModuleDef, m_size));
    P("type_slot_size", sizeof(PyType_Slot));
    P("type_spec_size", sizeof(PyType_Spec));
    P("method_size", sizeof(PyMethodDef));
    P("getset_size", sizeof(PyGetSetDef));
#ifdef Py_TPFLAGS_HAVE_VECTORCALL
    P("tp_vectorcall_offset", offsetof(PyTypeObject, tp_vectorcall_offset));
#endif
    return 0;
}
"""


def _layout_via_cc_probe(include_dirs) -> dict:
    import subprocess

    from .utils.link_utils import get_default_linkers

    probe_src = _write_snippet(suffix='.c', extra=_PROBE_MAIN)
    exe = probe_src[:-2] + ('.exe' if sys.platform == 'win32' else '')
    try:
        errors = []
        for linker in get_default_linkers():
            cmd = linker.split() + [
                '-I' + d for d in include_dirs
            ] + [probe_src, '-o', exe]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False,
            )
            if result.returncode == 0:
                break
            errors.append(result.stderr.strip().splitlines()[-1]
                          if result.stderr.strip() else str(result.returncode))
        else:
            raise RuntimeError('; '.join(errors) or 'no C compiler found')
        result = subprocess.run(
            [exe], capture_output=True, text=True, check=True,
        )
        layout = {}
        for line in result.stdout.splitlines():
            key, _, value = line.partition(' ')
            layout[key] = int(value)
        if layout.get('have_vectorcall') and 'tp_vectorcall_offset' not in layout:
            raise RuntimeError('probe missed tp_vectorcall_offset')
        _check(layout)
        return layout
    finally:
        for path in (probe_src, exe):
            if os.path.exists(path):
                os.remove(path)


def _include_dirs():
    found = []
    for key in ('INCLUDEPY', 'CONFINCLUDEPY'):
        path = sysconfig.get_config_var(key)
        if path and path not in found:
            found.append(path)
    path = sysconfig.get_path('include')
    if path and path not in found:
        found.append(path)
    if not found:
        raise RuntimeError(
            'cannot find C API headers for {} {}'.format(
                sys.implementation.name, sys.version.split()[0])
        )
    return found


def _write_snippet(suffix='.c', extra='') -> str:
    handle = tempfile.NamedTemporaryFile(
        prefix='pythoc-callable-layout-',
        suffix=suffix,
        delete=False,
    )
    handle.write((_SNIPPET + extra).encode('ascii'))
    handle.close()
    return handle.name


def _parse(source: str, include_dirs):
    from .cimport_clang import _load_cindex, resolve_parse_options

    cindex = _load_cindex()
    _target, _sysroot, args = resolve_parse_options(include_dirs=include_dirs)
    index = cindex.Index.create()
    tu = index.parse(source, args=args)
    errors = []
    for diag in tu.diagnostics:
        if diag.severity >= cindex.Diagnostic.Error:
            errors.append(str(diag))
    if errors:
        raise RuntimeError(
            'cannot read the {} {} C API from {}:\n{}'.format(
                sys.implementation.name,
                sys.version.split()[0],
                include_dirs[0],
                '\n'.join(errors[:8]),
            )
        )
    return tu


def _enum_values(cursor) -> dict:
    values = {}

    def walk(node):
        kind = node.kind.name
        if kind == 'ENUM_CONSTANT_DECL' and node.spelling:
            values[node.spelling] = int(node.enum_value)
        for child in node.get_children():
            walk(child)

    walk(cursor)
    return values


def _record(cursor, name: str) -> dict:
    found = []

    def walk(node):
        kind = node.kind.name
        if node.spelling != name:
            for child in node.get_children():
                walk(child)
            return
        if kind not in ('STRUCT_DECL', 'TYPEDEF_DECL'):
            for child in node.get_children():
                walk(child)
            return
        record = node.type.get_canonical()
        size = record.get_size()
        if size > 0:
            fields = {}
            decl = record.get_declaration()
            for child in decl.get_children():
                if child.kind.name != 'FIELD_DECL' or not child.spelling:
                    continue
                fields[child.spelling] = {
                    'offset': record.get_offset(child.spelling) // 8,
                    'size': child.type.get_size(),
                }
            found.append({'size': size, 'fields': fields})
        for child in node.get_children():
            walk(child)

    walk(cursor)
    if not found:
        raise RuntimeError('C API header has no {}'.format(name))
    return found[-1]


def _field(record: dict, name: str) -> dict:
    field = record['fields'].get(name)
    if field is None:
        known = ', '.join(sorted(record['fields']))
        raise RuntimeError(
            'C API type is missing {}: {}'.format(name, known)
        )
    return field


def _need(values: dict, name: str) -> int:
    if name not in values:
        raise RuntimeError('C API probe is missing {}'.format(name))
    return int(values[name])


def _check(layout: dict) -> None:
    if layout['pointer'] != 8 or layout['ssize'] != 8 or layout['int'] != 4:
        raise RuntimeError(
            'PythoC callable runtime needs a 64-bit C API, got '
            'pointer={} ssize={} int={}'.format(
                layout['pointer'], layout['ssize'], layout['int'])
        )
    expected = {
        'type_slot_size': 16,
        'type_spec_size': 32,
        'method_size': 32,
        'getset_size': 40,
    }
    for name, size in expected.items():
        if layout[name] != size:
            raise RuntimeError(
                'unexpected {} {} for this interpreter'.format(
                    name, layout[name])
            )
    if layout['tp_flags_size'] not in (4, 8):
        raise RuntimeError(
            'unexpected tp_flags size {}'.format(layout['tp_flags_size'])
        )
