"""Optional libclang-backed C import frontend."""

from __future__ import annotations

import os
from typing import Iterable, Optional

from .cimport_emitter import write_pythoc_module
from .cimport_ir import (
    CDeclIR,
    CEnumValueIR,
    CFieldIR,
    CModuleIR,
    CParamIR,
    CTypeIR,
)


class ClangCImportError(RuntimeError):
    """Raised when the libclang cimport backend cannot produce bindings."""


def _load_cindex():
    try:
        from clang import cindex
    except ImportError as exc:
        raise ClangCImportError(
            "clang Python bindings are not installed; install optional "
            "packages 'clang' and 'libclang' to use PC_CIMPORT_BACKEND=clang"
        ) from exc

    libclang_path = os.environ.get("PC_LIBCLANG_PATH")
    if libclang_path and not cindex.Config.loaded:
        if os.path.isdir(libclang_path):
            cindex.Config.set_library_path(libclang_path)
        else:
            cindex.Config.set_library_file(libclang_path)
    return cindex


def is_clang_backend_available() -> bool:
    try:
        cindex = _load_cindex()
        cindex.Index.create()
    except Exception:
        return False
    return True


def generate_bindings_to_file(
    path: str,
    lib: str,
    output_path: str,
    *,
    cflags: Optional[list[str]] = None,
    include_dirs: Optional[list[str]] = None,
    defines: Optional[list[str]] = None,
    target: Optional[str] = None,
    sysroot: Optional[str] = None,
    clang_args: Optional[list[str]] = None,
) -> None:
    module = parse_to_ir(
        path,
        cflags=cflags,
        include_dirs=include_dirs,
        defines=defines,
        target=target,
        sysroot=sysroot,
        clang_args=clang_args,
    )
    write_pythoc_module(module, lib, output_path)


def parse_to_ir(
    path: str,
    *,
    cflags: Optional[list[str]] = None,
    include_dirs: Optional[list[str]] = None,
    defines: Optional[list[str]] = None,
    target: Optional[str] = None,
    sysroot: Optional[str] = None,
    clang_args: Optional[list[str]] = None,
) -> CModuleIR:
    cindex = _load_cindex()
    args = _build_clang_args(
        cflags=cflags,
        include_dirs=include_dirs,
        defines=defines,
        target=target,
        sysroot=sysroot,
        clang_args=clang_args,
    )

    index = cindex.Index.create()
    options = (
        cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        | cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES
    )
    tu = index.parse(os.path.abspath(path), args=args, options=options)
    fatal = [
        diag
        for diag in tu.diagnostics
        if diag.severity >= cindex.Diagnostic.Error
    ]
    if fatal:
        rendered = "\n".join(str(diag) for diag in fatal[:5])
        raise ClangCImportError(f"libclang failed to parse {path}:\n{rendered}")

    module = CModuleIR()
    seen: set[tuple[str, str]] = set()
    main_file = os.path.abspath(path)
    for cursor in tu.cursor.get_children():
        if not _is_from_main_file(cursor, main_file):
            continue
        decl = _cursor_to_decl(cindex, cursor)
        if decl is None:
            continue
        key = (decl.kind, decl.name)
        if key in seen:
            continue
        seen.add(key)
        module.declarations.append(decl)
    return module


def _build_clang_args(
    *,
    cflags: Optional[list[str]],
    include_dirs: Optional[list[str]],
    defines: Optional[list[str]],
    target: Optional[str],
    sysroot: Optional[str],
    clang_args: Optional[list[str]],
) -> list[str]:
    args: list[str] = []
    if target:
        args.extend(["-target", target])
    if sysroot:
        args.append(f"--sysroot={sysroot}")
    for include_dir in include_dirs or []:
        args.extend(["-I", include_dir])
    for define in defines or []:
        args.append(f"-D{define}")
    args.extend(cflags or [])
    args.extend(clang_args or [])
    return args


def _is_from_main_file(cursor, main_file: str) -> bool:
    location = cursor.location
    if location is None or location.file is None:
        return False
    return os.path.abspath(str(location.file)) == main_file


def _cursor_to_decl(cindex, cursor) -> CDeclIR | None:
    kind = cursor.kind
    if kind == cindex.CursorKind.FUNCTION_DECL:
        if not cursor.spelling:
            return None
        params = [
            CParamIR(arg.spelling or None, _type_to_ir(cindex, arg.type))
            for arg in cursor.get_arguments()
        ]
        func_ty = CTypeIR(
            "function",
            return_type=_type_to_ir(cindex, cursor.result_type),
            params=params,
            is_variadic=cursor.type.is_function_variadic(),
        )
        return CDeclIR(
            "function",
            cursor.spelling,
            func_ty,
            is_definition=cursor.is_definition(),
        )

    if kind == cindex.CursorKind.STRUCT_DECL:
        name = cursor.spelling or _type_record_name(cursor.type, "AnonymousStruct")
        return CDeclIR("struct", name, fields=_fields_to_ir(cindex, cursor))

    if kind == cindex.CursorKind.UNION_DECL:
        name = cursor.spelling or _type_record_name(cursor.type, "AnonymousUnion")
        return CDeclIR("union", name, fields=_fields_to_ir(cindex, cursor))

    if kind == cindex.CursorKind.ENUM_DECL:
        name = cursor.spelling or _type_record_name(cursor.type, "AnonymousEnum")
        values = []
        for child in cursor.get_children():
            if child.kind == cindex.CursorKind.ENUM_CONSTANT_DECL:
                values.append(CEnumValueIR(child.spelling, child.enum_value))
        return CDeclIR("enum", name, values=values)

    if kind == cindex.CursorKind.TYPEDEF_DECL:
        if not cursor.spelling:
            return None
        return CDeclIR("typedef", cursor.spelling, _type_to_ir(cindex, cursor.underlying_typedef_type))

    if kind == cindex.CursorKind.VAR_DECL:
        if not cursor.spelling:
            return None
        return CDeclIR("var", cursor.spelling, _type_to_ir(cindex, cursor.type))

    return None


def _fields_to_ir(cindex, cursor) -> list[CFieldIR]:
    fields: list[CFieldIR] = []
    for child in cursor.get_children():
        if child.kind != cindex.CursorKind.FIELD_DECL:
            continue
        bit_width = None
        if child.is_bitfield():
            bit_width = child.get_bitfield_width()
        fields.append(CFieldIR(child.spelling or None, _type_to_ir(cindex, child.type), bit_width))
    return fields


def _type_to_ir(cindex, typ) -> CTypeIR:
    kind = typ.kind
    tk = cindex.TypeKind

    if kind == tk.ELABORATED:
        return _type_to_ir(cindex, typ.get_named_type())
    if kind == tk.TYPEDEF:
        return CTypeIR("typedef", name=typ.spelling)
    if kind == tk.POINTER:
        return CTypeIR("pointer", pointee=_type_to_ir(cindex, typ.get_pointee()))
    if kind in (tk.CONSTANTARRAY, tk.INCOMPLETEARRAY, tk.VARIABLEARRAY):
        size = typ.element_count if kind == tk.CONSTANTARRAY else -1
        return CTypeIR("array", element=_type_to_ir(cindex, typ.element_type), size=size)
    if kind in (tk.FUNCTIONPROTO, tk.FUNCTIONNOPROTO):
        params = [
            CParamIR(None, _type_to_ir(cindex, arg_type))
            for arg_type in _argument_types(typ)
        ]
        return CTypeIR(
            "function",
            return_type=_type_to_ir(cindex, typ.get_result()),
            params=params,
            is_variadic=typ.is_function_variadic(),
        )
    if kind == tk.RECORD:
        decl = typ.get_declaration()
        if decl.kind == cindex.CursorKind.UNION_DECL:
            return CTypeIR("union", name=decl.spelling or _type_record_name(typ, "AnonymousUnion"))
        return CTypeIR("struct", name=decl.spelling or _type_record_name(typ, "AnonymousStruct"))
    if kind == tk.ENUM:
        decl = typ.get_declaration()
        return CTypeIR("enum", name=decl.spelling or _type_record_name(typ, "AnonymousEnum"))

    primitive = _primitive_type_name(cindex, typ)
    if primitive is not None:
        return CTypeIR("primitive", name=primitive)

    spelling = typ.spelling
    if spelling:
        return CTypeIR("named", name=_strip_c_type_prefix(spelling))
    return CTypeIR("primitive", name="i32")


def _argument_types(typ) -> Iterable:
    try:
        return typ.argument_types()
    except Exception:
        return []


def _primitive_type_name(cindex, typ) -> str | None:
    tk = cindex.TypeKind
    kind = typ.kind
    spelling = typ.spelling

    if kind == tk.VOID:
        return "void"
    if kind == tk.BOOL:
        return "bool"
    if kind in (tk.CHAR_S, tk.CHAR_U):
        return "char" if spelling == "char" else ("u8" if kind == tk.CHAR_U else "i8")
    if kind == tk.SCHAR:
        return "i8"
    if kind == tk.UCHAR:
        return "u8"
    if kind == tk.SHORT:
        return "i16"
    if kind == tk.USHORT:
        return "u16"
    if kind == tk.INT:
        return "i32"
    if kind == tk.UINT:
        return "u32"
    if kind in (tk.LONG, tk.LONGLONG):
        return "i64"
    if kind in (tk.ULONG, tk.ULONGLONG):
        return "u64"
    if kind == tk.FLOAT:
        return "f32"
    if kind in (tk.DOUBLE, tk.LONGDOUBLE):
        return "f64"
    return None


def _type_record_name(typ, fallback: str) -> str:
    return _strip_c_type_prefix(typ.spelling) or fallback


def _strip_c_type_prefix(name: str) -> str:
    for prefix in ("struct ", "union ", "enum "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name
