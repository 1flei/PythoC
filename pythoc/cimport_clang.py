"""libclang-backed C import frontend."""

from __future__ import annotations

import math
import os
import platform
import re
import shlex
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple

from .cimport_emitter import write_pythoc_module
from .cimport_ir import (
    CDeclIR,
    CEnumValueIR,
    CFieldIR,
    CModuleIR,
    CParamIR,
    CTypeIR,
)
from .cimport_wrappers import render_stub_source, wrapper_prefix_for_path


class ClangCImportError(RuntimeError):
    """Raised when the libclang cimport backend cannot produce bindings."""


def _load_cindex():
    try:
        from clang import cindex
    except ImportError as exc:
        raise ClangCImportError(
            "clang Python bindings are not installed; pythoc requires "
            "the 'libclang' package for cimport. Install it with: "
            "pip install libclang"
        ) from exc

    from .config import config

    libclang_path = config.libclang_path
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
    enable_wrappers: bool = True,
    stub_path: Optional[str] = None,
    includes: bool = False,
) -> None:
    module = parse_to_ir(
        path,
        cflags=cflags,
        include_dirs=include_dirs,
        defines=defines,
        target=target,
        sysroot=sysroot,
        clang_args=clang_args,
        enable_wrappers=enable_wrappers,
        includes=includes,
    )
    prefix = wrapper_prefix_for_path(path) if enable_wrappers else None
    write_pythoc_module(module, lib, output_path, wrapper_prefix=prefix)
    if stub_path is not None:
        _write_wrapper_stub(module, path, prefix, stub_path)


def _write_wrapper_stub(
    module: CModuleIR,
    path: str,
    prefix: Optional[str],
    stub_path: str,
) -> None:
    """Write the inline-wrapper stub .c next to the bindings, or remove a
    stale stub when the current parse has no functions needing wrappers."""
    source = render_stub_source(module, path, prefix) if prefix else None
    if source is None:
        if os.path.exists(stub_path):
            os.remove(stub_path)
        return
    tmp_path = stub_path + ".tmp." + str(os.getpid())
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(source)
        os.replace(tmp_path, stub_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def parse_to_ir(
    path: str,
    *,
    cflags: Optional[list[str]] = None,
    include_dirs: Optional[list[str]] = None,
    defines: Optional[list[str]] = None,
    target: Optional[str] = None,
    sysroot: Optional[str] = None,
    clang_args: Optional[list[str]] = None,
    enable_wrappers: bool = True,
    includes: bool = False,
) -> CModuleIR:
    cindex = _load_cindex()
    effective_target, _effective_sysroot, args = resolve_parse_options(
        cflags=cflags,
        include_dirs=include_dirs,
        defines=defines,
        target=target,
        sysroot=sysroot,
        clang_args=clang_args,
    )

    index = cindex.Index.create()
    # Function bodies must be parsed: is_definition() (and with it the
    # header-defined inline/static wrapper detection) only reports True
    # when the body is part of the translation unit.
    options = cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
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
    seen: dict[tuple[str, str], int] = {}
    seen_is_main: dict[tuple[str, str], bool] = {}
    macro_slots: dict[str, int] = {}
    macro_is_main: dict[str, bool] = {}
    target_arch = _resolve_target_arch(effective_target)
    main_file = os.path.abspath(path)

    # Main-file declarations are always emitted.  With includes=True,
    # declarations pulled in transitively from included files are emitted
    # too (system headers frequently delegate to private sub-headers:
    # glibc math.h -> bits/mathcalls.h, macOS string.h -> _string.h).
    # The walk follows TU order, which matches C's declaration-before-use
    # order (an included file's declarations precede the includer's), so
    # the emitted module can bind names sequentially; dedup keeps the
    # first-seen slot and a main-file declaration wins the slot over an
    # included-file one.
    for cursor in tu.cursor.get_children():
        is_main = _is_from_main_file(cursor, main_file)
        if not is_main:
            if not includes:
                continue
            location = cursor.location
            if location is None or location.file is None:
                continue
        try:
            decl = _cursor_to_decl(cindex, tu, cursor, target_arch,
                                   enable_wrappers, main_file, includes)
        except Exception as exc:
            # One unconvertible declaration must not abort the whole import
            # (e.g. python-clang not knowing a TypeKind the platform SDK
            # uses).  Degrade it to a lazy error via _unsupported_symbols,
            # the same way unsupported types are handled.
            name = cursor.spelling or f"_unparsed_{len(module.declarations)}"
            key = ("error", name)
            if key not in seen:
                seen[key] = len(module.declarations)
                seen_is_main[key] = is_main
                module.declarations.append(CDeclIR(
                    "error", name,
                    reason=f"declaration could not be converted: {exc}"))
            continue
        if decl is None:
            continue
        if decl.kind == "macro":
            # A redefinition after #undef replaces the earlier definition,
            # matching the preprocessor's final value.  A main-file macro
            # wins over a same-named macro from an included file.
            slot = macro_slots.get(decl.name)
            if slot is None:
                macro_slots[decl.name] = len(module.declarations)
                macro_is_main[decl.name] = is_main
                module.declarations.append(decl)
            elif is_main or not macro_is_main[decl.name]:
                module.declarations[slot] = decl
                macro_is_main[decl.name] = is_main or macro_is_main[decl.name]
            continue
        key = (decl.kind, decl.name)
        slot = seen.get(key)
        if slot is not None:
            old = module.declarations[slot]
            old_is_main = seen_is_main[key]
            # A definition carries strictly more information than a plain
            # declaration (e.g. needs_wrapper detection); otherwise a
            # main-file declaration wins over an included-file one.
            if ((decl.is_definition and not old.is_definition)
                    or (is_main and not old_is_main and not old.is_definition)):
                module.declarations[slot] = decl
                seen_is_main[key] = is_main
            continue
        seen[key] = len(module.declarations)
        seen_is_main[key] = is_main
        module.declarations.append(decl)
    return module


def _resolve_target_arch(target: Optional[str]) -> str:
    triple = target or platform.machine().lower()
    arch = triple.split("-")[0].lower()
    if arch in ("arm64", "aarch64"):
        return "aarch64"
    if arch in ("x86_64", "amd64"):
        return "x86_64"
    return arch


def _split_env_args(value: Optional[str]) -> List[str]:
    """Shell-tokenise a whitespace-separated config string."""
    if not value:
        return []
    return shlex.split(value)


def _split_env_include_path(value: Optional[str]) -> List[str]:
    """Split PC_CIMPORT_INCLUDE_PATH (';' or os.pathsep separated)."""
    if not value:
        return []
    normalized = value.replace(';', os.pathsep)
    return [entry for entry in normalized.split(os.pathsep) if entry]


# Memoised per (cc, target): scraping runs a compiler subprocess.
_HOST_INCLUDE_CACHE: Dict[Tuple[str, str], List[str]] = {}


def _host_system_include_dirs(target: Optional[str] = None) -> List[str]:
    """Scrape the host C compiler's built-in #include <...> search dirs.

    Runs ``cc -E -v -x c -`` with stdin closed and parses the
    ``#include <...> search starts here:`` block from stderr.  When
    ``target`` is set it is passed as ``-target``; a compiler that rejects
    it (e.g. gcc) yields no dirs rather than host dirs that would be wrong
    for a cross target.  Returns [] when no C compiler is available --
    libclang usually finds its own resource headers without help.
    """
    from .utils.cc_utils import find_available_cc

    try:
        cc = find_available_cc()
    except RuntimeError:
        return []
    key = (cc, target or '')
    if key in _HOST_INCLUDE_CACHE:
        return list(_HOST_INCLUDE_CACHE[key])
    dirs: List[str] = []
    cmd = cc.split() + ['-E', '-v', '-x', 'c', '-']
    if target:
        cmd.extend(['-target', target])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=30, stdin=subprocess.DEVNULL)
        in_block = False
        for line in (result.stderr or '').splitlines():
            if '#include <...> search starts here:' in line:
                in_block = True
                continue
            if not in_block:
                continue
            if 'End of search list.' in line:
                break
            # Drop trailing annotations like "(framework directory)".
            entry = line.strip().split(' (', 1)[0].strip()
            if entry and os.path.isdir(entry):
                dirs.append(entry)
    except (OSError, subprocess.SubprocessError):
        dirs = []
    _HOST_INCLUDE_CACHE[key] = dirs
    return list(dirs)


def default_include_search_dirs(
    target: Optional[str] = None,
    sysroot: Optional[str] = None,
) -> List[str]:
    """Include dirs supplied by the environment and the host toolchain.

    Combines PC_CIMPORT_INCLUDE_PATH with the host compiler's built-in
    system include dirs.  Host scraping is skipped when an explicit
    sysroot is given: the user controls the header root, and host paths
    would be wrong inside it.
    """
    from .config import config

    effective_target = target or config.cimport_target
    effective_sysroot = sysroot or config.cimport_sysroot
    dirs = _split_env_include_path(config.cimport_include_path)
    if not effective_sysroot:
        dirs.extend(_host_system_include_dirs(effective_target))
    return dirs


def resolve_parse_options(
    *,
    cflags: Optional[list[str]] = None,
    include_dirs: Optional[list[str]] = None,
    defines: Optional[list[str]] = None,
    target: Optional[str] = None,
    sysroot: Optional[str] = None,
    clang_args: Optional[list[str]] = None,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Resolve the effective (target, sysroot, clang args) for one parse.

    Applies config/env defaults (PC_CIMPORT_TARGET / PC_CIMPORT_SYSROOT /
    PC_CIMPORT_CLANG_ARGS / PC_CIMPORT_INCLUDE_PATH) and appends the host
    compiler's built-in system include dirs as ``-isystem`` entries so
    real system headers resolve without explicit ``include_dirs``.
    """
    from .config import config

    effective_target = target or config.cimport_target
    effective_sysroot = sysroot or config.cimport_sysroot
    effective_clang_args = _split_env_args(config.cimport_clang_args)
    if clang_args:
        effective_clang_args.extend(clang_args)
    env_include_dirs = _split_env_include_path(config.cimport_include_path)
    system_dirs = (
        [] if effective_sysroot
        else _host_system_include_dirs(effective_target)
    )
    args = _build_clang_args(
        cflags=cflags,
        include_dirs=list(include_dirs or []) + env_include_dirs,
        defines=defines,
        target=effective_target,
        sysroot=effective_sysroot,
        clang_args=effective_clang_args,
        system_include_dirs=system_dirs,
    )
    return effective_target, effective_sysroot, args


def _build_clang_args(
    *,
    cflags: Optional[list[str]],
    include_dirs: Optional[list[str]],
    defines: Optional[list[str]],
    target: Optional[str],
    sysroot: Optional[str],
    clang_args: Optional[list[str]],
    system_include_dirs: Optional[list[str]] = None,
) -> list[str]:
    args: list[str] = []
    if target:
        args.extend(["-target", target])
    if sysroot:
        args.append(f"--sysroot={sysroot}")
    for include_dir in include_dirs or []:
        args.extend(["-I", include_dir])
    for system_dir in system_include_dirs or []:
        args.extend(["-isystem", system_dir])
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


def _is_from_emitted_file(cursor, main_file: str, include_all: bool) -> bool:
    """True when the cursor's file is part of the emitted declaration set."""
    location = cursor.location
    if location is None or location.file is None:
        return False
    if include_all:
        return True
    return os.path.abspath(str(location.file)) == main_file


def _cursor_to_decl(cindex, tu, cursor, target_arch: str,
                    enable_wrappers: bool, main_file: str,
                    include_all: bool = False) -> CDeclIR | None:
    kind = cursor.kind
    if kind == cindex.CursorKind.FUNCTION_DECL:
        if not cursor.spelling:
            return None
        params = [
            CParamIR(arg.spelling or None, _type_to_ir(cindex, arg.type, target_arch, main_file, include_all))
            for arg in cursor.get_arguments()
        ]
        func_ty = CTypeIR(
            "function",
            return_type=_type_to_ir(cindex, cursor.result_type, target_arch, main_file, include_all),
            params=params,
            is_variadic=_is_function_variadic(cindex, cursor.type),
        )
        is_static = cursor.storage_class == cindex.StorageClass.STATIC
        # A definition has no guaranteed external symbol when it has internal
        # linkage (static / static inline) or when it is a C99 inline
        # definition: the inline specifier without extern emits no code.
        # `extern inline` is excluded because it forces an external
        # definition wherever the header is compiled.
        needs_wrapper = (
            enable_wrappers
            and cursor.is_definition()
            and (
                is_static
                or (
                    cursor.storage_class != cindex.StorageClass.EXTERN
                    and _has_inline_specifier(cindex, tu, cursor)
                )
            )
        )
        return CDeclIR(
            "function",
            cursor.spelling,
            func_ty,
            storage="static" if is_static else None,
            is_definition=cursor.is_definition(),
            needs_wrapper=needs_wrapper,
            symbol=_asm_label(cindex, cursor),
        )

    if kind == cindex.CursorKind.STRUCT_DECL:
        name = cursor.spelling or _type_record_name(cursor.type, "AnonymousStruct")
        return CDeclIR(
            "struct",
            name,
            fields=_fields_to_ir(cindex, cursor, target_arch, main_file, include_all),
            size_bytes=_complete_type_size(cursor.type),
        )

    if kind == cindex.CursorKind.UNION_DECL:
        name = cursor.spelling or _type_record_name(cursor.type, "AnonymousUnion")
        return CDeclIR(
            "union",
            name,
            fields=_fields_to_ir(cindex, cursor, target_arch, main_file, include_all),
            size_bytes=_complete_type_size(cursor.type),
        )

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
        return CDeclIR("typedef", cursor.spelling, _type_to_ir(cindex, cursor.underlying_typedef_type, target_arch, main_file, include_all))

    if kind == cindex.CursorKind.VAR_DECL:
        if not cursor.spelling:
            return None
        storage = None
        if cursor.storage_class == cindex.StorageClass.STATIC:
            storage = "static"
        elif cursor.storage_class == cindex.StorageClass.EXTERN:
            storage = "extern"
        is_thread_local = False
        tls_kind = getattr(cursor, "tls_kind", None)
        if tls_kind is not None and tls_kind != cindex.TLSKind.NONE:
            is_thread_local = True
        return CDeclIR(
            "var",
            cursor.spelling,
            _type_to_ir(cindex, cursor.type, target_arch, main_file, include_all),
            storage=storage,
            is_definition=cursor.is_definition(),
            is_thread_local=is_thread_local,
            symbol=_asm_label(cindex, cursor),
        )

    if kind == cindex.CursorKind.MACRO_DEFINITION:
        if not cursor.spelling or _is_macro_functionlike(cindex, cursor):
            return None
        value = _macro_literal_value(cindex, tu, cursor)
        if value is None:
            return None
        return CDeclIR("macro", cursor.spelling, value=value)

    return None


_INLINE_SPECIFIER_SPELLINGS = frozenset({"inline", "__inline", "__inline__"})


def _has_inline_specifier(cindex, tu, cursor) -> bool:
    """True when the function's declaration specifiers contain `inline`.

    libclang exposes no direct inline-specifier query, so scan the
    declaration-specifier tokens (everything before the first '(' of the
    declarator; specifiers always precede it in C).
    """
    for token in tu.get_tokens(extent=cursor.extent):
        if token.kind == cindex.TokenKind.PUNCTUATION and token.spelling == "(":
            return False
        if (token.kind == cindex.TokenKind.KEYWORD
                and token.spelling in _INLINE_SPECIFIER_SPELLINGS):
            return True
    return False


def _is_macro_functionlike(cindex, cursor) -> bool:
    if hasattr(cursor, "is_macro_functionlike"):
        return bool(cursor.is_macro_functionlike())
    return bool(cindex.conf.lib.clang_Cursor_isMacroFunctionLike(cursor))


def _complete_type_size(typ) -> Optional[int]:
    try:
        size = typ.get_size()
    except Exception:
        return None
    return size if size >= 0 else None


_INT_SUFFIX_RE = re.compile(r"[uUlL]+$")
_FLOAT_SUFFIX_RE = re.compile(r"[fFlL]$")


def _macro_literal_value(cindex, tu, cursor):
    """Extract a numeric macro value following a conservative rule.

    Only a single LITERAL token, optionally preceded by a '-' punctuation
    token and optionally wrapped in one pair of parentheses (the common
    ``#define EOF (-1)`` idiom), is accepted.  Everything else
    (expressions, strings, empty bodies) yields None and the macro is
    skipped.
    """
    tokens = list(tu.get_tokens(extent=cursor.extent))
    body = tokens[1:]
    if (len(body) >= 3
            and body[0].kind == cindex.TokenKind.PUNCTUATION
            and body[0].spelling == "("
            and body[-1].kind == cindex.TokenKind.PUNCTUATION
            and body[-1].spelling == ")"):
        body = body[1:-1]
    sign = 1
    if len(body) == 1:
        literal = body[0]
    elif (len(body) == 2
          and body[0].kind == cindex.TokenKind.PUNCTUATION
          and body[0].spelling == "-"):
        sign = -1
        literal = body[1]
    else:
        return None
    if literal.kind != cindex.TokenKind.LITERAL:
        return None
    value = _parse_numeric_literal(literal.spelling)
    if value is None:
        return None
    return -value if sign < 0 else value


def _parse_numeric_literal(spelling: str):
    s = spelling.strip()
    int_body = _INT_SUFFIX_RE.sub("", s)
    if int_body.lower().startswith("0x"):
        try:
            return int(int_body, 16)
        except ValueError:
            return None
    if int_body.isdigit():
        if len(int_body) > 1 and int_body.startswith("0"):
            # C-style octal; Python's int(x, 0) rejects '010'
            if any(ch not in "01234567" for ch in int_body):
                return None
            return int(int_body, 8)
        return int(int_body, 10)
    float_body = _FLOAT_SUFFIX_RE.sub("", s)
    try:
        value = float(float_body)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _asm_label(cindex, cursor) -> Optional[str]:
    """Return the assembly label of a declaration, or None.

    clang_Cursor_getMangling yields the __asm__("label") name verbatim when
    one is attached (glibc __REDIRECT scenario); otherwise it yields the
    platform C mangling, which equals the spelling on ELF targets and the
    spelling with a leading underscore on Mach-O / 32-bit Windows.  Only a
    mangling matching neither convention is an actual asm label.
    """
    get_mangling = getattr(cindex.conf.lib, "clang_Cursor_getMangling", None)
    if get_mangling is None:
        return None
    try:
        mangling = get_mangling(cursor)
    except Exception:
        return None
    if not mangling:
        return None
    spelling = cursor.spelling
    if mangling == spelling or mangling == "_" + spelling:
        return None
    return mangling


def _field_offset_bytes(cindex, cursor) -> Optional[int]:
    """Byte offset of a FIELD_DECL within its record, None when unknown."""
    try:
        offset_bits = cursor.get_field_offsetof()
    except Exception:
        return None
    if offset_bits < 0 or offset_bits % 8 != 0:
        return None
    return offset_bits // 8


def _fields_to_ir(cindex, cursor, target_arch: str, main_file: str,
                  include_all: bool = False) -> list[CFieldIR]:
    fields: list[CFieldIR] = []
    for child in cursor.get_children():
        if child.kind != cindex.CursorKind.FIELD_DECL:
            continue
        bit_width = None
        offset_bytes = None
        if child.is_bitfield():
            bit_width = child.get_bitfield_width()
        else:
            offset_bytes = _field_offset_bytes(cindex, child)
        fields.append(CFieldIR(
            child.spelling or None,
            _type_to_ir(cindex, child.type, target_arch, main_file, include_all),
            bit_width,
            offset_bytes,
        ))
    return fields


def _type_to_ir(cindex, typ, target_arch: str, main_file: str,
                include_all: bool = False) -> CTypeIR:
    kind = typ.kind
    tk = cindex.TypeKind

    if kind == tk.ELABORATED:
        return _type_to_ir(cindex, typ.get_named_type(), target_arch, main_file, include_all)
    if kind == tk.TYPEDEF:
        # A typedef declared in an emitted file is emitted by the emitter,
        # so keep the name reference.  A typedef from a non-emitted header
        # is never emitted, so resolve through to the underlying type to
        # avoid dangling references.
        decl = typ.get_declaration()
        if decl is not None and _is_from_emitted_file(decl, main_file, include_all):
            return CTypeIR("typedef", name=typ.spelling)
        if decl is not None:
            return _type_to_ir(
                cindex, decl.underlying_typedef_type, target_arch, main_file, include_all)
        return CTypeIR("typedef", name=typ.spelling)
    if kind == tk.POINTER:
        pointee = typ.get_pointee()
        if pointee.kind in (tk.FUNCTIONPROTO, tk.FUNCTIONNOPROTO):
            # A C function pointer is a single indirection and pythoc's
            # func[...] is already a function-pointer value, so map
            # directly instead of double-wrapping as ptr[func[...]].
            return _type_to_ir(cindex, pointee, target_arch, main_file, include_all)
        return CTypeIR("pointer", pointee=_type_to_ir(cindex, pointee, target_arch, main_file, include_all))
    if kind in (tk.CONSTANTARRAY, tk.INCOMPLETEARRAY, tk.VARIABLEARRAY):
        size = typ.element_count if kind == tk.CONSTANTARRAY else -1
        return CTypeIR("array", element=_type_to_ir(cindex, typ.element_type, target_arch, main_file, include_all), size=size)
    if kind in (tk.FUNCTIONPROTO, tk.FUNCTIONNOPROTO):
        params = [
            CParamIR(None, _type_to_ir(cindex, arg_type, target_arch, main_file, include_all))
            for arg_type in _argument_types(typ)
        ]
        return CTypeIR(
            "function",
            return_type=_type_to_ir(cindex, typ.get_result(), target_arch, main_file, include_all),
            params=params,
            is_variadic=_is_function_variadic(cindex, typ),
        )
    if kind == tk.RECORD:
        decl = typ.get_declaration()
        if decl.kind == cindex.CursorKind.UNION_DECL:
            return CTypeIR("union", name=decl.spelling or _type_record_name(typ, "AnonymousUnion"))
        return CTypeIR("struct", name=decl.spelling or _type_record_name(typ, "AnonymousStruct"))
    if kind == tk.ENUM:
        decl = typ.get_declaration()
        return CTypeIR("enum", name=decl.spelling or _type_record_name(typ, "AnonymousEnum"))
    if kind == tk.LONGDOUBLE:
        size = _complete_type_size(typ)
        if size == 8:
            # Platforms where long double is identical to double (e.g.
            # macOS arm64, Windows).
            return CTypeIR("primitive", name="f64")
        if size == 16 and target_arch == "aarch64":
            return CTypeIR("primitive", name="f128")
        return CTypeIR(
            "unsupported",
            name="long double",
            reason=(
                f"'long double' ({size}-byte) has no matching pythoc type "
                f"on target {target_arch}"
            ),
        )

    primitive = _primitive_type_name(cindex, typ)
    if primitive is not None:
        return CTypeIR("primitive", name=primitive)

    spelling = typ.spelling
    if spelling:
        return CTypeIR("named", name=_strip_c_type_prefix(spelling))
    return CTypeIR("primitive", name="i32")


def _is_function_variadic(cindex, typ) -> bool:
    # libclang's is_function_variadic() asserts on FUNCTIONNOPROTO (K&R
    # no-proto `int f();`) types; those are never variadic.
    if typ.kind != cindex.TypeKind.FUNCTIONPROTO:
        return False
    return typ.is_function_variadic()


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
        # pythoc has no distinct char type (its `char` builtin is a
        # conversion function, not a type); plain C char maps to i8.
        return "u8" if kind == tk.CHAR_U else "i8"
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
    if kind == tk.DOUBLE:
        return "f64"
    return None


def _type_record_name(typ, fallback: str) -> str:
    return _strip_c_type_prefix(typ.spelling) or fallback


def _strip_c_type_prefix(name: str) -> str:
    for prefix in ("struct ", "union ", "enum "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name
