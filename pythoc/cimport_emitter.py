"""Emit pythoc source from normalized C import IR."""

from __future__ import annotations

import keyword

from .cimport_ir import CDeclIR, CFieldIR, CModuleIR, CParamIR, CTypeIR


_RESERVED_NAMES = {
    "compile",
    "extern",
    "enum",
    "ptr",
    "array",
    "struct",
    "union",
    "func",
    "void",
    "char",
    "bool",
}

_TAG_KINDS = ("struct", "union", "enum")


def _ident(name: str | None, fallback: str = "_unnamed") -> str:
    if not name:
        return fallback
    cleaned = []
    for i, ch in enumerate(name):
        if ch == "_" or ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    result = "".join(cleaned) or fallback
    if result[0].isdigit():
        result = "_" + result
    if keyword.iskeyword(result) or result in _RESERVED_NAMES:
        result += "_"
    return result


def _quote(value: str) -> str:
    return repr(value)


def _type_expr(ty: CTypeIR | None, renames: dict[str, str] | None = None) -> str:
    if ty is None:
        return "void"

    kind = ty.kind
    if kind == "primitive":
        return ty.name or "i32"
    if kind in ("named", "typedef"):
        # Typedefs and unknown named types keep their plain name; only tag
        # types are ever renamed (see _tag_renames).
        return _ident(ty.name, "i32")
    if kind == "pointer":
        return f"ptr[{_type_expr(ty.pointee, renames)}]"
    if kind == "array":
        elem = _type_expr(ty.element, renames)
        if ty.size is None or ty.size < 0:
            return f"ptr[{elem}]"
        return f"array[{elem}, {ty.size}]"
    if kind == "function":
        parts = [_type_expr(param.type, renames) for param in ty.params]
        parts.append(_type_expr(ty.return_type, renames))
        return "func[" + ", ".join(parts) + "]"
    if kind in _TAG_KINDS:
        name = _ident(ty.name, "i32")
        if renames:
            name = renames.get(name, name)
        return name
    return "i32"


def _unsupported_reason(ty: CTypeIR | None) -> str | None:
    if ty is None:
        return None
    if ty.kind == "unsupported":
        return ty.reason or f"unsupported C type '{ty.name}'"
    for child in (ty.pointee, ty.element, ty.return_type):
        reason = _unsupported_reason(child)
        if reason is not None:
            return reason
    for param in ty.params:
        reason = _unsupported_reason(param.type)
        if reason is not None:
            return reason
    return None


def _referenced_type_names(ty: CTypeIR | None, out: set[str],
                           renames: dict[str, str] | None = None) -> None:
    """Collect the emitted names of named types referenced by ty."""
    if ty is None:
        return
    if ty.kind in _TAG_KINDS and ty.name:
        name = _ident(ty.name)
        if renames:
            name = renames.get(name, name)
        out.add(name)
    elif ty.kind in ("typedef", "named") and ty.name:
        out.add(_ident(ty.name))
    for child in (ty.pointee, ty.element, ty.return_type):
        _referenced_type_names(child, out, renames)
    for param in ty.params:
        _referenced_type_names(param.type, out, renames)


def _decl_referenced_names(decl: CDeclIR,
                           renames: dict[str, str] | None = None) -> set[str]:
    out: set[str] = set()
    _referenced_type_names(decl.type, out, renames)
    for field in decl.fields:
        _referenced_type_names(field.type, out, renames)
    return out


def _decl_ident(decl: CDeclIR, renames: dict[str, str] | None = None,
                fallback: str = "_unnamed") -> str:
    """The module-level name a declaration is emitted under."""
    name = _ident(decl.name, fallback)
    if renames and decl.kind in _TAG_KINDS:
        name = renames.get(name, name)
    return name


def _decl_unsupported_reason(decl: CDeclIR,
                             wrapper_symbol: str | None) -> str | None:
    """Classify one declaration: None when emittable, else the lazy-error
    reason.  Mirrors the checks the emit helpers used to apply inline."""
    if decl.kind == "error":
        # The frontend failed to convert this declaration (e.g. a libclang
        # version that does not know a TypeKind the SDK uses); degrade to a
        # lazy error instead of aborting the whole import.
        return decl.reason or "declaration could not be converted"
    if decl.kind == "typedef":
        return _unsupported_reason(decl.type)
    if decl.kind == "var":
        if decl.storage == "static":
            return "static global has no external symbol (file-scope storage)"
        if decl.is_thread_local:
            return "thread-local global: TLS data symbols are not supported"
        return _unsupported_reason(decl.type)
    if decl.kind == "function":
        ty = decl.type
        reason = _unsupported_reason(ty)
        if reason is not None:
            return reason
        if decl.needs_wrapper and wrapper_symbol is None:
            if ty is not None and ty.is_variadic:
                return (
                    "variadic function defined in header without external "
                    "symbol (static/inline): varargs cannot be forwarded "
                    "through a generated wrapper"
                )
            return (
                "function defined in header without external symbol "
                "(static/inline) and no C wrapper could be generated for "
                "its signature"
            )
    return None


# Layout the emitted pythoc types have on the targets cimport supports
# (natural alignment, matching LLVM's default data layout).  cimport maps
# C long to i64 and pythoc pointers are hardcoded to 8 bytes, so cimport as
# a whole already assumes a 64-bit target.
_PRIMITIVE_LAYOUT = {
    "i8": (1, 1), "u8": (1, 1), "char": (1, 1), "bool": (1, 1),
    "i16": (2, 2), "u16": (2, 2),
    "i32": (4, 4), "u32": (4, 4), "f32": (4, 4),
    "i64": (8, 8), "u64": (8, 8), "f64": (8, 8),
    "f128": (16, 16),
}
_POINTER_LAYOUT = (8, 8)
# Enums are emitted as @enum(i32).
_ENUM_LAYOUT = (4, 4)
# Records referenced but never declared in the main file are emitted as
# opaque 1-byte placeholders.
_PLACEHOLDER_LAYOUT = (1, 1)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    remainder = value % alignment
    return value if remainder == 0 else value + alignment - remainder


class _LayoutModel:
    """Model of the memory layout the generated pythoc types will have.

    pythoc lays out @compile classes as LLVM structs: fields in declaration
    order, each aligned to its own natural alignment, total size padded to
    the maximum field alignment.  Unions become a byte buffer of the
    largest member padded to the largest member alignment.  A C record
    whose clang-observed field offsets or total size do not match this
    model (packed via #pragma pack/__attribute__, over-aligned via
    _Alignas, flexible array members, unverifiable field types, ...) cannot
    be represented faithfully as pythoc fields and must degrade to opaque
    storage.
    """

    def __init__(self, declarations: list[CDeclIR]):
        self._records: dict[str, CDeclIR] = {}
        self._typedefs: dict[str, CTypeIR] = {}
        for decl in declarations:
            if decl.kind in _TAG_KINDS and decl.kind != "enum":
                self._records.setdefault(_ident(decl.name), decl)
            elif decl.kind == "typedef":
                self._typedefs.setdefault(_ident(decl.name), decl.type)
        self._record_layouts: dict[int, tuple[int, int]] = {}
        self._record_accurate: dict[int, bool] = {}

    def type_layout(self, ty: CTypeIR | None,
                    _visiting: frozenset = frozenset()) -> tuple[int, int] | None:
        """(size, alignment) of the pythoc type emitted for ty, or None
        when the layout cannot be modeled."""
        if ty is None:
            return None
        kind = ty.kind
        if kind == "primitive":
            return _PRIMITIVE_LAYOUT.get(ty.name or "")
        if kind in ("pointer", "function"):
            return _POINTER_LAYOUT
        if kind == "array":
            if ty.size is None or ty.size < 0:
                # Emitted as ptr[elem].
                return _POINTER_LAYOUT
            element = self.type_layout(ty.element, _visiting)
            if element is None:
                return None
            elem_size, elem_align = element
            return (_align_up(elem_size, elem_align) * ty.size, elem_align)
        if kind == "enum":
            return _ENUM_LAYOUT
        if kind in ("struct", "union"):
            decl = self._records.get(_ident(ty.name)) if ty.name else None
            if decl is None:
                return _PLACEHOLDER_LAYOUT
            return self.record_layout(decl)
        if kind in ("typedef", "named"):
            name = _ident(ty.name) if ty.name else None
            if name is None or name in _visiting:
                return None
            if name in self._records:
                return self.record_layout(self._records[name])
            underlying = self._typedefs.get(name)
            if underlying is None:
                return None
            return self.type_layout(underlying, _visiting | {name})
        return None

    def record_layout(self, decl: CDeclIR) -> tuple[int, int]:
        """(size, alignment) of the pythoc type emitted for a record.
        Opaque records are emitted as array[u8, N] with alignment 1."""
        key = id(decl)
        layout = self._record_layouts.get(key)
        if layout is None:
            if not self.record_is_accurate(decl):
                layout = (_storage_size(decl), 1)
                self._record_layouts[key] = layout
            else:
                layout = self._record_layouts[key]
        return layout

    def record_is_accurate(self, decl: CDeclIR) -> bool:
        """True when the emitted field layout matches the C layout."""
        key = id(decl)
        cached = self._record_accurate.get(key)
        if cached is None:
            cached = self._check_record(decl)
            self._record_accurate[key] = cached
        return cached

    def _check_record(self, decl: CDeclIR) -> bool:
        if decl.size_bytes is None:
            # Incomplete declaration: no layout to verify against.
            return False
        offset = 0
        max_align = 1
        max_member_size = 0
        for field in decl.fields:
            if field.bit_width is not None:
                return False
            field_layout = self.type_layout(field.type)
            if field_layout is None:
                return False
            field_size, field_align = field_layout
            if decl.kind == "union":
                if field.offset_bytes not in (None, 0):
                    return False
                max_member_size = max(max_member_size, field_size)
            else:
                offset = _align_up(offset, field_align)
                if field.offset_bytes != offset:
                    return False
                offset += field_size
            max_align = max(max_align, field_align)
        if decl.kind == "union":
            size = _align_up(max_member_size, max_align)
        else:
            size = _align_up(offset, max_align)
        self._record_layouts[id(decl)] = (size, max_align)
        return size == decl.size_bytes


def _is_opaque_record(decl: CDeclIR, layout: _LayoutModel | None = None) -> bool:
    """Records whose layout cannot be expressed as pythoc fields (bitfields,
    unsupported field types, or a C layout pythoc's natural struct layout
    does not reproduce) degrade to opaque storage of matching size: the
    type stays layout-compatible for ptr[Name] use but real fields are
    not accessible."""
    if any(
        field.bit_width is not None or _unsupported_reason(field.type) is not None
        for field in decl.fields
    ):
        return True
    if layout is not None and not layout.record_is_accurate(decl):
        return True
    return False


def _storage_size(decl: CDeclIR) -> int:
    if decl.size_bytes is not None and decl.size_bytes > 0:
        return decl.size_bytes
    return 1


def _typedef_aliased_tag(decl: CDeclIR,
                         typedefs: dict[str, CDeclIR]) -> tuple[str, str] | None:
    """If a typedef resolves (through other typedefs) to a tag type, return
    (kind, ident) of that tag, else None."""
    ty = decl.type
    seen = set()
    while ty is not None and ty.kind == "typedef" and ty.name:
        name = _ident(ty.name)
        if name in seen or name not in typedefs:
            return None
        seen.add(name)
        ty = typedefs[name].type
    if ty is not None and ty.kind in _TAG_KINDS and ty.name:
        return (ty.kind, _ident(ty.name))
    return None


def _tag_renames(declarations: list[CDeclIR]) -> dict[str, str]:
    """Map colliding tag type idents to their disambiguated emitted names.

    C keeps tag names (struct/union/enum) in a namespace separate from
    ordinary identifiers (typedefs, functions, variables, enum constants),
    so `struct Foo { ... }; typedef int Foo;` is legal.  The generated
    module has a single flat namespace, so a colliding tag is emitted under
    a kind-prefixed name (struct_Foo / union_Foo / enum_Foo) while the
    ordinary identifier keeps the plain name.  A typedef aliasing its own
    tag (`typedef struct Foo Foo;`) names the same type and is not a
    collision.
    """
    typedefs = {
        _ident(decl.name): decl
        for decl in declarations if decl.kind == "typedef"
    }
    tags: dict[str, str] = {}  # ident -> tag kind
    for decl in declarations:
        if decl.kind in _TAG_KINDS:
            tags.setdefault(_ident(decl.name), decl.kind)
    ordinary: set[str] = set()
    for decl in declarations:
        if decl.kind in ("function", "var"):
            ordinary.add(_ident(decl.name))
        elif decl.kind == "enum":
            ordinary.update(_ident(value.name) for value in decl.values)
        elif decl.kind == "typedef":
            ident = _ident(decl.name)
            if ident in tags and _typedef_aliased_tag(decl, typedefs) == (
                    tags[ident], ident):
                continue
            ordinary.add(ident)
    taken = set(tags) | ordinary
    renames: dict[str, str] = {}
    for ident in sorted(ordinary & set(tags)):
        candidate = _ident(f"{tags[ident]}_{ident}")
        while candidate in taken:
            candidate = "_" + candidate
        taken.add(candidate)
        renames[ident] = candidate
    return renames


def _emit_header(lines: list[str]) -> None:
    lines.extend(
        [
            '"""Auto-generated pythoc bindings"""',
            "",
            "from pythoc import (",
            "    compile, extern, extern_global, enum, i8, i16, i32, i64,",
            "    u8, u16, u32, u64, f32, f64, f128, bool, ptr, array,",
            "    void, char, nullptr, sizeof, struct, union, func",
            ")",
            "",
        ]
    )


def _emit_opaque_record(lines: list[str], name: str, decl: CDeclIR) -> None:
    lines.append("@compile")
    lines.append(f"class {name}:")
    lines.append(f"    _storage: array[u8, {_storage_size(decl)}]")
    lines.append("")


def _emit_struct(lines: list[str], decl: CDeclIR,
                 renames: dict[str, str] | None = None,
                 layout: _LayoutModel | None = None) -> None:
    name = _decl_ident(decl, renames, "_AnonymousStruct")
    if _is_opaque_record(decl, layout):
        _emit_opaque_record(lines, name, decl)
        return
    lines.append("@compile")
    lines.append(f"class {name}:")
    if not decl.fields:
        lines.append("    pass")
    else:
        for index, field in enumerate(decl.fields):
            field_name = _ident(field.name, f"_field{index}")
            # Quote the type expression: C identifiers like __foo are
            # subject to Python's class-body name mangling when evaluated
            # directly, while a string annotation is resolved later by
            # pythoc against the module namespace (which also keeps
            # self-references working).  The field target itself is
            # un-mangled by pythoc's struct compilation.
            lines.append(
                f"    {field_name}: {_type_expr(field.type, renames)!r}")
    lines.append("")


def _emit_union(lines: list[str], decl: CDeclIR,
                renames: dict[str, str] | None = None,
                layout: _LayoutModel | None = None) -> None:
    name = _decl_ident(decl, renames, "_AnonymousUnion")
    if _is_opaque_record(decl, layout):
        _emit_opaque_record(lines, name, decl)
        return
    if not decl.fields:
        lines.append(f"{name} = union[]")
        lines.append("")
        return

    fields = []
    for index, field in enumerate(decl.fields):
        field_name = _ident(field.name, f"_field{index}")
        # Quote the field name: inside a subscript, name: type parses as a
        # slice whose start expression must be defined; a string literal is
        # self-contained and resolves to the same named-field form.
        fields.append(f"{field_name!r}: {_type_expr(field.type, renames)}")
    lines.append(f"{name} = union[" + ", ".join(fields) + "]")
    lines.append("")


def _emit_enum(lines: list[str], decl: CDeclIR,
               renames: dict[str, str] | None = None) -> None:
    name = _decl_ident(decl, renames, "_AnonymousEnum")
    lines.append("@enum(i32)")
    lines.append(f"class {name}:")
    if not decl.values:
        lines.append("    pass")
    else:
        for value in decl.values:
            value_name = _ident(value.name, "_value")
            if value.value is None:
                lines.append(f"    {value_name}: None")
            else:
                lines.append(f"    {value_name} = {value.value}")
    lines.append("")
    # C enum constants live in the enclosing scope, not inside the enum type
    for value in decl.values:
        if value.value is not None:
            lines.append(f"{_ident(value.name, '_value')} = {value.value}")
    if decl.values:
        lines.append("")


def _emit_function(lines: list[str], decl: CDeclIR, lib: str,
                   wrapper_symbol: str | None = None,
                   renames: dict[str, str] | None = None) -> None:
    ty = decl.type
    params = ty.params if ty else []
    ret = ty.return_type if ty else None

    if wrapper_symbol is not None:
        # The wrapper symbol is provided by a cc-compiled stub object
        # registered for linking, not by the user's library.
        lines.append(f"@extern(lib='', name={_quote(wrapper_symbol)})")
    elif decl.symbol:
        # __asm__ label: the binding keeps the C name, the linked symbol
        # uses the label (glibc __REDIRECT scenario).
        lines.append(f"@extern(lib={_quote(lib)}, name={_quote(decl.symbol)})")
    else:
        lines.append(f"@extern(lib={_quote(lib)})")
    rendered_params = []
    for index, param in enumerate(params):
        rendered_params.append(_format_param(index, param, renames))
    if ty and ty.is_variadic:
        rendered_params.append("*args")
    lines.append(
        f"def {_ident(decl.name, '_func')}("
        + ", ".join(rendered_params)
        + f") -> {_type_expr(ret, renames)}:"
    )
    lines.append("    pass")
    lines.append("")


def _format_param(index: int, param: CParamIR,
                  renames: dict[str, str] | None = None) -> str:
    return f"{_ident(param.name, f'arg{index}')}: {_type_expr(param.type, renames)}"


def _emit_typedef(lines: list[str], decl: CDeclIR,
                  renames: dict[str, str] | None = None) -> None:
    lines.append(f"{_ident(decl.name, '_typedef')} = {_type_expr(decl.type, renames)}")
    lines.append("")


def _emit_var(lines: list[str], decl: CDeclIR, lib: str,
              renames: dict[str, str] | None = None) -> None:
    name = _ident(decl.name, "_var")
    lines.append(
        f"{name} = extern_global({_type_expr(decl.type, renames)}, "
        f"name={_quote(decl.symbol or decl.name)}, lib={_quote(lib)})"
    )
    lines.append("")


def emit_pythoc_module(module: CModuleIR, lib: str,
                       wrapper_prefix: str | None = None) -> str:
    lines: list[str] = []
    _emit_header(lines)

    # C tag vs ordinary identifier namespaces: colliding tag types are
    # emitted under kind-prefixed names, and all references to them in
    # field/parameter/return types follow the rename.
    renames = _tag_renames(module.declarations)

    # Functions defined in the imported file without an external symbol
    # (static / C99 inline) bind to their generated wrapper symbol instead.
    wrapped: dict[str, str] = {}
    if wrapper_prefix is not None:
        from .cimport_wrappers import collect_wrappers
        wrapped = collect_wrappers(module, wrapper_prefix)

    layout = _LayoutModel(module.declarations)

    # Classify declarations before emitting anything: unsupported ones
    # become lazy errors via _unsupported_symbols, and declarations whose
    # types reference an unsupported name must degrade the same way so the
    # generated module never contains a dangling name.
    unsupported: dict[str, str] = {}
    emitted: list[CDeclIR] = []
    for decl in module.declarations:
        if decl.kind == "macro":
            emitted.append(decl)
            continue
        reason = _decl_unsupported_reason(decl, wrapped.get(decl.name))
        if reason is None:
            emitted.append(decl)
        else:
            unsupported[_decl_ident(decl, renames)] = reason

    changed = True
    while changed:
        changed = False
        for decl in list(emitted):
            if decl.kind == "macro":
                continue
            bad = sorted(
                name for name in _decl_referenced_names(decl, renames)
                if name in unsupported
            )
            if bad:
                emitted.remove(decl)
                unsupported[_decl_ident(decl, renames)] = (
                    f"references unsupported type '{bad[0]}': "
                    f"{unsupported[bad[0]]}"
                )
                changed = True

    # Declarations win over macros on name collision.
    declared_names: set[str] = set()
    for decl in module.declarations:
        if decl.kind == "macro":
            continue
        declared_names.add(_decl_ident(decl, renames))
        if decl.kind == "enum":
            declared_names.update(_ident(value.name) for value in decl.values)

    # Named types referenced by emitted declarations but defined only in
    # included headers (never part of the main-file declaration list) get
    # an opaque placeholder so the module imports cleanly and ptr[Name]
    # uses work; by-value use of such a type is not ABI-correct.
    defined_names = {
        _decl_ident(decl, renames)
        for decl in emitted
        if decl.kind in ("struct", "union", "enum", "typedef")
    }
    referenced_names: set[str] = set()
    for decl in emitted:
        referenced_names.update(_decl_referenced_names(decl, renames))
    placeholders = sorted(
        referenced_names - defined_names - set(unsupported) - declared_names
    )
    declared_names.update(placeholders)
    for name in placeholders:
        lines.append("@compile")
        lines.append(f"class {name}:")
        lines.append("    _storage: array[u8, 1]")
        lines.append("")

    for decl in emitted:
        if decl.kind == "struct":
            _emit_struct(lines, decl, renames, layout)
        elif decl.kind == "union":
            _emit_union(lines, decl, renames, layout)
        elif decl.kind == "enum":
            _emit_enum(lines, decl, renames)
        elif decl.kind == "function":
            _emit_function(lines, decl, lib,
                           wrapper_symbol=wrapped.get(decl.name),
                           renames=renames)
        elif decl.kind == "typedef":
            _emit_typedef(lines, decl, renames)
        elif decl.kind == "var":
            _emit_var(lines, decl, lib, renames)
        elif decl.kind == "macro":
            name = _ident(decl.name)
            if name not in declared_names:
                lines.append(f"{name} = {decl.value!r}")

    if unsupported:
        lines.append("")
        lines.append(f"_unsupported_symbols = {unsupported!r}")
        lines.append("")
        lines.append("def __getattr__(name):")
        lines.append("    reason = _unsupported_symbols.get(name)")
        lines.append("    if reason is not None:")
        lines.append("        raise RuntimeError(")
        lines.append("            'cimport symbol %s is not supported: %s' % (name, reason))")
        lines.append("    raise AttributeError(")
        lines.append("        'module %r has no attribute %r' % (__name__, name))")

    return "\n".join(lines).rstrip() + "\n"


def write_pythoc_module(module: CModuleIR, lib: str, output_path: str,
                        wrapper_prefix: str | None = None) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(emit_pythoc_module(module, lib, wrapper_prefix=wrapper_prefix))
