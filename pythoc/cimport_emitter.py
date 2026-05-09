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


def _type_expr(ty: CTypeIR | None) -> str:
    if ty is None:
        return "void"

    kind = ty.kind
    if kind == "primitive":
        return ty.name or "i32"
    if kind == "named":
        return _ident(ty.name, "i32")
    if kind == "pointer":
        return f"ptr[{_type_expr(ty.pointee)}]"
    if kind == "array":
        elem = _type_expr(ty.element)
        if ty.size is None or ty.size < 0:
            return f"ptr[{elem}]"
        return f"array[{elem}, {ty.size}]"
    if kind == "function":
        parts = [_type_expr(param.type) for param in ty.params]
        parts.append(_type_expr(ty.return_type))
        return "func[" + ", ".join(parts) + "]"
    if kind in {"struct", "union", "enum", "typedef"}:
        return _ident(ty.name, "i32")
    return "i32"


def _emit_header(lines: list[str]) -> None:
    lines.extend(
        [
            '"""Auto-generated pythoc bindings"""',
            "",
            "from pythoc import (",
            "    compile, extern, enum, i8, i16, i32, i64,",
            "    u8, u16, u32, u64, f32, f64, bool, ptr, array,",
            "    void, char, nullptr, sizeof, struct, union, func",
            ")",
            "",
        ]
    )


def _emit_struct(lines: list[str], decl: CDeclIR) -> None:
    lines.append("@compile")
    lines.append(f"class {_ident(decl.name, '_AnonymousStruct')}:")
    if not decl.fields:
        lines.append("    pass")
    else:
        for index, field in enumerate(decl.fields):
            field_name = _ident(field.name, f"_field{index}")
            lines.append(f"    {field_name}: {_type_expr(field.type)}")
    lines.append("")


def _emit_union(lines: list[str], decl: CDeclIR) -> None:
    name = _ident(decl.name, "_AnonymousUnion")
    if not decl.fields:
        lines.append(f"{name} = union[]")
        lines.append("")
        return

    fields = []
    for index, field in enumerate(decl.fields):
        field_name = _ident(field.name, f"_field{index}")
        fields.append(f"{field_name}: {_type_expr(field.type)}")
    lines.append(f"{name} = union[" + ", ".join(fields) + "]")
    lines.append("")


def _emit_enum(lines: list[str], decl: CDeclIR) -> None:
    lines.append("@enum(i32)")
    lines.append(f"class {_ident(decl.name, '_AnonymousEnum')}:")
    if not decl.values:
        lines.append("    pass")
    else:
        for value in decl.values:
            name = _ident(value.name, "_value")
            if value.value is None:
                lines.append(f"    {name}: None")
            else:
                lines.append(f"    {name} = {value.value}")
    lines.append("")


def _emit_function(lines: list[str], decl: CDeclIR, lib: str) -> None:
    ty = decl.type
    params = ty.params if ty else []
    ret = ty.return_type if ty else None

    lines.append(f"@extern(lib={_quote(lib)})")
    rendered_params = []
    for index, param in enumerate(params):
        rendered_params.append(_format_param(index, param))
    if ty and ty.is_variadic:
        rendered_params.append("*args")
    lines.append(
        f"def {_ident(decl.name, '_func')}("
        + ", ".join(rendered_params)
        + f") -> {_type_expr(ret)}:"
    )
    lines.append("    pass")
    lines.append("")


def _format_param(index: int, param: CParamIR) -> str:
    return f"{_ident(param.name, f'arg{index}')}: {_type_expr(param.type)}"


def _emit_typedef(lines: list[str], decl: CDeclIR) -> None:
    lines.append(f"{_ident(decl.name, '_typedef')} = {_type_expr(decl.type)}")
    lines.append("")


def _emit_var(lines: list[str], decl: CDeclIR) -> None:
    lines.append(f"# {_ident(decl.name, '_var')}: {_type_expr(decl.type)}")


def emit_pythoc_module(module: CModuleIR, lib: str) -> str:
    lines: list[str] = []
    _emit_header(lines)

    for decl in module.declarations:
        if decl.kind == "struct":
            _emit_struct(lines, decl)
        elif decl.kind == "union":
            _emit_union(lines, decl)
        elif decl.kind == "enum":
            _emit_enum(lines, decl)
        elif decl.kind == "function":
            _emit_function(lines, decl, lib)
        elif decl.kind == "typedef":
            _emit_typedef(lines, decl)
        elif decl.kind == "var":
            _emit_var(lines, decl)

    return "\n".join(lines).rstrip() + "\n"


def write_pythoc_module(module: CModuleIR, lib: str, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(emit_pythoc_module(module, lib))
