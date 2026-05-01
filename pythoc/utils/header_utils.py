"""C header generation for exported PythoC functions."""

import hashlib
import os
import re
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


_C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default",
    "do", "double", "else", "enum", "extern", "float", "for", "goto",
    "if", "inline", "int", "long", "register", "restrict", "return",
    "short", "signed", "sizeof", "static", "struct", "switch",
    "typedef", "union", "unsigned", "void", "volatile", "while",
    "_Bool", "_Complex", "_Imaginary",
}


_INTEGER_TYPES = {
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
}


_FLOAT_TYPES = {
    "f16": "_Float16",
    "bf16": "uint16_t",
    "f32": "float",
    "f64": "double",
    "f128": "__float128",
}


def export_c_header(symbols: Sequence[Any], output_path: str) -> str:
    """Write a C header for compiled function symbols."""
    function_infos = _collect_function_infos(symbols)
    collector = _AggregateCollector()
    prototypes = [
        _render_function_prototype(func_info, collector)
        for func_info in function_infos
    ]

    guard = _header_guard(output_path)
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdbool.h>",
        "#include <stdint.h>",
        "",
        "#ifdef __cplusplus",
        "extern \"C\" {",
        "#endif",
        "",
    ]

    aggregate_lines = collector.render_definitions()
    if aggregate_lines:
        lines.extend(aggregate_lines)
        lines.append("")

    lines.extend(prototypes)
    lines.extend([
        "",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        f"#endif /* {guard} */",
        "",
    ])

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines))
    return output_path


def _collect_function_infos(symbols: Sequence[Any]) -> List[Any]:
    result = []
    seen: Set[str] = set()
    for symbol in symbols:
        func_info = getattr(symbol, "_func_info", symbol)
        if not hasattr(func_info, "param_names"):
            raise TypeError(f"Cannot export non-compiled symbol: {symbol!r}")
        name = _function_symbol_name(func_info)
        if name not in seen:
            seen.add(name)
            result.append(func_info)
    if not result:
        raise RuntimeError("No compiled symbols selected for C header export.")
    return result


def _render_function_prototype(func_info: Any, collector: "_AggregateCollector") -> str:
    name = _function_symbol_name(func_info)
    return_type = getattr(func_info, "return_type_hint", None)
    if _is_omitted_c_type(return_type):
        ret_decl = "void"
    else:
        ret_decl = _declare_type(return_type, "", collector).strip()

    params = []
    for param_name in getattr(func_info, "param_names", []):
        param_type = func_info.param_type_hints[param_name]
        if _is_omitted_c_type(param_type):
            continue
        params.append(_declare_type(param_type, _c_identifier(param_name), collector))

    if getattr(func_info, "has_llvm_varargs", False):
        params.append("...")
    if not params:
        params.append("void")

    return f"{ret_decl} {_c_identifier(name)}({', '.join(params)});"


def _function_symbol_name(func_info: Any) -> str:
    binding = getattr(func_info, "binding_state", None)
    if binding is not None:
        actual = getattr(binding, "actual_func_name", None)
        if actual:
            return actual
    mangled = getattr(func_info, "mangled_name", None)
    return mangled or func_info.name


def _declare_type(pc_type: Any, declarator: str,
                  collector: "_AggregateCollector") -> str:
    qualifiers, pc_type = _strip_qualifiers(pc_type)
    declaration = _declare_unqualified_type(pc_type, declarator, collector)
    if qualifiers:
        declaration = " ".join(qualifiers) + " " + declaration
    return declaration


def _declare_unqualified_type(pc_type: Any, declarator: str,
                              collector: "_AggregateCollector") -> str:
    if pc_type is None:
        return _join_decl("void", declarator)

    if _is_refined_type(pc_type):
        return _declare_type(_refined_export_type(pc_type), declarator, collector)

    if _is_omitted_c_type(pc_type):
        return _join_decl("void", declarator)

    if _is_func_type(pc_type):
        return _declare_function_pointer(pc_type, declarator, collector)

    if _is_pointer_type(pc_type):
        pointee_type = getattr(pc_type, "pointee_type", None)
        if pointee_type is None:
            pointee_type = _void_type()
        return _declare_type(pointee_type, _pointer_declarator(declarator), collector)

    if _is_array_type(pc_type):
        element_type = getattr(pc_type, "element_type", None)
        dimensions = getattr(pc_type, "dimensions", None) or ()
        array_decl = _array_declarator(declarator, dimensions)
        return _declare_type(element_type, array_decl, collector)

    if _is_aggregate_type(pc_type):
        type_name = collector.add(pc_type)
        return _join_decl(type_name, declarator)

    type_name = _primitive_c_type(pc_type)
    if type_name is not None:
        return _join_decl(type_name, declarator)

    if isinstance(pc_type, str):
        return _join_decl(_c_identifier(pc_type), declarator)

    raise TypeError(f"Cannot export unsupported C type: {pc_type!r}")


def _strip_qualifiers(pc_type: Any) -> Tuple[List[str], Any]:
    qualifiers = []
    current = pc_type
    while isinstance(current, type) and hasattr(current, "qualified_type"):
        flags = current.get_qualifier_flags()
        if flags.get("const") and "const" not in qualifiers:
            qualifiers.append("const")
        if flags.get("volatile") and "volatile" not in qualifiers:
            qualifiers.append("volatile")
        current = current.qualified_type
    return qualifiers, current


def _declare_function_pointer(pc_type: Any, declarator: str,
                              collector: "_AggregateCollector") -> str:
    if getattr(pc_type, "return_type", None) is None:
        ret_decl = "void"
    elif _is_omitted_c_type(pc_type.return_type):
        ret_decl = "void"
    else:
        ret_decl = _declare_type(pc_type.return_type, "", collector).strip()

    params = []
    for param_type in getattr(pc_type, "param_types", None) or ():
        if _is_omitted_c_type(param_type):
            continue
        params.append(_declare_type(param_type, "", collector).strip())

    if getattr(pc_type, "has_llvm_varargs", False):
        params.append("...")
    if not params:
        params.append("void")

    return f"{ret_decl} (*{declarator})({', '.join(params)})"


def _primitive_c_type(pc_type: Any) -> str:
    name = _pc_type_name(pc_type)
    if name in _INTEGER_TYPES:
        return _INTEGER_TYPES[name]
    if name in _FLOAT_TYPES:
        return _FLOAT_TYPES[name]
    if name == "bool":
        return "bool"
    if name == "void":
        return "void"

    match = re.fullmatch(r"([iu])(\d+)", name or "")
    if match:
        prefix, width_text = match.groups()
        width = int(width_text)
        if width <= 8:
            storage_width = 8
        elif width <= 16:
            storage_width = 16
        elif width <= 32:
            storage_width = 32
        elif width <= 64:
            storage_width = 64
        else:
            raise TypeError(f"Cannot export integer type wider than 64 bits: {name}")
        return ("int" if prefix == "i" else "uint") + f"{storage_width}_t"

    return None


def _pc_type_name(pc_type: Any) -> str:
    if hasattr(pc_type, "get_name"):
        return pc_type.get_name()
    return getattr(pc_type, "__name__", str(pc_type))


def _is_pointer_type(pc_type: Any) -> bool:
    return bool(getattr(pc_type, "_is_pointer", False))


def _is_func_type(pc_type: Any) -> bool:
    return any(cls.__name__ == "func" for cls in getattr(pc_type, "__mro__", ()))


def _is_refined_type(pc_type: Any) -> bool:
    return bool(getattr(pc_type, "_is_refined", False))


def _refined_export_type(pc_type: Any) -> Any:
    base_type = getattr(pc_type, "_base_type", None)
    if base_type is not None:
        return base_type
    struct_type = getattr(pc_type, "_struct_type", None)
    if struct_type is not None:
        return struct_type
    param_types = getattr(pc_type, "_param_types", None)
    if param_types:
        return param_types[0]
    raise TypeError(f"Cannot export refined type without base type: {pc_type!r}")


def _is_omitted_c_type(pc_type: Any) -> bool:
    if pc_type is None:
        return False
    if _is_refined_type(pc_type):
        return _is_omitted_c_type(_refined_export_type(pc_type))
    if _is_func_type(pc_type):
        return False
    get_size = getattr(pc_type, "get_size_bytes", None)
    return bool(callable(get_size) and get_size() == 0 and _pc_type_name(pc_type) != "void")


def _is_array_type(pc_type: Any) -> bool:
    is_array = getattr(pc_type, "is_array", None)
    return bool(is_array() if callable(is_array) else False)


def _is_aggregate_type(pc_type: Any) -> bool:
    return _is_struct_type(pc_type) or _is_union_type(pc_type) or _is_enum_type(pc_type)


def _is_struct_type(pc_type: Any) -> bool:
    if getattr(pc_type, "_is_union", False):
        return False
    if getattr(pc_type, "_is_struct", False):
        return True
    is_struct = getattr(pc_type, "is_struct_type", None)
    return bool(is_struct() if callable(is_struct) else False)


def _is_enum_type(pc_type: Any) -> bool:
    is_enum = getattr(pc_type, "is_enum_type", None)
    return bool(is_enum() if callable(is_enum) else False)


def _is_union_type(pc_type: Any) -> bool:
    if getattr(pc_type, "_is_union", False):
        return True
    return any(cls.__name__ == "UnionType" for cls in getattr(pc_type, "__mro__", ()))


def _void_type() -> Any:
    from ..builtin_entities.types import void
    return void


def _join_decl(base: str, declarator: str) -> str:
    if declarator:
        return f"{base} {declarator}"
    return base


def _pointer_declarator(declarator: str) -> str:
    if not declarator:
        return "*"
    if declarator.startswith("["):
        return f"(*{declarator})"
    return f"*{declarator}"


def _array_declarator(declarator: str, dimensions: Iterable[Any]) -> str:
    if declarator.startswith("*"):
        declarator = f"({declarator})"
    result = declarator
    for dim in dimensions:
        result += f"[{int(dim)}]"
    return result


def _c_identifier(name: Any) -> str:
    text = str(name)
    text = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if not text:
        text = "pc_symbol"
    if text[0].isdigit() or text in _C_KEYWORDS:
        text = "pc_" + text
    return text


def _header_guard(output_path: str) -> str:
    base = os.path.basename(output_path)
    guard = _c_identifier(base).upper()
    return f"PYTHOC_{guard}_INCLUDED"


class _AggregateCollector:
    def __init__(self):
        self._types: List[Any] = []
        self._names: Dict[Any, str] = {}
        self._used_names: Set[str] = set()

    def add(self, pc_type: Any) -> str:
        if pc_type in self._names:
            return self._names[pc_type]

        base_name = _aggregate_base_name(pc_type)
        name = base_name
        if name in self._used_names:
            digest = hashlib.sha1(repr(pc_type).encode("utf-8")).hexdigest()[:8]
            name = f"{base_name}_{digest}"

        self._names[pc_type] = name
        self._used_names.add(name)
        self._types.append(pc_type)
        return name

    def render_definitions(self) -> List[str]:
        rendered: Set[Any] = set()
        lines: List[str] = []

        idx = 0
        while idx < len(self._types):
            self._render_one(self._types[idx], rendered, lines, set())
            idx += 1

        if not self._types:
            return []

        forward_lines = [
            f"typedef {_aggregate_tag_keyword(pc_type)} {self._names[pc_type]} "
            f"{self._names[pc_type]};"
            for pc_type in self._types
        ]
        return forward_lines + [""] + lines

    def _render_one(self, pc_type: Any, rendered: Set[Any], lines: List[str],
                    visiting: Set[Any]):
        if pc_type in rendered:
            return
        if pc_type in visiting:
            return
        visiting.add(pc_type)

        for field_type in _aggregate_value_field_types(pc_type):
            _, field_type = _strip_qualifiers(field_type)
            for dep_type in _aggregate_value_dependencies(field_type):
                self.add(dep_type)
                self._render_one(dep_type, rendered, lines, visiting)

        name = self.add(pc_type)
        fields = _aggregate_field_declarations(pc_type, self)
        lines.append(f"{_aggregate_tag_keyword(pc_type)} {name} {{")
        if fields:
            for field in fields:
                lines.append(f"    {field};")
        else:
            lines.append("    uint8_t _empty;")
        lines.append("};")
        lines.append("")
        rendered.add(pc_type)
        visiting.remove(pc_type)


def _aggregate_base_name(pc_type: Any) -> str:
    python_class = getattr(pc_type, "_python_class", None)
    if python_class is not None:
        raw_name = python_class.__name__
    elif getattr(pc_type, "_is_struct", False) or getattr(pc_type, "_is_union", False):
        raw_name = pc_type.__name__
    elif hasattr(pc_type, "_struct_info") and pc_type._struct_info is not None:
        raw_name = pc_type._struct_info.name
    else:
        raw_name = _pc_type_name(pc_type)
    return _c_identifier(raw_name)


def _aggregate_tag_keyword(pc_type: Any) -> str:
    if _is_union_type(pc_type):
        return "union"
    return "struct"


def _aggregate_value_field_types(pc_type: Any) -> List[Any]:
    if _is_struct_type(pc_type) or _is_union_type(pc_type):
        _ensure_fields_resolved(pc_type)
        return list(getattr(pc_type, "_field_types", None) or [])
    if _is_enum_type(pc_type):
        fields = []
        tag_type = getattr(pc_type, "_tag_type", None)
        payload_type = getattr(pc_type, "_union_payload", None)
        if tag_type is not None:
            fields.append(tag_type)
        if payload_type is not None:
            fields.append(payload_type)
        return fields
    return []


def _aggregate_value_dependencies(pc_type: Any) -> List[Any]:
    if _is_array_type(pc_type):
        return _aggregate_value_dependencies(getattr(pc_type, "element_type", None))
    if _is_aggregate_type(pc_type):
        return [pc_type]
    return []


def _aggregate_field_declarations(pc_type: Any,
                                  collector: _AggregateCollector) -> List[str]:
    if _is_struct_type(pc_type):
        _ensure_fields_resolved(pc_type)
        field_types = getattr(pc_type, "_field_types", None) or []
        field_names = getattr(pc_type, "_field_names", None) or []
        result = []
        for idx, field_type in enumerate(field_types):
            if _is_omitted_c_type(field_type):
                continue
            field_name = field_names[idx] if idx < len(field_names) else None
            if not field_name:
                field_name = f"field_{idx}"
            result.append(_declare_type(field_type, _c_identifier(field_name), collector))
        return result

    if _is_enum_type(pc_type):
        tag_type = getattr(pc_type, "_tag_type", None)
        payload_type = getattr(pc_type, "_union_payload", None)
        result = []
        if tag_type is not None and not _is_omitted_c_type(tag_type):
            result.append(_declare_type(tag_type, "tag", collector))
        if payload_type is not None and not _is_omitted_c_type(payload_type):
            result.append(_declare_type(payload_type, "payload", collector))
        return result

    if _is_union_type(pc_type):
        _ensure_fields_resolved(pc_type)
        field_types = getattr(pc_type, "_field_types", None) or []
        field_names = getattr(pc_type, "_field_names", None) or []
        result = []
        for idx, field_type in enumerate(field_types):
            if _is_omitted_c_type(field_type):
                continue
            field_name = field_names[idx] if idx < len(field_names) else None
            if not field_name:
                field_name = f"field_{idx}"
            result.append(_declare_type(field_type, _c_identifier(field_name), collector))
        return result

    return []


def _ensure_fields_resolved(pc_type: Any):
    resolver = getattr(pc_type, "_ensure_field_types_resolved", None)
    if callable(resolver):
        resolver()
