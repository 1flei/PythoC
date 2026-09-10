"""C wrapper generation for header-defined inline/static functions.

A C function defined in a header as ``static inline`` (or plain ``static``,
or C99 ``inline`` without ``extern``) has no guaranteed external symbol in
any library, so a plain ``@extern`` binding to it is dead.  This module
renders a stub ``.c`` file that includes the original header and defines a
non-static forwarding wrapper per such function:

    int __pythoc_inl_<hash>_add(int a, int b) { return add(a, b); }

The stub is compiled with the system C compiler by the cimport driver and
the resulting object file is registered for linking, so the wrapper symbol
resolves both in JIT group libraries and AOT executables.

The wrapper symbol embeds a short hash of the absolute header path so two
different headers defining an identically named static inline function can
be imported into the same process without symbol collisions.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional

from .cimport_ir import CDeclIR, CModuleIR, CTypeIR


_WRAPPER_PREFIX = "__pythoc_inl_"

_C_PRIMITIVES = {
    "void": "void",
    "bool": "_Bool",
    "char": "char",
    "i8": "signed char",
    "u8": "unsigned char",
    "i16": "short",
    "u16": "unsigned short",
    "i32": "int",
    "u32": "unsigned int",
    "i64": "long long",
    "u64": "unsigned long long",
    "f32": "float",
    "f64": "double",
    "f128": "long double",
}


def wrapper_prefix_for_path(path: str) -> str:
    """Deterministic wrapper symbol prefix for one imported file."""
    digest = hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:8]
    return f"{_WRAPPER_PREFIX}{digest}_"


def stub_paths(cache_dir: str, basename: str) -> tuple[str, str]:
    """Return (stub .c path, stub .o path) inside the cimport cache dir."""
    return (
        os.path.join(cache_dir, f"{basename}_pythoc_inl.c"),
        os.path.join(cache_dir, f"{basename}_pythoc_inl.o"),
    )


def render_c_type(ty: Optional[CTypeIR], declarator: str = "") -> Optional[str]:
    """Render a CTypeIR back to C source text around ``declarator``.

    Returns None when the type cannot be expressed as C (``unsupported``
    types or unknown primitives), in which case no wrapper is generated.
    Array types in parameter position decay to pointers, matching C.
    """
    if ty is None:
        return "void" + (f" {declarator}" if declarator else "")
    kind = ty.kind
    if kind == "primitive":
        c_name = _C_PRIMITIVES.get(ty.name or "")
        if c_name is None:
            return None
        return c_name + (f" {declarator}" if declarator else "")
    if kind in ("struct", "union", "enum"):
        if not ty.name:
            return None
        return f"{kind} {ty.name}" + (f" {declarator}" if declarator else "")
    if kind in ("typedef", "named"):
        if not ty.name:
            return None
        return ty.name + (f" {declarator}" if declarator else "")
    if kind == "pointer":
        inner = "*" + declarator
        if ty.pointee is not None and ty.pointee.kind in ("function", "array"):
            inner = "(" + inner + ")"
        return render_c_type(ty.pointee, inner)
    if kind == "array":
        # Function parameters of array type decay to pointers to the element
        # type (C17 6.7.6.3p7).
        inner = "*" + declarator
        if ty.element is not None and ty.element.kind in ("function", "array"):
            inner = "(" + inner + ")"
        return render_c_type(ty.element, inner)
    if kind == "function":
        params = _render_params(ty, name_params=False)
        if params is None:
            return None
        return render_c_type(ty.return_type, f"{declarator}({params})")
    return None


def _render_params(ty: CTypeIR, name_params: bool) -> Optional[str]:
    parts = []
    for index, param in enumerate(ty.params):
        name = param.name or f"arg{index}"
        declarator = name
        # Function-typed parameters are adjusted to function pointers
        # (C17 6.7.6.3p8).
        if param.type is not None and param.type.kind == "function":
            declarator = f"(*{name})"
        rendered = render_c_type(param.type, declarator if name_params else "")
        if rendered is None:
            return None
        parts.append(rendered)
    return ", ".join(parts) if parts else "void"


def render_wrapper(decl: CDeclIR, symbol: str) -> Optional[str]:
    """Render one forwarding wrapper definition, or None if unrenderable."""
    ty = decl.type
    if ty is None:
        return None
    params = _render_params(ty, name_params=True)
    if params is None:
        return None
    signature = render_c_type(ty.return_type, f"{symbol}({params})")
    if signature is None:
        return None
    arg_names = ", ".join(
        param.name or f"arg{index}" for index, param in enumerate(ty.params)
    )
    call = f"{decl.name}({arg_names})"
    is_void = ty.return_type is None or (
        ty.return_type.kind == "primitive" and ty.return_type.name == "void"
    )
    if is_void:
        body = f"    {call};"
    else:
        body = f"    return {call};"
    return f"{signature}\n{{\n{body}\n}}"


def collect_wrappers(module: CModuleIR, prefix: str) -> Dict[str, str]:
    """Map C function name -> wrapper symbol for every wrappable function.

    A function is wrapped iff it was marked ``needs_wrapper`` by the clang
    frontend, is not variadic (varargs cannot be forwarded through a typed
    wrapper), and its signature can be rendered back to C.
    """
    symbols: Dict[str, str] = {}
    for decl in module.declarations:
        if decl.kind != "function" or not decl.needs_wrapper:
            continue
        if decl.type is None or decl.type.is_variadic:
            continue
        symbol = prefix + decl.name
        if render_wrapper(decl, symbol) is None:
            continue
        symbols[decl.name] = symbol
    return symbols


def render_stub_source(
    module: CModuleIR, header_path: str, prefix: str
) -> Optional[str]:
    """Render the full stub .c source, or None when no wrappers are needed."""
    symbols = collect_wrappers(module, prefix)
    if not symbols:
        return None
    include_path = os.path.abspath(header_path).replace("\\", "/")
    lines = [
        "/* Auto-generated by pythoc cimport: forwarding wrappers for",
        "   header-defined functions that have no external symbol. */",
        f'#include "{include_path}"',
        "",
    ]
    for decl in module.declarations:
        symbol = symbols.get(decl.name)
        if symbol is None:
            continue
        if decl.storage != "static":
            # C99 inline definitions do not emit an external symbol unless
            # the function is also declared with extern in the translation
            # unit (C17 6.7.4p7); the redeclaration forces emission so the
            # wrapper's call always resolves.
            prototype = _render_params(decl.type, name_params=False)
            rendered = render_c_type(decl.type.return_type, f"{decl.name}({prototype})")
            lines.append(f"extern {rendered};")
        lines.append(render_wrapper(decl, symbol))
        lines.append("")
    return "\n".join(lines)
