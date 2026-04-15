"""
Varargs resolution for PC functions.

Two modes:
- *args: T  (annotated) -- compile-time expansion. The caller passes
  individual arguments that are packed into a T value inside the function.
  Works for any T whose field layout is known at compile time (struct,
  union, enum, named structs, array, ...).
- *args     (bare)      -- LLVM C ABI varargs. The caller uses the
  platform va_list mechanism. Access via explicit va_start/va_arg/va_end
  builtins.
"""

import ast
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

from ..schema_protocol import (
    get_field_layout_types,
    get_schema_field_types,
    has_field_layout,
    is_schema_type,
)


@dataclass
class ResolvedVarArgs:
    """Resolved varargs metadata with PC element types."""
    param_name: Optional[str]
    parsed_type: Optional[Any] = None
    element_types: List[Any] = field(default_factory=list)

    @property
    def is_typed(self) -> bool:
        """True when the varargs have a type annotation (compile-time expansion)."""
        return self.parsed_type is not None

    @property
    def has_llvm_varargs(self) -> bool:
        """True only for bare *args -- real C ABI varargs."""
        return self.param_name is not None and not self.is_typed

    # --- Backward compatibility ------------------------------------------
    # Several call-sites still branch on ``kind``.  Keep a derived property
    # so they keep working without a coordinated rewrite.

    @property
    def kind(self) -> str:
        if self.parsed_type is None:
            return "none"
        return "typed"


def _extract_element_types(
    annotation: ast.AST,
    parsed_type: Any,
    type_resolver,
) -> List[Any]:
    """Extract field / element types from the varargs annotation.

    Strategy (in priority order):
    1. Named schema type (e.g. ``Point``) -> field types from schema protocol
    2. Subscript annotation (e.g. ``struct[i32, f64]``) -> resolve each slice item
    3. Array type -> [element_type] * size
    4. Single type fallback
    """
    # 1. Named / registered schema type -- authoritative even if empty
    if is_schema_type(parsed_type):
        result = []
        for ft in get_schema_field_types(parsed_type):
            if isinstance(ft, str):
                resolved = type_resolver.parse_annotation(ft)
                result.append(resolved if resolved is not None else ft)
            else:
                result.append(ft)
        return result

    # 2. Inline subscript (struct[i32, f64], union[i32, f64], ...)
    if isinstance(annotation, ast.Subscript):
        slice_node = annotation.slice
        items = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) else [slice_node]
        result = []
        for item in items:
            elem = type_resolver.parse_annotation(item)
            if elem is not None:
                result.append(elem)
        if result:
            return result

    # 3. field_layout (covers literal carriers, etc.)
    if has_field_layout(parsed_type):
        layout = get_field_layout_types(parsed_type)
        if layout:
            return list(layout)

    # 4. Array type: element_type * count
    elem_type = getattr(parsed_type, 'element_type', None)
    dims = getattr(parsed_type, 'dimensions', None)
    if elem_type is not None and dims and len(dims) == 1:
        return [elem_type] * dims[0]

    # 5. Fallback: single element
    if parsed_type is not None:
        return [parsed_type]

    return []


def resolve_varargs(func_node: ast.FunctionDef, type_resolver) -> ResolvedVarArgs:
    """Resolve *args into PC element types for declaration building.

    Returns a ``ResolvedVarArgs`` whose ``element_types`` lists the
    individual parameter types to expand, or is empty for bare ``*args``.
    """
    if not func_node.args.vararg:
        return ResolvedVarArgs(param_name=None)

    vararg = func_node.args.vararg
    if not vararg.annotation:
        # Bare *args -- C ABI varargs
        return ResolvedVarArgs(param_name=vararg.arg)

    annotation = vararg.annotation
    parsed_type = type_resolver.parse_annotation(annotation)

    element_types = _extract_element_types(annotation, parsed_type, type_resolver)

    return ResolvedVarArgs(
        param_name=vararg.arg,
        parsed_type=parsed_type,
        element_types=element_types,
    )


# ---- Legacy API kept for callers that have not been updated yet ----------

def detect_varargs(
    func_node: ast.FunctionDef, type_resolver,
) -> Tuple[str, Optional[List[Any]], Optional[str]]:
    """Legacy wrapper. Prefer ``resolve_varargs`` in new code."""
    resolved = resolve_varargs(func_node, type_resolver)
    if not resolved.is_typed:
        return ("none", None, resolved.param_name)
    return (resolved.kind, resolved.element_types, resolved.param_name)
