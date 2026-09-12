from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class RuntimeCapture:
    name: str
    pc_type: type
    value_ref: Optional[Any] = None


@dataclass(frozen=True)
class ClosureCapturePlan:
    bindings: Dict[str, Any]
    runtime: List[RuntimeCapture]


def extract_value_from_vref(vref) -> Any:
    th = getattr(vref, "type_hint", None)
    if th is not None:
        if hasattr(th, "is_constant") and th.is_constant():
            return th.get_constant_value()
        if hasattr(th, "get_python_object"):
            return th.get_python_object()
    return vref.get_python_value()


def build_closure_capture_plan(
    captured_vars: Iterable[str],
    visible: Mapping[str, Any],
) -> ClosureCapturePlan:
    bindings: Dict[str, Any] = {}
    runtime: List[RuntimeCapture] = []

    for var_name in captured_vars:
        vinfo = visible.get(var_name)
        if vinfo is None or vinfo.value_ref is None:
            continue

        vref = vinfo.value_ref
        if vref.is_python_value():
            try:
                bindings[var_name] = extract_value_from_vref(vref)
            except TypeError:
                pass
            continue

        if vref.is_pcvalue():
            pc_type = vref.get_pc_type()
            if pc_type is not None:
                runtime.append(RuntimeCapture(var_name, pc_type, vref))

    return ClosureCapturePlan(bindings=bindings, runtime=runtime)


def normalize_runtime_captures(captures) -> List[RuntimeCapture]:
    result: List[RuntimeCapture] = []
    for item in captures or []:
        if isinstance(item, RuntimeCapture):
            result.append(item)
            continue
        if len(item) == 2:
            name, pc_type = item
            result.append(RuntimeCapture(name, pc_type, None))
            continue
        if len(item) == 3:
            name, pc_type, value_ref = item
            result.append(RuntimeCapture(name, pc_type, value_ref))
            continue
        raise TypeError("invalid runtime capture descriptor")
    return result
