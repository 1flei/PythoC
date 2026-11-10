from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


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


def snapshot_closure_captures(
    references: Mapping[str, Any],
    reference_names: Mapping[str, str],
) -> ClosureCapturePlan:
    """Snapshot resolved bindings for materialization into independent state."""
    bindings: Dict[str, Any] = {}
    runtime: List[RuntimeCapture] = []
    for alias, vinfo in references.items():
        vref = vinfo.value_ref
        if vref.is_python_value():
            bindings[alias] = extract_value_from_vref(vref)
            continue

        if vref.is_pcvalue():
            pc_type = vref.get_pc_type()
            if pc_type is not None:
                captured = vref.clone(
                    var_name=vinfo.name, linear_path=(),
                    linear_owner_id=vinfo.var_id,
                )
                runtime.append(RuntimeCapture(
                    reference_names[alias], pc_type, captured))

    return ClosureCapturePlan(bindings, runtime)


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
