"""Target-aware memory layout for PythoC semantic types."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional, Tuple

from llvmlite import binding, ir
from .config import config


try:
    binding.initialize()
except RuntimeError:
    pass
binding.initialize_native_target()


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


@dataclass(frozen=True)
class TypeLayout:
    """Memory layout of a PythoC type on one target."""

    storage_size: int
    allocation_size: int
    alignment: int
    stride: int
    bit_width: Optional[int] = None
    field_offsets: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.storage_size < 0 or self.allocation_size < 0:
            raise ValueError("type layout sizes must be non-negative")
        if self.storage_size > self.allocation_size:
            raise ValueError("storage size cannot exceed allocation size")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("type layout alignment must be a power of two")
        if self.stride < self.allocation_size:
            raise ValueError(
                "type layout stride cannot be smaller than allocation size")
        if self.allocation_size % self.alignment:
            raise ValueError("allocation size must preserve type alignment")
        if self.stride % self.alignment:
            raise ValueError("type layout stride must preserve type alignment")


def default_target_triple() -> str:
    triple = binding.get_default_triple()
    if sys.platform == "win32":
        return triple.replace("-windows-msvc", "-windows-gnu")
    return triple


def _resolve_target_cpu(value=None):
    cpu = config.target_cpu if value is None else value
    if cpu in ("native", "host"):
        return binding.get_host_cpu_name()
    if cpu in (None, "", "generic"):
        return ""
    return cpu


def _resolve_target_features(value=None):
    features = config.target_features if value is None else value
    if features in ("native", "host"):
        return binding.get_host_cpu_features().flatten()
    if features in (None, "", "generic"):
        return ""
    return features


def target_machine_identity():
    """Return the resolved CPU and features used for layout and emission."""
    return _resolve_target_cpu(), _resolve_target_features()


def _target_machine_cache_key(triple):
    return (triple, *target_machine_identity())


def _create_target_machine(target, **kwargs):
    cpu, features = target_machine_identity()
    return target.create_target_machine(cpu=cpu, features=features, **kwargs)


@lru_cache(maxsize=None)
def _get_target_machine(triple, cpu, features):
    target = binding.Target.from_triple(triple)
    return target.create_target_machine(cpu=cpu, features=features)


def get_target_machine(triple: Optional[str] = None):
    return _get_target_machine(
        *_target_machine_cache_key(triple or default_target_triple()))


def _llvm_context(llvm_type):
    """Find the defining context without expanding recursive named types."""
    if isinstance(llvm_type, ir.IdentifiedStructType):
        return llvm_type.context
    if isinstance(llvm_type, (ir.ArrayType, ir.VectorType)):
        return _llvm_context(llvm_type.element)
    if isinstance(llvm_type, ir.PointerType) and not llvm_type.is_opaque:
        return _llvm_context(llvm_type.pointee)
    if isinstance(llvm_type, ir.LiteralStructType):
        for element in llvm_type.elements:
            context = _llvm_context(element)
            if context is not None:
                return context
    return None


class TargetLayout:
    """Compose PythoC layouts from target primitive layout facts."""

    def __init__(self, triple: Optional[str] = None, *, target_data=None):
        self.triple = triple or default_target_triple()
        self.target_data = (
            get_target_machine(self.triple).target_data
            if target_data is None else target_data
        )

    def primitive(
        self,
        llvm_type: ir.Type,
        *,
        storage_size: Optional[int] = None,
        bit_width: Optional[int] = None,
    ) -> TypeLayout:
        if isinstance(llvm_type, ir.VoidType):
            return TypeLayout(0, 0, 1, 0, bit_width=0)

        context = _llvm_context(llvm_type)
        if context is None:
            return self._cached_primitive(llvm_type, storage_size, bit_width)
        return self._primitive(llvm_type, storage_size, bit_width, context)

    @lru_cache(maxsize=1024)
    def _cached_primitive(self, llvm_type, storage_size, bit_width):
        return self._primitive(llvm_type, storage_size, bit_width, None)

    def _primitive(self, llvm_type, storage_size, bit_width, context):
        allocation_size = llvm_type.get_abi_size(self.target_data, context)
        alignment = llvm_type.get_abi_alignment(self.target_data, context)
        stored = allocation_size if storage_size is None else storage_size
        return TypeLayout(
            storage_size=stored,
            allocation_size=allocation_size,
            alignment=alignment,
            stride=align_up(allocation_size, alignment),
            bit_width=bit_width,
        )

    def array(
        self,
        element: TypeLayout,
        dimensions: Iterable[int],
    ) -> TypeLayout:
        result = element
        for count in reversed(tuple(dimensions)):
            allocation_size = result.stride * count
            result = TypeLayout(
                storage_size=allocation_size,
                allocation_size=allocation_size,
                alignment=result.alignment,
                stride=allocation_size,
            )
        return result

    def struct(self, fields: Iterable[TypeLayout]) -> TypeLayout:
        offsets = []
        end = 0
        alignment = 1
        for field in fields:
            end = align_up(end, field.alignment)
            offsets.append(end)
            end += field.allocation_size
            alignment = max(alignment, field.alignment)

        allocation_size = align_up(end, alignment)
        return TypeLayout(
            storage_size=end,
            allocation_size=allocation_size,
            alignment=alignment,
            stride=allocation_size,
            field_offsets=tuple(offsets),
        )

    def union(self, fields: Iterable[TypeLayout]) -> TypeLayout:
        field_layouts = tuple(fields)
        if not field_layouts:
            return TypeLayout(0, 0, 1, 0)

        storage_size = max(field.allocation_size for field in field_layouts)
        alignment = max(field.alignment for field in field_layouts)
        allocation_size = align_up(storage_size, alignment)
        return TypeLayout(
            storage_size=storage_size,
            allocation_size=allocation_size,
            alignment=alignment,
            stride=allocation_size,
            field_offsets=(0,) * len(field_layouts),
        )

    @lru_cache(maxsize=32)
    def alignment_carrier(self, alignment: int) -> ir.Type:
        """Return storage with exactly the requested allocation and alignment."""
        candidates = [ir.IntType(width) for width in (8, 16, 32, 64, 128)]
        candidates.append(ir.VectorType(ir.IntType(8), alignment))
        for candidate in candidates:
            layout = self.primitive(candidate)
            if (
                layout.allocation_size == alignment
                and layout.alignment == alignment
            ):
                return candidate
        raise ValueError(
            f"target {self.triple} has no storage carrier for alignment "
            f"{alignment}"
        )


@lru_cache(maxsize=None)
def _get_target_layout(triple, cpu, features):
    machine = _get_target_machine(triple, cpu, features)
    return TargetLayout(triple, target_data=machine.target_data)


def get_target_layout(triple: Optional[str] = None) -> TargetLayout:
    return _get_target_layout(
        *_target_machine_cache_key(triple or default_target_triple()))


def get_type_layout(pc_type) -> TypeLayout:
    layout = pc_type.get_layout()
    if not isinstance(layout, TypeLayout):
        raise TypeError(f"{pc_type!r} did not provide a TypeLayout")
    return layout


def validate_ctypes_layout(pc_type, ctype, fields=()):
    """Reject host ctypes representations that cannot model native storage."""
    import ctypes

    layout = get_type_layout(pc_type)
    actual = ctypes.sizeof(ctype), ctypes.alignment(ctype)
    expected = layout.allocation_size, layout.alignment
    if actual != expected:
        raise ValueError(
            f"ctypes cannot represent {pc_type.get_name()} layout: "
            f"size/alignment {actual} != {expected}"
        )
    for index, name in fields:
        if getattr(ctype, name).offset != layout.field_offsets[index]:
            raise ValueError(
                f"ctypes cannot represent {pc_type.get_name()} field {name!r}"
            )
    return ctype
