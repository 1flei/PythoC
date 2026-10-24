"""Target-aware memory layout for PythoC semantic types."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Tuple

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

    storage_bits: int
    allocation_bits: int
    alignment_bits: int
    stride_bits: int
    bit_width: Optional[int] = None
    field_offsets_bits: Tuple[int, ...] = ()
    field_widths_bits: Tuple[int, ...] = ()
    field_storage_units_bits: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.storage_bits < 0 or self.allocation_bits < 0:
            raise ValueError("type layout sizes must be non-negative")
        if self.storage_bits > self.allocation_bits:
            raise ValueError("storage size cannot exceed allocation size")
        if (
            self.alignment_bits <= 0
            or self.alignment_bits & (self.alignment_bits - 1)
        ):
            raise ValueError("type layout alignment must be a power of two")
        if self.stride_bits < self.allocation_bits:
            raise ValueError("type layout stride cannot be smaller than allocation size")
        if self.allocation_bits % self.alignment_bits:
            raise ValueError("allocation size must preserve type alignment")
        if self.stride_bits % self.alignment_bits:
            raise ValueError("type layout stride must preserve type alignment")
        field_count = len(self.field_offsets_bits)
        for values in (
            self.field_widths_bits,
            self.field_storage_units_bits,
        ):
            if values and len(values) != field_count:
                raise ValueError("field layout metadata lengths must match")

    @classmethod
    def empty(cls) -> "TypeLayout":
        return cls(0, 0, 8, 0, bit_width=0)

    @staticmethod
    def _bytes(bits: int, name: str) -> int:
        if bits % 8:
            raise ValueError(f"{name} is not byte-addressable")
        return bits // 8

    @property
    def storage_size(self) -> int:
        return self._bytes(self.storage_bits, "storage size")

    @property
    def allocation_size(self) -> int:
        return self._bytes(self.allocation_bits, "allocation size")

    @property
    def alignment(self) -> int:
        return self._bytes(self.alignment_bits, "ABI alignment")

    @property
    def stride(self) -> int:
        return self._bytes(self.stride_bits, "stride")

    @property
    def field_offsets(self) -> Tuple[int, ...]:
        return tuple(
            self._bytes(offset, "field offset")
            for offset in self.field_offsets_bits
        )


def _validate_power_of_two(value: Optional[int], name: str) -> None:
    if value is not None and type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value is not None and (value <= 0 or value & (value - 1)):
        raise ValueError(f"{name} must be a positive power of two")


def _normalize_named_bits(
    values,
    name: str,
) -> Tuple[Tuple[str, int], ...]:
    if values is None:
        return ()
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping from field names to bits")
    normalized = []
    for field_name, bits in values.items():
        if not isinstance(field_name, str):
            raise TypeError(f"{name} keys must be field names")
        if not isinstance(bits, int):
            raise TypeError(f"{name} values must be integers")
        _validate_power_of_two(bits, f"{name}[{field_name!r}]")
        normalized.append((field_name, bits))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class AggregateLayoutConfig:
    """Orthogonal bit-placement and ABI configuration for an aggregate."""

    alignment: Optional[int] = None
    storage_unit: Optional[int] = None
    cross_unit: bool = False
    bit_order: str = "lsb0"
    abi_alignment: Optional[int] = None
    storage_unit_overrides: Tuple[Tuple[str, int], ...] = ()
    unit_break_before: Tuple[Tuple[str, int], ...] = ()
    explicit: bool = False

    def __post_init__(self) -> None:
        _validate_power_of_two(self.alignment, "alignment")
        _validate_power_of_two(self.storage_unit, "storage_unit")
        _validate_power_of_two(self.abi_alignment, "abi_alignment")
        if not isinstance(self.cross_unit, bool):
            raise TypeError("cross_unit must be a bool")
        if self.bit_order not in ("lsb0", "msb0"):
            raise ValueError("bit_order must be 'lsb0' or 'msb0'")
        for name, bits in (
            self.storage_unit_overrides + self.unit_break_before
        ):
            if not isinstance(name, str) or not isinstance(bits, int):
                raise TypeError("field layout overrides must contain name/bit pairs")
            _validate_power_of_two(bits, f"layout override for {name!r}")
        for value, name in (
            (self.storage_unit, "storage_unit"),
            (self.abi_alignment, "abi_alignment"),
        ):
            if value is not None and value % 8:
                raise ValueError(f"{name} must be byte-addressable")

    @property
    def is_natural(self) -> bool:
        return not self.explicit and (
            self.alignment is None
            and self.storage_unit is None
            and self.abi_alignment is None
            and not self.storage_unit_overrides
            and not self.unit_break_before
            and not self.cross_unit
            and self.bit_order == "lsb0"
        )

    @property
    def override_map(self):
        return MappingProxyType(dict(self.storage_unit_overrides))

    @property
    def break_map(self):
        return MappingProxyType(dict(self.unit_break_before))

    def identity(self) -> str:
        return repr((
            self.alignment,
            self.storage_unit,
            self.cross_unit,
            self.bit_order,
            self.abi_alignment,
            self.storage_unit_overrides,
            self.unit_break_before,
            self.explicit,
        ))


NATURAL_LAYOUT = AggregateLayoutConfig()


def make_aggregate_layout_config(**kwargs) -> AggregateLayoutConfig:
    allowed = {
        "alignment",
        "storage_unit",
        "cross_unit",
        "bit_order",
        "abi_alignment",
        "storage_unit_overrides",
        "unit_break_before",
    }
    unknown = set(kwargs) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unknown aggregate layout option(s): {names}")
    normalized = dict(kwargs)
    normalized["storage_unit_overrides"] = _normalize_named_bits(
        normalized.get("storage_unit_overrides"),
        "storage_unit_overrides",
    )
    normalized["unit_break_before"] = _normalize_named_bits(
        normalized.get("unit_break_before"),
        "unit_break_before",
    )
    normalized["explicit"] = bool(kwargs)
    return AggregateLayoutConfig(**normalized)


def _integer_bit_width(pc_type) -> int:
    layout = get_type_layout(pc_type)
    if layout.bit_width is None or layout.bit_width <= 0:
        raise TypeError("bit-layout fields must be integer types")
    return layout.bit_width


def _automatic_storage_unit(widths: Iterable[int]) -> int:
    maximum = max(widths, default=1)
    unit = 8
    while unit < maximum:
        unit *= 2
    return unit


def field_segments(offset, width, unit, bit_order):
    """Map value bits to contiguous slices of native-endian storage units."""
    segments = []
    for value_bit in range(width):
        logical = offset + value_bit
        if bit_order == 'lsb0':
            physical = logical
        else:
            ordered = offset + width - 1 - value_bit
            physical = ordered // unit * unit + unit - 1 - ordered % unit
        start, bit = divmod(physical, unit)
        start *= unit
        if (segments and segments[-1][0] == start
                and segments[-1][1] + segments[-1][3] == bit):
            old_start, old_bit, old_value, count = segments[-1]
            segments[-1] = old_start, old_bit, old_value, count + 1
        else:
            segments.append((start, bit, value_bit, 1))
    return tuple(segments)


def _validate_disjoint_fields(names, offsets, widths, units, bit_order):
    """Reject struct configurations that alias physical memory bits."""
    little_endian = str(get_target_layout().target_data).startswith('e')
    occupied = set()
    for name, offset, width, unit in zip(names, offsets, widths, units):
        field_bits = set()
        for start, bit, _, count in field_segments(offset, width, unit, bit_order):
            for index in range(bit, bit + count):
                byte, within = divmod(index, 8)
                if not little_endian:
                    byte = unit // 8 - 1 - byte
                field_bits.add(start + byte * 8 + within)
        if occupied & field_bits:
            raise ValueError(f'bit-layout struct field {name!r} overlaps another field')
        occupied.update(field_bits)


def bit_aggregate_layout(
    field_names: Iterable[str],
    field_types: Iterable,
    config: AggregateLayoutConfig,
    *,
    is_union: bool,
) -> TypeLayout:
    names = tuple(field_names)
    types = tuple(field_types)
    if len(names) != len(types):
        raise ValueError("field names and field types must have equal lengths")

    widths = tuple(_integer_bit_width(field) for field in types)
    default_unit = config.storage_unit or _automatic_storage_unit(widths)
    placement_alignment = config.alignment or 1
    overrides = config.override_map
    breaks = config.break_map
    unknown = (set(overrides) | set(breaks)) - set(names)
    if unknown:
        raise ValueError(
            "layout override names are not fields: "
            + ", ".join(sorted(unknown))
        )
    if is_union and breaks:
        raise ValueError("unit_break_before is not valid for unions")

    units = tuple(overrides.get(name, default_unit) for name in names)
    if any(unit % 8 for unit in units):
        raise ValueError("storage units must be byte-addressable")
    if not config.cross_unit:
        for name, width, unit in zip(names, widths, units):
            if width > unit:
                raise ValueError(
                    f"field {name!r} is wider than its storage unit"
                )

    if is_union:
        offsets = (0,) * len(names)
        storage_bits = max(
            (align_up(width, unit) for width, unit in zip(widths, units)),
            default=0,
        )
    else:
        offsets_list = []
        cursor = 0
        for name, width, unit in zip(names, widths, units):
            boundary = breaks.get(name)
            if boundary is not None:
                cursor = align_up(cursor, boundary)
            cursor = align_up(cursor, placement_alignment)
            if (
                not config.cross_unit
                and cursor // unit != (cursor + width - 1) // unit
            ):
                cursor = align_up(cursor, unit)
            offsets_list.append(cursor)
            cursor += width
        offsets = tuple(offsets_list)
        _validate_disjoint_fields(names, offsets, widths, units, config.bit_order)
        storage_bits = max(
            (
                align_up(offset + width, unit)
                for offset, width, unit in zip(offsets, widths, units)
            ),
            default=0,
        )

    abi_alignment = config.abi_alignment or max(units, default=8)
    if abi_alignment % 8:
        raise ValueError("abi_alignment must be byte-addressable")
    allocation_alignment = max(8, abi_alignment)
    allocation_bits = align_up(storage_bits, allocation_alignment)
    if allocation_bits == 0:
        allocation_bits = allocation_alignment

    return TypeLayout(
        storage_bits=storage_bits,
        allocation_bits=allocation_bits,
        alignment_bits=allocation_alignment,
        stride_bits=allocation_bits,
        field_offsets_bits=offsets,
        field_widths_bits=widths,
        field_storage_units_bits=units,
    )


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
            return TypeLayout.empty()

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
            storage_bits=stored * 8,
            allocation_bits=allocation_size * 8,
            alignment_bits=alignment * 8,
            stride_bits=align_up(allocation_size, alignment) * 8,
            bit_width=bit_width,
        )

    def array(
        self,
        element: TypeLayout,
        dimensions: Iterable[int],
    ) -> TypeLayout:
        result = element
        for count in reversed(tuple(dimensions)):
            allocation_bits = result.stride_bits * count
            result = TypeLayout(
                storage_bits=allocation_bits,
                allocation_bits=allocation_bits,
                alignment_bits=result.alignment_bits,
                stride_bits=allocation_bits,
            )
        return result

    def struct(self, fields: Iterable[TypeLayout]) -> TypeLayout:
        offsets = []
        end = 0
        alignment = 8
        for field in fields:
            end = align_up(end, field.alignment_bits)
            offsets.append(end)
            end += field.allocation_bits
            alignment = max(alignment, field.alignment_bits)

        allocation_size = align_up(end, alignment)
        return TypeLayout(
            storage_bits=end,
            allocation_bits=allocation_size,
            alignment_bits=alignment,
            stride_bits=allocation_size,
            field_offsets_bits=tuple(offsets),
        )

    def union(self, fields: Iterable[TypeLayout]) -> TypeLayout:
        field_layouts = tuple(fields)
        if not field_layouts:
            return TypeLayout.empty()

        storage_size = max(field.allocation_bits for field in field_layouts)
        alignment = max(field.alignment_bits for field in field_layouts)
        allocation_size = align_up(storage_size, alignment)
        return TypeLayout(
            storage_bits=storage_size,
            allocation_bits=allocation_size,
            alignment_bits=alignment,
            stride_bits=allocation_size,
            field_offsets_bits=(0,) * len(field_layouts),
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
