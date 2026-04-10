"""pc_tuple literal carrier.

pc_tuple models tuple literal results as compile-time carriers that preserve
ValueRef elements without committing to a struct lowering strategy.
"""

from __future__ import annotations

from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..valueref import ValueRef


class PCTupleTypeMeta(type):
    """Metaclass providing tuple-like compile-time interfaces."""

    def __iter__(cls):
        elements = cls._elements if cls._elements else []
        return iter(elements)

    def __len__(cls):
        return len(cls._elements) if cls._elements else 0


class PCTupleType(metaclass=PCTupleTypeMeta):
    """Tuple literal carrier with stored ValueRef elements."""

    _elements: List['ValueRef'] = None
    _is_pc_tuple: bool = True
    _canonical_name: str = 'pc_tuple'

    @classmethod
    def get_name(cls) -> str:
        return getattr(cls, '_canonical_name', 'pc_tuple')

    @classmethod
    def is_pc_tuple(cls) -> bool:
        return getattr(cls, '_is_pc_tuple', False)

    @classmethod
    def get_elements(cls) -> List['ValueRef']:
        return cls._elements if cls._elements else []

    @classmethod
    def get_length(cls) -> int:
        return len(cls._elements) if cls._elements else 0

    @classmethod
    def get_element(cls, index: int) -> 'ValueRef':
        if cls._elements is None or index >= len(cls._elements):
            raise IndexError(f"pc_tuple index {index} out of range")
        return cls._elements[index]

    @classmethod
    def get_element_types(cls) -> List[Any]:
        if cls._elements is None:
            return []
        return [elem.type_hint for elem in cls._elements]


def create_pc_tuple_type(elements: List['ValueRef']) -> type:
    """Create a pc_tuple type from a list of ValueRefs."""
    from ..type_id import get_type_id

    field_types = [elem.get_pc_type() for elem in elements]
    type_parts = [get_type_id(ft) for ft in field_types]
    canonical_name = f"pc_tuple[{', '.join(type_parts)}]"

    return PCTupleTypeMeta(
        canonical_name,
        (PCTupleType,),
        {
            '_canonical_name': canonical_name,
            '_elements': elements,
            '_is_pc_tuple': True,
        },
    )


class pc_tuple(PCTupleType):
    """Factory class for pc_tuple carriers."""

    @classmethod
    def get_name(cls) -> str:
        return 'pc_tuple'

    @classmethod
    def from_elements(cls, elements: List['ValueRef']) -> type:
        return create_pc_tuple_type(elements)


__all__ = ['pc_tuple', 'PCTupleType', 'create_pc_tuple_type']
