"""pc_dict literal carrier.

pc_dict models dict literal results as compile-time carriers that preserve
key/value ValueRef pairs without forcing an eager lowering target.
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..valueref import ValueRef


class PCDictTypeMeta(type):
    """Metaclass providing mapping-like compile-time interfaces."""

    def __len__(cls):
        return len(cls._entries) if cls._entries else 0


class PCDictType(metaclass=PCDictTypeMeta):
    """Dict literal carrier with stored key/value ValueRefs."""

    _entries: List[Tuple['ValueRef', 'ValueRef']] = None
    _is_pc_dict: bool = True
    _canonical_name: str = 'pc_dict'

    @classmethod
    def get_name(cls) -> str:
        return getattr(cls, '_canonical_name', 'pc_dict')

    @classmethod
    def is_pc_dict(cls) -> bool:
        return getattr(cls, '_is_pc_dict', False)

    @classmethod
    def get_entries(cls) -> List[Tuple['ValueRef', 'ValueRef']]:
        return cls._entries if cls._entries else []

    @classmethod
    def get_length(cls) -> int:
        return len(cls._entries) if cls._entries else 0

    @classmethod
    def get_value(cls, key):
        for key_ref, value_ref in cls.get_entries():
            if key_ref.is_python_value() and key_ref.get_python_value() == key:
                return value_ref
        raise KeyError(key)


def create_pc_dict_type(entries: List[Tuple['ValueRef', 'ValueRef']]) -> type:
    """Create a pc_dict type from key/value ValueRef pairs."""
    return PCDictTypeMeta(
        'pc_dict',
        (PCDictType,),
        {
            '_canonical_name': 'pc_dict',
            '_entries': entries,
            '_is_pc_dict': True,
        },
    )


class pc_dict(PCDictType):
    """Factory class for pc_dict carriers."""

    @classmethod
    def get_name(cls) -> str:
        return 'pc_dict'

    @classmethod
    def from_entries(cls, entries: List[Tuple['ValueRef', 'ValueRef']]) -> type:
        return create_pc_dict_type(entries)


__all__ = ['pc_dict', 'PCDictType', 'create_pc_dict_type']
