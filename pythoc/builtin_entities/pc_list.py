"""
PC List Type Implementation

pc_list is a struct-like type that wraps list literals containing ValueRefs.
It represents the result of evaluating a list literal like [f32(1.5), f32(2.5)].

Design:
- Inherits from StructType for field access semantics
- Elements are stored as ValueRefs (both pyconst for Python values and IR values)
- Can be converted to array type via type_converter
- visit_List returns pc_list when list contains IR values

Usage:
    [1, 2, 3]           -> pyconst[list[1, 2, 3]]  (pure Python, existing behavior)
    [f32(1.5), f32(2.5)] -> pc_list[f32, f32]      (with IR values)
"""

from typing import List, Optional, Any, TYPE_CHECKING
from llvmlite import ir

from .struct import StructType, StructTypeMeta, create_struct_type
from ..logger import logger

if TYPE_CHECKING:
    from ..valueref import ValueRef


class PCListTypeMeta(StructTypeMeta):
    """Metaclass for PCListType that enables iteration over the class itself.
    
    This allows `for x in pc_list_type:` to iterate over the stored elements.
    Inherits from StructTypeMeta to be compatible with StructType's metaclass chain.
    """
    
    def __iter__(cls):
        """Iterate over stored elements.
        
        This enables for-loop iteration over pc_list types:
            for elem in [1, 2, 3]:  # where [1, 2, 3] becomes pc_list
                ...
        """
        elements = cls._elements if cls._elements else []
        return iter(elements)
    
    def __len__(cls):
        """Return number of elements."""
        return len(cls._elements) if cls._elements else 0


class PCListType(StructType, metaclass=PCListTypeMeta):
    """List literal type - wraps list[ValueRef] in struct-like semantics.
    
    This type represents the result of evaluating a list literal that contains
    IR values (not pure Python values). It stores the actual ValueRefs for
    later conversion to array or other types.
    
    Attributes:
        _elements: List of ValueRef elements
        _is_pc_list: Flag to identify pc_list types
    """
    
    _elements: List['ValueRef'] = None
    _is_pc_list: bool = True
    
    @classmethod
    def get_name(cls) -> str:
        """Return type name"""
        if cls._canonical_name:
            return cls._canonical_name
        return 'pc_list'
    
    @classmethod
    def is_pc_list(cls) -> bool:
        """Check if this is a pc_list type."""
        return getattr(cls, '_is_pc_list', False)
    
    @classmethod
    def get_elements(cls) -> List['ValueRef']:
        """Get the stored ValueRef elements."""
        return cls._elements if cls._elements else []
    
    @classmethod
    def get_length(cls) -> int:
        """Get number of elements."""
        return len(cls._elements) if cls._elements else 0
    
    @classmethod
    def get_element(cls, index: int) -> 'ValueRef':
        """Get element at index."""
        if cls._elements is None or index >= len(cls._elements):
            raise IndexError(f"pc_list index {index} out of range")
        return cls._elements[index]
    
    @classmethod
    def get_element_types(cls) -> List[Any]:
        """Get list of element types (from ValueRef type_hints)."""
        if cls._elements is None:
            return []
        return [elem.type_hint for elem in cls._elements]


def create_pc_list_type(elements: List['ValueRef']) -> type:
    """Create a pc_list type from a list of ValueRefs.
    
    Args:
        elements: List of ValueRef elements
    
    Returns:
        PCListType subclass with stored elements
    """
    from .python_type import pyconst
    
    # Build field types from elements
    # Python values become pyconst[value], IR values keep their type_hint
    field_types = []
    for elem in elements:
        field_types.append(elem.get_pc_type())
    
    # Generate canonical name
    from ..type_id import get_type_id
    type_parts = [get_type_id(ft) for ft in field_types]
    canonical_name = f"pc_list[{', '.join(type_parts)}]"
    
    # Create new pc_list type using PCListTypeMeta to enable iteration
    new_type = PCListTypeMeta(
        canonical_name,
        (PCListType,),
        {
            '_canonical_name': canonical_name,
            '_field_types': field_types,
            '_field_names': None,  # pc_list has no named fields
            '_python_class': None,
            '_struct_info': None,
            '_structure_hash': None,
            '_elements': elements,
            '_is_pc_list': True,
        }
    )
    
    return new_type


class pc_list(PCListType):
    """Factory class for pc_list types.
    
    This is the user-facing class that provides:
    - pc_list.from_elements(elements) - create from ValueRef list
    - pc_list[type1, type2, ...] - type subscript (less common)
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'pc_list'
    
    @classmethod
    def from_elements(cls, elements: List['ValueRef']) -> type:
        """Create pc_list type from list of ValueRefs.
        
        This is the primary way to create pc_list types from visit_List.
        
        Args:
            elements: List of ValueRef elements
        
        Returns:
            PCListType subclass
        """
        return create_pc_list_type(elements)
    
    @classmethod
    def __class_getitem__(cls, item):
        """Support pc_list[type1, type2, ...] subscript syntax.
        
        This is less common - usually pc_list is created via from_elements().
        """
        # Normalize item to tuple
        if not isinstance(item, tuple):
            item = (item,)
        
        # Create pc_list type with given field types (no elements)
        from ..type_id import get_type_id
        type_parts = [get_type_id(ft) for ft in item]
        canonical_name = f"pc_list[{', '.join(type_parts)}]"
        
        new_type = PCListTypeMeta(
            canonical_name,
            (PCListType,),
            {
                '_canonical_name': canonical_name,
                '_field_types': list(item),
                '_field_names': None,
                '_python_class': None,
                '_struct_info': None,
                '_structure_hash': None,
                '_elements': None,  # No elements when created via subscript
                '_is_pc_list': True,
            }
        )
        
        return new_type


__all__ = ['pc_list', 'PCListType', 'create_pc_list_type']
