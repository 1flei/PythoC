# -*- coding: utf-8 -*-
from typing import Any, List


def mangle_function_name(func_name: str, param_types: List[Any]) -> str:
    """
    Generate a mangled function name for overloading.
    
    Format: _Z<name_len><name><param_encodings>
    """
    if not param_types:
        return func_name

    from ..type_id import get_type_id
    
    # IDs are lossless but variable length; frame each component so adjacent
    # parameter types cannot be parsed as a different signature.
    param_encodings = ''.join(
        f'{len(type_id)}_{type_id}' for type_id in
        (get_type_id(pt) for pt in param_types))
    return f'_Z{len(func_name)}{func_name}{param_encodings}'
