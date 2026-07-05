"""
Common definitions (stddef.h)

The C preprocessor expands object-like macros but leaves typedef names such as
``size_t`` intact, so they survive into generated bindings. These aliases give
those names their canonical 64-bit ABI representation.
"""

from ..builtin_entities import i32, i64, u64
from ..forward_ref import mark_type_defined

size_t = u64
ssize_t = i64
ptrdiff_t = i64
wchar_t = i32

for _name in ('size_t', 'ssize_t', 'ptrdiff_t', 'wchar_t'):
    mark_type_defined(_name, globals()[_name])

