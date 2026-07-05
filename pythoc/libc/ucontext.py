"""
User context type (ucontext.h / sys/ucontext.h).

For the current targets the full layout is not needed; the pointer is treated
as opaque.  If translated code begins accessing fields, this binding must be
expanded to match the target ABI.
"""

from ..builtin_entities import i8
from ..forward_ref import mark_type_defined

ucontext_t = i8
mark_type_defined("ucontext_t", ucontext_t)

__all__ = ['ucontext_t']
