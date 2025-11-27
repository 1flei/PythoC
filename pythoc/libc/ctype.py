"""
Character classification and conversion (ctype.h)
P0 subset: is* and to* family
"""

from ..decorators import extern
from ..builtin_entities import i32

# Classification
@extern(lib='c')
def isalnum(c: i32) -> i32: pass

@extern(lib='c')
def isalpha(c: i32) -> i32: pass

@extern(lib='c')
def isblank(c: i32) -> i32: pass

@extern(lib='c')
def iscntrl(c: i32) -> i32: pass

@extern(lib='c')
def isdigit(c: i32) -> i32: pass

@extern(lib='c')
def isgraph(c: i32) -> i32: pass

@extern(lib='c')
def islower(c: i32) -> i32: pass

@extern(lib='c')
def isprint(c: i32) -> i32: pass

@extern(lib='c')
def ispunct(c: i32) -> i32: pass

@extern(lib='c')
def isspace(c: i32) -> i32: pass

@extern(lib='c')
def isupper(c: i32) -> i32: pass

@extern(lib='c')
def isxdigit(c: i32) -> i32: pass

# Conversion
@extern(lib='c')
def tolower(c: i32) -> i32: pass

@extern(lib='c')
def toupper(c: i32) -> i32: pass
