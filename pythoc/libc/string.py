"""
String Manipulation Functions (string.h)
"""

from ..decorators import extern
from ..builtin_entities import ptr, i8, i32, i64

# String length
@extern(lib='c')
def strlen(s: ptr[i8]) -> i64:
    """Get string length"""
    pass

# String copying
@extern(lib='c')
def strcpy(dest: ptr[i8], src: ptr[i8]) -> ptr[i8]:
    """Copy string"""
    pass

@extern(lib='c')
def strncpy(dest: ptr[i8], src: ptr[i8], n: i64) -> ptr[i8]:
    """Copy string with limit"""
    pass

# String concatenation
@extern(lib='c')
def strcat(dest: ptr[i8], src: ptr[i8]) -> ptr[i8]:
    """Concatenate strings"""
    pass

@extern(lib='c')
def strncat(dest: ptr[i8], src: ptr[i8], n: i64) -> ptr[i8]:
    """Concatenate strings with limit"""
    pass

# String comparison
@extern(lib='c')
def strcmp(s1: ptr[i8], s2: ptr[i8]) -> i32:
    """Compare strings"""
    pass

@extern(lib='c')
def strncmp(s1: ptr[i8], s2: ptr[i8], n: i64) -> i32:
    """Compare strings with limit"""
    pass

# String searching
@extern(lib='c')
def strchr(s: ptr[i8], c: i32) -> ptr[i8]:
    """Find character in string"""
    pass

@extern(lib='c')
def strstr(haystack: ptr[i8], needle: ptr[i8]) -> ptr[i8]:
    """Find substring"""
    pass

# Memory operations
@extern(lib='c')
def memcpy(dest: ptr[i8], src: ptr[i8], n: i64) -> ptr[i8]:
    """Copy memory block"""
    pass

@extern(lib='c')
def memset(s: ptr[i8], c: i32, n: i64) -> ptr[i8]:
    """Fill memory with constant byte"""
    pass

@extern(lib='c')
def memcmp(s1: ptr[i8], s2: ptr[i8], n: i64) -> i32:
    """Compare memory blocks"""
    pass

@extern(lib='c')
def memmove(dest: ptr[i8], src: ptr[i8], n: i64) -> ptr[i8]:
    """Move memory block (handles overlap)"""
    pass

# P0 additions: token/suffix searches/spans
@extern(lib='c')
def strtok(s: ptr[i8], delim: ptr[i8]) -> ptr[i8]:
    """Tokenize string (not thread-safe)"""
    pass

@extern(lib='c')
def strtok_r(s: ptr[i8], delim: ptr[i8], saveptr: ptr[ptr[i8]]) -> ptr[i8]:
    """Tokenize string (reentrant)"""
    pass

@extern(lib='c')
def strrchr(s: ptr[i8], c: i32) -> ptr[i8]:
    """Find last occurrence of character"""
    pass

@extern(lib='c')
def strspn(s: ptr[i8], accept: ptr[i8]) -> i64:
    """Span of characters in accept"""
    pass

@extern(lib='c')
def strcspn(s: ptr[i8], reject: ptr[i8]) -> i64:
    """Span of characters not in reject"""
    pass

@extern(lib='c')
def strpbrk(s: ptr[i8], accept: ptr[i8]) -> ptr[i8]:
    """Search any of a set of bytes"""
    pass