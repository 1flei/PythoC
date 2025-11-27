"""
Standard Library Functions (stdlib.h)
"""

from ..decorators import extern
from ..builtin_entities import ptr, i8, i32, i64, f64

# Memory management
@extern(lib='c')
def malloc(size: i64) -> ptr[i8]:
    """Allocate memory"""
    pass

@extern(lib='c')
def free(ptr: ptr[i8]) -> None:
    """Free allocated memory"""
    pass

@extern(lib='c')
def calloc(count: i64, size: i64) -> ptr[i8]:
    """Allocate and zero-initialize memory"""
    pass

@extern(lib='c')
def realloc(ptr: ptr[i8], size: i64) -> ptr[i8]:
    """Reallocate memory"""
    pass

# Program control
@extern(lib='c')
def exit(status: i32) -> None:
    """Exit program"""
    pass

@extern(lib='c')
def abort() -> None:
    """Abort program"""
    pass

# String conversion
@extern(lib='c')
def atoi(str: ptr[i8]) -> i32:
    """Convert string to integer"""
    pass

@extern(lib='c')
def atol(str: ptr[i8]) -> i64:
    """Convert string to long integer"""
    pass

@extern(lib='c')
def atof(str: ptr[i8]) -> f64:
    """Convert string to double"""
    pass

@extern(lib='c')
def strtol(str: ptr[i8], endptr: ptr[ptr[i8]], base: i32) -> i64:
    """Convert string to long integer"""
    pass

@extern(lib='c')
def strtod(str: ptr[i8], endptr: ptr[ptr[i8]]) -> f64:
    """Convert string to double"""
    pass

# Random numbers
@extern(lib='c')
def rand() -> i32:
    """Generate random number"""
    pass

@extern(lib='c')
def srand(seed: i32) -> None:
    """Seed random number generator"""
    pass

# System interaction
@extern(lib='c')
def system(command: ptr[i8]) -> i32:
    """Execute system command"""
    pass

# P0 additions: env, sort/search, more strto*
@extern(lib='c')
def getenv(name: ptr[i8]) -> ptr[i8]:
    """Get environment variable"""
    pass

@extern(lib='c')
def qsort(base: ptr[i8], nmemb: i64, size: i64, compar: ptr[i8]) -> None:
    """Sort array with comparator (function pointer approximated as ptr[i8])"""
    pass

@extern(lib='c')
def bsearch(key: ptr[i8], base: ptr[i8], nmemb: i64, size: i64, compar: ptr[i8]) -> ptr[i8]:
    """Binary search in array (function pointer approximated as ptr[i8])"""
    pass

@extern(lib='c')
def strtoul(str: ptr[i8], endptr: ptr[ptr[i8]], base: i32) -> i64:
    """Convert string to unsigned long (mapped to i64)"""
    pass

@extern(lib='c')
def strtoll(str: ptr[i8], endptr: ptr[ptr[i8]], base: i32) -> i64:
    """Convert string to long long (mapped to i64)"""
    pass

@extern(lib='c')
def strtoull(str: ptr[i8], endptr: ptr[ptr[i8]], base: i32) -> i64:
    """Convert string to unsigned long long (mapped to i64)"""
    pass