"""
Standard Library Functions (stdlib.h)
"""

from ..decorators import extern
from ..builtin_entities import ptr, i8, i32, i64, f64, void

# Memory management
@extern(lib='c')
def malloc(size: i64) -> ptr[void]:
    """Allocate memory"""
    pass

@extern(lib='c')
def free(ptr: ptr[void]) -> None:
    """Free allocated memory"""
    pass

@extern(lib='c')
def calloc(count: i64, size: i64) -> ptr[void]:
    """Allocate and zero-initialize memory"""
    pass

@extern(lib='c')
def realloc(ptr: ptr[void], size: i64) -> ptr[void]:
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
def qsort(base: ptr[void], nmemb: i64, size: i64, compar: ptr[void]) -> None:
    """Sort array with comparator (function pointer approximated as ptr[void])"""
    pass

@extern(lib='c')
def bsearch(key: ptr[void], base: ptr[void], nmemb: i64, size: i64, compar: ptr[void]) -> ptr[void]:
    """Binary search in array (function pointer approximated as ptr[void])"""
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