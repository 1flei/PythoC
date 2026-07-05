"""
Dynamic linking API (dlfcn.h).

These declarations are required by portable C sources that load shared
libraries at runtime (e.g. tinycc's run-time DLL support).
"""

from ..decorators import extern
from ..builtin_entities import ptr, i8, i32, void


@extern(lib='c')
def dlopen(filename: ptr[i8], flags: i32) -> ptr[void]:
    """Open a shared object and return a handle."""
    pass


@extern(lib='c')
def dlsym(handle: ptr[void], symbol: ptr[i8]) -> ptr[void]:
    """Obtain the address of a symbol from a shared object."""
    pass


@extern(lib='c')
def dlclose(handle: ptr[void]) -> i32:
    """Close a shared object handle."""
    pass


@extern(lib='c')
def dlerror() -> ptr[i8]:
    """Return a human-readable string describing the most recent dl error."""
    pass
