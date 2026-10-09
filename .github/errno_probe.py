import ctypes

libc = ctypes.CDLL(None)
err = libc.__errno_location
err.restype = ctypes.POINTER(ctypes.c_int)

from pythoc import compile, i32
from pythoc.libc.errno import __errno_location


@compile
def read_errno() -> i32:
    return __errno_location()[0]


err().contents.value = 0
print('[probe] calling read_errno', flush=True)
r = read_errno()
print('[probe] kernel observed on first call:', r, flush=True)
print('[probe] after call errno =', err().contents.value, flush=True)
