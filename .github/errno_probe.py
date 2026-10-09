import ctypes

libc = ctypes.CDLL(None)
err = libc.__errno_location
err.restype = ctypes.POINTER(ctypes.c_int)


def show(tag):
    print('[probe]', tag, 'errno =', err().contents.value, flush=True)


from pythoc import compile, i32
from pythoc.libc.errno import __errno_location


@compile
def nop() -> i32:
    return 0


@compile
def read_errno() -> i32:
    return __errno_location()[0]


import pythoc.python_call as pc

_orig_install = pc.install_library


def _inst(lib, w):
    show('before install_library')
    _orig_install(lib, w)
    show('after install_library')


pc.install_library = _inst

err().contents.value = 0
pc.resolve_compiled_callable(nop)
show('after manual resolve')
err().contents.value = 0
nop()
show('after nop dispatch+entry')
err().contents.value = 0
r = read_errno()
print('[probe] kernel observed on first read_errno call:', r, flush=True)
show('after read_errno call')
err().contents.value = 7
r = read_errno()
print('[probe] kernel observed on second call (pre=7):', r, flush=True)
