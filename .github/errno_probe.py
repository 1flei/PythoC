import ctypes

libc = ctypes.CDLL(None)
err = libc.__errno_location
err.restype = ctypes.POINTER(ctypes.c_int)

_orig_cdll = ctypes.CDLL
_loaded = []


class LoggingCDLL(_orig_cdll):
    def __init__(self, name, mode=0, **kw):
        _loaded.append((name, mode))
        super().__init__(name, mode=mode, **kw)


def show(tag):
    print('[probe]', tag, 'errno =', err().contents.value, flush=True)


ctypes.CDLL = LoggingCDLL

from pythoc import compile, i32
from pythoc.libc.errno import __errno_location


@compile
def nop() -> i32:
    return 0


@compile
def read_errno() -> i32:
    return __errno_location()[0]


show('after decorate')
err().contents.value = 0
nop()
show('after nop() first call')
err().contents.value = 0
r = read_errno()
print('[probe] kernel observed on first read_errno call:', r, flush=True)
show('after read_errno')
err().contents.value = 7
r = read_errno()
print('[probe] kernel observed on second call (pre=7):', r, flush=True)
for name, mode in _loaded:
    if name and ('build' in name or 'pythoc' in name):
        print('[probe] CDLL', name, 'mode=', oct(mode), flush=True)
