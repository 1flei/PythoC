import ctypes

libc = ctypes.CDLL(None)
err = libc.__errno_location
err.restype = ctypes.POINTER(ctypes.c_int)


def show(tag):
    print('[probe]', tag, 'errno =', err().contents.value, flush=True)


from pythoc import compile, i32
from pythoc.libc.errno import __errno_location


@compile
def read_errno() -> i32:
    return __errno_location()[0]


show('after decorate')

import pythoc.native_executor as ne
import pythoc.python_adapter as pa

_orig_execute = ne.MultiSOExecutor.execute_function


def _execute(self, *a, **k):
    show('before execute_function')
    r = _orig_execute(self, *a, **k)
    show('after execute_function')
    return r


ne.MultiSOExecutor.execute_function = _execute

_orig_adapter_path = pa._development_adapter_path


def _adapter_path(*a, **k):
    show('before _development_adapter_path')
    r = _orig_adapter_path(*a, **k)
    show('after _development_adapter_path')
    return r


pa._development_adapter_path = _adapter_path

_orig_load = pa._load_adapter_library


def _load(p):
    show('before _load_adapter_library')
    r = _orig_load(p)
    show('after _load_adapter_library')
    return r


pa._load_adapter_library = _load

err().contents.value = 0
show('reset before first call')
r = read_errno()
print('[probe] kernel observed on first call:', r, flush=True)
show('after first call')
err().contents.value = 7
r = read_errno()
print('[probe] kernel observed on second call (pre=7):', r, flush=True)
show('after second call')
