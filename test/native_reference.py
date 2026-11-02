"""Own a temporary C reference library until its last test call."""

import _ctypes
from contextlib import contextmanager
import ctypes
from pathlib import Path
import sys
import tempfile

from pythoc.utils.cc_utils import compile_c_to_object
from pythoc.utils.link_utils import get_shared_lib_extension, link_files


@contextmanager
def c_reference(source, libraries=()):
    """Compile and load C code; references must not escape this context."""
    with tempfile.TemporaryDirectory(prefix='pythoc-c-reference-') as temp:
        root = Path(temp)
        obj = compile_c_to_object(str(source), str(root / 'reference.o'))
        path = link_files(
            [obj], str(root / ('reference' + get_shared_lib_extension())),
            shared=True, link_objects=[], link_libraries=list(libraries),
        )
        library = ctypes.CDLL(path)
        try:
            yield library
        finally:
            # ctypes keeps library handles open even after the last reference
            # is dropped. Windows cannot delete a mapped DLL.
            close = (_ctypes.FreeLibrary if sys.platform == 'win32'
                     else _ctypes.dlclose)
            close(library._handle)
            library._handle = 0
