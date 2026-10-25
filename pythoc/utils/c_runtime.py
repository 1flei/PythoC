"""Load the host C runtime shared with Python's native library calls."""

import ctypes
from functools import lru_cache
import sys


@lru_cache(maxsize=1)
def load_c_runtime():
    """Modern Windows Python and the GNU Windows toolchain use the UCRT."""
    return ctypes.CDLL('ucrtbase.dll' if sys.platform == 'win32' else None)
