"""
Terminal ioctl support (<sys/ioctl.h> / <sys/ttycom.h>).

Only the pieces real-world sources use: ``struct winsize`` / ``struct
ttysize`` layouts (macOS) and the variadic ``ioctl`` entry point.
"""

from ..decorators import compile, extern
from ..builtin_entities import i32, u16, u64, ptr
from ..forward_ref import mark_type_defined
from ._platform import IS_MACOS


if IS_MACOS:
    @compile
    class winsize:
        ws_row: u16
        ws_col: u16
        ws_xpixel: u16
        ws_ypixel: u16

    @compile
    class ttysize:
        ts_lines: u16
        ts_cols: u16
        ts_xx: u16
        ts_yy: u16
else:
    # glibc layout (no ttysize on Linux; provide it for source compat).
    @compile
    class winsize:
        ws_row: u16
        ws_col: u16
        ws_xpixel: u16
        ws_ypixel: u16

    @compile
    class ttysize:
        ts_lines: u16
        ts_cols: u16
        ts_xx: u16
        ts_yy: u16


@extern(lib='c')
def ioctl(fd: i32, request: u64, *args) -> i32:
    """Control device parameters."""
    pass


for _name in ("winsize", "ttysize"):
    mark_type_defined(_name, globals()[_name])


__all__ = ["winsize", "ttysize", "ioctl"]
