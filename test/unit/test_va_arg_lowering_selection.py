"""Unit tests for platform va_arg lowering selection.

The lowering must match the platform's va_list model exactly: on
arm64-apple-darwin va_list is a plain stack cursor, not the AAPCS
register-save-area struct used on Linux.  Emitting AAPCS reads on
Darwin leaves the register path reading uninitialized memory and only
"works" by stack-garbage luck.
"""

import unittest

from pythoc.builder.abi.varargs import (
    AArch64AAPCSVAArgLowering,
    AArch64DarwinVAArgLowering,
    VoidPtrVAArgLowering,
    X86_64SysVVAArgLowering,
    get_va_arg_lowering,
)


class TestVAArgLoweringSelection(unittest.TestCase):
    def test_apple_silicon_darwin_uses_stack_cursor(self):
        for triple in (
            'arm64-apple-darwin23.5.0',
            'arm64-apple-macosx14.0',
            'aarch64-apple-darwin',
        ):
            self.assertIsInstance(
                get_va_arg_lowering(triple),
                AArch64DarwinVAArgLowering,
                triple,
            )

    def test_linux_aarch64_uses_aapcs(self):
        self.assertIsInstance(
            get_va_arg_lowering('aarch64-unknown-linux-gnu'),
            AArch64AAPCSVAArgLowering,
        )

    def test_x86_64_uses_sysv(self):
        for triple in ('x86_64-unknown-linux-gnu', 'x86_64-apple-darwin'):
            self.assertIsInstance(
                get_va_arg_lowering(triple),
                X86_64SysVVAArgLowering,
                triple,
            )

    def test_windows_uses_voidptr(self):
        for triple in ('x86_64-w64-windows-gnu', 'aarch64-windows-msvc'):
            self.assertIsInstance(
                get_va_arg_lowering(triple),
                VoidPtrVAArgLowering,
                triple,
            )


if __name__ == '__main__':
    unittest.main()
