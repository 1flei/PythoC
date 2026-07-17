"""
Platform and architecture detection for libc bindings.

The C library structures and constants vary across operating systems and
CPU architectures.  This module centralizes the host platform detection so
that individual libc binding modules can select the correct ABI definitions.
"""

import platform
import sys

# Operating system
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')

# CPU architecture
MACHINE = platform.machine().lower()
IS_X86_64 = MACHINE in ('x86_64', 'amd64')
IS_ARM64 = MACHINE in ('arm64', 'aarch64')
IS_X86 = MACHINE in ('i386', 'i686', 'x86')

# Convenience tuples for common dispatch patterns
PLATFORM_MACOS_X86_64 = IS_MACOS and IS_X86_64
PLATFORM_MACOS_ARM64 = IS_MACOS and IS_ARM64
PLATFORM_LINUX_X86_64 = IS_LINUX and IS_X86_64
PLATFORM_LINUX_ARM64 = IS_LINUX and IS_ARM64
PLATFORM_WINDOWS_X86_64 = IS_WINDOWS and IS_X86_64
PLATFORM_WINDOWS_ARM64 = IS_WINDOWS and IS_ARM64

__all__ = [
    'IS_WINDOWS', 'IS_MACOS', 'IS_LINUX',
    'MACHINE', 'IS_X86_64', 'IS_ARM64', 'IS_X86',
    'PLATFORM_MACOS_X86_64', 'PLATFORM_MACOS_ARM64',
    'PLATFORM_LINUX_X86_64', 'PLATFORM_LINUX_ARM64',
    'PLATFORM_WINDOWS_X86_64', 'PLATFORM_WINDOWS_ARM64',
]
