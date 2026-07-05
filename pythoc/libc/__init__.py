"""
PC Compiler - Standard C Library Functions
Provides pre-defined extern declarations for common C library functions
"""

from ..decorators import extern

# Export all C library functions
from .stdio import *
from .stdlib import *
from .string import *
from .math import *
from .memory import *
from .ctype import *
from .stddef import *
from .stdint import *
from .time import *
from .unistd import *
from .dlfcn import *
from .errno import *
from .signal import *
from .setjmp import *
from .stdarg import *
from .dispatch import *
from .semaphore import *
from .ucontext import *

__all__ = [
    # stdio.h
    'FILE',
    'printf', 'scanf', 'puts', 'getchar', 'putchar', 'fopen', 'fclose', 'freopen',
    'fread', 'fwrite', 'fgets', 'fputs', 'fprintf', 'fscanf', 'fflush',

    # stdlib.h functions
    'malloc', 'free', 'calloc', 'realloc', 'exit', 'abort', 'atoi', 'atof',
    'strtol', 'strtod', 'rand', 'srand', 'system',

    # string.h functions
    'strlen', 'strcpy', 'strncpy', 'strcat', 'strncat', 'strcmp', 'strncmp',
    'strchr', 'strstr', 'memcpy', 'memset', 'memcmp', 'memmove',

    # math.h functions
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh',
    'exp', 'log', 'log10', 'pow', 'sqrt', 'ceil', 'floor', 'fabs', 'fmod',

    # Memory management utilities
    'memalloc', 'memfree', 'memzero',

    # ctype.h
    'isalpha', 'isdigit', 'isspace', 'isalnum', 'isupper', 'islower',
    'toupper', 'tolower',

    # stddef.h typedefs
    'size_t', 'ssize_t', 'ptrdiff_t', 'wchar_t',

    # stdint.h / inttypes.h typedefs
    'int8_t', 'int16_t', 'int32_t', 'int64_t',
    'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
    'intptr_t', 'uintptr_t', 'intmax_t', 'uintmax_t',

    # time.h / sys/time.h
    'timeval', 'gettimeofday', 'time',

    # unistd.h
    'sysconf', 'getpid', 'getcwd', 'close', 'read', 'write', 'lseek',
    'access', 'isatty', 'getpagesize', 'mprotect',

    # dlfcn.h
    'dlopen', 'dlsym', 'dlclose', 'dlerror',

    # errno.h
    '__error',

    # signal.h
    'siginfo_t', 'signal', 'raise_', 'sigaction', 'sigemptyset', 'sigfillset',

    # setjmp.h
    'jmp_buf', 'sigjmp_buf', 'setjmp', 'longjmp',

    # stdarg.h
    'va_list',

    # dispatch.h
    'dispatch_semaphore_t', 'dispatch_time_t',
    'dispatch_semaphore_create', 'dispatch_semaphore_wait', 'dispatch_semaphore_signal',

    # semaphore.h
    'sem_t', 'sem_init', 'sem_wait', 'sem_post', 'sem_destroy',

    # ucontext.h
    'ucontext_t',
]