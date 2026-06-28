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
from .time import *
from .unistd import *

__all__ = [
    # stdio.h functions
    'printf', 'scanf', 'puts', 'getchar', 'putchar', 'fopen', 'fclose',
    'fread', 'fwrite', 'fgets', 'fputs', 'fprintf', 'fscanf', 'fflush',
    '__stdinp', '__stdoutp', '__stderrp',

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

    # time.h / sys/time.h
    'timeval', 'gettimeofday', 'time',

    # unistd.h
    'sysconf', 'getpid', 'getcwd', 'close', 'read', 'write', 'lseek',
    'access', 'isatty',
]