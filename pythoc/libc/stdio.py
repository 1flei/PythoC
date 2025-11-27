"""
Standard I/O Library Functions (stdio.h)
"""

from ..decorators import extern
from ..builtin_entities import ptr, i8, i32, i64

# Input/Output functions
@extern(lib='c')
def printf(format: ptr[i8], *args) -> i32:
    """Print formatted output to stdout"""
    pass

@extern(lib='c')
def scanf(format: ptr[i8], *args) -> i32:
    """Read formatted input from stdin"""
    pass

@extern(lib='c')
def puts(s: ptr[i8]) -> i32:
    """Write string to stdout with newline"""
    pass

@extern(lib='c')
def getchar() -> i32:
    """Read a character from stdin"""
    pass

@extern(lib='c')
def putchar(c: i32) -> i32:
    """Write a character to stdout"""
    pass

# File operations (simplified signatures)
@extern(lib='c')
def fopen(filename: ptr[i8], mode: ptr[i8]) -> ptr[i8]:
    """Open a file"""
    pass

@extern(lib='c')
def fclose(stream: ptr[i8]) -> i32:
    """Close a file"""
    pass

@extern(lib='c')
def fread(ptr: ptr[i8], size: i64, count: i64, stream: ptr[i8]) -> i64:
    """Read data from file"""
    pass

@extern(lib='c')
def fwrite(ptr: ptr[i8], size: i64, count: i64, stream: ptr[i8]) -> i64:
    """Write data to file"""
    pass

@extern(lib='c')
def fgets(s: ptr[i8], size: i32, stream: ptr[i8]) -> ptr[i8]:
    """Read string from file"""
    pass

@extern(lib='c')
def fputs(s: ptr[i8], stream: ptr[i8]) -> i32:
    """Write string to file"""
    pass

@extern(lib='c')
def fprintf(stream: ptr[i8], format: ptr[i8], *args) -> i32:
    """Print formatted output to file"""
    pass

@extern(lib='c')
def fscanf(stream: ptr[i8], format: ptr[i8], *args) -> i32:
    """Read formatted input from file"""
    pass

@extern(lib='c')
def fflush(stream: ptr[i8]) -> i32:
    """Flush file buffer"""
    pass

# P0 additions: buffering, seeking, char IO, formatted to buffers
@extern(lib='c')
def fseek(stream: ptr[i8], offset: i64, whence: i32) -> i32:
    """Reposition stream position indicator"""
    pass

@extern(lib='c')
def ftell(stream: ptr[i8]) -> i64:
    """Obtain current value of the file position indicator"""
    pass

@extern(lib='c')
def rewind(stream: ptr[i8]) -> None:
    """Set file position to the beginning of the file"""
    pass

@extern(lib='c')
def ferror(stream: ptr[i8]) -> i32:
    """Test error indicator for a stream"""
    pass

@extern(lib='c')
def feof(stream: ptr[i8]) -> i32:
    """Test end-of-file indicator for a stream"""
    pass

@extern(lib='c')
def clearerr(stream: ptr[i8]) -> None:
    """Clear error and EOF indicators for a stream"""
    pass

@extern(lib='c')
def setvbuf(stream: ptr[i8], buf: ptr[i8], mode: i32, size: i64) -> i32:
    """Set buffering mode and buffer"""
    pass

@extern(lib='c')
def setbuf(stream: ptr[i8], buf: ptr[i8]) -> None:
    """Set buffer for a file stream"""
    pass

@extern(lib='c')
def fgetc(stream: ptr[i8]) -> i32:
    """Get next character from stream"""
    pass

@extern(lib='c')
def fputc(c: i32, stream: ptr[i8]) -> i32:
    """Write character to stream"""
    pass

@extern(lib='c')
def ungetc(c: i32, stream: ptr[i8]) -> i32:
    """Push character back onto stream"""
    pass

@extern(lib='c')
def sprintf(s: ptr[i8], format: ptr[i8], *args) -> i32:
    """Write formatted output to string"""
    pass

@extern(lib='c')
def snprintf(s: ptr[i8], n: i64, format: ptr[i8], *args) -> i32:
    """Write formatted output to sized buffer"""
    pass

@extern(lib='c')
def vprintf(format: ptr[i8], ap: ptr[i8]) -> i32:
    """Formatted print using va_list"""
    pass

@extern(lib='c')
def vfprintf(stream: ptr[i8], format: ptr[i8], ap: ptr[i8]) -> i32:
    """Formatted file print using va_list"""
    pass

@extern(lib='c')
def vsnprintf(s: ptr[i8], n: i64, format: ptr[i8], ap: ptr[i8]) -> i32:
    """Formatted to sized buffer using va_list"""
    pass