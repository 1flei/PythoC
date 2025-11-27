#!/usr/bin/env python3
"""
PC translation of nsieve benchmark (equivalent to test/example/nsieve.c)

Implements the classic sieve of Eratosthenes using C-like heap memory
(flags array allocated via malloc, initialized with memset), and prints
prime counts for three sizes like the C version.
"""

from pythoc import i8, i32, u8, ptr, compile, seq
from pythoc.libc.stdlib import malloc, free, atoi
from pythoc.libc.string import memset
from pythoc.libc.stdio import printf


@compile
def nsieve(m: i32):
    count: i32 = 0

    # Allocate m bytes for boolean flags (unsigned char in C)
    flags: ptr[u8] = ptr[u8](malloc(m))

    # memset(flags, 1, m)
    memset(flags, 1, m)

    for i in seq(2, m):
        if flags[i] != 0:
            count += 1
            for j in seq(i + i, m, i):
                flags[j] = 0

    free(flags)
    printf("Primes up to %8u %8u\n", m, count)


@compile
def main(argc: i32, argv: ptr[ptr[i8]]) -> i32:
    m = atoi(argv[1])
    for k in seq(3):
        size: i32 = 10000 << (m - k)
        nsieve(size)
    return 0

if __name__ == "__main__":
    from pythoc import compile_to_executable
    compile_to_executable()
