"""
Common type definitions shared across multiple files
"""

from __future__ import annotations
from pythoc import compile, i32, ptr


@compile
class Node:
    value: i32
    next: ptr[Node]


@compile
class Stack:
    top: ptr[Node]
    size: i32
