"""
Simplified multi-file test - only tests type sharing
No cross-file function calls yet
"""

from __future__ import annotations
from pythoc import compile, i32, ptr, sizeof, nullptr
from pythoc.libc.stdlib import malloc
from pythoc.libc.stdio import printf
from .type_defs import Node, Stack


@compile
def create_and_test_node() -> i32:
    """Create a node and test it - all in one file"""
    # Create a node
    node = ptr[Node](malloc(sizeof(Node)))
    node.value = i32(42)
    node.next = nullptr
    
    printf("Node value: %d\n", node.value)
    return node.value


@compile
def create_and_test_stack() -> i32:
    """Create a stack and test it - all in one file"""
    # Create a stack
    stack = ptr[Stack](malloc(sizeof(Stack)))
    stack.top = nullptr
    stack.size = i32(0)
    
    printf("Stack size: %d\n", stack.size)
    
    # Create a node and push it
    node = ptr[Node](malloc(sizeof(Node)))
    node.value = i32(100)
    node.next = stack.top
    stack.top = node
    stack.size = stack.size + i32(1)
    
    printf("After push, stack size: %d\n", stack.size)
    printf("Top value: %d\n", stack.top.value)
    
    return stack.size


@compile
def main() -> i32:
    """Main entry point"""
    printf("\n=== Simple Multi-File Test ===\n\n")
    
    printf("Testing Node:\n")
    create_and_test_node()
    
    printf("\nTesting Stack:\n")
    create_and_test_stack()
    
    printf("\n=== Test completed ===\n")
    return i32(0)


if __name__ == '__main__':
    main()
