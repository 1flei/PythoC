"""
Stack operations - using Node from node_ops
"""

from pythoc import compile, i32, ptr, nullptr, sizeof
from pythoc.libc.stdlib import malloc, free
from type_defs import Node, Stack
from node_ops import create_node


@compile
def stack_init() -> ptr[Stack]:
    """Initialize an empty stack"""
    stack = ptr[Stack](malloc(sizeof(Stack)))
    stack.top = nullptr
    stack.size = i32(0)
    return stack


@compile
def stack_push(stack: ptr[Stack], value: i32) -> i32:
    """Push a value onto the stack, return 1 on success"""
    new_node = create_node(value)
    new_node.next = stack.top
    stack.top = new_node
    stack.size = stack.size + i32(1)
    return i32(1)


@compile
def stack_pop(stack: ptr[Stack]) -> i32:
    """Pop a value from the stack, return the value (or 0 if empty)"""
    if stack.top == nullptr:
        return i32(0)
    
    value = stack.top.value
    old_top = stack.top
    stack.top = stack.top.next
    stack.size = stack.size - i32(1)
    
    free(ptr[i32](old_top))
    return value


@compile
def stack_peek(stack: ptr[Stack]) -> i32:
    """Peek at the top value without removing it"""
    if stack.top == nullptr:
        return i32(0)
    return stack.top.value


@compile
def stack_is_empty(stack: ptr[Stack]) -> i32:
    """Check if stack is empty, return 1 if empty, 0 otherwise"""
    if stack.size == i32(0):
        return i32(1)
    return i32(0)


@compile
def stack_get_size(stack: ptr[Stack]) -> i32:
    """Get the current size of the stack"""
    return stack.size
