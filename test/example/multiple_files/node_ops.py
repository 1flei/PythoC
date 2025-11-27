"""
Node operations - basic linked list node manipulation
"""

from pythoc import compile, i32, ptr, nullptr, sizeof
from pythoc.libc.stdlib import malloc
from .type_defs import Node


@compile
def create_node(value: i32) -> ptr[Node]:
    """Create a new node with given value"""
    node = ptr[Node](malloc(sizeof(Node)))
    node.value = value
    node.next = nullptr
    return node


@compile
def node_append(head: ptr[Node], value: i32) -> ptr[Node]:
    """Append a node to the end of the list, return new head"""
    new_node = create_node(value)
    
    if head == nullptr:
        return new_node
    
    current = head
    while current.next != nullptr:
        current = current.next
    
    current.next = new_node
    return head


@compile
def node_count(head: ptr[Node]) -> i32:
    """Count nodes in the list"""
    count = i32(0)
    current = head
    
    while current != nullptr:
        count = count + i32(1)
        current = current.next
    
    return count


@compile
def node_sum(head: ptr[Node]) -> i32:
    """Sum all values in the list"""
    total = i32(0)
    current = head
    
    while current != nullptr:
        total = total + current.value
        current = current.next
    
    return total
