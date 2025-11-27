"""
Main test file that uses all modules
Demonstrates multi-file compilation and cross-module references
"""

from pythoc import compile, i32, ptr, nullptr
from pythoc.libc.stdio import printf

from type_defs import Node, Stack
from node_ops import create_node, node_append, node_count, node_sum
from stack_ops import (
    stack_init, stack_push, stack_pop, 
    stack_peek, stack_is_empty, stack_get_size
)
from utils import list_to_stack, stack_sum


@compile
def test_node_operations() -> i32:
    """Test basic node operations"""
    printf("=== Testing Node Operations ===\n")
    
    # Create a linked list: 10 -> 20 -> 30
    head = create_node(i32(10))
    head = node_append(head, i32(20))
    head = node_append(head, i32(30))
    
    count = node_count(head)
    total = node_sum(head)
    
    printf("List count: %d\n", count)
    printf("List sum: %d\n", total)
    
    return i32(0)


@compile
def test_stack_operations() -> i32:
    """Test stack operations"""
    printf("=== Testing Stack Operations ===\n")
    
    stack = stack_init()
    
    # Push some values
    stack_push(stack, i32(100))
    stack_push(stack, i32(200))
    stack_push(stack, i32(300))
    
    size = stack_get_size(stack)
    printf("Stack size: %d\n", size)
    
    # Peek at top
    top_value = stack_peek(stack)
    printf("Top value: %d\n", top_value)
    
    # Pop values
    val1 = stack_pop(stack)
    val2 = stack_pop(stack)
    val3 = stack_pop(stack)
    
    printf("Popped: %d, %d, %d\n", val1, val2, val3)
    
    # Check if empty
    is_empty = stack_is_empty(stack)
    printf("Stack is empty: %d\n", is_empty)
    
    return i32(0)


@compile
def test_cross_module() -> i32:
    """Test cross-module functionality"""
    printf("=== Testing Cross-Module Operations ===\n")
    
    # Create a list
    head = create_node(i32(5))
    head = node_append(head, i32(15))
    head = node_append(head, i32(25))
    
    list_sum = node_sum(head)
    printf("List sum: %d\n", list_sum)
    
    # Transfer to stack
    stack = stack_init()
    transferred = list_to_stack(head, stack)
    printf("Transferred %d items to stack\n", transferred)
    
    # Sum the stack
    stack_total = stack_sum(stack)
    printf("Stack sum: %d\n", stack_total)
    
    return i32(0)


@compile
def main() -> i32:
    """Main entry point"""
    printf("\n*** Multi-File Test Demo ***\n\n")
    
    test_node_operations()
    printf("\n")
    
    test_stack_operations()
    printf("\n")
    
    test_cross_module()
    printf("\n")
    
    printf("*** All tests completed ***\n")
    return i32(0)

if __name__ == '__main__':
    from pythoc import compile_to_executable
    compile_to_executable()