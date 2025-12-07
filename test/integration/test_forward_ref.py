"""
Test forward reference and circular reference support for composite types.

Tests cover:
1. Self-reference in struct (ptr[Self])
2. Forward reference (A references B, B defined later)
3. Circular reference (A references B, B references A)
4. Enum with forward-referenced payload types
5. Nested forward references (ptr[ptr[Self]])
6. Multiple circular references (A -> B -> C -> A)
"""

from pythoc import compile, enum, i32, i8, ptr, sizeof
from pythoc.libc.stdlib import malloc, free


# =============================================================================
# Define all types first (before any function calls)
# =============================================================================

# Test 1: Self-reference in struct
@compile
class Node:
    """Linked list node with self-reference"""
    value: i32
    next: ptr['Node']  # Self-reference


# Test 2: Forward reference (A references B, B defined later)
@compile
class Container:
    """Container that references Item (defined later)"""
    item: ptr['Item']
    count: i32


@compile
class Item:
    """Item referenced by Container"""
    value: i32
    name_len: i32


# Test 3: Circular reference (A references B, B references A)
@compile
class Parent:
    """Parent that references Child"""
    child: ptr['Child']
    id: i32


@compile
class Child:
    """Child that references Parent (circular)"""
    parent: ptr[Parent]  # Can use Parent directly since it's defined
    value: i32


# Test 4: Enum with self-referential payload
@enum(i8)
class TreeNode:
    """Binary tree node as enum"""
    Leaf: i32                           # Leaf with value
    Branch: ptr['TreeNode']             # Branch pointing to another node (self-ref)


# Test 5: Enum with forward-referenced struct payload
@compile
class Point:
    """Point struct for enum payload"""
    x: i32
    y: i32


@enum(i8)
class Shape:
    """Shape enum with struct payload"""
    Empty: None
    Circle: i32                         # radius
    Rectangle: Point                    # width, height as Point


# Test 6: Deep nesting with forward reference
@compile
class DeepNode:
    """Node with pointer to pointer to self"""
    value: i32
    indirect: ptr[ptr['DeepNode']]  # ptr to ptr to self


# Test 7: Multiple circular references (A -> B -> C -> A)
@compile
class NodeA:
    """First node in cycle"""
    value: i32
    next: ptr['NodeB']


@compile
class NodeB:
    """Second node in cycle"""
    value: i32
    next: ptr['NodeC']


@compile
class NodeC:
    """Third node in cycle, back to A"""
    value: i32
    next: ptr[NodeA]  # Completes the cycle


# =============================================================================
# Define all test functions
# =============================================================================

@compile
def test_self_ref() -> i32:
    """Test self-referential struct"""
    n1: ptr[Node] = ptr[Node](malloc(sizeof(Node)))
    n2: ptr[Node] = ptr[Node](malloc(sizeof(Node)))
    
    n1.value = 10
    n1.next = n2
    
    n2.value = 20
    n2.next = ptr[Node](0)  # null
    
    result: i32 = n1.value + n1.next.value  # 10 + 20 = 30
    
    free(n1)
    free(n2)
    return result


@compile
def test_forward_ref() -> i32:
    """Test forward reference"""
    c: ptr[Container] = ptr[Container](malloc(sizeof(Container)))
    it: ptr[Item] = ptr[Item](malloc(sizeof(Item)))
    
    it.value = 42
    it.name_len = 5
    
    c.item = it
    c.count = 1
    
    result: i32 = c.item.value + c.count  # 42 + 1 = 43
    
    free(c)
    free(it)
    return result


@compile
def test_circular_ref() -> i32:
    """Test circular reference"""
    p: ptr[Parent] = ptr[Parent](malloc(sizeof(Parent)))
    c: ptr[Child] = ptr[Child](malloc(sizeof(Child)))
    
    p.id = 100
    p.child = c
    
    c.value = 50
    c.parent = p
    
    # Navigate the cycle: p -> child -> parent -> id
    result: i32 = p.child.parent.id + c.value  # 100 + 50 = 150
    
    free(p)
    free(c)
    return result


@compile
def test_enum_self_ref() -> i32:
    """Test enum with self-referential payload"""
    # Create a leaf
    leaf: TreeNode = TreeNode(TreeNode.Leaf, 42)
    
    # Check leaf value using match with tuple syntax
    result: i32 = -1
    match leaf:
        case (TreeNode.Leaf, val):
            result = val  # Should be 42
        case _:
            result = -1
    return result


@compile
def test_enum_struct_payload() -> i32:
    """Test enum with struct payload"""
    p: Point
    p.x = 10
    p.y = 20
    
    rect: Shape = Shape(Shape.Rectangle, p)
    
    # Access payload using match with tuple syntax
    result: i32 = -1
    match rect:
        case (Shape.Rectangle, pt):
            result = pt.x + pt.y  # 10 + 20 = 30
        case _:
            result = -1
    return result


@compile
def test_deep_nesting() -> i32:
    """Test deep nesting"""
    n: ptr[DeepNode] = ptr[DeepNode](malloc(sizeof(DeepNode)))
    n.value = 99
    
    # Create ptr to ptr
    pp: ptr[ptr[DeepNode]] = ptr[ptr[DeepNode]](malloc(sizeof(ptr[DeepNode])))
    pp[0] = n
    
    n.indirect = pp
    
    # Access through double indirection
    result: i32 = n.indirect[0].value  # Should be 99
    
    free(pp)
    free(n)
    return result


@compile
def test_multi_circular() -> i32:
    """Test multiple circular references"""
    a: ptr[NodeA] = ptr[NodeA](malloc(sizeof(NodeA)))
    b: ptr[NodeB] = ptr[NodeB](malloc(sizeof(NodeB)))
    c: ptr[NodeC] = ptr[NodeC](malloc(sizeof(NodeC)))
    
    a.value = 1
    a.next = b
    
    b.value = 2
    b.next = c
    
    c.value = 3
    c.next = a
    
    # Navigate the full cycle: a -> b -> c -> a
    result: i32 = a.value + a.next.value + a.next.next.value + a.next.next.next.value
    # 1 + 2 + 3 + 1 = 7
    
    free(a)
    free(b)
    free(c)
    return result


# =============================================================================
# Print type info
# =============================================================================

print("Test 1: Self-reference in struct")
print(f"  Node defined: {Node}")
print(f"  Node._field_types: {Node._field_types}")

print("\nTest 2: Forward reference")
print(f"  Container defined: {Container}")
print(f"  Item defined: {Item}")
print(f"  Container._field_types: {Container._field_types}")

print("\nTest 3: Circular reference")
print(f"  Parent defined: {Parent}")
print(f"  Child defined: {Child}")
print(f"  Parent._field_types: {Parent._field_types}")
print(f"  Child._field_types: {Child._field_types}")

print("\nTest 4: Enum with self-referential payload")
print(f"  TreeNode defined: {TreeNode}")
print(f"  TreeNode._variant_types: {TreeNode._variant_types}")

print("\nTest 5: Enum with forward-referenced struct payload")
print(f"  Shape defined: {Shape}")
print(f"  Shape._variant_types: {Shape._variant_types}")

print("\nTest 6: Deep nesting (ptr[ptr[Self]])")
print(f"  DeepNode defined: {DeepNode}")
print(f"  DeepNode._field_types: {DeepNode._field_types}")

print("\nTest 7: Multiple circular references (A -> B -> C -> A)")
print(f"  NodeA defined: {NodeA}")
print(f"  NodeB defined: {NodeB}")
print(f"  NodeC defined: {NodeC}")


# =============================================================================
# Run tests
# =============================================================================

result1 = test_self_ref()
print(f"\n  test_self_ref() = {result1} (expected 30)")
assert result1 == 30, f"Test 1 failed: expected 30, got {result1}"
print("  PASS")

result2 = test_forward_ref()
print(f"  test_forward_ref() = {result2} (expected 43)")
assert result2 == 43, f"Test 2 failed: expected 43, got {result2}"
print("  PASS")

result3 = test_circular_ref()
print(f"  test_circular_ref() = {result3} (expected 150)")
assert result3 == 150, f"Test 3 failed: expected 150, got {result3}"
print("  PASS")

result4 = test_enum_self_ref()
print(f"  test_enum_self_ref() = {result4} (expected 42)")
assert result4 == 42, f"Test 4 failed: expected 42, got {result4}"
print("  PASS")

result5 = test_enum_struct_payload()
print(f"  test_enum_struct_payload() = {result5} (expected 30)")
assert result5 == 30, f"Test 5 failed: expected 30, got {result5}"
print("  PASS")

result6 = test_deep_nesting()
print(f"  test_deep_nesting() = {result6} (expected 99)")
assert result6 == 99, f"Test 6 failed: expected 99, got {result6}"
print("  PASS")

result7 = test_multi_circular()
print(f"  test_multi_circular() = {result7} (expected 7)")
assert result7 == 7, f"Test 7 failed: expected 7, got {result7}"
print("  PASS")


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("ALL FORWARD REFERENCE TESTS PASSED!")
print("=" * 60)
