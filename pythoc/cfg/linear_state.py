"""Immutable active-token sets with path sharing between CFG snapshots."""

from collections.abc import Collection


class _Branch:
    __slots__ = ('left', 'right', 'size')

    def __init__(self, left, right):
        self.left, self.right = left, right
        self.size = _size(left) + _size(right)


def _size(node):
    if node is None:
        return 0
    return node.size if isinstance(node, _Branch) else 1


def _branch(left, right):
    # A sole leaf needs no routing bits. A branch retains its bit position.
    if left is None and not isinstance(right, _Branch):
        return right
    if right is None and not isinstance(left, _Branch):
        return left
    return _Branch(left, right)


def _replace(node, token, present, bit=1):
    if node is None:
        return token if present else None
    if not isinstance(node, _Branch):
        if node == token:
            return node if present else None
        if not present:
            return node
        if (node & bit) != (token & bit):
            return (_Branch(node, token) if token & bit
                    else _Branch(token, node))
        child = _replace(node, token, present, bit << 1)
        return _Branch(None, child) if token & bit else _Branch(child, None)
    if token & bit:
        right = _replace(node.right, token, present, bit << 1)
        return node if right is node.right else _branch(node.left, right)
    left = _replace(node.left, token, present, bit << 1)
    return node if left is node.left else _branch(left, node.right)


def _same(first, second):
    if first is second:
        return True
    if isinstance(first, _Branch) and isinstance(second, _Branch):
        return (first.size == second.size
                and _same(first.left, second.left)
                and _same(first.right, second.right))
    return first == second


class LinearState(Collection):
    """A persistent set of dense token IDs allocated by one linear checker.

    Absent tokens are invalid. Updating a token copies only its radix path;
    forwarding an unchanged state to another block shares the entire tree.
    Dense IDs bound the path length by the number of tracked token paths.
    """

    __slots__ = ('_root',)

    def __init__(self, root=None):
        self._root = root

    def with_token(self, token, present):
        root = _replace(self._root, token, present)
        return self if root is self._root else LinearState(root)

    def __contains__(self, token):
        node, bit = self._root, 1
        while isinstance(node, _Branch):
            node = node.right if token & bit else node.left
            bit <<= 1
        return node == token

    def __iter__(self):
        pending = [self._root]
        while pending:
            node = pending.pop()
            if isinstance(node, _Branch):
                pending.extend((node.right, node.left))
            elif node is not None:
                yield node

    def __len__(self):
        return _size(self._root)

    def __eq__(self, other):
        if not isinstance(other, LinearState):
            return NotImplemented
        return _same(self._root, other._root)
