"""Source coordinates carried by authored and generated syntax."""

import ast
from typing import NamedTuple, Optional


class SourceLocation(NamedTuple):
    filename: str
    line: int
    column: Optional[int] = None

    @classmethod
    def from_frame(cls, frame):
        return cls(frame.f_code.co_filename, frame.f_lineno)

    def __str__(self):
        filename = self.filename.replace('\\', '/')
        filename = filename.rsplit('/pc/', 1)[-1] if '/pc/' in filename else filename.rsplit('/', 1)[-1]
        column = '' if self.column is None else f':{self.column}'
        return f'{filename}:{self.line}{column}'


def ast_location(node, source):
    """Read a stored origin, or apply the file and offset of a parsed AST."""
    if isinstance(node, SourceLocation):
        return node
    origin = getattr(node, '_pc_source_location', None)
    filename, offset = source
    if origin is None and filename and hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
        return SourceLocation(filename, node.lineno + offset, node.col_offset)
    return origin


def set_ast_source(node, source):
    """Record the origin of parsed syntax before it is moved to another file."""
    for child in ast.walk(node):
        location = ast_location(child, source)
        if location is not None:
            child._pc_source_location = location


def anchor_generated_ast(node, location):
    """Attach a producer location to syntax without an authored origin."""
    for child in ast.walk(node):
        if 'lineno' in child._attributes and not hasattr(child, '_pc_source_location'):
            child._pc_source_location = location


def copy_ast_location(node, original):
    """Copy AST coordinates and the associated file origin."""
    ast.copy_location(node, original)
    node.__dict__.pop('_pc_source_location', None)
    if hasattr(original, '_pc_source_location'):
        node._pc_source_location = original._pc_source_location
    return node


def class_namespace_key(class_name, member_name):
    """Return the class dictionary key for an identifier from its source."""
    prefix = class_name.lstrip('_')
    if prefix and member_name.startswith('__') and not member_name.endswith('__'):
        return f'_{prefix}{member_name}'
    return member_name


def class_annotation_nodes(class_ast):
    """Index the source declarations by their Python class dictionary keys."""
    if class_ast is None:
        return {}
    return {
        class_namespace_key(class_ast.name, stmt.target.id): stmt
        for stmt in class_ast.body
        if isinstance(stmt, ast.AnnAssign) and stmt.simple
    }


def call_source_name(call):
    """Display the source callee for diagnostics on an already-bound call."""
    return _source_expression(call.func)


def _source_expression(node):
    if isinstance(node, ast.Name):
        return getattr(node, '_pc_source_name', node.id)
    if isinstance(node, ast.Attribute):
        return f'{_source_expression(node.value)}.{node.attr}'
    return ast.unparse(node)


def _diagnose_expansion_at_call(function):
    """Mark a library expansion whose implementation validates a caller's use."""
    function._pc_diagnose_expansion_at_call = True
    return function
