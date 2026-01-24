"""
Match statement exhaustiveness checking

This module implements exhaustiveness checking for match statements using
a pattern matrix algorithm (Maranget-style).

Key design decisions:
- Pattern matrix algorithm is sound for product types (preserves cross-field correlations)
- Guards are treated as potentially False (conservative but sound)
- Reuses pattern semantics from stmt_match.py to avoid drift
- Wildcard and variable bindings are equivalent for exhaustiveness
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Tuple, Any, Union
from enum import Enum as PyEnum
import ast

from .logger import logger


class PatternKind(PyEnum):
    """Kinds of normalized patterns"""
    WILDCARD = "wildcard"      # _ or variable binding
    LITERAL = "literal"        # concrete value (True, False, 42, etc.)
    CONSTRUCTOR = "constructor"  # enum variant or struct/tuple constructor
    OR = "or"                  # disjunction of patterns


@dataclass
class NormalizedPattern:
    """Normalized pattern for exhaustiveness checking.

    This is a shared representation that captures pattern semantics
    without AST-specific details.
    """
    kind: PatternKind
    # For LITERAL: the value
    value: Optional[Any] = None
    # For CONSTRUCTOR: tag/variant info + sub-patterns
    constructor_tag: Optional[int] = None
    constructor_name: Optional[str] = None
    sub_patterns: List['NormalizedPattern'] = field(default_factory=list)
    # For OR: list of alternatives
    alternatives: List['NormalizedPattern'] = field(default_factory=list)
    # Type information for this pattern position
    type_info: Optional[Any] = None
    # Original AST node (for error messages)
    ast_node: Optional[Any] = None

    @staticmethod
    def wildcard(type_info: Any = None) -> 'NormalizedPattern':
        return NormalizedPattern(kind=PatternKind.WILDCARD, type_info=type_info)

    @staticmethod
    def literal(value: Any, type_info: Any = None) -> 'NormalizedPattern':
        return NormalizedPattern(kind=PatternKind.LITERAL, value=value, type_info=type_info)

    @staticmethod
    def constructor(tag: int, name: str, subs: List['NormalizedPattern'],
                   type_info: Any = None) -> 'NormalizedPattern':
        return NormalizedPattern(
            kind=PatternKind.CONSTRUCTOR,
            constructor_tag=tag,
            constructor_name=name,
            sub_patterns=subs or [],
            type_info=type_info
        )

    @staticmethod
    def or_pattern(alternatives: List['NormalizedPattern'],
                  type_info: Any = None) -> 'NormalizedPattern':
        return NormalizedPattern(
            kind=PatternKind.OR,
            alternatives=alternatives,
            type_info=type_info
        )

    def is_wildcard(self) -> bool:
        return self.kind == PatternKind.WILDCARD


@dataclass
class PatternRow:
    """A row in the pattern matrix (one case clause)."""
    patterns: List[NormalizedPattern]  # One per column (subject component)
    has_guard: bool = False
    original_case: Optional[Any] = None  # ast.match_case for error reporting


@dataclass
class PatternMatrix:
    """Matrix of patterns for exhaustiveness checking.

    Rows = case clauses
    Columns = subject components (for multi-subject or tuple destructuring)
    """
    rows: List[PatternRow]
    column_types: List[Any]  # Type of each column

    def is_empty(self) -> bool:
        """No rows = no patterns."""
        return len(self.rows) == 0

    def first_column_constructors(self) -> Set[int]:
        """Get all constructor tags in first column (for specialization)."""
        tags = set()
        for row in self.rows:
            if row.patterns and row.patterns[0].kind == PatternKind.CONSTRUCTOR:
                tags.add(row.patterns[0].constructor_tag)
            elif row.patterns and row.patterns[0].kind == PatternKind.OR:
                # Collect constructor tags from OR alternatives
                for alt in row.patterns[0].alternatives:
                    if alt.kind == PatternKind.CONSTRUCTOR:
                        tags.add(alt.constructor_tag)
        return tags

    def first_column_literals(self) -> Set[Any]:
        """Get all literal values in first column."""
        literals = set()
        for row in self.rows:
            if row.patterns and row.patterns[0].kind == PatternKind.LITERAL:
                literals.add(row.patterns[0].value)
            elif row.patterns and row.patterns[0].kind == PatternKind.OR:
                for alt in row.patterns[0].alternatives:
                    if alt.kind == PatternKind.LITERAL:
                        literals.add(alt.value)
        return literals


class TypeInfo:
    """Helper class to provide type information for exhaustiveness checking."""

    @staticmethod
    def is_finite(pc_type, _seen: set = None) -> bool:
        """Check if a type has a finite set of possible values.

        A type is finite if we can enumerate all possible values at compile time.
        - Bool: finite (True, False)
        - Enum: finite (all variants with finite payloads)
        - Struct: finite if ALL fields are finite types
        - Int/Float/Ptr: not finite (too many values)

        Uses _seen set to handle recursive types (which are treated as infinite).
        """
        if pc_type is None:
            return False

        # Handle circular references - treat as infinite
        if _seen is None:
            _seen = set()

        type_id = id(pc_type)
        if type_id in _seen:
            # Recursive type - treat as infinite to avoid infinite loop
            return False
        _seen.add(type_id)

        # Bool is finite (True, False)
        from .builtin_entities import bool as pc_bool
        if pc_type is pc_bool:
            return True

        # Enum types are finite if all payloads are finite
        if hasattr(pc_type, '_is_enum') and pc_type._is_enum:
            variant_types = getattr(pc_type, '_variant_types', [])
            from .builtin_entities.types import void
            for payload_type in variant_types:
                if payload_type is not None and payload_type != void:
                    # Check if payload type is finite
                    if not TypeInfo.is_finite(payload_type, _seen.copy()):
                        # Enum with infinite payload is still considered "finite"
                        # for constructor enumeration - the payload exhaustiveness
                        # will be checked recursively by the pattern matrix algorithm
                        pass
            return True

        # Struct types are finite if ALL fields have finite types
        if hasattr(pc_type, '_is_struct') and pc_type._is_struct:
            field_types = getattr(pc_type, '_field_types', [])
            if not field_types:
                # Empty struct is finite (one constructor, no fields)
                return True
            # Check if all fields are finite
            for field_type in field_types:
                if not TypeInfo.is_finite(field_type, _seen.copy()):
                    return False
            return True

        # Everything else (int, float, ptr, union, etc.) is not finite
        return False

    @staticmethod
    def get_all_constructors(pc_type) -> List[Tuple[int, str, List[Any]]]:
        """Get all constructors for a finite type.

        Returns list of (tag, name, sub_types) tuples.

        For struct types, there's a single constructor (tag=0) with sub_types
        being the field types.
        """
        from .builtin_entities import bool as pc_bool

        # Bool constructors: True (1), False (0)
        if pc_type is pc_bool:
            return [(1, "True", []), (0, "False", [])]

        # Enum constructors: each variant
        if hasattr(pc_type, '_is_enum') and pc_type._is_enum:
            result = []
            variant_names = getattr(pc_type, '_variant_names', [])
            variant_types = getattr(pc_type, '_variant_types', [])
            tag_values = getattr(pc_type, '_tag_values', {})

            for i, name in enumerate(variant_names):
                tag = tag_values.get(name, i)
                payload_type = variant_types[i] if i < len(variant_types) else None
                # Payload sub-types: if has payload, one sub-type
                from .builtin_entities.types import void
                if payload_type is not None and payload_type != void:
                    sub_types = [payload_type]
                else:
                    sub_types = []
                result.append((tag, name, sub_types))
            return result

        # Struct constructors: single constructor with field types as sub-types
        if hasattr(pc_type, '_is_struct') and pc_type._is_struct:
            field_types = getattr(pc_type, '_field_types', [])
            struct_name = getattr(pc_type, '__name__', 'struct')
            # Struct has exactly one constructor (tag=0) with all fields as sub-patterns
            return [(0, struct_name, list(field_types))]

        return []

    @staticmethod
    def get_constructor_arity(pc_type, tag: int) -> int:
        """Get the number of sub-patterns for a constructor."""
        from .builtin_entities import bool as pc_bool

        # Bool has no sub-patterns
        if pc_type is pc_bool:
            return 0

        # Enum: find variant by tag and return 1 if has payload, 0 otherwise
        if hasattr(pc_type, '_is_enum') and pc_type._is_enum:
            variant_names = getattr(pc_type, '_variant_names', [])
            variant_types = getattr(pc_type, '_variant_types', [])
            tag_values = getattr(pc_type, '_tag_values', {})

            for i, name in enumerate(variant_names):
                if tag_values.get(name, i) == tag:
                    payload_type = variant_types[i] if i < len(variant_types) else None
                    from .builtin_entities.types import void
                    if payload_type is not None and payload_type != void:
                        return 1
                    return 0
            return 0

        # Struct: arity is the number of fields
        if hasattr(pc_type, '_is_struct') and pc_type._is_struct:
            field_types = getattr(pc_type, '_field_types', [])
            return len(field_types)

        # Array: arity is the number of elements
        if hasattr(pc_type, 'is_array') and pc_type.is_array():
            # Get array length from type
            length = getattr(pc_type, 'length', None)
            if length is not None:
                return length
            # Fallback: check _field_types
            field_types = getattr(pc_type, '_field_types', [])
            return len(field_types)

        return 0

    @staticmethod
    def get_constructor_sub_types(pc_type, tag: int) -> List[Any]:
        """Get the types of sub-patterns for a constructor."""
        from .builtin_entities import bool as pc_bool

        # Bool has no sub-types
        if pc_type is pc_bool:
            return []

        # Enum: find variant by tag and return payload type
        if hasattr(pc_type, '_is_enum') and pc_type._is_enum:
            variant_names = getattr(pc_type, '_variant_names', [])
            variant_types = getattr(pc_type, '_variant_types', [])
            tag_values = getattr(pc_type, '_tag_values', {})

            for i, name in enumerate(variant_names):
                if tag_values.get(name, i) == tag:
                    payload_type = variant_types[i] if i < len(variant_types) else None
                    from .builtin_entities.types import void
                    if payload_type is not None and payload_type != void:
                        return [payload_type]
                    return []
            return []

        # Struct: sub-types are the field types
        if hasattr(pc_type, '_is_struct') and pc_type._is_struct:
            field_types = getattr(pc_type, '_field_types', [])
            return list(field_types)

        # Array: sub-types are element type repeated for each position
        if hasattr(pc_type, 'is_array') and pc_type.is_array():
            element_type = getattr(pc_type, 'element_type', None)
            length = getattr(pc_type, 'length', None)
            if element_type and length:
                return [element_type] * length
            # Fallback: check _field_types
            field_types = getattr(pc_type, '_field_types', [])
            return list(field_types)

        return []

    @staticmethod
    def describe_constructor(pc_type, tag: int) -> str:
        """Get a human-readable name for a constructor."""
        from .builtin_entities import bool as pc_bool

        if pc_type is pc_bool:
            return "True" if tag == 1 else "False"

        if hasattr(pc_type, '_is_enum') and pc_type._is_enum:
            variant_names = getattr(pc_type, '_variant_names', [])
            tag_values = getattr(pc_type, '_tag_values', {})
            type_name = getattr(pc_type, '__name__', 'enum')

            for name in variant_names:
                if tag_values.get(name) == tag:
                    return f"{type_name}.{name}"
            return f"{type_name}.<tag={tag}>"

        if hasattr(pc_type, '_is_struct') and pc_type._is_struct:
            struct_name = getattr(pc_type, '__name__', 'struct')
            return struct_name

        return f"<tag={tag}>"


def is_exhaustive(matrix: PatternMatrix) -> Tuple[bool, List[str]]:
    """Check if pattern matrix is exhaustive.

    Uses Maranget-style pattern matrix algorithm.

    Returns (is_exhaustive, list_of_uncovered_patterns).
    """
    if matrix.is_empty():
        # No patterns = not exhaustive (unless no possible values)
        if not matrix.column_types:
            return (True, [])  # No columns = trivially exhaustive
        return (False, ["_"])

    if not matrix.rows[0].patterns:
        # No columns = exhaustive if we have at least one row without guard
        for row in matrix.rows:
            if not row.has_guard:
                return (True, [])
        return (False, ["_"])

    # Get the type of first column
    col_type = matrix.column_types[0] if matrix.column_types else None

    # Case 1: First column has a wildcard/catch-all row without guard
    for row in matrix.rows:
        if row.patterns[0].is_wildcard() and not row.has_guard:
            # Check remaining columns with this row specialized
            remaining = specialize_default(matrix)
            return is_exhaustive(remaining)

    # Case 2: First column type is finite (bool, enum, struct with finite fields)
    if TypeInfo.is_finite(col_type):
        all_constructors = TypeInfo.get_all_constructors(col_type)
        covered_constructors = matrix.first_column_constructors()

        # Also check for literals that might represent constructors (bool case)
        covered_literals = matrix.first_column_literals()

        uncovered = []
        for ctor_tag, ctor_name, ctor_sub_types in all_constructors:
            # Check if covered by constructor pattern
            in_constructors = ctor_tag in covered_constructors
            # Check if covered by literal (for bool: True=1, False=0)
            in_literals = False
            from .builtin_entities import bool as pc_bool
            if col_type is pc_bool:
                if ctor_tag == 1 and True in covered_literals:
                    in_literals = True
                elif ctor_tag == 0 and False in covered_literals:
                    in_literals = True

            if not in_constructors and not in_literals:
                # This constructor is not covered at all
                uncovered.append(TypeInfo.describe_constructor(col_type, ctor_tag))
            else:
                # Check if this constructor's sub-patterns are exhaustive
                specialized = specialize(matrix, col_type, ctor_tag)
                sub_exhaustive, sub_uncovered = is_exhaustive(specialized)
                if not sub_exhaustive:
                    ctor_desc = TypeInfo.describe_constructor(col_type, ctor_tag)
                    for u in sub_uncovered:
                        uncovered.append(f"({ctor_desc}, {u})")

        return (len(uncovered) == 0, uncovered)

    # Case 3: Struct type with infinite fields - still has single constructor
    # We can specialize on the struct constructor and check field patterns
    if hasattr(col_type, '_is_struct') and col_type._is_struct:
        # Struct has exactly one constructor (tag=0)
        # Check if any row has a constructor pattern for structs
        covered_constructors = matrix.first_column_constructors()

        if 0 in covered_constructors:
            # Specialize on the struct constructor and check sub-patterns
            specialized = specialize(matrix, col_type, 0)
            sub_exhaustive, sub_uncovered = is_exhaustive(specialized)
            if not sub_exhaustive:
                struct_name = getattr(col_type, '__name__', 'struct')
                return (False, [f"({struct_name}, {u})" for u in sub_uncovered])
            return (True, [])
        else:
            # No struct constructor pattern - not exhaustive
            struct_name = getattr(col_type, '__name__', 'struct')
            return (False, [f"_ (catch-all required for {struct_name})"])

    # Case 4: Array type - has single constructor, check element patterns
    if hasattr(col_type, 'is_array') and col_type.is_array():
        # Array has exactly one constructor (tag=0)
        covered_constructors = matrix.first_column_constructors()

        if 0 in covered_constructors:
            # Specialize on the array constructor and check element patterns
            specialized = specialize(matrix, col_type, 0)
            sub_exhaustive, sub_uncovered = is_exhaustive(specialized)
            if not sub_exhaustive:
                array_name = getattr(col_type, '__name__', 'array')
                return (False, [f"({array_name}, {u})" for u in sub_uncovered])
            return (True, [])
        else:
            # No array constructor pattern - not exhaustive
            array_name = getattr(col_type, '__name__', 'array')
            return (False, [f"_ (catch-all required for {array_name})"])

    # Case 5: First column type is infinite (int, float, etc.)
    # Must have a wildcard - already checked above, so not exhaustive
    type_name = getattr(col_type, '__name__', str(col_type)) if col_type else 'unknown'
    return (False, [f"_ (catch-all required for {type_name})"])


def specialize(matrix: PatternMatrix, col_type: Any, constructor_tag: int) -> PatternMatrix:
    """Specialize matrix for a specific constructor.

    - Keep rows where first column matches constructor or is wildcard
    - Replace first column with constructor's sub-patterns
    """
    new_rows = []
    arity = TypeInfo.get_constructor_arity(col_type, constructor_tag)
    sub_types = TypeInfo.get_constructor_sub_types(col_type, constructor_tag)

    for row in matrix.rows:
        first = row.patterns[0]

        if first.kind == PatternKind.CONSTRUCTOR and first.constructor_tag == constructor_tag:
            # Exact match: expand sub-patterns
            new_patterns = first.sub_patterns + row.patterns[1:]
            new_rows.append(PatternRow(new_patterns, row.has_guard, row.original_case))

        elif first.kind == PatternKind.LITERAL:
            # Check if literal matches constructor (for bool: True=1, False=0)
            from .builtin_entities import bool as pc_bool
            if col_type is pc_bool:
                literal_tag = 1 if first.value is True else (0 if first.value is False else None)
                if literal_tag == constructor_tag:
                    # Literal match: no sub-patterns for bool
                    new_patterns = row.patterns[1:]
                    new_rows.append(PatternRow(new_patterns, row.has_guard, row.original_case))

        elif first.is_wildcard():
            # Wildcard matches any constructor: expand with wildcards for sub-patterns
            wildcards = [NormalizedPattern.wildcard(t) for t in sub_types]
            new_patterns = wildcards + row.patterns[1:]
            new_rows.append(PatternRow(new_patterns, row.has_guard, row.original_case))

        elif first.kind == PatternKind.OR:
            # OR pattern: check if any alternative matches this constructor
            for alt in first.alternatives:
                if alt.kind == PatternKind.CONSTRUCTOR and alt.constructor_tag == constructor_tag:
                    # This alternative matches
                    new_patterns = alt.sub_patterns + row.patterns[1:]
                    new_rows.append(PatternRow(new_patterns, row.has_guard, row.original_case))
                    break
                elif alt.kind == PatternKind.LITERAL:
                    # Check literal for bool
                    from .builtin_entities import bool as pc_bool
                    if col_type is pc_bool:
                        literal_tag = 1 if alt.value is True else (0 if alt.value is False else None)
                        if literal_tag == constructor_tag:
                            new_patterns = row.patterns[1:]
                            new_rows.append(PatternRow(new_patterns, row.has_guard, row.original_case))
                            break
        # Otherwise: row doesn't match, skip

    # Update column types
    new_col_types = sub_types + matrix.column_types[1:]

    return PatternMatrix(new_rows, new_col_types)


def specialize_default(matrix: PatternMatrix) -> PatternMatrix:
    """Default specialization: keep rows with wildcard in first column."""
    new_rows = []
    for row in matrix.rows:
        if row.patterns[0].is_wildcard():
            new_rows.append(PatternRow(row.patterns[1:], row.has_guard, row.original_case))

    return PatternMatrix(new_rows, matrix.column_types[1:])


class PatternNormalizer:
    """Convert AST patterns to normalized form.

    This class captures the pattern semantics used by stmt_match.py
    to ensure the exhaustiveness checker and codegen agree.
    """

    def __init__(self, visitor=None):
        """Initialize normalizer.

        Args:
            visitor: Optional AST visitor for expression evaluation
        """
        self.visitor = visitor

    def normalize(self, pattern: ast.pattern, subject_type: Any) -> NormalizedPattern:
        """Convert AST pattern to normalized form."""

        # MatchAs: wildcard or binding (possibly with sub-pattern)
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is None:
                # case _: or case x: - both are wildcards for exhaustiveness
                return NormalizedPattern.wildcard(subject_type)
            else:
                # case P(...) as x: recurse into sub-pattern
                return self.normalize(pattern.pattern, subject_type)

        # MatchSingleton: True, False, None
        if isinstance(pattern, ast.MatchSingleton):
            return NormalizedPattern.literal(pattern.value, subject_type)

        # MatchValue: literal value
        if isinstance(pattern, ast.MatchValue):
            value = self._extract_constant_value(pattern.value)

            # Check if this is an enum variant constant
            if self._is_enum_variant(pattern.value, subject_type):
                tag = self._resolve_enum_tag(pattern.value, subject_type)
                name = self._get_variant_name(subject_type, tag)
                return NormalizedPattern.constructor(tag, name, [], subject_type)

            return NormalizedPattern.literal(value, subject_type)

        # MatchOr: expand into alternatives
        if isinstance(pattern, ast.MatchOr):
            alts = [self.normalize(p, subject_type) for p in pattern.patterns]
            return NormalizedPattern.or_pattern(alts, subject_type)

        # MatchSequence: tuple/struct pattern or enum pattern
        if isinstance(pattern, ast.MatchSequence):
            return self._normalize_sequence(pattern, subject_type)

        # MatchClass: struct destructuring
        if isinstance(pattern, ast.MatchClass):
            return self._normalize_class(pattern, subject_type)

        # Fallback: treat as wildcard (conservative)
        return NormalizedPattern.wildcard(subject_type)

    def _normalize_sequence(self, pattern: ast.MatchSequence,
                           subject_type: Any) -> NormalizedPattern:
        """Normalize sequence pattern.

        Handles:
        - Enum tuple syntax: (Status.Ok, payload)
        - Regular tuple/struct patterns
        """
        # Check if this is enum pattern matching: (EnumClass.Variant, x)
        is_enum = hasattr(subject_type, '_is_enum') and subject_type._is_enum

        if is_enum and len(pattern.patterns) >= 1:
            first_pattern = pattern.patterns[0]

            # Check if first element is a variant tag
            if isinstance(first_pattern, ast.MatchValue):
                if self._is_enum_variant(first_pattern.value, subject_type):
                    tag = self._resolve_enum_tag(first_pattern.value, subject_type)
                    name = self._get_variant_name(subject_type, tag)

                    # Get payload pattern(s)
                    sub_patterns = []
                    if len(pattern.patterns) > 1:
                        # Has payload
                        payload_type = self._get_variant_payload_type(subject_type, tag)
                        payload_pattern = self.normalize(pattern.patterns[1], payload_type)
                        sub_patterns = [payload_pattern]

                    return NormalizedPattern.constructor(tag, name, sub_patterns, subject_type)

        # Regular tuple/struct pattern - treat as single constructor with sub-patterns
        field_types = self._get_field_types(subject_type)
        sub_patterns = []
        for i, sub_pat in enumerate(pattern.patterns):
            field_type = field_types[i] if i < len(field_types) else None
            sub_patterns.append(self.normalize(sub_pat, field_type))

        return NormalizedPattern.constructor(0, "tuple", sub_patterns, subject_type)

    def _normalize_class(self, pattern: ast.MatchClass,
                        subject_type: Any) -> NormalizedPattern:
        """Normalize class (struct) pattern."""
        # Get field types from subject type
        field_names = getattr(subject_type, '_field_names', [])
        field_types = getattr(subject_type, '_field_types', [])

        # Build sub-patterns in field order
        sub_patterns = []
        for fname, ftype in zip(field_names, field_types):
            # Check if this field is specified in pattern
            if fname in pattern.kwd_attrs:
                idx = pattern.kwd_attrs.index(fname)
                sub_pat = self.normalize(pattern.kwd_patterns[idx], ftype)
            else:
                # Field not in pattern = wildcard
                sub_pat = NormalizedPattern.wildcard(ftype)
            sub_patterns.append(sub_pat)

        return NormalizedPattern.constructor(0, "struct", sub_patterns, subject_type)

    def _extract_constant_value(self, node: ast.expr) -> Any:
        """Extract constant value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant):
                return -node.operand.value
        # Can't determine constant value
        return None

    def _is_enum_variant(self, node: ast.expr, subject_type: Any) -> bool:
        """Check if AST node is an enum variant access like Status.Ok."""
        if not hasattr(subject_type, '_is_enum') or not subject_type._is_enum:
            return False

        if isinstance(node, ast.Attribute):
            # Status.Ok -> check if 'Ok' is a variant
            variant_names = getattr(subject_type, '_variant_names', [])
            return node.attr in variant_names

        return False

    def _resolve_enum_tag(self, node: ast.expr, subject_type: Any) -> int:
        """Resolve enum variant node to tag value."""
        if isinstance(node, ast.Attribute):
            attr_name = node.attr
            tag_values = getattr(subject_type, '_tag_values', {})
            return tag_values.get(attr_name, 0)

        # Try constant evaluation
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value

        return 0

    def _get_variant_name(self, subject_type: Any, tag: int) -> str:
        """Get variant name from tag."""
        if hasattr(subject_type, '_is_enum') and subject_type._is_enum:
            variant_names = getattr(subject_type, '_variant_names', [])
            tag_values = getattr(subject_type, '_tag_values', {})
            for name in variant_names:
                if tag_values.get(name) == tag:
                    return name
        return f"<tag={tag}>"

    def _get_variant_payload_type(self, subject_type: Any, tag: int) -> Any:
        """Get payload type for enum variant."""
        if hasattr(subject_type, '_is_enum') and subject_type._is_enum:
            variant_names = getattr(subject_type, '_variant_names', [])
            variant_types = getattr(subject_type, '_variant_types', [])
            tag_values = getattr(subject_type, '_tag_values', {})

            for i, name in enumerate(variant_names):
                if tag_values.get(name) == tag:
                    return variant_types[i] if i < len(variant_types) else None
        return None

    def _get_field_types(self, subject_type: Any) -> List[Any]:
        """Get field types for struct/tuple type."""
        if hasattr(subject_type, '_field_types'):
            return subject_type._field_types
        return []


def check_match_exhaustiveness(node: ast.Match, subject_types: List[Any],
                               visitor=None) -> None:
    """Check if a match statement is exhaustive.

    Raises CompileError if not exhaustive.

    Args:
        node: ast.Match node
        subject_types: List of types for subjects
        visitor: Optional AST visitor for expression evaluation
    """
    # Fast path: unguarded catch-all = exhaustive
    for case in node.cases:
        pattern = case.pattern
        if isinstance(pattern, ast.MatchAs) and pattern.pattern is None:
            if case.guard is None:
                return  # Immediately exhaustive

    # Build pattern matrix
    normalizer = PatternNormalizer(visitor)
    rows = []

    for case in node.cases:
        # Normalize the pattern
        if len(subject_types) == 1:
            normalized = normalizer.normalize(case.pattern, subject_types[0])
            row_patterns = [normalized]
        else:
            # Multi-subject: pattern should be a sequence or wildcard
            if isinstance(case.pattern, ast.MatchSequence):
                row_patterns = []
                for i, sub_pat in enumerate(case.pattern.patterns):
                    stype = subject_types[i] if i < len(subject_types) else None
                    row_patterns.append(normalizer.normalize(sub_pat, stype))
                # Pad with wildcards if fewer patterns than subjects
                while len(row_patterns) < len(subject_types):
                    stype = subject_types[len(row_patterns)]
                    row_patterns.append(NormalizedPattern.wildcard(stype))
            elif isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None:
                # Single wildcard/binding for multiple subjects
                # Expand to wildcards for all columns
                row_patterns = [NormalizedPattern.wildcard(stype) for stype in subject_types]
            elif isinstance(case.pattern, ast.MatchOr):
                # OR pattern for multi-subject - normalize and check if any alt is a tuple
                # This handles cases like: case (True, True) | (False, False):
                normalized = normalizer.normalize(case.pattern, subject_types[0] if subject_types else None)
                # If it normalizes to an OR of constructors with sub-patterns, we need to expand
                if normalized.kind == PatternKind.OR:
                    # Check if all alternatives are sequences that we can expand
                    all_tuples = True
                    for alt in normalized.alternatives:
                        if alt.kind != PatternKind.CONSTRUCTOR or len(alt.sub_patterns) != len(subject_types):
                            all_tuples = False
                            break
                    if all_tuples:
                        # Keep as single pattern for now - the algorithm will handle OR alternatives
                        row_patterns = [normalized]
                    else:
                        row_patterns = [normalized]
                else:
                    row_patterns = [normalized]
            else:
                # Single pattern for multiple subjects - normalize as single column
                # This handles complex patterns that apply to a tuple type
                normalized = normalizer.normalize(case.pattern, subject_types[0] if subject_types else None)
                row_patterns = [normalized]

        has_guard = case.guard is not None
        rows.append(PatternRow(row_patterns, has_guard, case))

    matrix = PatternMatrix(rows, subject_types)

    # Check exhaustiveness
    exhaustive, uncovered = is_exhaustive(matrix)

    if not exhaustive:
        # Build error message
        msg = "Non-exhaustive match statement."
        if uncovered:
            msg += f" Uncovered cases: {', '.join(uncovered)}"

        # Check if any guards were present
        has_guards = any(c.guard is not None for c in node.cases)
        if has_guards:
            msg += "\nNote: Guard conditions (if clauses) are treated as potentially False. "
            msg += "Add a wildcard case (_) to ensure exhaustiveness."

        logger.error(msg, node=node, exc_type=ValueError)
