"""
Match statement exhaustiveness checking

This module implements exhaustiveness checking for match statements using
a pattern matrix algorithm (Maranget-style).

Key design decisions:
- Pattern matrix algorithm is sound for product types (preserves cross-field correlations)
- Guards are treated as potentially False (conservative but sound)
- Shares resolved patterns with code generation
- Wildcard and variable bindings are equivalent for exhaustiveness
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Tuple, Any
from enum import Enum as PyEnum
import ast

from .logger import logger
from .schema_protocol import (
    get_schema_field_names, get_schema_field_types, is_schema_type,
)


class PatternKind(PyEnum):
    """Kinds of normalized patterns"""
    WILDCARD = "wildcard"      # _ or variable binding
    LITERAL = "literal"        # concrete value (True, False, 42, etc.)
    CONSTRUCTOR = "constructor"  # enum variant or struct/tuple constructor
    OR = "or"                  # disjunction of patterns


@dataclass
class NormalizedPattern:
    """Resolved pattern shared by exhaustiveness checking and code generation.

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
    value_ref: Optional[Any] = None
    bindings: Tuple[str, ...] = ()
    pointer_capture: bool = False
    field_indices: Tuple[int, ...] = ()
    payload_index: Optional[int] = None

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
        if is_schema_type(pc_type) and getattr(pc_type, '_is_struct', False):
            field_types = get_schema_field_types(pc_type)
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
        if is_schema_type(pc_type) and getattr(pc_type, '_is_struct', False):
            field_types = get_schema_field_types(pc_type)
            struct_name = getattr(pc_type, '__name__', 'struct')
            # Struct has exactly one constructor (tag=0) with all fields as sub-patterns
            return [(0, struct_name, list(field_types))]

        return []

    @staticmethod
    def get_constructor_arity(pc_type, tag: int) -> int:
        return len(TypeInfo.get_constructor_sub_types(pc_type, tag))

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
        if is_schema_type(pc_type) and getattr(pc_type, '_is_struct', False):
            field_types = get_schema_field_types(pc_type)
            return list(field_types)

        if getattr(pc_type, 'dimensions', None) is not None:
            from .type_descriptor import array_decay_type
            return [array_decay_type(pc_type).pointee_type] * pc_type.dimensions[0]

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
    """Resolve patterns once for both coverage checking and code generation."""

    def __init__(self, visitor):
        self.visitor = visitor

    def normalize_case(self, pattern, subject_types):
        if len(subject_types) == 1:
            return self.normalize(pattern, subject_types[0])
        if isinstance(pattern, ast.MatchOr):
            result = NormalizedPattern.or_pattern([
                self.normalize_case(item, subject_types) for item in pattern.patterns])
        elif isinstance(pattern, ast.MatchAs) and pattern.pattern is None:
            if pattern.name is not None:
                raise TypeError('Multi-subject match requires individual captures')
            result = NormalizedPattern.constructor(0, 'subjects', [
                NormalizedPattern.wildcard(ty) for ty in subject_types])
        elif isinstance(pattern, ast.MatchSequence):
            if len(pattern.patterns) != len(subject_types):
                raise TypeError('Multi-subject pattern count does not match subjects')
            result = NormalizedPattern.constructor(0, 'subjects', [
                self.normalize(item, ty)
                for item, ty in zip(pattern.patterns, subject_types)])
        else:
            raise TypeError('Multi-subject match requires a sequence pattern')
        result.ast_node = pattern
        return result

    def normalize(self, pattern, subject_type):
        from dataclasses import replace
        if isinstance(pattern, ast.MatchAs):
            result = (NormalizedPattern.wildcard(subject_type)
                      if pattern.pattern is None
                      else self.normalize(pattern.pattern, subject_type))
            if pattern.name is not None:
                result = replace(result, bindings=result.bindings + (pattern.name,))
        elif isinstance(pattern, ast.MatchSingleton):
            if pattern.value is None:
                raise TypeError('None singleton patterns are unsupported')
            result = self._value(ast.Constant(pattern.value), subject_type)
        elif isinstance(pattern, ast.MatchValue):
            result = self._value(pattern.value, subject_type)
        elif isinstance(pattern, ast.MatchOr):
            result = NormalizedPattern.or_pattern([
                self.normalize(item, subject_type) for item in pattern.patterns],
                subject_type)
        elif isinstance(pattern, ast.MatchClass):
            result = self._class(pattern, subject_type)
        elif isinstance(pattern, ast.MatchSequence):
            result = self._sequence(pattern, subject_type)
        else:
            raise NotImplementedError(
                f'Unsupported match pattern: {type(pattern).__name__}')
        result.ast_node = pattern
        return result

    def _resolve_value(self, expression):
        from .type_descriptor import _constant_value
        value = self.visitor.visit_rvalue_expression(expression)
        known, constant = _constant_value(value)
        return value, constant if known else None

    def _value(self, expression, subject_type):
        value_ref, value = self._resolve_value(expression)
        if getattr(subject_type, '_is_enum', False) and value is not None:
            tags = subject_type._tag_values
            for index, name in enumerate(subject_type._variant_names):
                if tags[name] == value:
                    fields = TypeInfo.get_constructor_sub_types(subject_type, value)
                    result = NormalizedPattern.constructor(value, name, [
                        NormalizedPattern.wildcard(ty) for ty in fields], subject_type)
                    result.payload_index = index
                    result.value_ref = value_ref
                    return result
            raise TypeError(f'{value!r} is not a tag of {subject_type.get_name()}')
        result = NormalizedPattern.literal(value, subject_type)
        result.value_ref = value_ref
        return result

    def _class(self, pattern, subject_type):
        from .builtin_entities import ptr
        from .type_converter import strip_qualifiers
        from .type_descriptor import same_type
        from .type_resolver import TypeResolver
        _, pattern_type = self._resolve_value(pattern.cls)
        if pattern_type is ptr:
            if pattern.kwd_attrs or len(pattern.patterns) != 1:
                raise TypeError('ptr pattern requires one capture name')
            inner = pattern.patterns[0]
            if not isinstance(inner, ast.MatchAs) or inner.pattern is not None:
                raise TypeError('ptr pattern requires a capture name')
            result = NormalizedPattern.wildcard(subject_type)
            result.pointer_capture = True
            result.bindings = (inner.name,) if inner.name is not None else ()
            return result
        pattern_type = TypeResolver._require_type(pattern_type)
        if not same_type(strip_qualifiers(pattern_type), strip_qualifiers(subject_type)):
            raise TypeError(
                f'Pattern type {pattern_type.get_name()} does not match '
                f'subject type {subject_type.get_name()}')
        if not is_schema_type(pattern_type) or getattr(pattern_type, '_is_enum', False):
            raise TypeError('Class patterns require a struct type')
        fields = get_schema_field_types(pattern_type)
        names = get_schema_field_names(pattern_type)
        if len(pattern.patterns) > len(fields):
            raise TypeError('Too many positional fields in class pattern')
        specified = dict(enumerate(pattern.patterns))
        for name, item in zip(pattern.kwd_attrs, pattern.kwd_patterns):
            if name not in names:
                raise TypeError(f'Unknown pattern field: {name}')
            index = names.index(name)
            if index in specified:
                raise TypeError(f'Duplicate pattern field: {name}')
            specified[index] = item
        subs = [self.normalize(specified[index], ty) if index in specified
                else NormalizedPattern.wildcard(ty)
                for index, ty in enumerate(fields)]
        result = NormalizedPattern.constructor(0, 'struct', subs, subject_type)
        result.field_indices = tuple(specified)
        return result

    def _sequence(self, pattern, subject_type):
        if (getattr(subject_type, '_is_enum', False) and pattern.patterns
                and isinstance(pattern.patterns[0], ast.MatchValue)):
            result = self._value(pattern.patterns[0].value, subject_type)
            if result.kind == PatternKind.LITERAL:
                if len(pattern.patterns) == 1:
                    return result
                if len(pattern.patterns) != 2:
                    raise TypeError('Enum patterns accept a tag and one payload pattern')
                fields = get_schema_field_types(subject_type)
                result = NormalizedPattern.constructor(None, 'enum-fields', [
                    result, self.normalize(pattern.patterns[1], fields[1])], subject_type)
                result.field_indices = (0, 1)
                return result
            if len(pattern.patterns) > 2:
                raise TypeError('Enum patterns accept a tag and one payload pattern')
            if len(pattern.patterns) == 2:
                fields = TypeInfo.get_constructor_sub_types(
                    subject_type, result.constructor_tag)
                item = pattern.patterns[1]
                if fields:
                    result.sub_patterns = [self.normalize(item, fields[0])]
                    result.field_indices = (0,)
                elif not (isinstance(item, ast.MatchAs) and item.pattern is None
                          and item.name is None):
                    raise TypeError('This enum variant has no payload to match')
            return result
        fields = get_schema_field_types(subject_type)
        if getattr(subject_type, 'dimensions', None) is not None:
            fields = TypeInfo.get_constructor_sub_types(subject_type, 0)
        if len(pattern.patterns) != len(fields):
            raise TypeError('Sequence pattern length does not match subject fields')
        result = NormalizedPattern.constructor(0, 'sequence', [
            self.normalize(item, ty) for item, ty in zip(pattern.patterns, fields)],
            subject_type)
        result.field_indices = tuple(range(len(fields)))
        return result


def _case_alternatives(pattern, multi_subject):
    if multi_subject and pattern.kind == PatternKind.OR:
        for alternative in pattern.alternatives:
            yield from _case_alternatives(alternative, True)
    else:
        yield pattern.sub_patterns if multi_subject else [pattern]


def _check_linear_variant_exhaustiveness(node, subject_types, patterns):
    from .schema_protocol import is_linear_schema_type
    for index, subject_type in enumerate(subject_types):
        if not getattr(subject_type, '_is_enum', False):
            continue
        linear = {subject_type._tag_values[name]
                  for name, ty in zip(subject_type._variant_names,
                                      subject_type._variant_types)
                  if ty is not None and is_linear_schema_type(ty)}
        if not linear:
            continue
        matched, has_wildcard = set(), False
        for case, pattern in zip(node.cases, patterns):
            for row in _case_alternatives(pattern, len(subject_types) > 1):
                item = row[index]
                alternatives = (item.alternatives if item.kind == PatternKind.OR
                                else [item])
                for alternative in alternatives:
                    has_wildcard |= alternative.is_wildcard()
                    if case.guard is None and alternative.kind == PatternKind.CONSTRUCTOR:
                        matched.add(alternative.constructor_tag)
        if has_wildcard and linear - matched:
            names = [name for name, tag in subject_type._tag_values.items()
                     if tag in linear - matched]
            logger.error(
                'Match with wildcard does not explicitly handle linear variants: '
                + ', '.join(names) + '. Linear payloads must be explicitly matched '
                'and consumed.', node=node, exc_type=ValueError)


def check_match_exhaustiveness(node, subject_types, patterns):
    """Check coverage of the same resolved patterns consumed by codegen."""
    _check_linear_variant_exhaustiveness(node, subject_types, patterns)
    rows = [PatternRow(row, case.guard is not None, case)
            for case, pattern in zip(node.cases, patterns)
            for row in _case_alternatives(pattern, len(subject_types) > 1)]
    exhaustive, uncovered = is_exhaustive(PatternMatrix(rows, subject_types))
    if not exhaustive:
        message = 'Non-exhaustive match statement.'
        if uncovered:
            message += " Uncovered cases: " + ', '.join(uncovered)
        if any(case.guard is not None for case in node.cases):
            message += ('\nNote: Guard conditions (if clauses) are treated as '
                        'potentially False. Add a wildcard case (_) to ensure '
                        'exhaustiveness.')
        logger.error(message, node=node, exc_type=ValueError)
