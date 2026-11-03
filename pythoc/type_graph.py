"""PythoC record declarations, recursive views and storage dependencies."""

from dataclasses import dataclass
import threading


def record_view(pc_type):
    """Unwrap a decorated class without consulting a backend representation."""
    if getattr(pc_type, '_is_refined', False):
        return None
    view = getattr(pc_type, '_struct_type', None)
    if getattr(view, '_pc_record', False):
        pc_type = view
    return pc_type if getattr(pc_type, '_pc_record', False) else None


def named_record(pc_type):
    view = record_view(pc_type)
    if view is not None and (view._python_class is not None or view._force_identified):
        return view
    return None


def record_name(view):
    name = view.get_name()
    suffix = getattr(view._python_class, '_anonymous_suffix', None)
    return name + suffix if suffix else name


def _owner(view):
    return view._python_class if view._python_class is not None else view


def _resolve(pc_type):
    if isinstance(pc_type, str):
        from .forward_ref import get_defined_type
        resolved = get_defined_type(pc_type)
        return _resolve(resolved) if resolved is not None else None
    view = record_view(pc_type)
    if view is not None:
        return view
    union_view = getattr(pc_type, '_union_type', None)
    return union_view if union_view is not None else pc_type


def _fields(view):
    view._ensure_field_types_resolved()
    fields = tuple(view._field_types or ())
    names = tuple(view._field_names or (None,) * len(fields))
    if len(names) != len(fields):
        raise TypeError(f'Incomplete PythoC record definition: {view.get_name()}')
    return names, fields, view._layout_config.identity()


def _is_union(pc_type):
    from .builtin_entities.union import UnionType
    return isinstance(pc_type, type) and issubclass(pc_type, UnionType)


def _shape(pc_type):
    """Describe semantic constructors and whether their edges store a value."""
    if pc_type is None:
        return ('void',), ()
    qualified = getattr(pc_type, 'qualified_type', None)
    if qualified is not None:
        flags = tuple(sorted(pc_type.get_qualifier_flags().items()))
        return ('qualified', flags), ((qualified, True),)
    if getattr(pc_type, '_is_refined', False):
        base = pc_type._base_type
        if base is None:
            base = pc_type._struct_type
        from .type_descriptor import leaf_key, predicate_shape
        if base is None and pc_type._predicates is None:
            return ('leaf', leaf_key(pc_type)), ()
        shapes = [predicate_shape(pred) for pred in pc_type._predicates or ()]
        predicates = tuple((head, len(types)) for head, types in shapes)
        children = ((base, True),) + tuple(
            (parameter, None) for _, types in shapes for parameter in types)
        return ('refined', predicates, tuple(pc_type._tags or ())), children
    params = getattr(pc_type, 'param_types', None)
    if params is None and getattr(pc_type, '_is_function_type', False):
        from .type_descriptor import leaf_key
        return ('leaf', leaf_key(pc_type)), ()
    if params is not None:
        flags = tuple(bool(getattr(pc_type, name, False)) for name in (
            'has_varargs', 'has_llvm_varargs', 'has_kwargs'))
        names = pc_type._get_parameter_names()
        components = (*params, pc_type.return_type)
        head = ('function', names, flags, pc_type.calling_convention)
        return head, tuple((t, False) for t in components)
    if getattr(pc_type, '_is_pointer', False):
        from .builtin_entities.types import void
        pointee = getattr(pc_type, 'pointee_type', None)
        if pointee is None or pointee is void:
            return ('pointer',), ()
        return ('pointer',), ((pointee, False),)
    dimensions = getattr(pc_type, 'dimensions', None)
    if dimensions is not None:
        return ('array', tuple(dimensions)), ((pc_type.element_type, True),)
    lanes = getattr(pc_type, 'lanes', None)
    if lanes is not None and getattr(pc_type, 'element_type', None) is not None:
        return ('simd', lanes), ((pc_type.element_type, True),)
    fields = getattr(pc_type, '_field_types', None)
    if fields is not None:
        ensure = getattr(pc_type, '_ensure_field_types_resolved', None)
        if ensure is not None:
            ensure()
            fields = pc_type._field_types
        names = tuple(pc_type._field_names or (None,) * len(fields))
        if len(names) != len(fields):
            raise TypeError('Incomplete PythoC record definition: '
                            + pc_type.get_name())
        kind = ('enum' if getattr(pc_type, '_tag_type', None) is not None
                else 'union' if _is_union(pc_type) else 'struct')
        name = (record_name(pc_type) if named_record(pc_type) is not None
                else pc_type.get_name())
        layout = pc_type._layout_config.identity()
        if kind == 'enum':
            from .type_descriptor import atom
            variants = (tuple(pc_type._variant_names or ()),
                        atom(pc_type._tag_values),
                        tuple(pc_type._variant_types or ()))
            return (kind, name, names, layout, variants[:2]), (
                tuple((t, True) for t in fields)
                + tuple((t, None) for t in variants[2]))
        return (kind, name, names, layout), tuple((t, True) for t in fields)
    from .type_descriptor import leaf_key
    return ('leaf', leaf_key(pc_type)), ()


def _equivalent(first, second, seen=None, resolver=None):
    if first is second:
        return True
    def resolve(current):
        if isinstance(current, str) and resolver is not None:
            local = resolver(current)
            if local is not None:
                current = local
        return _resolve(current)

    a, b = resolve(first), resolve(second)
    if a is None or b is None:
        return isinstance(first, str) and first == second
    first_record, second_record = named_record(a), named_record(b)
    if first_record is not None or second_record is not None:
        first_fields = tuple(getattr(first_record, '_field_types', ()) or ())
        second_fields = tuple(getattr(second_record, '_field_types', ()) or ())
        if any(field is None for field in first_fields + second_fields):
            return (first_record is not None and second_record is not None
                    and _owner(first_record) is _owner(second_record))
    seen = set() if seen is None else seen
    pair = (id(a), id(b))
    if pair in seen:
        return True
    seen.add(pair)
    head_a, children_a = _shape(a)
    head_b, children_b = _shape(b)
    from .type_descriptor import _wire
    return (_wire(head_a) == _wire(head_b) and len(children_a) == len(children_b)
            and all(_equivalent(x, y, seen, resolver)
                    for (x, _), (y, _) in zip(children_a, children_b)))


@dataclass(frozen=True, eq=False)
class RecordDefinition:
    """A canonical declaration; object identity survives late completion."""

    name: str
    representative: type
    field_names: tuple
    field_types: tuple
    layout: tuple


class TypeGraph:
    """Session-owned validation and canonicalization of record declarations."""

    def __init__(self):
        self._lock = threading.RLock()
        self._declarations = {}
        self._canonical = {}
        self._by_name = {}
        self._definitions = {}
        self._checked = {}
        self._revision = -1
        self.validating = False

    def _collect(self, root):
        pending, nodes, records = [(root, True)], {}, []
        while pending:
            original, stored = pending.pop()
            pc_type = _resolve(original)
            if pc_type is None:
                if stored:
                    raise TypeError(f'Incomplete PythoC value type: {original}')
                continue
            key = id(pc_type)
            if key in nodes:
                continue
            _, children = _shape(pc_type)
            nodes[key] = (pc_type, children)
            if named_record(pc_type) is not None:
                records.append(pc_type)
            # Identity-only enum variant edges are compared and projected,
            # but the union payload already owns their storage validation.
            pending.extend((child, stored) for child, stored in children
                           if stored is not None)
        return nodes, records

    def _check_storage(self, nodes):
        active, complete = set(), set()

        def visit(pc_type):
            key = _owner(pc_type) if named_record(pc_type) is not None else pc_type
            if key in active:
                raise TypeError('Illegal by-value recursion in PythoC type '
                                + getattr(pc_type, '__name__', str(pc_type)))
            if key in complete:
                return
            active.add(key)
            for child, stored in nodes[id(pc_type)][1]:
                resolved = _resolve(child)
                if stored and resolved is not None:
                    visit(resolved)
            active.remove(key)
            complete.add(key)

        for pc_type, _ in nodes.values():
            visit(pc_type)

    def validate(self, root, revision):
        with self._lock:
            if self._revision != revision:
                self._checked.clear()
                self._revision = revision
            snapshot = _fields(root) if named_record(root) is not None else None
            if snapshot is not None and self._checked.get(root) == snapshot:
                return
            if self.validating:
                return
            self.validating = True
            declarations = self._declarations.copy()
            canonical = self._canonical.copy()
            by_name = {name: list(items) for name, items in self._by_name.items()}
            try:
                nodes, records = self._collect(root)
                for view in records:
                    owner = _owner(view)
                    names, fields, layout = _fields(view)
                    definition = declarations.get(owner)
                    if definition is None:
                        definition = RecordDefinition(
                            record_name(view), view, names, fields, layout)
                        declarations[owner] = definition
                    if (definition.field_names != names or definition.layout != layout
                            or len(definition.field_types) != len(fields)
                            or not all(_equivalent(a, b) for a, b in
                                       zip(definition.field_types, fields))):
                        raise TypeError('Conflicting PythoC record views: '
                                        + definition.name)
                    canonical.setdefault(owner, definition.representative)
                self._check_storage(nodes)
                for view in records:
                    owner = _owner(view)
                    name = record_name(view)
                    candidates = by_name.setdefault(name, [])
                    equivalent = next((other for other in candidates
                                       if _equivalent(view, other)), None)
                    if equivalent is not None:
                        canonical[owner] = equivalent
                    else:
                        candidates.append(view)
                self._declarations = declarations
                self._canonical = canonical
                self._by_name = by_name
                self._checked.update((view, _fields(view)) for view in records)
            finally:
                self.validating = False

    def canonical_record(self, view, revision):
        with self._lock:
            if self.validating:
                return self._canonical.get(_owner(view), view)
            self.validate(view, revision)
            return self._canonical[_owner(view)]

    def definition(self, view, revision):
        # A canonical declaration remains the same type when a forward
        # reference completes. Mangling strings are not declaration identities.
        with self._lock:
            canonical = self.canonical_record(view, revision)
            definition = self._definitions.get(canonical)
            if definition is None:
                names, fields, layout = _fields(canonical)
                definition = RecordDefinition(
                    record_name(canonical), canonical, names, fields, layout)
                self._definitions[canonical] = definition
            return definition


def _current_graph():
    from .session import CompileSession
    session = CompileSession.current()
    return session.type_graph, session.forward_refs.revision


def canonical_record_type(pc_type):
    view = named_record(pc_type)
    if view is None:
        return pc_type
    graph, revision = _current_graph()
    return graph.canonical_record(view, revision)


def record_definition(pc_type):
    graph, revision = _current_graph()
    return graph.definition(named_record(pc_type), revision)


def validate_type_graph(pc_type):
    graph, revision = _current_graph()
    view = record_view(pc_type)
    graph.validate(view if view is not None else pc_type, revision)


def validation_in_progress():
    from .session import CompileSession
    try:
        return CompileSession.current().type_graph.validating
    except RuntimeError:
        return False
