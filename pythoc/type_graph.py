"""PythoC record declarations, recursive views and storage dependencies."""

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import weakref


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
    from .type_resolver import TypeResolver
    pc_type = TypeResolver.resolve_type(pc_type, required=False)
    if pc_type is None:
        return None
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


def _equivalent(first, second, resolver=None):
    """Compare directly while the visited graphs admit a one-to-one mapping.

    Sharing and different cycle unrollings require bisimulation refinement.
    Switching on a mapping conflict bounds pair traversal by graph size,
    while shallow mismatches never need to project the remaining graph.
    """
    if first is second:
        return True
    from .type_descriptor import _wire, type_key
    from .type_resolver import TypeReference, TypeResolver

    def resolve(value):
        if resolver is not None and isinstance(value, (str, TypeReference)):
            resolved = resolver(value)
            if resolved is not None:
                value = resolved
        return _resolve(value)

    forward, backward = {}, {}
    pending = [(first, second)]
    while pending:
        left, right = pending.pop()
        if left is right:
            continue
        a, b = resolve(left), resolve(right)
        if a is None or b is None:
            if TypeResolver.reference_key(left) != TypeResolver.reference_key(right):
                return False
            continue
        if a is b:
            continue
        key_a, key_b = id(a), id(b)
        if key_a in forward and forward[key_a] == key_b:
            continue
        record_a, record_b = named_record(a), named_record(b)
        if any(field is None for record in (record_a, record_b) if record is not None
               for field in record._field_types or ()):
            if (record_a is None or record_b is None
                    or _owner(record_a) is not _owner(record_b)):
                return False
            continue
        head_a, children_a = _shape(a)
        head_b, children_b = _shape(b)
        if _wire(head_a) != _wire(head_b) or len(children_a) != len(children_b):
            return False
        if key_a in forward or key_b in backward:
            return type_key(first, resolver, _validate=False) == type_key(
                second, resolver, _validate=False)
        forward[key_a], backward[key_b] = key_b, key_a
        pending.extend((a, b) for (a, _), (b, _) in zip(children_a, children_b))
    return True


@dataclass(frozen=True, eq=False)
class RecordDefinition:
    """A canonical declaration; object identity survives late completion."""

    name: str
    representative: type
    field_names: tuple
    field_types: tuple
    layout: tuple
    declaration: object = None


@dataclass
class _TypeBinding:
    value: object
    frozen: bool = False


class _RecordIndex:
    """Keep the earliest declaration for each interned structural identity."""

    def __init__(self):
        self.views = {}
        self.buckets = {}

    def representative(self, view, key):
        if view not in self.views:
            self.views[view] = len(self.views)
        current = key(view)
        candidate = self.buckets.get(current)
        if candidate is not None and key(candidate) != current:
            del self.buckets[current]
            candidate = None
        if candidate is None or self.views[view] < self.views[candidate]:
            candidate = view
            self.buckets[current] = candidate
        return candidate


class TypeGraph:
    """Session-owned validation and canonicalization of record declarations."""

    def __init__(self):
        self._lock = threading.RLock()
        self._declarations = {}
        self._canonical = {}
        self._by_name = {}
        from .type_descriptor import StructuralTypeIndex
        self._structural_index = StructuralTypeIndex()
        self._definitions = {}
        self._checked = {}
        self._pending_names = {}
        self._pending_references = weakref.WeakKeyDictionary()
        self._dirty_roots = set()
        self.validating = False
        self.ctypes_types = weakref.WeakKeyDictionary()
        self.type_bindings = weakref.WeakKeyDictionary()
        self._preparation = None
        self.resolving_expressions = set()
        self.binding_revision = 0
        self.prepared_types = weakref.WeakSet()
        self.type_scopes = weakref.WeakValueDictionary()

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
        for root, _ in nodes.values():
            pending = [(root, False)]
            while pending:
                current, finish = pending.pop()
                key = _owner(current) if named_record(current) is not None else current
                if finish:
                    active.remove(key)
                    complete.add(key)
                    continue
                if key in active:
                    raise TypeError('Illegal by-value recursion in PythoC type '
                                    + getattr(current, '__name__', str(current)))
                if key in complete:
                    continue
                active.add(key)
                pending.append((current, True))
                for child, stored in nodes[id(current)][1]:
                    if stored:
                        resolved = _resolve(child)
                        if resolved is not None:
                            pending.append((resolved, False))

    def validate(self, root, revision):
        with self._lock:
            if self.validating:
                return
            snapshot = _fields(root) if named_record(root) is not None else None
            if (self._preparation is None and snapshot is not None
                    and self._checked.get(root) == (revision, snapshot)):
                return
            self.validating = True
            declarations = {}
            canonical = {}
            try:
                nodes, records = self._collect(root)
                for view in records:
                    owner = _owner(view)
                    names, fields, layout = _fields(view)
                    declaration = getattr(owner, '_type_declaration', None)
                    definition = declarations.get(owner, self._declarations.get(owner))
                    if definition is None:
                        definition = RecordDefinition(
                            record_name(view), view, names, fields, layout,
                            declaration)
                        declarations[owner] = definition
                    if (definition.field_names != names or definition.layout != layout
                            or len(definition.field_types) != len(fields)
                            or not all(_equivalent(a, b) for a, b in
                                       zip(definition.field_types, fields))):
                        # A declaration's storage projection can change while
                        # its scoped aliases are still awaiting compilation.
                        # A second view must still agree with that projection.
                        if (declaration is None or definition.declaration is not declaration
                                or definition.representative is not view
                                or definition.layout != layout):
                            raise TypeError('Conflicting PythoC record views: '
                                            + definition.name)
                        definition = RecordDefinition(
                            record_name(view), view, names, fields, layout,
                            declaration)
                        declarations[owner] = definition
                    canonical[owner] = definition.representative
                self._check_storage(nodes)
                unresolved, references = set(), set()

                def pending(reference, names):
                    from .type_resolver import TypeReference
                    unresolved.update(names)
                    if isinstance(reference, TypeReference):
                        references.add(reference)

                keys = self._structural_index.intern(root, on_unresolved=pending)

                def key(view):
                    if id(view) not in keys:
                        keys.update(self._structural_index.intern(view))
                    return keys[id(view)]

                for view in records:
                    owner = _owner(view)
                    name = record_name(view)
                    index = self._by_name.setdefault(name, _RecordIndex())
                    canonical[owner] = index.representative(view, key)
                self._watch_pending(root, unresolved, references)
                self._declarations.update(declarations)
                self._canonical.update(canonical)
                self._checked.update((view, (revision, _fields(view))) for view in records)
            finally:
                self.validating = False

    def _watch_pending(self, root, names, references):
        from .session import CompileSession
        for reference in references:
            self._pending_references.setdefault(reference, set()).add(root)
        waiting = self._pending_names.setdefault(root, set())
        for name in names - waiting:
            waiting.add(name)

            def completed(value, root=root, name=name):
                with self._lock:
                    self._pending_names[root].discard(name)
                    self._dirty_roots.add(root)

            CompileSession.current().forward_refs.register_callback(
                name, completed, include_existing=False)

    def canonical_record(self, view, revision):
        with self._lock:
            if self.validating:
                return self._canonical.get(_owner(view), view)
            while self._dirty_roots:
                root = self._dirty_roots.pop()
                self.validate(root, revision)
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

    @contextmanager
    def binding_transaction(self):
        """Commit reachable bindings together, after successful preparation."""
        if self._preparation is not None:
            yield
            return
        bindings, prepared = {}, set()
        self._preparation = (bindings, prepared)
        try:
            yield
            for reference, binding in bindings.items():
                previous = self.type_bindings.get(reference)
                if (previous.value if previous is not None else None) is not binding.value:
                    self.binding_revision += 1
                    self._dirty_roots.update(self._pending_references.get(reference, ()))
                self.type_bindings[reference] = binding
            self.prepared_types.update(prepared)
            for reference in bindings:
                scope = getattr(reference.namespace, 'type_scope', None)
                if scope is not None:
                    scope.finish()
        except BaseException:
            if bindings:
                self.binding_revision += 1
            raise
        finally:
            self._preparation = None

    def get_binding(self, reference):
        if self._preparation is not None and reference in self._preparation[0]:
            return self._preparation[0][reference]
        return self.type_bindings.get(reference)

    def record_bindings(self, references, value, freeze):
        """Record observations while the resolver holds the graph lock."""
        destination = (self.type_bindings if self._preparation is None
                       else self._preparation[0])
        for reference in references:
            previous = self.get_binding(reference)
            if previous is not None and previous.frozen:
                continue
            if (self._preparation is None
                    and (previous.value if previous is not None else None) is not value):
                self.binding_revision += 1
                self._dirty_roots.update(self._pending_references.get(reference, ()))
            destination[reference] = _TypeBinding(value, freeze and value is not None)

    def prepare_type(self, pc_type):
        """Fix reachable bindings before layout, ABI, or LLVM construction."""
        from .type_resolver import TypeResolver
        if pc_type is None:
            return
        with self._lock, self.binding_transaction():
            prepared = self.prepared_types
            pending, seen, active = [(pc_type, True, 0, False)], {}, {}
            complete = True
            while pending:
                current, stored, depth, finish = pending.pop()
                if finish:
                    del active[id(current)]
                    continue
                current = TypeResolver.resolve_type(current, required=stored, freeze=True)
                current = _resolve(current)
                if current is None:
                    complete = complete and stored
                    continue
                if isinstance(current, type) and (
                        current in prepared or current in self._preparation[1]):
                    continue
                # Unions and enums have finite backing storage even when pointers
                # refer back to them; unlike pointer-only aliases, they break the
                # recursive LLVM construction dependency.
                depth += (named_record(current) is not None
                          or getattr(current, '_is_union', False)
                          or getattr(current, '_is_enum', False))
                key = id(current)
                if key in active:
                    if depth <= active[key]:
                        raise TypeError('Recursive type constructors require a named record')
                    continue
                if key in seen:
                    continue
                seen[key] = current
                active[key] = depth
                pending.append((current, stored, depth, True))
                _, children = _shape(current)
                owner = getattr(current, '_python_class', None)
                declaration = getattr(owner, '_type_declaration', None)
                if declaration is not None:
                    # Static fields are absent from the instance graph, but their
                    # bindings also determine this declaration's storage projection.
                    children = tuple((pc_type, True) for _, _, pc_type, _ in declaration)
                pending.extend((child, bool(stored), depth, False) for child, stored in children
                               if stored is not None)
            if seen:
                from .session import CompileSession
                self.validate(pc_type, (CompileSession.current().forward_refs.revision,
                                        self.binding_revision))
            if complete:
                self._preparation[1].update(
                    value for value in seen.values() if isinstance(value, type))


def _current_graph():
    from .session import CompileSession
    session = CompileSession.current()
    return session.type_graph, (session.forward_refs.revision,
                                session.type_graph.binding_revision)


def canonical_record_type(pc_type):
    view = named_record(pc_type)
    if view is None:
        return pc_type
    graph, revision = _current_graph()
    return graph.canonical_record(view, revision)


def record_definition(pc_type):
    prepare_type(pc_type)
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


def prepare_type(pc_type):
    """Prepare a type in the active compilation session."""
    from .session import CompileSession
    CompileSession.current().type_graph.prepare_type(pc_type)
