"""Pure projections and relations over authoritative Python type objects.

The Python classes/instances remain the only type definitions. ``type_key``
does not own state: it reads their fields and emits a lossless wire projection.
A digest of that wire form is only an index, never proof of equality.
"""

import base64
from collections import defaultdict, deque
import json
import math
import uuid
import weakref


_process_namespace = uuid.uuid4().hex
_opaque_ids = {}
_next_opaque_id = 0


def _opaque(value):
    """Give unsupported runtime atoms a collision-free process-local identity."""
    global _next_opaque_id
    address = id(value)
    entry = _opaque_ids.get(address)
    if entry is not None and entry[0]() is value:
        return ('opaque', _process_namespace, entry[1])
    _next_opaque_id += 1
    number = _next_opaque_id
    try:
        reference = weakref.ref(value)
    except TypeError:
        # Unweakrefable atoms are rare in a type declaration. Keep them alive
        # so their address cannot be reused for a different semantic atom.
        reference = lambda value=value: value
    _opaque_ids[address] = (reference, number)
    return ('opaque', _process_namespace, number)


def atom(value, _active=None):
    """Project Python values; cyclic edges retain process-local identity."""
    active = set() if _active is None else _active
    container = type(value) in (tuple, list, frozenset, set, dict)
    if not container:
        return _atom(value, active)
    address = id(value)
    if address in active:
        return _opaque(value)
    active.add(address)
    try:
        return _atom(value, active)
    finally:
        active.remove(address)


def _atom(value, active):
    kind = type(value)
    if value is None:
        return ('none',)
    if kind is bool:
        return ('bool', value)
    if kind is int:
        return ('int', str(value))
    if kind is str:
        return ('str', value)
    if kind is bytes:
        return ('bytes', value.hex())
    if kind is float:
        if math.isnan(value):
            return _opaque(value)
        return ('float', value.hex() if math.isfinite(value) else repr(value))
    if kind in (tuple, list):
        return (kind.__name__, tuple(atom(item, active) for item in value))
    if kind in (frozenset, set):
        return (kind.__name__, tuple(sorted(
            (atom(item, active) for item in value), key=_wire)))
    if kind is dict:
        pairs = ((atom(key, active), atom(item, active)) for key, item in value.items())
        return ('dict', tuple(sorted(pairs, key=lambda pair: _wire(pair[0]))))
    return _opaque(value)


def _wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def predicate_shape(predicate):
    """Read a declared builtin predicate family or retain opaque identity."""
    try:
        contract = vars(predicate).get('_pc_refinement_contract')
    except TypeError:
        contract = None
    if contract is None:
        return ('opaque-predicate', atom(predicate)), ()
    info = predicate._func_info
    parameters = tuple(info.param_type_hints[name] for name in info.param_names)
    return ('std-predicate', atom(contract)), parameters


def same_predicate(first, second, context=None):
    """Predicate equality shares the same heads and type edges as identity."""
    first_head, first_types = predicate_shape(first)
    second_head, second_types = predicate_shape(second)
    return (_wire(first_head) == _wire(second_head)
            and len(first_types) == len(second_types)
            and all(same_type(a, b, resolver=context)
                    for a, b in zip(first_types, second_types)))


def leaf_key(pc_type):
    from .builtin_entities.python_type import PythonType
    if isinstance(pc_type, PythonType):
        if pc_type._is_constant:
            return ('pyconst', atom(pc_type._constant_value))
        return ('python-object', _opaque(pc_type._python_object))
    if isinstance(pc_type, type):
        # Builtin scalar declarations have stable, unique names in this
        # module. Unknown user-defined leaves require declaration identity.
        from .builtin_entities import types as builtin_types
        if vars(builtin_types).get(pc_type.__name__) is pc_type:
            return ('builtin', 'pythoc.builtin_entities.types', pc_type.__name__)
        if pc_type.__module__.startswith('pythoc.builtin_entities'):
            import importlib
            module = importlib.import_module(pc_type.__module__)
            if vars(module).get(pc_type.__name__) is pc_type:
                return ('builtin', pc_type.__module__, pc_type.__name__)
        return ('declaration', _opaque(pc_type))
    raise TypeError(f'Unsupported PythoC type leaf: {pc_type!r}')


def _bisimulation_classes(heads, edges):
    """Refine ordered type edges using smaller-half partition splitting."""
    groups = defaultdict(set)
    for node, (head, children) in enumerate(zip(heads, edges)):
        groups[(head, len(children))].add(node)
    blocks = [groups[key] for key in sorted(groups)]
    classes = [0] * len(heads)
    for number, block in enumerate(blocks):
        for node in block:
            classes[node] = number
    predecessors = [[] for _ in heads]
    for source, children in enumerate(edges):
        for position, target in enumerate(children):
            predecessors[target].append((source, position))
    pending = deque(range(len(blocks)))
    queued = set(pending)
    while pending:
        splitter = pending.popleft()
        queued.remove(splitter)
        incoming = defaultdict(set)
        for target in blocks[splitter]:
            for source, position in predecessors[target]:
                incoming[position].add(source)
        for position in sorted(incoming):
            sources = incoming[position]
            affected = defaultdict(set)
            for source in sources:
                affected[classes[source]].add(source)
            for number in sorted(affected):
                selected = affected[number]
                block = blocks[number]
                if len(selected) == len(block):
                    continue
                if len(selected) > len(block) // 2:
                    selected = block - selected
                block.difference_update(selected)
                other = len(blocks)
                blocks.append(selected)
                for node in selected:
                    classes[node] = other
                queued.add(other)
                pending.append(other)
    return classes


def _type_projection(pc_type, resolver=None, *, _validate=True, on_unresolved=None):
    """Read each reachable type into an ordered finite graph.

    This treats a forward-declared pointer and a completed pointer as the
    same type once the name resolves, even if they unfold a recursive record
    through different numbers of intermediate Python specialization objects.
    """
    from .type_graph import (
        IncompleteRecordError, _owner, _resolve, _shape, named_record, record_name,
        validate_type_graph,
    )
    from .type_resolver import TypeReference, TypeResolver, UnresolvedTypeName

    heads = []
    edges = []
    indices = {}
    direct_indices = {}
    objects = []

    validated = False

    def normalize(current, is_reference):
        nonlocal validated
        override = resolver(current) if is_reference and resolver is not None else None
        if override is not None:
            current = _resolve(override)
        else:
            resolved = _resolve(current)
            if is_reference and resolved is None:
                if on_unresolved is not None:
                    expression = (current.expression
                                  if isinstance(current, TypeReference) else current)
                    namespace = current.namespace if isinstance(current, TypeReference) else None
                    on_unresolved(current, TypeResolver._missing_expression_names(
                        expression, namespace))
                from .type_id import note_unresolved_name_token
                note_unresolved_name_token()
                from .session import CompileSession
                return ('forward', _opaque(CompileSession.current()),
                        TypeResolver.reference_key(current))
            current = resolved
        if current is None:
            return ('void',)
        record = named_record(current)
        if record is not None and any(field is None for field in record._field_types or ()):
            return ('incomplete-record', _opaque(_owner(record)), record_name(record))
        try:
            if _validate and not validated and record is not None:
                validate_type_graph(record)
                validated = True
            return current
        except (IncompleteRecordError, UnresolvedTypeName):
            from .type_id import note_unresolved_name_token
            note_unresolved_name_token()
            return ('incomplete-record', _opaque(_owner(record)), record_name(record))

    def add(original):
        identity = id(original)
        known = direct_indices.get(identity)
        if known is not None:
            return known[1]
        is_reference = TypeResolver.is_reference(original)
        current = normalize(original, is_reference)
        key = current if isinstance(current, tuple) else id(current)
        index = indices.get(key)
        if index is None:
            index = len(heads)
            indices[key] = index
            objects.append(current)
            heads.append(None)
            edges.append(None)
        if not is_reference and not isinstance(current, tuple):
            direct_indices[identity] = (original, index)
        return index

    # A type key can be requested before a declaration is complete (for a
    # specialization name). Value-use validation remains TypeGraph's job.
    root = add(pc_type)
    for index, current in enumerate(objects):
        head, children = (current, ()) if isinstance(current, tuple) else _shape(current)
        heads[index] = _wire(head).decode('utf-8')
        edges[index] = tuple(add(child) for child, _ in children)
    return root, objects, heads, edges


def type_key(pc_type, resolver=None, *, _validate=True):
    """Project a type to a lossless, rooted finite bisimulation quotient."""
    root, _, heads, edges = _type_projection(pc_type, resolver, _validate=_validate)
    # Class refinement computes the greatest structural equivalence of this
    # finite graph. Sharing and cycle-unrolling do not affect the result.
    classes = _bisimulation_classes(heads, edges)
    representatives = {}
    for index, number in enumerate(classes):
        representatives.setdefault(number, index)
    queue = [classes[root]]
    positions = {classes[root]: 0}
    output = []
    for number in queue:
        index = representatives[number]
        children = []
        for child in edges[index]:
            child_class = classes[child]
            if child_class not in positions:
                positions[child_class] = len(queue)
                queue.append(child_class)
            children.append(positions[child_class])
        output.append((heads[index], tuple(children)))
    return ('graph', tuple(output))


class StructuralTypeIndex:
    """Intern finite type graphs without serializing every rooted subgraph.

    Acyclic edges use interned child identities. Cyclic components use a
    canonical quotient shared by all of their members. Dictionary equality
    checks the full component, so a hash collision cannot merge types.
    """

    def __init__(self):
        self.components = {}
        self.next_identity = 0

    def intern(self, pc_type, *, on_unresolved=None):
        _, objects, heads, edges = _type_projection(
            pc_type, _validate=False, on_unresolved=on_unresolved)
        classes = _bisimulation_classes(heads, edges)
        representatives = {}
        for node, number in enumerate(classes):
            representatives.setdefault(number, node)
        numbers = list(representatives)
        positions = {number: pos for pos, number in enumerate(numbers)}
        qheads = [heads[representatives[number]] for number in numbers]
        qedges = [tuple(positions[classes[child]] for child in
                        edges[representatives[number]]) for number in numbers]
        reverse = [[] for _ in numbers]
        for node, children in enumerate(qedges):
            for child in children:
                reverse[child].append(node)
        visited, finished = set(), []
        for root in range(len(numbers)):
            pending = [(root, False)]
            while pending:
                node, finish = pending.pop()
                if finish:
                    finished.append(node)
                elif node not in visited:
                    visited.add(node)
                    pending.append((node, True))
                    pending.extend((child, False) for child in qedges[node]
                                   if child not in visited)
        components, assigned = [], set()
        for root in reversed(finished):
            if root in assigned:
                continue
            component, pending = [], [root]
            assigned.add(root)
            while pending:
                node = pending.pop()
                component.append(node)
                for parent in reverse[node]:
                    if parent not in assigned:
                        assigned.add(parent)
                        pending.append(parent)
            components.append(component)
        identities = {}
        for component in reversed(components):
            order = component
            if len(component) > 1:
                order = _canonical_component_order(component, qheads, qedges, identities)
            ranks = {node: rank for rank, node in enumerate(order)}
            key = tuple((qheads[node], tuple(
                ('internal', ranks[child]) if child in ranks
                else ('external', identities[child])
                for child in qedges[node])) for node in order)
            tokens = self.components.get(key)
            if tokens is None:
                tokens = tuple(range(self.next_identity,
                                     self.next_identity + len(order)))
                self.next_identity += len(order)
                self.components[key] = tokens
            identities.update(zip(order, tokens))
        return {id(value): identities[positions[classes[node]]]
                for node, value in enumerate(objects) if not isinstance(value, tuple)}


def _canonical_component_order(component, heads, edges, identities):
    local = {node: pos for pos, node in enumerate(component)}
    local_heads = [heads[node] for node in component]
    local_edges = []
    external = {}
    for node in component:
        children = []
        for child in edges[node]:
            if child in local:
                children.append(local[child])
            else:
                identity = identities[child]
                if identity not in external:
                    external[identity] = len(local_heads)
                    local_heads.append(_wire(('external', identity)).decode('ascii'))
                children.append(external[identity])
        local_edges.append(tuple(children))
    local_edges.extend(() for _ in external)
    partition = _bisimulation_classes(local_heads, local_edges)
    return sorted(component, key=lambda node: partition[local[node]])


def encode_type_key(key):
    return 't2_' + base64.urlsafe_b64encode(_wire(key)).decode('ascii').rstrip('=')


def same_type(source, target, *, resolver=None):
    """Exact PythoC type equality, including qualifiers and refinements."""
    from .type_graph import _equivalent
    return _equivalent(source, target, resolver=resolver)


def _unqualified(pc_type):
    while getattr(pc_type, 'qualified_type', None) is not None:
        pc_type = pc_type.qualified_type
    return pc_type


def _refinement_chain(pc_type):
    items = []
    while getattr(pc_type, '_is_refined', False):
        base = pc_type._base_type or pc_type._struct_type
        items.append(pc_type)
        pc_type = base
    return items, pc_type


def _flags(pc_type):
    """Read storage qualifiers through qualifier and refinement wrappers."""
    flags, seen = {}, set()
    while pc_type is not None and id(pc_type) not in seen:
        seen.add(id(pc_type))
        from .type_resolver import TypeReference
        if isinstance(pc_type, (str, TypeReference)):
            from .type_graph import _resolve
            pc_type = _resolve(pc_type)
            if pc_type is None:
                break
        refined = getattr(pc_type, '_is_refined', False)
        local = getattr(pc_type, '_qualifier_flags', None)
        if local is None:
            getter = None if refined else getattr(pc_type, 'get_qualifier_flags', None)
            local = getter() if getter is not None else {}
        for name, enabled in local.items():
            flags[name] = flags.get(name, False) or enabled
        qualified = getattr(pc_type, 'qualified_type', None)
        if qualified is not None:
            pc_type = qualified
        elif refined:
            base = pc_type._base_type
            pc_type = pc_type._struct_type if base is None else base
        else:
            break
    return flags


def _dropped_pointee_qualifiers(source, target):
    source_flags = _flags(source)
    present = tuple(flag for flag in ('const', 'volatile') if source_flags.get(flag))
    if not present:
        return present
    target_flags = _flags(target)
    return tuple(flag for flag in present if not target_flags.get(flag))


def integer_value(pc_type, raw):
    """Read an integer constant using its PythoC precision and signedness."""
    from .type_converter import get_base_type
    pc_type = get_base_type(pc_type)
    width = pc_type._bit_width
    bits = raw & ((1 << width) - 1)
    if getattr(pc_type, '_is_bool', False):
        return bool(bits)
    if pc_type._is_signed and bits & (1 << (width - 1)):
        bits -= 1 << width
    return bits


def _constant_value(value):
    """Read a compile-time value without treating LLVM type as type policy."""
    from .valueref import ValueRef
    if not isinstance(value, ValueRef):
        return True, value
    known, raw = value.try_get_constant()
    if (known and isinstance(raw, int)
            and (getattr(value.type_hint, '_is_integer', False)
                 or getattr(value.type_hint, '_is_bool', False))):
        raw = integer_value(value.type_hint, raw)
    return known, raw


def is_null_pointer_constant(value):
    """Recognize a known null in the PythoC pointer value domain."""
    from .type_converter import get_base_type

    if value.is_python_value() or not getattr(get_base_type(value.type_hint), '_is_pointer', False):
        return False
    known, constant = _constant_value(value)
    return known and constant is None


def nullable_pointer_type(null, other_type):
    """Retain a concrete pointer type when the other branch is untyped null."""
    from .builtin_entities.types import void
    from .type_converter import get_base_type

    source = get_base_type(null.type_hint)
    if (getattr(source, 'pointee_type', ...) in (None, void)
            and is_null_pointer_constant(null)
            and getattr(get_base_type(other_type), '_is_pointer', False)
            and can_convert(null.type_hint, other_type, value=null)):
        return other_type
    return None


def _literal_item_relation(item, target, mode, context):
    from .builtin_entities.python_type import PythonType
    from .valueref import ValueRef
    if isinstance(item, ValueRef):
        return can_convert(item.type_hint, target, mode=mode,
                           value=item, context=context)
    return can_convert(PythonType.wrap(item, is_constant=True), target,
                       mode=mode, value=item, context=context)


def _enum_payload(target, tag):
    names = getattr(target, '_variant_names', ()) or ()
    variants = getattr(target, '_variant_types', ()) or ()
    tags = getattr(target, '_tag_values', None) or {}
    for name, payload in zip(names, variants):
        candidate = tags.get(name) if isinstance(tags, dict) else getattr(target, name, None)
        if type(candidate) is type(tag) and candidate == tag:
            return True, payload
    return False, None


def enum_constructor_payload(target, args, context=None):
    """Validate an enum constructor and return (admitted, payload type).

    Enums without payloads support integral casts, including runtime C enum
    discriminants. Tagged unions require a known declared variant.
    """
    from .builtin_entities.types import void
    if len(args) == 1 and all(item in (None, void)
                              for item in target._variant_types):
        return _literal_item_relation(args[0], target._tag_type,
                                      'implicit', context), None
    if not _literal_relation(tuple(args), target, 'implicit', context):
        return False, None
    _, tag = _constant_value(args[0])
    return _enum_payload(target, tag)


def enum_runtime_variants(target, args, context=None):
    """Variants a runtime constructor may select after checking its tag."""
    from .builtin_entities.types import void
    if len(args) not in (1, 2):
        return ()
    if not _literal_item_relation(args[0], target._tag_type, 'implicit', context):
        return ()
    variants = []
    for name, payload in zip(target._variant_names, target._variant_types):
        valid = (len(args) == 1 if payload in (None, void) else
                 len(args) == 2 and _literal_item_relation(
                     args[1], payload, 'implicit', context))
        if valid:
            variants.append((target._tag_values[name], payload))
    return tuple(variants)


def enum_constructor_error(target, args):
    """Describe a rejected tag or payload after constructor validation."""
    from .builtin_entities.types import void
    prefix = f'{target.get_name()} requires a known valid variant tag and its declared payload'
    if len(args) not in (1, 2):
        return f'{prefix}. Expected 1 or 2 arguments, got {len(args)}.'
    if not _literal_item_relation(args[0], target._tag_type, 'implicit', None):
        return (f'{prefix}. The first argument must have tag type '
                f'{format_type(target._tag_type)}, got {format_type(args[0].type_hint)}.')
    known, tag = _constant_value(args[0])
    if not known:
        return f'{prefix}. No variant accepts this argument list for a runtime tag.'
    found, payload = _enum_payload(target, tag)
    if not found:
        return f'{prefix}. Tag {tag!r} does not select a declared variant.'
    if payload in (None, void):
        return f'{prefix}. Tag {tag} takes no payload.'
    if len(args) == 1:
        return f'{prefix}. Tag {tag} requires a payload of type {format_type(payload)}.'
    detail = conversion_message(args[1].type_hint, payload, args[1], mode='implicit')
    return f'{prefix}. Payload for tag {tag}: {detail}'


def array_decay_type(source):
    """Array storage qualifiers qualify the decayed pointee, not the pointer."""
    from .builtin_entities.types import ptr
    from .ir_helpers import propagate_qualifiers
    from .type_converter import get_base_type
    base = get_base_type(source)
    if getattr(base, 'element_type', None) is None:
        raise TypeError('Cannot decay array without element_type')
    element = propagate_qualifiers(source, _array_item_type(base))
    return ptr[element]


def _array_item_type(array_type):
    if len(array_type.dimensions) == 1:
        return array_type.element_type
    from .builtin_entities.array import array
    return array[(array_type.element_type, *array_type.dimensions[1:])]


def _contains_storage(source, flag, _seen=None):
    """Inspect only stored values, excluding pointer and signature edges."""
    from .builtin_entities.python_type import PythonType
    from .literal_protocol import iter_literal_value_refs
    from .type_graph import _shape, _resolve
    seen = set() if _seen is None else _seen
    if id(source) in seen:
        return False
    seen.add(id(source))
    if isinstance(source, PythonType):
        return any(_contains_storage(item.type_hint, flag, seen)
                   for item in iter_literal_value_refs(source._python_object))
    source = _resolve(source)
    if source is None:
        return False
    matched = flag(source) if callable(flag) else getattr(source, flag, False)
    if matched:
        return True
    dimensions = getattr(source, 'dimensions', None)
    if dimensions is not None and math.prod(dimensions) == 0:
        return False
    _, children = _shape(source)
    return any(stored is True and _contains_storage(child, flag, seen)
               for child, stored in children)


def contains_linear_storage(source):
    """Find owned tokens through value-storage edges and literal carriers."""
    return _contains_storage(source, '_is_linear')


def contains_const_storage(source):
    """A whole-value store must not overwrite a const subobject."""
    from .ir_helpers import is_const
    return _contains_storage(source, is_const)


def contains_refinement_storage(source):
    """Find proof-bearing values without treating a raw pointer as a proof."""
    return _contains_storage(source, '_is_refined')


def validate_static_storage(pc_type, node, *, zero_initialized=False):
    """Apply the same storage admission to local and class static members."""
    from .builtin_entities.types import void
    from .logger import logger
    if _unqualified(pc_type) is void:
        logger.error('Static storage requires a value type', node=node, exc_type=TypeError)
    if contains_linear_storage(pc_type):
        logger.error('Static storage of linear values is unsupported',
                     node=node, exc_type=TypeError)
    if zero_initialized and not can_zero_initialize(pc_type):
        logger.error('Cannot implicitly zero-initialize proof-bearing static storage',
                     node=node, exc_type=TypeError)


def can_zero_initialize(target, _active=None):
    """Whether implicit aggregate zero filling can inhabit this value type."""
    from .builtin_entities.types import void
    from .builtin_entities.python_type import PythonType
    from .schema_protocol import get_schema_field_types, is_schema_type
    from .type_graph import _resolve
    target = _resolve(_unqualified(target))
    if target is None:
        return False
    if getattr(target, '_is_refined', False) or getattr(target, '_is_linear', False):
        return False
    if isinstance(target, PythonType):
        return target._is_constant
    if target is void or getattr(target, '_is_pointer', False):
        return True
    if any(getattr(target, flag, False)
           for flag in ('_is_integer', '_is_bool', '_is_float')):
        return True
    active = set() if _active is None else _active
    if id(target) in active:
        return False
    active = active | {id(target)}
    dimensions = getattr(target, 'dimensions', None)
    if dimensions is not None:
        return math.prod(dimensions) == 0 or can_zero_initialize(target.element_type, active)
    if getattr(target, '_is_enum', False):
        payloads = target._variant_types
        if all(item in (None, void) for item in payloads):
            return True
        found, payload = _enum_payload(target, 0)
        return found and (payload in (None, void) or can_zero_initialize(payload, active))
    if is_schema_type(target):
        return all(can_zero_initialize(field, active)
                   for field in get_schema_field_types(target))
    element = getattr(target, 'element_type', None)
    if getattr(target, 'lanes', None) is not None:
        return can_zero_initialize(element, active)
    return False


def _callable_type(literal):
    if getattr(literal, '_is_compiled', False):
        if getattr(literal, '_is_parametric', False):
            return None
        info = getattr(literal, '_func_info', None)
        return getattr(info, 'callable_pc_type', None)
    if getattr(literal, '_is_extern', False):
        from .callable_lowering import extern_callable_type
        actual, _, _ = extern_callable_type(literal)
        return actual
    return None


def _literal_relation(literal, target, mode, context):
    from .builtin_entities.types import void
    from .literal_protocol import (get_mapping_entries, get_sequence_elements,
                                   is_mapping_carrier, is_sequence_carrier,
                                   select_union_member)
    from .schema_protocol import (get_schema_field_names,
                                  get_schema_field_types, is_schema_type)
    if getattr(literal, '_is_compiled', False) or getattr(literal, '_is_extern', False):
        actual = _callable_type(literal)
        return actual is not None and can_convert(actual, target,
                                                  mode=mode, context=context)
    if is_sequence_carrier(literal):
        items = get_sequence_elements(literal)
        if getattr(target, '_is_enum', False):
            if len(items) not in (1, 2):
                return False
            known, tag = _constant_value(items[0])
            if not known:
                return False
            found, payload = _enum_payload(target, tag)
            if not found:
                return False
            return (len(items) == 1 if payload in (None, void) else
                    len(items) == 2 and _literal_item_relation(items[1], payload, mode, context))
        if getattr(target, 'dimensions', None) is not None:
            # The materializer permits omitted trailing elements and fills zeroes.
            element = _array_item_type(target)
            return (len(items) <= target.dimensions[0]
                    and (len(items) == target.dimensions[0] or can_zero_initialize(element))
                    and all(_literal_item_relation(item, element, mode, context)
                            for item in items))
        if is_schema_type(target):
            fields = get_schema_field_types(target)
            return (len(items) <= len(fields)
                    and all(can_zero_initialize(field) for field in fields[len(items):])
                    and all(_literal_item_relation(item, field, mode, context)
                            for item, field in zip(items, fields)))
        return False
    if is_mapping_carrier(literal):
        if getattr(target, '_is_union', False):
            selected = select_union_member(literal, target)
            return selected is not None and _literal_item_relation(
                selected[2], selected[1], mode, context)
        if not is_schema_type(target) or getattr(target, '_is_enum', False):
            return False
        names = get_schema_field_names(target)
        if any(name is None for name in names):
            return False
        entries = get_mapping_entries(literal)
        mapped = {}
        for key, item in entries:
            known, key = _constant_value(key)
            if not known or type(key) is not str or key in mapped:
                return False
            mapped[key] = item
        return (set(mapped) == set(names)
                and all(_literal_item_relation(mapped[name], field, mode, context)
                        for name, field in zip(names, get_schema_field_types(target))))
    if type(literal) is str:
        if getattr(target, '_is_pointer', False):
            from .builtin_entities.types import i8, u8, void
            pointee = _unqualified(getattr(target, 'pointee_type', None))
            return (mode == 'explicit' or pointee is None or pointee is void
                    or any(same_type(pointee, byte, resolver=context)
                           for byte in (i8, u8)))
        if getattr(target, 'dimensions', None) is not None:
            element = target.element_type
            size = getattr(element, '_size_bytes', None)
            return (size == 1 and getattr(element, '_is_integer', False)
                    and not getattr(element, '_is_bool', False)
                    and len(literal.encode('utf-8')) <= math.prod(target.dimensions))
        return False
    if type(literal) in (int, float, bool):
        if (getattr(target, '_is_integer', False) or getattr(target, '_is_float', False)
                or getattr(target, '_is_bool', False)):
            return True
        if getattr(target, '_is_pointer', False):
            return mode == 'explicit' and type(literal) is int
        if getattr(target, '_is_enum', False):
            found, payload = _enum_payload(target, literal)
            return found and payload in (None, void)
        if getattr(target, 'dimensions', None) is not None or is_schema_type(target):
            return literal == 0 and can_zero_initialize(target)
    return False


def format_type(pc_type, parents=()):
    """Render type components used by conversion diagnostics."""
    from .builtin_entities.python_type import PythonType
    from .builtin_entities.types import void
    from .literal_protocol import is_mapping_carrier, is_sequence_carrier
    from .schema_protocol import is_schema_type, iter_schema_fields

    if pc_type is None:
        return 'void'
    if isinstance(pc_type, PythonType):
        literal = pc_type.get_python_object()
        if is_sequence_carrier(literal):
            return 'sequence literal'
        if is_mapping_carrier(literal):
            return 'mapping literal'
        if pc_type._is_constant and callable(literal):
            name = getattr(literal, '__name__', type(literal).__name__)
            return f'pyconst[{name}]'
        return pc_type.get_instance_name()
    if not isinstance(pc_type, type):
        return str(pc_type)
    if pc_type in parents:
        return pc_type.__name__
    chain = (*parents, pc_type)
    qualified = getattr(pc_type, 'qualified_type', None)
    if qualified is not None:
        return f'{pc_type.get_qualifier_name()}[{format_type(qualified, chain)}]'
    parameters = getattr(pc_type, 'param_types', None)
    if parameters is not None:
        items = [format_type(parameter, chain) for parameter in parameters]
        for flag, label in (('has_varargs', '*args'), ('has_llvm_varargs', '...'),
                            ('has_kwargs', '**kwargs')):
            if getattr(pc_type, flag, False):
                items.append(label)
        items.append(format_type(pc_type.return_type, chain))
        return f'func[{", ".join(items)}] ({pc_type.calling_convention})'
    if getattr(pc_type, '_is_function_type', False):
        return 'func'
    if getattr(pc_type, '_is_pointer', False):
        pointee = getattr(pc_type, 'pointee_type', None) or void
        return f'ptr[{format_type(pointee, chain)}]'
    dimensions = getattr(pc_type, 'dimensions', None)
    if dimensions is not None:
        items = [format_type(pc_type.element_type, chain), *map(str, dimensions)]
        return f'array[{", ".join(items)}]'
    if getattr(pc_type, 'lanes', None) is not None:
        return f'simd[{format_type(pc_type.element_type, chain)}, {pc_type.lanes}]'
    if getattr(pc_type, '_is_refined', False):
        base = pc_type._base_type or pc_type._struct_type
        if base is not None:
            items = [format_type(base, chain)]
            items.extend(getattr(p, '__name__', repr(p)) for p in pc_type._predicates or ())
            items.extend(map(repr, pc_type._tags or ()))
            return f'refined[{", ".join(items)}]'
    if getattr(pc_type, '_variant_names', None) is not None:
        variants = []
        for name, payload in zip(pc_type._variant_names, pc_type._variant_types):
            tag = pc_type._tag_values[name]
            field = f'{name}: {format_type(payload, chain)}' if payload not in (None, void) else name
            variants.append(f'{field} = {tag}')
        return f'{pc_type.get_name()} ({format_type(pc_type._tag_type, chain)}) {{{", ".join(variants)}}}'
    if is_schema_type(pc_type):
        from .type_graph import named_record
        members = []
        for _, name, member in iter_schema_fields(pc_type):
            rendered = format_type(member, chain)
            members.append(f'{name}: {rendered}' if name else rendered)
        if named_record(pc_type) is not None:
            return f'{pc_type.get_name()} {{{", ".join(members)}}}'
        family = 'union' if getattr(pc_type, '_is_union', False) else 'struct'
        return f'{family}[{", ".join(members)}]'
    get_name = getattr(pc_type, 'get_name', None)
    return get_name() if get_name is not None else pc_type.__name__


def _record_conversion_detail(source, target, mode, context, seen):
    """Find the first aggregate field rejected by the normal relation."""
    from .schema_protocol import get_schema_field_types, is_schema_type, iter_schema_fields
    source, target = _unqualified(source), _unqualified(target)
    if not all(is_schema_type(t) and not getattr(t, '_is_enum', False)
               and not getattr(t, '_is_union', False) for t in (source, target)):
        return None
    pair = id(source), id(target)
    if pair in seen:
        return None
    seen.add(pair)
    source_fields = get_schema_field_types(source)
    target_fields = iter_schema_fields(target)
    if len(source_fields) != len(target_fields):
        return f'field count differs ({len(source_fields)} vs {len(target_fields)})'
    for first, (index, name, second) in zip(source_fields, target_fields):
        compatible = (same_type(first, second, resolver=context) if mode == 'identity'
                      else can_convert(first, second, mode=mode, context=context))
        if not compatible:
            field = repr(name) if name is not None else str(index + 1)
            detail = _record_conversion_detail(first, second, mode, context, seen)
            mismatch = detail or f'{format_type(first)} vs {format_type(second)}'
            return f'field {field}: {mismatch}'
    return None


def conversion_message(source, target, value, *, mode, context=None):
    """Explain a rejected conversion using the same directional relation."""
    message = conversion_failure(source, target, value, mode, context)
    if message is not None:
        return message
    operation = 'implicitly convert' if mode == 'implicit' else 'convert'
    message = f"Cannot {operation} '{format_type(source)}' to '{format_type(target)}'"
    dropped = _implicit_pointer_qualifier_loss(source, target) if mode == 'implicit' else ()
    if dropped:
        message += f": would drop pointee {' and '.join(dropped)}"
    detail = _record_conversion_detail(source, target, mode, context, set())
    if detail is not None:
        message += f': incompatible fields, {detail}'
    elif (getattr(_unqualified(source), 'pointee_type', None) is not None
          and getattr(_unqualified(target), 'pointee_type', None) is not None):
        detail = _record_conversion_detail(_unqualified(source).pointee_type,
                                           _unqualified(target).pointee_type,
                                           'identity', context, set())
        if detail is not None:
            message += f': pointed-to record {detail}'
    if mode == 'implicit' and can_convert(source, target, mode='explicit',
                                          value=value, context=context):
        message += '. An explicit cast is allowed.'
        if 'const' in dropped:
            message += ' Writes still require writable storage.'
        if 'volatile' in dropped:
            message += ' Keep volatile accesses for storage that requires them.'
    return message


def _implicit_pointer_qualifier_loss(source, target):
    target = _unqualified(target)
    if not getattr(target, '_is_pointer', False) or getattr(target, '_is_function_type', False):
        return ()
    source_base = _refinement_chain(_unqualified(source))[1]
    if getattr(source_base, 'dimensions', None) is not None:
        source_base = array_decay_type(source)
    if not getattr(source_base, '_is_pointer', False) or getattr(source_base, '_is_function_type', False):
        return ()
    return _dropped_pointee_qualifiers(
        getattr(source_base, 'pointee_type', None), getattr(target, 'pointee_type', None))


def conversion_failure(source, target, value, mode, context):
    """Explain common value-dependent rejections at the semantic boundary."""
    from .builtin_entities.python_type import PythonType
    from .literal_protocol import (get_mapping_entries, get_sequence_elements,
                                   is_mapping_carrier, is_sequence_carrier,
                                   select_union_member)
    from .schema_protocol import is_schema_type, iter_schema_fields
    from .builtin_entities.types import void
    if target is void and contains_linear_storage(source):
        return 'Cannot discard a linear value with a cast; use consume()'
    if isinstance(target, PythonType):
        if contains_linear_storage(source):
            return 'Cannot discard a linear value into a compile-time constant'
        return f'Cannot assign {format_type(source)} to {format_type(target)}'
    if not isinstance(source, PythonType):
        return None
    known, literal = _constant_value(value) if value is not None else (True, source._python_object)
    if not known:
        return None
    if getattr(literal, '_is_compiled', False) or getattr(literal, '_is_extern', False):
        if getattr(literal, '_is_parametric', False):
            return ('A parametric function cannot be used as a first-class value '
                    'without supplying its compile-time arguments first')
        actual = _callable_type(literal)
        if actual is not None:
            name = getattr(literal, '__name__', getattr(literal, 'func_name', repr(literal)))
            return (f"Function '{name}' has signature {format_type(actual)}, "
                    f"but is being used as {format_type(target)}")
    target = _unqualified(target)
    if is_mapping_carrier(literal) and getattr(target, '_is_union', False):
        selected = select_union_member(literal, target)
        if selected is None:
            return 'A union initializer must select exactly one named member'
        name, field, item = selected
        return _literal_member_error(item, field, f'member {name!r}', mode, context)
    if is_mapping_carrier(literal) and is_schema_type(target):
        fields = iter_schema_fields(target)
        names = [name for _, name, _ in fields]
        if any(name is None for name in names):
            return f'Cannot initialize {format_type(target)} by name: some fields are unnamed'
        mapped = {}
        for key, item in get_mapping_entries(literal):
            key_known, key = _constant_value(key)
            if not key_known or type(key) is not str:
                return 'Cannot initialize fields by name: keys must be constant strings'
            if key not in names:
                return f"pc_dict key '{key}' is not a field of {target}"
            if key in mapped:
                return f"Cannot initialize field {key!r} twice"
            mapped[key] = item
        missing = [name for name in names if name not in mapped]
        if missing:
            return f"pc_dict is missing fields for {target}: {', '.join(missing)}"
        for _, name, field in fields:
            if not _literal_item_relation(mapped[name], field, mode, context):
                return _literal_member_error(mapped[name], field, f'field {name!r}', mode, context)
    if is_sequence_carrier(literal) and not getattr(target, '_is_enum', False):
        items = get_sequence_elements(literal)
        if getattr(target, 'dimensions', None) is not None:
            element = _array_item_type(target)
            fields = [(index, None, element)
                      for index in range(min(target.dimensions[0], len(items) + 1))]
            label = 'element'
        elif is_schema_type(target):
            fields = iter_schema_fields(target)
            label = 'field'
        else:
            return None
        if len(items) > len(fields):
            return (f'Cannot initialize {format_type(target)} with {len(items)} values. '
                    f'Expected at most {len(fields)}.')
        for index, name, field in fields:
            subject = f'{label} {name!r}' if name is not None else f'{label} {index + 1}'
            if index < len(items):
                if not _literal_item_relation(items[index], field, mode, context):
                    return _literal_member_error(items[index], field, subject, mode, context)
            elif not can_zero_initialize(field):
                return f'Cannot omit initializer for {subject} of type {format_type(field)}'
    if isinstance(literal, type):
        return f'Cannot promote type {format_type(literal)} to a value of type {format_type(target)}'
    if (not isinstance(literal, (int, float, bool, str, type(None)))
            and not is_mapping_carrier(literal)
            and not is_sequence_carrier(literal)):
        return f"Cannot promote Python value of type {type(literal).__name__} to PC type"
    return None


def _literal_member_error(item, field, subject, mode, context):
    from .builtin_entities.python_type import PythonType
    from .valueref import ValueRef
    source = item.type_hint if isinstance(item, ValueRef) else PythonType.wrap(item, is_constant=True)
    return f'{conversion_message(source, field, item, mode=mode, context=context)} ({subject})'


def can_convert(source, target, *, mode='implicit', value=None, context=None, _seen=None):
    """Directional PythoC conversion policy; never asks LLVM for equality.

    ``mode`` is ``implicit`` or ``explicit``. The trusted refinement constructors
    are separate operations and intentionally do not call this as an override.
    """
    if mode not in ('implicit', 'explicit'):
        raise ValueError(f'Unknown conversion mode: {mode}')
    if same_type(source, target, resolver=context):
        return True
    _seen = set() if _seen is None else _seen
    pair = (id(source), id(target), mode)
    if pair in _seen:
        return True
    _seen.add(pair)

    original_source = source
    source_flags, target_flags = _flags(source), _flags(target)
    if source_flags or target_flags:
        # At the value level qualifiers describe storage. Pointer pointee
        # qualifiers are checked below before recursive conversion.
        source = _unqualified(source)
        target = _unqualified(target)
        if same_type(source, target, resolver=context):
            return True

    if any(getattr(item, '_is_function_type', False)
           and getattr(item, 'param_types', None) is None for item in (source, target)):
        return False
    source_refinements, source_base = _refinement_chain(source)
    target_refinements, target_base = _refinement_chain(target)
    if (source_refinements and not target_refinements
            and getattr(source_base, 'dimensions', None) is not None
            and getattr(target, '_is_pointer', False)):
        return can_convert(array_decay_type(original_source), target,
                           mode=mode, context=context)
    if source_refinements or target_refinements:
        if not target_refinements:
            return can_convert(source_base, target_base, mode=mode, value=value,
                               context=context, _seen=_seen)
        if not same_type(source_base, target_base, resolver=context):
            return False
        source_predicates = [pred for item in source_refinements
                             for pred in item._predicates or ()]
        source_tags = {atom(tag) for item in source_refinements
                       for tag in item._tags or ()}
        return (all(any(same_predicate(pred, source_pred, context)
                        for source_pred in source_predicates)
                    for item in target_refinements for pred in item._predicates or ())
                and all(atom(tag) in source_tags for item in target_refinements
                        for tag in item._tags or ()))

    from .builtin_entities.types import void
    if target is void:
        return not contains_linear_storage(source)

    from .builtin_entities.python_type import PythonType
    if isinstance(target, PythonType):
        if contains_linear_storage(source):
            return False
        if not getattr(target, '_is_constant', False):
            return False
        known, actual = _constant_value(value) if value is not None else (False, None)
        if not known and isinstance(source, PythonType) and source._is_constant:
            known, actual = True, source._constant_value
        return known and atom(actual) == atom(target._constant_value)
    if isinstance(source, PythonType):
        known, literal = _constant_value(value) if value is not None else (True, source._python_object)
        return known and _literal_relation(literal, target, mode, context)

    source_func = getattr(source, 'param_types', None)
    target_func = getattr(target, 'param_types', None)
    if source_func is not None or target_func is not None:
        if source_func is None or target_func is None:
            if (source_func is None and target_func is not None
                    and getattr(source, '_is_pointer', False) and value is not None):
                known, actual = _constant_value(value)
                if known and actual is None:
                    return True
            return (mode == 'explicit' and getattr(source, '_is_pointer', False)
                    and (getattr(target, '_is_pointer', False)
                         or getattr(target, '_is_integer', False)
                         or getattr(target, '_is_bool', False))) or (
                    mode == 'explicit' and getattr(target, '_is_pointer', False)
                    and (getattr(source, '_is_integer', False)
                         or getattr(source, '_is_bool', False))) or (
                    source_func is not None and target_func is None
                    and getattr(target, '_is_pointer', False)
                    and _unqualified(getattr(target, 'pointee_type', None)) in (None, void))
        if len(source_func) != len(target_func):
            return False
        flags = ('has_varargs', 'has_llvm_varargs', 'has_kwargs')
        return (source.calling_convention == target.calling_convention
                and all(bool(getattr(source, flag, False)) == bool(getattr(target, flag, False)) for flag in flags)
                and all(same_type(a, b, resolver=context) for a, b in zip(source_func, target_func))
                and same_type(source.return_type, target.return_type, resolver=context))

    source_ptr = getattr(source, '_is_pointer', False)
    target_ptr = getattr(target, '_is_pointer', False)
    source_dimensions = getattr(source, 'dimensions', None)
    if source_dimensions is not None and target_ptr:
        return can_convert(array_decay_type(original_source), target,
                           mode=mode, context=context)
    if source_ptr and target_ptr:
        if value is not None:
            known, actual = _constant_value(value)
            if known and actual is None:
                return True
        if mode == 'explicit':
            return True
        source_pointee = getattr(source, 'pointee_type', None)
        target_pointee = getattr(target, 'pointee_type', None)
        if _dropped_pointee_qualifiers(source_pointee, target_pointee):
            return False
        first = _unqualified(source_pointee)
        second = _unqualified(target_pointee)
        return (first in (None, void) or second in (None, void)
                or same_type(first, second, resolver=context))
    if source_ptr or target_ptr:
        if mode != 'explicit':
            if source_ptr and target_ptr and value is not None:
                known, actual = _constant_value(value)
                return known and actual is None
            return False
        return ((source_ptr and (getattr(target, '_is_integer', False)
                                or getattr(target, '_is_bool', False)))
                or (target_ptr and (getattr(source, '_is_integer', False)
                                    or getattr(source, '_is_bool', False))))

    source_dimensions = getattr(source, 'dimensions', None)
    target_dimensions = getattr(target, 'dimensions', None)
    if source_dimensions is not None or target_dimensions is not None:
        return (source_dimensions == target_dimensions
                and same_type(getattr(source, 'element_type', None),
                              getattr(target, 'element_type', None),
                              resolver=context))
    source_lanes = getattr(source, 'lanes', None)
    target_lanes = getattr(target, 'lanes', None)
    if source_lanes is not None or target_lanes is not None:
        return (source_lanes == target_lanes
                and same_type(getattr(source, 'element_type', None),
                              getattr(target, 'element_type', None),
                              resolver=context))

    source_fields = getattr(source, '_field_types', None)
    target_fields = getattr(target, '_field_types', None)
    if source_fields is not None and getattr(target, '_is_enum', False):
        if len(source_fields) not in (1, 2) or not isinstance(source_fields[0], PythonType):
            return False
        tag_type = source_fields[0]
        if not tag_type._is_constant:
            return False
        found, payload = _enum_payload(target, tag_type._constant_value)
        if not found:
            return False
        from .builtin_entities.types import void
        if payload in (None, void):
            return len(source_fields) == 1
        return (len(source_fields) == 2
                and can_convert(source_fields[1], payload, mode=mode,
                                context=context, _seen=_seen))
    if (getattr(source, '_is_enum', False) or getattr(target, '_is_enum', False)
            or getattr(source, '_is_union', False) or getattr(target, '_is_union', False)):
        return False
    if source_fields is not None or target_fields is not None:
        if source_fields is None or target_fields is None:
            return False
        return (len(source_fields) == len(target_fields)
                and all(can_convert(a, b, mode=mode, context=context, _seen=_seen)
                        for a, b in zip(source_fields, target_fields)))

    if ((getattr(source, '_is_integer', False) or getattr(source, '_is_bool', False))
            and (getattr(target, '_is_integer', False) or getattr(target, '_is_bool', False))):
        return True
    if (getattr(source, '_is_integer', False) or getattr(source, '_is_float', False)
            or getattr(source, '_is_bool', False)) and (
            getattr(target, '_is_integer', False) or getattr(target, '_is_float', False)
            or getattr(target, '_is_bool', False)):
        return True
    return False


def common_numeric_type(left, right):
    """Select a symmetric numeric result using PythoC metadata only.

    PythoC preserves explicitly sized integers instead of promoting every small
    integer to i32. Equal-width mixed integers use the unsigned type; a wider
    signed integer can represent every value of a narrower unsigned integer.
    Arithmetic forgets storage qualifiers and refinement proofs.
    """
    from .type_converter import get_base_type
    from .builtin_entities import types
    left, right = get_base_type(left), get_base_type(right)

    def numeric(item):
        return any(getattr(item, flag, False)
                   for flag in ('_is_integer', '_is_bool', '_is_float'))

    if not numeric(left) or not numeric(right):
        return None
    floats = [item for item in (left, right) if getattr(item, '_is_float', False)]
    if floats:
        if len(floats) == 1:
            return floats[0]
        if same_type(*floats):
            return floats[0]
        if floats[0]._size_bytes == floats[1]._size_bytes:
            return types.f32
        return max(floats, key=lambda item: item._size_bytes)
    width = max(left._bit_width, right._bit_width)
    widest = [item for item in (left, right) if item._bit_width == width]
    if all(getattr(item, '_is_bool', False) for item in widest):
        return types.bool
    signed = all(item._is_signed for item in widest)
    return getattr(types, ('i' if signed else 'u') + str(width))


def common_value_type(left, right):
    """Select a common semantic type before lowering either branch."""
    from .builtin_entities.types import ptr, void
    from .builtin_entities.qualifiers import const, volatile
    if left is None or right is None:
        return None
    left, right = _unqualified(left), _unqualified(right)
    if same_type(left, right):
        return left
    if getattr(left, '_is_refined', False) and getattr(right, '_is_refined', False):
        # Selection preserves an existing weaker proof on both paths. Numeric
        # promotion below intentionally erases proofs for arithmetic instead.
        left_to_right = can_convert(left, right)
        right_to_left = can_convert(right, left)
        if left_to_right and right_to_left:
            from .type_id import get_type_id
            return min((left, right), key=get_type_id)
        if left_to_right:
            return right
        if right_to_left:
            return left
    numeric = common_numeric_type(left, right)
    if numeric is not None:
        return numeric
    if (getattr(left, '_is_pointer', False)
            and getattr(right, '_is_pointer', False)
            and getattr(left, 'param_types', None) is None
            and getattr(right, 'param_types', None) is None):
        first = getattr(left, 'pointee_type', None)
        second = getattr(right, 'pointee_type', None)
        base_first, base_second = _unqualified(first), _unqualified(second)
        if base_first in (None, void):
            pointee = base_second
        elif base_second in (None, void) or same_type(base_first, base_second):
            pointee = base_first
        else:
            return None
        pointee = void if pointee is None else pointee
        if _flags(first).get('const') or _flags(second).get('const'):
            pointee = const[pointee]
        if _flags(first).get('volatile') or _flags(second).get('volatile'):
            pointee = volatile[pointee]
        candidate = ptr[pointee]
        return candidate if can_convert(left, candidate) and can_convert(right, candidate) else None
    left_to_right = can_convert(left, right)
    right_to_left = can_convert(right, left)
    if left_to_right and right_to_left:
        # Convertible records/signatures can retain different labels. Use a
        # stable tie break so swapping branches cannot change the result type.
        from .type_id import get_type_id
        return min((left, right), key=get_type_id)
    if left_to_right:
        return right
    if right_to_left:
        return left
    return None


def can_subtract_pointers(left, right, context=None):
    """Pointer differences require a shared element stride and semantics."""
    from .builtin_entities.types import void
    from .type_converter import get_base_type
    left, right = get_base_type(left), get_base_type(right)
    if not all(getattr(item, '_is_pointer', False)
               and getattr(item, 'param_types', None) is None for item in (left, right)):
        return False
    first = _unqualified(getattr(left, 'pointee_type', None)) or void
    second = _unqualified(getattr(right, 'pointee_type', None)) or void
    return same_type(first, second, resolver=context)


def can_compare_pointers(left, right, *, left_value=None, right_value=None,
                         context=None):
    """Pointer comparison checks semantic pointees before backend alignment."""
    if not (getattr(left, '_is_pointer', False)
            and getattr(right, '_is_pointer', False)):
        return False
    if (can_convert(left, right, value=left_value, context=context)
            or can_convert(right, left, value=right_value, context=context)):
        return True
    # Comparing addresses does not read or write their pointees. Storage
    # qualifiers may differ on both sides without discarding either qualifier.
    first = getattr(left, 'pointee_type', None)
    second = getattr(right, 'pointee_type', None)
    return (first is not None and second is not None
            and same_type(_unqualified(first), _unqualified(second), resolver=context))


def is_integer_operand(source, *, value=None):
    """Integer category for indexing, where numeric casts are not implicit."""
    from .builtin_entities.python_type import PythonType
    from .type_converter import get_base_type
    if isinstance(source, PythonType):
        known, literal = _constant_value(value) if value is not None else (True, source._python_object)
        return known and type(literal) in (int, bool)
    source = get_base_type(source)
    return bool(getattr(source, '_is_integer', False) or getattr(source, '_is_bool', False))


def default_argument_promotion(source):
    """C varargs promotions, selected solely from PythoC value types."""
    from .type_converter import get_base_type
    from .builtin_entities.types import i32, f32, f64
    source = get_base_type(source)
    if (getattr(source, '_is_integer', False) or getattr(source, '_is_bool', False)):
        return i32 if source._bit_width < 32 else None
    return f64 if same_type(source, f32) else None


def atomic_value_is_supported(pointee, operation):
    """Atomic value admission, including storage and proof restrictions."""
    from .type_converter import get_base_type
    if operation not in ('load', 'store', 'add', 'sub', 'and', 'or', 'xchg', 'cas'):
        raise ValueError(f'Unknown atomic operation: {operation}')
    if operation != 'load' and _flags(pointee).get('const'):
        return False
    arithmetic = operation in ('add', 'sub', 'and', 'or')
    if arithmetic and _refinement_chain(_unqualified(pointee))[0]:
        return False
    base = get_base_type(pointee)
    if getattr(base, '_is_integer', False):
        return base._bit_width in (8, 16, 32, 64)
    if getattr(base, '_is_bool', False):
        return not arithmetic
    return not arithmetic and bool(getattr(base, '_is_pointer', False))


def specialization_key(value, _active=None):
    """One lossless input projection for specialization and artifact caches."""
    from .utils.naming import NormalizedSuffix
    from .builtin_entities.base import _is_type_like
    if isinstance(value, NormalizedSuffix):
        return ('normalized-suffix', value.identity)
    if _is_type_like(value):
        from .type_id import get_type_id
        return ('pc-type', get_type_id(value))
    if getattr(value, '_is_compiled', False):
        binding = getattr(value, '_binding', None)
        if binding is not None and binding.registration_identity is not None:
            return ('compiled-callable', binding.source_file, binding.original_name,
                    specialization_key(binding.compile_suffix),
                    specialization_key(binding.effect_suffix),
                    binding.registration_identity)
    kind = type(value)
    if kind not in (tuple, list, dict):
        return atom(value)
    active = set() if _active is None else _active
    if id(value) in active:
        return _opaque(value)
    active.add(id(value))
    try:
        if kind is dict:
            pairs = ((specialization_key(key, active), specialization_key(item, active))
                     for key, item in value.items())
            return ('dict', tuple(sorted(pairs, key=lambda pair: _wire(pair[0]))))
        return (kind.__name__, tuple(specialization_key(item, active) for item in value))
    finally:
        active.remove(id(value))


def validate_member_names(names, context, node):
    """Validate labels without conflating unnamed fields with named members."""
    from .logger import logger
    present = [name for name in names or () if name is not None]
    if any(not isinstance(name, str) or not name for name in present):
        logger.error(f'{context}: member names must be nonempty strings',
                     node=node, exc_type=TypeError)
    if len(set(present)) != len(present):
        logger.error(f'{context}: duplicate member names', node=node, exc_type=TypeError)


def validate_type_argument(value, context, node, *, allow_void=True, allow_incomplete=False):
    """Admit Python objects that actually implement the PythoC type protocol."""
    from .builtin_entities.base import _is_type_like
    from .builtin_entities.types import void
    from .logger import logger
    if value is None and allow_incomplete:
        return
    if isinstance(value, str) and value:
        return
    if not _is_type_like(value):
        logger.error(f'{context}: expected a PythoC type or quoted forward reference',
                     node=node, exc_type=TypeError)
    if not allow_void and _unqualified(value) is void:
        logger.error(f'{context}: void is not a value element or parameter type',
                     node=node, exc_type=TypeError)
