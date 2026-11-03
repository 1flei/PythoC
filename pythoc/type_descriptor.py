"""Pure projections and relations over authoritative Python type objects.

The Python classes/instances remain the only type definitions. ``type_key``
does not own state: it reads their fields and emits a lossless wire projection.
A digest of that wire form is only an index, never proof of equality.
"""

import base64
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
        if (pc_type.__module__.startswith('pythoc.builtin_entities')
                or pc_type.__module__ == 'pythoc.std.float_types'):
            import importlib
            module = importlib.import_module(pc_type.__module__)
            if vars(module).get(pc_type.__name__) is pc_type:
                return ('builtin', pc_type.__module__, pc_type.__name__)
        return ('declaration', _opaque(pc_type))
    raise TypeError(f'Unsupported PythoC type leaf: {pc_type!r}')


def type_key(pc_type, resolver=None):
    """Project a type graph to a canonical finite bisimulation quotient.

    This treats a forward-declared pointer and a completed pointer as the
    same type once the name resolves, even if they unfold a recursive record
    through different numbers of intermediate Python specialization objects.
    """
    from .type_graph import (
        _owner, _resolve, _shape, canonical_record_type, named_record,
        record_name,
    )
    heads = []
    edges = []
    indices = {}

    def normalize(current, allow_incomplete):
        if isinstance(current, str):
            resolved = resolver(current) if resolver is not None else None
            if resolved is None:
                resolved = _resolve(current)
            if resolved is None:
                from .type_id import note_unresolved_name_token
                note_unresolved_name_token()
                from .session import CompileSession
                return ('forward', _opaque(CompileSession.current()), current)
            current = resolved
        current = _resolve(current)
        if current is None:
            return ('void',)
        try:
            return canonical_record_type(current)
        except TypeError as error:
            if not allow_incomplete or 'Incomplete PythoC' not in str(error):
                raise
            view = named_record(current)
            if view is None:
                raise
            from .type_id import note_unresolved_name_token
            note_unresolved_name_token()
            return ('incomplete-record', _opaque(_owner(view)),
                    record_name(view))

    def add(current, allow_incomplete=False):
        current = normalize(current, allow_incomplete)
        key = current if isinstance(current, tuple) else id(current)
        if key in indices:
            return indices[key]
        index = len(heads)
        indices[key] = index
        heads.append(None)
        edges.append(None)
        if isinstance(current, tuple):
            head, children = current, ()
        else:
            head, children = _shape(current)
        heads[index] = _wire(head).decode('utf-8')
        edges[index] = tuple(add(child, allow_incomplete or not stored)
                             for child, stored in children)
        return index

    # A type key can be requested before a declaration is complete (for a
    # specialization name). Value-use validation remains TypeGraph's job.
    root = add(pc_type, allow_incomplete=True)
    # Class refinement computes the greatest structural equivalence of this
    # finite graph. Sharing and cycle-unrolling do not affect the result.
    def classify(signatures):
        ordered = sorted(set(signatures))
        numbers = {signature: number for number, signature in enumerate(ordered)}
        return tuple(numbers[signature] for signature in signatures)

    classes = classify(tuple((head, len(children)) for head, children in zip(heads, edges)))
    while True:
        signatures = tuple((classes[index], head,
                            tuple(classes[child] for child in children))
                           for index, (head, children) in enumerate(zip(heads, edges)))
        updated = classify(signatures)
        # Including the previous class makes each pass a strict refinement.
        # Once the class count stops growing, only numeric labels could move.
        if len(set(updated)) == len(set(classes)):
            break
        classes = updated
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
        if isinstance(pc_type, str):
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
    if isinstance(value, ValueRef):
        if value.is_python_value():
            return True, value.get_python_value()
        ir_value = value.ir_value
        if hasattr(ir_value, 'constant'):
            raw = ir_value.constant
            if (isinstance(raw, int)
                    and (getattr(value.type_hint, '_is_integer', False)
                         or getattr(value.type_hint, '_is_bool', False))):
                raw = integer_value(value.type_hint, raw)
            return True, raw
        return False, None
    return True, value


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


def array_decay_type(source):
    """Array storage qualifiers qualify the decayed pointee, not the pointer."""
    from .builtin_entities.array import array
    from .builtin_entities.types import ptr
    from .ir_helpers import propagate_qualifiers
    from .type_converter import get_base_type
    base = get_base_type(source)
    if getattr(base, 'element_type', None) is None:
        raise TypeError('Cannot decay array without element_type')
    dimensions = base.dimensions
    element = (array[(base.element_type,) + tuple(dimensions[1:])]
               if len(dimensions) > 1 else base.element_type)
    element = propagate_qualifiers(source, element)
    return ptr[element]


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


def validate_static_storage(pc_type, *, zero_initialized=False):
    """Apply the same storage admission to local and class static members."""
    from .builtin_entities.types import void
    if _unqualified(pc_type) is void:
        raise TypeError('Static storage requires a value type')
    if contains_linear_storage(pc_type):
        raise TypeError('Static storage of linear values is unsupported')
    if zero_initialized and not can_zero_initialize(pc_type):
        raise TypeError('Cannot implicitly zero-initialize proof-bearing static storage')


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
                                   is_mapping_carrier, is_sequence_carrier)
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
            element = target.element_type
            if len(target.dimensions) > 1:
                from .builtin_entities.array import array
                element = array[(element,) + tuple(target.dimensions[1:])]
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
            from .builtin_entities.types import i8, void
            pointee = getattr(target, 'pointee_type', None)
            return (mode == 'explicit' or pointee is None or pointee is void
                    or same_type(pointee, i8, resolver=context))
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


def conversion_failure(source, target, value):
    """Explain common value-dependent rejections at the semantic boundary."""
    from .builtin_entities.python_type import PythonType
    from .literal_protocol import (get_mapping_entries, is_mapping_carrier,
                                   is_sequence_carrier)
    from .schema_protocol import (get_schema_field_names,
                                  is_schema_type)
    from .builtin_entities.types import void
    if target is void and contains_linear_storage(source):
        return 'Cannot discard a linear value with a cast; use consume()'
    if isinstance(target, PythonType) and contains_linear_storage(source):
        return 'Cannot discard a linear value into a compile-time constant'
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
            return (f"Function '{name}' has signature {actual.get_name()}, "
                    f"but is being used as {target.get_name()}")
    if is_mapping_carrier(literal) and is_schema_type(target):
        names = get_schema_field_names(target)
        keys = []
        for key, _item in get_mapping_entries(literal):
            key_known, key = _constant_value(key)
            if key_known:
                keys.append(key)
        for key in keys:
            if key not in names:
                return f"pc_dict key '{key}' is not a field of {target}"
        missing = [name for name in names if name not in keys]
        if missing:
            return f"pc_dict is missing fields for {target}: {', '.join(missing)}"
    if (not isinstance(literal, (int, float, bool, str, type(None)))
            and not is_mapping_carrier(literal)
            and not is_sequence_carrier(literal)):
        return f"Cannot promote Python value of type {type(literal).__name__} to PC type"
    return None


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
        if any(_flags(source_pointee).get(flag)
               and not _flags(target_pointee).get(flag)
               for flag in ('const', 'volatile')):
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
    if operation not in ('load', 'store', 'add', 'and', 'or', 'xchg', 'cas'):
        raise ValueError(f'Unknown atomic operation: {operation}')
    if operation != 'load' and _flags(pointee).get('const'):
        return False
    arithmetic = operation in ('add', 'and', 'or')
    if arithmetic and _refinement_chain(_unqualified(pointee))[0]:
        return False
    base = get_base_type(pointee)
    if getattr(base, '_is_integer', False):
        return base._bit_width in (8, 16, 32, 64)
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


def validate_member_names(names, context):
    """Validate labels without conflating unnamed fields with named members."""
    present = [name for name in names or () if name is not None]
    if any(not isinstance(name, str) or not name for name in present):
        raise TypeError(f'{context}: member names must be nonempty strings')
    if len(set(present)) != len(present):
        raise TypeError(f'{context}: duplicate member names')


def validate_type_argument(value, context, *, allow_void=True, allow_incomplete=False):
    """Admit Python objects that actually implement the PythoC type protocol."""
    from .builtin_entities.base import _is_type_like
    from .builtin_entities.types import void
    if value is None and allow_incomplete:
        return
    if isinstance(value, str) and value:
        return
    if not _is_type_like(value):
        raise TypeError(f'{context}: expected a PythoC type or quoted forward reference')
    if not allow_void and _unqualified(value) is void:
        raise TypeError(f'{context}: void is not a value element or parameter type')
