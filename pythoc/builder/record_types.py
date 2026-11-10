"""Materialize validated records using target storage representations."""

import threading
import weakref


_materialization_lock = threading.RLock()


def materialize_record(definition, context):
    """Declare a record before lowering its possibly recursive fields."""
    from ..session import CompileSession
    with CompileSession.current().type_graph._lock, _materialization_lock:
        state = getattr(context, '_pythoc_records', None)
        if state is None:
            state = context._pythoc_records = (weakref.WeakKeyDictionary(), {})
        records, representations = state
        if definition in records:
            return records[definition]
        name, suffix = definition.name, 0
        while name in context.identified_types:
            suffix += 1
            name = f'{definition.name}.{suffix}'
        record = context.get_identified_type(name)
        records[definition] = record
        try:
            fields = tuple(field.get_llvm_type(context)
                           for field in definition.field_types)
            fields = tuple(field for field in fields if field is not None)
            existing = representations.get((definition.name, fields))
            if existing is not None:
                records[definition] = existing
                return existing
            record.set_body(*fields)
            representations[(definition.name, fields)] = record
        except Exception:
            del records[definition]
            raise
        return record
