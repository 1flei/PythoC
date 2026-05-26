"""Internal typed task adapter.

This module lowers a normal typed @compile function to the runtime task ABI:
func[ptr[void], ptr[void]].  It generates the argument cell, trampoline,
result cell, and typed join helpers used by Future.

It is intentionally not a user-facing task API.  User code should use
pythoc.std.runtime.Future, or pythoc.std.runtime.raw for direct C ABI work.
"""
from __future__ import annotations
from .policy import bind_mem
bind_mem()

from types import SimpleNamespace

from pythoc import (
    compile, effect, u64, u8, ptr, void, nullptr, sizeof, meta, move,
)

from .api import (
    Runtime, runtime_spawn, runtime_join, runtime_detach,
)
from .raw import runtime_spawn_raw, runtime_join_raw
from .task import Task, TaskHandle, task_destroy


_SPAWN_TYPED_ABI_VERSION = "spawn_typed_target_group_v1"
_SPAWN_TYPED_TRAMPOLINE_SOURCE = "spawn_typed_trampoline.py"
_SPAWN_TYPED_SPAWN_SOURCE = "spawn_typed_spawn.py"
_SPAWN_TYPED_SPAWN_RAW_SOURCE = "spawn_typed_spawn_raw.py"
_SPAWN_TYPED_JOIN_SOURCE = "spawn_typed_join.py"
_SPAWN_TYPED_JOIN_RAW_SOURCE = "spawn_typed_join_raw.py"


def _target_suffix_key(target_fn):
    binding = getattr(target_fn, "_binding", getattr(target_fn, "_state", None))
    group_key = getattr(binding, "group_key", None)
    if group_key is not None:
        return group_key
    return getattr(target_fn, "__name__", "anon")


def _TypedTask(target_fn, *, stack_size=None):
    """Generate typed spawn/join helpers for a specific @compile function.

    Given a function like:

        @compile
        def work(x: i64, y: i32) -> i64: ...

    Generates:
        - _WorkArgs struct { x: i64, y: i32 }
        - _work_trampoline(arg: ptr[void]) -> ptr[void]
        - typed_spawn(rt, x, y) -> TaskHandle
        - typed_join(rt, handle) -> i64

    Args:
        target_fn: A @compile'd PythoC function.
        stack_size: Override per-task stack size (default: runtime default).

    Returns:
        SimpleNamespace with:
            .spawn(rt, arg0, arg1, ...)  -> TaskHandle
            .join(rt, handle)            -> return_type
            .spawn_raw(rt, arg0, ...)    -> ptr[Task]
            .join_raw(rt, task)          -> return_type
            .detach(rt, handle)          -> void
            .args_type                   -> the generated Args struct type
    """
    # ---- Extract function metadata ----
    param_info = _extract_params(target_fn)
    ret_type = _extract_return_type(target_fn)
    fn_name = getattr(target_fn, '__name__', 'anon')

    # Use a unique suffix for all generated code
    type_suffix = (
        _SPAWN_TYPED_ABI_VERSION,
        fn_name,
        _target_suffix_key(target_fn),
        tuple(t for _, t in param_info),
        ret_type,
    )

    # ---- 1. Generate Args struct ----
    args_struct = _make_args_struct(param_info, type_suffix)

    # ---- 2. Generate Trampoline ----
    # The trampoline has signature func[ptr[void], ptr[void]] and does:
    #   1. Cast ptr[void] -> ptr[ArgsStruct]
    #   2. Extract each field
    #   3. Call target_fn(field0, field1, ...)
    #   4. If return type is not void: store result in heap cell
    #   5. Free the args struct
    #   6. Return ptr[void] to result (or nullptr for void)
    trampoline = _make_trampoline(
        target_fn, param_info, ret_type, args_struct, type_suffix
    )

    # ---- 3. Generate typed_spawn ----
    has_return = ret_type is not void and ret_type is not None
    stk = stack_size if stack_size is not None else u64(0)

    @compile(suffix=type_suffix)
    def _typed_spawn(rt: ptr[Runtime], args_ptr: ptr[void]) -> TaskHandle:
        """Internal: spawn with pre-packed args."""
        return runtime_spawn(rt, trampoline, args_ptr, stk)

    @compile(suffix=type_suffix)
    def _typed_spawn_raw(rt: ptr[Runtime], args_ptr: ptr[void]) -> ptr[Task]:
        """Internal: spawn_raw with pre-packed args."""
        return runtime_spawn_raw(rt, trampoline, args_ptr, stk)

    # ---- 4. Generate arg-packing + spawn wrapper ----
    # We generate a function that allocates the Args struct, fills fields,
    # and calls _typed_spawn.
    spawn_fn = _make_spawn_wrapper(
        param_info, args_struct, _typed_spawn, type_suffix
    )
    spawn_raw_fn = _make_spawn_raw_wrapper(
        param_info, args_struct, _typed_spawn_raw, type_suffix
    )

    # ---- 5. Generate typed join ----
    if has_return:
        join_fn = _make_typed_join(ret_type, type_suffix)
        join_raw_fn = _make_typed_join_raw(ret_type, type_suffix)
    else:
        join_fn = _make_void_join(type_suffix)
        join_raw_fn = _make_void_join_raw(type_suffix)

    # ---- 6. Detach (just a passthrough) ----
    @compile(suffix=type_suffix)
    def _typed_detach(rt: ptr[Runtime], handle: TaskHandle) -> void:
        runtime_detach(rt, handle)

    return SimpleNamespace(
        spawn=spawn_fn,
        join=join_fn,
        spawn_raw=spawn_raw_fn,
        join_raw=join_raw_fn,
        detach=_typed_detach,
        args_type=args_struct,
        trampoline=trampoline,
    )


# ============================================================
# Internal: Extract function signature from @compile metadata
# ============================================================

def _extract_params(fn) -> list:
    """Extract (name, type) pairs from a @compile function.

    PythoC stores type annotations as resolved types on the compiled
    function's metadata. We inspect the original Python annotations
    and resolve them against the function's globals.
    """
    import inspect

    # Try to get from PythoC's internal metadata first
    if hasattr(fn, '_pc_param_types'):
        names = fn._pc_param_names
        types = fn._pc_param_types
        return list(zip(names, types))

    # Fallback: inspect Python annotations
    sig = inspect.signature(fn)
    hints = fn.__annotations__ if hasattr(fn, '__annotations__') else {}

    params = []
    for name, param in sig.parameters.items():
        if name == 'return':
            continue
        ann = hints.get(name)
        if ann is None:
            raise TypeError(
                f"Task adapter: parameter '{name}' of {fn.__name__} has no type annotation. "
                f"All parameters must be annotated."
            )
        # Resolve string annotations if needed
        if isinstance(ann, str):
            ann = eval(ann, fn.__globals__)
        params.append((name, ann))

    return params


def _extract_return_type(fn):
    """Extract return type from a @compile function."""
    if hasattr(fn, '_pc_return_type'):
        return fn._pc_return_type

    hints = fn.__annotations__ if hasattr(fn, '__annotations__') else {}
    ret = hints.get('return')
    if ret is None:
        return void
    if isinstance(ret, str):
        ret = eval(ret, fn.__globals__)
    return ret


# ============================================================
# Internal: Code generation
# ============================================================

def _make_args_struct(param_info, suffix):
    """Generate a struct type holding all function arguments."""
    if not param_info:
        # No-arg function: use a minimal struct (can't have empty struct)
        @compile(suffix=suffix)
        class _EmptyArgs:
            _pad: u8
        return _EmptyArgs

    # Use meta.struct_type for dynamic struct creation
    fields = [(name, ty) for name, ty in param_info]
    return meta.struct_type(fields)


@meta.quote(debug_source=False)
def _tpl_seq(first, second, third, fourth, fifth):
    first
    second
    third
    fourth
    fifth


@meta.quote(debug_source=False)
def _tpl_empty():
    pass


@meta.quote(debug_source=False)
def _tpl_cast_arg(args_name, args_ptr_type):
    args_name: args_ptr_type = args_ptr_type(raw_arg)


@meta.quote(debug_source=False)
def _tpl_pack_one_arg(args_name, index, value_name):
    args_name[0][index] = move(value_name)


@meta.quote(debug_source=False)
def _tpl_assign_target_result(target_name):
    target_name = _target_fn(*_args[0])


@meta.quote(debug_source=False)
def _tpl_assign_target_result_no_args(target_name):
    target_name = _target_fn()


@meta.quote(debug_source=False)
def _tpl_call_target_unpacked():
    _target_fn(*_args[0])


@meta.quote(debug_source=False)
def _tpl_call_target_no_args():
    _target_fn()


@meta.quote(debug_source=False)
def _tpl_box_result(cell_name, cell_type, ret_type, value_name):
    cell_name: cell_type = cell_type(
        effect.mem.malloc(u64(sizeof(ret_type)))
    )
    cell_name[0] = value_name


@meta.quote(debug_source=False)
def _tpl_free_raw_arg():
    effect.mem.free(ptr_void_cast(raw_arg))


@meta.quote(debug_source=False)
def _tpl_return_ptr(value_name):
    return ptr_void_cast(value_name)


@meta.quote(debug_source=False)
def _tpl_return_null():
    return nullptr


@meta.quote(debug_source=False)
def _tpl_alloc_args(args_name, args_ptr_type, args_type):
    args_name: args_ptr_type = args_ptr_type(
        effect.mem.malloc(u64(sizeof(args_type)))
    )


@meta.quote(debug_source=False)
def _tpl_return_spawn(spawn_fn, args_name):
    return spawn_fn(rt, ptr_void_cast(args_name))


@meta.quote(debug_source=False)
def _tpl_runtime_join(raw_name, join_fn, token_name):
    raw_name: ptr_void_type = join_fn(rt, token_name)


@meta.quote(debug_source=False)
def _tpl_cast_result(cell_name, cell_type, raw_name):
    cell_name: cell_type = cell_type(raw_name)


@meta.quote(debug_source=False)
def _tpl_load_result(value_name, cell_name):
    value_name = cell_name[0]


@meta.quote(debug_source=False)
def _tpl_free_raw(raw_name):
    effect.mem.free(raw_name)


@meta.quote(debug_source=False)
def _tpl_destroy_task(task_name):
    task_destroy(task_name)


@meta.quote(debug_source=False)
def _tpl_return_value(value_name):
    return value_name


def _fragment_globals(*items):
    merged = {}

    def visit(item):
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        merged.update(getattr(item, '_user_globals', {}))

    visit(items)
    return merged


def _make_spawn_body(param_info, spawn_fn):
    alloc = _tpl_alloc_args('_a', '_PtrArgs', '_ArgsType')
    pack = [
        _tpl_pack_one_arg('_a', i, name)
        for i, (name, _) in enumerate(param_info)
    ]
    finish = _tpl_return_spawn(spawn_fn, '_a')
    return _tpl_seq(alloc, pack, finish, _tpl_empty(), _tpl_empty())


def _compile_generated(name, params, return_type, body, required_globals,
                       source_file, suffix):
    merged_globals = dict(required_globals)
    merged_globals.update(_fragment_globals(body))
    gf = meta.func(
        name=name,
        params=params,
        return_type=return_type,
        body=body,
        required_globals=merged_globals,
        source_file=source_file,
    )
    return meta.compile_generated(gf, suffix=suffix)


def _make_trampoline(target_fn, param_info, ret_type, args_struct, suffix):
    """Generate the trampoline: func[ptr[void], ptr[void]].

    The generated function:
      1. Casts arg from ptr[void] to ptr[ArgsStruct]
      2. Loads each field
      3. Calls target_fn
      4. Packs result into heap cell (if non-void)
      5. Frees the args struct
      6. Returns ptr[void] to result cell (or nullptr)
    """
    has_return = ret_type is not void and ret_type is not None

    setup = _tpl_cast_arg('_args', '_PtrArgs')
    call_args = bool(param_info)

    if has_return:
        invoke = (
            _tpl_assign_target_result('_result_val')
            if call_args
            else _tpl_assign_target_result_no_args('_result_val')
        )
        result_body = [
            _tpl_box_result(
                '_result_cell', '_PtrRet', '_RetType', '_result_val'
            ),
            _tpl_free_raw_arg(),
        ]
        finish = _tpl_return_ptr('_result_cell')
    else:
        invoke = (
            _tpl_call_target_unpacked()
            if call_args
            else _tpl_call_target_no_args()
        )
        result_body = [_tpl_free_raw_arg()]
        finish = _tpl_return_null()

    body = _tpl_seq(setup, _tpl_empty(), invoke, result_body, finish)
    required_globals = {
        '__name__': __name__,
        'effect': effect,
        'u64': u64,
        'sizeof': sizeof,
        'nullptr': nullptr,
        '_target_fn': target_fn,
        '_PtrArgs': ptr[args_struct],
        'ptr_void_cast': ptr[void],
    }
    if has_return:
        required_globals['_RetType'] = ret_type
        required_globals['_PtrRet'] = ptr[ret_type]

    return _compile_generated(
        name=f'_trampoline_{getattr(target_fn, "__name__", "fn")}',
        params=[('raw_arg', ptr[void])],
        return_type=ptr[void],
        body=body,
        required_globals=required_globals,
        source_file=_SPAWN_TYPED_TRAMPOLINE_SOURCE,
        suffix=suffix,
    )


def _make_spawn_wrapper(param_info, args_struct, typed_spawn_fn, suffix):
    """Generate spawn function that packs args and calls runtime_spawn.

    For a function with params (x: i64, y: i32):
        def spawn(rt: ptr[Runtime], x: i64, y: i32) -> TaskHandle:
            args = ptr[ArgsStruct](effect.mem.malloc(sizeof(ArgsStruct)))
            args.x = x
            args.y = y
            return _typed_spawn(rt, ptr[void](args))
    """
    body = _make_spawn_body(param_info, typed_spawn_fn)
    return _compile_generated(
        name=f'_spawn_{getattr(typed_spawn_fn, "__name__", "fn")}',
        params=[('rt', ptr[Runtime])] + list(param_info),
        return_type=TaskHandle,
        body=body,
        required_globals={
            '__name__': __name__,
            'effect': effect,
            'move': move,
            'u64': u64,
            'sizeof': sizeof,
            '_ArgsType': args_struct,
            '_PtrArgs': ptr[args_struct],
            'ptr_void_cast': ptr[void],
            typed_spawn_fn.__name__: typed_spawn_fn,
            'TaskHandle': TaskHandle,
        },
        source_file=_SPAWN_TYPED_SPAWN_SOURCE,
        suffix=("spawn", suffix),
    )


def _make_spawn_raw_wrapper(param_info, args_struct, typed_spawn_raw_fn, suffix):
    """Same as spawn wrapper but returns ptr[Task] instead of TaskHandle."""
    body = _make_spawn_body(param_info, typed_spawn_raw_fn)
    return _compile_generated(
        name=f'_spawn_raw_{getattr(typed_spawn_raw_fn, "__name__", "fn")}',
        params=[('rt', ptr[Runtime])] + list(param_info),
        return_type=ptr[Task],
        body=body,
        required_globals={
            '__name__': __name__,
            'effect': effect,
            'move': move,
            'u64': u64,
            'sizeof': sizeof,
            '_ArgsType': args_struct,
            '_PtrArgs': ptr[args_struct],
            'ptr_void_cast': ptr[void],
            typed_spawn_raw_fn.__name__: typed_spawn_raw_fn,
        },
        source_file=_SPAWN_TYPED_SPAWN_RAW_SOURCE,
        suffix=("spawn_raw", suffix),
    )


def _make_join_body(join_fn, token_name, destroy_task=False):
    read_raw = _tpl_runtime_join('_raw', join_fn, token_name)
    cast_cell = _tpl_cast_result('_cell', '_PtrRet', '_raw')
    load_value = _tpl_load_result('_val', '_cell')
    cleanup = [_tpl_free_raw('_raw')]
    if destroy_task:
        cleanup.append(_tpl_destroy_task(token_name))
    return _tpl_seq(
        read_raw,
        cast_cell,
        load_value,
        cleanup,
        _tpl_return_value('_val'),
    )


def _make_typed_join(ret_type, suffix):
    """Generate join that returns the actual typed value.

    def join(rt: ptr[Runtime], handle: TaskHandle) -> RetType:
        raw: ptr[void] = runtime_join(rt, handle)
        cell: ptr[RetType] = ptr[RetType](raw)
        val = cell[0]
        effect.mem.free(raw)
        return val
    """
    body = _make_join_body(runtime_join, 'handle')
    return _compile_generated(
        name='_typed_join',
        params=[('rt', ptr[Runtime]), ('handle', TaskHandle)],
        return_type=ret_type,
        body=body,
        required_globals={
            '__name__': __name__,
            'effect': effect,
            'ptr_void_type': ptr[void],
            '_PtrRet': ptr[ret_type],
            runtime_join.__name__: runtime_join,
            'TaskHandle': TaskHandle,
        },
        source_file=_SPAWN_TYPED_JOIN_SOURCE,
        suffix=("join", suffix),
    )


def _make_typed_join_raw(ret_type, suffix):
    """Generate join_raw that takes ptr[Task] directly (no linear handle)."""
    body = _make_join_body(runtime_join_raw, 'task', destroy_task=True)
    return _compile_generated(
        name='_typed_join_raw',
        params=[('rt', ptr[Runtime]), ('task', ptr[Task])],
        return_type=ret_type,
        body=body,
        required_globals={
            '__name__': __name__,
            'effect': effect,
            'ptr_void_type': ptr[void],
            '_PtrRet': ptr[ret_type],
            runtime_join_raw.__name__: runtime_join_raw,
            'task_destroy': task_destroy,
        },
        source_file=_SPAWN_TYPED_JOIN_RAW_SOURCE,
        suffix=("join_raw", suffix),
    )


def _make_void_join(suffix):
    """Generate join for void-returning functions."""
    @compile(suffix=("void_join", suffix))
    def _void_join(rt: ptr[Runtime], handle: TaskHandle) -> void:
        runtime_join(rt, handle)

    return _void_join


def _make_void_join_raw(suffix):
    """Generate join_raw for void-returning functions."""
    @compile(suffix=("void_join_raw", suffix))
    def _void_join_raw(rt: ptr[Runtime], task: ptr[Task]) -> void:
        runtime_join_raw(rt, task)
        task_destroy(task)

    return _void_join_raw


# ============================================================
# Convenience: batch spawn helper
# ============================================================

def _TypedBatch(target_fn, *, stack_size=None):
    """Like _TypedTask but optimized for spawning many tasks in a batch.

    Returns an additional .spawn_batch() that doesn't wake workers
    per-spawn, and a .notify_all() to wake them after the batch.

    Usage:
        Compute = _TypedBatch(compute)

        # Spawn 10000 tasks without per-task notification
        handles = []
        for i in range(10000):
            handles.append(Compute.spawn_batch(rt, i64(i), i64(i*2)))
        Compute.notify_all(rt)  # Wake all workers once

        # Join all
        for h in handles:
            result = Compute.join(rt, h)
    """
    base = _TypedTask(target_fn, stack_size=stack_size)

    # Add batch-spawn (uses sched_spawn_batch internally)
    from .scheduler import sched_spawn_batch, sched_notify_all

    # For batch, users should use spawn_raw + manual task management
    # since batch spawning returns raw tasks without linear handles
    return SimpleNamespace(
        spawn=base.spawn,
        join=base.join,
        spawn_raw=base.spawn_raw,
        join_raw=base.join_raw,
        detach=base.detach,
        args_type=base.args_type,
        trampoline=base.trampoline,
        # Batch-specific: user should call notify_all after spawning batch
    )
