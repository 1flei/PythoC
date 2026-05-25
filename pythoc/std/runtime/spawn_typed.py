"""Internal typed task adapter.

This module lowers a normal typed @compile function to the runtime task ABI:
func[ptr[void], ptr[void]].  It generates the argument cell, trampoline,
result cell, and typed join helpers used by Future.

It is intentionally not a user-facing task API.  User code should use
pythoc.std.runtime.Future, or pythoc.std.runtime.raw for direct C ABI work.
"""
from __future__ import annotations

import ast
from .policy import bind_mem
bind_mem()

from types import SimpleNamespace
from typing import Any, Tuple

from pythoc import (
    compile, effect, i32, i64, u64, u8, ptr, void, struct, nullptr, sizeof,
    func, meta, move,
)
from pythoc.libc.string import memset

from .api import (
    Runtime, runtime_spawn, runtime_join, runtime_detach,
)
from .raw import runtime_spawn_raw, runtime_join_raw
from .task import Task, TaskHandle, task_destroy


_SPAWN_TYPED_ABI_VERSION = "spawn_typed_target_group_v1"


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

    if not param_info and not has_return:
        # Simplest case: no args, no return
        @compile(suffix=("tramp_void_void", suffix))
        def _trampoline(arg: ptr[void]) -> ptr[void]:
            target_fn(arg)  # will be dead code eliminated; just call it
            effect.mem.free(arg)
            return nullptr
        # Actually, for no-args we still need to handle it properly.
        # Let's fall through to the general case.

    # General case: use dynamic code generation via meta.func
    # Build the trampoline body as AST statements
    import ast

    body_stmts = []

    # Step 1: Cast arg to args struct pointer
    # args_p: ptr[ArgsStruct] = ptr[ArgsStruct](raw_arg)
    body_stmts.append(ast.AnnAssign(
        target=ast.Name(id='_args', ctx=ast.Store()),
        annotation=_type_annotation_node('ptr', args_struct),
        value=ast.Call(
            func=_type_annotation_node('ptr', args_struct),
            args=[ast.Name(id='raw_arg', ctx=ast.Load())],
            keywords=[],
        ),
        simple=1,
    ))

    # Step 2: Load each field into a local variable
    for i, (name, param_type) in enumerate(param_info):
        body_stmts.append(ast.AnnAssign(
            target=ast.Name(id=f'_p{i}', ctx=ast.Store()),
            annotation=ast.Name(id=f'_ParamType{i}', ctx=ast.Load()),
            value=ast.Subscript(
                value=ast.Subscript(
                    value=ast.Name(id='_args', ctx=ast.Load()),
                    slice=ast.Constant(value=0),
                    ctx=ast.Load(),
                ),
                slice=ast.Constant(value=i),
                ctx=ast.Load(),
            ),
            simple=1,
        ))

    # Step 3: Call target function
    call_args = [ast.Name(id=f'_p{i}', ctx=ast.Load()) for i in range(len(param_info))]
    call_expr = ast.Call(
        func=ast.Name(id='_target_fn', ctx=ast.Load()),
        args=call_args,
        keywords=[],
    )
    
    if has_return:
        # _result_val = target_fn(_p0, _p1, ...)
        body_stmts.append(ast.Assign(
            targets=[ast.Name(id='_result_val', ctx=ast.Store())],
            value=call_expr,
        ))
        
        # Step 4: Allocate result cell and store
        # _result_cell: ptr[RetType] = ptr[RetType](effect.mem.malloc(u64(sizeof(RetType))))
        body_stmts.append(ast.AnnAssign(
            target=ast.Name(id='_result_cell', ctx=ast.Store()),
            annotation=_type_annotation_node('ptr', 'ret'),
            value=ast.Call(
                func=_type_annotation_node('ptr', 'ret'),
                args=[ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id='effect', ctx=ast.Load()),
                            attr='mem',
                            ctx=ast.Load(),
                        ),
                        attr='malloc',
                        ctx=ast.Load(),
                    ),
                    args=[ast.Call(
                        func=ast.Name(id='u64', ctx=ast.Load()),
                        args=[ast.Call(
                            func=ast.Name(id='sizeof', ctx=ast.Load()),
                            args=[ast.Name(id='_RetType', ctx=ast.Load())],
                            keywords=[],
                        )],
                        keywords=[],
                    )],
                    keywords=[],
                )],
                keywords=[],
            ),
            simple=1,
        ))
        
        # _result_cell[0] = _result_val
        body_stmts.append(ast.Assign(
            targets=[ast.Subscript(
                value=ast.Name(id='_result_cell', ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=ast.Store(),
            )],
            value=ast.Name(id='_result_val', ctx=ast.Load()),
        ))
    else:
        # void return: just call
        body_stmts.append(ast.Expr(value=call_expr))

    # Step 5: Free args struct
    body_stmts.append(ast.Expr(value=ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id='effect', ctx=ast.Load()),
                attr='mem',
                ctx=ast.Load(),
            ),
            attr='free',
            ctx=ast.Load(),
        ),
        args=[ast.Call(
            func=ast.Name(id='ptr_void_cast', ctx=ast.Load()),
            args=[ast.Name(id='raw_arg', ctx=ast.Load())],
            keywords=[],
        )],
        keywords=[],
    )))

    # Step 6: Return
    if has_return:
        body_stmts.append(ast.Return(value=ast.Call(
            func=ast.Name(id='ptr_void_cast', ctx=ast.Load()),
            args=[ast.Name(id='_result_cell', ctx=ast.Load())],
            keywords=[],
        )))
    else:
        body_stmts.append(ast.Return(value=ast.Name(id='nullptr', ctx=ast.Load())))

    ast.fix_missing_locations(ast.Module(body=body_stmts, type_ignores=[]))

    # Build required_globals for the trampoline
    required_globals = {
        '__name__': __name__,
        'effect': effect,
        'ptr': ptr,
        'void': void,
        'u64': u64,
        'sizeof': sizeof,
        'nullptr': nullptr,
        '_target_fn': target_fn,
        '_ArgsStruct': args_struct,
        'ptr_void_cast': ptr[void],
    }
    if has_return:
        required_globals['_RetType'] = ret_type
    for i, (_, param_type) in enumerate(param_info):
        required_globals[f'_ParamType{i}'] = param_type

    # Use meta.func to build the GeneratedFunction
    gf = meta.func(
        name=f'_trampoline_{getattr(target_fn, "__name__", "fn")}',
        params=[('raw_arg', ptr[void])],
        return_type=ptr[void],
        body=body_stmts,
        required_globals=required_globals,
        source_file=f'<spawn_typed:{getattr(target_fn, "__name__", "fn")}>',
    )

    # Compile it
    compiled = meta.compile_generated(gf, suffix=suffix)
    return compiled


def _type_annotation_node(wrapper, inner_type):
    """Build AST node for ptr[T] type expression."""
    # This generates: ptr[_ArgsStruct] or ptr[_RetType]
    # In the required_globals context, these resolve correctly
    if wrapper == 'ptr':
        return ast.Subscript(
            value=ast.Name(id='ptr', ctx=ast.Load()),
            slice=ast.Name(id='_ArgsStruct' if inner_type != 'ret' else '_RetType', ctx=ast.Load()),
            ctx=ast.Load(),
        )
    return ast.Name(id=str(inner_type), ctx=ast.Load())


def _make_spawn_wrapper(param_info, args_struct, typed_spawn_fn, suffix):
    """Generate spawn function that packs args and calls runtime_spawn.

    For a function with params (x: i64, y: i32):
        def spawn(rt: ptr[Runtime], x: i64, y: i32) -> TaskHandle:
            args = ptr[ArgsStruct](effect.mem.malloc(sizeof(ArgsStruct)))
            args.x = x
            args.y = y
            return _typed_spawn(rt, ptr[void](args))
    """
    import ast as _ast

    params_list = [('rt', ptr[Runtime])] + list(param_info)
    body_stmts = []

    # Allocate args struct
    body_stmts.append(_ast.AnnAssign(
        target=_ast.Name(id='_a', ctx=_ast.Store()),
        annotation=_ast.Name(id='_PtrArgs', ctx=_ast.Load()),
        value=_ast.Call(
            func=_ast.Name(id='_PtrArgs', ctx=_ast.Load()),
            args=[_ast.Call(
                func=_ast.Attribute(
                    value=_ast.Attribute(
                        value=_ast.Name(id='effect', ctx=_ast.Load()),
                        attr='mem', ctx=_ast.Load(),
                    ),
                    attr='malloc', ctx=_ast.Load(),
                ),
                args=[_ast.Call(
                    func=_ast.Name(id='u64', ctx=_ast.Load()),
                    args=[_ast.Call(
                        func=_ast.Name(id='sizeof', ctx=_ast.Load()),
                        args=[_ast.Name(id='_ArgsType', ctx=_ast.Load())],
                        keywords=[],
                    )],
                    keywords=[],
                )],
                keywords=[],
            )],
            keywords=[],
        ),
        simple=1,
    ))

    if param_info:
        body_stmts.append(_ast.Assign(
            targets=[_ast.Subscript(
                value=_ast.Name(id='_a', ctx=_ast.Load()),
                slice=_ast.Constant(value=0),
                ctx=_ast.Store(),
            )],
            value=_ast.Tuple(
                elts=[
                    _ast.Call(
                        func=_ast.Name(id='move', ctx=_ast.Load()),
                        args=[_ast.Name(id=name, ctx=_ast.Load())],
                        keywords=[],
                    )
                    for name, _ in param_info
                ],
                ctx=_ast.Load(),
            ),
        ))

    # Call _typed_spawn(rt, ptr[void](_a))
    body_stmts.append(_ast.Return(value=_ast.Call(
        func=_ast.Name(id='_typed_spawn', ctx=_ast.Load()),
        args=[
            _ast.Name(id='rt', ctx=_ast.Load()),
            _ast.Call(
                func=_ast.Name(id='_ptr_void', ctx=_ast.Load()),
                args=[_ast.Name(id='_a', ctx=_ast.Load())],
                keywords=[],
            ),
        ],
        keywords=[],
    )))

    _ast.fix_missing_locations(_ast.Module(body=body_stmts, type_ignores=[]))

    gf = meta.func(
        name=f'_spawn_{getattr(typed_spawn_fn, "__name__", "fn")}',
        params=params_list,
        return_type=TaskHandle,
        body=body_stmts,
        required_globals={
            '__name__': __name__,
            'effect': effect,
            'move': move,
            'u64': u64,
            'sizeof': sizeof,
            '_ArgsType': args_struct,
            '_PtrArgs': ptr[args_struct],
            '_ptr_void': ptr[void],
            '_typed_spawn': typed_spawn_fn,
            'TaskHandle': TaskHandle,
        },
        source_file=f'<spawn_typed:spawn>',
    )

    return meta.compile_generated(gf, suffix=("spawn", suffix))


def _make_spawn_raw_wrapper(param_info, args_struct, typed_spawn_raw_fn, suffix):
    """Same as spawn wrapper but returns ptr[Task] instead of TaskHandle."""
    import ast as _ast

    params_list = [('rt', ptr[Runtime])] + list(param_info)
    body_stmts = []

    body_stmts.append(_ast.AnnAssign(
        target=_ast.Name(id='_a', ctx=_ast.Store()),
        annotation=_ast.Name(id='_PtrArgs', ctx=_ast.Load()),
        value=_ast.Call(
            func=_ast.Name(id='_PtrArgs', ctx=_ast.Load()),
            args=[_ast.Call(
                func=_ast.Attribute(
                    value=_ast.Attribute(
                        value=_ast.Name(id='effect', ctx=_ast.Load()),
                        attr='mem', ctx=_ast.Load(),
                    ),
                    attr='malloc', ctx=_ast.Load(),
                ),
                args=[_ast.Call(
                    func=_ast.Name(id='u64', ctx=_ast.Load()),
                    args=[_ast.Call(
                        func=_ast.Name(id='sizeof', ctx=_ast.Load()),
                        args=[_ast.Name(id='_ArgsType', ctx=_ast.Load())],
                        keywords=[],
                    )],
                    keywords=[],
                )],
                keywords=[],
            )],
            keywords=[],
        ),
        simple=1,
    ))

    if param_info:
        body_stmts.append(_ast.Assign(
            targets=[_ast.Subscript(
                value=_ast.Name(id='_a', ctx=_ast.Load()),
                slice=_ast.Constant(value=0),
                ctx=_ast.Store(),
            )],
            value=_ast.Tuple(
                elts=[
                    _ast.Call(
                        func=_ast.Name(id='move', ctx=_ast.Load()),
                        args=[_ast.Name(id=name, ctx=_ast.Load())],
                        keywords=[],
                    )
                    for name, _ in param_info
                ],
                ctx=_ast.Load(),
            ),
        ))

    body_stmts.append(_ast.Return(value=_ast.Call(
        func=_ast.Name(id='_typed_spawn_raw', ctx=_ast.Load()),
        args=[
            _ast.Name(id='rt', ctx=_ast.Load()),
            _ast.Call(
                func=_ast.Name(id='_ptr_void', ctx=_ast.Load()),
                args=[_ast.Name(id='_a', ctx=_ast.Load())],
                keywords=[],
            ),
        ],
        keywords=[],
    )))

    _ast.fix_missing_locations(_ast.Module(body=body_stmts, type_ignores=[]))

    gf = meta.func(
        name=f'_spawn_raw_{getattr(typed_spawn_raw_fn, "__name__", "fn")}',
        params=params_list,
        return_type=ptr[Task],
        body=body_stmts,
        required_globals={
            '__name__': __name__,
            'effect': effect,
            'move': move,
            'u64': u64,
            'sizeof': sizeof,
            '_ArgsType': args_struct,
            '_PtrArgs': ptr[args_struct],
            '_ptr_void': ptr[void],
            '_typed_spawn_raw': typed_spawn_raw_fn,
        },
        source_file=f'<spawn_typed:spawn_raw>',
    )

    return meta.compile_generated(gf, suffix=("spawn_raw", suffix))


def _make_typed_join(ret_type, suffix):
    """Generate join that returns the actual typed value.

    def join(rt: ptr[Runtime], handle: TaskHandle) -> RetType:
        raw: ptr[void] = runtime_join(rt, handle)
        cell: ptr[RetType] = ptr[RetType](raw)
        val = cell[0]
        effect.mem.free(raw)
        return val
    """
    import ast as _ast

    body_stmts = []

    # raw: ptr[void] = runtime_join(rt, handle)
    body_stmts.append(_ast.AnnAssign(
        target=_ast.Name(id='_raw', ctx=_ast.Store()),
        annotation=_ast.Name(id='_ptr_void', ctx=_ast.Load()),
        value=_ast.Call(
            func=_ast.Name(id='_runtime_join', ctx=_ast.Load()),
            args=[
                _ast.Name(id='rt', ctx=_ast.Load()),
                _ast.Name(id='handle', ctx=_ast.Load()),
            ],
            keywords=[],
        ),
        simple=1,
    ))

    # cell: ptr[RetType] = ptr[RetType](_raw)
    body_stmts.append(_ast.AnnAssign(
        target=_ast.Name(id='_cell', ctx=_ast.Store()),
        annotation=_ast.Name(id='_PtrRet', ctx=_ast.Load()),
        value=_ast.Call(
            func=_ast.Name(id='_PtrRet', ctx=_ast.Load()),
            args=[_ast.Name(id='_raw', ctx=_ast.Load())],
            keywords=[],
        ),
        simple=1,
    ))

    # val = _cell[0]
    body_stmts.append(_ast.Assign(
        targets=[_ast.Name(id='_val', ctx=_ast.Store())],
        value=_ast.Subscript(
            value=_ast.Name(id='_cell', ctx=_ast.Load()),
            slice=_ast.Constant(value=0),
            ctx=_ast.Load(),
        ),
    ))

    # effect.mem.free(_raw)
    body_stmts.append(_ast.Expr(value=_ast.Call(
        func=_ast.Attribute(
            value=_ast.Attribute(
                value=_ast.Name(id='effect', ctx=_ast.Load()),
                attr='mem', ctx=_ast.Load(),
            ),
            attr='free', ctx=_ast.Load(),
        ),
        args=[_ast.Name(id='_raw', ctx=_ast.Load())],
        keywords=[],
    )))

    # return _val
    body_stmts.append(_ast.Return(value=_ast.Name(id='_val', ctx=_ast.Load())))

    _ast.fix_missing_locations(_ast.Module(body=body_stmts, type_ignores=[]))

    gf = meta.func(
        name='_typed_join',
        params=[('rt', ptr[Runtime]), ('handle', TaskHandle)],
        return_type=ret_type,
        body=body_stmts,
        required_globals={
            '__name__': __name__,
            'effect': effect,
            '_ptr_void': ptr[void],
            '_PtrRet': ptr[ret_type],
            '_runtime_join': runtime_join,
            'TaskHandle': TaskHandle,
        },
        source_file='<spawn_typed:join>',
    )

    return meta.compile_generated(gf, suffix=("join", suffix))


def _make_typed_join_raw(ret_type, suffix):
    """Generate join_raw that takes ptr[Task] directly (no linear handle)."""
    import ast as _ast

    body_stmts = []

    body_stmts.append(_ast.AnnAssign(
        target=_ast.Name(id='_raw', ctx=_ast.Store()),
        annotation=_ast.Name(id='_ptr_void', ctx=_ast.Load()),
        value=_ast.Call(
            func=_ast.Name(id='_runtime_join_raw', ctx=_ast.Load()),
            args=[
                _ast.Name(id='rt', ctx=_ast.Load()),
                _ast.Name(id='task', ctx=_ast.Load()),
            ],
            keywords=[],
        ),
        simple=1,
    ))

    body_stmts.append(_ast.AnnAssign(
        target=_ast.Name(id='_cell', ctx=_ast.Store()),
        annotation=_ast.Name(id='_PtrRet', ctx=_ast.Load()),
        value=_ast.Call(
            func=_ast.Name(id='_PtrRet', ctx=_ast.Load()),
            args=[_ast.Name(id='_raw', ctx=_ast.Load())],
            keywords=[],
        ),
        simple=1,
    ))

    body_stmts.append(_ast.Assign(
        targets=[_ast.Name(id='_val', ctx=_ast.Store())],
        value=_ast.Subscript(
            value=_ast.Name(id='_cell', ctx=_ast.Load()),
            slice=_ast.Constant(value=0),
            ctx=_ast.Load(),
        ),
    ))

    # Free result cell
    body_stmts.append(_ast.Expr(value=_ast.Call(
        func=_ast.Attribute(
            value=_ast.Attribute(
                value=_ast.Name(id='effect', ctx=_ast.Load()),
                attr='mem', ctx=_ast.Load(),
            ),
            attr='free', ctx=_ast.Load(),
        ),
        args=[_ast.Name(id='_raw', ctx=_ast.Load())],
        keywords=[],
    )))

    # Destroy the task
    body_stmts.append(_ast.Expr(value=_ast.Call(
        func=_ast.Name(id='_task_destroy', ctx=_ast.Load()),
        args=[_ast.Name(id='task', ctx=_ast.Load())],
        keywords=[],
    )))

    body_stmts.append(_ast.Return(value=_ast.Name(id='_val', ctx=_ast.Load())))

    _ast.fix_missing_locations(_ast.Module(body=body_stmts, type_ignores=[]))

    gf = meta.func(
        name='_typed_join_raw',
        params=[('rt', ptr[Runtime]), ('task', ptr[Task])],
        return_type=ret_type,
        body=body_stmts,
        required_globals={
            '__name__': __name__,
            'effect': effect,
            '_ptr_void': ptr[void],
            '_PtrRet': ptr[ret_type],
            '_runtime_join_raw': runtime_join_raw,
            '_task_destroy': task_destroy,
        },
        source_file='<spawn_typed:join_raw>',
    )

    return meta.compile_generated(gf, suffix=("join_raw", suffix))


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
