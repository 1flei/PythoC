"""Typed linear Future API over the active executor effect.

Future models async result ownership.  The active effect.executor decides how
the task is scheduled: N:M runtime, single-thread coroutine, thread pool, or
another backend.  The default backend is the N:M runtime executor.
"""
from __future__ import annotations

import ast
import copy
from types import SimpleNamespace

from .policy import bind_mem
bind_mem()

from pythoc import compile, effect, inline, meta, ptr, void, u64, sizeof, move
from pythoc.builtin_entities.python_type import PythonType
from pythoc.effect import get_current_compilation_context
from pythoc.logger import logger
from pythoc.type_converter import get_base_type
from pythoc.valueref import wrap_value

from .spawn_typed import _TypedTask, _extract_params, _extract_return_type
from .executor_effect import DefaultExecutor, ExecutorHandle

effect.default(executor=DefaultExecutor)

_FUTURE_REGISTRY = {}
_FUTURE_TASK_CACHE = {}
_FUTURE_DO_TASK_CACHE = {}
_FUTURE_ABI_VERSION = "future_do_returns_future_v4"


def _current_executor_binding(suffix):
    ctx = get_current_compilation_context()
    if ctx is not None:
        overrides = ctx.get("effect_overrides") or {}
        if "executor" in overrides:
            return overrides["executor"], ctx.get("effect_suffix")
        executor_suffix = suffix if suffix is not None else None
        return DefaultExecutor, executor_suffix
    executor_impl = effect._resolve_effect("executor", __name__) or DefaultExecutor
    executor_suffix = suffix if suffix is not None else effect._get_current_suffix()
    return executor_impl, executor_suffix


def _executor_binding_from_visitor(visitor, suffix=None):
    binding = getattr(visitor, "binding_state", None)
    if binding is None:
        return _current_executor_binding(suffix)

    executor_suffix = getattr(binding, "effect_suffix", None)
    overrides = getattr(binding, "captured_effect_context", None) or {}
    if executor_suffix is not None and "executor" in overrides:
        return overrides["executor"], executor_suffix
    return DefaultExecutor, suffix


def _FutureTask(target_fn, *, stack_size=None, suffix=None, executor_binding=None):
    """Generate typed linear Future helpers for a @compile function.

    The returned namespace provides:
        spawn(args...) -> FutureType
        join(future) -> return_type
        view(future) -> generator[return_type]
        do(genexp) -> return_type
        detach(future) -> void

    A Future stores the linear backend handle returned by the active executor.
    """
    if executor_binding is None:
        executor_impl, executor_suffix = _current_executor_binding(suffix)
    else:
        executor_impl, executor_suffix = executor_binding
    cache_key = (target_fn, stack_size, id(executor_impl), executor_suffix)
    cached = _FUTURE_TASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    typed = _TypedTask(target_fn, stack_size=stack_size)
    param_info = _extract_params(target_fn)
    ret_type = _extract_return_type(target_fn)
    fn_name = getattr(target_fn, '__name__', 'anon')
    type_suffix = (
        _FUTURE_ABI_VERSION,
        "future",
        fn_name,
        tuple(t for _, t in param_info),
        ret_type,
        executor_suffix,
    )
    has_return = ret_type is not void and ret_type is not None
    stk = stack_size if stack_size is not None else u64(0)

    @compile(suffix=type_suffix)
    class _Future:
        handle: ExecutorHandle

    @compile(suffix=type_suffix)
    def _future_new(handle: ExecutorHandle) -> _Future:
        f: _Future
        f.handle = move(handle)
        return f

    spawn_fn = _make_future_spawn_wrapper(
        param_info,
        typed.args_type,
        typed.trampoline,
        stk,
        executor_impl.spawn,
        _future_new,
        _Future,
        type_suffix,
    )

    if has_return:
        @compile(suffix=type_suffix)
        def _future_join(f: _Future) -> ret_type:
            raw: ptr[void] = executor_impl.join(f.handle)
            cell: ptr[ret_type] = ptr[ret_type](raw)
            value = cell[0]
            effect.mem.free(raw)
            return value

        @compile(suffix=type_suffix)
        def _future_view(f: _Future) -> ret_type:
            yield _future_join(f)
        _future_view.defer_linear_transfer = True

        @inline
        def _future_do(genexp) -> ret_type:
            for value in genexp:
                return value

    else:
        @compile(suffix=type_suffix)
        def _future_join(f: _Future) -> void:
            executor_impl.join(f.handle)

        @compile(suffix=type_suffix)
        def _future_view(f: _Future) -> void:
            _future_join(f)
        _future_view.defer_linear_transfer = True

        @inline
        def _future_do(genexp) -> void:
            for _ in genexp:
                pass

    @compile(suffix=type_suffix)
    def _future_detach(f: _Future) -> void:
        executor_impl.detach(f.handle)

    ns = SimpleNamespace(
        type=_Future,
        ret_type=ret_type,
        suffix=type_suffix,
        executor_impl=executor_impl,
        executor_suffix=executor_suffix,
        spawn=spawn_fn,
        join=_future_join,
        view=_future_view,
        do=_future_do,
        detach=_future_detach,
    )
    _FUTURE_REGISTRY[_Future] = ns
    _FUTURE_TASK_CACHE[cache_key] = ns
    return ns


def _python_func_ref(fn):
    return wrap_value(
        fn,
        kind="python",
        type_hint=PythonType.wrap(fn, is_constant=True),
    )


def _target_from_arg(arg, node):
    if not arg.is_python_value():
        logger.error("Future.spawn requires a compile-time function", node=node)
    return arg.get_python_value()


def _harness_for_future(arg, node):
    future_type = get_base_type(getattr(arg, "type_hint", None))
    harness = _FUTURE_REGISTRY.get(future_type)
    if harness is None:
        logger.error("Future operation requires a Future value", node=node)
    return harness


def _call_python_fn(visitor, fn, args, node):
    return visitor.value_dispatcher.handle_call(
        _python_func_ref(fn),
        args,
        node,
    )


def _is_future_view_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "view"
        and node.args
    )


def _make_future_do_task(genexp_node, output_harness, view_harnesses, future_types):
    key = (
        _FUTURE_ABI_VERSION,
        ast.dump(genexp_node),
        output_harness.suffix,
        tuple(h.suffix for h in view_harnesses),
    )
    cached = _FUTURE_DO_TASK_CACHE.get(key)
    if cached is not None:
        return cached

    genexp = copy.deepcopy(genexp_node)
    view_idx = 0
    for gen in genexp.generators:
        if _is_future_view_call(gen.iter):
            gen.iter = ast.Call(
                func=ast.Name(id=f'_future_view_{view_idx}', ctx=ast.Load()),
                args=[ast.Name(id=f'_future_{view_idx}', ctx=ast.Load())],
                keywords=[],
            )
            view_idx = view_idx + 1

    body_stmts = [
        ast.Return(value=ast.Call(
            func=ast.Name(id='_future_do_value', ctx=ast.Load()),
            args=[genexp],
            keywords=[],
        ))
    ]
    ast.fix_missing_locations(ast.Module(body=body_stmts, type_ignores=[]))

    required_globals = {
        '__name__': __name__,
        '_future_do_value': output_harness.do,
    }
    for i, harness in enumerate(view_harnesses):
        required_globals[f'_future_view_{i}'] = harness.view

    gf = meta.func(
        name='_future_do_task',
        params=[
            (f'_future_{i}', future_type)
            for i, future_type in enumerate(future_types)
        ],
        return_type=output_harness.ret_type,
        body=body_stmts,
        required_globals=required_globals,
        source_file='<runtime_future:do>',
    )

    compiled = meta.compile_generated(gf, suffix=("future_do", key))
    compiled._pc_param_names = [name for name, _ in gf.params]
    compiled._pc_param_types = [param_type for _, param_type in gf.params]
    compiled._pc_return_type = output_harness.ret_type
    _FUTURE_DO_TASK_CACHE[key] = compiled
    return compiled


class _FutureSpawnOp:
    defer_linear_transfer = True

    def handle_call(self, visitor, func_ref, args, node):
        if not args:
            logger.error("Future.spawn requires a function", node=node)
        target_fn = _target_from_arg(args[0], node)
        harness = _FutureTask(
            target_fn,
            executor_binding=_executor_binding_from_visitor(visitor),
        )
        return _call_python_fn(visitor, harness.spawn, args[1:], node)


class _FutureJoinOp:
    defer_linear_transfer = True

    def handle_call(self, visitor, func_ref, args, node):
        if len(args) != 1:
            logger.error("Future.join expects one Future value", node=node)
        harness = _harness_for_future(args[0], node)
        return _call_python_fn(visitor, harness.join, args, node)


class _FutureViewOp:
    defer_linear_transfer = True

    def handle_call(self, visitor, func_ref, args, node):
        if len(args) != 1:
            logger.error("Future.view expects one Future value", node=node)
        harness = _harness_for_future(args[0], node)
        return _call_python_fn(visitor, harness.view, args, node)


class _FutureDetachOp:
    defer_linear_transfer = True

    def handle_call(self, visitor, func_ref, args, node):
        if len(args) != 1:
            logger.error("Future.detach expects one Future value", node=node)
        harness = _harness_for_future(args[0], node)
        return _call_python_fn(visitor, harness.detach, args, node)


class _FutureDoOp:
    defer_linear_transfer = True

    def handle_call(self, visitor, func_ref, args, node):
        if len(args) != 1:
            logger.error("Future.do expects one generator expression", node=node)
        output_harness, view_harnesses, future_refs = self._future_sources(
            visitor, node
        )
        future_types = [
            get_base_type(getattr(ref, "type_hint", None))
            for ref in future_refs
        ]
        task_fn = _make_future_do_task(
            node.args[0],
            output_harness,
            view_harnesses,
            future_types,
        )
        task_harness = _FutureTask(
            task_fn,
            executor_binding=(
                output_harness.executor_impl,
                output_harness.executor_suffix,
            ),
        )
        return _call_python_fn(visitor, task_harness.spawn, future_refs, node)

    def _future_sources(self, visitor, node):
        if not node.args or not isinstance(node.args[0], ast.GeneratorExp):
            logger.error("Future.do expects a generator expression", node=node)
        future_refs = []
        view_harnesses = []
        for gen in node.args[0].generators:
            iter_node = gen.iter
            if _is_future_view_call(iter_node):
                future_ref = visitor.visit_rvalue_expression(iter_node.args[0])
                future_refs.append(future_ref)
                view_harnesses.append(_harness_for_future(future_ref, iter_node))
        if view_harnesses:
            return view_harnesses[0], view_harnesses, future_refs
        logger.error("Future.do requires at least one Future.view source", node=node)


class _FutureFacade:
    spawn = _FutureSpawnOp()
    join = _FutureJoinOp()
    view = _FutureViewOp()
    detach = _FutureDetachOp()
    do = _FutureDoOp()


def _make_future_spawn_wrapper(
    param_info,
    args_struct,
    trampoline,
    stack_size,
    executor_spawn_fn,
    future_new_fn,
    future_type,
    suffix,
):
    """Generate spawn(args...) -> FutureType."""
    import ast as _ast

    params_list = list(param_info)
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

    body_stmts.append(_ast.AnnAssign(
        target=_ast.Name(id='_handle', ctx=_ast.Store()),
        annotation=_ast.Name(id='ExecutorHandle', ctx=_ast.Load()),
        value=_ast.Call(
            func=_ast.Name(id='_executor_spawn', ctx=_ast.Load()),
            args=[
                _ast.Name(id='_trampoline', ctx=_ast.Load()),
                _ast.Call(
                    func=_ast.Name(id='_ptr_void', ctx=_ast.Load()),
                    args=[_ast.Name(id='_a', ctx=_ast.Load())],
                    keywords=[],
                ),
                _ast.Name(id='_stack_size', ctx=_ast.Load()),
            ],
            keywords=[],
        ),
        simple=1,
    ))

    body_stmts.append(_ast.Return(value=_ast.Call(
        func=_ast.Name(id='_future_new', ctx=_ast.Load()),
        args=[
            _ast.Name(id='_handle', ctx=_ast.Load()),
        ],
        keywords=[],
    )))

    _ast.fix_missing_locations(_ast.Module(body=body_stmts, type_ignores=[]))

    gf = meta.func(
        name=f'_future_spawn_{getattr(trampoline, "__name__", "fn")}',
        params=params_list,
        return_type=future_type,
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
            '_trampoline': trampoline,
            '_stack_size': stack_size,
            '_executor_spawn': executor_spawn_fn,
            '_future_new': future_new_fn,
            'ExecutorHandle': ExecutorHandle,
        },
        source_file='<runtime_future:spawn>',
    )

    return meta.compile_generated(gf, suffix=("future_spawn", suffix))


Future = _FutureFacade()


__all__ = ["Future"]
