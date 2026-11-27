import inspect
import ast
import sys
import textwrap
from functools import wraps
from ..logger import logger
from ..source import SourceLocation, set_ast_source


def inline(func=None, *, cls_method=False, method=False):
    """Decorator to mark a function for inline execution
    
    Can be used on standalone Python functions and class methods.
    When used on class methods in BuiltinEntity, automatically creates
    the appropriate classmethod handler and attaches it to the enclosing class.
    
    Args:
        func: The function to decorate (when used without arguments)
        cls_method: If True, creates handle_{func_name} classmethod and
            automatically attaches it to the enclosing class during class creation.
            The handler name is derived from the decorated function name.
            Examples: iter_begin -> handle_iter_begin, call -> handle_call
        method: If True, treats as instance method where first param is self (Python object).
            Creates handle_{func_name} classmethod that extracts self from args[0].
            Examples: self.iter_begin() -> handle_iter_begin
    
    Examples:
        @inline
        def add(a, b):
            return a + b
        
        class counter(BuiltinEntity):
            @inline(cls_method=True)
            def iter_begin(counter_obj):
                return 0
            
            @inline(method=True)
            def iter_begin(self) -> i32:
                return 0
            
            # After class creation, counter.handle_iter_begin is automatically available
    """
    def decorator(f):
        try:
            from ..utils.inspect_utils import get_object_sourcelines
            lines, start_line = get_object_sourcelines(f)
            filename = inspect.getfile(f)
        except (OSError, TypeError) as error:
            logger.error(f'Cannot get source code for function {f.__name__}: {error}',
                         node=call_site)
        source = textwrap.dedent(''.join(lines))
        try:
            func_ast = ast.parse(source, filename=filename).body[0]
        except (SyntaxError, IndexError) as error:
            logger.error(error, node=SourceLocation(filename, start_line))
        set_ast_source(func_ast, (filename, start_line - 1))
        func_ast._pc_diagnose_expansion_at_call = getattr(
            f, '_pc_diagnose_expansion_at_call', False)
        if not isinstance(func_ast, ast.FunctionDef):
            logger.error(f'Expected function definition, got {type(func_ast).__name__}',
                         node=func_ast, exc_type=ValueError)

        from ..call_normalization import inline_parameter_defaults
        from .visible import definition_names, get_all_accessible_symbols
        from ..type_resolver import TypeResolver

        defaults = inline_parameter_defaults(f)
        names = definition_names(f, func_ast)
        captured_symbols = TypeResolver.capture_type_namespace(names, definition=f)
        merged_func_globals = get_all_accessible_symbols(
            f, include_builtins=True, captured_symbols=captured_symbols, names=names)

        def definition_globals():
            values = {name: f.__globals__[name] for name in names if name in f.__globals__}
            values.update(merged_func_globals)
            return values

        return _make_inline_callable(
            func_ast, definition_globals, defaults=defaults, python_func=f,
            cls_method=cls_method, method=method)

    call_site = logger.current_location() or SourceLocation.from_frame(sys._getframe(1))
    if cls_method and method:
        logger.error("Cannot use both cls_method=True and method=True",
                     node=call_site, exc_type=ValueError)
    
    # Handle both @inline and @inline(...) usage
    if func is None:
        # Called with arguments: @inline(cls_method=..., method=...)
        return decorator
    else:
        # Called without arguments: @inline
        return decorator(func)


def _make_inline_callable(func_ast, definition_globals, *, defaults=None,
                          python_func=None, cls_method=False, method=False):
    """Bind Python and generated inline bodies through one call/expansion path."""
    from ..call_normalization import (
        NormalizedCallArgs, bind_call_args, lower_inline_signature,
    )

    func_ast, signature = lower_inline_signature(func_ast, defaults or {})
    param_names = signature['param_names']

    def create_handler(visitor, args, node):
        if not isinstance(args, NormalizedCallArgs):
            args = bind_call_args(
                signature, args, getattr(args, 'keywords', {}), node, visitor=visitor)
        from .._inline import InlineAdapter
        adapter = InlineAdapter(
            visitor, dict(zip(param_names, args)), func_globals=definition_globals())
        return adapter.execute_inline(func_ast, node)

    def classmethod_handler(cls, visitor, func_ref, args, node):
        return create_handler(visitor, args, node)

    def handle_call_method(visitor, func_ref, args, node):
        return create_handler(visitor, args, node)

    def wrapper(*args, **kwargs):
        if python_func is None:
            node = logger.current_location() or SourceLocation.from_frame(sys._getframe(1))
            logger.error('Generated inline functions can only be called in compiled code',
                         node=node, exc_type=TypeError)
        return python_func(*args, **kwargs)

    if python_func is None:
        wrapper.__name__ = func_ast.name
    else:
        wrapper = wraps(python_func)(wrapper)

    if cls_method or method:
        handler_name = f'handle_{wrapper.__name__}'
        handler_method = classmethod(classmethod_handler)
        setattr(wrapper, handler_name, handler_method)
        wrapper._handler_name = handler_name
        wrapper._handler_method = handler_method

    wrapper._is_inline = True
    wrapper.defer_linear_transfer = True
    wrapper._pc_compile_level_callable = True
    wrapper._func_ast = func_ast
    wrapper._param_names = param_names
    wrapper._pc_call_signature = signature
    wrapper._original_func = python_func
    wrapper._inline_cls_method = cls_method
    wrapper._inline_method = method
    wrapper.handle_call = handle_call_method
    return wrapper

def register_inline_handlers(cls):
    """Helper to register all @inline(cls_method=True) or @inline(method=True) handlers on a class.
    
    Call this after class definition to automatically attach handle_* methods.
    Or better yet, use it as a class decorator:
    
    @register_inline_handlers
    class MyClass(BuiltinEntity):
        @inline(cls_method=True)
        def iter_begin(obj):
            ...
        
        @inline(method=True)
        def iter_begin(self):
            ...
    """
    for name in dir(cls):
        attr = getattr(cls, name)
        # Check for either cls_method or method
        is_cls_method = hasattr(attr, '_inline_cls_method') and attr._inline_cls_method
        is_method = hasattr(attr, '_inline_method') and attr._inline_method
        if is_cls_method or is_method:
            if hasattr(attr, '_handler_name') and hasattr(attr, '_handler_method'):
                setattr(cls, attr._handler_name, attr._handler_method)
    return cls
