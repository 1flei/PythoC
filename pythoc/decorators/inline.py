import inspect
import ast
import textwrap
from functools import wraps
from ..logger import logger, set_source_context


# Registry to track inline methods that need to be attached to classes
_pending_inline_handlers = []


def inline(func=None, *, cls_method=False, method=False):
    """Decorator to mark a function for inline execution
    
    Can be used on both standalone functions and class methods.
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
            source = inspect.getsource(f)
            source = textwrap.dedent(source)
            # Get function start line for accurate error messages
            try:
                _, start_line = inspect.getsourcelines(f)
                source_file = inspect.getfile(f)
                set_source_context(source_file, start_line - 1)
            except (OSError, TypeError):
                pass
            tree = ast.parse(source)
            func_ast = tree.body[0]
            if not isinstance(func_ast, ast.FunctionDef):
                raise ValueError(f"Expected function definition, got {type(func_ast).__name__}")
        except (OSError, TypeError) as e:
            raise RuntimeError(f"Cannot get source code for function {f.__name__}: {e}")

        param_names = [arg.arg for arg in func_ast.args.args]

        def create_handler(visitor, args, node, self_obj=None):
            # For method=True, first arg is self (Python object), rest are IR args
            if method:
                if len(args) < 1:
                    raise TypeError(f"{f.__name__}() missing self argument")
                # args[0] is iterable (Python object or wrapped value)
                # Extract the actual Python object
                if hasattr(args[0], 'kind') and args[0].kind == 'python':
                    self_obj = args[0].get_python_value()
                elif hasattr(args[0], 'value') and hasattr(args[0].value, '__class__'):
                    # Direct Python object
                    self_obj = args[0].value
                else:
                    # Assume it's the object itself
                    self_obj = args[0]
                
                # Remaining args are the method parameters (excluding self)
                method_args = args[1:]
                expected_param_count = len(param_names) - 1  # Exclude self
                if len(method_args) != expected_param_count:
                    raise TypeError(f"{f.__name__}() takes {expected_param_count} arguments (excluding self), got {len(method_args)}")
                
                # Bind self to the first param_name, rest to method_args
                param_bindings = {param_names[0]: self_obj}
                param_bindings.update(dict(zip(param_names[1:], method_args)))
            else:
                # Standard inline or cls_method
                if len(args) != len(param_names):
                    raise TypeError(f"{f.__name__}() takes {len(param_names)} arguments, got {len(args)}")
                param_bindings = dict(zip(param_names, args))
            
            # Execute inline to generate IR using unified kernel
            # Pass func_globals so adapter can merge them with caller's globals
            from ..inline import InlineAdapter
            adapter = InlineAdapter(visitor, param_bindings, func_globals=f.__globals__)
            result = adapter.execute_inline(func_ast)
            return result

        # Create a classmethod handler that ignores the cls parameter
        def classmethod_handler(cls, visitor, func_ref, args, node):
            return create_handler(visitor, args, node)

        # Create handle_call for PythonType interception
        def handle_call_method(visitor, func_ref, args, node):
            return create_handler(visitor, args, node)

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        # Determine handler name from function name if cls_method=True or method=True
        if cls_method or method:
            handler_name = f'handle_{f.__name__}'
            handler_method = classmethod(classmethod_handler)
            # Attach the classmethod handler to the wrapper
            setattr(wrapper, handler_name, handler_method)
            # Store for later registration
            wrapper._handler_name = handler_name
            wrapper._handler_method = handler_method
        
        # Common metadata
        wrapper._is_inline = True
        wrapper._func_ast = func_ast
        wrapper._param_names = param_names
        wrapper._original_func = f
        wrapper._inline_cls_method = cls_method
        wrapper._inline_method = method
        wrapper.handle_call = handle_call_method  # For PythonType interception
        
        return wrapper
    
    # Validate parameters
    if cls_method and method:
        raise ValueError("Cannot use both cls_method=True and method=True")
    
    # Handle both @inline and @inline(...) usage
    if func is None:
        # Called with arguments: @inline(cls_method=..., method=...)
        return decorator
    else:
        # Called without arguments: @inline
        return decorator(func)


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
