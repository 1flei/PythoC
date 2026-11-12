"""Annotation evaluation and definition-scoped PythoC type binding."""

import ast
from dataclasses import dataclass, field
import inspect
import keyword
import symtable
from types import CodeType
from typing import Any, Optional

from llvmlite import ir


@dataclass(eq=False)
class _TypeScope:
    """One lexical activation, retaining a frame only for pending local names."""

    code: object
    local_names: frozenset
    declarations: dict = field(default_factory=dict)
    pending: set = field(default_factory=set)
    frame: object = None

    def lookup(self, name):
        if name in self.declarations:
            return self.declarations[name]
        if self.frame is not None:
            return self.frame.f_locals.get(name, _MISSING)
        return _MISSING

    def define(self, name, value):
        self.declarations[name] = value
        self.pending.discard(name)
        if not self.pending:
            self.release()

    def release(self):
        if self.frame is not None:
            from .session import CompileSession
            CompileSession.current().type_graph.type_scopes.pop(id(self.frame), None)
            self.frame = None

    def finish(self):
        values = {name: self.lookup(name) for name in self.pending}
        if all(value is not _MISSING for value in values.values()):
            self.declarations.update(values)
            self.pending.clear()
            self.release()

    def contains(self, code):
        pending = list(self.code.co_consts)
        while pending:
            constant = pending.pop()
            if constant is code:
                return True
            if isinstance(constant, CodeType):
                pending.extend(constant.co_consts)
        return False


@dataclass(eq=False)
class TypeReference:
    """A scoped expression; evaluation and frozen bindings live in TypeGraph."""

    expression: str
    namespace: dict

    def __repr__(self):
        return f'TypeReference({self.expression!r})'

    def get_name(self):
        return self.expression

    def get_type_id(self, _visited=None):
        from .type_id import get_type_id
        return get_type_id(self)

    def get_llvm_type(self, module_context=None):
        return TypeResolver.resolve_type(self, freeze=True).get_llvm_type(module_context)

    def get_layout(self):
        return TypeResolver.resolve_type(self, freeze=True).get_layout()

    def get_ctypes_type(self):
        return TypeResolver.resolve_type(self, freeze=True).get_ctypes_type()



class _DependencySyntax(ast.NodeTransformer):
    def visit_Slice(self, node):
        # PythoC slices carry named items. Only the type expression is read;
        # the lower bound is a label, and the step is not evaluated.
        return (self.visit(node.upper) if node.upper is not None
                else ast.Constant(None))


_MISSING = object()


class TypeResolver:
    """Evaluate annotations and bind type references through one API."""

    def __init__(self, module_context=None, user_globals=None, visitor=None):
        """
        Initialize the type resolver.

        Args:
            module_context: Optional LLVM module context for struct types
            user_globals: Optional user global namespace for type alias resolution
            visitor: AST visitor for expression evaluation, created lazily
                when not provided
        """
        self.module_context = module_context
        self.user_globals = user_globals if user_globals is not None else {}
        self.visitor = visitor
        self._constexpr_visitor = None  # Lazy-created constexpr visitor

    def _get_constexpr_visitor(self):
        """Get or create a constexpr visitor for type evaluation.

        This visitor uses ConstexprBackend and can evaluate type expressions
        without LLVM context.
        """
        if self._constexpr_visitor is None:
            from .backend import ConstexprBackend
            from .ast_visitor import LLVMIRVisitor

            backend = ConstexprBackend(user_globals=self.user_globals)
            self._constexpr_visitor = LLVMIRVisitor(
                backend=backend,
                user_globals=self.user_globals
            )
        return self._constexpr_visitor

    def parse_annotation(self, annotation, *, defer_unresolved=False) -> Optional[Any]:
        """Evaluate an annotation, deferring it when dependencies are missing."""
        if (defer_unresolved and self.visitor is None
                and isinstance(annotation, (str, ast.AST))):
            expression = (ast.unparse(annotation)
                          if isinstance(annotation, ast.AST) else annotation)
            if TypeResolver._missing_expression_names(
                    expression, self.user_globals, registry=False):
                return TypeReference(expression, TypeResolver._reference_namespace(
                    expression, self.user_globals))
        if annotation is None:
            return None
        result = self._extract_type_from_valueref(
            self._evaluate_type_expression(annotation))
        return TypeResolver.bind_type_references(result, self.user_globals)

    def resolve_type_value(self, value_ref):
        """Validate an evaluated type value without executing syntax."""
        return self._extract_type_from_valueref(value_ref)

    @staticmethod
    def annotation_expression(annotation):
        """Expose names in quoted annotations before binding or evaluation."""
        while isinstance(annotation, str) or (
                isinstance(annotation, ast.Constant)
                and isinstance(annotation.value, str)):
            text = annotation if isinstance(annotation, str) else annotation.value
            annotation = ast.parse(text, mode="eval").body
        return annotation

    def _evaluate_type_expression(self, node):
        """
        Evaluate type annotation AST, returning ValueRef(kind='python').

        Uses provided visitor if available (to access local type variables),
        otherwise uses constexpr visitor.

        Args:
            node: AST node, string, or Python value

        Returns:
            ValueRef(kind='python', value=<type>)
        """
        node = self.annotation_expression(node)
        if not isinstance(node, ast.AST):
            return self._wrap_as_python_value(node)

        # Use provided visitor (for local type variable access) or constexpr visitor
        # The visitor's visit_List will check is_constexpr() or element types
        # to decide whether to return Python list or pc_list
        visitor = self.visitor if self.visitor is not None else self._get_constexpr_visitor()
        if self.visitor is None:
            expression = ast.unparse(node)
            names = TypeResolver.expression_names(expression)
            namespace = TypeResolver._reference_namespace(expression, self.user_globals)
            namespace.update(TypeResolver._resolution_namespace(self.user_globals, names))
            namespace.type_binding_namespace = self.user_globals
            visitor.ctx.user_globals = namespace
        return visitor.visit_expression(node)

    def _wrap_as_python_value(self, value):
        """Wrap Python value as a canonical compile-time Python constant."""
        from .valueref import ValueRef, wrap_python_constant
        if isinstance(value, ValueRef):
            return value
        return wrap_python_constant(value)

    def _extract_type_from_valueref(self, value_ref):
        """Extract type from ValueRef(kind='python') or raw value."""
        from .valueref import ValueRef
        from .type_check import is_python_value

        if value_ref is None:
            return None

        if isinstance(value_ref, ValueRef):
            if not is_python_value(value_ref):
                raise TypeError(f"Expected Python type ValueRef, got {value_ref!r}")
            value = value_ref.value
        else:
            value = value_ref

        if isinstance(value, TypeReference):
            return value

        # None value -> void type (for return type annotation like -> None)
        if value is None:
            from .builtin_entities import void
            return void

        if isinstance(value, str):
            return TypeResolver.resolve_type(value, self.user_globals)

        return TypeResolver._require_type(value)

    @staticmethod
    def _require_type(value):
        """Validate evaluated and directly named runtime types identically."""
        from .builtin_entities.base import _is_type_like

        if getattr(value, '_is_param', False):
            raise TypeError("'param' cannot be used as a runtime type")
        if not _is_type_like(value):
            raise TypeError(f"Cannot use as type: {value} (type: {type(value)})")
        return value

    def annotation_to_llvm_type(self, annotation) -> ir.Type:
        """
        Parse type annotation and return LLVM type.

        This is a convenience method that combines parse_annotation and
        extraction of LLVM type.

        Args:
            annotation: AST node representing a type annotation

        Returns:
            LLVM IR type
        """
        builtin_type = self.parse_annotation(annotation)

        if builtin_type is None:
            raise TypeError(f"TypeResolver: Cannot resolve type annotation: {annotation}")

        return builtin_type.get_llvm_type(self.module_context)

    @staticmethod
    def resolve_type(value, namespace=None, *, required=True, freeze=False):
        """Observe names, or prepare and freeze a deferred expression.

        Optional observations never execute a deferred constructor. Explicit
        preparation evaluates it once per successful session binding; failed
        preparation leaves its references available for a later retry.
        """
        if not isinstance(value, (str, TypeReference)):
            return value
        from .session import CompileSession
        graph = CompileSession.current().type_graph
        with graph._lock:
            if freeze and graph._preparation is None:
                with graph.binding_transaction():
                    resolved = TypeResolver._resolve_type(value, namespace, required, True, graph)
                    if resolved is not None:
                        graph.prepare_type(resolved)
                    return resolved
            return TypeResolver._resolve_type(value, namespace, required, freeze, graph)

    @staticmethod
    def bind_type_references(pc_type, namespace=None, *, self_types=None, local_lookup=None):
        """Capture scope for type components, preserving string labels and tags."""
        from .builtin_entities.base import _is_type_like

        def reference(value):
            expression = value.expression if isinstance(value, TypeReference) else value
            if isinstance(value, TypeReference):
                owner = getattr(value.namespace, 'type_owner_name', None)
                if self_types is not None and owner in self_types:
                    value.namespace[owner] = self_types[owner]
                    scope = getattr(value.namespace, 'type_scope', None)
                    if scope is not None:
                        scope.define(owner, self_types[owner])
                    if expression == owner:
                        return self_types[owner]
                return value
            scope = TypeResolver._reference_namespace(expression, namespace)
            if local_lookup is not None:
                for name in TypeResolver.expression_names(expression):
                    found, value = local_lookup(name)
                    if found:
                        scope[name] = value
            known = (TypeResolver._lookup_name(scope, expression, registry=False)
                     if expression.isidentifier() else _MISSING)
            return known if _is_type_like(known) else TypeReference(expression, scope)

        return TypeResolver._map_type_components(pc_type, reference)

    @staticmethod
    def resolve_signature_types(info):
        """Use the same completed signature for codegen and Python adapters."""
        from .type_graph import prepare_type
        binding = getattr(info, 'binding_state', None)
        namespace = getattr(binding, 'compilation_globals', None)
        from .session import CompileSession
        graph = CompileSession.current().type_graph
        with graph._lock, graph.binding_transaction():
            parameters = {name: TypeResolver._resolve_type_components(
                pc_type, namespace, freeze=True)
                for name, pc_type in info.param_type_hints.items()}
            result = (TypeResolver._resolve_type_components(
                info.return_type_hint, namespace, freeze=True)
                if info.return_type_hint is not None else None)
            for pc_type in (*parameters.values(), result):
                prepare_type(pc_type)
        info.param_type_hints.update(parameters)
        info.return_type_hint = result

    @staticmethod
    def capture_type_namespace(names=None, *, definition=None):
        """Capture names from the definition's lexical activation."""
        from .decorators.visible import AccessibleSymbols

        if definition is not None and names is None:
            names = set()
            for annotation in getattr(definition, '__annotations__', {}).values():
                if isinstance(annotation, str):
                    names.update(TypeResolver.expression_names(annotation))
                else:
                    def observe(value):
                        expression = (value.expression
                                      if isinstance(value, TypeReference) else value)
                        names.update(TypeResolver.expression_names(expression))
                        return value
                    TypeResolver._map_type_components(annotation, observe)
        frame = inspect.currentframe()
        try:
            frame = frame.f_back
            parent = (definition.__qualname__.rsplit('.', 1)[0]
                      if definition is not None and '.' in definition.__qualname__ else '')
            parent = parent.removesuffix('.<locals>')
            while frame is not None:
                module = str(frame.f_globals.get('__name__', ''))
                if definition is not None:
                    qualname = frame.f_code.co_qualname
                    if module == definition.__module__ and (
                            qualname == parent
                            or (not parent and frame.f_code.co_name == '<module>')):
                        break
                elif (not (module == 'pythoc' or module.startswith('pythoc.'))
                      or module.startswith(('pythoc.std', 'pythoc.libc'))):
                    break
                frame = frame.f_back
            if frame is None:
                return AccessibleSymbols()
            requested = set(frame.f_globals) | set(frame.f_locals) if names is None else set(names)
            namespace = {name: frame.f_locals[name] if name in frame.f_locals
                         else frame.f_globals[name] for name in requested
                         if name in frame.f_locals or name in frame.f_globals}
            owner = None
            if '__module__' in frame.f_locals and '__qualname__' in frame.f_locals:
                owner = frame.f_locals['__qualname__'].rsplit('.', 1)[-1]
                namespace.pop(owner, None)
            result = AccessibleSymbols(namespace, live_globals=frame.f_globals)
            result.type_owner_name = owner
            result.type_owner_namespace = frame.f_locals if owner is not None else None
            scope_frame = frame.f_back if owner is not None else frame
            if scope_frame is not None and scope_frame.f_locals is not scope_frame.f_globals:
                from .session import CompileSession
                scopes = CompileSession.current().type_graph.type_scopes
                scope = scopes.get(id(scope_frame))
                if scope is None:
                    local_names = frozenset(scope_frame.f_code.co_varnames
                                            + scope_frame.f_code.co_cellvars
                                            + scope_frame.f_code.co_freevars)
                    scope = _TypeScope(scope_frame.f_code, local_names)
                missing = requested.intersection(scope.local_names) - scope_frame.f_locals.keys()
                scope.pending.update(missing - scope.declarations.keys())
                if scope.pending:
                    scope.frame = scope_frame
                    scopes[id(scope_frame)] = scope
                result.type_scope = scope
                result.type_locals = {name: value for name, value in scope_frame.f_locals.items()
                                      if name in requested}
            return result
        finally:
            del frame

    @staticmethod
    def expression_names(expression):
        names = set()
        for node in ast.walk(TypeResolver._dependency_tree(expression)):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                try:
                    names.update(TypeResolver.expression_names(node.value))
                except SyntaxError:
                    pass
        return names

    @staticmethod
    def reference_key(value):
        """Distinguish lexical references while preserving shared opaque tags."""
        if isinstance(value, TypeReference):
            names = TypeResolver.expression_names(value.expression)
            owner = getattr(value.namespace, 'type_owner_name', None)
            if owner in names:
                return (value.expression, id(value.namespace.type_owner_namespace))
            scope = getattr(value.namespace, 'type_scope', None)
            if scope is not None and names.intersection(scope.local_names):
                return (value.expression, id(scope))
            return (value.expression, None)
        return (value, None)

    @staticmethod
    def publish_type_definition(name, pc_type):
        """Make a local declaration visible to references from this factory call."""
        if not isinstance(pc_type, type):
            return
        namespace = TypeResolver.capture_type_namespace((), definition=pc_type)
        scope = getattr(namespace, 'type_scope', None)
        if scope is not None and name in scope.local_names:
            scope.define(name, pc_type)

    @staticmethod
    def _missing_expression_names(expression, namespace, *, registry=True):
        """Check immediate dependencies before executing any annotation code."""
        tree = TypeResolver._dependency_tree(expression)
        node = tree.body
        while isinstance(node, ast.Constant) and isinstance(node.value, str):
            expression = node.value
            tree = TypeResolver._dependency_tree(expression)
            node = tree.body
        symbols = symtable.symtable(
            ast.unparse(tree), '<annotation>', 'eval').get_symbols()
        return [symbol.get_name() for symbol in symbols
                if symbol.is_referenced() and TypeResolver._lookup_name(
                    namespace, symbol.get_name(), registry=registry) is _MISSING]

    @staticmethod
    def _reference_namespace(expression, namespace):
        """Capture expression dependencies, without copying a module per name."""
        from .decorators.visible import AccessibleSymbols

        names = TypeResolver.expression_names(expression)
        if namespace is None:
            return TypeResolver.capture_type_namespace(names)
        captured = {}
        for name in names:
            value = TypeResolver._lookup_name(namespace, name, registry=False)
            if value is not _MISSING:
                captured[name] = value
        live = getattr(namespace, 'live_globals', None)
        result = AccessibleSymbols(
            captured, live_globals=live if live is not None else namespace)
        result.type_scope = getattr(namespace, 'type_scope', None)
        result.type_owner_name = getattr(namespace, 'type_owner_name', None)
        result.type_owner_namespace = getattr(namespace, 'type_owner_namespace', None)
        local = getattr(namespace, 'type_locals', None)
        result.type_locals = (
            {name: value for name, value in local.items() if name in names}
            if local is not None else None)
        return result

    @staticmethod
    def _resolution_namespace(namespace, names):
        """Use captures, then late defining-module names, then the session registry."""
        result = {}
        for name in (*names, '__builtins__', '__name__', '__file__', '__package__'):
            value = TypeResolver._lookup_name(namespace, name)
            if value is not _MISSING:
                result[name] = value
        return result

    @staticmethod
    def _resolve_type_components(pc_type, namespace=None, *, freeze=False):
        """Resolve a type's components once at a compilation boundary."""
        def reference(value):
            resolved = TypeResolver.resolve_type(value, namespace, required=False, freeze=freeze)
            return value if resolved is None else resolved

        root = TypeResolver.resolve_type(pc_type, namespace, freeze=freeze)
        return TypeResolver._map_type_components(root, reference)

    @staticmethod
    def _lookup_name(namespace, name, *, registry=True):
        if namespace is not None:
            if name == getattr(namespace, 'type_owner_name', None):
                return namespace.get(name, _MISSING)
            owner = getattr(namespace, 'type_owner_namespace', None)
            if owner is not None and name in owner:
                return owner[name]
            scope = getattr(namespace, 'type_scope', None)
            if scope is not None and name in scope.local_names:
                captured = getattr(namespace, 'type_locals', None)
                if captured is not None and name in captured:
                    return captured[name]
                return scope.lookup(name)
            if name in namespace:
                return namespace[name]
            live = getattr(namespace, 'live_globals', None)
            if live is not None and name in live:
                return live[name]
        from .session import CompileSession
        value = (CompileSession.current().forward_refs.get_defined_type(name)
                 if registry else None)
        if value is None:
            from .builtin_entities import get_builtin_entity
            value = get_builtin_entity(name)
        return _MISSING if value is None else value

    @staticmethod
    def _dependency_tree(expression):
        """Project PythoC labels away for dependency analysis, never execution."""
        return _DependencySyntax().visit(ast.parse(expression, mode='eval'))

    @staticmethod
    def _resolve_type(value, namespace, required, freeze, graph):
        seen = set()
        references = []
        while isinstance(value, (str, TypeReference)):
            reference = value if isinstance(value, TypeReference) else None
            binding = graph.get_binding(reference) if reference is not None else None
            if binding is not None and binding.frozen:
                value = binding.value
                continue
            if isinstance(value, TypeReference):
                references.append(value)
                namespace = value.namespace
                expression = value.expression
            else:
                expression = value
            key = (expression, id(namespace))
            if key in seen:
                raise TypeError(f'Cyclic type alias: {expression}')
            seen.add(key)
            direct = (TypeResolver._lookup_name(namespace, expression)
                      if expression.isidentifier() else _MISSING)
            if direct is not _MISSING:
                value = direct
            else:
                missing = (
                    [expression]
                    if expression.isidentifier() and not keyword.iskeyword(expression)
                    else TypeResolver._missing_expression_names(expression, namespace))
                if missing:
                    graph.record_bindings(references, None, False)
                    if not required:
                        return None
                    raise NameError(f"Type annotation name {missing[0]!r} is not defined")
                if not freeze and graph._preparation is None:
                    if not required:
                        return None
                    raise TypeError('Deferred type expressions require preparation: ' + expression)
                names = TypeResolver.expression_names(expression)
                bindings = TypeResolver._resolution_namespace(namespace, names)
                if key in graph.resolving_expressions:
                    raise TypeError(f'Cyclic type expression: {expression}')
                graph.resolving_expressions.add(key)
                try:
                    evaluation_namespace = TypeResolver._reference_namespace(
                        expression, namespace)
                    evaluation_namespace.update(bindings)
                    value = TypeResolver(user_globals=evaluation_namespace).parse_annotation(
                        expression)
                finally:
                    graph.resolving_expressions.remove(key)
            if not isinstance(value, (str, TypeReference)):
                try:
                    TypeResolver._require_type(value)
                except TypeError as error:
                    raise TypeError(
                        f'Invalid type annotation {expression!r}: {error}') from error
        graph.record_bindings(references, value, freeze)
        return value

    @staticmethod
    def _map_type_components(pc_type, reference):
        """Rebuild changed constructors, leaving named declarations authoritative."""
        memo = {}

        def bind(value):
            if isinstance(value, (str, TypeReference)):
                result = reference(value)
                return result if isinstance(result, (str, TypeReference)) else bind(result)
            if not isinstance(value, type):
                return value
            if id(value) in memo:
                return memo[id(value)]
            memo[id(value)] = value
            mapper = getattr(value, 'map_type_components', None)
            result = mapper(bind) if mapper is not None else value
            memo[id(value)] = result
            return result

        return bind(pc_type)
