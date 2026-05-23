import os
import sys
import atexit
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Tuple
from ..utils.link_utils import file_lock
from .deps import get_dependency_tracker, GroupKey


def _atomic_replace(src: str, dst: str):
    """Atomically replace *dst* with *src* via rename.

    On Windows ``os.rename`` fails when *dst* already exists, so we fall
    back to ``os.replace`` (Python 3.3+).  If the file is still locked by
    another process we retry briefly.
    """
    import time
    for attempt in range(5):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            # Another process may still hold the file open briefly.
            time.sleep(0.05 * (attempt + 1))
    # Last attempt — let it raise if it still fails.
    os.replace(src, dst)


@dataclass
class _CompileIteration:
    """One scheduler-visible slice of group compilation work."""

    items: List[Tuple[Any, Any]]
    symbols: Set[str] = field(default_factory=set)


@dataclass
class _CompileResult:
    """Result produced by compiling a group before artifact publication."""

    compiled_count: int = 0
    compiled_symbols: Set[str] = field(default_factory=set)


@dataclass
class _GroupObjectTaskResult:
    """Worker result for one group-object build task."""

    group_key: Tuple
    group: dict
    status: Optional[str]
    compiled_symbols: Set[str] = field(default_factory=set)


class OutputManager:
    """
    Manages compilation groups and output file generation.
    
    A compilation group represents a set of functions compiled into the same
    .ll/.o/.so files. Functions are grouped by a 4-tuple:
    - (source_file, scope, compile_suffix, effect_suffix)
    
    Supports both 3-tuple (legacy) and 4-tuple group keys for compatibility.
    """
    
    def __init__(self):
        """Initialize output manager with empty group registry."""
        # Key: group_key tuple (3 or 4 elements)
        # Value: dict with compiler, wrappers, file paths, etc.
        self._pending_groups = {}

        # All groups (including completed ones) for compile_to_executable lookup
        self._all_groups = {}

        # Pending compilation callbacks: group_key -> [(callback, func_info), ...]
        # callback signature: (compiler) -> None
        self._pending_compilations = {}

        # Cached compilation callbacks: group_key -> [(callback, func_info), ...]
        # These are compilations that were skipped due to cache hit but may need
        # to be restored if the group is reopened for new functions.
        self._cached_compilations = {}

        # Track if flush has been called (to avoid double compilation)
        self._flushed_groups = set()

        # Protects scheduler-visible mutable state while object build tasks run.
        self._state_lock = threading.RLock()

        # Dynamic scheduling may need multiple tasks for the same group when
        # compilation enqueues more work while an earlier group task is running.
        self._group_object_task_seq = 0

        # Diagnostic strict mode: when enabled, fail if codegen queues new work
        # during object build.  This identifies dependencies/specializations that
        # should eventually move into a build planning phase.
        self._active_build_groups = set()

        # Planning-time group dependencies discovered from import/registration
        # metadata, not from codegen visitors.
        self._planning_group_deps = {}
    
    def _next_group_object_task_id(self, group_key):
        """Return a unique scheduler task id for one group-object attempt."""
        with self._state_lock:
            self._group_object_task_seq += 1
            seq = self._group_object_task_seq
        return f"compile-object:{repr(group_key)}:{seq}"

    def get_or_create_group(self, group_key, compiler, ir_file, obj_file, so_file, 
                           source_file):
        """
        Get existing group or create a new one.
        
        Args:
            group_key: Group key tuple (3 or 4 elements)
            compiler: LLVMCompiler instance for this group
            ir_file: Path to output .ll file
            obj_file: Path to output .o file
            so_file: Path to output .so file
            source_file: Original source file path
        
        Returns:
            dict: Group info with keys: compiler, wrappers, ir_file, obj_file, so_file, source_file
        """
        with self._state_lock:
            # Check if already in _all_groups (including completed groups)
            if group_key in self._all_groups:
                group = self._all_groups[group_key]
                in_flushed = group_key in self._flushed_groups

                # If the group was already flushed, it may still be safe to add more
                # functions as long as the corresponding shared library has NOT been
                # loaded yet. This is important for flows where internal native
                # execution (e.g. bindgen) triggers a flush mid-import, but the user
                # module continues to define more @compile functions afterwards.
                if in_flushed:
                    from ..native_executor import get_multi_so_executor
                    executor = get_multi_so_executor()

                    existing_so_file = group.get('so_file')
                    if existing_so_file and existing_so_file in executor.loaded_libs:
                        source_file_path = group_key[0] if group_key else 'unknown'
                        from ..logger import logger
                        logger.error(
                            f"Cannot define new compiled function after native execution has started. "
                            f"File '{source_file_path}' was already compiled and loaded. "
                            f"Move all @compile decorated functions before any code that triggers execution."
                        )
                        raise RuntimeError(
                            f"Cannot define new @compile function after module '{source_file_path}' "
                            f"has started native execution. All @compile functions must be defined "
                            f"before any compiled function is called."
                        )

                    # Re-open the group for further compilation. We must also ensure
                    # it is considered pending again, otherwise wrappers won't be
                    # added.
                    if group_key in self._flushed_groups:
                        self._flushed_groups.remove(group_key)
                    self._pending_groups[group_key] = group

                    # Restore cached compilations back to pending. These are functions
                    # that were skipped due to cache hit but now need to be recompiled
                    # together with the new functions to create a complete .o file.
                    if group_key in self._cached_compilations:
                        cached = self._cached_compilations.pop(group_key)
                        if group_key not in self._pending_compilations:
                            self._pending_compilations[group_key] = []
                        self._pending_compilations[group_key].extend(cached)

                return group
            
            if group_key not in self._pending_groups:
                group = {
                    'compiler': compiler,
                    'wrappers': [],
                    'all_wrappers': [],
                    'source_file': source_file,
                    'ir_file': ir_file,
                    'obj_file': obj_file,
                    'so_file': so_file,
                }
                self._pending_groups[group_key] = group
                self._all_groups[group_key] = group
            
            return self._pending_groups[group_key]
    
    def add_wrapper_to_group(self, group_key, wrapper):
        """
        Add a compiled function wrapper to its group.
        
        Args:
            group_key: Group identifier
            wrapper: Function wrapper to add
        """
        with self._state_lock:
            group = self._all_groups.get(group_key)
            if group is not None:
                all_wrappers = group.setdefault('all_wrappers', [])
                if wrapper not in all_wrappers:
                    all_wrappers.append(wrapper)
            if group_key in self._pending_groups:
                group = self._pending_groups[group_key]
                if wrapper not in group['wrappers']:
                    group['wrappers'].append(wrapper)

    def get_group_wrappers(self, group_key):
        """Return persistent wrapper inventory for a group."""
        with self._state_lock:
            group = self._all_groups.get(group_key)
            if not group:
                return []
            return list(group.get('all_wrappers') or group.get('wrappers') or [])

    def get_group_effect_specialization(self, group_key, effect_key, wrapper_ids):
        """Return a completed group-level effect specialization if current."""
        with self._state_lock:
            group = self._all_groups.get(group_key)
            if not group:
                return None
            cache = group.get('effect_specialization_cache') or {}
            cached = cache.get(effect_key)
            if not cached or cached.get('wrapper_ids') != wrapper_ids:
                return None
            return cached

    def record_group_effect_specialization(
        self,
        group_key,
        effect_key,
        wrapper_ids,
        specialized_by_name,
    ):
        """Record that one base group has been specialized for an effect."""
        with self._state_lock:
            group = self._all_groups.get(group_key)
            if not group:
                return
            cache = group.setdefault('effect_specialization_cache', {})
            cache[effect_key] = {
                'wrapper_ids': wrapper_ids,
                'specialized_by_name': dict(specialized_by_name),
            }

    def _iter_compiled_values(self, root):
        """Yield compiled wrappers reachable from simple import-time values."""
        seen = set()
        stack = [root]
        while stack:
            value = stack.pop()
            value_id = id(value)
            if value_id in seen:
                continue
            seen.add(value_id)

            if callable(value) and getattr(value, '_is_compiled', False):
                yield value
                continue

            if isinstance(value, dict):
                stack.extend(value.values())
                continue

            if isinstance(value, (list, tuple, set, frozenset)):
                stack.extend(value)
                continue

            value_dict = getattr(value, '__dict__', None)
            if isinstance(value_dict, dict):
                stack.extend(value_dict.values())

    def record_group_planning_dependency(self, group_key, target_group_key):
        """Record a registration-time group dependency for build planning."""
        if not group_key or not target_group_key or group_key == target_group_key:
            return
        # Module globals contain all previously defined functions from the same
        # source file.  Those are not import/factory dependencies and treating
        # them as such would over-specialize the whole module under unrelated
        # effect suffixes.
        if len(group_key) >= 1 and len(target_group_key) >= 1 and group_key[0] == target_group_key[0]:
            return
        with self._state_lock:
            self._planning_group_deps.setdefault(group_key, set()).add(target_group_key)

    def record_group_planning_deps_from_globals(self, group_key, globals_dict):
        """Record compiled-wrapper group deps visible from a function's globals."""
        if not isinstance(globals_dict, dict):
            return
        for value in globals_dict.values():
            for wrapper in self._iter_compiled_values(value):
                binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
                target_group_key = getattr(binding, 'group_key', None)
                self.record_group_planning_dependency(group_key, target_group_key)

    def get_group_planning_dependencies(self, group_key):
        """Return registration-time group dependencies for build planning."""
        with self._state_lock:
            return set(self._planning_group_deps.get(group_key, set()))
    
    def queue_compilation(self, group_key, callback, func_info):
        """
        Queue a function for deferred compilation.
        
        Args:
            group_key: Group identifier
            callback: Callable (compiler) -> None that compiles the function
            func_info: FunctionInfo for forward declaration
        """
        with self._state_lock:
            if os.environ.get('PC_FAIL_ON_BUILD_TIME_QUEUE') and self._active_build_groups:
                func_name = getattr(func_info, 'mangled_name', None) or getattr(func_info, 'name', '<unknown>')
                raise RuntimeError(
                    "queue_compilation called during object build; "
                    f"func={func_name}, group={group_key}, "
                    f"active_build_groups={sorted(map(repr, self._active_build_groups))}"
                )

            if group_key not in self._pending_compilations:
                self._pending_compilations[group_key] = []
            self._pending_compilations[group_key].append((callback, func_info))

            # If this group already exists but was previously flushed, it might not
            # currently be in `_pending_groups`. In that case, pending compilations
            # would never be processed. Ensure the group is marked pending again.
            if group_key in self._all_groups and group_key not in self._pending_groups:
                group = self._all_groups[group_key]

                # If the group's shared library has already been loaded, we must not
                # allow new compilations into it (same rule as in `get_or_create_group`).
                from ..native_executor import get_multi_so_executor
                executor = get_multi_so_executor()
                so_file = group.get('so_file')
                if so_file and so_file in executor.loaded_libs:
                    source_file_path = group_key[0] if group_key else 'unknown'
                    raise RuntimeError(
                        f"Cannot define new @compile function after module '{source_file_path}' "
                        f"has started native execution. All @compile functions must be defined "
                        f"before any compiled function is called."
                    )

                if group_key in self._flushed_groups:
                    self._flushed_groups.remove(group_key)

                self._pending_groups[group_key] = group

                # If we previously cached skipped compilations for this group (cache hit),
                # restore them so we rebuild a complete object file.
                if group_key in self._cached_compilations:
                    cached = self._cached_compilations.pop(group_key)
                    self._pending_compilations[group_key].extend(cached)

            pending_count = len(self._pending_compilations[group_key])

        from ..logger import logger
        logger.debug(f"queue_compilation: {func_info.name}, group_key={group_key}, total_pending={pending_count}")

    def _get_pending_symbols(self, group_key):
        """Return symbol names currently expected to be materialized for a group."""
        with self._state_lock:
            pending = list(self._pending_compilations.get(group_key, []))
        return {
            func_info.mangled_name or func_info.name
            for _, func_info in pending
        }

    def _get_cached_compiled_symbols(self, group_key, group):
        """Load the compiled symbol set for a group's current object file."""
        compiled_symbols = group.get('compiled_symbols')
        if compiled_symbols:
            return set(compiled_symbols)

        obj_file = group.get('obj_file')
        if not obj_file:
            return set()

        dep_tracker = get_dependency_tracker()
        deps = dep_tracker.load_deps(obj_file)
        if not deps or not getattr(deps, 'compiled_symbols', None):
            return set()

        compiled_symbols = set(deps.compiled_symbols)
        group['compiled_symbols'] = compiled_symbols
        return compiled_symbols

    def _cached_object_covers_pending_symbols(self, group_key, group):
        """Check whether cached object metadata covers all currently pending defs."""
        pending_symbols = self._get_pending_symbols(group_key)
        if not pending_symbols:
            return True

        compiled_symbols = self._get_cached_compiled_symbols(group_key, group)
        if not compiled_symbols:
            return False

        missing_symbols = pending_symbols - compiled_symbols
        if missing_symbols:
            from ..logger import logger
            logger.debug(
                f"Cache miss for {group_key}: object missing pending symbols {sorted(missing_symbols)}"
            )
            return False

        # #1/#9: Check AST content hash — detect stale cache when source_file
        # stays the same but the generated AST body changed.
        if group.get('_ast_content_hashes'):
            import hashlib
            combined = '|'.join(sorted(group['_ast_content_hashes']))
            current_hash = hashlib.sha256(
                combined.encode('utf-8')
            ).hexdigest()[:16]

            obj_file = group.get('obj_file')
            if obj_file:
                from .deps import get_dependency_tracker
                dep_tracker = get_dependency_tracker()
                cached_deps = dep_tracker.load_deps(obj_file)
                if cached_deps and cached_deps.ast_content_hash:
                    if cached_deps.ast_content_hash != current_hash:
                        from ..logger import logger
                        logger.debug(
                            f"Cache miss for {group_key}: AST content hash changed "
                            f"({cached_deps.ast_content_hash} -> {current_hash})"
                        )
                        return False

        if not self._cached_dependency_outputs_exist(group):
            return False

        return True

    def _cached_dependency_outputs_exist(self, group):
        """Check whether persisted dependent groups still have materialized outputs."""
        obj_file = group.get('obj_file')
        if not obj_file:
            return True

        from .deps import get_dependency_tracker
        from ..utils.link_utils import get_shared_lib_extension

        dep_tracker = get_dependency_tracker()
        deps = dep_tracker.load_deps(obj_file)
        if not deps:
            return True

        lib_ext = get_shared_lib_extension()
        cwd = os.getcwd()

        for group_dep in deps.group_dependencies:
            target_group = getattr(group_dep, 'target_group', None)
            if target_group is None or not target_group.file:
                continue

            source_file = target_group.file
            if source_file.startswith(cwd + os.sep) or source_file.startswith(cwd + '/'):
                rel_path = os.path.relpath(source_file, cwd)
            else:
                base_name = os.path.splitext(os.path.basename(source_file))[0]
                rel_path = f"external/{base_name}.py"

            build_dir = os.path.join('build', os.path.dirname(rel_path))
            base_name = os.path.splitext(os.path.basename(source_file))[0]
            file_suffix = target_group.get_file_suffix()
            file_base = f"{base_name}.{file_suffix}" if file_suffix else base_name

            dep_obj_file = os.path.join(build_dir, f"{file_base}.o")
            dep_so_file = os.path.join(build_dir, f"{file_base}{lib_ext}")
            if not os.path.exists(dep_obj_file) or not os.path.exists(dep_so_file):
                from ..logger import logger
                logger.debug(
                    f"Cache miss for {group.get('source_file')}: dependent group output missing {dep_obj_file} / {dep_so_file}"
                )
                return False

        return True
    
    def _forward_declare_function(self, compiler, func_info):
        """
        Add forward declaration for a function in the module.
        
        Args:
            compiler: LLVMCompiler instance
            func_info: FunctionInfo with signature information
        """
        from llvmlite import ir
        
        func_name = func_info.mangled_name or func_info.name
        
        # Check if already declared
        try:
            compiler.module.get_global(func_name)
            return  # Already exists
        except KeyError:
            pass
        
        # Build LLVM function type from func_info
        module_context = compiler.module.context
        param_llvm_types = []
        for param_name in func_info.param_names:
            pc_type = func_info.param_type_hints.get(param_name)
            if pc_type is not None and hasattr(pc_type, 'get_llvm_type'):
                param_llvm_types.append(pc_type.get_llvm_type(module_context))
            else:
                # Fallback to i32 if type unknown
                param_llvm_types.append(ir.IntType(32))
        
        if func_info.return_type_hint is not None and hasattr(func_info.return_type_hint, 'get_llvm_type'):
            return_type = func_info.return_type_hint.get_llvm_type(module_context)
        else:
            return_type = ir.VoidType()
        
        # Use LLVMBuilder to declare function with proper ABI handling
        from ..builder import LLVMBuilder
        temp_builder = LLVMBuilder()
        func_wrapper = temp_builder.declare_function(
            compiler.module, func_name,
            param_llvm_types, return_type,
            var_arg=func_info.has_llvm_varargs
        )
    
    def get_ir_text(self, group_key):
        """Return the (optimised) LLVM IR text for *group_key*.

        Intended for tests and tooling that want to inspect the
        emitted IR without depending on the on-disk ``.ll``.  The
        ``.ll`` is a pure debug artefact: when a group's ``.o`` is
        served from cache (cache hit) we deliberately do *not* rewrite
        it, because doing so bumps the ``.o`` mtime and forces a
        relink of the dependent shared library -- on Windows that
        fails with ``EACCES`` whenever the DLL is already mapped into
        the running process.

        This method instead materialises the IR purely in memory by
        replaying the cached compilation callbacks against a fresh
        ``LLVMCompiler``.  ``flush_all()`` and the on-disk ``.o`` are
        left untouched.

        Returns:
            The optimised IR text as a string.

        Raises:
            KeyError: if no group with this key has been registered.
        """
        group = self._all_groups.get(group_key)
        if group is None:
            raise KeyError(
                f"no group registered for key {group_key!r}; "
                f"call the @compile-decorated function first"
            )

        compiler = group.get('compiler')
        if compiler is None:
            raise KeyError(
                f"group {group_key!r} has no compiler attached"
            )

        # If the live compiler module already carries function bodies
        # (cache miss path was taken during a prior flush), we can
        # serve directly from it.
        existing_ir = compiler.get_ir()
        if 'define ' in existing_ir:
            return existing_ir

        # Cache-hit path: callbacks are parked in _cached_compilations.
        # Replay them against a throwaway compiler that shares the
        # same source_file/user_globals so that imports resolve
        # consistently.
        cached = self._cached_compilations.get(group_key, [])
        if not cached:
            raise KeyError(
                f"group {group_key!r} has neither in-memory IR nor "
                f"cached compilation callbacks"
            )

        from ..compiler import LLVMCompiler
        scratch = LLVMCompiler(user_globals=getattr(compiler, 'user_globals', None))

        # Phase 1: forward declare all functions on the scratch
        # compiler so that references between them resolve.
        for _callback, func_info in cached:
            self._forward_declare_function(scratch, func_info)

        # Phase 2: replay each compile callback against the scratch
        # compiler. The callback signature is ``(compiler) -> None``,
        # so passing the throwaway compiler keeps disk state
        # untouched.
        for callback, _func_info in cached:
            callback(scratch)

        if not scratch.verify_module():
            raise RuntimeError(
                f"in-memory IR materialisation failed verification "
                f"for group {group_key!r}"
            )

        from ..config import config
        scratch.optimize_module(optimization_level=int(config.opt_level))
        return scratch.get_ir()

    def _drain_next_compile_iteration(self, group_key, compiled_funcs) -> _CompileIteration:
        """Move one pending slice out of global queues for this group.

        This is intentionally the only step in the group compile loop that
        mutates ``_pending_compilations``.  Later refactors can make this a
        scheduler commit step instead of doing it inside the worker.
        """
        with self._state_lock:
            pending = list(self._pending_compilations.get(group_key, []))
            if not pending:
                return _CompileIteration(items=[])

            # Clear pending to avoid re-processing.  Compiling this slice may enqueue
            # more work for the same group; the outer loop will drain it next.
            del self._pending_compilations[group_key]
            # If queue_compilation re-marked this same group as pending while this
            # task is active, keep that work inside this task instead of creating
            # a duplicate dynamic scheduler task with the same id.
            self._pending_groups.pop(group_key, None)

        new_pending = []
        symbols = set()
        for callback, func_info in pending:
            func_key = func_info.mangled_name or func_info.name
            if func_key not in compiled_funcs:
                new_pending.append((callback, func_info))
                compiled_funcs.add(func_key)
                symbols.add(func_key)

        return _CompileIteration(items=new_pending, symbols=symbols)

    def _build_group_scope(self, iteration: _CompileIteration):
        """Build the in-group symbol scope used for self/mutual recursion."""
        group_scope = {}
        for _callback, func_info in iteration.items:
            if func_info.wrapper is not None:
                group_scope[func_info.name] = func_info.wrapper
        return group_scope

    def _inject_group_scope(self, iteration: _CompileIteration, group_scope):
        """Inject group-local function wrappers into compilation globals."""
        for _callback, func_info in iteration.items:
            if func_info.compilation_globals is not None:
                # Existing entries take precedence for non-function entries.
                for name, wrapper in group_scope.items():
                    if name not in func_info.compilation_globals:
                        func_info.compilation_globals[name] = wrapper

    def _prepare_compile_iteration(self, compiler, iteration: _CompileIteration):
        """Prepare one compile slice before function body lowering."""
        from ..logger import logger

        group_scope = self._build_group_scope(iteration)
        self._inject_group_scope(iteration, group_scope)
        logger.debug(
            f"Injected group scope with {len(group_scope)} functions: {list(group_scope.keys())}"
        )

        # Phase 1: Forward declare all new functions.
        for _callback, func_info in iteration.items:
            self._forward_declare_function(compiler, func_info)

    def _compile_iteration_bodies(self, compiler, iteration: _CompileIteration):
        """Compile one slice of function bodies.

        This is the part we ultimately want to make pure with respect to
        PythoC global state. Today callbacks may still record deps or enqueue
        effect specializations; the surrounding methods isolate that fact.
        """
        for callback, _func_info in iteration.items:
            callback(compiler)

    def _compile_pending_for_group(self, group_key, group) -> _CompileResult:
        """
        Compile all pending functions for a group using two-pass iterations.

        The method is split into explicit phases so the scheduler can later
        move queue mutation and dependency/effect commits out of object emission.
        """
        if not group:
            return _CompileResult()

        compiler = group['compiler']
        from ..logger import logger
        logger.debug(
            f"_compile_pending_for_group: group_key={group_key}, "
            f"pending={len(self._pending_compilations.get(group_key, []))}"
        )

        compiled_funcs = set()
        result = _CompileResult()

        # Loop until no more pending compilations for this group.  Compiling one
        # slice can still enqueue effect/suffix specializations for the next one.
        while True:
            iteration = self._drain_next_compile_iteration(group_key, compiled_funcs)
            if not iteration.items:
                break

            result.compiled_count += len(iteration.items)
            result.compiled_symbols |= iteration.symbols

            self._prepare_compile_iteration(compiler, iteration)
            self._compile_iteration_bodies(compiler, iteration)

        return result
    
    def _group_object_cache_hit(self, group_key, group) -> bool:
        """Return whether the current object artifact covers pending work."""
        from .cache import BuildCache

        source_file = group.get('source_file')
        obj_file = group['obj_file']
        return (
            bool(source_file)
            and BuildCache.check_obj_uptodate(obj_file, source_file)
            and self._cached_object_covers_pending_symbols(group_key, group)
        )

    def _commit_cached_group(self, group_key, group):
        """Commit OutputManager state after a cache hit."""
        from ..config import config

        obj_file = group['obj_file']
        ir_file = group.get('ir_file')
        if config.save_ir and ir_file and not os.path.exists(ir_file):
            from ..logger import logger
            logger.debug(
                f"save_ir=True but {ir_file} is absent on cache hit; "
                f"keeping cached .o untouched. Remove {obj_file} to force "
                f"a rebuild that writes the .ll."
            )

        self._restore_deps_from_cache(group_key, group)
        self._flushed_groups.add(group_key)

        if group_key in self._pending_compilations:
            self._cached_compilations[group_key] = self._pending_compilations.pop(group_key)
        group['wrappers'] = []

        return 'cached'

    def _commit_empty_group(self, group_key, group):
        """Commit OutputManager state when no functions materialized."""
        self._flushed_groups.add(group_key)
        group['wrappers'] = []
        return 'empty'

    def _publish_group_object(self, group_key, group, compile_result: _CompileResult):
        """Publish the compiled LLVM module as IR, object, and deps files."""
        compiler = group['compiler']
        obj_file = group['obj_file']

        existing_symbols = set(group.get('compiled_symbols', set()))
        compiled_symbols = existing_symbols | compile_result.compiled_symbols

        if not compiler.verify_module():
            raise RuntimeError(f"Module verification failed for group {group_key}")

        from ..config import config

        if config.save_unopt_ir:
            unopt_ir_file = group['ir_file'].replace('.ll', '.unopt.ll')
            with open(unopt_ir_file, 'w') as f:
                f.write(str(compiler.module))

        opt_level = int(config.opt_level)
        compiler.optimize_module(optimization_level=opt_level)

        # Write .o atomically so concurrent readers never see a half-written file.
        # The .ll text is a debug artifact, controlled by config.save_ir.
        tmp_obj = obj_file + '.tmp.' + str(os.getpid())
        if config.save_ir:
            compiler.save_ir_to_file(group['ir_file'])
        compiler.compile_to_object(tmp_obj)
        _atomic_replace(tmp_obj, obj_file)

        group['compiled_symbols'] = compiled_symbols
        self._save_group_deps(
            group_key,
            compiler,
            obj_file,
            group,
            compiled_symbols=compiled_symbols,
        )

    def _commit_compiled_group(self, group_key, group):
        """Commit OutputManager state after publishing a fresh object."""
        self._flushed_groups.add(group_key)
        group['wrappers'] = []
        return 'compiled'

    def _build_group_object_task(self, group_key, group):
        """Worker phase for compiling one pending group to object/deps artifacts."""
        with self._state_lock:
            if group_key in self._flushed_groups or group.get('compilation_failed', False):
                return _GroupObjectTaskResult(group_key, group, None)
            self._active_build_groups.add(group_key)

        try:
            obj_file = group['obj_file']
            # File lock covers the entire cache-check -> compile -> write cycle.
            # This remains the cross-process guard; the scheduler handles only
            # in-process task ordering.
            lockfile_path = obj_file + '.lock'

            with file_lock(lockfile_path):
                # Check cache inside the lock so that waiting processes
                # see the .o written by the winner and skip compilation.
                if self._group_object_cache_hit(group_key, group):
                    return _GroupObjectTaskResult(group_key, group, 'cached')

                # Cache miss -- this process is the first to compile this .o.
                try:
                    compile_result = self._compile_pending_for_group(group_key, group)
                except Exception:
                    with self._state_lock:
                        group['compilation_failed'] = True
                    raise

                # If no functions were actually compiled (all deferred),
                # skip writing an empty .o file.
                if compile_result.compiled_count == 0:
                    return _GroupObjectTaskResult(group_key, group, 'empty')

                if not group.get('wrappers'):
                    return _GroupObjectTaskResult(group_key, group, None)

                self._publish_group_object(group_key, group, compile_result)

            return _GroupObjectTaskResult(
                group_key,
                group,
                'compiled',
                compiled_symbols=set(compile_result.compiled_symbols),
            )
        finally:
            with self._state_lock:
                self._active_build_groups.discard(group_key)

    def _commit_group_object_task_result(self, task_result):
        """Scheduler-thread commit phase for a group-object task."""
        result = task_result.value
        if not isinstance(result, _GroupObjectTaskResult):
            return self._plan_pending_group_tasks()

        group_key = result.group_key
        group = result.group
        with self._state_lock:
            if result.status == 'cached':
                self._commit_cached_group(group_key, group)
            elif result.status == 'empty':
                self._commit_empty_group(group_key, group)
            elif result.status == 'compiled':
                if result.compiled_symbols:
                    existing = set(group.get('compiled_symbols', set()))
                    group['compiled_symbols'] = existing | result.compiled_symbols
                self._commit_compiled_group(group_key, group)

        return self._plan_pending_group_tasks()

    def _flush_group_object(self, group_key, group):
        """Compile and commit one group immediately (legacy helper)."""
        task_result = type('_ImmediateTaskResult', (), {})()
        task_result.value = self._build_group_object_task(group_key, group)
        self._commit_group_object_task_result(task_result)
        return task_result.value.status

    def _take_pending_groups(self):
        """Atomically take the current pending group batch for scheduling."""
        with self._state_lock:
            batch = list(self._pending_groups.items())
            self._pending_groups.clear()
        return batch

    def _effect_bindings_for_group(self, group_key):
        """Return captured effect bindings for an effect-specialized group."""
        with self._state_lock:
            pending = list(self._pending_compilations.get(group_key, []))
            group = self._all_groups.get(group_key, {})
            wrappers = list(group.get('all_wrappers') or group.get('wrappers') or [])
        for _callback, func_info in pending:
            binding = getattr(func_info, 'binding_state', None)
            captured = getattr(binding, 'captured_effect_context', None)
            if captured:
                return captured
        for wrapper in wrappers:
            binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
            captured = getattr(binding, 'captured_effect_context', None)
            if captured:
                return captured
        return None

    def _iter_compiled_global_refs(self, wrappers):
        """Yield compiled wrappers referenced by wrapper global namespaces."""
        seen = set()
        for wrapper in wrappers:
            binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
            globals_dict = getattr(binding, 'compilation_globals', None) if binding else None
            if not isinstance(globals_dict, dict):
                continue
            for value in globals_dict.values():
                candidates = [value]
                value_dict = getattr(value, '__dict__', None)
                if isinstance(value_dict, dict):
                    candidates.extend(value_dict.values())
                for candidate in candidates:
                    if not callable(candidate) or not getattr(candidate, '_is_compiled', False):
                        continue
                    if id(candidate) in seen:
                        continue
                    seen.add(id(candidate))
                    yield candidate

    def _pre_materialize_referenced_default_templates(self):
        """Materialize default template wrappers referenced by pending groups."""
        from ..decorators.compile import materialize_specialization, DEFAULT_EFFECT_KEY

        while True:
            with self._state_lock:
                groups = list(self._pending_groups)
            changed = False
            for group_key in groups:
                wrappers = self.get_group_wrappers(group_key)
                candidates = list(self._iter_compiled_global_refs(wrappers))
                for dep_group_key in self.get_group_planning_dependencies(group_key):
                    candidates.extend(self.get_group_wrappers(dep_group_key))
                for candidate in candidates:
                    binding = getattr(candidate, '_binding', getattr(candidate, '_state', None))
                    if not binding or not getattr(binding, 'is_template', False):
                        continue
                    before = len(self._pending_compilations.get(binding.group_key, []))
                    materialize_specialization(candidate, DEFAULT_EFFECT_KEY, {})
                    after = len(self._pending_compilations.get(binding.group_key, []))
                    changed = changed or after > before
            if not changed:
                break

    def _pre_materialize_effect_groups(self):
        """Eagerly materialize whole base groups for pending effect groups."""
        from ..decorators.compile import materialize_group_specialization

        while True:
            with self._state_lock:
                effect_groups = []
                for group_key in self._pending_groups:
                    if len(group_key) != 4 or group_key[3] is None:
                        continue
                    wrappers = self.get_group_wrappers(group_key)
                    if any(
                        getattr(
                            getattr(wrapper, '_binding', getattr(wrapper, '_state', None)),
                            'is_group_specialization',
                            False,
                        )
                        for wrapper in wrappers
                    ):
                        effect_groups.append(group_key)

            expanded = False
            for group_key in effect_groups:
                effect_suffix = group_key[3]
                base_group_key = (group_key[0], group_key[1], group_key[2], None)
                base_wrappers = self.get_group_wrappers(base_group_key)
                if not base_wrappers:
                    continue
                effect_bindings = self._effect_bindings_for_group(group_key)
                if not effect_bindings:
                    continue

                before = sum(len(v) for v in self._pending_compilations.values())
                materialize_group_specialization(
                    base_wrappers[0], effect_suffix, effect_bindings,
                )

                scan_wrappers = base_wrappers + self.get_group_wrappers(group_key)
                for candidate in self._iter_compiled_global_refs(scan_wrappers):
                    binding = getattr(candidate, '_binding', getattr(candidate, '_state', None))
                    if binding is None or getattr(candidate, '_pc_effect_impl', False):
                        continue
                    if binding.effect_suffix == effect_suffix and binding.group_key and len(binding.group_key) == 4:
                        candidate_base_key = (
                            binding.group_key[0], binding.group_key[1], binding.group_key[2], None
                        )
                        candidate_base_wrappers = self.get_group_wrappers(candidate_base_key)
                        if candidate_base_wrappers:
                            materialize_group_specialization(
                                candidate_base_wrappers[0], effect_suffix, effect_bindings,
                            )
                    elif binding.effect_suffix is None:
                        materialize_group_specialization(
                            candidate, effect_suffix, effect_bindings,
                        )

                planning_deps = (
                    self.get_group_planning_dependencies(base_group_key)
                    | self.get_group_planning_dependencies(group_key)
                )
                for dep_group_key in planning_deps:
                    if len(dep_group_key) != 4:
                        continue
                    dep_base_key = (
                        dep_group_key[0], dep_group_key[1], dep_group_key[2], None
                    )
                    dep_base_wrappers = self.get_group_wrappers(dep_base_key)
                    if dep_base_wrappers:
                        materialize_group_specialization(
                            dep_base_wrappers[0], effect_suffix, effect_bindings,
                        )

                after = sum(len(v) for v in self._pending_compilations.values())
                expanded = expanded or after > before

            if not expanded:
                break

    def _group_needs_effect_context_resource(self, group_key) -> bool:
        """Return whether compiling this group touches the global effect context."""
        with self._state_lock:
            pending = list(self._pending_compilations.get(group_key, []))
        for _callback, func_info in pending:
            binding = getattr(func_info, 'binding_state', None)
            if binding is None:
                continue
            if getattr(binding, 'effect_suffix', None):
                return True
            captured = getattr(binding, 'captured_effect_context', None)
            if captured:
                return True
        return False

    def _plan_pending_group_tasks(self):
        """Create scheduler tasks for groups queued during compilation."""
        batch = self._take_pending_groups()
        if not batch:
            return []
        from .planner import plan_group_object_tasks
        return plan_group_object_tasks(self, batch)

    def _requeue_unfinished_groups(self):
        """Restore unfinished groups after scheduler failure."""
        with self._state_lock:
            for group_key, group in self._all_groups.items():
                if (
                    group_key not in self._flushed_groups
                    and not group.get('compilation_failed', False)
                ):
                    self._pending_groups.setdefault(group_key, group)

    def flush_all(self):
        """
        Flush all pending output files to disk.
        
        This should be called before native execution to ensure
        all .ll and .o files have been written.
        
        Uses two-pass compilation to support mutual recursion:
        1. Forward declare all functions in each group
        2. Compile all function bodies
        3. Write .ll and .o files
        
        Cache check is done here, at flush time:
        - If .o is up-to-date (newer than source), skip compilation
        - Otherwise, compile and regenerate .o
        """
        from ..logger import logger
        # Check if any group has already loaded its library
        from ..native_executor import get_multi_so_executor
        executor = get_multi_so_executor()
        
        with self._state_lock:
            pending_groups = list(self._pending_groups.items())
            pending_compilations = [
                (k, len(v)) for k, v in self._pending_compilations.items()
            ]

        for group_key, group in pending_groups:
            so_file = group.get('so_file')
            # Check if THIS SPECIFIC so_file is already loaded
            # (not just any library from the same source file)
            if so_file and so_file in executor.loaded_libs:
                # Check if this group has new pending compilations
                with self._state_lock:
                    has_pending = bool(self._pending_compilations.get(group_key))
                if has_pending:
                    source_file = group.get('source_file', so_file)
                    raise RuntimeError(
                        f"Cannot compile new functions in '{source_file}' after native execution has started. "
                        f"All @compile decorators must be executed before calling any compiled functions."
                    )
        
        # Process groups through the scheduler.  New groups may be added during
        # compilation; task commit callbacks feed them back into the same run.
        logger.debug(f"flush_all: _pending_groups={[k for k, _ in pending_groups]}")
        logger.debug(f"flush_all: _pending_compilations={pending_compilations}")
        self._pre_materialize_effect_groups()
        self._pre_materialize_referenced_default_templates()
        from .scheduler import BuildScheduler, BuildSchedulerError
        object_workers = int(os.environ.get('PC_OBJECT_BUILD_WORKERS', '1') or '1')
        tasks = self._plan_pending_group_tasks()
        if tasks:
            try:
                BuildScheduler(max_workers=object_workers).run(tasks)
            except BuildSchedulerError as exc:
                self._requeue_unfinished_groups()
                if len(exc.failures) == 1:
                    raise next(iter(exc.failures.values())) from None
                raise
            except Exception:
                self._requeue_unfinished_groups()
                raise

        # Don't clear pending groups - they serve as metadata cache for subsequent runs
    
    def _restore_deps_from_cache(self, group_key, group):
        """
        Restore dependency information from .deps file on cache hit.
        
        This ensures that link libraries/objects are properly registered
        even when we skip compilation.

        
        Args:
            group_key: Group key tuple
            group: Group info dict
        """
        obj_file = group['obj_file']
        compiler = group['compiler']
        
        dep_tracker = get_dependency_tracker()
        deps = dep_tracker.load_deps(obj_file)
        if deps:
            group['compiled_symbols'] = set(getattr(deps, 'compiled_symbols', []))

            # Restore imported_user_functions from group-level deps
            if not hasattr(compiler, 'imported_user_functions'):
                compiler.imported_user_functions = {}
            for group_dep in deps.group_dependencies:
                if group_dep.target_group:
                    # For group-level deps, we can't restore the exact function name
                    # but we can restore the file dependency
                    target_file = group_dep.target_group.file
                    # Use a placeholder name based on the group key
                    placeholder_name = f"group_{hash(group_dep.target_group.to_tuple()) & 0xFFFFFF:06x}"
                    compiler.imported_user_functions[placeholder_name] = target_file
            
            # Restore link libraries and objects to registry
            from ..registry import get_unified_registry
            registry = get_unified_registry()
            for lib in deps.link_libraries:
                registry.add_link_library(lib)
            for obj in deps.link_objects:
                registry.add_link_object(obj)


    
    def _save_group_deps(self, group_key, compiler, obj_file, group=None, compiled_symbols=None):
        """
        Save dependency information for a compiled group.

        Persists link libraries/objects to `.deps` file. Group dependencies
        are recorded at call time in `compile.py` and `type_converter.py`.


        Args:
            group_key: Group key tuple
            compiler: LLVMCompiler with compilation results
            obj_file: Path to .o file
            group: Optional group info dict (for accessing wrappers)
            compiled_symbols: Optional complete symbol set for the current .o
        """
        from .deps import get_dependency_tracker

        dep_tracker = get_dependency_tracker()
        with dep_tracker._lock:
            # Get or create deps for this group
            group_deps = dep_tracker.get_or_create_group_deps(group_key)
            
            # Set source mtime
            if group is None:
                group = self._all_groups.get(group_key)
            if group and group.get('source_file'):
                source_file = group['source_file']
                if os.path.exists(source_file):
                    group_deps.source_mtime = os.path.getmtime(source_file)
            
            # Link libraries and objects are recorded incrementally during
            # compilation (via callable_lowering.record_extern_dependency and
            # cimport).  We do NOT dump the global registry here — that would
            # pollute every group's .deps with unrelated libraries from other
            # groups compiled in the same process.
            
            if compiled_symbols is not None:
                group_deps.compiled_symbols = sorted(compiled_symbols)

            # Persist AST content hash for meta-generated code invalidation.
            if group and group.get('_ast_content_hashes'):
                import hashlib
                combined = '|'.join(sorted(group['_ast_content_hashes']))
                group_deps.ast_content_hash = hashlib.sha256(
                    combined.encode('utf-8')
                ).hexdigest()[:16]

            # Propagate transitive effects from dependent groups.
            # If this group calls functions in groups that use effects (e.g., c_ast uses
            # effect.mem), those effects should be reflected in this group's effects_used
            # so that the effect specialization check in type_converter.py works correctly
            # for callers that reference this group.
            for dep in group_deps.group_dependencies:
                dep_group_deps = dep_tracker.get_deps(dep.target_group.to_tuple())
                if dep_group_deps and dep_group_deps.effects_used:
                    group_deps.effects_used |= dep_group_deps.effects_used
            
            # Save to file
            dep_tracker.save_deps(group_key, obj_file)
    
    def get_group(self, group_key):
        """
        Get group info by key.
        
        Args:
            group_key: Group identifier
        
        Returns:
            dict or None: Group info if exists
        """
        return self._all_groups.get(group_key)
    
    def get_all_groups(self):
        """
        Get all groups (including completed ones).
        
        Returns:
            dict: All groups mapping group_key -> group info
        """
        return self._all_groups
    
    def clear_all(self):
        """Clear all pending groups (for testing/reset)."""
        self._pending_groups.clear()
        self._all_groups.clear()
        self._pending_compilations.clear()
        self._cached_compilations.clear()
        self._flushed_groups.clear()
        self._group_object_task_seq = 0
        self._active_build_groups.clear()
        self._planning_group_deps.clear()
    
    def clear_failed_group(self, group_key):
        """
        Clear a failed compilation group to allow retry or cleanup.
        
        Args:
            group_key: Group identifier to clear
        """
        if group_key in self._pending_groups:
            del self._pending_groups[group_key]
        if group_key in self._all_groups:
            del self._all_groups[group_key]
        if group_key in self._pending_compilations:
            del self._pending_compilations[group_key]
        if group_key in self._flushed_groups:
            self._flushed_groups.remove(group_key)


# Global singleton instance
_output_manager = OutputManager()

# Track if atexit handler is registered
_atexit_registered = False


def _atexit_flush():
    """Flush all pending compilations at interpreter shutdown.

    NOTE: If compilation fails here, `logger.error()` may call `sys.exit(1)`.
    A `SystemExit` escaping an atexit callback is printed as
    "Exception ignored in atexit callback" with a traceback, which is noisy.

    We therefore intercept `SystemExit`, flush stdio, and terminate with the
    same exit code via `os._exit()` to keep output clean.
    """
    try:
        _output_manager.flush_all()
    except SystemExit as e:
        # Preserve exit status without triggering atexit traceback printing.
        try:
            sys.stderr.flush()
            sys.stdout.flush()
        finally:
            code = e.code
            if isinstance(code, int):
                os._exit(code)
            os._exit(1)
    except Exception:
        # Silently ignore errors during atexit
        # (e.g., if program is terminating due to another error)
        pass


def _ensure_atexit_registered():
    """Register atexit handler if not already registered."""
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_atexit_flush)
        _atexit_registered = True


def get_output_manager():
    """Get the global OutputManager singleton."""
    _ensure_atexit_registered()
    return _output_manager


def flush_all_pending_outputs():
    """
    Convenience function to flush all pending outputs.
    
    This is the main entry point used by the runtime.
    """
    _output_manager.flush_all()


def clear_failed_group(group_key):
    """
    Clear a failed compilation group.
    
    This is useful for error testing where a group fails to compile
    and we want to clean up before the next test.
    
    Args:
        group_key: (source_file, scope, suffix) tuple
    """
    _output_manager.clear_failed_group(group_key)
