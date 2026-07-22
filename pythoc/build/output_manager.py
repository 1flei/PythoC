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


@dataclass
class _CodegenTaskResult:
    """Worker result for one codegen (Phase 1) task."""

    group_key: Tuple
    group: dict
    status: Optional[str]
    compile_result: Optional["_CompileResult"] = None


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

        # Effect specialization graph — replaces BFS-based planning deps.
        from ..effect_graph import EffectGraph
        self._effect_graph = EffectGraph()
    
    def _next_group_object_task_id(self, group_key):
        """Return a unique scheduler task id for one group-object attempt."""
        with self._state_lock:
            self._group_object_task_seq += 1
            seq = self._group_object_task_seq
        return f"compile-object:{repr(group_key)}:{seq}"

    def _next_compile_task_id(self, group_key):
        """Return a unique scheduler task id for one Phase 2 compile-object attempt."""
        with self._state_lock:
            self._group_object_task_seq += 1
            seq = self._group_object_task_seq
        return f"publish-object:{repr(group_key)}:{seq}"

    def _reject_if_flushed(self, group_key, context_msg=""):
        """Reject attempts to add to a flushed group — groups are immutable after flush.

        This replaces the old group-reopen mechanism. If the group's .so has been
        dlopen'd, the error is more specific (native execution started). Otherwise
        it's a code-ordering issue (flush triggered before all @compile defined).
        """
        if group_key not in self._all_groups:
            return  # Group doesn't exist yet — nothing to reject
        if group_key in self._pending_groups:
            return  # Still pending — fine to add

        # Group was flushed. Reject — no reopen in any case.
        from ..native_executor import get_multi_so_executor
        executor = get_multi_so_executor()
        group = self._all_groups[group_key]
        so_file = group.get('so_file')
        source_file_path = group_key[0] if group_key else 'unknown'

        if so_file and so_file in executor.loaded_libs:
            raise RuntimeError(
                f"Cannot define new @compile function after module '{source_file_path}' "
                f"has started native execution. All @compile functions must be defined "
                f"before any compiled function is called."
            )
        raise RuntimeError(
            f"Cannot add new @compile function to module '{source_file_path}' "
            f"after flush has already occurred. This typically happens when a "
            f"module-level statement triggers native execution (e.g. bindgen, "
            f"calling a compiled function) before all @compile functions are "
            f"defined. Move all @compile decorators before any code that "
            f"triggers execution."
        )

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
            # Reject if group already flushed — groups are immutable after flush.
            self._reject_if_flushed(group_key)

            if group_key in self._all_groups:
                return self._all_groups[group_key]
            
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

            # Reject if group already flushed — groups are immutable after flush.
            self._reject_if_flushed(group_key)

            if group_key not in self._pending_compilations:
                self._pending_compilations[group_key] = []
            self._pending_compilations[group_key].append((callback, func_info))

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

        # Changing debug_info settings must invalidate cached objects, otherwise
        # a previously-built object without DWARF would be reused after enabling
        # PC_DEBUG_INFO (and vice versa).
        obj_file = group.get('obj_file')
        if obj_file:
            from .deps import get_dependency_tracker
            from ..config import config
            dep_tracker = get_dependency_tracker()
            cached_deps = dep_tracker.load_deps(obj_file)
            if cached_deps and cached_deps.debug_info != bool(config.debug_info):
                from ..logger import logger
                logger.debug(
                    f"Cache miss for {group_key}: debug_info changed "
                    f"({cached_deps.debug_info} -> {config.debug_info})"
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
            # source_embed deps are validated by mtime, not output existence
            # (an inlined-only module may emit no object of its own).
            if getattr(group_dep, 'dependency_type', None) == "source_embed":
                continue
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

    def _record_type_layout_deps_for_iteration(self, group_key, iteration, seen_globals_ids):
        """Record type-layout (source_embed) deps for one compile iteration.

        Runs once per unique ``compilation_globals`` dict per group
        (deduplicated via ``seen_globals_ids``) so the scan cost is
        O(unique module namespaces), not O(functions).
        """
        from .deps import get_dependency_tracker
        dep_tracker = get_dependency_tracker()
        for _callback, func_info in iteration.items:
            wrapper = getattr(func_info, 'wrapper', None)
            binding = getattr(wrapper, '_binding', getattr(wrapper, '_state', None))
            globals_dict = getattr(binding, 'compilation_globals', None)
            if not isinstance(globals_dict, dict):
                continue
            gid = id(globals_dict)
            if gid in seen_globals_ids:
                continue
            seen_globals_ids.add(gid)
            dep_tracker.record_type_layout_deps_from_globals(group_key, globals_dict)

    def _compile_pending_for_group(self, group_key, group) -> _CompileResult:
        """
        Compile all pending functions for a group using two-pass iterations.

        The method is split into explicit phases so the scheduler can later
        move queue mutation and dependency/effect commits out of object emission.

        Codegen of one slice can enqueue effect/suffix specializations for
        the next slice of this same group; the drain loop re-checks
        ``_pending_compilations`` on every iteration and picks those up.
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
        seen_globals_ids = set()
        group_effect_deps = set()

        # Loop until no more pending compilations for this group.  Compiling one
        # slice can still enqueue effect/suffix specializations for the next one.
        while True:
            iteration = self._drain_next_compile_iteration(group_key, compiled_funcs)
            if not iteration.items:
                break

            self._record_type_layout_deps_for_iteration(
                group_key, iteration, seen_globals_ids,
            )

            _it_count = len(iteration.items)
            result.compiled_count += _it_count
            result.compiled_symbols |= iteration.symbols

            self._prepare_compile_iteration(compiler, iteration)
            self._compile_iteration_bodies(compiler, iteration)

            for _callback, func_info in iteration.items:
                effect_deps = getattr(func_info, 'effect_dependencies', None)
                if effect_deps:
                    group_effect_deps |= effect_deps

        # Declare the group's effect usage on its EffectGraph node exactly
        # once, now that every function of the group has been compiled.
        # Before this point has_effects_declared is False and callers stay
        # conservative; after it the answer is definitive for the whole group.
        if result.compiled_count:
            from ..effect_graph import node_id_from_group_key
            self._effect_graph.set_node_effects(
                node_id_from_group_key(group_key), group_effect_deps,
            )

        return result
    
    def _group_object_cache_hit(self, group_key, group) -> bool:
        """Return whether the current object artifact covers pending work."""
        from .cache import BuildCache

        source_file = group.get('source_file')
        obj_file = group['obj_file']
        return (
            bool(source_file)
            and BuildCache.check_obj_uptodate(obj_file, source_file)
            and self._cached_source_embed_deps_uptodate(group)
            and self._cached_object_covers_pending_symbols(group_key, group)
        )

    def _cached_source_embed_deps_uptodate(self, group) -> bool:
        """Check cached object is newer than every source-embed dependency.

        A source-embed dependency means this group's object embeds, by value,
        code or a type layout owned by another source file (e.g. a cross-module
        inlined yield generator and the structs it allocates). The cache only
        tracks each group's own mtime, so without this check a layout change in
        the dependency would be served from a stale .o whose baked-in frame size
        no longer matches -> silent memory corruption. We conservatively treat
        the cache as stale whenever a dependency source is newer than the .o.
        """
        obj_file = group.get('obj_file')
        if not obj_file:
            return True

        from .deps import get_dependency_tracker
        from .cache import BuildCache

        deps = get_dependency_tracker().load_deps(obj_file)
        if not deps:
            return True

        for group_dep in deps.group_dependencies:
            if getattr(group_dep, 'dependency_type', None) != "source_embed":
                continue
            target_group = getattr(group_dep, 'target_group', None)
            if target_group is None or not target_group.file:
                continue
            if not BuildCache.check_obj_uptodate(obj_file, target_group.file):
                from ..logger import logger
                logger.debug(
                    f"Cache miss for {group.get('source_file')}: source-embed "
                    f"dependency {target_group.file} is newer than {obj_file}"
                )
                return False

        return True

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

    # ------------------------------------------------------------------
    # Phase-split build (Scheme A): codegen (serial) + compile (parallel)
    # ------------------------------------------------------------------

    def _codegen_group_task(self, group_key, group):
        """Phase 1 worker: codegen (IR generation) without .o publication.

        Codegen is serialized via implicit DAG chaining (see
        _plan_pending_codegen_tasks).  No resource lock needed.
        Writes no artifacts; the cache check runs under the file lock so a
        concurrent process publishing the same .o cannot be misread.
        """
        with self._state_lock:
            if group_key in self._flushed_groups or group.get('compilation_failed', False):
                return _CodegenTaskResult(group_key, group, None)
            self._active_build_groups.add(group_key)

        try:
            # Cache check under the file lock: another process may be
            # mid-publish of the same .o/.deps, and only the .o rename is
            # atomic.  Codegen is serialized, so the lock never contends
            # in-process.
            lockfile_path = group['obj_file'] + '.lock'
            with file_lock(lockfile_path):
                if self._group_object_cache_hit(group_key, group):
                    return _CodegenTaskResult(group_key, group, 'cached')

            try:
                compile_result = self._compile_pending_for_group(group_key, group)
            except Exception:
                with self._state_lock:
                    group['compilation_failed'] = True
                raise

            if compile_result.compiled_count == 0:
                return _CodegenTaskResult(group_key, group, 'empty')

            if not group.get('wrappers'):
                return _CodegenTaskResult(group_key, group, None)

            return _CodegenTaskResult(group_key, group, 'compiled', compile_result)
        finally:
            with self._state_lock:
                self._active_build_groups.discard(group_key)

    def _codegen_on_success(self, task_result):
        """Phase 1 on_success: dispatch compile task + discover new codegen tasks."""
        result = task_result.value
        if not isinstance(result, _CodegenTaskResult):
            return self._plan_pending_codegen_tasks()

        new_tasks = []
        group_key = result.group_key
        group = result.group

        if result.status == 'cached':
            with self._state_lock:
                self._commit_cached_group(group_key, group)
        elif result.status == 'empty':
            with self._state_lock:
                self._commit_empty_group(group_key, group)
        elif result.status == 'compiled':
            from .planner import plan_compile_object_task
            new_tasks.append(plan_compile_object_task(
                self, group_key, group, result.compile_result,
            ))
        # status == None: already flushed or failed — nothing to dispatch

        new_tasks.extend(self._plan_pending_codegen_tasks())
        return new_tasks

    def _compile_object_task(self, group_key, group, compile_result):
        """Phase 2 worker: publish .o file from codegen-produced IR.

        Acquires the file lock and re-checks cache inside it.  Multiple
        compile tasks for different groups run in parallel.
        """
        obj_file = group['obj_file']
        lockfile_path = obj_file + '.lock'

        with file_lock(lockfile_path):
            # Re-check cache inside the lock — another process may have
            # published the .o while we were waiting.
            if self._group_object_cache_hit(group_key, group):
                return _GroupObjectTaskResult(group_key, group, 'cached')

            self._publish_group_object(group_key, group, compile_result)

        return _GroupObjectTaskResult(
            group_key, group,
            'compiled',
            compiled_symbols=set(compile_result.compiled_symbols),
        )

    def _compile_on_success(self, task_result):
        """Phase 2 on_success: commit state.

        Returns no new tasks on purpose: codegen tasks are chained only from
        ``_codegen_on_success``.  If compile completions also planned codegen
        tasks, two codegen tasks could run concurrently and break the
        serialization that Phase 1 relies on.
        """
        result = task_result.value
        if not isinstance(result, _GroupObjectTaskResult):
            return []

        group_key = result.group_key
        group = result.group

        with self._state_lock:
            if result.status == 'cached':
                self._commit_cached_group(group_key, group)
            elif result.status == 'compiled':
                if result.compiled_symbols:
                    existing = set(group.get('compiled_symbols', set()))
                    group['compiled_symbols'] = existing | result.compiled_symbols
                self._commit_compiled_group(group_key, group)

        return []

    def _plan_pending_codegen_tasks(self):
        """Create **one** codegen task for the next pending group.

        Codegen is serialized via implicit DAG chaining: each codegen task's
        ``on_success`` calls this method again to create the next one.  Only
        one codegen task exists at a time, so no resource lock is needed.
        Compile tasks (Phase 2) run in parallel on other workers.
        """
        batch = self._take_pending_groups()
        if not batch:
            return []
        # Take only the first pending group — chaining handles the rest.
        first = batch[0]
        # Put remaining back so they're available for the next chain link.
        if len(batch) > 1:
            with self._state_lock:
                for gk, g in batch[1:]:
                    self._pending_groups[gk] = g
        from .planner import plan_codegen_tasks
        return plan_codegen_tasks(self, [first])

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

    def _node_group_index(self):
        """Snapshot mapping NodeID -> [group_key] for graph neighbor lookups."""
        from ..effect_graph import node_id_from_group_key
        with self._state_lock:
            index = {}
            for gk in self._all_groups:
                index.setdefault(node_id_from_group_key(gk), []).append(gk)
        return index

    def _pre_materialize_referenced_default_templates(self):
        """Materialize default template wrappers referenced by pending groups."""
        from ..decorators.compile import materialize_specialization, DEFAULT_EFFECT_KEY
        from ..effect_graph import node_id_from_group_key

        while True:
            with self._state_lock:
                groups = list(self._pending_groups)
            node_index = self._node_group_index()
            changed = False
            for group_key in groups:
                # Candidates: this group's own wrappers plus wrappers of
                # groups the EffectGraph says this group depends on.
                candidates = list(self.get_group_wrappers(group_key))
                caller_node = node_id_from_group_key(group_key)
                for neighbor_node in self._effect_graph.get_neighbors(caller_node):
                    for other_gk in node_index.get(neighbor_node, ()):
                        candidates.extend(self.get_group_wrappers(other_gk))
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
        from ..effect_graph import node_id_from_group_key

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
            node_index = self._node_group_index()

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

                # Propagate the specialization to groups the EffectGraph says
                # this group depends on.  Transitively dependent groups are
                # reached by the outer fixpoint: once a neighbor's effect
                # group exists, it is processed in a later iteration.
                planning_nodes = (
                    self._effect_graph.get_neighbors(node_id_from_group_key(base_group_key))
                    | self._effect_graph.get_neighbors(node_id_from_group_key(group_key))
                )
                for neighbor_node in planning_nodes:
                    for dep_group_key in node_index.get(neighbor_node, ()):
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
        # Phase-split build (Scheme A): codegen is serialized via implicit
        # DAG chaining (one at a time); compile tasks run in parallel on
        # remaining workers.  object_workers>=2 enables pipelining.
        tasks = self._plan_pending_codegen_tasks()
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

            # Record whether debug info was enabled for this object, so that
            # toggling PC_DEBUG_INFO invalidates the cache.
            from ..config import config
            group_deps.debug_info = bool(config.debug_info)

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
        from ..effect_graph import EffectGraph
        self._effect_graph = EffectGraph()
    
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
