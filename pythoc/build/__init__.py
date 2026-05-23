from .cache import BuildCache
from .output_manager import (
    OutputManager, 
    get_output_manager, 
    flush_all_pending_outputs,
    _ensure_atexit_registered,
)
from .deps import (
    GroupKey,
    GroupDeps,
    DependencyTracker,
    get_dependency_tracker,
)
from .scheduler import (
    BuildScheduler,
    BuildSchedulerError,
    BuildTask,
    TaskResult,
)
from .planner import (
    plan_group_object_tasks,
    plan_link_shared_tasks,
    plan_stub_import_library_tasks,
)

__all__ = [
    'BuildCache',
    'OutputManager',
    'get_output_manager',
    'flush_all_pending_outputs',
    '_ensure_atexit_registered',
    # Dependency tracking
    'GroupKey',
    'GroupDeps',
    'DependencyTracker',
    'get_dependency_tracker',
    'BuildScheduler',
    'BuildSchedulerError',
    'BuildTask',
    'TaskResult',
    'plan_group_object_tasks',
    'plan_link_shared_tasks',
    'plan_stub_import_library_tasks',
]
