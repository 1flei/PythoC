"""Build task planning helpers.

This module centralizes construction of scheduler-visible build tasks.  The
scheduler owns ordering/parallelism; callers provide domain operations and commit
callbacks.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

from .scheduler import BuildTask


def _deps_file_for_obj(obj_file: Optional[str]) -> Optional[str]:
    if not obj_file:
        return None
    return obj_file[:-2] + '.deps' if obj_file.endswith('.o') else obj_file + '.deps'


def plan_group_object_tasks(output_manager, groups: Sequence[Tuple[Tuple, dict]]) -> List[BuildTask]:
    """Plan tasks that compile pending groups into object/deps artifacts."""
    tasks: List[BuildTask] = []
    for group_key, group in groups:
        obj_file = group.get('obj_file')
        task_id = output_manager._next_group_object_task_id(group_key)
        resources = [
            f"group:{repr(group_key)}",
            f"compiler:{id(group.get('compiler'))}",
        ]
        if output_manager._group_needs_effect_context_resource(group_key):
            resources.append("effect-context")
        tasks.append(BuildTask(
            id=task_id,
            kind='compile_group_to_object',
            inputs=(group.get('source_file'),),
            outputs=(
                group.get('ir_file'),
                obj_file,
                _deps_file_for_obj(obj_file),
            ),
            resources=tuple(resource for resource in resources if resource),
            run=lambda group_key=group_key, group=group: output_manager._build_group_object_task(
                group_key,
                group,
            ),
            on_success=output_manager._commit_group_object_task_result,
        ))
    return tasks


def plan_link_shared_tasks(link_jobs, link_fn) -> List[BuildTask]:
    """Plan shared-library link tasks.

    Each job is either ``(obj, so, libs)`` or ``(obj, so, libs, deps)`` where
    ``deps`` is a sequence of scheduler task ids that must complete first.
    """
    tasks: List[BuildTask] = []
    for job in link_jobs:
        if len(job) == 3:
            obj, so, libs = job
            deps = ()
        else:
            obj, so, libs, deps = job
        tasks.append(BuildTask(
            id=f"link-shared:{os.path.abspath(so)}",
            kind='link_shared_library',
            deps=tuple(deps or ()),
            inputs=(obj,) + tuple(libs or []),
            outputs=(so,),
            run=lambda obj=obj, so=so, libs=libs: link_fn(obj, so, libs),
        ))
    return tasks


def plan_stub_import_library_tasks(stub_jobs: Iterable[Tuple[str, str, str]], generate_fn) -> List[BuildTask]:
    """Plan Windows stub import-library generation tasks."""
    tasks: List[BuildTask] = []
    for obj, dll, imp in stub_jobs:
        tasks.append(BuildTask(
            id=f"stub-implib:{os.path.abspath(imp)}",
            kind='generate_stub_import_library',
            inputs=(obj,),
            outputs=(imp, os.path.splitext(imp)[0] + '.exports.def'),
            run=lambda obj=obj, dll=dll, imp=imp: generate_fn(obj, dll, imp),
        ))
    return tasks
