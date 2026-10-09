"""Lower artifact plans to the existing build scheduler."""

from __future__ import annotations

import os
from typing import List, Optional

from ..build.scheduler import BuildScheduler, BuildTask
from .model import (
    ArtifactKind,
    ArtifactPhase,
    ArtifactPlan,
    NativeArtifact,
)


def plan_artifact_tasks(plan: ArtifactPlan) -> List[BuildTask]:
    pre_steps = [
        step for step in plan.steps
        if step.phase is ArtifactPhase.PRE_LINK
    ]
    post_steps = [
        step for step in plan.steps
        if step.phase is ArtifactPhase.POST_LINK
    ]
    tasks = [
        BuildTask(
            id=step.id,
            kind=step.kind,
            inputs=step.inputs,
            outputs=step.outputs,
            resources=step.resources,
            run=step.run,
            cache_check=step.cache_check,
        )
        for step in pre_steps
    ]

    link_task = _link_task(plan, tuple(step.id for step in pre_steps))
    tasks.append(link_task)
    tasks.extend(
        BuildTask(
            id=step.id,
            kind=step.kind,
            deps=(link_task.id,),
            inputs=(plan.output_path,) + step.inputs,
            outputs=step.outputs,
            resources=step.resources,
            run=step.run,
            cache_check=step.cache_check,
        )
        for step in post_steps
    )
    return tasks


def build_artifact(
    plan: ArtifactPlan,
    *,
    scheduler: Optional[BuildScheduler] = None,
) -> NativeArtifact:
    tasks = plan_artifact_tasks(plan)
    runner = scheduler or BuildScheduler()
    runner.run(tasks)
    if not os.path.isfile(plan.output_path):
        raise RuntimeError(
            "artifact build did not produce {!r}".format(plan.output_path)
        )
    return NativeArtifact.from_plan(plan)


def _link_task(plan: ArtifactPlan, dependencies) -> BuildTask:
    output = os.path.abspath(plan.output_path)
    libraries = _ordered_unique(
        tuple(plan.link.link_libraries) + tuple(plan.link_libraries)
    )
    inputs = (
        plan.object_files
        + tuple(plan.link_objects)
        + libraries
    )
    common = {
        "deps": dependencies,
        "inputs": inputs,
        "outputs": (plan.output_path,),
    }

    if plan.kind is ArtifactKind.STATIC_LIBRARY:
        from ..utils.link_utils import archive_files

        return BuildTask(
            id="archive-static:{}".format(output),
            kind="archive_static_library",
            run=lambda: archive_files(
                list(plan.object_files),
                plan.output_path,
            ),
            **common,
        )

    if plan.kind is ArtifactKind.EXECUTABLE:
        from ..utils.link_utils import try_link_with_linkers

        return BuildTask(
            id="link-executable:{}".format(output),
            kind="link_executable",
            run=lambda: try_link_with_linkers(
                list(plan.object_files),
                plan.output_path,
                shared=False,
                link_objects=(
                    list(plan.link_objects)
                    if plan.link_objects
                    else None
                ),
                link_libraries=list(libraries),
                extra_flags=list(plan.extra_link_flags),
            ),
            **common,
        )

    if plan.kind in (
        ArtifactKind.SHARED_LIBRARY,
        ArtifactKind.PYTHON_EXTENSION,
    ):
        from ..utils.link_utils import link_files

        return BuildTask(
            id="link-shared:{}".format(output),
            kind="link_shared_library",
            run=lambda: link_files(
                list(plan.object_files),
                plan.output_path,
                shared=True,
                link_objects=list(plan.link_objects),
                link_libraries=list(libraries),
                extra_flags=list(plan.extra_link_flags),
            ),
            **common,
        )

    raise ValueError("unsupported artifact kind {!r}".format(plan.kind))


def _ordered_unique(items):
    return tuple(dict.fromkeys(item for item in items if item))
