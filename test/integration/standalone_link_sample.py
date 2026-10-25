"""Standalone helper/library dependency sample for subprocess tests."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from pythoc import compile, compile_to_executable, extern, i32, ptr
from pythoc import callable_type, python_entry
from pythoc.artifact import ArtifactRole, LinkPlan
from pythoc.build.output_manager import get_output_manager
from pythoc.c_helpers import post_inc


@extern(lib=os.environ["PC_TEST_LINK_LIBRARY"])
def link_probe_value() -> i32:
    pass


@compile
def main() -> i32:
    value: i32 = 4
    previous: i32 = post_inc(i32, ptr(value))
    return previous + value + link_probe_value() - 46


if __name__ == "__main__":
    groups = get_output_manager().get_all_groups()
    for wrapper in (python_entry.pythoc_make_module,
                    callable_type.PyInit__callable):
        assert groups[wrapper._binding.group_key]["artifact_role"] is (
            ArtifactRole.PYTHON_RUNTIME
        )
    assert main() == 0
    plan = LinkPlan.from_all_groups()
    runtime_groups = [
        group for group in get_output_manager().get_all_groups().values()
        if group.get("artifact_role", ArtifactRole.NATIVE)
        is not ArtifactRole.NATIVE
    ]
    assert all(group.get("obj_file") not in plan.obj_files
               for group in runtime_groups)
    snapshot = {
        str(Path(path).resolve()): {
            "mtime_ns": Path(path).stat().st_mtime_ns,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        for path in plan.obj_files
    }
    executable = str(Path("build/standalone-link-sample").resolve())
    compile_to_executable(output_path=executable)
    result = subprocess.run([executable], check=False)
    assert result.returncode == 0
    print("LINK_STATE=" + json.dumps({
        "objects": snapshot, "libraries": plan.link_libraries,
        "executable": executable,
        "runtime_group_count": len(runtime_groups),
    }))
