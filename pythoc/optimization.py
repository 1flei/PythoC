"""Version-compatible native LLVM optimization pipelines."""

import llvmlite
from llvmlite import binding

from .target_layout import _create_target_machine


_HAS_DEFAULT_NEW_PM = hasattr(binding, 'create_pass_builder')
_PIPELINE_POLICY = 'llvm-default-v1'


def _validate_level(level):
    if type(level) is not int or not 0 <= level <= 3:
        raise ValueError('optimization level must be an integer from 0 to 3')


def optimization_identity(level):
    """Identify output-affecting pipeline settings across cached builds."""
    _validate_level(level)
    manager = 'new' if _HAS_DEFAULT_NEW_PM else 'legacy'
    llvm_version = '.'.join(map(str, binding.llvm_version_info))
    return (f'{_PIPELINE_POLICY}:llvmlite={llvmlite.__version__}:'
            f'llvm={llvm_version}:pm={manager}:O{level}')


def optimize_ir(ir_text, level):
    """Use the installed LLVM's default pipeline, preserving old bindings."""
    _validate_level(level)
    with binding.parse_assembly(ir_text) as module:
        module.verify()
        # Older PassBuilder bindings reject O0. Leave the IR unoptimized.
        if level == 0:
            return str(module)
        target = binding.Target.from_triple(module.triple)
        with _create_target_machine(target) as target_machine:
            if _HAS_DEFAULT_NEW_PM:
                _run_new(module, target_machine, level)
            else:
                _run_legacy(module, target_machine, level)
        module.verify()
        return str(module)


def _run_new(module, target_machine, level):
    with binding.create_pipeline_tuning_options(speed_level=level) as tuning:
        with binding.create_pass_builder(target_machine, tuning) as builder:
            with builder.getModulePassManager() as manager:
                manager.run(module, builder)


def _run_legacy(module, target_machine, level):
    """Best-effort equivalent using the default pipeline of older LLVMs."""
    with binding.create_pass_manager_builder() as builder:
        builder.opt_level = level
        # Legacy bindings require explicitly installing the normal inliner.
        builder.inlining_threshold = 250 if level == 3 else 225
        builder.loop_vectorize = level >= 2
        builder.slp_vectorize = level >= 2
        builder.disable_unroll_loops = level < 2
        with binding.create_module_pass_manager() as manager:
            target_machine.add_analysis_passes(manager)
            builder.populate(manager)
            manager.run(module)
