"""Experimental narrow repair: target Mamba groups are not draft attention.

Only repairs the unannotated EAGLE fallback. Explicitly annotated groups and
non-hybrid models retain stock behavior. Prompt-only storage is required.
"""
import json


def install():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import SchedulerOffloadConfig
    from vllm.v1.kv_cache_interface import MambaSpec
    original = SchedulerOffloadConfig.from_spec
    def from_spec(cls, spec, vllm_config, kv_cache_config):
        result = original(spec, vllm_config, kv_cache_config)
        if any(g.is_eagle_group for g in kv_cache_config.kv_cache_groups):
            return result
        mamba = {i for i, g in enumerate(kv_cache_config.kv_cache_groups) if isinstance(g.kv_cache_spec, MambaSpec)}
        if not mamba or not any(g.is_eagle_group for g in result.kv_group_configs):
            return result
        if not result.offload_prompt_only:
            raise ValueError('experimental Mamba offload repair requires prompt-only storage')
        groups = tuple(g._replace(is_eagle_group=False) if g.group_idx in mamba else g for g in result.kv_group_configs)
        print('B70_OFFLOAD_GROUP_FIX ' + json.dumps({'mamba_groups': sorted(mamba), 'remaining_eagle_groups': [g.group_idx for g in groups if g.is_eagle_group], 'prompt_only': True}), flush=True)
        return result._replace(kv_group_configs=groups)
    SchedulerOffloadConfig.from_spec = classmethod(from_spec)
