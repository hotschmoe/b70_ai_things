"""Install opt-in hooks when vLLM imports their modules in its normal order."""
import importlib
import importlib.abc
import importlib.machinery
import os
import sys

TARGETS = {}
if os.environ.get('B70_TP_HOST_TRACE_DIR'):
    TARGETS['vllm.v1.worker.gpu_model_runner'] = ['b70_tp_host_trace']
if os.environ.get('B70_KV_MODE') in ('record', 'load'):
    TARGETS['vllm.model_executor.layers.attention.attention'] = ['kv_calibration_hook']
modules = []
if os.environ.get('B70_OFFLOAD_GROUP_FIX') == '1':
    modules.append('kv_offload_group_fix')
if os.environ.get('B70_OFFLOAD_TRACE') == '1':
    modules.append('kv_offload_trace')
if modules:
    TARGETS['vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler'] = modules


class HookLoader:
    def __init__(self, original, hooks):
        self.original, self.hooks = original, hooks

    def __getattr__(self, name):
        return getattr(self.original, name)

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        try:
            for name in self.hooks:
                importlib.import_module(name).install()
                print('B70_HOOK_INSTALLED ' + name, flush=True)
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(78)


class HookFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = HookLoader(spec.loader, TARGETS.pop(fullname))
        return spec


if TARGETS:
    sys.meta_path.insert(0, HookFinder())
