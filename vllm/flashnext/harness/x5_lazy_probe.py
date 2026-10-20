"""Exercise X1's 2066-replay/RNG gate and normal interpreter finalization."""
import ctypes
import os
import runpy

import torch

helper = '/mnt/vm_8tb/b70/build/vllm-flashnext/x5/eventless-lazy/libx5_eventless_helper.so'
loaded_helper = ctypes.CDLL(helper, mode=ctypes.RTLD_GLOBAL)
os.environ['X1_EVENTLESS_SHIM'] = helper
runpy.run_path('/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness/x1_eventless_probe.py', run_name='__main__')
torch.xpu.synchronize()
print('X5_LAZY_PROBE_NORMAL_PYTHON_EXIT', flush=True)
