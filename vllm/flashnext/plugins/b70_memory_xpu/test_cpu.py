"""CPU checks of forward equivalence and workspace ownership (no XPU calls)."""
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from b70_memory_xpu.fixes import alias_dense_shards, install


torch.manual_seed(2)
layer = nn.Module()
layer.weight = nn.Parameter(torch.randn(5, 7), requires_grad=False)
dense = layer.weight.detach().clone()
layer.b70_shards = {"full": SimpleNamespace(packed=None, dense=dense)}
x = torch.randn(3, 7)
before = torch.nn.functional.linear(x, layer.weight)
alias_dense_shards(layer)
assert torch.equal(before, torch.nn.functional.linear(x, layer.weight))
assert layer.weight.data_ptr() == dense.data_ptr()

layer.weight = nn.Parameter(torch.empty(5, 7), requires_grad=False)
layer.b70_shards = {"b": SimpleNamespace(packed=None, dense=dense[:2]),
                    "a": SimpleNamespace(packed=None, dense=dense[2:])}
before = torch.cat([torch.nn.functional.linear(x, s.dense) for s in layer.b70_shards.values()], -1)
alias_dense_shards(layer)
after = torch.cat([torch.nn.functional.linear(x, s.dense) for s in layer.b70_shards.values()], -1)
assert torch.equal(before, after) and layer.weight.numel() == 0

# Install only needs the method class. Avoid importing vLLM or detecting a
# device in this CPU unit test; exercise the real dequant scratch implementation.
stub = ModuleType("vllm.models.qwen4_exp.gguf.linear")
stub.B70GGUFLinearMethod = type("Method", (), {"process_weights_after_loading": lambda self, layer: None})
sys.modules[stub.__name__] = stub
from b70_gguf_xpu import prefill_dq
install()
device = torch.device("cpu")
prefill_dq._SCRATCH.clear()
w = prefill_dq._scratch(("dq", 21, 9, 7), (9, 7), torch.float32, device)
w.copy_(torch.randn_like(w))
y = x @ w.t()
expected = y.clone()
v = prefill_dq._scratch(("dq", 2, 3, 7), (3, 7), torch.float32, device)
assert v.data_ptr() == w.data_ptr()
v.fill_(50)
assert torch.equal(y, expected)
assert len(prefill_dq._SCRATCH) == 1
v = prefill_dq._scratch(("dq", 23, 11, 7), (11, 7), torch.float32, device)
assert v.numel() == 77 and len(prefill_dq._SCRATCH) == 1
print("CPU_ALIAS_FORWARD_OK STREAM_ARENA_OWNERSHIP_OK")

# Check exception-safe restoration of allocator settings without a device.
from contextlib import contextmanager
platform = ModuleType("vllm.platforms")
platform.current_platform = SimpleNamespace(is_xpu=lambda: True)
worker = ModuleType("vllm.v1.worker.gpu_worker")

class Worker:
    @contextmanager
    def _scoped_allocator_max_split(self, max_split_size_mb):
        yield

worker.Worker = Worker
sys.modules[platform.__name__] = platform
sys.modules[worker.__name__] = worker
settings = ["garbage_collection_threshold:0.8"]
torch._C._accelerator_getAllocatorSettings = lambda: settings[0]
torch._C._accelerator_setAllocatorSettings = lambda value: settings.__setitem__(0, value)
from b70_memory_xpu.allocator import install as install_allocator
install_allocator()
original = settings[0]
try:
    with Worker()._scoped_allocator_max_split(20):
        assert settings[0].endswith("max_split_size_mb:20")
        raise RuntimeError("exercise cleanup")
except RuntimeError:
    pass
assert settings[0] == original
print("ALLOCATOR_EXCEPTION_RESTORE_OK")
