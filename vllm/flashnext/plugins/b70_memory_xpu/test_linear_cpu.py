"""Exercise the real GGUF loading/forward paths on CPU, without device mounts."""
import torch
from torch import nn

from b70_memory_xpu.fixes import alias_dense_shards
from vllm.models.qwen4_exp.gguf.linear import B70GGUFLinearMethod

torch.set_num_threads(1)
torch.manual_seed(9)
method = B70GGUFLinearMethod()
layer = nn.Module()
method.create_weights(layer, 16, [8], 16, 8, torch.bfloat16)
method.load_shard(layer, None, torch.randn(8, 16).bfloat16(), 30, (8, 16))
method.process_weights_after_loading(layer)
x = torch.randn(4, 16).bfloat16()
before = method.apply(layer, x)
assert layer.weight.data_ptr() != layer.b70_shards["full"].dense.data_ptr()
alias_dense_shards(layer)
assert torch.equal(before, method.apply(layer, x))
assert layer.weight.data_ptr() == layer.b70_shards["full"].dense.data_ptr()

layer = nn.Module()
method.create_weights(layer, 16, [3, 5], 16, 8, torch.bfloat16)
method.load_shard(layer, "a", torch.randn(3, 16).bfloat16(), 30, (3, 16))
method.load_shard(layer, "b", torch.randn(5, 16).bfloat16(), 30, (5, 16))
method.process_weights_after_loading(layer)
before = method.apply(layer, x)
alias_dense_shards(layer)
assert layer.weight.numel() == 0
assert torch.equal(before, method.apply(layer, x))

layer = nn.Module()
method.create_weights(layer, 4, [8], 4, 8, torch.bfloat16)
layer.weight = nn.Parameter(layer.weight.unsqueeze(1), requires_grad=False)
method.load_shard(layer, None, torch.randn(8, 4).float(), 0, (8, 4))
method.process_weights_after_loading(layer)
x = torch.randn(2, 8, 9).bfloat16()
before = torch.nn.functional.conv1d(x, layer.weight, groups=8)
alias_dense_shards(layer)
assert layer.weight.shape == (8, 1, 4)
assert torch.equal(before, torch.nn.functional.conv1d(x, layer.weight, groups=8))
print("REAL_GGUF_FULL_MERGED_CONV_FORWARD_OK")
