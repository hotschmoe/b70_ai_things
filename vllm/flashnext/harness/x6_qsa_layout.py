"""CPU contract probe of the real XPU platform's hybrid cache alignment."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from b70_qsa_xpu import kv_spec
from vllm.config import CacheConfig
from vllm.model_executor.layers.attention.attention import Attention
from vllm.platforms.xpu import XPUPlatform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

kv_spec.register_specs()

def registered(backend):
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer.get_attn_backend = lambda: backend
    return layer

layers = {f'layers.{i}.attn': registered(kv_spec.B70QsaBackend) for i in range(3, 48, 4)}
layers['layers.0.gdn'] = registered(GDNAttentionBackend)
cache = CacheConfig(mamba_cache_mode='align')
model = NS(is_hybrid=True, use_mla=False, dtype=torch.bfloat16, architecture='Qwen4ExpForConditionalGeneration',
           get_num_kv_heads=lambda _: 2, get_head_size=lambda: 256)
cfg = NS(cache_config=cache, model_config=model, parallel_config=NS(),
         compilation_config=NS(static_forward_context=layers))
fake_model = NS(get_mamba_specs_from_config=lambda _: [NS(page_size_bytes=3268608)])
with patch('vllm.model_executor.models.ModelRegistry.resolve_model_cls', return_value=(fake_model, 'x6')):
    XPUPlatform.update_block_size_for_backend(cfg)
assert XPUPlatform._find_non_ssm_backends(cfg) == [kv_spec.B70QsaBackend]
spec = kv_spec.make_qsa_spec(n_kv=2, head_dim=256, idx_dim=128, idx_block=4,
                             block_size=cache.block_size, pooled=True)
assert cache.block_size % 64 == 0
assert spec.page_size_bytes == cache.mamba_page_size_padded
assert cache.mamba_block_size == cache.block_size
print('X6_QSA_HYBRID_OK', 'layers=12', 'block=' + str(cache.block_size), 'page=' + str(spec.page_size_bytes))
