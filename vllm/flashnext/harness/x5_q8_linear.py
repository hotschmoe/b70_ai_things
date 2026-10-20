"""Opt-in Q8_0 linear payloads using the already supported O3 kernels.

The original loader dequantizes Q8_0 to BF16 despite current kernels
supporting Q8_0. Keep only known 2-D block projections packed. Embeddings,
norms, routers, convolution, and every other type use the original loader.
This changes accumulation/activation quantization, so quality and fresh
greedy/MTP gates are required; it is not a bit-preserving optimization.
"""
import re
import sys


def install():
    from vllm.models.qwen4_exp.gguf import serve_loader as loader

    if getattr(loader, '_x5_q8_linear', False):
        return
    original = loader._tensor_payload
    names = re.compile(r'blk\.\d+\.(?:attn_(?:q|k|v|qkv|gate|output)|'
                       r'ssm_out|ffn_(?:gate|up|down)_shexp)\.weight')

    def payload(tensor):
        if int(tensor.tensor_type) == 8 and names.fullmatch(tensor.name):
            ne = tuple(int(v) for v in tensor.shape)
            assert len(ne) == 2
            raw = loader._owning_uint8(tensor.data)
            print(f'X5_Q8_LINEAR {tensor.name} shape={ne}', file=sys.stderr, flush=True)
            return raw, 8, ne
        return original(tensor)

    loader._tensor_payload = payload
    loader._x5_q8_linear = True
