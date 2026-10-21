"""Synthetic qwen4 model: real vLLM KV lifecycle and FULL decode graphs.

No quality/performance claim: random dummy weights, narrow trunk/experts,
48 layers including all12 QSA24/2/256 layers, PLE disabled, single card.
"""
import json
import os
from pathlib import Path
import signal
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
import torch
from vllm import LLM, SamplingParams
from vllm.models.qwen4_exp.xpu.model import Qwen4ExpDecoderLayer
from vllm.models.qwen4_exp.xpu.indexer_weights import Qwen4ExpIndexerWeights, indexer_dims
from b70_qsa_xpu.model_hook import attach_qsa
from b70_qsa_xpu import register

register()
original = Qwen4ExpDecoderLayer.__init__

def with_indexer(self, vllm_config, layer_type, prefix=''):
    original(self, vllm_config, layer_type, prefix)
    if getattr(self, 'self_attn', None) is not None:
        cfg = vllm_config.model_config.hf_text_config
        heads, kv_heads, dim = indexer_dims(cfg)
        self.self_attn.indexer = Qwen4ExpIndexerWeights(
            hidden_size=cfg.hidden_size, index_n_heads=heads, index_kv_heads=kv_heads,
            index_head_dim=dim, rms_norm_eps=cfg.rms_norm_eps, quant_config=None,
            params_dtype=vllm_config.model_config.dtype, prefix=prefix + '.self_attn.indexer')
        self.qsa_runner = attach_qsa(self)

Qwen4ExpDecoderLayer.__init__ = with_indexer

def inspect_qsa(model):
    rows = []
    for layer in model.modules():
        runner = getattr(layer, 'qsa_runner', None)
        if runner is None:
            continue
        cache = runner.cache_layer.kv_cache
        assert runner.batch is not None and runner.buf is None
        assert cache.dtype == torch.uint8 and runner.batch.pages.data_ptr() == cache.data_ptr()
        assert runner._cache_token == runner._cache_id()
        rows.append(dict(name=runner.cache_layer.layer_name, shape=list(cache.shape),
                         block=runner.batch.block_size, dtype=str(cache.dtype)))
    assert len(rows) == 12, rows
    return rows

@torch.inference_mode()
def main():
    signal.alarm(180)
    llm = LLM(model=os.environ['X6_TINY_MODEL'], load_format='dummy', dtype='bfloat16',
              language_model_only=True, max_model_len=4096, max_num_batched_tokens=128,
              max_num_seqs=2, gpu_memory_utilization=0.16, enable_prefix_caching=False,
              mamba_cache_mode='align', enforce_eager=False,
              compilation_config=dict(mode=0, cudagraph_mode='FULL_DECODE_ONLY',
                                      cudagraph_capture_sizes=[1,2], cudagraph_num_of_warmups=2))
    print('X6_TINY_BOUND', json.dumps(llm.apply_model(inspect_qsa)), flush=True)
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    for n in (1,2):
        out = llm.generate([dict(prompt_token_ids=[1,2,3,4])] * n, params, use_tqdm=False)
        assert len(out) == n and all(len(r.outputs[0].token_ids) == 8 for r in out)
        print('X6_TINY_GENERATE_OK', n, [r.outputs[0].token_ids for r in out], flush=True)
    llm.apply_model(inspect_qsa)
    print('X6_TINY_FULL_DECODE_OK', flush=True)
    llm.llm_engine.engine_core.shutdown()
    signal.alarm(0)

if __name__ == '__main__':
    main()
