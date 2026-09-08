"""Model-specific static KV scale collection/loading for the pinned R187 ABI.

Source port of the retained NVFP4 block-10 idea. No old scale artifacts used.
"""
import atexit
import hashlib
import json
import math
import os
from pathlib import Path


def install():
    import torch
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention.attention import Attention
    mode = os.environ['B70_KV_MODE']
    out = Path(os.environ['B70_KV_OUT'])
    records = {}
    def dump():
        if not records:
            return
        path = out / ('kv-' + mode + '-' + str(os.getpid()) + '.json')
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(records, indent=2, sort_keys=True) + '\n')
        tmp.replace(path)
    atexit.register(dump)
    if mode == 'record':
        original = Attention.forward
        def forward(self, query, key, value, *args, **kwargs):
            context = get_forward_context()
            # Skip dummy profiling/graph warmups, which may use synthetic state.
            if context.attn_metadata is not None and key is not None and value is not None:
                name = self.layer_name
                from vllm.distributed import get_tensor_model_parallel_rank
                rec = records.setdefault(name, {'rank': get_tensor_model_parallel_rank(), 'n': 0, 'tokens': 0, 'kv_dtype': self.kv_cache_dtype, 'q_dtype': str(query.dtype), 'k_dtype': str(key.dtype), 'v_dtype': str(value.dtype), 'q_amax': 0., 'k_amax': 0., 'v_amax': 0.})
                for label, tensor in [('q', query), ('k', key), ('v', value)]:
                    maximum = float(tensor.detach().abs().amax().item())
                    if not math.isfinite(maximum):
                        raise RuntimeError('nonfinite calibration activation: ' + name + ' ' + label)
                    rec[label + '_amax'] = max(rec[label + '_amax'], maximum)
                rec['n'] += 1
                rec['tokens'] += key.shape[0]
                # Atomic per-process files, with bounded crash loss.
                if rec['n'] % 20 == 0 or rec['n'] == 1:
                    dump()
            return original(self, query, key, value, *args, **kwargs)
        Attention.forward = forward
    else:
        artifact_bytes = Path(os.environ['B70_KV_SCALES']).read_bytes()
        artifact = json.loads(artifact_bytes)
        scales = artifact['layers']
        original = Attention.process_weights_after_loading
        def process(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if not str(self.kv_cache_dtype).startswith('fp8'):
                return result
            name = self.layer_name
            if name not in scales:
                raise RuntimeError('missing calibrated scales: ' + name)
            rec = scales[name]
            for label in ('q', 'k', 'v'):
                scale = rec[label + '_scale']
                if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                    raise RuntimeError('invalid calibrated scale: ' + name)
                getattr(self, '_' + label + '_scale').fill_(scale)
                setattr(self, '_' + label + '_scale_float', scale)
                mirror = getattr(self, '_' + label + '_scale_cpu', None)
                if mirror is not None:
                    mirror.fill_(scale)
            from vllm.distributed import get_tensor_model_parallel_rank
            records[name] = {'rank': get_tensor_model_parallel_rank(), 'kv_dtype': self.kv_cache_dtype, **rec,
                             'artifact_sha256': hashlib.sha256(artifact_bytes).hexdigest()}
            dump()
            print('B70_KV_SCALE_LOADED ' + name + ' ' + json.dumps(records[name]), flush=True)
            return result
        Attention.process_weights_after_loading = process
