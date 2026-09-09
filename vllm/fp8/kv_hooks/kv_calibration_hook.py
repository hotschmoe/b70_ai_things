"""Model-specific static KV scale collection/loading for R187/R276 research.

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
            if ((out / 'COLLECT').exists() and context.attn_metadata is not None
                    and key is not None and value is not None):
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
        legacy = (artifact.get('weights') == 'qwen3.8-27b/fp8-official'
                  and artifact.get('schema') == 'b70.qwen38-official-fp8-kv-scales.v1')
        current = (artifact.get('schema') == 'b70.qwen38-kv-scales.v2'
                   and artifact.get('weights') == 'qwen3.8-27b/int4-autoround-gptq-relabel-r212')
        if not (legacy or current):
            raise RuntimeError('unexpected calibration model identity')
        if current:
            import vllm
            import sysconfig
            package = Path(vllm.__file__).resolve().parent
            if package != Path(sysconfig.get_path('purelib')) / 'vllm':
                raise RuntimeError('calibration requires installed vLLM package')
            expected_ops = artifact.get('provenance', {}).get('xpu_ops_sha256')
            actual_ops = hashlib.sha256((package / '_xpu_ops.py').read_bytes()).hexdigest()
            if not expected_ops or expected_ops != actual_ops:
                raise RuntimeError('calibration XPU routing fingerprint mismatch')
        expected_config = artifact.get('model_config_sha256')
        actual_config = hashlib.sha256((Path(os.environ.get('B70_KV_MODEL_ROOT', '/model')) / 'config.json').read_bytes()).hexdigest()
        if not expected_config or expected_config != actual_config:
            raise RuntimeError('calibration model configuration fingerprint mismatch')
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
                             'query_quantized': self.query_quant is not None,
                             'attention_impl': type(self.impl).__name__,
                             'artifact_sha256': hashlib.sha256(artifact_bytes).hexdigest()}
            dump()
            print('B70_KV_SCALE_LOADED ' + name + ' ' + json.dumps(records[name]), flush=True)
            return result
        Attention.process_weights_after_loading = process
        if os.environ.get('B70_KV_AUDIT') == '1':
            audit = {}
            def dump_audit():
                path = out / ('kv-audit-' + str(os.getpid()) + '.json')
                temp = path.with_suffix('.tmp')
                temp.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
                temp.replace(path)
            atexit.register(dump_audit)
            original_forward = Attention.forward
            def audit_forward(self, query, key, value, *args, **kwargs):
                if ((out / 'AUDIT').exists() and key is not None and value is not None
                        and get_forward_context().attn_metadata is not None):
                    name = self.layer_name
                    from vllm.distributed import get_tensor_model_parallel_rank
                    rec = audit.setdefault(name, {'rank': get_tensor_model_parallel_rank(),
                        'forwards': 0, 'k_values': 0, 'v_values': 0,
                        'k_clipped': 0, 'v_clipped': 0, 'k_amax': 0., 'v_amax': 0.})
                    for label, tensor in [('k', key), ('v', value)]:
                        bound = 448. * scales[name][label + '_scale']
                        absolute = tensor.detach().abs()
                        maximum = float(absolute.amax().item())
                        if not math.isfinite(maximum):
                            raise RuntimeError('nonfinite held-out KV activation: ' + name)
                        rec[label + '_amax'] = max(rec[label + '_amax'], maximum)
                        rec[label + '_values'] += tensor.numel()
                        rec[label + '_clipped'] += int((absolute > bound).sum().item())
                    rec['forwards'] += 1
                    if rec['forwards'] == 1 or rec['forwards'] % 20 == 0:
                        dump_audit()
                return original_forward(self, query, key, value, *args, **kwargs)
            Attention.forward = audit_forward
