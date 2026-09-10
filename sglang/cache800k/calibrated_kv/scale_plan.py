"""CPU-only exact-artifact mapper. Does not import torch or mutate a model."""
import hashlib
import json
import math
import struct

WEIGHTS = 'qwen3.8-27b/int4-autoround-gptq-relabel-r212'
SOURCE_OPS = '6ee6b8db18759873246aca28e85ca6d2ba177eb08bfd3b9b0f0feea168cee9b3'
TARGET_LAYERS = list(range(3,64,4))
DRAFT = 'mtp.layers.0.self_attn.attn'
NAMES = {f'language_model.model.layers.{i}.self_attn.attn' for i in TARGET_LAYERS} | {DRAFT}


def make_plan(artifact_bytes, config_bytes, *, role, tp_rank, tp_size,
              module_inventory, backend='triton', kv_dtype='fp8_e4m3',
              pp_size=1, dcp_size=1, query_quantized=False):
    artifact=json.loads(artifact_bytes)
    config=json.loads(config_bytes)
    if artifact.get('schema') != 'b70.qwen38-kv-scales.v2' or artifact.get('weights') != WEIGHTS:
        raise ValueError('calibration identity mismatch')
    if artifact.get('model_config_sha256') != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError('model config bytes mismatch; transformed config needs explicit provenance port')
    if artifact.get('provenance',{}).get('xpu_ops_sha256') != SOURCE_OPS:
        raise ValueError('unexpected source calibration routing provenance')
    if artifact.get('expected_layers') != 17 or set(artifact.get('layers',{})) != NAMES:
        raise ValueError('expected exactly 16 target and one MTP calibrated attention layers')
    text=config.get('text_config',{})
    if (text.get('num_hidden_layers') !=64 or text.get('num_key_value_heads') !=4
        or [i for i,x in enumerate(text.get('layer_types',[])) if x=='full_attention'] != TARGET_LAYERS):
        raise ValueError('model attention layout mismatch')
    if tp_size not in (1,2) or tp_rank not in range(tp_size) or pp_size !=1 or dcp_size !=1:
        raise ValueError('unmapped TP/PP/DCP topology')
    if role not in ('target','draft'):
        raise ValueError('explicit target/draft role required')
    if backend != 'triton' or kv_dtype != 'fp8_e4m3' or query_quantized:
        raise ValueError('only FP16-query Triton E4M3FN plumbing reviewed; intel_xpu prefill has no descales')
    observations=artifact.get('observations',{})
    if set(observations) != {'0','1'}:
        raise ValueError('expected both TP2 calibration ranks')
    for rank,obs in observations.items():
        if set(obs) !=NAMES:
            raise ValueError('incomplete rank layer observations')
    headroom=artifact.get('headroom')
    if not isinstance(headroom,(int,float)) or isinstance(headroom,bool) or not math.isfinite(headroom) or headroom<1:
        raise ValueError('invalid headroom')
    for name,record in artifact['layers'].items():
        if set(record) != {'q_scale','k_scale','v_scale'}:
            raise ValueError('unexpected scale fields')
        for label in ('q','k','v'):
            scale=record[label+'_scale']
            if not isinstance(scale,(int,float)) or isinstance(scale,bool) or not math.isfinite(scale) or scale<=0:
                raise ValueError('invalid nonpositive/nonfinite scale')
            amax=[]
            for rank in ('0','1'):
                value=observations[rank][name][label+'_amax']
                if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                    raise ValueError('invalid observation')
                amax.append(value)
            expected=max(amax)*headroom/448.0
            if not math.isclose(scale,expected,rel_tol=1e-12,abs_tol=1e-15):
                raise ValueError('scale does not equal merged-rank E4M3FN calibration')
    paths={f'model.layers.{i}.attn':i for i in TARGET_LAYERS} if role=='target' else {'model.model.layers.0.attn':0}
    if module_inventory != paths:
        raise ValueError('actual RadixAttention module paths/layer IDs do not match reviewed layout')
    rows=[]
    for path,index in paths.items():
        name=f'language_model.model.layers.{index}.self_attn.attn' if role=='target' else DRAFT
        rec=artifact['layers'][name]
        def f32(value):return struct.unpack('f',struct.pack('f',value))[0]
        rows.append(dict(module_path=path,layer_id=index,source_layer=name,
                         k_scale=f32(rec['k_scale']),v_scale=f32(rec['v_scale']),
                         source_q_scale=rec['q_scale'],query_scale_applied=False))
    return dict(schema='b70.sglang-qwen38-kv-port-plan.v1',role=role,tp_rank=tp_rank,tp_size=tp_size,
                calibration_ranks=[0,1],rank_mapping='same merged max scale on every target/draft TP rank',
                artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
                config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                backend=backend,kv_dtype=kv_dtype,storage='torch.float8_e4m3fn, finite max448',
                rows=rows,status='CPU mapping only; no tensors assigned, no GPU qualification')
