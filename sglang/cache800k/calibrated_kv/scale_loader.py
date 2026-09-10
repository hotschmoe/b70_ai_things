"""Schema-scoped experimental loader; GPU allocation only inside serving lease.

Installed with scale_plan.py under sglang.srt.model_executor.b70_calibrated_kv.
No implicit unit scales, alternate backend, or partial target/draft coverage.
"""
import hashlib
import json
from pathlib import Path

EXPECTED_ARTIFACT_SHA256 = 'd53fb6565c485658b345d0b48aaf3ea4f5e0a48748e09a24bc6a29234c026a82'

try:
    from .scale_plan import make_plan
except ImportError:  # Standalone CPU fixture runner.
    from scale_plan import make_plan


def verify_model_files(artifact, model_root):
    root=Path(model_root).resolve()
    files=artifact.get('provenance',{}).get('model_files',{})
    if not files or 'config.json' not in files or not any(n.endswith('.safetensors') for n in files):
        raise ValueError('missing full model-file provenance')
    index_name='model.safetensors.index.json'
    if index_name not in files:
        raise ValueError('reviewed model requires a hashed safetensors index')
    disk_weights={p.name for p in root.glob('*.safetensors')}
    manifest_weights={n for n in files if n.endswith('.safetensors')}
    if disk_weights !=manifest_weights:
        raise ValueError('unlisted or missing on-disk weight shards')
    if {p.name for p in root.glob('*.index.json')} != {index_name}:
        raise ValueError('unexpected weight index alternative')
    if any(root.glob('*.bin')) or any(root.glob('*.pt')) or any(root.glob('*.pth')):
        raise ValueError('unexpected alternative weight format')
    for name,record in files.items():
        if Path(name).is_absolute() or len(Path(name).parts)!=1:
            raise ValueError('invalid manifest filename')
        path=root/name
        if path.stat().st_size !=record['bytes']:
            raise ValueError('model-file size mismatch: '+name)
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b''):
                digest.update(chunk)
        if digest.hexdigest()!=record['sha256']:
            raise ValueError('model-file digest mismatch: '+name)

    index=json.loads((root/index_name).read_bytes())
    weight_map=index.get('weight_map',{})
    if not weight_map or not set(weight_map.values()).issubset(manifest_weights):
        raise ValueError('index references unverified weight shard')


def install_validated_plan(model, plan, *, radix_attention_type, tensor_factory):
    """Separate allocation from mutation; tensor_factory is torch-backed at runtime.

    CPU fixtures inject inert scalar objects. The live entry point below enforces
    XPU float16 and creates float32 device tensors. No old native code is loaded.
    """
    modules={n:m for n,m in model.named_modules() if isinstance(m,radix_attention_type)}
    expected={row['module_path']:row['layer_id'] for row in plan['rows']}
    if {n:m.layer_id for n,m in modules.items()} !=expected:
        raise ValueError('model inventory changed after validation')
    fields=('k_scale','v_scale','k_scale_float','v_scale_float')
    for layer in modules.values():
        if any(not hasattr(layer,key) or getattr(layer,key) is not None for key in fields):
            raise ValueError('refusing missing fields, existing scales, or repeated installation')
    staged=[]
    for row in plan['rows']:
        staged.append((modules[row['module_path']],row,
                       tensor_factory(row['k_scale']),tensor_factory(row['v_scale'])))
    for layer,row,k,v in staged:
        # RadixAttention starts with ordinary None attributes. Remove them before
        # nn.Module.register_buffer, which otherwise rejects the existing name.
        delattr(layer,'k_scale');delattr(layer,'v_scale')
        layer.register_buffer('k_scale',k,persistent=True)
        layer.register_buffer('v_scale',v,persistent=True)
        layer.k_scale_float=row['k_scale']
        layer.v_scale_float=row['v_scale']
    return len(staged)


def try_load(*, model, scale_path, model_root, kv_dtype, is_draft_worker,
             tp_rank,tp_size,pp_size,dcp_size,dp_attention,
             prefill_backend,decode_backend,device,gpu_id,model_dtype):
    """Return False only for absent path or a different artifact schema."""
    if scale_path is None:
        return False
    artifact_bytes=Path(scale_path).read_bytes()
    artifact=json.loads(artifact_bytes)
    if artifact.get('schema')!='b70.qwen38-kv-scales.v2':
        return False
    if hashlib.sha256(artifact_bytes).hexdigest()!=EXPECTED_ARTIFACT_SHA256:
        raise ValueError('unreviewed calibration artifact digest')
    if device!='xpu' or dp_attention:
        raise ValueError('calibrated port only reviewed for XPU without DP attention')
    if prefill_backend!='triton' or decode_backend!='triton':
        raise ValueError('calibrated port requires BOTH prefill and decode Triton')
    role='draft' if is_draft_worker else 'target'
    expected_class='Qwen3_5ForCausalLMMTP' if is_draft_worker else 'Qwen3_5ForConditionalGeneration'
    if type(model).__name__ !=expected_class:
        raise ValueError('unreviewed target/draft model class')
    import torch
    from sglang.srt.layers.radix_attention import RadixAttention
    if model_dtype !=torch.float16:
        raise ValueError('source calibration query/computation dtype must be float16')
    inventory={n:m.layer_id for n,m in model.named_modules() if isinstance(m,RadixAttention)}
    config_bytes=(Path(model_root)/'config.json').read_bytes()
    plan=make_plan(artifact_bytes,config_bytes,role=role,tp_rank=tp_rank,tp_size=tp_size,
                   pp_size=pp_size,dcp_size=dcp_size,module_inventory=inventory,
                   backend=prefill_backend,kv_dtype=kv_dtype,query_quantized=False)
    verify_model_files(artifact,model_root)
    # Query quantization must remain absent, not merely ignored in the plan.
    for _,layer in model.named_modules():
        if isinstance(layer,RadixAttention) and getattr(layer,'q_scale_float',None) is not None:
            raise ValueError('unexpected query scaling')
    target_device=torch.device(device,gpu_id)
    coverage=install_validated_plan(model,plan,radix_attention_type=RadixAttention,
        tensor_factory=lambda value:torch.tensor(value,dtype=torch.float32,device=target_device))
    print('B70_SGLANG_KV_SCALES_LOADED '+json.dumps(dict(plan,coverage=coverage),sort_keys=True),flush=True)
    model._b70_calibrated_kv_identity=plan
    return True
