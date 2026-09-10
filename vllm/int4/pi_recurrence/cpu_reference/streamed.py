"""Bounded same-weight FP32 CPU reference building blocks.

No full-model entry point. All compute is independent Torch; dequantized weights
round to FP16, but activations/accumulation are FP32, a deliberate confound.
"""
import ast
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open

SOURCE_SHA = '67cf849081143a998f0e189c551e4b5f532365a055805570747385e400372abb'
NAMES = {
 'Qwen3_5TextRotaryEmbedding','Qwen3_5RMSNormGated','apply_mask_to_padding_states',
 'causal_conv1d_update','causal_conv1d_fn','l2norm','torch_chunk_gated_delta_rule',
 'torch_recurrent_gated_delta_rule','Qwen3_5GatedDeltaNet','rotate_half',
 'apply_rotary_pos_emb','repeat_kv','eager_attention_forward','Qwen3_5Attention',
 'Qwen3_5MLP','Qwen3_5RMSNorm','Qwen3_5DecoderLayer'}


def blocked_attention(module, query, key, value, attention_mask, scaling,
                      dropout=0., query_block=128, **kwargs):
    if attention_mask is not None or dropout != 0 or module.training:
        raise ValueError('Only eval, unpadded causal attention is supported')
    if query.device.type != 'cpu' or query.dtype != torch.float32:
        raise ValueError('CPU FP32 only')
    batch, heads, length, dim = query.shape
    if batch != 1 or key.shape[2] != length or value.shape != key.shape:
        raise ValueError('Only batch1 uncached same-length QKV supported')
    groups = module.num_key_value_groups
    if heads != key.shape[1]*groups: raise ValueError('GQA shape')
    out = torch.empty_like(query)
    for h in range(heads):
        kh = key[0,h//groups]; vh = value[0,h//groups]
        for start in range(0,length,query_block):
            end=min(start+query_block,length)
            scores=query[0,h,start:end] @ kh[:end].T * scaling
            mask=torch.arange(end)[None,:] > torch.arange(start,end)[:,None]
            scores.masked_fill_(mask,float('-inf'))
            out[0,h,start:end] = scores.softmax(-1) @ vh[:end]
    return out.transpose(1,2).contiguous(), None


class OnlyEager:
    @staticmethod
    def get_interface(name, fallback):
        if name != 'eager': raise ValueError('Only explicit eager reference allowed')
        return blocked_attention


def source_namespace(path):
    import transformers.models.qwen3_5.modeling_qwen3_5 as installed
    source=Path(path).read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Model source identity mismatch')
    if hashlib.sha256(Path(installed.__file__).read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Installed dependency source differs')
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in NAMES]
    if {n.name for n in nodes} != NAMES: raise ValueError('Source inventory')
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)):
                node.decorator_list=[d for d in node.decorator_list if isinstance(d,ast.Name) and d.id in ('staticmethod','classmethod','property')]
    ns=dict(vars(installed))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
    ns['ALL_ATTENTION_FUNCTIONS']=OnlyEager
    return ns


class Weights:
    def __init__(self, root):
        self.root=Path(root)
        config=json.loads((self.root/'config.json').read_text())
        q=config['quantization_config']
        if not (q['bits']==4 and q['group_size']==128 and q['sym'] is True and q['desc_act'] is False):
            raise ValueError('Unsupported GPTQ contract')
        self.index=json.loads((self.root/'model.safetensors.index.json').read_text())['weight_map']
        self.headers={}
        for shard in sorted(set(self.index.values())):
            if Path(shard).name != shard: raise ValueError('Nonlocal shard')
            with safe_open(self.root/shard,framework='pt',device='cpu') as f:
                for name in f.keys():
                    if self.index.get(name)!=shard: raise ValueError('Shard/index mismatch')
                    self.headers[name]=tuple(f.get_slice(name).get_shape())
        if set(self.headers)!=set(self.index): raise ValueError('Incomplete index')

    def read(self,name,sl=None):
        with safe_open(self.root/self.index[name],framework='pt',device='cpu') as f:
            return f.get_tensor(name) if sl is None else f.get_slice(name)[sl]

    def mapping(self, layer, number):
        prefix=f'model.language_model.layers.{number}.'
        actual={k for k in self.index if k.startswith(prefix)}
        used=set(); receipts=[]
        for name,param in layer.named_parameters():
            direct=prefix+name
            if direct in actual:
                if self.headers[direct] != tuple(param.shape): raise ValueError((direct,'shape'))
                keys=[direct]; quant=False
            else:
                if not name.endswith('.weight') or param.ndim!=2: raise ValueError(('Missing',direct))
                stem=direct[:-6]
                keys=[stem+x for x in ('qweight','qzeros','scales')]
                n,k=param.shape
                expected=[(k//8,n),(k//128,n//8),(k//128,n)]
                if k%128 or n%8 or any(self.headers.get(key)!=shape for key,shape in zip(keys,expected)):
                    raise ValueError(('Quant shape',direct,expected))
                quant=True
            used.update(keys)
            receipts.append({'parameter':name,'shape':list(param.shape),'quantized':quant,'keys':keys})
        if used!=actual: raise ValueError(('Unhandled layer tensors',sorted(actual-used),sorted(used-actual)))
        return receipts

    def load_layer(self,layer,number):
        receipts=self.mapping(layer,number)
        for rec in receipts:
            nshape=rec['shape']; keys=rec['keys']
            if not rec['quantized']:
                value=self.read(keys[0])
                if value.dtype not in (torch.float16,torch.float32,torch.bfloat16): raise ValueError('Unsupported dense dtype')
                value=value.float() if rec['parameter'].endswith('.A_log') else value.half().float()
            else:
                n,k=nshape
                value=torch.empty(n,k,dtype=torch.float32)
                shifts=torch.arange(8,dtype=torch.int64)[None,:,None]*4
                for begin in range(0,n,128):
                    end=min(n,begin+128)
                    packed=self.read(keys[0],(slice(None),slice(begin,end)))
                    scale=self.read(keys[2],(slice(None),slice(begin,end)))
                    if packed.dtype!=torch.int32 or scale.dtype!=torch.float16: raise ValueError('Quant dtype')
                    q=((packed.long()[:,None,:] >> shifts)&15).reshape(k,end-begin)
                    dense=(q.float()-8)*scale.float().repeat_interleave(128,dim=0)
                    value[begin:end]=dense.half().float().T
            if not torch.isfinite(value).all(): raise ValueError('Nonfinite weight')
            parent,attr=rec['parameter'].rsplit('.',1) if '.' in rec['parameter'] else ('',rec['parameter'])
            setattr(layer.get_submodule(parent),attr,torch.nn.Parameter(value,requires_grad=False))
        return receipts

    def embedding(self, ids):
        name='model.language_model.embed_tokens.weight'
        if ids.ndim!=2 or ids.shape[0]!=1: raise ValueError('Embedding batch1')
        # Lookup one token row at a time; no full FP32 embedding materialization.
        unique,inverse=torch.unique(ids.flatten(),sorted=True,return_inverse=True)
        rows=torch.stack([self.read(name,(int(i),slice(None))).half().float() for i in unique])
        return rows[inverse].reshape(*ids.shape,-1)

    def head(self,hidden,chunk=512):
        name='lm_head.weight'; n,k=self.headers[name]
        if hidden.ndim!=2 or hidden.shape[1]!=k or hidden.shape[0]>16:
            raise ValueError('At most16 selected positions')
        output=torch.empty(hidden.shape[0],n)
        for start in range(0,n,chunk):
            end=min(n,start+chunk)
            w=self.read(name,(slice(start,end),slice(None))).half().float()
            output[:,start:end]=F.linear(hidden,w)
        return output
