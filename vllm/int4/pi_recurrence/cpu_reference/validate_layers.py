"""CPU-only mapping plus isolated real-layer validation; not a model forward."""
import argparse
import gc
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from streamed import Weights, source_namespace, blocked_attention


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(); torch.set_num_threads(8);torch.manual_seed(42)
    from transformers import AutoConfig
    cfg=AutoConfig.from_pretrained(a.model,local_files_only=True).text_config
    cfg._attn_implementation='eager'
    if cfg.rope_parameters['rope_type']!='default': raise ValueError('Only default RoPE reviewed')
    ns=source_namespace(a.source); weights=Weights(a.model)
    mapping=[]
    for number in range(cfg.num_hidden_layers):
        with torch.device('meta'): layer=ns['Qwen3_5DecoderLayer'](cfg,number)
        mapping.append({'layer':number,'type':cfg.layer_types[number],'parameters':weights.mapping(layer,number)})
        del layer
    # Independent dense causal fixture across query block boundaries and GQA groups.
    mod=SimpleNamespace(training=False,num_key_value_groups=2)
    q=torch.randn(1,4,131,8);k=torch.randn(1,2,131,8);v=torch.randn_like(k)
    blocked,_=blocked_attention(mod,q,k,v,None,.125,query_block=17)
    mask=torch.ones(131,131,dtype=torch.bool).triu(1)
    kk=k.repeat_interleave(2,dim=1);vv=v.repeat_interleave(2,dim=1)
    dense=((q@kk.transpose(-1,-2))*.125).masked_fill(mask,float('-inf')).softmax(-1)@vv
    torch.testing.assert_close(blocked,dense.transpose(1,2),atol=5e-7,rtol=1e-5)
    try: blocked_attention(mod,q,k,v,mask,.125)
    except ValueError: pass
    else: raise AssertionError('Mask rejection')
    # Real embedding slice and one isolated layer of each kind, not chained.
    ids=torch.arange(65).reshape(1,-1)
    hidden=weights.embedding(ids)
    rotary=ns['Qwen3_5TextRotaryEmbedding'](cfg)
    pos=torch.arange(65).reshape(1,-1)
    with torch.inference_mode(): position_embeddings=rotary(hidden,pos)
    results=[]
    for number in (0,3):
        start=time.perf_counter()
        with torch.device('meta'): layer=ns['Qwen3_5DecoderLayer'](cfg,number)
        receipt=weights.load_layer(layer,number);layer.eval()
        loaded=time.perf_counter()
        with torch.inference_mode():
            y=layer(hidden,position_embeddings=position_embeddings,attention_mask=None)
            # Causality: changing suffix cannot affect first32 outputs.
            changed=hidden.clone();changed[:,32:]*=-1
            z=layer(changed,position_embeddings=position_embeddings,attention_mask=None)
        assert y.shape==hidden.shape and torch.isfinite(y).all()
        torch.testing.assert_close(y[:,:32],z[:,:32],atol=2e-5,rtol=2e-5)
        results.append({'layer':number,'type':cfg.layer_types[number], 'load_seconds':loaded-start,
          'two_forward_seconds':time.perf_counter()-loaded,'output_abs_max':float(y.abs().max()),
          'prefix_causality_max_abs':float((y[:,:32]-z[:,:32]).abs().max()),
          'parameter_bytes':sum(p.numel()*p.element_size() for p in layer.parameters()),
          'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        del layer,y,z;gc.collect()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    result={'passed':True,'scope':'64-layer meta mapping; isolated real layers0 and3 at65 tokens; NOT a model forward',
      'layers':results,'attention_fixture_max_abs':float((blocked-dense.transpose(1,2)).abs().max()),
      'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'fp32_compute_confounds':True,'head_full_vocabulary_not_executed':True}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
