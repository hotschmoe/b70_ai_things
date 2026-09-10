"""One isolated full-length layer, CPU only; hard600-second alarm."""
import argparse
import json
import resource
import signal
import time
from pathlib import Path
import torch
from streamed import Weights,source_namespace


def limits():
    if Path('/dev/dri').exists(): raise RuntimeError('No GPU device exposure permitted')
    mem=(Path('/sys/fs/cgroup')/'memory.max').read_text().strip()
    cpu=(Path('/sys/fs/cgroup')/'cpu.max').read_text().split()
    if mem=='max' or int(mem)>12*1024**3 or cpu[0]=='max' or int(cpu[0])/int(cpu[1])>8:
        raise RuntimeError('Requires explicit <=12GiB and <=8CPU cgroup caps')
    return {'memory_max':int(mem),'cpu_max':cpu}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--ids',required=True);p.add_argument('--layer',type=int,choices=[0,3],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();caps=limits();signal.alarm(600);torch.set_num_threads(8)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'started.json').write_text(json.dumps({'layer':a.layer,'caps':caps,'deadline_seconds':600})+'\n')
    from transformers import AutoConfig
    cfg=AutoConfig.from_pretrained(a.model,local_files_only=True).text_config;cfg._attn_implementation='eager'
    if cfg.rope_parameters['rope_type']!='default':raise ValueError('Unsupported RoPE')
    ns=source_namespace(a.source);w=Weights(a.model)
    case=max(json.loads(Path(a.ids).read_text()),key=lambda c:len(c['combined_ids']))
    ids=torch.tensor(case['combined_ids']).reshape(1,-1)
    if ids.shape[1]!=6102:raise ValueError('Expected frozen6102-token prefix')
    start=time.perf_counter();hidden=w.embedding(ids)
    rotary=ns['Qwen3_5TextRotaryEmbedding'](cfg)
    pos=torch.arange(ids.shape[1]).reshape(1,-1)
    with torch.inference_mode():pe=rotary(hidden,pos)
    with torch.device('meta'):layer=ns['Qwen3_5DecoderLayer'](cfg,a.layer)
    w.load_layer(layer,a.layer);layer.eval();loaded=time.perf_counter()
    with torch.inference_mode():y=layer(hidden,position_embeddings=pe,attention_mask=None)
    elapsed=time.perf_counter()-loaded
    if y.shape!=hidden.shape or not torch.isfinite(y).all():raise ValueError('Invalid output')
    result={'passed':True,'scope':'isolated layer on actual prefix embeddings; NOT intermediate activations or full model',
      'layer':a.layer,'tokens':ids.shape[1],'load_embedding_seconds':loaded-start,
      'forward_seconds':elapsed,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'output_abs_max':float(y.abs().max()),'caps':caps,'direct_rounding':'FP16 then FP32 except A_log FP32; norm+1 FP32',
      'fp32_compute_confound':True}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));signal.alarm(0)
if __name__=='__main__':main()
