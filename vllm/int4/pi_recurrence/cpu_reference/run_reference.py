"""Review-gated, layer-streamed CPU teacher-forced scoring.

No generation. Requires an explicit immutable plan and --run-full. FP32 compute
is intentionally different from serving FP16. No GPU or external kernels.
"""
import argparse
import gc
import hashlib
import json
import os
import resource
import time
from pathlib import Path
import torch
from streamed import Weights,source_namespace,SOURCE_SHA
from validate_long_layer import limits

MANIFEST_SHA='f0f3c197466b8cc5206cf5cc005fc73e1bbcad28539aeb5f9dd378a2310d898d'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def stamp(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def verify_plan(plan):
    if plan['image']!='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067':
        raise ValueError('Unreviewed image')
    for name,sha in plan['sources'].items():
        if Path(name).name!=name or digest(Path(__file__).parent/name)!=sha:raise ValueError(('Source',name))
    if set(plan['sources'])!={'streamed.py','run_reference.py','validate_long_layer.py'}:raise ValueError('Source inventory')
    for name in ('ids','payload'):
        if digest(plan[name])!=plan[name+'_sha256']:raise ValueError(('Input hash',name))
    if digest(plan['model_manifest'])!=MANIFEST_SHA:raise ValueError('Model manifest pin')
    manifest=json.loads(Path(plan['model_manifest']).read_text())
    if len(manifest)!=17:raise ValueError('Expected17 verified model files')
    before={};root=Path(plan['model'])
    for name,rec in manifest.items():
        if Path(name).name!=name:raise ValueError('Nonlocal model path')
        p=root/name;before[name]=stamp(p)
        if p.stat().st_size!=rec['bytes'] or digest(p)!=rec['sha256']:raise ValueError(('Model identity',name))
        if stamp(p)!=before[name]:raise ValueError('Model changed while hashing')
    if {p.name for p in root.glob('*.safetensors')} != {n for n in manifest if n.endswith('.safetensors')}:
        raise ValueError('Weight inventory differs')
    if digest(plan['model_source'])!=SOURCE_SHA:raise ValueError('Transformers source')
    return before


def score_rows(logits,positions,expected_ids,tokenizer):
    rows=[]
    for row,p in zip(logits,positions):
        targets=sorted(set([248046,248044,expected_ids[p+1]]))
        top=torch.topk(row,20)
        rows.append({'causal_input_position_zero_based':p,'next_position':p+1,
          'expected_next_token':expected_ids[p+1],
          'targets':[{'id':i,'logit':float(row[i]),'rank':int((row>row[i]).sum())+1,
                      'margin_to_top':float(top.values[0]-row[i]),'decoded':tokenizer.decode([i])} for i in targets],
          'top20':[{'id':int(i),'logit':float(v),'decoded':tokenizer.decode([int(i)])} for i,v in zip(top.indices,top.values)]})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--run-full',action='store_true');a=p.parse_args()
    if not a.run_full:raise SystemExit('Full pass not enabled; review plan then explicitly pass --run-full')
    if digest(a.plan)!=a.plan_sha256:raise ValueError('Plan hash')
    caps=limits();torch.set_num_threads(8)
    if torch.get_num_threads()>8:raise ValueError('Thread cap')
    plan=json.loads(Path(a.plan).read_text());out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.time();(out/'STARTED.json').write_text(json.dumps({'started':started,'caps':caps,'plan_sha256':a.plan_sha256})+'\n')
    before=verify_plan(plan)
    from transformers import AutoConfig,AutoTokenizer
    cfg=AutoConfig.from_pretrained(plan['model'],local_files_only=True).text_config;cfg._attn_implementation='eager'
    if cfg.num_hidden_layers!=64 or cfg.rope_parameters['rope_type']!='default':raise ValueError('Configuration scope')
    tok=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
    payload=json.loads(Path(plan['payload']).read_text())
    if payload.get('chat_template_kwargs')!={'enable_thinking':False}:raise ValueError('Thinking scope')
    rendered=tok.apply_chat_template(payload['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prompt=tok.encode(rendered,add_special_tokens=False)
    observed=json.loads(Path(plan['ids']).read_text())
    if observed.get('observation_complete') is not True or observed.get('done') is not True or observed.get('errors') or observed['token_ids'][-1]!=248046:
        raise ValueError('Incomplete raw token observation')
    if observed['prompt_token_ids']!=prompt or len(observed['token_ids'])!=2058:
        raise ValueError('Observed token identity')
    input_ids=observed['prompt_token_ids']+observed['token_ids'][:-1]
    expected_ids=tok.encode(rendered+json.dumps(list(range(1,513))),add_special_tokens=False)
    if len(prompt)!=4045 or len(input_ids)!=6102 or input_ids!=expected_ids[:len(input_ids)]:raise ValueError('Prefix mismatch')
    positions=list(range(len(input_ids)-8,len(input_ids)))
    if 6098 not in positions or 6101 not in positions:raise ValueError('Missing observed boundaries')
    ns=source_namespace(plan['model_source']);weights=Weights(plan['model'])
    # Exhaustive metadata validation precedes the first real layer computation.
    mapping=[]
    for i in range(64):
        with torch.device('meta'):layer=ns['Qwen3_5DecoderLayer'](cfg,i)
        mapping.append(weights.mapping(layer,i));del layer
    (out/'mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    with torch.inference_mode():
        hidden=weights.embedding(torch.tensor(input_ids).reshape(1,-1))
        rotary=ns['Qwen3_5TextRotaryEmbedding'](cfg)
        pe=rotary(hidden,torch.arange(len(input_ids)).reshape(1,-1))
        for i in range(64):
            layer_start=time.perf_counter()
            with torch.device('meta'):layer=ns['Qwen3_5DecoderLayer'](cfg,i)
            weights.load_layer(layer,i);layer.eval()
            hidden=layer(hidden,position_embeddings=pe,attention_mask=None)
            if not torch.isfinite(hidden).all():raise ValueError(('Nonfinite layer',i))
            record={'layer':i,'seconds':time.perf_counter()-layer_start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            with (out/'layers.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            del layer;gc.collect()
        selected=hidden[0,positions].clone();del hidden;gc.collect()
        norm_weight=weights.read('model.language_model.norm.weight').half().float()
        if norm_weight.shape!=(cfg.hidden_size,):raise ValueError('Final norm shape')
        selected=selected*torch.rsqrt(selected.pow(2).mean(-1,keepdim=True)+cfg.rms_norm_eps)
        selected=selected*(1.+norm_weight)
        logits=weights.head(selected)
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite final logits')
        scores=score_rows(logits,positions,expected_ids,tok)
    after={name:stamp(Path(plan['model'])/name) for name in before}
    if before!=after:raise ValueError('Model file changed during computation')
    manifest=json.loads(Path(plan['model_manifest']).read_text())
    for name,rec in manifest.items():
        if digest(Path(plan['model'])/name)!=rec['sha256'] or stamp(Path(plan['model'])/name)!=before[name]:
            raise ValueError(('Post-run model identity',name))
    result={'passed':True,'scope':'Independent same-quant teacher-forced FP32 scoring, not endpoint qualification',
      'original_sampled_ids_available':True,'observed_response_ids':observed['response_ids'],'excluded_terminal_token':248046,'canonical_input_length':len(input_ids),'scores':scores,
      'model_manifest_sha256':MANIFEST_SHA,'plan_sha256':a.plan_sha256,'caps':caps,
      'elapsed_seconds':time.time()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'file_stats_before':before,'file_stats_after':after,'fp32_compute_confound':True}
    (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','elapsed_seconds','peak_rss_kib')}))
if __name__=='__main__':main()
