"""Read-only final fresh-artifact review. Does not freeze or alter digests."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

CORPUS_SHA='40e6621e96cfff0f43151be75f124e5100ac00872c0b5a06ba0fe1b940457e6d'
MODEL_MANIFEST_SHA='f0f3c197466b8cc5206cf5cc005fc73e1bbcad28539aeb5f9dd378a2310d898d'
IMAGE='sha256:0328900cf1f8f29f5a8e76ed21a3eff71ef54d0abf495a88fdb86e93e880b077'
OLD_ARTIFACT='d53fb6565c485658b345d0b48aaf3ea4f5e0a48748e09a24bc6a29234c026a82'


def digest(raw):return hashlib.sha256(raw).hexdigest()
def integer(x):return type(x) is int and x>0


def check_rows(corpus, rows):
    expected=list(range(256))+['long32000','long64000','long80000','long96000',
        'continuation-Python iterator and generator error handling',
        'continuation-Rust ownership and lifetimes']
    assert [x['id'] for x in corpus]==expected,'wrong corpus identity/order'
    assert [x['id'] for x in rows]==expected,'incomplete/duplicate/reordered response IDs'
    for item,row in zip(corpus,rows):
        assert row['prompt_sha256']==digest(item['prompt'].encode()),'response prompt not bound to corpus'
        assert row['text_sha256']==digest(row['text'].encode()),'response text hash mismatch'
        assert not row['error'] and row['client_cancelled'] is False
        assert row['finish_reason'] in ('stop','length')
        assert integer(row['usage']['prompt_tokens']) and integer(row['usage']['completion_tokens'])
        assert row['usage']['completion_tokens']<=item['limit']
        for field in ('text','reasoning'):
            text=row.get(field,'')
            assert isinstance(text,str)
            assert not re.search(r'(.)\1{31}',text),'degenerate calibration response'
            lines=[x.strip() for x in text.splitlines() if x.strip()]
            assert not any(len(set(lines[i:i+8]))==1 for i in range(max(0,len(lines)-7)))


def review(root,config):
    run=root/'record-eager-prefixoff'
    assert (run/'exit.rc').read_text().strip()=='0'
    assert (run/'WORKLOADS_PASSED').exists()
    for name in ('post-failure.txt','force-remove.log','recovery.log','recovery-required.txt'):
        assert not (run/name).exists(),'lifecycle requires explicit failure review: '+name
    evidence={}
    for stage in ('pre','post'):
        p=run/(stage+'-xpu-health.log');text=p.read_text()
        assert all(token in text for token in ('card 0: OK','card 1: OK','HEALTHY (cards 0 1)'))
        assert not any(token in text for token in ('HEALTH_FAIL','HEALTH_OK False','WEDGED','INCONCLUSIVE'))
        evidence[p.name]=digest(p.read_bytes())
        p=run/(stage+'-xpu-collective-health.log');text=p.read_text()
        assert 'COLLECTIVE_HEALTH_OK' in text and 'xpu-collective-health: HEALTHY' in text
        evidence[p.name]=digest(p.read_bytes())
    assert (run/'stop.log').read_text().strip(),'missing clean teardown log'
    manifest_raw=(run/'manifest.json').read_bytes();meta=json.loads(manifest_raw);args=meta['args']
    assert meta['image']==IMAGE
    assert args['tensor_parallel_size']==2 and args['p2p']==0 and args['mtp']==3
    assert args['eager'] and args['prefix_off'] and args['kv_dtype']=='auto' and args['hook']=='record'
    corpus_raw=(run/'02-calibrate/corpus.json').read_bytes()
    assert digest(corpus_raw)==CORPUS_SHA
    responses_raw=(run/'02-calibrate/responses.jsonl').read_bytes()
    corpus=json.loads(corpus_raw);rows=[json.loads(x) for x in responses_raw.splitlines()]
    check_rows(corpus,rows)
    model_raw=(root/'fresh-preservation/model-files.json').read_bytes()
    assert digest(model_raw)==MODEL_MANIFEST_SHA
    files=json.loads(model_raw)
    pre=json.loads((root/'pre-verified-model.json').read_bytes())
    post=json.loads((root/'post-verified-model.json').read_bytes())
    assert pre==post and set(pre)==set(files)
    for name,value in files.items():
        assert pre[name]['sha256']==value['sha256'] and pre[name]['stat'][2]==value['bytes']
    artifact_path=root/'fresh-scales.json';raw=artifact_path.read_bytes();artifact=json.loads(raw)
    assert digest(raw)!=OLD_ARTIFACT,'old artifact is not fresh'
    provenance=artifact['provenance']
    assert provenance['image']==IMAGE and provenance['model_files']==files
    assert provenance['source_sha256']==meta['source_sha256']
    assert provenance['calibration_manifest_sha256']==digest(manifest_raw)
    assert provenance['corpus_sha256']==digest(corpus_raw)
    assert provenance['responses_sha256']==digest(responses_raw)
    assert provenance['requests']==262
    assert provenance['prompt_tokens']==sum(x['usage']['prompt_tokens'] for x in rows)
    assert provenance['completion_tokens']==sum(x['usage']['completion_tokens'] for x in rows)
    for rank,records in artifact['observations'].items():
        assert rank in ('0','1')
        for rec in records.values():
            assert type(rec['rank']) is int and rec['rank']==int(rank)
            assert integer(rec['n']) and integer(rec['tokens'])
            assert rec['kv_dtype']=='auto'
            for key in ('q','k','v'):
                assert rec[key+'_dtype']=='torch.float16'
                value=rec[key+'_amax']
                assert type(value) in (int,float) and math.isfinite(value) and value>0
    assert {str(p):digest(p.read_bytes()) for p in sorted(run.glob('kv-record-*.json'))}==artifact['sources']
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'calibrated_kv'))
    from scale_plan import make_plan
    plans=[]
    for role in ('target','draft'):
        inventory=({f'model.layers.{i}.attn':i for i in range(3,64,4)} if role=='target'
                   else {'model.model.layers.0.attn':0})
        for tp in (1,2):
            for rank in range(tp):
                plans.append(make_plan(raw,config.read_bytes(),role=role,tp_rank=rank,tp_size=tp,module_inventory=inventory))
    evidence.update({str(p.relative_to(root)):digest(p.read_bytes()) for p in (
        run/'exit.rc',run/'WORKLOADS_PASSED',run/'stop.log',root/'pre-verified-model.json',
        root/'post-verified-model.json',root/'fresh-preservation/model-files.json')})
    return dict(review_passed=True,artifact_sha256=digest(raw),artifact_path=str(artifact_path),
                requests=262,target_layers=16,draft_layers=1,reviewed_plans=len(plans),
                evidence=evidence,scope='artifact acceptance source/CPU gate; no backend qualification')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();print(json.dumps(review(a.root,a.config),indent=2))
