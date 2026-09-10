#!/usr/bin/env python3
"""Frozen long retrieval, guide and cancellation checks; no tools are executed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'fp8'))
from kv_campaign_probe import stream,request,generation_issue

def validate(row,item,reuse=False,cancel=False):
    assert row['error'] is None
    assert not re.search(r'(.{1,80}?)\1{15,}',row['text']+row['reasoning'],re.S),'repeated output unit'
    if cancel:
        assert row['client_cancelled'] and row['finish_reason'] is None
        return
    assert not row['client_cancelled'] and row['finish_reason']=='stop','incomplete response'
    assert row['usage']['prompt_tokens']==item['prompt_tokens'],'runtime tokenizer count mismatch'
    assert 0<row['usage']['completion_tokens']<=item['max_tokens']
    hit=row['usage']['prompt_tokens_details']['cached_tokens']
    assert type(hit) is int and 0<=hit<=item['prompt_tokens']
    if reuse:assert hit>0,'cache reuse not observed'
    if item['kind']=='retrieval':assert json.loads(row['text'])==item['expected'],'wrong long-context secret'
    else:
        assert generation_issue(row) is None
        assert len(row['text'])>=1000
        for word in ['Invariant','Implementation','Tests','Complexity','Concurrency']:
            assert re.search(r'(?im)^\s*(?:#+\s*)?(?:\*\*)?'+word+r'\b',row['text']), 'missing guide section '+word
        assert 'def ' in row['text'] and '```' in row['text']
        assert not row['text'].count('```')%2

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:18125');p.add_argument('--model',default='hotschmoe-dd');p.add_argument('--alias',required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['retrieval','reuse','guide','cancel','recovery'],required=True);p.add_argument('--namespace',required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    identity=json.loads(request(a.base,'/v1/models'))
    assert {a.model,a.alias} <= {m['id'] for m in identity['data']}
    (a.out/'models.json').write_text(json.dumps(identity,indent=2)+'\n')
    (a.out/'metrics-before.txt').write_text(request(a.base,'/metrics'))
    rows=json.loads(a.corpus.read_text())
    assert len(rows)==8 and len({r['id'] for r in rows})==8
    for item in rows:
        assert hashlib.sha256(item['prompt'].encode()).hexdigest()==item['prompt_sha256']
        assert 180000<=item['prompt_tokens']<=190000 and item['prompt_tokens']+item['max_tokens']<200000
    selected=[r for r in rows if r['kind']==('guide' if a.mode in ('guide','cancel') else 'retrieval')]
    if a.mode=='cancel':selected=selected[:1]
    def run(item):
        row=stream(a.base,a.model,item['prompt'],item['max_tokens'],salt=a.namespace,timeout=1200,cancel_after_chunks=8 if a.mode=='cancel' else None)
        row.update(id=item['id'],mode=a.mode,expected_prompt_tokens=item['prompt_tokens'])
        try:validate(row,item,reuse=a.mode in ('reuse','recovery'),cancel=a.mode=='cancel');row['passed']=True
        except Exception as exc:row.update(passed=False,quality_error=ascii(exc))
        (a.out/(item['id']+'.json')).write_text(json.dumps(row,ensure_ascii=True,indent=2)+'\n')
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,selected))
    passed=all(r['passed'] for r in results)
    if a.mode=='cancel' and passed:
        # Client stream closure is a cancellation request, not scheduler proof.
        # Require the running/waiting gauges to drain before recovery begins.
        deadline=time.monotonic()+60;drained=False
        while time.monotonic()<deadline:
            metrics=request(a.base,'/metrics')
            running=[float(m.group(2)) for m in re.finditer(r'^vllm:num_requests_(running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)$',metrics,re.M)]
            if running and all(n==0 for n in running):drained=True;break
            time.sleep(1)
        passed=drained
        (a.out/'cancellation-drain.json').write_text(json.dumps(dict(passed=drained,scope='HTTP closure plus idle gauges; no worker cancellation trace'))+'\n')
    (a.out/'metrics-after.txt').write_text(request(a.base,'/metrics'))
    summary=dict(passed=passed,checks=len(results),mode=a.mode,corpus_sha256=hashlib.sha256(a.corpus.read_bytes()).hexdigest(),limitations='Guide structure/degeneration gate is not full semantic certification. No external tools executed.')
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
