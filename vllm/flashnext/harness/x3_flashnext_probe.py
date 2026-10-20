"""Exact 12288-token real-model prefill, identity and concurrent coherence."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from bench_latency import identity, post_json, make_prompt, stream_completion, summarize_itl

BASE = 'http://127.0.0.1:18135'
ROOT = Path('/mnt/vm_8tb/b70/build/vllm-flashnext/x3')
MODEL = 'hotschmoe-dd'
report = {'identity':identity(BASE,MODEL,'qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-x3-prefill')}
assert report['identity']['first_ok'] and report['identity']['alias_ok'],report

def math(a,b,context=''):
    result=post_json(BASE+'/v1/chat/completions',dict(model=MODEL,messages=[{'role':'user','content':context+f'\nReturn only the integer: {a} * {b}.'}],temperature=0,max_tokens=16,chat_template_kwargs={'enable_thinking':False}),300)
    text=result['choices'][0]['message']['content'].strip()
    assert text==str(a*b),text
    return dict(a=a,b=b,text=text,usage=result.get('usage'))

report['coherence_c1']=[math(17,19),math(23,29)]
with ThreadPoolExecutor(2) as pool:
    report['coherence_c2']=list(pool.map(lambda pair:math(*pair),[(17,19),(23,29)]))
print('COHERENCE',json.dumps(report),flush=True)
prompt,_=make_prompt(BASE,MODEL,12400,'x3-exact-fn')
encoded=post_json(BASE+'/tokenize',{'model':MODEL,'prompt':prompt},300)
ids=encoded['tokens'][:12288]
assert len(ids)==12288,len(ids)
report['prefill']=[]
for i in range(5):
    if i==0:
        (ROOT/'flashnext-profile-mode').write_text('1 native')
    if i==1:
        (ROOT/'flashnext-profile-mode').write_text('1 pad')
    if i==2:
        (ROOT/'flashnext-profile-mode').write_text('0 pad')
    row=stream_completion(BASE,MODEL,ids,1,300)
    summary=summarize_itl(row)
    summary['profiled']=i<2
    summary['ep_reduce_scatter']='native' if i==0 else 'pad'
    summary['first_token']=row['text']
    assert row['prompt_tokens']==12288,row
    report['prefill'].append(summary)
    print('FLASHNEXT_PREFILL',i,json.dumps(summary),flush=True)
    (ROOT/'flashnext-serve.json').write_text(json.dumps(report,indent=2))
report['long_coherence']=math(17,19,prompt)
assert report['long_coherence']['usage']['prompt_tokens']>=12288
(ROOT/'flashnext-serve.json').write_text(json.dumps(report,indent=2))
print('LONG_COHERENCE',json.dumps(report['long_coherence']),flush=True)
print('DONE_FLASHNEXT',flush=True)
