#!/usr/bin/env python3
"""CPU-only exact-tokenizer padding matrix. No inference or generated tool use.

Only neutral padding and max_tokens differ from the original array256 task.
Padding can change model behavior: a moving failure position is evidence of
boundary association, not by itself proof of a cache implementation defect.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--runtime-out',type=Path,required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    import transformers
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,trust_remote_code=False)
    m=json.loads(a.corpus.read_text());old=next(i for i in m['cases'] if i['id']=='array-cap1024-repeat0');raw=Path(old['payload']).read_bytes();assert hashlib.sha256(raw).hexdigest()==old['sha256'];base=json.loads(raw);base['max_tokens']=2048
    def ids(payload):
        return tok.apply_chat_template(payload['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    n=len(ids(base));assert n==4035,(n,'expected original runtime count')
    a.out.mkdir(parents=True,exist_ok=False);cases=[]
    for delta in [0,128,256,512]:
        payload=copy.deepcopy(base)
        if delta:
            for count in range(1,1024):
                payload['messages'][0]['content']='Neutral padding:'+(' pad'*count)+'\n'+base['messages'][0]['content']
                if len(ids(payload))==n+delta:break
            else:raise AssertionError('no exact padding found')
        token_ids=ids(payload)
        for repeat in range(2):
            name=f'array256-padding{delta}-repeat{repeat}';path=a.out/(name+'.json');path.write_text(json.dumps(payload,ensure_ascii=True,indent=2)+'\n')
            cases.append(dict(id=name,kind='exact_array',payload=str(a.runtime_out/path.name),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cache_group=f'boundary-{delta}',require_hit=bool(repeat),expected=old['expected'],prompt_tokens=len(token_ids),token_ids_sha256=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),next_1600_boundary=4800,tokens_to_boundary=4800-len(token_ids)))
    result=dict(cases=cases,limitations=['Controlled padding diagnostic; no exact Pi replay. Failure alignment does not alone prove causality. No tools executed. Original failures preserved.'],transformers=transformers.__version__,tokenizer_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.model.iterdir() if p.name.startswith('tokenizer') or p.name in ('chat_template.jinja','config.json')})
    (a.out/'manifest.json').write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n');print(json.dumps([dict(id=i['id'],prompt_tokens=i['prompt_tokens'],tokens_to_boundary=i['tokens_to_boundary']) for i in cases]))
if __name__=='__main__':main()
