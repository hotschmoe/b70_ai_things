#!/usr/bin/env python3
"""CPU-only corpus freezing using the serving image's exact tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
GUIDE='Write a practical Python LRU cache implementation and testing guide, 500 to 700 words. Use exactly these section headings: Invariant, Implementation, Tests, Complexity, Concurrency. Include Python code and concrete eviction, update and capacity-zero tests. Explain the need for synchronization during concurrent operations. Finish the guide, do not continue the archive. Do not call tools.'

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    import transformers
    tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,trust_remote_code=False)
    def count(prompt):
        ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        return len(ids),hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    rows=[]
    for i in range(4):
        secrets=['B70-'+hashlib.sha256(f'long200k-{i}-{j}'.encode()).hexdigest()[:20] for j in range(3)]
        def make(n):
            parts=[f'Archive {i}. Three AUTHORIZED_SECRET records contain the only authoritative access codes. Retain all three in record order. Other records are ordinary background.\n']
            for j in range(n):
                if j in (n//4,n//2,3*n//4):parts.append('AUTHORIZED_SECRET='+secrets[(n//4,n//2,3*n//4).index(j)]+'\n')
                parts.append(f'Record {j:06d}: the warehouse checked its inventory ledger, stored the receipt, and completed the scheduled audit.\n')
            return ''.join(parts)
        query='\nReturn ONLY a JSON array of the three AUTHORIZED_SECRET values in record order. No markdown, explanation or tool calls.'
        low,high=1,8000
        while low<high:
            mid=(low+high)//2
            if count(make(mid)+query)[0]<185000:low=mid+1
            else:high=mid
        prefix=make(low)
        for kind,suffix,limit in [('retrieval',query,128),('guide','\n'+GUIDE,2048)]:
            prompt=prefix+suffix;n,ids_sha=count(prompt)
            assert 180000<=n<=190000 and n+limit<200000
            rows.append(dict(id=f'{kind}-{i}',index=i,kind=kind,prompt=prompt,prompt_tokens=n,token_ids_sha256=ids_sha,
                             prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),expected=secrets if kind=='retrieval' else None,max_tokens=limit))
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'corpus.json').write_text(json.dumps(rows,ensure_ascii=True,indent=2)+'\n')
    paths=[p for p in a.model.iterdir() if p.name.startswith('tokenizer') or p.name in ('chat_template.jinja','config.json','special_tokens_map.json','added_tokens.json')]
    (a.out/'tokenizer-identity.json').write_text(json.dumps(dict(transformers=transformers.__version__,tokenizer_class=type(tokenizer).__name__,files={p.name:digest(p) for p in paths},corpus_sha256=digest(a.out/'corpus.json'),counts=[dict(id=r['id'],prompt_tokens=r['prompt_tokens']) for r in rows]),indent=2)+'\n')
    print(json.dumps(dict(status='CPU_TOKENIZED',counts=[r['prompt_tokens'] for r in rows])))
if __name__=='__main__':main()
