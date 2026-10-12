"""CPU tokenizer-only corpus preparation; no engine or weight-file imports."""
import hashlib,json,sys
from pathlib import Path
sys.path[:0]=['/tools','/serve']
from strata_tokenizer import Tokenizer
from frontend import ChatTemplate
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tp=Path('/tokenizer');v=json.loads((tp/'vocab.json').read_bytes());tokens=[None]*len(v)
for token,i in v.items():tokens[i]=token
cfg=json.loads((tp/'tokenizer.json').read_bytes())
tok=Tokenizer(tokens,(tp/'merges.txt').read_text().split('\n'),json.loads((tp/'token_type.json').read_bytes()),cfg['pre'],cfg['special_ids'])
tpl=ChatTemplate(tp/'chat_template.jinja');seed=json.loads(Path('/seed.json').read_bytes());rows=[]
for case in seed['cases']:
 context='\n'.join(case['context_lines']);context_ids=tok.encode(context)
 def candidate(n,padding=''):
  body=tok.decode(context_ids[:n],errors='strict')+padding+'\n\nTask: '+case['instruction']
  messages=[{'role':'system','content':seed['system']},{'role':'user','content':body}]
  rendered=tpl.render(messages,enable_thinking=False)
  return messages,rendered,tok.encode(rendered,parse_special=True)
 lo,hi=0,len(context_ids)
 while lo<hi:
  mid=(lo+hi+1)//2
  if len(candidate(mid)[2])<=2048:lo=mid
  else:hi=mid-1
 found=None
 for n in range(lo,max(-1,lo-64),-1):
  for pad in range(65):
   value=candidate(n,'\n'+' detail'*pad if pad else '')
   if len(value[2])==2048:found=value;break
  if found:break
 if found is None:raise ValueError('Cannot fit exact P2048 without altering instruction: '+case['id'])
 messages,rendered,ids=found
 if any(type(x)is not int or not 0<=x<len(tokens) for x in ids):raise ValueError('Invalid token ID')
 rows.append({'id':case['id'],'role':case['role'],'messages':messages,'rendered':rendered,'ids':ids,'input_tokens':len(ids),'max_new_tokens':512,'native_request':'GEN 512 '+','.join(map(str,ids)),'target_min_useful_tokens':256})
result={'schema':1,'seed_sha256':sha(Path('/seed.json')),'worker_sha256':sha(Path(__file__)),'tokenizer_sha256':{p.name:sha(p) for p in tp.iterdir() if p.is_file()},'software_sha256':{str(p):sha(p)for p in [Path('/tools/strata_tokenizer.py'),Path('/serve/frontend.py')]},'cases':rows,'context':4096,'thinking':False,'actual_model_payload_read':False,'actual_GPU_touch':False,'natural_output_length_observed':False}
with Path('/results/corpus.json').open('x',encoding='ascii') as f:json.dump(result,f,indent=2,ensure_ascii=True,allow_nan=False);f.write('\n')
print(json.dumps({'cases':len(rows),'input_tokens':[len(r['ids'])for r in rows],'output':'/results/corpus.json'}))
