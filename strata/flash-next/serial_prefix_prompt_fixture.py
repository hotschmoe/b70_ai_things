#!/usr/bin/env python3
"""Run CPU-only in the pinned Python runtime with source/pack read-only."""
import hashlib,json,sys
from pathlib import Path
sys.path[:0]=['/src/tools','/src/serve']
from strata_tokenizer import Tokenizer
from frontend import ChatTemplate
p=Path('/pack/tokenizer');v=json.loads((p/'vocab.json').read_text());tokens=[None]*len(v)
for token,i in v.items():tokens[i]=token
cfg=json.loads((p/'tokenizer.json').read_text());tok=Tokenizer(tokens,(p/'merges.txt').read_text().split('\n'),json.loads((p/'token_type.json').read_text()),cfg['pre'],cfg['special_ids']);tpl=ChatTemplate(p/'chat_template.jinja')
system='Answer accurately and briefly. Return only the requested answer. Do not add explanations.'
def render(messages):return tok.encode(tpl.render(messages,enable_thinking=False),parse_special=True)
def common(a,b):
 n=0
 for x,y in zip(a,b):
  if x!=y:break
  n+=1
 return n
cases={}
for key,q in [('A','What is 17 plus 25? Answer with the number only.'),('B','What is 19 plus 23? Answer with the number only.'),('C','Name the capital of France. Answer with the city only.')]:cases[key]=render([{'role':'system','content':system},{'role':'user','content':q}])
long='Use the following inventory to answer the final question. '+''.join('Shelf %d contains exactly three sealed boxes. '%i for i in range(1,17))
for key,q in [('pin_A','How many boxes are on shelf 3? Answer with the number only.'),('pin_B','How many boxes are on shelf 8? Answer with the number only.')]:cases[key]=render([{'role':'system','content':system},{'role':'user','content':long+q}])
history=[{'role':'system','content':system},{'role':'user','content':'What is 2 plus 2?'},{'role':'assistant','content':'4'}]
for key,q in [('turn_A','What is 3 plus 3? Answer with the number only.'),('turn_B','What is 5 plus 5? Answer with the number only.')]:cases[key]=render(history+[{'role':'user','content':q}])
for key,q in [('evict_A','What is 17 plus 25? Answer with the number only.'),('evict_B','What is 19 plus 23? Answer with the number only.'),('evict_C','Name the capital of France. Answer with the city only.'),('evict_D','What is 17 plus 25? Answer with the number only.')]:
 cases[key]=render([{'role':'system','content':'Independent context '+key+'. '+system},{'role':'user','content':q}])
turn=tok.encode('<|im_start|>',parse_special=True);assert len(turn)==1
root=next(i for i in range(1,len(cases['A'])) if cases['A'][i]==turn[0])
turn_at=max(i for i,t in enumerate(cases['turn_A']) if t==turn[0] and i<len(cases['turn_A'])-3)
# Engine's last turn boundary is the assistant header; common history ends at the latest user header.
cases['turn_B']=cases['turn_A']+tok.encode(' ',parse_special=True)
turn_shared=max(i for i,t in enumerate(cases['turn_A']) if t==turn[0])
result={'schema':1,'transport':'native GEN using exact exported tokenizer/template','thinking':False,'vocab':len(tokens),'turn_token':turn[0],'root':root,'pin':common(cases['pin_A'],cases['pin_B'])-1,'turn_shared':turn_shared,'cases':cases,'turn_boundaries':{k:max(i for i,t in enumerate(v) if t==turn[0]) for k,v in cases.items()},'tokenizer_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir() if x.is_file()},'texts':{'system':system,'pin_inventory':long},'useful_prompts':3,'decode_probes':{text:tok.encode(text) for text in ['42','Paris','3','6']}}
assert result['root']<result['pin']<min(len(cases['pin_A']),len(cases['pin_B']))-1
print(json.dumps(result,ensure_ascii=True))
