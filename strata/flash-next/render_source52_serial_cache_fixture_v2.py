"""Tokenizer-only authentic divergent suffix; no engine or weight imports."""
import hashlib,json,sys
from pathlib import Path
sys.path[:0]=['/tools','/serve']
from strata_tokenizer import Tokenizer
from frontend import ChatTemplate
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
corpus=json.loads(Path('/corpus.json').read_bytes());original=corpus['cases'][0]
tp=Path('/tokenizer');v=json.loads((tp/'vocab.json').read_bytes());tokens=[None]*len(v)
for token,i in v.items():tokens[i]=token
cfg=json.loads((tp/'tokenizer.json').read_bytes())
tok=Tokenizer(tokens,(tp/'merges.txt').read_text().split('\n'),json.loads((tp/'token_type.json').read_bytes()),cfg['pre'],cfg['special_ids'])
tpl=ChatTemplate(tp/'chat_template.jinja')
messages=json.loads(json.dumps(original['messages']));body=messages[-1]['content'];cut=body.rfind('\n\nTask: ')
if cut<0:raise ValueError('Original final Task boundary absent')
messages[-1]['content']=body[:cut]+'\n\nTask: Write a detailed operational review focused on rollback and acceptance criteria. Compare alternatives and distinguish observations from proposals.'
rendered=tpl.render(messages,enable_thinking=False);b=tok.encode(rendered,parse_special=True);a=tok.encode(tpl.render(original['messages'],enable_thinking=False),parse_special=True)
if a!=original['ids'] or len(a)!=2048 or a[:1536]!=b[:1536] or a==b or not 1536<len(b)<=3072:raise ValueError('Authentic shared1536 divergent token contract differs')
result={'schema':1,'A':a,'B':b,'pin':1536,'A_messages':original['messages'],'B_messages':messages,'B_rendered':rendered,'producer_sha256':sha(Path(__file__)),'input_corpus_sha256':sha(Path('/corpus.json')),'software_sha256':{str(p):sha(p)for p in[Path('/tools/strata_tokenizer.py'),Path('/serve/frontend.py')]},'tokenizer_sha256':{p.name:sha(p)for p in tp.iterdir()if p.is_file()},'actual_GPU_touch':False,'actual_model_payload_read':False}
with Path('/results/cache-fixture.json').open('x',encoding='ascii')as f:json.dump(result,f,indent=2,ensure_ascii=True);f.write('\n')
print(json.dumps({'A_tokens':len(a),'B_tokens':len(b),'pin':1536,'output':'/results/cache-fixture.json'}))
