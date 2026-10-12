"""CPU-only token decoding after the closed native experiment; no weight imports."""
import hashlib,json,sys
from pathlib import Path
sys.path[:0]=['/tools']
from strata_tokenizer import Tokenizer
p=Path('/tokenizer');v=json.loads((p/'vocab.json').read_bytes());tokens=[None]*len(v)
for token,i in v.items():tokens[i]=token
cfg=json.loads((p/'tokenizer.json').read_bytes());t=Tokenizer(tokens,(p/'merges.txt').read_text().split('\n'),json.loads((p/'token_type.json').read_bytes()),cfg['pre'],cfg['special_ids'])
source=Path('/requests.json');rows=json.loads(source.read_bytes());out=[]
for row in rows:
 ids=row['raw']['output_ids'];useful=[i for i in ids if i not in(248044,248046)];text=t.decode(useful,errors='strict');out.append({'ordinal':row['ordinal'],'useful_tokens':len(useful),'text':text,'decoded_text_utf8_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),'meaningful_output_review_required':True})
result={'requests_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'tokenizer_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in p.iterdir()if x.is_file()},'outputs':out,'model_payload_read':False,'GPU_touch':False}
with Path('/results/decoded.json').open('x',encoding='ascii')as f:json.dump(result,f,indent=2,ensure_ascii=True);f.write('\n')
