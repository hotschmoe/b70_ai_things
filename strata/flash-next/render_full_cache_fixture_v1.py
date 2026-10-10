"""Root-owned CPU-only tokenizer producer; run only in declared readonly fixture image."""
import hashlib,json,sys
from pathlib import Path
sys.path[:0]=['/tools','/serve']
from strata_tokenizer import Tokenizer
from frontend import ChatTemplate

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 tp=Path('/tokenizer');v=json.loads((tp/'vocab.json').read_text());tokens=[None]*len(v)
 for token,index in v.items():tokens[index]=token
 cfg=json.loads((tp/'tokenizer.json').read_text());tokenizer=Tokenizer(tokens,(tp/'merges.txt').read_text().split('\n'),json.loads((tp/'token_type.json').read_text()),cfg['pre'],cfg['special_ids']);template=ChatTemplate(tp/'chat_template.jinja');seed=json.loads(Path('/seed.json').read_text());rows={}
 for phase,messages in seed['messages'].items():
  rows[phase]=[]
  for index,message in enumerate(messages):
   rendered=template.render(message,enable_thinking=False);ids=tokenizer.encode(rendered,parse_special=True)
   if not 0<len(ids)<1984 or not all(type(t) is int and 0<=t<248320 for t in ids):raise ValueError('Bounded authentic source token fixture required')
   rows[phase].append({'logical_index':index,'messages':message,'rendered':rendered,'ids':ids})
 result={'schema':1,'source_seed_sha256':sha(Path('/seed.json')),'fixtures':rows,'tokenizer_file_sha256':{p.name:sha(p) for p in tp.iterdir() if p.is_file()},'source_file_sha256':{str(p):sha(p) for p in [Path('/tools/strata_tokenizer.py'),Path('/serve/frontend.py')]},'producer_source_sha256':sha(Path(__file__)),'actual_GPU_touch':False,'actual_model_payload_read':False,'runtime':{'python':sys.version,'executable':sys.executable},'shared_cache_runtime_qualified':False}
 Path('/results/fixtures.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='ascii')
if __name__=='__main__':main()
