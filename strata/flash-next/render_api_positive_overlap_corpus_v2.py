"""Container-only original tokenizer/template rendering; no model engine."""
import hashlib,json,sys
from pathlib import Path
sys.path[:0]=['/tools','/serve']
from strata_tokenizer import Tokenizer
from frontend import ChatTemplate
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
tp=Path('/tokenizer');v=json.loads((tp/'vocab.json').read_text());tokens=[None]*len(v)
for token,i in v.items():tokens[i]=token
cfg=json.loads((tp/'tokenizer.json').read_text());t=Tokenizer(tokens,(tp/'merges.txt').read_text().split('\n'),json.loads((tp/'token_type.json').read_text()),cfg['pre'],cfg['special_ids']);template=ChatTemplate(tp/'chat_template.jinja')
seed=json.loads(Path('/seed.json').read_bytes());rows={}
for phase in ('warm','target'):
 rows[phase]=[]
 for i,message in enumerate(seed['messages'][phase]):
  rendered=template.render(message,enable_thinking=False);ids=t.encode(rendered,parse_special=True)
  if not 0<len(ids)<1984 or not all(type(x)is int and 0<=x<248320 for x in ids):raise ValueError('Actual token shape invalid')
  rows[phase].append({'logical_index':i,'messages':message,'rendered':rendered,'ids':ids})
result={'schema':1,'overlap_corpus_fixture_generation':2,'source_seed_sha256':sha(Path('/seed.json')),'producer_sha256':sha(Path(__file__)),'fixtures':rows,'tokenizer_file_sha256':{p.name:sha(p) for p in tp.iterdir() if p.is_file()},'source_file_sha256':{str(p):sha(p) for p in [Path('/tools/strata_tokenizer.py'),Path('/serve/frontend.py')]},'actual_GPU_touch':False,'actual_model_payload_read':False,'runtime':{'python':sys.version,'executable':sys.executable},'initial_checkpoint_nonreuse_qualified':False,'actual_CPU_continuation_observed':False,'actual_GPU_positive_overlap_observed':False,'warm_two_row_runtime_qualified':False,'matched_buffer_only_A_B_input_equivalent':False}
path=Path('/results/fixtures.json')
with path.open('x',encoding='ascii') as f:f.write(json.dumps(result,indent=2,ensure_ascii=True,allow_nan=False)+'\n')
