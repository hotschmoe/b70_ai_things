#!/usr/bin/env python3
"""Exact tokenizer/template CPU cases. Never read an original GGUF weight payload."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T185206Z-d5q32zc7/source')
BASELINE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f12-20261009/c1-onecard-segmented-integrated29-prepared-v8')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=True)+'\n',encoding='ascii')

def generate(source,baseline,out,lane='source29'):
 prepared=json.loads((baseline/'prepared.json').read_bytes());manifest=json.loads((baseline/'artifact-identity.json').read_bytes());cfg=json.loads((baseline/'server-config.json').read_bytes());pack=Path(prepared['pack'])/'tokenizer';runtime=manifest['runtime']
 for name,digest in runtime['python_sources'].items():require(sha(source/name)==digest,'Actual consumed Python source differs '+name)
 for name,digest in manifest['tokenizer_files'].items():require(sha(pack/name)==digest,'Qualified tokenizer/header/template export differs '+name)
 require(cfg.get('effort_position','start')=='start','This case contract pins actual default reasoning-effort placement')
 sys.path.insert(0,str(source));sys.path.insert(0,str(source/'tools'));server=importlib.import_module('serve.server');ST=importlib.import_module('strata_tokenizer')
 v=json.loads((pack/'vocab.json').read_bytes());tokens=[None]*len(v)
 for text,i in v.items():tokens[i]=text
 settings=json.loads((pack/'tokenizer.json').read_bytes());tok=ST.Tokenizer(tokens,(pack/'merges.txt').read_text().split('\n'),json.loads((pack/'token_type.json').read_bytes()),settings['pre'],settings['special_ids']);service=server.Service.__new__(server.Service);service.tok=tok;service.template=server.ChatTemplate(pack/'chat_template.jinja');service.literals=server.literal_tags(tok.control_tokens);service.effort_end=False
 source_text=(source/'sycl/src/program/generate.cpp').read_text()
 needles=['int64_t reread_to = -1;', 'cache_prefix_ceiling>0 && o.prompt_cache > 0', 'const int64_t read_from = reread_to > 0 ? 0 : resume;', 'if (reread && c != nullptr)', 'conversations.best(ids,req_imgs,want_cvec,cache_prefix_ceiling)']
 require(all(n in source_text for n in needles),'Consumed source no-cache/reread contract changed')
 require('STRATA_CKPT_REREAD' not in cfg['env'],'No checkpoint reread env may be present, including value0')
 def messages(phase,index):
  return [{'role':'system','content':'Write a long numbered list. Continue until the token limit. Do not explain or end early.'},{'role':'user','content':f'{phase} independent request {index}. List the integers from 1 through 500, one item per line.'}]
 def render(msgs):return service.encode_prompt(msgs,None,{'enable_thinking':False})
 out.mkdir(parents=True,exist_ok=False);roster=[]
 for n in (2,4,6):
  warm=[messages('Warm',i) for i in range(n)];target=[messages('Target',i) for i in range(n)];api_warm=[render(m) for m in warm];api_target=[render(m) for m in target]
  # Native warmups deliberately use a real tokenizer-encoded plain prefix
  # before the exact chat template, so token0 differs from every target. This
  # is a numeric warmup input, not an API-render or natural completion claim.
  prefix=tok.encode('Warm numerical preamble.\n',parse_special=False);native_warm=[prefix+ids for ids in api_warm]
  require(prefix and all(ids[0]!=target_ids[0] for ids in native_warm for target_ids in api_target),'Native unrelated complete-prefix contract failed')
  require(all(1<=len(ids)<192 for ids in native_warm+api_warm+api_target),'Bounded ASCII corpus unexpectedly changed tokenizer geometry')
  require(all(tok.decode(tok.encode(m[1]['content'],parse_special=False))==m[1]['content'] for m in warm+target),'Actual tokenizer roundtrip failed')
  policy=[{'logical_index':i,'role':'admission','values':{'reused':0,'read_from':0,'reread_to':-1}} for i in range(n)]
  # Only index1 can survive cancellation with >=32 remaining tokens; peers
  # have8 and therefore cannot enter _may_go_solo, index0 is HTTP-canceled.
  policy.append({'logical_index':1,'role':'solo_migration','values':{'reused':0,'read_from':0,'reread_to':-1}})
  spec={'schema':1,'source_lane':lane,'slots':n,'port':18520+n,'tokens':{'warm':native_warm,'target':api_target},'messages':{'warm':warm,'target':target},'api_token_ids':{'warm':api_warm,'target':api_target},'cancel_index':0,'max_new_by_request':[64,64]+[8]*(n-2),'actual_counter_policy':policy,'counter_contract':'o.prompt_cache0 suppresses main/checkpoint/idle-slot lookups; conversation_cache_mib0 has no parked incoming; absent STRATA_CKPT_REREAD leaves reread_to=-1, read_from=resume0. Solo migration re-evaluates own exact prefix from0; no cache state-transfer reuse claim.','source_sha256':{name:sha(source/name) for name in ['sycl/src/program/generate.cpp','serve/server.py','serve/frontend.py','tools/strata_tokenizer.py']},'tokenizer_sha256':manifest['tokenizer_files'],'baseline_prepared_sha256':sha(baseline/'prepared.json'),'generator_sha256':sha(Path(__file__)),'CPU_only':True,'actual_model_forward_executed':False,'actual_GPU_executed':False,'genuine_harness_preparation_created':False}
  file=out/('case'+str(n)+'.json');write(file,spec);roster.append({'slots':n,'path':str(file),'sha256':sha(file),'native_target_lengths':[len(x) for x in api_target],'native_warm_lengths':[len(x) for x in native_warm]})
 receipt={'schema':1,'passed':True,'source':str(source),'baseline':str(baseline),'source_lane':lane,'rows':roster,'permitted_payload_reads':'only tokenizer/header/template exports; no original GGUF or backend weight payload','host_python':sys.version,'host_regex':importlib.import_module('regex').__version__,'host_jinja':importlib.import_module('jinja2').__version__,'actual_runtime_tokenizer_match_required_again_at_API_boundary':True,'no_GPU_or_actual_model_forward':True};write(out/'receipt.json',receipt);return receipt

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--baseline',type=Path,default=BASELINE);p.add_argument('--output',type=Path,required=True);p.add_argument('--lane',choices=['source29','source31'],default='source29');a=p.parse_args();result=generate(a.source,a.baseline,a.output,a.lane);print(json.dumps({'passed':result['passed'],'requested_cases':[2,4,6],'GPU_executed':False}));return 0
if __name__=='__main__':raise SystemExit(main())
