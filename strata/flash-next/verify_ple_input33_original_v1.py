#!/usr/bin/env python3
"""Original-source ngram row and IQ4_NL F32 embedding equality, not PLE/math proof."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from original_first_gdn_layer_v1 import OriginalTensorRows
from ple_owned_history_storage_v1 import ngram_rows,PleHashConsts
ROOT=Path(__file__).resolve().parents[2]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def preflight(proof):
 if proof.get('passed') is not True or proof.get('actual_matching_SFD_request_terminal_observed') is not True or len(proof.get('frames',[]))!=4:raise ValueError('Exact recollected actual4 source frames required')
 if {f['binding']['request'] for f in proof['frames']}!={1,2,3,4} or sorted(len(f['binding']['gen_ids']) for f in proof['frames'])!=[1,2,4,8]:raise ValueError('Exact source33 fourprefix identity roster required')
 for frame in proof['frames']:
  f=frame['binding'];ids=f['gen_ids'];pos=f['pos']
  if f['T']!=1 or pos!=len(ids)-1 or f['stage']!=0 or f['lb']!=0 or f['le'] not in (32,48) or f['window_tokens']!=ids[pos:pos+1] or f['prev']!=[ids[pos-2] if pos>=2 else -1,ids[pos-1] if pos>=1 else -1]:raise ValueError('Actual original-input window geometry differs')
  for name,size in [('embedding',10240),('row_ids',64)]:
   field=frame['fields'][name];path=Path(field['path'])
   if path.is_symlink() or field['bytes']!=size or path.stat().st_size!=size or sha(path)!=field['sha256']:raise ValueError('Actual input rawfield changed before original lookup')
 return True

def verify(proof,provider):
 preflight(proof);results=[]
 for frame in proof['frames']:
  f=frame['binding'];prev=list(f['prev']);wanted=[]
  for token in f['window_tokens']:wanted.extend(ngram_rows(int(token),prev,PleHashConsts()));prev=[prev[1],int(token)]
  rows=frame['fields']['row_ids'];raw=Path(rows['path']).read_bytes()
  if sha(rows['path'])!=rows['sha256'] or np.frombuffer(raw,dtype='<u4').tolist()!=wanted or f['row_ids']!=wanted:raise ValueError('Actual staged native row IDs differ from independent original hash')
  emb=frame['fields']['embedding'];actual=Path(emb['path']).read_bytes();expected=np.asarray(provider.rows('per_layer_token_embd.weight',wanted),dtype='<f4').reshape(f['T'],2560).tobytes()
  if sha(emb['path'])!=emb['sha256'] or actual!=expected:raise ValueError('Actual prelaunch embedding differs bitwise from independent original IQ4_NL rows')
  results.append({'request':f['request'],'source_rows':wanted,'row_ids_sha256':rows['sha256'],'embedding_sha256':emb['sha256'],'original_F32_embedding_sha256':hashlib.sha256(expected).hexdigest(),'bitwise_equal':True})
 if len(results)!=4 or {r['request'] for r in results}!={1,2,3,4}:raise ValueError('Exact4 original stageinput comparisons required')
 return {'schema':1,'passed':True,'results':results,'actual_original_staged_input_bitwise':True,'GPU_PLE_input_consumption_observed':False,'original_owned_PLE_history_qualified':False,'full_model_math_qualified':False,'scope':'Actual host prelaunch row IDs+F32 embedding equal independent original hash/IQ4_NL decoder. GPU consumption/key/value/norm/history/conv/outputs separate.'}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('proof','model-identity','output','directory','producer-log','requests'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--binding-sha256',required=True);p.add_argument('--le',type=int,required=True)
 a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve previous original-input proof')
 proof=json.loads(a.proof.read_bytes())
 from collect_ple_input33_v1 import collect
 requests={int(k):v for k,v in json.loads(a.requests.read_bytes()).items()};current=collect(a.directory,requests,a.binding_sha256,a.producer_log,(0,a.le));
 if current!=proof:raise ValueError('Supplied sourceproof differs from actual directory/log/request recollection')
 preflight(proof)
 provider=OriginalTensorRows(ROOT/'strata/flash-next/original-gguf-reference-foundation-plan-v1.json',a.model_identity);file,tensor,sig=provider.reader.tensors['per_layer_token_embd.weight']
 if tensor['type']!='IQ4_NL' or tensor['shape_ggml_order']!=[160,320001536] or provider.reader.files[0]['metadata']['qwen4exp.ple.eos_token_id']['value']!=248044:raise ValueError('Selected original PLE source role/constants differ')
 result=verify(proof,provider);result.update(proof_sha256=sha(a.proof),model_identity_sha256=sha(a.model_identity),original_source_binding={'path':file['path'],'type':tensor['type'],'shape':tensor['shape_ggml_order'],'absolute_offset':tensor['absolute_offset'],'packed_bytes':tensor['packed_bytes'],'stat':sig},checker_sha256=sha(Path(__file__)),collector_sha256=sha(Path(__file__).with_name('collect_ple_input33_v1.py')),producer_log_sha256=sha(a.producer_log),requests_sha256=sha(a.requests));a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii');print(json.dumps({'passed':True,'actual_original_staged_input_bitwise':True,'full_model_math_qualified':False}))
if __name__=='__main__':main()
