"""Root-only original packed weight export; never imports native captured operands."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
from prepare_prefill_gdn_projection_operator_v1 import ROOT,HERE,require,sha
from original_first_gdn_layer_v1 import OriginalTensorRows
from original_grouped_quantized_reference_v1 import original_packed_rows

def export(provider,out):
 require(provider.actual_source is True,'Independent original provider required');records={}
 for role,name,shape in [('attn_qkv.weight','qkv.q80',(2560,10240)),('attn_gate.weight','z.q80',(2560,6144)),('ssm_out.weight','out.q80',(6144,2560))]:
  key='blk.0.'+role;require(provider.shape(key)==shape and provider.reader.tensors[key][1]['type']=='Q8_0','Actual original packed Q80 shape/type required '+role);b=original_packed_rows(provider.reader,key,range(shape[1]));(out/name).write_bytes(b);records[name]={'role':key,'original_type':'Q8_0','shape_ggml_order':list(shape),'storage':'unchanged_original_packed_Q8_0','bytes':len(b),'sha256':sha(out/name)}
 for role,name in [('ssm_alpha.weight','alpha.bf16'),('ssm_beta.weight','beta.bf16')]:
  key='blk.0.'+role;require(provider.shape(key)==(2560,48)and provider.reader.tensors[key][1]['type']=='F32','Original F32 AB weight required');x=np.asarray(provider.rows(key,range(48)),dtype='<f4');require(x.shape==(48,2560)and np.isfinite(x).all(),'Complete original AB rows required');u=x.view('<u4');b=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2').tobytes();(out/name).write_bytes(b);records[name]={'role':key,'original_type':'F32','shape_ggml_order':[2560,48],'storage':'original_F32_to_loader_BF16_RNE','bytes':len(b),'sha256':sha(out/name)}
 for role,name in [('ssm_dt.bias','dt.f32'),('ssm_a','ssm_a.f32')]:
  key='blk.0.'+role;require(provider.shape(key)==(48,)and provider.reader.tensors[key][1]['type']=='F32','Original F32 vector required');x=np.asarray(provider.rows(key,[0]),dtype='<f4');require(x.size==48 and np.isfinite(x).all(),'Complete original vector required');(out/name).write_bytes(x.tobytes());records[name]={'role':key,'original_type':'F32','shape_ggml_order':[48],'storage':'unchanged_original_F32','bytes':x.nbytes,'sha256':sha(out/name)}
 return records

def main(a):
 from run_prefill_gdn_projection_operator_v1 import binding
 source=binding();require(sha(a.model_identity)==a.identity_sha256,'Caller fresh complete model identity required');out=a.output.resolve();require(not out.exists(),'New original export root required');out.mkdir();r={'schema':1,'passed':False,'errors':[],'started_epoch':time.time(),'model_identity_sha256':a.identity_sha256,'source_plan_sha256':source,'native_values_used_as_primary_inputs':False,'original_weight_export_only':True,'roles':{}};pins={Path(__file__).resolve():sha(Path(__file__).resolve()),a.model_identity:a.identity_sha256}
 try:r['roles']=export(OriginalTensorRows(HERE/'original-gguf-reference-foundation-plan-v1.json',a.model_identity),out)
 except BaseException as e:r['errors'].append(type(e).__name__+': '+str(e));(out/'exception.txt').write_text(traceback.format_exc(),encoding='ascii',errors='backslashreplace')
 finally:
  try:
   require(binding()==source,'Closing export source changed')
   for p,h in pins.items():require(sha(p)==h,'Closing caller/source bytes changed')
  except BaseException as e:r['errors'].append('byte-close: '+str(e))
  try:
   from run_source44_num38_original_reference_v2 import closing_identity
   r['postCPU_full4']=closing_identity(out,time.time())
  except BaseException as e:r['errors'].append('closing-full4: '+str(e))
  r.update(passed=not r['errors'],finished_epoch=time.time(),artifacts={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='report.json'});(out/'report.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii')
 require(r['passed'],'Original export/source/closing failed');return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model-identity',type=Path,required=True);p.add_argument('--identity-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
