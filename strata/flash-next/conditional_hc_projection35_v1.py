#!/usr/bin/env python3
"""Captured-input CONDITIONAL original HC selected-projection diagnostic only.
Original full48 owned reference is untouched; native values never enter it.
"""
import argparse,array,contextlib,hashlib,io,json,math,os,struct,time,traceback
from pathlib import Path
import numpy as np
import hc_f32_arithmetic35_host_v2 as host
from original_first_gdn_layer_v1 import OriginalTensorRows
from original_gguf_vector_decoder_v2 import decode as decode_original
from explore_owned_layer0_gdn_num10_v1 import num10_admission,load_target
from explore_full48_original_prefix1_v4 import identity_admission,sha,read,write,require
from original_math_scalar import metrics
from source_page_watchdog_v3 import guard,preserve
from run_source_upload_oracle_full_v2 import full_buffered_identity
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SOURCE_PLAN=HERE/'conditional-hc-projection35-source-plan-v1.json'
PREFIX=4;POSITION=3;TILE=64<<20
ROSTER={'up':{'role':'blk.0.hc_attn_up.weight','rows':[0,1,1123,10239],'kind':'Q8_0','shape':(320,10240),'input':'attn_hc_low','native_output':'attn_hc_gate'},'inject':{'role':'blk.0.hc_attn_inject.weight','rows':[0,1,2,3],'kind':'F32','shape':(10240,4),'input':'attn_hc_normalized','native_output':'attn_hc_inject'},'down':{'role':'blk.0.hc_attn_down.weight','rows':[0,1,159,319],'kind':'Q8_0','shape':(10240,320),'input':'attn_hc_normalized','native_output':None}}
HOST_ROOT=Path('/mnt/vm_8tb/b70/build/hc-f32-arithmetic35-host-v1-container-20261010')
HOST_RUNTIME_ROOT=Path('/mnt/vm_8tb/b70/build/hc35-host-runtime-v1-20261010')
HOST_RUNTIME_REPORT_SHA='7ab01bf3efae5daac463a56b3ba620fb2ee20e883a6d104808e99c5ad17e8931'
HOST_PINS={'compile-receipt.json':'1fba58350b0fe82987bf1916e7f1a663cbadfee5ee8bad985d2cb51cf9bdbac1','analytic-receipt.json':'b05cf281b7bc1ec63a4164459194e465589b488951c974e06b9d4feb65bff1cb','v2-final-adapter-receipt.json':'425d397b0a5d6228f5990af0392132d0e9fe68c8fad3e95be2f4fb6a7ef22309','hc35-host':'d97ea12b9b805e3e8d9520ae8452dcf2a8129216e7666af6f5e44926e80b48e8'}

def source_binding():
 plan=read(SOURCE_PLAN)
 for name,want in plan['files'].items():require(sha(ROOT/name)==want,'Conditional projection frozen source changed '+name)
 host.source_binding();return plan['files']

def host_admission(helper,build_root,host_runtime_root):
 helper=Path(helper).resolve();build_root=Path(build_root).resolve();host_runtime_root=Path(host_runtime_root).resolve();require(host_runtime_root==HOST_RUNTIME_ROOT.resolve() and sha(host_runtime_root/'report.json')==HOST_RUNTIME_REPORT_SHA,'Exact finalized supplemental host runtime proof differs');require(build_root==HOST_ROOT.resolve() and helper==build_root/'hc35-host','Exact finalized helper build association differs')
 for name,want in HOST_PINS.items():require(sha(build_root/name)==want,'Finalized host compile/analytic/adapter/binary identity differs '+name)
 compiled=read(build_root/'compile-receipt.json');analytic=read(build_root/'analytic-receipt.json');adapter=read(build_root/'v2-final-adapter-receipt.json')
 require(compiled['passed'] is True and compiled['compile_return_code']==0 and compiled['source_unchanged'] is True and compiled['source_sha256']==sha(HERE/'hc_f32_arithmetic35_host_v1.cpp') and compiled['helper_sha256']==HOST_PINS['hc35-host'],'Finalized host compile proof differs')
 require(analytic['passed'] is True and analytic['compile_receipt_sha256']==HOST_PINS['compile-receipt.json'] and analytic['helper_sha256']==HOST_PINS['hc35-host'] and len(analytic['rows'])==15 and all(row['passed'] for row in analytic['rows']),'Actual15 host analytic proof absent')
 require(adapter['passed'] is True and adapter['adapter_sha256']==sha(Path(host.__file__)) and adapter['source_plan_sha256']==sha(HERE/'hc-f32-arithmetic35-host-source-plan-v2.json') and [row['label'] for row in adapter['rows']]==['fused','dot_xor','q8_signed'] and all(row['passed'] and row['receipt']['helper_sha256']==HOST_PINS['hc35-host'] and row['receipt']['host_gradual_input_output_probes_passed'] for row in adapter['rows']),'Actual final V2 adapter3 proof absent')
 for receipt in (compiled,analytic,adapter):require(receipt['GPU_touch'] is False and receipt['model_payload_read'] is False and receipt['device_intrinsics_qualified'] is False and receipt['model_math_qualified'] is False,'Host analytic evidence scope differs')
 from qualify_hc35_host_runtime_v1 import finalized_binding
 current=finalized_binding(host_runtime_root,helper,build_root)
 return {'compile_analytic_adapter_pins':HOST_PINS,'current_host_runtime_finalized_binding':current,'host_helper_sha256':HOST_PINS['hc35-host'],'full_model_math_qualified':False,'tolerance_gate':None}

def selected_cases(include_down=False):
 return {name:dict(value,rows=list(value['rows'])) for name,value in ROSTER.items() if include_down or name!='down'}

def original_role(provider,case):
 file,tensor,sig=provider.reader.tensors[case['role']];require(tensor['type']==case['kind'] and tuple(tensor['shape_ggml_order'])==case['shape'],'Exact original HC role/type/row geometry differs '+case['role'])
 return file,tensor,sig

def packed_row(provider,case,index):
 file,tensor,sig=original_role(provider,case);require(type(index) is int and index in case['rows'],'Only predeclared HC rows admitted');cols=case['shape'][0];row_bytes=cols//32*34 if case['kind']=='Q8_0' else cols*4;require(row_bytes<=TILE,'Selected original HC row exceeds bound');offset=tensor['absolute_offset']+index*row_bytes
 require(offset>=file['tensor_data_offset'] and offset+row_bytes<=file['size_bytes'],'Selected original row extent differs');fd=os.open(file['path'],os.O_RDONLY)
 try:
  def stable():
   st=os.fstat(fd);require([st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]==sig,'Original HC shard stat changed at row boundary')
  stable();raw=os.pread(fd,row_bytes,offset);stable();require(len(raw)==row_bytes,'Short original HC row read')
 finally:os.close(fd)
 decoded=np.asarray(provider.rows(case['role'],[index])[0],dtype=np.float64);require(decoded.shape==(cols,) and np.isfinite(decoded).all(),'Original decoded selected HC row differs');own_decode=np.asarray(decode_original(raw,case['kind'],'original_fp64'),dtype=np.float64).reshape(-1);require(np.array_equal(decoded,own_decode),'Packed/independent original row decoder disagreement')
 return raw,decoded,{'role':case['role'],'row':index,'kind':case['kind'],'columns':cols,'original_file':file['path'],'source_stat5':sig,'absolute_offset':offset,'packed_bytes':len(raw),'packed_sha256':hashlib.sha256(raw).hexdigest(),'original_decoded_row_sha256':hashlib.sha256(np.asarray(decoded,dtype='<f8').tobytes()).hexdigest()}

def selected_projection(provider,values,helper,helper_sha,include_down=False,runner=None):
 # Native field values are EXPLICIT conditional operands in this separate lane.
 runner=host.run_helper if runner is None else runner;result={};weights={}
 for name,case in selected_cases(include_down).items():
  operand=np.asarray(values[case['input']],dtype=np.float64).reshape(-1);require(operand.shape==(case['shape'][0],) and np.isfinite(operand).all(),'Captured conditional input shape/finite differs');input_bytes=np.asarray(operand,dtype='<f4').tobytes();require(np.array_equal(operand,np.frombuffer(input_bytes,dtype='<f4').astype(np.float64)),'Captured conditional operand must be exact F32 widened');f32_out=[];f64_out=[];bindings=[];calls=[]
  for index in case['rows']:
   packed,decoded,identity=packed_row(provider,case,index);op='q8dot' if case['kind']=='Q8_0' else 'dot';raw,receipt=runner(helper,helper_sha,op,packed,input_bytes);require(isinstance(raw,bytes) and len(raw)==4,'Host selected-output dtype/extent must be scalar LE_F32');require(receipt.get('helper_sha256')==helper_sha and receipt.get('input_sha256')==[hashlib.sha256(packed).hexdigest(),hashlib.sha256(input_bytes).hexdigest()] and receipt.get('output_sha256')==hashlib.sha256(raw).hexdigest(),'Host scalar operand/output/helper source receipt differs');require(receipt.get('host_gradual_input_output_probes_passed') is True and receipt.get('implementation_arithmetic_qualified') is False and receipt.get('device_intrinsics_qualified') is False and receipt.get('model_math_qualified') is False and receipt.get('tolerance_gate') is None,'Host scalar operation scope/probes differ');value=struct.unpack('<f',raw)[0];require(math.isfinite(value),'Nonfinite host selected output');f32_out.append(value);f64_out.append(float(decoded@operand));bindings.append(identity);calls.append(receipt);weights[(name,index)]=packed
  f32_result=np.asarray(f32_out,dtype='<f4');f64_result=np.asarray(f64_out,dtype='<f8');require(np.isfinite(f64_result).all(),'Nonfinite conditional F64 mathematical lane');native=None
  if case['native_output'] is not None:
   native_all=np.asarray(values[case['native_output']],dtype='<f4').reshape(-1);require(native_all.shape==(case['shape'][1],) and np.isfinite(native_all).all(),'Native selected output source extent/finite differs');native=native_all[case['rows']].copy()
  comparisons={}
  if native is not None:
   for lane,value in [('source32_FMA_XOR_F32',f32_result),('original_sameinput_F64',f64_result)]:comparisons[lane]={**metrics(value.astype(np.float64),native.astype(np.float64)),'F32_bytes_equal':np.asarray(value,dtype='<f4').tobytes()==native.tobytes(),'numeric_gate_assigned':False,'tolerance_gate':None}
  result[name]={'role':case['role'],'rows':case['rows'],'input_field':case['input'],'captured_inputs_used':True,'scope':'CONDITIONAL actualnative operand; selected originalweights only; no original-owned reference attribution','source32_F32_output':f32_result,'mathematical_F64_output':f64_result,'selected_native_output':native,'conditional_input_F32_bytes':input_bytes,'original_rows':bindings,'host_calls':calls,'comparisons':comparisons,'native_output_observed':native is not None,'down_preSiLU_vs_postSiLU_comparison_forbidden':name=='down','full_model_math_qualified':False,'numeric_pass_claim':False}
 return result,weights

def save_raw(output,label,raw,encoding,shape):
 path=output/label;require(not path.exists(),'Preserve conditional raw evidence');path.write_bytes(raw);return {'path':str(path),'sha256':sha(path),'bytes':len(raw),'encoding':encoding,'shape':list(shape)}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--native-run-root',type=Path,required=True);ap.add_argument('--model-identity',type=Path,required=True);ap.add_argument('--helper',type=Path,required=True);ap.add_argument('--helper-build-root',type=Path,required=True);ap.add_argument('--host-runtime-root',type=Path,required=True);ap.add_argument('--include-down',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 report={'schema':1,'status':'incomplete','started_epoch':time.time(),'scope':'CONDITIONAL suppliednative normalized/low operands; originalowned math unchanged','captured_inputs_used':True,'full_model_math_qualified':False,'numeric_pass_claim':False,'tolerance_gate':None,'prefix':PREFIX,'position':POSITION,'errors':[]};lock_path=HERE/'model-lock.json';lock=read(lock_path);shards=[ROOT/lock['destination']/row['path'] for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')];admitted=False;terminal=None
 def watch(label):
  try:return guard(shards[2])
  except Exception:report['preserved_'+label]=preserve(shards[2],out,label);raise
 try:
  report['dependency_sha256']=source_binding();report['initial_known_pages']=watch('initial-failure');report['host_admission']=host_admission(a.helper,a.helper_build_root,a.host_runtime_root);parent,child,plan,binding=num10_admission(a.native_run_root);report['native_finalized_binding']=binding;boundary=max(parent['child_terminal_epoch'],child['finished_epoch'],binding['posthealth_finished']);report['pre_original_identity']=identity_admission(a.model_identity,lock_path,shards,boundary);report['pre_known_pages']=watch('pre-failure');ids,values,_,target=load_target(a.native_run_root,plan,PREFIX,binding['requests']);report['native_observation_binding']=target;require(len(ids)==PREFIX and target['prefix']==PREFIX,'Exact prefix4row3 target required');report['case_roster']=selected_cases(a.include_down);write(out/'report.json',report);admitted=True
  numpy_config=io.StringIO()
  with contextlib.redirect_stdout(numpy_config):np.show_config()
  report['mathematical_F64_runtime']={'numpy_version':np.__version__,'numpy_module_sha256':sha(np.__file__),'numpy_configuration':numpy_config.getvalue(),'numpy_binary_sha256':{str(p):sha(p) for p in sorted(Path(np.__file__).parent.rglob('*.so'))},'env':{k:v for k,v in os.environ.items() if k.startswith(('OMP_','MKL_','OPENBLAS_','NUMEXPR_'))},'numeric_runtime_qualified':False};write(out/'report.json',report)
  provider=OriginalTensorRows(HERE/'original-gguf-reference-foundation-plan-v1.json',a.model_identity);results,weights=selected_projection(provider,values,a.helper,HOST_PINS['hc35-host'],a.include_down);terminal=time.time();report['computation_terminal_epoch']=terminal;arrays={};clean={}
  for name,case in results.items():
   row=dict(case);rows=case['rows'];arrays[name+'_source32_F32']=save_raw(out,name+'-source32-output.f32',case['source32_F32_output'].tobytes(),'LE_F32',[len(rows)]);arrays[name+'_mathematical_F64']=save_raw(out,name+'-mathematical-output.f64',case['mathematical_F64_output'].tobytes(),'LE_F64',[len(rows)]);arrays[name+'_conditional_input']=save_raw(out,name+'-actual-conditional-input.f32',case['conditional_input_F32_bytes'],'LE_F32',[len(case['conditional_input_F32_bytes'])//4])
   if case['selected_native_output'] is not None:arrays[name+'_native_selected']=save_raw(out,name+'-native-selected.f32',case['selected_native_output'].tobytes(),'LE_F32',[len(rows)])
   for key in ('source32_F32_output','mathematical_F64_output','selected_native_output','conditional_input_F32_bytes'):row.pop(key)
   clean[name]=row
  for (name,index),raw in weights.items():arrays[name+'_original_row'+str(index)]=save_raw(out,name+'-original-row'+str(index)+('.q8_0' if ROSTER[name]['kind']=='Q8_0' else '.f32'),raw,ROSTER[name]['kind'],[ROSTER[name]['shape'][0]])
  report['results']=clean;report['raw_arrays']=arrays;report['post_host_admission']=host_admission(a.helper,a.helper_build_root,a.host_runtime_root);require(report['post_host_admission']==report['host_admission'],'Pinned host runtime/helper changed');_,_,postplan,postbinding=num10_admission(a.native_run_root);require(postplan==plan and postbinding==binding,'Finalized native target changed during conditional diagnostic');source_binding();report['post_known_pages']=watch('post-failure');identity_admission(a.model_identity,lock_path,shards,boundary);write(out/'report.json',report)
 except Exception as error:
  terminal=time.time();report['errors'].append(str(error));report['traceback']=traceback.format_exc()
  if 'initial_known_pages' in report:
   try:report['failure_page_views']=preserve(shards[2],out,'conditional-admission-failure')
   except Exception as preservation_error:report['errors'].append('Page preservation: '+str(preservation_error))
 finally:
  try:
   require(admitted,'Fullscan skipped: conditional original payload admission not reached');after=terminal or time.time();report['before_full4_pages']=watch('before-full4-failure');identity=full_buffered_identity(lock_path,lock,shards,out/'post-conditional-model-identity.json',after);require(identity['passed'],'NEW postCPU publisher all4 hash failed');report['post_conditional_identity']=identity_admission(out/'post-conditional-model-identity.json',lock_path,shards,after);report['final_known_pages']=watch('final-failure')
  except Exception as error:
   report['errors'].append('Post-source proof: '+str(error))
   if admitted:
    try:report['preserved_post_failure']=preserve(shards[2],out,'post-failure')
    except Exception as preserve_error:report['errors'].append('Source preservation: '+str(preserve_error))
  report['finished_epoch']=time.time();report['status']='conditional exploratory complete; NO numerical qualification' if not report['errors'] else 'FAILED conditional source/admission/execution; preserve evidence';write(out/'report.json',report)
 print(json.dumps({'status':report['status'],'errors':report['errors'],'groups':list(report.get('results',{})),'numeric_pass_claim':False}));return 1 if report['errors'] else 0
if __name__=='__main__':raise SystemExit(main())
