"""Current43 NUM10 target adapter; original equations and conditional lanes."""
import hashlib,math,os,re,stat,time
from pathlib import Path
import numpy as np
from serial37_canonical_json_v3 import read_unique,canonical
from independent_q8_1_activation_v1 import encode,decode,NATIVE
HERE=Path(__file__).resolve().parent;PLAN=HERE/'source43-num10-original-reference-source-plan-v1.json'
SHAPES={'residual_input':(4,2560),'attn_hc_normalized':(4,2560),'attn_hc_low':(320,),'attn_hc_gate':(4,2560),'attn_hc_inject':(4,),'attn_mixed':(2560,),'gdn_state_before':(128,48,128),'gdn_conv_before':(10240,3),'gdn_qkv':(10240,),'gdn_z':(6144,),'gdn_decay_beta':(2,48),'gdn_normalized_qkv':(10240,),'gdn_output_gated':(6144,),'gdn_block_output':(2560,),'gdn_state_after':(128,48,128),'gdn_conv_after':(10240,3),'residual_after_attn':(4,2560)}
PACKETS={'attn_input_q81':2880,'gdn_output_q81':6912}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sig(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def consume(path,size,want):
 p=Path(path);require(p.is_absolute()and not any(x.is_symlink()for x in(p,*p.parents))and p.is_file(),'Nonaliased original target required');before=sig(p.stat());require(before[2]==size and size<=4<<20,'Exact bounded target extent required')
 with p.open('rb')as f:
  require(sig(os.fstat(f.fileno()))==before,'Original target descriptor differs');raw=f.read(size+1);require(sig(os.fstat(f.fileno()))==before,'Original target descriptor changed')
 require(len(raw)==size and hashlib.sha256(raw).hexdigest()==want and sig(p.stat())==before and p.read_bytes()==raw and sig(p.stat())==before,'Consumed target SHA/stat/reread changed');return raw

def source_binding():
 row=read_unique(PLAN);root=HERE.parents[1]
 for name,w in row['files'].items():require(hashlib.sha256((root/name).read_bytes()).hexdigest()==w,'Reference source changed '+name)
 return hashlib.sha256(PLAN.read_bytes()).hexdigest()
def field_contract():
 source=HERE/'layer0_numerical_contract_v1.hpp';fields=re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',source.read_text());require(len(fields)==33,'Exact NUM10 software layout required')
 return [{'name':n,'encoding':e,'bytes':int(b),'provenance':'DERIVED_gpu_native_exp_from_actual_gate_up'if i in(25,29)else'actual_buffer'}for i,(n,e,b)in enumerate(fields)]

def load_targets(binding,rows):
 require(binding['source_generation']==43 and type(binding['materialized_mode'])is int and binding['materialized_mode']==1 and binding['NUM10_enabled']is True and binding['QSA3_enabled']is False,'Actual43 mode1 NUM10-only binding required');root=Path(binding['capture_root']);pid=binding['engine_pid'];require(type(pid)is int and pid>0 and len(rows)==4,'Four actual same-engine fresh captures required');out=[];contract=field_contract();requests=[]
 for row in rows:
  raw=row['raw'];item=row['layer0'];frame=item['frame'];request=frame['request'];require(raw['ids']==[248045]and type(raw['fresh'])is int and raw['fresh']==1 and raw.get('pin')is None and type(frame['position'])is int and frame['position']==0 and frame['gen_ids']==[248045]and frame['token']==248045 and frame['rows']==1 and frame['layer']==0 and frame['stage']==0 and frame['reused']==0,'Exact fresh accepted token/zero-history capture required');require(frame['pid']==pid and type(request)is int and request>0 and request not in requests and frame['binding_sha256']==binding['plan_sha256']and frame['completed_nonce']==((pid&0xffffffff)<<32)|request,'Actual owner/request/nonce differs');requests.append(request)
  require(frame['request_replay_marker_verified']is True and frame['producer_mapping_verified']is True and frame['complete_preregistered_layout']is True and frame['graph_key']in(2,3) and frame['raw_fused_hidden_observed']is False and frame['full_model_math_qualified']is False,'Actual complete source mapping required');metadata=Path(item['metadata']);consume(metadata,metadata.stat().st_size,item['metadata_sha256']);require(metadata.parent==root/'layer0'and read_unique(metadata)==frame and hashlib.sha256(metadata.read_bytes()).hexdigest()==item['metadata_sha256'],'Actual frame metadata changed');observed=item['observed'];require(len(frame['fields'])==33 and all(all(a[k]==w[k]for k in('name','bytes','encoding','provenance'))for a,w in zip(frame['fields'],contract)),'Metadata raw33 contract differs');require(len(observed)==33 and [f['name']for f in observed]==[f['name']for f in contract],'Exact raw33 roster required');values={};packets={};roster=[]
  for f,want in zip(observed,contract):
   require(all(f[k]==want[k]for k in('name','bytes','encoding','provenance')),'Actual raw33 shape/provenance differs');path=Path(f['path']);require(path.parent==root/'layer0','Actual target path escaped');data=consume(path,f['bytes'],f['sha256']);roster.append(f)
   if f['name']in SHAPES:
    a=np.frombuffer(data,dtype='<f4').reshape(SHAPES[f['name']]).copy();require(np.isfinite(a).all(),'Finite current target required');values[f['name']]=a
   elif f['name']in PACKETS:packets[f['name']]=data
  require(set(values)==set(SHAPES)and set(packets)==set(PACKETS),'Complete HC/GDN targets required');require(not any(values['gdn_state_before'].tobytes())and not any(values['gdn_conv_before'].tobytes()),'Each actual fresh request must have zero state/conv')
  out.append({'request':request,'values':values,'packets':packets,'raw33':roster})
 return out

def metrics(own,native):
 a=np.asarray(own,dtype='<f4');b=np.asarray(native,dtype='<f4');require(a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all(),'Finite equal comparison geometry required');x=a.astype(np.float64);y=b.astype(np.float64);d=x-y
 return {'nmse':float(np.sum(d*d)/max(1e-30,np.sum(y*y))),'max_normalized':float(np.max(np.abs(d))/max(1e-6,np.max(np.abs(y)))),'bitwise_equal':a.tobytes()==b.tobytes(),'denominator':'native target; frozen original comparison convention','numeric_gate_assigned':False}

def independent_first(model):
 """Unchanged FirstGdnLayer FP64 equations, no captured argument accepted."""
 embedding=np.asarray(model.p.rows('token_embd.weight',[248045])[0],dtype=np.float64);residual=np.broadcast_to(embedding,(model.g.streams,model.g.embd)).copy();hc=model.hc_read(residual,'attn');zero=model.initial_state();require(not np.any(zero['recurrent'])and not np.any(zero['conv']),'Own initial state must be zero');after,block,gdn=model.mixer(hc['mixed'],{k:v.copy()for k,v in zero.items()});written=model.hc_write(residual,block,hc['inject']);fields={'residual_input':residual,'attn_hc_normalized':hc['normalized'],'attn_hc_low':hc['low_silu'],'attn_hc_gate':hc['gate'],'attn_hc_inject':hc['inject'],'attn_mixed':hc['mixed'],'gdn_state_before':zero['recurrent'].transpose(1,0,2),'gdn_conv_before':zero['conv'].T,'gdn_qkv':gdn['qkv'],'gdn_z':model.project('blk.0.attn_gate.weight',hc['mixed']).reshape(-1),'gdn_decay_beta':np.stack([gdn['log_decay'],gdn['beta']]),'gdn_normalized_qkv':np.concatenate([gdn[n].reshape(-1)for n in('q','k','v')]),'gdn_output_gated':gdn['gated'].reshape(-1),'gdn_block_output':block,'gdn_state_after':after['recurrent'].transpose(1,0,2),'gdn_conv_after':after['conv'].T,'residual_after_attn':written}
 return fields

def conditional(model,targets):
 """Explicit captured input diagnostics; never enter independent_first."""
 zero=model.initial_state();require(not np.any(zero['recurrent'])and not np.any(zero['conv']),'Conditional state remains own zero');after,block,gdn=model.mixer(targets['attn_mixed'].copy(),{k:v.copy()for k,v in zero.items()})
 packet=encode(np.asarray(targets['gdn_output_gated'],dtype='<f4').tobytes(),1,6144,NATIVE);activation=decode(packet,1,6144)['reconstructed_activation'][0];projected=model.project('blk.0.ssm_out.weight',activation)
 return {'native_HC_mixed_to_original_zero_GDN':{'captured_input_used':True,'captured_state_used':False,'block':block,'gated':gdn['gated'].reshape(-1),'state':after},'native_gated_to_original_Q81_projection':{'captured_input_used':True,'captured_state_used':False,'packet':packet,'output':projected,'stored_sum_used':False,'projection':'original decoded Q8_0 weights dot independently encoded Q8_1 d*codes; not native F32 reducer'}}

def evaluate(model,targets,conditional_requested=False):
 require(model.p.actual_source is True,'Actual independently admitted original provider required');primary=[];seams=[]
 for target in targets:
  own=independent_first(model);require(np.asarray(own['residual_input'],dtype='<f4').tobytes()==target['values']['residual_input'].tobytes(),'Actual zero-token embedding target differs from independently owned embedding');checks={n:metrics(v,target['values'][n])for n,v in own.items()};packets={n:encode(np.asarray(own[f],dtype='<f4').tobytes(),1,cols,NATIVE)for n,f,cols in(('attn_input_q81','attn_mixed',2560),('gdn_output_q81','gdn_output_gated',6144))};primary.append({'request':target['request'],'fields_owned':own,'comparisons':checks,'packet_exact':{n:raw==target['packets'][n]for n,raw in packets.items()},'captured_inputs_used':False,'lane':'unchanged original_first_gdn_layer_v1 FP64 equations; native storage/intrinsics unqualified'})
  if conditional_requested:seams.append({'request':target['request'],'lanes':conditional(model,target['values']),'full_model_math_qualified':False})
 return {'independent':primary,'conditional':seams,'full_model_math_qualified':False,'numeric_pass_claim':False,'tolerance_gate':None,'reference_replaced':False,'computation_terminal_epoch':time.time()}

def enclosed(root,model_factory,capture_admit,closing_scan,conditional_requested=False):
 """Caller supplies current43 public reader and actual post-CPU publisher scan."""
 source=source_binding();owner=os.getpid();before=capture_admit(root);result=None;terminal=None;errors=[]
 try:
  from original_first_gdn_layer_v1 import FirstGdnLayer
  targets=load_targets(before,before['requests']);model=model_factory();require(type(model)is FirstGdnLayer,'Unchanged original FP64 model required; storage/device variants separate');result=evaluate(model,targets,conditional_requested);terminal=result['computation_terminal_epoch']
 except BaseException as error:terminal=time.time();errors.append(str(error));raise
 finally:
  terminal=terminal or time.time()
  # Both closings are attempted even after reference or one closing failure.
  try:require(canonical(capture_admit(root))==canonical(before),'Actual43 capture changed across original computation')
  except BaseException as error:errors.append('Capture closing: '+str(error))
  try:
   identity=closing_scan(terminal);lock=read_unique(HERE/'model-lock.json');wanted=[r for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')];rows=identity['rows'];require(all(type(identity[n])in(int,float)and math.isfinite(identity[n])for n in('started','finished'))and identity['lock_sha256']==hashlib.sha256((HERE/'model-lock.json').read_bytes()).hexdigest()and identity['model_revision']==lock['revision']and identity['passed']is True and identity['started']>=terminal and identity['finished']>=identity['started']and len(rows)==len(wanted)==4,'Actual complete postCPU identity/chronology required')
   for row,w in zip(rows,wanted):require(row['passed']is True and Path(row['path'])==HERE.parents[1]/lock['destination']/w['path'] and row['sha256']==row['expected_sha256']==w['sha256']and row['bytes']==w['size']and row['stat_before']==row['stat_after']==sig(Path(row['path']).stat()),'Exact current publisher closing required')
  except BaseException as error:errors.append('Publisher closing: '+str(error))
  try:require(os.getpid()==owner and source_binding()==source,'Original source/owner changed')
  except BaseException as error:errors.append('Source closing: '+str(error))
  require(not errors,'Original reference or closing failed: '+'; '.join(errors))
 result.update(source_plan_sha256=source,producer_pid=owner,current43_capture_before=before,current43_capture_after=before,postCPU_identity=identity)
 return result
