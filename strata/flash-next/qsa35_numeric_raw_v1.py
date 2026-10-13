"""Exact current QSA35 comparison targets; captures never become primary operands."""
import collections,hashlib,json,re
from pathlib import Path
import numpy as np
from serial37_canonical_json_v3 import canonical
from source59_literal1_numeric_lpdisabled_v1 import numeric
from collect_prefix_residual30_v2 import collect as p30_collect
from collect_ple_input33_v1 import collect as ple_collect
from audit_prefix30_lifecycle_v1 import audit_text
from layer0_numerical_qualification_v10 import compare_numeric
HEADER='sycl/include/strata/core/layer3_qsa_target_observer.hpp'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def consume(path,expected_bytes,expected_sha):
 p=Path(path);require(p.resolve()==p and p.is_file()and not p.is_symlink(),'Confined canonical capture required');before=p.stat();require(before.st_size==expected_bytes<=4<<20,'Exact bounded target extent required');data=p.read_bytes();require(len(data)==expected_bytes and sha(p)==expected_sha and p.stat()==before,'Captured bytes/stat changed');return data

def fields(actor):
 sdk=actor.parent.fresh;proof=sdk.binding();require(proof['source_successor']=='qsa35_session_math_v1','Exact new QSA35 SDK proof required');path=sdk.SDK/'source'/HEADER;want=actor.c.read(sdk.PLAN)['expected_patched_source_sha256'][HEADER];require(sha(path)==want,'Actual QSA35 header changed');rows=re.findall(r'\{"([A-Za-z0-9_]+)",(\d+),"([^"]+)",(true|false)\}',path.read_text());require(len(rows)==29 and len({r[0]for r in rows})==29,'Exactly original29 source field layouts required');return {n:dict(bytes=int(b),encoding=e,rows=flag=='true')for n,b,e,flag in rows}
def frame(path,binding,spec):
 raw=Path(path).read_bytes();require(len(raw)<1<<20,'Metadata bound');f=json.loads(raw);require(all(type(f[k])is int for k in('schema','layer','pid','request','first_position','rows'))and type(f['owner'])is bool,'Typed source coordinates required');require(f['schema']==1 and f['binding_sha256']==binding and f['layer']==35 and f['pid']==1 and f['request']in range(1,5),'Actual layer35/PID/request/source required');stage=f['stage'];require(type(stage)is int and stage in(0,1)and(f['lb'],f['le'])==((0,24)if stage==0 else(24,48)) and f['owner']is(stage==1),'Actual 24/24 stage1 ownership required');require(f['route']=='verifier'and f['first_position']==0 and f['rows']==1 and f['max_cells']==4096 if f['owner'] else f['route']=='verifier'and f['first_position']==0 and f['rows']==1,'Exact first literal verifier frame required');require(f['nonce']==(1<<32)|(f['request']<<24)|(1<<8)|(stage+1),'Actual request/stage nonce differs');require(f['device_nonce_observed']is f['owner'],'Owner GPU nonce/nonowner acknowledgment required');require(f['qsa_ordinal_global']==8 and f['qsa_ordinal_local']==(35-f['lb'])//4,'Actual recorded late QSA ordinals differ')
 for key in('captures_are_math_inputs','full_model_math_qualified','internal_norm_argument_or_rsqrt_observed','internal_attention_score_softmax_observed','forward_state_is_committed_state','unused_padding_is_native_math_target'):require(f[key]is False,'Target scope cannot become numerical/committed-state authority')
 expected=spec if f['owner']else{};require(len(f['fields'])==len(expected)and {x['name']for x in f['fields']}==set(expected),'Unique complete original29/zero quota required');payload={};views={}
 for x in f['fields']:
  s=spec[x['name']];require(x['bytes']==s['bytes']and x['encoding']==s['encoding']and Path(x['file']).name==x['file'],'Exact original field extent/encoding/path required');p=Path(path).parent/x['file'];data=consume(p,x['bytes'],sha(p));payload[x['name']]=data;views[x['name']]=dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),encoding=x['encoding'])
 if f['owner']:
  require(f['n_pages']>0 and f['n_slots']==f['n_pages'] and f['indexer_before_active_pooled_rows']==0 and f['indexer_after_active_pooled_rows']==1,'Actual fresh resident-state bounds differ')
  require(np.frombuffer(payload['page_before'],dtype='<i4').tolist()==[0]and np.frombuffer(payload['page_resolved'],dtype='<i4').tolist()==[0],'Actual physical page0 mapping required');require(np.frombuffer(payload['step'],dtype='<i4').tolist()==[0,1,0,1]and np.frombuffer(payload['selected_ids_capacity4'],dtype='<i4')[0]==0,'Actual one-cell causal selection required; padding not used')
  for name,data in payload.items():
   if spec[name]['encoding'].startswith('LE_F32'):require(np.isfinite(np.frombuffer(data,dtype='<f4')).all(),'Active native F32 target nonfinite')
   if name in('K_physical_page0_after_resolve','V_physical_page0_after_resolve'):require(np.isfinite(np.frombuffer(data,dtype='<f2').reshape(2,4,256)[:,0]).all(),'Actual active FP16 cell nonfinite')
 require(Path(path).read_bytes()==raw,'Metadata changed');return dict(metadata=str(path),metadata_sha256=hashlib.sha256(raw).hexdigest(),frame=f,fields=views,actual_one_causal_cell_observed=f['owner'],captures_are_primary_math_inputs=False,unused_padding_compared_as_math=False)
def collect(directory,binding,actor):
 directory=Path(directory).resolve();c=actor.c;recipe=c.read(directory/'recipe.json');require(recipe['QSA35_enabled']is True and recipe['NUM33_enabled']is False and recipe['native_env']['STRATA_HC_EXPLICIT_SQRT_DIVIDE']=='1','Actual QSA35/sqrt1/noNUM33 required');spec=fields(actor);rows=c.read(directory/'requests.json');actor.requests(rows,recipe['requests']);collected=[];expected_sfd=set()
 for r in rows:
  require(canonical(r['raw'])==canonical(c.read(directory/('request-%d.json'%r['ordinal']))),'Actual individual request copy differs');meta=numeric(r['raw'],directory,[(0,24),(24,48)]);heads=[x for x in meta['vectors']if x['phase']=='first_logits_before_sampler'];require(len(heads)==1,'One full original first head required');h=heads[0];values=np.frombuffer(consume(h['path'],248320*4,h['sha256']),dtype='<f4');require(np.isfinite(values).all()and values[r['raw']['output_ids'][0]]==np.max(values),'Complete finite head/emitted argmax required');collected.append(dict(ordinal=r['ordinal'],raw=r['raw'],meta=meta));expected_sfd.update(Path(x['path']).name for x in meta['vectors'])
 actual={r['ordinal']:[248045]for r in rows};stages={0:(0,24),1:(24,48)};p30=p30_collect(directory/'p30',actual,stages,binding,directory/'engine.combined.log');ple=ple_collect(directory/'ple-input',actual,binding,directory/'engine.combined.log',(0,24));log=(directory/'engine.combined.log').read_text();life=audit_text(log,stages,True);require(life['passed']and len(p30['frames'])==8 and len(ple['frames'])==4,'Full paired P30/PLE/lifetime census required');p30_names=set();ple_names=set();joins=[]
 for record in p30['frames']:
  p30_names.add(Path(record['metadata']).name);require(record['binding']['pid']==1,'P30 actual nativePID1');owner=next(r for r in collected if r['ordinal']==record['binding']['request']);resid={int(x['layer']):x for x in owner['meta']['residuals']}
  for x in record['fields']:
   consume(x['path'],40960,x['sha256']);p30_names.add(Path(x['path']).name)
   if x['phase']=='ffn':require(x['sha256']==resid[x['layer']]['sha256'],'Actual same-request P30/SFD finalFFN differs');joins.append((owner['ordinal'],x['layer']))
 for record in ple['frames']:
  ple_names.add(Path(record['metadata']).name)
  for x in record['fields'].values():consume(x['path'],Path(x['path']).stat().st_size,x['sha256']);ple_names.add(Path(x['path']).name)
 require(len(joins)==len(set(joins))==192,'All192 P30/SFD same-owner joins required');targets=[frame(p,binding,spec)for p in sorted((directory/'qsa').glob('*.json'))];require(len(targets)==8 and {(x['frame']['request'],x['frame']['stage'])for x in targets}=={(r,s)for r in range(1,5)for s in(0,1)},'Four exact owner+nonowner QSA35 frames required');qsa_names={Path(x['metadata']).name for x in targets}|{Path(f['path']).name for x in targets for f in x['fields'].values()}
 for name,want in [('captures',expected_sfd),('p30',p30_names),('ple-input',ple_names),('qsa',qsa_names)]:require({p.name for p in(directory/name).iterdir()}==want,'Exact capture roster differs '+name)
 require(not list((directory/'layer0').glob('*')),'NUM33OFF emitted layer0 targets');markers=[x for x in log.splitlines()if x.startswith('QSA3 ')];quota=sum(x['bytes']*(2 if x['rows']else 1)for x in spec.values())+16
 for stage,(lb,le)in stages.items():
  owner=int(stage==1);a=f'QSA3 allocation pid=1 stage={stage} lb={lb} le={le} owner={owner} bytes={quota if owner else 0} target_only=1';b=f'QSA3 release_returned pid=1 stage={stage} owner={owner}';require(markers.count(a)==markers.count(b)==1,'Typed observer allocation/release required')
 for x in targets:
  f=x['frame'];name=Path(x['metadata']).name;mark=f"QSA3 frame pid=1 request={f['request']} stage={f['stage']} owner={int(f['owner'])} route=verifier p0=0 rows=1 fields={29 if f['owner']else 0} nonce={f['nonce']} metadata=/results/qsa/{name}";require(markers.count(mark)==1,'Exact source frame marker required')
 require(len(markers)==12,'No stale or extra QSA markers');repeat=[compare_numeric(collected[0],r)for r in collected[1:]];owner=[x for x in targets if x['frame']['owner']]
 def active_images(x):
  result={}
  for name,f in x['fields'].items():
   data=consume(f['path'],f['bytes'],f['sha256'])
   if name.endswith('physical_page0_before'):continue
   if name in('K_physical_page0_after_resolve','V_physical_page0_after_resolve'):data=np.frombuffer(data,dtype='<f2').reshape(2,4,256)[:,:1].copy().tobytes()
   elif name=='selected_ids_capacity4':data=data[:4]
   elif name=='indexer_before':continue
   elif name=='indexer_after':data=data[:1536]+data[2048:2560]+data[3072:3076] # exclude dead128 and inactive pooled slot
   result[name]=hashlib.sha256(data).hexdigest()
  return result
 require(all(active_images(x)==active_images(owner[0])for x in owner),'QSA active fields repeat bytes differ')

 return dict(schema=1,source_generation=62,source_successor='qsa35_session_math_v1',requests=collected,P30=p30,PLE_input33=ple,P30_SFD_lifetime=life,QSA35=targets,full49_repeat_comparisons=repeat,actual_QSA_fields=29,owner_frames=4,nonowner_zero_frames=4,numeric_pass_assigned=False,full_model_math_qualified=False,captures_are_primary_math_inputs=False,unused_padding_compared_as_math=False)
