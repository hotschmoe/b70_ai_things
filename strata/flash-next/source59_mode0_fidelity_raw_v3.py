"""Existing NUM33/P30 source schemas on actual59; raw math targets only."""
import ast,copy,json,re
from pathlib import Path
import numpy as np
import layer0_numerical_qualification_v10 as software
import source44_num38_numeric_v2 as numeric_software
import audit_layer0_capture_lifecycle_v2 as lifetime_software
from serial37_canonical_json_v3 import canonical
import run_source59_mode0_fidelity_v3 as actor
c=actor.c;require=actor.require

def field_contract():
 sdk=actor.parent.fresh;source=sdk.SDK/'source/sycl/include/strata/core/layer0_numerical_contract.hpp';want=c.read(sdk.PLAN)['expected_patched_source_sha256']['sycl/include/strata/core/layer0_numerical_contract.hpp'];require(c.sha(source)==want,'Current59 NUM33 field contract changed')
 fields=re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',source.read_text());require(len(fields)==33 and sum(int(row[2])for row in fields)==7023048,'Current59 observed33 field/extent roster required')
 return [{'name':n,'encoding':e,'bytes':int(b),'observed':True,'provenance':'DERIVED_gpu_native_exp_from_actual_gate_up'if i in(25,29)else'actual_buffer'}for i,(n,e,b)in enumerate(fields)]
def frame(raw,directory,binding):
 node=copy.deepcopy(next(n for n in ast.parse(Path(software.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='collect_frame'))
 namespace={**vars(software),'field_contract':field_contract,'live_blob_bindings':lifetime_software.live_blob_bindings}
 exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[current59-NUM33-existing-schema-only]','exec'),namespace)
 result=namespace['collect_frame'](raw,Path(directory),binding);f=result['frame']
 require('numerical_source_generation' not in f and 'gdn_core_output_observed' not in f,'Actual current33 contract has no44 core extension; never fabricate38 fields')
 require(f['pid']==1 and all(type(f[k])is int for k in ('schema','pid','request','stage','layer','rows','position','token','reused','completed_nonce','graph_key')),'Actual directexec PID1 and typed NUM fields required')
 require(f['position']==0 and f['token']==248045 and f['reused']==0,'Actual zero-position literal source differs')
 fields={v['name']:v for v in result['observed']}
 for name,size in [('gdn_state_before',3145728),('gdn_conv_before',122880)]:
  p=Path(fields[name]['path']);data=p.read_bytes();require(len(data)==fields[name]['bytes']==size and not any(data),'Actual each-request before-state not bytezero')
 return result

def collect(directory,binding):
 directory=Path(directory).resolve();recipe=c.read(directory/'recipe.json');require(recipe['engine_generation']==59 and type(recipe['materialized_mode'])is int and recipe['materialized_mode']==0 and recipe['NUM33_enabled']is True,'Actual59 mode0 observed descriptor required')
 rows=c.read(directory/'requests.json');actor.requests(rows,recipe['requests']);collected=[]
 for row in rows:
  require(canonical(row['raw'])==canonical(c.read(directory/('request-%d.json'%row['ordinal']))),'Original individual request changed')
  meta=numeric_software.port()[0](row['raw'],directory,[(0,24),(24,48)]);layer=frame(row['raw'],directory,binding)
  require(layer['frame']['request']==row['ordinal'] and layer['frame']['binding_sha256']==binding,'NUM ordinal/current source binding differs')
  vectors=meta['vectors'];heads=[v for v in vectors if v['phase']=='first_logits_before_sampler'];require(len(heads)==1,'Exact full first head required')
  head=heads[0];data=Path(head['path']).read_bytes();values=np.frombuffer(data,dtype='<f4');require(len(values)==248320 and np.isfinite(values).all()and len(data)==248320*4 and c.sha(Path(head['path']))==head['sha256'],'All full first logits finite/extent/hash required')
  require(values[row['raw']['output_ids'][0]]==np.max(values),'Actual native first token must attain head maximum')
  collected.append({'ordinal':row['ordinal'],'raw':row['raw'],'meta':meta,'layer0':layer,'full_head_finite_and_emitted_argmax':True})
 # P30 actual raw prefix coverage is all48 input/attention/FFN per request,
 # with distinct stage0/1 owners; not only SFD postFFN signatures.
 from collect_prefix_residual30_v2 import collect as p30_collect
 from collect_ple_input33_v1 import collect as ple_collect
 from audit_prefix30_lifecycle_v1 import audit_text
 actual={r['ordinal']:[248045]for r in collected};stages={0:(0,24),1:(24,48)}
 p30=p30_collect(directory/'p30',actual,stages,binding,directory/'engine.combined.log')
 ple=ple_collect(directory/'ple-input',actual,binding,directory/'engine.combined.log',(0,24))
 log=(directory/'engine.combined.log').read_text();life=audit_text(log,stages,True);require(life['passed'],'Actual current P30/SFD ownership/lifetime failed')
 require(len(p30['frames'])==8 and len(ple['frames'])==4,'Exact four-request paired P30 and PLE rosters required')
 # Exact raw file rosters, original owner/last-FFN joins and immutable extents.
 from source41_model_cli_contract_v1 import consume
 expected_num=set();expected_p30=set();expected_ple=set();expected_sfd=set();joins=[]
 for row in collected:
  require(int(row['meta']['logits'][0]['pid'])==row['layer0']['frame']['pid'] and int(row['meta']['logits'][0]['request'])==row['ordinal'],'SFD/NUM actual owner/ordinal differs')
  for vector in row['meta']['vectors']:expected_sfd.add(Path(vector['path']).name)
  record=row['layer0'];expected_num.add(Path(record['metadata']).name)
  for field in record['observed']:
   p=Path(field['path']);data=consume(p,4<<20);require(c.sha(p)==field['sha256'] and len(data)==field['bytes'],'Exact current NUM consumed bytes changed');expected_num.add(p.name)
 for record in p30['frames']:
  expected_p30.add(Path(record['metadata']).name);consume(record['metadata'],1<<20);identity=record['binding'];require(identity['pid']==1,'Actual P30 owner must equal native PID1')
  own=next(r for r in collected if r['ordinal']==identity['request']);sfd={int(v['layer']):v for v in own['meta']['residuals']}
  for field in record['fields']:
   p=Path(field['path']);data=consume(p,81920);require(c.sha(p)==field['sha256'] and len(data)==field['bytes'],'Exact current P30 consumed bytes changed');expected_p30.add(p.name)
   if field['phase']=='ffn':
    other=sfd[field['layer']];require(other['sha256']==field['sha256'],'Actual P30/SFD same-request last FFN differs');joins.append((identity['request'],field['layer']))
 for record in ple['frames']:
  expected_ple.add(Path(record['metadata']).name);consume(record['metadata'],1<<20)
  for field in record['fields'].values():
   p=Path(field['path']);data=consume(p,10240);require(c.sha(p)==field['sha256'] and len(data)==field['bytes'],'Exact current PLE consumed bytes changed');expected_ple.add(p.name)
 require(len(joins)==len(set(joins))==192 and set(joins)=={(r,l)for r in range(1,5)for l in range(48)},'Exact all192 request/layer final FFN joins required')
 for name,want in [('layer0',expected_num),('p30',expected_p30),('ple-input',expected_ple),('captures',expected_sfd)]:require({p.name for p in (directory/name).iterdir()}==want,'Exact current raw capture file roster differs '+name)
 from layer0_numerical_qualification_v10 import compare_numeric
 repetitions=[compare_numeric(collected[0],r)for r in collected[1:]]
 require(len({canonical([{'name':v['name'],'sha256':v['sha256']}for v in r['layer0']['observed']])for r in collected})==1,'Actual all33 field repeat bytes differ')
 return {'schema':1,'source_generation':59,'materialized_mode':0,'native_numerical_observer_field_count':33,'extra44_core_RMS_fields_observed':False,'engine_pid':1,'binding_sha256':binding,'requests':collected,'P30':p30,'PLE_input33':ple,'P30_SFD_lifetime':life,'full49_repeat_comparisons':repetitions,'all33_fields_repeat_bitwise':True,'full_head_finite_and_emitted_argmax':True,'full_model_math_qualified':False,'captured_values_are_comparison_targets_only':True}
