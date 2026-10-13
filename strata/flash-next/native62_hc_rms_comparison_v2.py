"""Immutable59 baseline and independent own48 targets; comparison-only authority."""
from pathlib import Path
import numpy as np
from serial37_canonical_json_v3 import read_unique,canonical
from native_qsa41_targets_v1 import consume as bounded_consume
from compare_current_full48_num10_v1 import metrics
import hashlib
ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')
BASE=ROOT/'source59-mode0-fidelity-v3-run-v1';PUBLIC=ROOT/'source59-mode0-fidelity-v3-recovered-component-v1.json';OWN=ROOT/'current59-full48-mode0-ownstate-reference-v3'
PRODUCER_SHA='2efcaa63076d6421ea3476e29aea62c44e38809d2199c42ca23d6a54ae947bc8';PUBLIC_SHA='0df16e374eb0fbbd5883292475ac369c4e6fb5e5ae7a2d406d0fad7e04431d41';OWN_SHA='d9fabca3bca0e9d1fd94d507820c0bf179c65eb405d2856a46c47f29aefdc093'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def consume(path,limit,expected):
 path=Path(path);require(sha(path)==expected,'Exact consumed comparison artifact SHA required');data=bounded_consume(path,limit);require(hashlib.sha256(data).hexdigest()==expected and sha(path)==expected,'Comparison artifact changed during consume');return data

def tensor_map(raw):
 rows={}
 def put(key,value):
  require(key not in rows,'Duplicate raw tensor key');rows[key]=value
 for r in raw['requests']:
  ordinal=r['ordinal'];require(r['raw']['ids']==[248045]and r['raw']['fresh']==1 and r['raw']['pin']is None,'Actual literal/fresh baseline scope required')
  for f in r['layer0']['observed']:put((ordinal,'NUM33',f['name']),f)
  for f in r['meta']['vectors']:put((ordinal,'SFD',f['phase'],f['layer']),f)
 for frame in raw['P30']['frames']:
  for f in frame['fields']:put((frame['binding']['request'],'P30',f['layer'],f['phase']),f)
 return rows

def baseline(raw):
 require(sha(BASE/'report.json')==PRODUCER_SHA and sha(PUBLIC)==PUBLIC_SHA,'Exact authenticated historical comparison targets required');report=read_unique(BASE/'report.json');pub=read_unique(PUBLIC);require(report['passed']is False and pub['passed']is True and pub['original_producer_passed']is False and pub['producer_status_rewritten']is False and pub['source_generation']==59 and pub['materialized_mode']==0,'Historical component scope must remain unchanged');old=tensor_map(pub['current_mode0_targets']);new=tensor_map(raw);require(set(old)==set(new),'Complete original33/all49/all144 phase roster required');pins={BASE/'report.json':PRODUCER_SHA,PUBLIC:PUBLIC_SHA}
 try:
  for key,v in old.items():
   p=Path(v['path']);require(p.resolve()==p and p.is_relative_to(BASE)and not p.is_symlink(),'Historical target path escaped');require(report['artifacts'][str(p.relative_to(BASE))]==v['sha256'],'Original producer target join differs');data=consume(p,4<<20,v['sha256']);pins[p]=v['sha256'];require(new[key]['sha256']==v['sha256'],'Combined OFF tensor is not bitwise original59 '+str(key))
  for r,o in zip(raw['requests'],pub['current_mode0_targets']['requests']):require(r['raw']['output_ids']==o['raw']['output_ids']and r['raw']['done'].split()[0:3]+r['raw']['done'].split()[5:]==o['raw']['done'].split()[0:3]+o['raw']['done'].split()[5:],'Original OFF output/finish differ')
  return {'all33_and49_and144_phase_tensors_bitwise':True,'tensors_checked':len(old),'original_producer_passed':False,'historical_current_model_health_source_authority_transferred':False,'comparison_targets_only':True,'producer_sha256':PRODUCER_SHA,'recovered_component_sha256':PUBLIC_SHA}
 finally:
  for p,w in pins.items():require(sha(p)==w,'Consumed historical target changed '+str(p))

def independent_metrics(raw):
 require(sha(OWN/'report.json')==OWN_SHA,'Exact independent own full48 execution required');own=read_unique(OWN/'report.json');require(own['passed']is True and own['errors']==[]and own['captured_inputs_used']is False and own['materialized_mode']==0 and own['numeric_pass_assigned']is False,'Independent own-state source/compute scope differs');pins={OWN/'report.json':OWN_SHA};fields={};head=None
 try:
  for l in range(48):
   for phase in('input','attention','ffn'):
    name='own-l%d-%s'%(l,phase);d=own['arrays'][name];require(d['shape']==[4,2560]and d['bytes']==40960,'Independent phase shape differs');p=OWN/(name+'.f32');require(own['artifacts'][p.name]==d['sha256'],'Original own array/report join differs');data=consume(p,40960,d['sha256']);pins[p]=d['sha256'];fields[(l,phase)]=np.frombuffer(data,dtype='<f4').reshape(4,2560).copy()
  name='own-first-head';d=own['arrays'][name];p=OWN/(name+'.f32');require(d['shape']==[248320]and d['bytes']==248320*4 and own['artifacts'][p.name]==d['sha256'],'Independent complete head shape/join differs');head=np.frombuffer(consume(p,248320*4,d['sha256']),dtype='<f4').copy();pins[p]=d['sha256'];comparisons=[]
  for r in raw['requests']:
   native={}
   for frame in raw['P30']['frames']:
    if frame['binding']['request']==r['ordinal']:
     for f in frame['fields']:native[(f['layer'],f['phase'])]=np.frombuffer(consume(f['path'],40960,f['sha256']),dtype='<f4').reshape(4,2560)
   require(set(native)==set(fields),'All48/3phase native rows required');h=r['meta']['logits'][0];actualhead=np.frombuffer(consume(h['path'],248320*4,h['sha256']),dtype='<f4');comparisons.append({'request':r['ordinal'],'layers':[{'layer':l,**{k:metrics(fields[(l,k)],native[(l,k)])for k in('input','attention','ffn')}}for l in range(48)],'head':metrics(head,actualhead)})
  return {'independent_own_report_sha256':OWN_SHA,'comparisons':comparisons,'unsupported_reference_contracts':own['unsupported'],'numeric_pass_assigned':False,'full_model_math_qualified':False,'native_operands_used_as_primary':False,'original_own_states_and_weights':True}
 finally:
  for p,w in pins.items():require(sha(p)==w,'Consumed independent own array changed '+str(p))
