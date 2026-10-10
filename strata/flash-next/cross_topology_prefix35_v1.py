#!/usr/bin/env python3
"""Read-only source35/P30V4 topology control; no tolerance, math or speed claim.
Actual execution reads finalized diagnostic arrays and source-page guards only.
No bulk model-weight scan, original reference computation or GPU workload.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import validate_prefix_residual30_final_v4 as admission
from collect_prefix_residual30_v2 import window_nonce,ROUTES,PHASES
from explore_full48_original_prefix1_v4 import compare_observation

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREFIXES=(1,2,4,8)
sha,read,require=admission.sha,admission.read,admission.require


def source_binding():
 plan=read(HERE/'cross-topology-prefix35-source-plan-v1.json')
 for name,digest in plan['files'].items():require(sha(ROOT/name)==digest,'Frozen cross-topology dependency changed '+name)
 return plan['files']


def matched_plans(left,right,left_env,right_env,left_chain,right_chain):
 require(left['cards']==[0] and right['cards']==[0,1],'Exact onecard then split32/16 pair required')
 for p in (left,right):
  admission.ctrl.base.geometry_contract(p['args'])
  require(all(p['args'].count(flag)==1 for flag in admission.ctrl.base.GEOMETRY),'Duplicate geometry argument')
  require(p['prefixes']=={str(n):p['tokens']['cases']['A'][:n] for n in PREFIXES} and p['max_new']==1 and p['primary_model_name']=='hotschmoe-dd','Exact accepted prefix input/model geometry differs')
 require(not any(flag in left['args'] for flag in ('--layer-split','--split-device','--trim-stage-weights')),'Onecard topology contains split/trim')
 for flag,value in (('--layer-split','32'),('--split-device','1')):
  require(right['args'].count(flag)==1 and right['args'][right['args'].index(flag)+1]==value,'Exact paired split topology differs')
 require(right['args'].count('--trim-stage-weights')==1,'Pair trim absent/duplicated')
 normalize=lambda args:admission.ctrl.base.topology_args(args)
 require(normalize(left['args'])==normalize(right['args']),'Arguments differ beyond topology split/device/trim')
 for key in ('image','engine_receipt_sha256','source_plan_sha256','driver_sha256','token_fixture_sha256','tokens','prefixes','dependency_sha256','transport'):
  require(left[key]==right[key],'Matched source/image/fixture differs '+key)
 require(Path(left['engine_root']).resolve()==Path(right['engine_root']).resolve() and left_chain['generation']==right_chain['generation'],'Actual same SDK/source generation required')
 require(left_chain['generation'].get('prompt_verifier_P30_capture35') is True,'Source35 compiled marker absent')
 require(left_env.get('ZE_AFFINITY_MASK')=='0' and right_env.get('ZE_AFFINITY_MASK')=='0,1','Exact topology affinity differs')
 clean=lambda env:{k:v for k,v in env.items() if k!='ZE_AFFINITY_MASK'}
 require(clean(left_env)==clean(right_env) and 'STRATA_VERIFY_EAGER' not in left_env and 'STRATA_VERIFY_EAGER' not in right_env,'Effective math environment differs/normal graph EAGER enabled')
 return {'common_args':normalize(left['args']),'common_env':clean(left_env),'engine_receipt_sha256':left['engine_receipt_sha256'],'source_plan_sha256':left['source_plan_sha256'],'image':left['image'],'geometry':'CTX2048/prefill64/PLE65536/FP16KV','topologies':[[0],[0,1]],'pair_layer_split':[32,16]}


def raw_vector(field,root,rows,floats):
 path=Path(field['path']);require(path.resolve().parent==root.resolve() and not path.is_symlink(),'Raw field path escaped admitted arm')
 require(field['bytes']==rows*floats*4 and sha(path)==field['sha256'],'Current raw SHA/extent changed')
 raw=path.read_bytes();require(len(raw)==field['bytes'] and hashlib.sha256(raw).hexdigest()==field['sha256'],'Raw read changed after admission')
 value=np.frombuffer(raw,dtype='<f4').reshape(rows,floats);require(np.isfinite(value).all(),'Raw nonfinite array');return value


def phase_rows(capture,requests,stages,root):
 require([row['prefix'] for row in requests]==list(PREFIXES),'Four actual prefixes missing/duplicated')
 result={};seen_fields=set();by_request={int(row['meta']['logits'][0]['request']):row for row in requests}
 require(len(by_request)==4,'Duplicate actual request identity')
 for frame in capture['frames']:
  d=frame['binding'];require(d.get('schema')==2 and d['route'] in ROUTES and d['nonce']==window_nonce(d['pid'],d['request'],d['first_position'],d['rows'],d['route']),'Actual schema2 window association differs')
  require(d['request'] in by_request and stages.get(d['stage'])==(d['lb'],d['le']),'Actual request/stage ownership differs')
  row=by_request[d['request']];n=row['prefix'];p=d['first_position'];count=d['rows'];ids=row['raw']['ids'];head=row['meta']['logits'][0]
  require(len(ids)==n and d['gen_ids']==ids and d['pid']==int(head['pid']) and d['row_coverage_complete'] is True and d['normal_dispatch']==ROUTES[d['route']][1],'Actual request/PID/token/dispatch binding differs')
  require(1<=count<=8 and 0<=p and p+count<=n,'Actual prefix row extent differs')
  require((d['route']=='verifier' and count==1 and p==n-1) or (d['route']!='verifier' and p+count<=n-1 and (d['route']!='prompt_verifier' or count in (1,2))),'Actual final/earlier window association differs')
  require(d['row_floats']==10240 and d['source_extent_bytes']==count*40960,'Actual source matrix geometry differs')
  for field in frame['fields']:
   layer,phase=field['layer'],field['phase'];require(d['lb']<=layer<d['le'] and phase in PHASES and field['encoding']=='LE_F32[rows,4,2560]','Layer/phase/layout differs')
   require(field['path'] not in seen_fields,'Duplicate raw field identity');seen_fields.add(field['path']);value=raw_vector(field,root,count,10240)
   for offset in range(count):
    key=(n,layer,phase,p+offset);require(key not in result,'Duplicate absolute prefix row');result[key]=(value[offset],{'path':field['path'],'sha256':field['sha256'],'matrix_rows':count,'matrix_row':offset,'absolute_position':p+offset,'stage':d['stage'],'nonce':d['nonce'],'route':d['route']})
 expected={(n,l,phase,p) for n in PREFIXES for l in range(48) for phase in PHASES for p in range(n)}
 require(set(result)==expected,'All2160 actual absolute prefix rowvectors required');return result


def compare_runs(left_root,right_root):
 dependencies=source_binding();left=admission.finalized_binding(left_root);right=admission.finalized_binding(right_root)
 lp,rp=left[2],right[2];le=read(Path(lp['prepared'])/'server-config.json')['env'];re=read(Path(rp['prepared'])/'server-config.json')['env'];matched=matched_plans(lp,rp,le,re,left[3]['chain'],right[3]['chain'])
 maps=[];requests=[]
 for root,admitted in ((left_root,left),(right_root,right)):
  rows=read(root/'child/p30_on/requests.json');requests.append(rows);stages=dict(enumerate(admission.ctrl.expected_stage_ranges(admitted[2]['args'])));maps.append(phase_rows(admitted[3]['capture'],rows,stages,root/'child/p30_on/p30'))
 pairs=[]
 for key in sorted(maps[0]):
  lv,lb=maps[0][key];rv,rb=maps[1][key];pairs.append({'prefix':key[0],'layer':key[1],'phase':key[2],'absolute_position':key[3],'left':lb,'right':rb,**compare_observation(lv,rv)})
 heads=[]
 for lr,rr in zip(*requests):
  require(lr['raw']['ids']==rr['raw']['ids']==lp['prefixes'][str(lr['prefix'])] and lr['prefix']==rr['prefix'],'Cross-topology accepted GEN IDs differ')
  lh,rh=lr['meta']['logits'][0],rr['meta']['logits'][0];lv=raw_vector({**lh,'bytes':int(lh['bytes'])},left_root/'child/p30_on/captures',1,248320);rv=raw_vector({**rh,'bytes':int(rh['bytes'])},right_root/'child/p30_on/captures',1,248320)
  heads.append({'prefix':lr['prefix'],'left':lh,'right':rh,'generated_ids_equal':lr['raw']['output_ids']==rr['raw']['output_ids'],**compare_observation(lv,rv)})
 require(len(pairs)==2160 and len(heads)==4,'Complete cross-topology comparison count differs')
 allpairs=heads+pairs
 return {'schema':1,'comparison_completed':True,'scope':'Matched native source35 cross-topology observation control; no original reference or math/speed qualification','bindings':{'dependencies':dependencies,'left':left[3],'right':right[3]},'matched':matched,'heads':heads,'rowvectors':pairs,'counts':{'full_heads':4,'rowvectors':2160},'all_bitwise_equal':all(row['bitwise_equal'] for row in allpairs),'first_bitwise_difference':next(({'kind':'head' if i<4 else 'rowvector',**row} for i,row in enumerate(allpairs) if not row['bitwise_equal']),None),'numeric_pass_claim':False,'tolerance_gate':None,'full_model_math_qualified':False,'native_values_fed_to_original_reference':False}


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ('onecard-run-root','pair-run-root','output'):ap.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();require(not a.output.exists(),'Preserve prior evidence; choose NEW output');report={'schema':1,'comparison_completed':False,'errors':[],'numeric_pass_claim':False,'full_model_math_qualified':False,'tolerance_gate':None,'started_epoch':time.time()}
 try:report.update(compare_runs(a.onecard_run_root.resolve(),a.pair_run_root.resolve()))
 except Exception as error:report['errors'].append(str(error))
 report['finished_epoch']=time.time();a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='ascii');print(json.dumps({'comparison_completed':report['comparison_completed'],'all_bitwise_equal':report.get('all_bitwise_equal'),'errors':report['errors']}));return 1 if report['errors'] else 0
if __name__=='__main__':raise SystemExit(main())
