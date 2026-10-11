#!/usr/bin/env python3
"""Direct historical43 mode1 residual/head targets; no model reads/readmission."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from serial37_canonical_json_v3 import read_unique
import current_full48_own_state_bulk_v1 as candidate
require=candidate.require
OWN_SHA='37dba63f1dcbe8d2ffb28280c5ebf184155fa762442bc1b710e5cdf9799367b4'
NATIVE_SHA='d6713012b9d821ce7507a00d827044931bf7108501889e9df970bdf42f6d4f20'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'current-full48-num10-comparison-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def artifact(root,report,p):
 p=Path(p);require(p.is_file()and not p.is_symlink()and p.resolve()==p and sha(p)==report['artifacts'][str(p.relative_to(root))],'Original report artifact changed '+str(p));return p

def metrics(a,b):
 a=np.asarray(a,dtype='<f4');b=np.asarray(b,dtype='<f4');require(a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all(),'Finite equal target shape required');d=a.astype(np.float64)-b.astype(np.float64);y=b.astype(np.float64)
 return {'nmse':float(np.sum(d*d)/max(1e-30,np.sum(y*y))),'max_absolute':float(np.max(np.abs(d))),'max_normalized':float(np.max(np.abs(d))/max(1e-6,np.max(np.abs(y)))),'bitwise_equal':a.tobytes()==b.tobytes(),'numeric_pass_claim':False}
def array(root,report,p,shape):
 p=artifact(root,report,p);raw=p.read_bytes();require(len(raw)==int(np.prod(shape))*4 and hashlib.sha256(raw).hexdigest()==report['artifacts'][str(p.relative_to(root))],'Array shape/hash changed');return np.frombuffer(raw,dtype='<f4').reshape(shape).copy()
def execute(args):
 ownroot=args.own_root.resolve();native=args.native_root.resolve();own=read_unique(ownroot/'report.json');nr=read_unique(native/'report.json');require(sha(ownroot/'report.json')==OWN_SHA and sha(native/'report.json')==NATIVE_SHA,'Exact executed own and native reports required');require(own['passed']is True and own['errors']==[]and own['postCPU_full4']['passed']is True and own['source_generation']==43 and own['materialized_mode']==1 and own['accepted_ids']==[248045]and own['captured_inputs_used']is False and own['state_initialized_from_zero']is True,'Own executed mode1/prefix1 scope differs');require(nr['passed']is True and nr['errors']==[],'Historical native producer failed')
 oc=read_unique(artifact(ownroot,own,ownroot/'postCPU-full4.json'));nc=read_unique(artifact(native,nr,native/'post-full4.json'));require(oc==own['postCPU_full4']and nc==nr['post_full4']and nc['passed']is True and oc['lock_sha256']==nc['lock_sha256']and oc['model_revision']==nc['model_revision'],'Recorded complete identities differ')
 def roster(v):
  require(len(v['rows'])==4 and all(r['passed']is True and r['sha256']==r['expected_sha256']for r in v['rows']),'Recorded complete four shards required');return [(r['path'],r['bytes'],r['sha256'])for r in v['rows']]
 require(roster(oc)==roster(nc),'Own and target original publisher bytes differ')
 sp=read_unique(artifact(native,nr,native/'source-plan.snapshot.json'));require(sha(native/'source-plan.snapshot.json')==nr['source_plan_sha256'],'Historical original source snapshot differs');artifact(native,nr,native/'producer.snapshot.py');corpus=read_unique(artifact(native,nr,native/'plan.snapshot.json'));require(sha(native/'plan.snapshot.json')==nr['plan_sha256'],'Historical corpus differs');build=read_unique(candidate.sdk43.PLAN);require(sha(candidate.sdk43.PLAN)==candidate.sdk43.PLAN_SHA and corpus['source43']['plan_sha256']==candidate.sdk43.PLAN_SHA,'Matched actual43 source proposal differs')
 for name in candidate.SOURCE_FILES:require(corpus['source43']['current_all70_source_sha256'][name]==build['expected_patched_source_sha256'][name],'Actual recorded native math source differs '+name)
 config=read_unique(args.config);require(sha(args.config)==own['config_sha256'],'Exact own config required');candidate.schedule([248045],config['args'],config['env']);directory=native/'source43-num10-ON';recipe=read_unique(artifact(native,nr,directory/'recipe.json'));result=read_unique(artifact(native,nr,directory/'result.json'));phase=read_unique(artifact(native,nr,directory/'phase_receipt.json'));artifact(native,nr,directory/'command.json');require(recipe['engine_generation']==43 and type(recipe['engine_generation'])is int and recipe['materialized_mode']==1 and type(recipe['materialized_mode'])is int and recipe['native_argv'][2:]==config['args'],'Matched mode1/native argument geometry required')
 for key,value in config['env'].items():require(recipe['native_env'].get(key)==value,'Actual declared native math environment differs '+key)
 require(result['passed']is True and result['errors']==[]and result['error']is None and result['engine_rc']==0 and type(result['engine_rc'])is int and result['forced_cleanup']is False and result['removal']['removed']is True and phase['passed']is True and phase['error']is None and phase['result_sha256']==sha(directory/'result.json')and phase['command_sha256']==sha(directory/'command.json'),'Original recorded normal terminal/phase failed')
 rows=read_unique(artifact(native,nr,directory/'requests.json'));require([r['ordinal']for r in rows]==[1,2,3,4],'Original four target requests required');checks=[]
 for row in rows:
  ordinal=row['ordinal'];raw=row['raw'];require(raw['ids']==[248045]and type(raw['fresh'])is int and raw['fresh']==1 and raw['pin']is None and raw['stop_sent']is False,'Exact fresh literalprefix1 required');require(read_unique(artifact(native,nr,directory/('request-%d.json'%ordinal)))==raw,'Original request row differs')
  record={'request':ordinal,'layers':{str(l):{'input':None,'attention':None,'ffn':None}for l in range(48)}};seen=set()
  for line in raw['stderr']:
   if not line.startswith('SFD vector '):continue
   fields=dict(re.findall(r'(\w+)=([^ ]+)',line));layer=int(fields['layer']);kind=fields['phase'];require(fields['pid']=='1'and int(fields['request'])==ordinal and fields['stage']=='0'and fields['lb']=='0'and fields['le']=='48'and fields['pos']=='0'and fields['token']=='248045'and fields['reused']=='0','Original SFD owner/position differs');path=Path(fields['file']);require(path.parent==Path('/results/captures'),'Original SFD target escaped')
   if kind=='first_window_residual':require(0<=layer<48 and layer not in seen,'Duplicate SFD layer');seen.add(layer);record['layers'][str(layer)]['ffn']=metrics(array(ownroot,own,ownroot/('own-l%d-ffn.f32'%layer),(4,2560)),array(native,nr,directory/'captures'/path.name,(4,2560)))
   elif kind=='first_logits_before_sampler':require(layer==-1 and 'head'not in record,'Duplicate native head');record['head']=metrics(array(ownroot,own,ownroot/'own-first-head.f32',(248320,)),array(native,nr,directory/'captures'/path.name,(248320,)))
   else:raise ValueError('Unexpected SFD phase')
  require(seen==set(range(48))and 'head'in record,'Exact native full49 roster required');meta=read_unique(artifact(native,nr,directory/'layer0'/('l0q8-1-r%d-s0.json'%ordinal)));require(meta['pid']==1 and meta['request']==ordinal and meta['token']==248045 and meta['position']==0 and meta['rows']==1 and meta['binding_sha256']==nr['source_plan_sha256'],'Original NUM10 first row owner differs')
  for field in meta['fields']:
   if field['name']in('residual_input','residual_after_attn'):
    key='input'if field['name']=='residual_input'else'attention';p=Path(field['file']);require(p.parent==Path('/results/layer0')and field['bytes']==40960,'Original layer0 residual extent differs');record['layers']['0'][key]=metrics(array(ownroot,own,ownroot/('own-l0-%s.f32'%key),(4,2560)),array(native,nr,directory/'layer0'/p.name,(4,2560)))
  require(record['layers']['0']['input']is not None and record['layers']['0']['attention']is not None,'Both observed layer0 phases required');record['largest_post_FFN_divergences']=sorted([{'layer':int(k),**v['ffn']}for k,v in record['layers'].items()],key=lambda x:x['nmse'],reverse=True)[:8];checks.append(record)
 require(sha(ownroot/'report.json')==OWN_SHA and sha(native/'report.json')==NATIVE_SHA,'Original report changed during comparison');return {'schema':1,'own_report_sha256':OWN_SHA,'historical_target_report_sha256':NATIVE_SHA,'actual_source_generation':43,'source44_observer_compatibility_used':False,'materialized_mode':1,'prefix_ids':[248045],'comparisons':checks,'observed_native_scopes':'48 post-FFN residuals + head + layer0 input/attention per request','all_other_input_attention_observed':False,'provider_or_model_reads':False,'current_native_qualification_transferred':False,'full_model_math_qualified':False,'numeric_pass_claim':False,'fresh_model_scan_performed':False}
def main():
 p=argparse.ArgumentParser();p.add_argument('--own-root',type=Path,required=True);p.add_argument('--native-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external comparison required');plan=read_unique(PLAN)
 for name,want in plan['files'].items():require(sha(ROOT/name)==want,'Comparison source changed '+name)
 r=execute(a)
 for name,want in plan['files'].items():require(sha(ROOT/name)==want,'Closing comparison source changed '+name)
 r['comparison_source_plan_sha256']=sha(PLAN);a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='ascii')
if __name__=='__main__':main()
