#!/usr/bin/env python3
"""Retired-witness CPU comparison ABBA; never current admission authority."""
import argparse,copy,hashlib,json,math,os,platform,stat,sys,time
from pathlib import Path
from serial37_canonical_json_v3 import canonical,unique_object,finite_constant
HERE=Path(__file__).resolve().parent;PLAN=HERE/'masked43-canonical-benchmark-source-plan-v1.json';MAX_BYTES=160<<20
LABELS=['entry','ready','current-use-2','current-use-3:predevice','current-use-4:postrun','current-use-5','current-use-6:postproof']
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sig(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def read_bound(path,wanted,limit=MAX_BYTES):
 p=Path(path);require(p.is_absolute()and not any(x.is_symlink()for x in(p,*p.parents))and p.is_file(),'Original regular nonaliased input required');s=p.stat();before=sig(s);require(stat.S_ISREG(s.st_mode)and before[2]<=limit,'Bounded input before read required');h=hashlib.sha256();parts=[]
 with p.open('rb')as f:
  require(sig(os.fstat(f.fileno()))==before,'Input descriptor differs')
  for raw in iter(lambda:f.read(8<<20),b''):h.update(raw);parts.append(raw)
  require(sig(os.fstat(f.fileno()))==before,'Input descriptor mutated')
 data=b''.join(parts);require(len(data)==before[2]and sig(p.stat())==before and h.hexdigest()==wanted,'Actual consumed SHA/stat differs')
 require(not any(x.is_symlink()for x in(p,*p.parents)),'Input alias changed during read');return data,{'path':str(p),'bytes':len(data),'sha256':h.hexdigest(),'stat5':before}
def read_unique(raw):return json.loads(raw,object_pairs_hook=unique_object,parse_constant=finite_constant)
def source_binding():
 body=PLAN.read_bytes();record=read_unique(body);root=HERE.parents[1]
 for n,w in record['files'].items():read_bound(root/n,w,16<<20)
 return hashlib.sha256(body).hexdigest()
def baseline(boundaries,initial):
 # Exactly the existing three canonicalizations per successful view.
 return [canonical(b['view'])==canonical(boundaries[0]['view'])==canonical(initial)for b in boundaries]
def within_call(boundaries,initial):
 # No module/global cache: both retained strings live only in this invocation.
 entry=canonical(boundaries[0]['view']);current=canonical(initial)
 return [canonical(b['view'])==entry==current for b in boundaries]
def validate(witness,report,witness_sha):
 require(type(witness['schema'])is int and witness['schema']==1 and type(witness['calls'])is int and witness['calls']==6 and len(witness['boundaries'])==7,'Exact successful retired six-call/seven-view fixture required');require([b['label']for b in witness['boundaries']]==LABELS,'Exact original view comparison roster required');require(type(witness['owner_pid'])is int and witness['owner_pid']==report['producer_pid']and witness['owner_pid']>0,'Original saved producer association required');require(report['producer_operation_witness_sha256']==report['artifact_sha256']['producer-operation-witness.json']==witness_sha,'Original report/raw witness hash association differs');require(report['passed']is True and report['errors']==[],'Retired original successful report fixture required')
 for b in witness['boundaries']:require(type(b['owner_pid'])is int and b['owner_pid']==witness['owner_pid'],'Original saved boundary owner differs')
 return {'retired_producer_pid':witness['owner_pid'],'views':7,'saved_witness_only':True,'current_semantic_or_model_admission_authority':False,'all_other_original_witness_predicates_executed':False}
def negatives():
 base={'v':[1,1.0,True,-0.0],'d':{'x':3}};controls={}
 for name,mutate in [('bool_to_int',lambda x:x['v'].__setitem__(2,1)),('int_to_float',lambda x:x['v'].__setitem__(0,1.0)),('signed_zero',lambda x:x['v'].__setitem__(3,0.0)),('nested_change',lambda x:x['d'].__setitem__('x',4))]:
  rows=[{'view':copy.deepcopy(base)}for _ in range(7)];mutate(rows[6]['view']);a=baseline(rows,copy.deepcopy(base));b=within_call(rows,copy.deepcopy(base));require(a==b and len(a)==7 and a[:-1]==[True]*6 and a[-1]is False,'Typed last-view counterexample lost');controls[name]=True
 for compare in(baseline,within_call):
  try:compare([{'view':{1:'x','1':'y'}}]*7,{1:'x','1':'y'})
  except ValueError:continue
  raise ValueError('Duplicate normalized key was accepted')
 controls['duplicate_key']=True;return controls

def run(witness_path,witness_sha,report_path,report_sha,out,rounds=2):
 require(type(rounds)is int and 1<=rounds<=8,'Bounded ABBA rounds required');require(Path(witness_path).parent==Path(report_path).parent and Path(witness_path).name=='producer-operation-witness.json','Exact retired original report/witness pair required');out=Path(out);require(out.is_absolute()and not any(p.is_symlink()for p in out.parents)and not out.exists()and Path(report_path).parent!=out and Path(report_path).parent not in out.parents,'New external diagnostic output required');out.mkdir(parents=True);result={'schema':1,'passed':False,'errors':[],'producer_pid':os.getpid(),'started_epoch':time.time(),'scope':'CPU saved-witness canonical comparison only','current_semantic_or_model_admission_authority':False,'GPU_execution':False,'serving_speed_qualified':False,'model_math_qualified':False};began=time.perf_counter()
 try:
  source=source_binding();raw,before=read_bound(witness_path,witness_sha);report_raw,report_before=read_bound(report_path,report_sha);witness=read_unique(raw);report=read_unique(report_raw);scope=validate(witness,report,witness_sha);boundaries=witness['boundaries'];initial=copy.deepcopy(boundaries[0]['view']);del raw,report_raw;result['setup_seconds']=time.perf_counter()-began;result['negative_controls']=negatives();result['original_fixture_scope']=scope;result['initial_view_origin']='deep copy of saved entry; NOT live current ByteEpoch initial';result['witness_before']=before;result['original_report_before']=report_before;result['source_plan_sha256']=source;result['runtime']={'python':sys.version,'executable':str(Path(sys.executable).resolve()),'kernel':platform.release(),'affinity':sorted(os.sched_getaffinity(0)),'argv':sys.argv,'timing_clock':'perf_counter'}
  rows=[];expected=[True]*7
  for iteration in range(rounds):
   for method in('A','B','B','A'):
    start=time.perf_counter();values=(baseline if method=='A'else within_call)(boundaries,initial);elapsed=time.perf_counter()-start;require(values==expected,'Every original saved comparison must be true');rows.append({'round':iteration,'method':method,'elapsed_seconds':elapsed,'comparisons':values,'successful_canonical_calls':21 if method=='A'else 9});(out/'trials.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
  result['trials']=rows;result['all_seven_comparisons_exact']=True;result['passed']=True
 except BaseException as error:result['errors'].append(type(error).__name__+': '+str(error))
 finally:
  if 'witness_before'in result:
   try:
    _,final_witness=read_bound(witness_path,witness_sha);_,final_report=read_bound(report_path,report_sha);require(final_witness==result['witness_before']and final_report==result['original_report_before']and source_binding()==result['source_plan_sha256'],'Final original raw/report/source binding changed');result['final_witness_binding']=final_witness;result['final_original_report_binding']=final_report;result['witness_after']=final_witness;result['original_report_after']=final_report;result['source_after_sha256']=result['source_plan_sha256']
   except BaseException as error:result['passed']=False;result['errors'].append('Closing: '+type(error).__name__+': '+str(error))
  result['finished_epoch']=time.time();result['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in out.iterdir()if p.is_file()and p.name!='report.json'};(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 require(result['passed'],'Saved-witness diagnostic failed; original inputs preserved');return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--witness',type=Path,required=True);p.add_argument('--expected-witness-sha256',required=True);p.add_argument('--original-report',type=Path,required=True);p.add_argument('--expected-report-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=2);a=p.parse_args();run(a.witness,a.expected_witness_sha256,a.original_report,a.expected_report_sha256,a.output,a.rounds)
if __name__=='__main__':main()
