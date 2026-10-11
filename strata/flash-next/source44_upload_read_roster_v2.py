"""Explicit consumed-file roster for one unchanged source44 upload admission."""
import hashlib,os,stat,time
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical
MAX_JSON=64<<20;MAX_FILES=4096;MAX_BYTES=8<<30

def require(ok,msg):
 if not ok:raise ValueError(msg)
def path(value):
 p=Path(value);require(p.is_absolute(),'Absolute consumed path required')
 for part in (p,*p.parents):require(not part.is_symlink(),'Consumed alias refused: '+str(part))
 require(p.is_file(),'Consumed regular file required: '+str(p));return p

def json_read(value):
 p=path(value);require(p.stat().st_size<=MAX_JSON,'Consumed JSON oversized before parse');return read_unique(p)

def roster(prerequisite):
 """Follow only fields actually consumed by frozen SDK/oracle/upload readers."""
 import source44_sdk_binding_v1 as sdk
 repo=Path(__file__).resolve().parents[2];paths=set()
 def add(value):
  p=path(value);paths.add(str(p));require(len(paths)<=MAX_FILES,'Consumed roster too large');return p
 def read(value):return json_read(add(value))
 record=read(prerequisite);upload=add(record['upload_receipt']);oracle=add(record['oracle_receipt']);add(record['pack_receipt'])
 engine=read(sdk.SDK/'receipt.json');sdk_plan=read(sdk.PLAN);add(sdk.SDK/'plan.snapshot.json');add(sdk.SDK/'build/CMakeCache.txt')
 for name in sdk_plan['expected_patched_source_sha256']:add(sdk.SDK/'source'/name)
 for name in engine['binary_sha256']:add(name)
 source_plan=read(repo/'strata/flash-next/source44-upload-source-plan-v1.json')
 for name in source_plan['files']:add(repo/name)
 plan_path=repo/'strata/flash-next/native-source-upload-plan-full-source44-v1.json';read(plan_path)
 oracle_row=read(oracle);plan=read(oracle.parent/'plan.snapshot.json');add(oracle.parent/'oracle.cpp');add(oracle.parent/'source-upload-oracle')
 for name in oracle_row['library_sha256']:add(name)
 # These external JSON references are read by original_upload_gate, not discovered
 # by a directory glob or by treating unrelated command strings as evidence.
 add(plan['source_roster']['path']);read(plan['source_roster']['receipt']);read(plan['inventory'])
 report=read(upload);add(upload.parent/'controller.py');add(upload.parent/'source-page-watchdog.py');read(upload.parent/'post-model-identity.json')
 add(repo/'strata/flash-next/model-lock.json');add(repo/'strata/flash-next/run_source_upload44_oracle_v1.py');add(repo/'strata/flash-next/source_page_watchdog_v3.py');add(repo/'strata/flash-next/parse_usm_logical_free_trace.py')
 for row in report['cases']:
  case=row['case'];require(isinstance(case,str)and case and '/'not in case and case not in('.','..'),'Exact case basename required')
  add(upload.parent/(case+'.log'));read(upload.parent/(case+'-logical-free.json'))
 ordered=sorted(paths);require(sum(path(p).stat().st_size for p in ordered)<=MAX_BYTES,'Consumed total byte quota exceeded before seal');return ordered

def signature(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,stat.S_IFMT(s.st_mode)]
def seal(paths):
 require(paths==sorted(set(paths))and len(paths)<=MAX_FILES,'Exact unique sorted roster required');started=time.time();rows={};total=0
 for name in paths:
  p=path(name);before=signature(p.stat());total+=before[2];require(total<=MAX_BYTES,'Consumed byte quota exceeded');h=hashlib.sha256();count=0
  with p.open('rb')as stream:
   require(signature(os.fstat(stream.fileno()))==before,'Consumed descriptor changed before read')
   for raw in iter(lambda:stream.read(8<<20),b''):h.update(raw);count+=len(raw)
   require(signature(os.fstat(stream.fileno()))==before,'Consumed descriptor changed during read')
  require(signature(path(name).stat())==before and count==before[2],'Consumed file changed during seal')
  rows[name]={'sha256':h.hexdigest(),'bytes':count,'stat6':before}
 return {'owner_pid':os.getpid(),'started_epoch':started,'finished_epoch':time.time(),'rows':rows,'bytes':total}

def bracket(prerequisite,call):
 owner=os.getpid();paths=roster(prerequisite);before=seal(paths);began=time.time()
 result=call();ended=time.time();require(os.getpid()==owner,'Foreign upload owner refused')
 require(roster(prerequisite)==paths,'Consumed upload roster changed');after=seal(paths)
 require(canonical(before['rows'])==canonical(after['rows']),'Consumed upload bytes/stat changed across original call')
 return result,{'owner_pid':owner,'original_call_started_epoch':began,'original_call_finished_epoch':ended,'before':before,'after':after,'raw_read_roster_bracketed':True,'model_payload_bytes_rehashed':False,'model_stat_predicates_retained_in_original_call':True,'actual_successful_original_upload_calls':1}
