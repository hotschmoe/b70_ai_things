"""Unqualified composition primitive; caller closes all reads with fresh full4."""
import hashlib,math,os,time
from pathlib import Path
import validate_source41_model_cli_v3 as original
from serial37_canonical_json_v3 import read_unique,canonical
HERE=Path(__file__).resolve().parent;PLAN=HERE/'source41-off-deferred-identity-source-plan-v2.json'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def epoch(value):
 require(type(value)in(int,float)and math.isfinite(value)and value>0,'Finite typed epoch required');return value
def source_binding():
 record=read_unique(PLAN);root=HERE.parents[1]
 for path,want in record['files'].items():require(sha(root/path)==want,'Deferred source identity changed '+path)
 return sha(PLAN)

class DeferredOffIdentity:
 def __init__(self,root):
  self.owner=os.getpid();self.root=Path(root).resolve();self.source=source_binding();self.started=time.time();self.finished=None;self.error=None;self.predicates=None;self.reads_finished=None;self.scan=None;self.close_attempts=[]
 def owned(self):require(os.getpid()==self.owner,'Deferred identity cannot transfer between processes')
 def recollect(self):
  self.owned();require(self.finished is None,'Exactly one original OFF predicate recollection required')
  try:
   report=read_unique(self.root/'report.json');require(report['phase']=='off-equivalence','Only exact original V3 OFF evidence supported')
   self.predicates=original._finalized_binding(self.root)
   require(self.predicates['OFF_equivalence_passed']is True and 'current_public_full4_rehashed'not in self.predicates,'Only original nonclosing predicates supported')
  except BaseException as exc:self.error=type(exc).__name__+': '+str(exc)
  finally:self.finished=time.time()
  return self.view()
 def mark_prerequisite_reads_finished(self,source_check):
  """Caller invokes after ALL prerequisite reads, before its first-arm scan."""
  self.owned();require(self.finished is not None and self.reads_finished is None,'Recollection and one final read boundary required')
  try:source_check();require(source_binding()==self.source,'Source changed before caller scan')
  except BaseException as exc:self.error=(self.error+'; 'if self.error else'')+'Final prerequisite/source check: '+type(exc).__name__+': '+str(exc)
  finally:self.reads_finished=time.time()
  return self.reads_finished
 def close_with_caller_scan(self,scan_path,pages_before,pages_after):
  attempt={'scan_path':str(scan_path),'started_epoch':time.time(),'error':None};self.close_attempts.append(attempt)
  try:return self._close_with_caller_scan(scan_path,pages_before,pages_after)
  except BaseException as exc:
   attempt['error']=type(exc).__name__+': '+str(exc);self.error=(self.error+'; 'if self.error else'')+'Closing identity attempt: '+attempt['error'];raise
  finally:attempt['finished_epoch']=time.time()
 def _close_with_caller_scan(self,scan_path,pages_before,pages_after):
  self.owned();require(self.reads_finished is not None and self.scan is None,'Final prerequisite read boundary must precede one closing scan')
  path=Path(scan_path);require(path.is_file()and not path.is_symlink()and path.resolve()==path,'Original caller scan file required');raw=path.read_bytes();value=read_unique(path);require(path.read_bytes()==raw,'Caller scan changed during consumption')
  lock=read_unique(HERE/'model-lock.json');wanted=[r for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')];require(len(wanted)==len(value['rows'])==4 and value['passed']is True and value['lock_sha256']==sha(HERE/'model-lock.json')and value['model_revision']==lock['revision'],'Fresh ordered publisher full4 required')
  for row,want in zip(value['rows'],wanted):
   model=HERE.parents[1]/lock['destination']/want['path'];require(row['path']==str(model)and row['passed']is True and row['bytes']==want['size']and row['sha256']==row['expected_sha256']==want['sha256']and row['stat_before']==row['stat_after']==original.c.stat_signature(model),'Current caller publisher bytes/stat differ')
  third=Path(value['rows'][2]['path']);original.pages(pages_before,third);original.pages(pages_after,third)
  ordered=[self.started,self.finished,self.reads_finished,pages_before['epoch'],value['started'],value['finished'],pages_after['epoch'],time.time()]
  for x in ordered:epoch(x)
  require(ordered==sorted(ordered),'Caller full4/page scan must close all prerequisite reads')
  require(source_binding()==self.source,'Source changed after caller scan');self.scan={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'started':value['started'],'finished':value['finished'],'page_before_epoch':pages_before['epoch'],'page_after_epoch':pages_after['epoch']};return self.view()
 def require_closed(self):
  self.owned();require(self.scan is not None,'Mandatory fresh caller identity scan missing, including on failure');require(self.error is None,'Original predicate failure retained after identity closure: '+str(self.error));return self.view()
 def view(self):
  return {'schema':1,'owner_pid':self.owner,'original_V3_root':str(self.root),'original_predicate_binding':self.predicates,'error':self.error,'started_epoch':self.started,'predicate_terminal_epoch':self.finished,'all_prerequisite_reads_terminal_epoch':self.reads_finished,'caller_identity_scan':self.scan,'closing_attempts':self.close_attempts,'identity_deferred':self.scan is None,'fresh_scan_required_on_failure':True,'separate_V3_current_public_full4_rehashed_claimed':False,'standalone_model_or_runtime_qualified':False,'full_model_math_qualified':False,'scope':'Original V3 OFF predicates only; caller scan closes prerequisite reads; actual arm pre/post gates remain mandatory'}
