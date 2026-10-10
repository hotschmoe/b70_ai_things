"""Narrow USM byte-parser memo proposal; generic EvidenceEpoch purity is unchanged."""
import base64,copy,hashlib,json,os,subprocess,sys,tempfile,time
from pathlib import Path
from serial37_canonical_json_v3 import canonical,unique_object,finite_constant
from operation_evidence_snapshot_v2 import EvidenceEpoch
HERE=Path(__file__).resolve().parent
WORKER=HERE/'logical_free_immutable_worker_v2.py'
WORKER_SHA='133e2fc6b2bf0f2ee529cde18b08119ae60dc0ce7846f6e5538448d3b0a0c204'
PARSER=HERE/'parse_usm_logical_free_trace.py'
PARSER_SHA='71714821736fe98790b80f85d1ab72ac59f58ee6336298721f0aaee2753ca665'
PROGRAM_SOURCE_NAMES=('logical_free_immutable_epoch_v2.py','operation_evidence_snapshot_v2.py','operation_pack_hash_witness_v2.py','serial37_canonical_json_v3.py','logical_free_worker_owned_v2.py')
MAX_OUTPUT=32<<20

def require(ok,message):
 if not ok:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def fixed_source():
 require(WORKER.is_file() and not WORKER.is_symlink() and WORKER.absolute()==WORKER.resolve() and digest(WORKER.read_bytes())==WORKER_SHA,'Current fixed worker source differs');raw=PARSER.read_bytes();require(PARSER.is_file() and not PARSER.is_symlink() and PARSER.absolute()==PARSER.resolve() and digest(raw)==PARSER_SHA,'Current original parser source differs');return raw

def invoke(request):
 raw=(canonical(request)+'\n').encode('ascii');command=[str(Path(sys.executable).resolve()),'-I','-B','-S',str(WORKER)];started=time.time();retired=None;stdout=None;stderr=None
 from logical_free_worker_owned_v2 import execute
 with tempfile.TemporaryDirectory(prefix='logical-byte-worker-')as empty:
  out=Path(empty)/'stdout';err=Path(empty)/'stderr'
  try:
   retired=execute(command,raw,out,err,empty)
   require(out.stat().st_size<=MAX_OUTPUT and err.stat().st_size<=MAX_OUTPUT,'Kernel-limited regular worker output exceeded bound')
   with out.open('rb')as handle:stdout=handle.read(MAX_OUTPUT+1)
   with err.open('rb')as handle:stderr=handle.read(MAX_OUTPUT+1)
   require(len(stdout)<=MAX_OUTPUT and not stderr,'Bounded clean worker stdout/stderr required');value=json.loads(stdout,object_pairs_hook=unique_object,parse_constant=finite_constant);require(type(value.get('passed'))is bool and retired['return_code']in (0,1) and (retired['return_code']==0)==value['passed'],'Original worker status contract differs');identity=value['process'];require(type(identity['pid'])is int and identity['pid']==retired['child_pid'] and type(identity['parent_pid'])is int and identity['parent_pid']==os.getpid() and type(identity['start_ticks'])is int and identity['start_ticks']==retired['worker_start_ticks'],'Actual owned worker PID/start identity differs');require(type(value['schema'])is int and value['schema']==1 and value['input_sha256']==digest(raw) and value['parser_source_sha256']==PARSER_SHA,'Actual worker input/source differs')
   counts=value['parse_counts'];require(set(counts)=={'positive','negative'} and type(counts['positive'])is int and 0<=counts['positive']<=1 and type(counts['negative'])is int and 0<=counts['negative']<=3,'Typed actual worker counters required')
   if value['passed']:require(counts=={'positive':1 if request['mode']=='parse'else 0,'negative':3 if request['mode']=='parse'else 0},'Actual successful worker counter scope differs')
   return value,{'command':command,'input_sha256':digest(raw),'stdout_sha256':digest(stdout),'stderr_sha256':digest(stderr),'return_code':retired['return_code'],'started_epoch':started,'finished_epoch':time.time(),'fresh_process':True,'child_pid':retired['child_pid'],'worker_identity':identity,'worker_retirement':retired,'stdout_stderr_regular_sinks_closed':True,'stdout_stderr_pipe_drain_claimed':False,'process_terminal':True,'empty_cwd':True,'clean_environment':{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8'},'model_or_GPU_input':False,'worker_passed':value['passed'],'parse_counts':counts,'error':value.get('error')}
  except BaseException as exc:
   files={}
   for name,path in (('stdout',out),('stderr',err)):
    if path.is_file():
     size=path.stat().st_size;files[name]={'bytes':size,'sha256':digest(path.read_bytes())if size<=MAX_OUTPUT else None,'complete_bytes_hash_observed':size<=MAX_OUTPUT}
   exc.logical_invocation_diagnostic={'command':command,'input_sha256':digest(raw),'started_epoch':started,'finished_epoch':time.time(),'worker_retirement':retired or getattr(exc,'owned_worker_retirement',None),'output_file_observations':files,'return_code':retired['return_code']if retired else None,'child_pid':retired['child_pid']if retired else None,'worker_passed':False,'parse_counts':None,'parse_counts_observed':False,'exception_type':type(exc).__name__,'error':str(exc),'actual_subprocess_attempt_observed':retired is not None or getattr(exc,'owned_worker_retirement',None)is not None}
   raise

def context():return {'worker_path':str(WORKER),'worker_sha256':WORKER_SHA,'parser_path':str(PARSER),'parser_sha256':PARSER_SHA,'program_source_names':list(PROGRAM_SOURCE_NAMES),'source_root':str(HERE),'python_executable':str(Path(sys.executable).resolve()),'max_output_bytes':MAX_OUTPUT}

class LogicalEpoch:
 def __init__(self,roster,max_seconds=900):
  original=fixed_source();self.context=context();self.base_roster=[(Path(p).resolve(),d)for p,d in roster];require(len(self.base_roster)==len(set(p for p,d in self.base_roster)),'Duplicate initial case/source roster');declared=dict(self.base_roster);require(declared.get(WORKER)==WORKER_SHA and declared.get(PARSER)==PARSER_SHA,'Explicit complete frozen parser and worker sources required')
  require(all(HERE/name in declared for name in PROGRAM_SOURCE_NAMES),'Complete host source/dependency roster required')
  require(all(type(d)is str and len(d)==64 and all(c in '0123456789abcdef'for c in d)for p,d in self.base_roster),'Every input must have explicit expected byte SHA')
  probe,self.probe_command=invoke({'mode':'probe','parser_source_base64':base64.b64encode(original).decode('ascii')});require(probe['passed']is True,'Actual runtime probe failed');self.runtime=probe['runtime'];require(all(type(self.runtime[k])is int and self.runtime[k]==1 for k in ('isolated','no_site','dont_write_bytecode')),'Actual isolated runtime flags required');require(type(self.runtime['output_file_limit_bytes'])is int and self.runtime['output_file_limit_bytes']==MAX_OUTPUT,'Actual kernel output file bound required');combined=dict(self.base_roster)
  for row in self.runtime['mapped_and_module_files']:
   p=Path(row['path']);require(p not in combined or combined[p]==row['sha256'],'Runtime/source expected SHA conflict');combined[p]=row['sha256']
  self.complete_roster=list(combined.items());self.evidence=EvidenceEpoch(self.complete_roster,max_seconds);self.memo={};self.calls=0;self.executions=0;self.commands=[];self.errors=[];self.failed=False;self.source_raw=original;self.runtime_key=canonical(self.runtime)
 def owned(self,current_roster):
  self.evidence._owned();require(canonical(context())==canonical(self.context),'Narrow parser immutable execution context changed');fixed_source();require(len(current_roster)==len(set(str(Path(p).resolve())for p,d in current_roster)),'Duplicate current case/source roster');require(canonical({str(Path(p).resolve()):d for p,d in current_roster})==canonical({str(p):d for p,d in self.base_roster}),'Complete current case/source roster differs');self.evidence.require_roster(self.complete_roster)
 def collect(self,log,logical,expected_owner_count,*,current_roster):
  self.owned(current_roster);require(not self.failed,'Failed parser epoch cannot reuse any semantic result');require(type(expected_owner_count)is int and expected_owner_count>=0,'Exact typed expected owner count required');log=Path(log).resolve();logical=Path(logical).resolve();base=dict(self.base_roster);require(log in base and logical in base and log not in (WORKER,PARSER) and logical not in (WORKER,PARSER) and log!=logical,'Exact declared distinct case input paths required');raw=self.evidence.bytes_for(log);ledger=self.evidence.bytes_for(logical);key=canonical({'execution_context':self.context,'source_sha256':PARSER_SHA,'worker_sha256':WORKER_SHA,'program_source_bytes':{name:digest(self.evidence.bytes_for(HERE/name))for name in PROGRAM_SOURCE_NAMES},'runtime':self.runtime_key,'log_path':str(log),'log_sha256':digest(raw),'logical_path':str(logical),'logical_sha256':digest(ledger),'require_owners':True,'expected_owner_count':expected_owner_count});self.calls+=1
  if key not in self.memo:
   request={'mode':'parse','parser_source_base64':base64.b64encode(self.source_raw).decode('ascii'),'log_base64':base64.b64encode(raw).decode('ascii'),'logical_base64':base64.b64encode(ledger).decode('ascii'),'require_owners':True,'expected_owner_count':expected_owner_count}
   try:response,command=invoke(request)
   except BaseException as exc:
    self.failed=True;row=getattr(exc,'logical_invocation_diagnostic',None) or {'command':[str(Path(sys.executable).resolve()),'-I','-B','-S',str(WORKER)],'input_sha256':digest((canonical(request)+'\n').encode('ascii')),'started_epoch':None,'finished_epoch':None,'worker_retirement':None,'actual_subprocess_attempt_observed':False,'worker_passed':False,'parse_counts':None,'parse_counts_observed':False,'exception_type':type(exc).__name__,'error':str(exc),'return_code':None,'child_pid':None};self.commands.append(copy.deepcopy(row));self.errors.append({'exception_type':type(exc).__name__,'error':str(exc),'invocation_index':len(self.commands)-1});raise
   self.commands.append(command)
   try:
    require(response['passed']is True,'Isolated original parser refused bytes: '+str(response.get('error')));require(canonical(response['runtime'])==self.runtime_key,'Fresh worker actual source/runtime closure differs');self.memo[key]=copy.deepcopy(response['result']);self.executions+=1
   except BaseException as exc:self.failed=True;self.errors.append({'exception_type':type(exc).__name__,'error':str(exc),'invocation_index':len(self.commands)-1});raise
  return copy.deepcopy(self.memo[key])
 def seal_predevice(self,*,current_roster):self.owned(current_roster);self.evidence.seal_predevice()
 def finalize(self,*,current_roster):
  self.owned(current_roster);proof=self.evidence.finalize();return {'schema':1,'passed':not self.failed and not self.errors,'worker_errors':self.errors,'owner_pid':os.getpid(),'evidence_byte_witness':proof,'parser_source_sha256':PARSER_SHA,'worker_source_sha256':WORKER_SHA,'worker_runtime':self.runtime,'worker_probe_command':self.probe_command,'worker_commands':self.commands,'semantic_calls':self.calls,'semantic_executions':self.executions,'parse_counts_complete':all(row['parse_counts']is not None for row in self.commands),'positive_parse_count':sum(row['parse_counts']['positive']for row in self.commands)if all(row['parse_counts']is not None for row in self.commands)else None,'negative_parse_count':sum(row['parse_counts']['negative']for row in self.commands)if all(row['parse_counts']is not None for row in self.commands)else None,'live_guards_reused':False,'generic_purity_contract_weakened':False,'saved_result_imported':False,'actual_upload_or_model_runtime_admission':False,'performance_qualified':False}
