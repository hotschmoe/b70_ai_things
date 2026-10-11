"""Per-call original predicate execution plus its actual validated upload return."""
import hashlib,os,time
from pathlib import Path
import qualify_indexer_materialized41_masked_v2 as original
import indexer_materialized41_masked_inputs_v2 as inputs
from serial37_canonical_json_v3 import read_unique,canonical
HERE=Path(__file__).resolve().parent;PLAN=HERE/'indexer-materialized41-masked-upload-view-source-plan-v1.json'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_binding():
 row=read_unique(PLAN);root=HERE.parents[1]
 for path,want in row['files'].items():require(sha(root/path)==want,'Upload-view source changed '+path)
 return sha(PLAN)
def finalized_binding(root):
 root=Path(root).resolve();source=source_binding();pid=os.getpid();started=time.time();calls=[];namespace=original.derived();original_modules=namespace['modules']
 class PerCallBaseline:
  @staticmethod
  def finalized_binding(directory,adjudication=None):
   require(os.getpid()==pid and not calls,'One same-PID original Baseline call required');begin=time.time()
   modules=original_modules();prepared,proof=modules[-1].finalized_binding(directory,adjudication)
   require(proof['model_baseline_or_constructor_proof_transferred']is False and 'actual_fresh41_upload_binding'in proof,'Actual upload-only original Baseline return required')
   calls.append({'started_epoch':begin,'finished_epoch':time.time(),'prepared':prepared,'proof':proof});return prepared,proof
 def per_call_modules():return (*original_modules()[:-1],PerCallBaseline)
 # Only the fresh derived function namespace is changed, never original globals/classes.
 namespace['modules']=per_call_modules
 component=namespace['finalized_binding'](root)
 witness_admission=inputs.admit_operation_witness(root,read_unique(root/'report.json'))
 require(os.getpid()==pid and len(calls)==1,'Exactly one successful original Baseline execution required');witness=inputs.operation_witness()
 require(type(witness['owner_pid'])is int and witness['owner_pid']==pid and witness['one_full_semantic_admission_per_owned_process']is True and witness['whole_current_input_bytes_on_each_use']is True and witness['generic_persistent_cached_PASS_used']is False and witness['boundaries'],'Same live original byte operation required')
 last=witness['boundaries'][-1];require(last['owner_pid']==pid and started<=last['finished_epoch']<=time.time(),'Actual current-call byte seal required')
 require(source_binding()==source,'Upload-view source changed during original predicates')
 return {'schema':1,'source_plan_sha256':source,'original_masked_component_binding':component,'actual_revalidated_upload_binding':calls[0]['proof']['actual_fresh41_upload_binding'],'same_call':{'owner_pid':pid,'started_epoch':started,'finished_epoch':time.time(),'original_Baseline_calls':1,'original_Baseline_started_epoch':calls[0]['started_epoch'],'original_Baseline_finished_epoch':calls[0]['finished_epoch'],'original_operation_admission':witness_admission,'last_current_byte_seal_epoch':last['finished_epoch'],'current_byte_boundary_count':len(witness['boundaries']),'original_module_globals_or_class_mutated':False,'saved_PASS_substituted':False,'upload_raw_roster_independently_bracketed_by_reference_epoch':False},'model_or_constructor_proof_transferred':False,'full_model_math_qualified':False}
