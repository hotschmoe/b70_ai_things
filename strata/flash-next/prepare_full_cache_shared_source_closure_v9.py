"""NEW complete shared V7 runtime source; frozen V6 and failures preserved."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'full-cache-shared-runtime-source-plan-v9.json'
TESTS=sorted(p.stem for p in HERE.glob('test_full_cache_shared*cpu_v2.py'))+sorted(p.stem for p in HERE.glob('test_full_cache_shared*cpu_v9.py'))+['test_registry_c140_shared_association_cpu_v3','test_full_cache_shared_registry_source_cpu_v4','test_incremental_native_semantic_cursor_cpu_v1','test_full_cache_semantic_phase_stream_cpu_v1','test_full_cache_postterminal_stop_cpu_v1','test_full_cache_shared_v7_integration_cpu_v1','test_eos_waiter_observer40_cpu_v1','test_full_cache_shared_v9_routing_cpu_v1','test_full_cache_flat_producer_import_cpu_v1','test_full_cache_owned_container_logs_cpu_v1']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 prior=HERE/'full-cache-shared-runtime-source-plan-v8.json';wanted='01b148929d192db55274d53d9a16f67e07ba875ec1a9d1ae2d7387e24846c634'
 if sha(prior)!=wanted:raise ValueError('Frozen source V6 changed')
 files=dict(json.loads(prior.read_bytes())['files']);files[str(prior.relative_to(ROOT))]=wanted
 names=['full_cache_shared_flat_bootstrap_v1.py','full_cache_trace_process_identity_v1.py','full_cache_owned_container_logs_v1.py','full_cache_flat_producer_import_v1.py','test_full_cache_flat_producer_import_cpu_v1.py','test_full_cache_owned_container_logs_cpu_v1.py','buffered_native_trace_epoch_v5.py','incremental_native_semantic_cursor_v1.py','full_cache_semantic_phase_stream_v1.py','full_cache_postterminal_stop_contract_v1.py','full_cache_shared_native_lifetime_v3.py','full_cache_shared_native_lifetime_v4.py','full_cache_shared_native_lifetime_v5.py','eos_waiter_terminal_source40_v2.py','test_full_cache_shared_v9_routing_cpu_v1.py','test_incremental_native_semantic_cursor_cpu_v1.py','test_full_cache_semantic_phase_stream_cpu_v1.py','test_full_cache_postterminal_stop_cpu_v1.py','test_full_cache_shared_v7_integration_cpu_v1.py','full-cache-shared-runtime-v9-design.md']
 eos=HERE/'eos-waiter-observer40-source-plan-v1.json'
 if sha(eos)!='dd6f6c9e8254a86368f82406c3634e79969476a4203cc4a89263dadfeb6a097c':raise ValueError('Frozen actual waiter observer source plan changed')
 files.update(json.loads(eos.read_bytes())['files']);files[str(eos.relative_to(ROOT))]=sha(eos)
 for p in list(HERE.glob('*full_cache_shared*v9.py'))+[HERE/n for n in names]+[Path(__file__)]:
  raw=p.read_bytes();raw.decode('ascii')
  if p.suffix=='.py':ast.parse(raw,filename=str(p))
  files[str(p.relative_to(ROOT))]=sha(p)
 if 'evals/configs/models.yaml'in files:raise ValueError('Current registry admitted by named exact171 association, never old raw digest')
 return {'schema':9,'status':'SOURCE/CPU full runtime successor; actual new GPU experiment required','files':dict(sorted(files.items())),'actual_source40_bindings':json.loads(eos.read_bytes())['actual_source40_bindings'],'inherited_source_plan_sha256':wanted,'actual_baseline_parent_generation_required':1403,'actual_prepared_registry_models_required':171,'historical169_proof_transferred':False,'engine_plan_sha256':'d87740b5cbd5310895b057541ae5537034b993e38bb20e6248350a7452dbc2e6','original_9_actor_and_persisted_fresh49_scope_preserved':True,'active_consumer_generation':9,'registered_alias_generation':3,'text_budget_bytes_per_actor':32<<30,'raw_capture_quota_bytes':512<<20,'raw_capture_quota_scope':'captures files only','terminal_drain_seconds_unchanged':60,'original_V6_raw_errors_rewritten':False,'old_failed_run_promoted':False,'cpu_command':'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest '+' '.join(TESTS)+' -q','full_cache_runtime_qualified':False,'model_math_qualified':False,'physical_residency_qualified':False,'latency_qualified':False}
if __name__=='__main__':
 if OUTPUT.exists():raise SystemExit('Existing immutable V8 ledger must be preserved')
 OUTPUT.write_text(json.dumps(plan(),indent=2,ensure_ascii=True)+'\n',encoding='ascii')
