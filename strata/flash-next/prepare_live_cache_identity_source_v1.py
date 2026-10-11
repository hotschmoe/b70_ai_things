"""Freeze actual frontend/loaded-owner refusal and independent cold control."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUTPUT=HERE/'live-cache-identity-refusal-source-plan-v1.json'
TESTS=['test_live_cache_identity_namespace_cpu_v1','test_live_cache_identity_actor_cpu_v1','test_live_cache_identity_recollection_cpu_v1']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 prior=HERE/'full-cache-shared-runtime-source-plan-v8.json'
 if sha(prior)!='01b148929d192db55274d53d9a16f67e07ba875ec1a9d1ae2d7387e24846c634':raise ValueError('Immutable current fullcache source predecessor changed')
 files=dict(json.loads(prior.read_bytes())['files']);files[str(prior.relative_to(ROOT))]=sha(prior)
 seeds=list(HERE.glob('*live*identity*v1.py'))+[HERE/'live_cache_loaded_owner_v1.py',HERE/'live-cache-identity-experiment-v1-progress.md',Path(__file__)]
 for p in seeds:
  raw=p.read_bytes();raw.decode('ascii')
  if p.suffix=='.py':ast.parse(raw)
  files[str(p.relative_to(ROOT))]=sha(p)
 return {'schema':1,'status':'SOURCE/CPU actual frontend loaded-owner isolation, fresh actual execution required','files':dict(sorted(files.items())),'actual_alternate_native_weights_loaded':False,'native_cache_model_identity_key_qualified':False,'supported_alternate_weight_hot_reload_endpoint':False,'scope':['actual loaded frontend metadata/object mutation refused before original prepare','actual original native restart entry refused before old owner close','same locked original weights two fresh native incarnations','prime/refusal/repeat complete49 versus independently owned fresh cold controls'],'full_cache_runtime_qualified':False,'model_math_qualified':False,'latency_qualified':False,'cpu_command':'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest '+' '.join(TESTS)+' -q','prerequisite_fullcache_source_plan_sha256':sha(prior),'actual_fresh_baseline_generation':1403,'actual_prepared_registry_models':171}
if __name__=='__main__':
 if OUTPUT.exists():raise SystemExit('Preserve immutable existing live identity ledger')
 OUTPUT.write_text(json.dumps(plan(),indent=2,ensure_ascii=True)+'\n',encoding='ascii')
