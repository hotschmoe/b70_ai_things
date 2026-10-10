"""NEW explicit171 source consumer closure; inherited625 remain immutable."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'full-cache-shared-runtime-source-plan-v5.json'
TESTS=sorted(p.stem for p in HERE.glob('test_full_cache_shared*cpu_v2.py'))+sorted(p.stem for p in HERE.glob('test_full_cache_shared*cpu_v5.py'))+['test_registry_c140_shared_association_cpu_v3','test_full_cache_shared_registry_source_cpu_v4']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 from full_cache_shared_registry_source_v4 import REGISTRY
 PRIOR_PLAN=HERE/'full-cache-shared-runtime-source-plan-v4.json';PRIOR_SHA='22f15ff917aaace6a4b8982586fb6ce79072e48b39d7d533b61027da405f5718'
 if sha(PRIOR_PLAN)!=PRIOR_SHA:raise ValueError('Frozen V4 source ledger changed')
 prior=json.loads(PRIOR_PLAN.read_bytes());files={n:w for n,w in prior['files'].items()if n!=REGISTRY};files[str(PRIOR_PLAN.relative_to(ROOT))]=PRIOR_SHA
 seeds=list(HERE.glob('*full_cache_shared*v5.py'))+[HERE/'full_cache_shared_registry_source_v4.py',HERE/'full_cache_shared_baseline_admission_v4.py',HERE/'test_full_cache_shared_registry_source_cpu_v4.py',Path(__file__),HERE/'full-cache-shared-runtime-v5-design.md']
 for p in seeds:
  raw=p.read_bytes();raw.decode('ascii')
  if p.suffix=='.py':ast.parse(raw,filename=str(p))
  files[str(p.relative_to(ROOT))]=sha(p)
 if REGISTRY in files:raise ValueError('Mutable registry cannot be raw source predicate')
 return {'schema':5,'status':'NEW source/CPU exact171 consumer and prelaunch recipe/env/failed-recovery successor, no runtime qualification','files':dict(sorted(files.items())),'inherited_source_plan_sha256':PRIOR_SHA,'inherited_nonregistry_bytes':682,'registry_consumer':'full_cache_shared_registry_source_v4.py','actual_registry_models_required':171,'original_global_hash_gate_passed':False,'original_prepared169_proof_rewritten':False,'active_consumer_generation':5,'unchanged_research_alias_generation':3,'engine_plan_sha256':'d87740b5cbd5310895b057541ae5537034b993e38bb20e6248350a7452dbc2e6','actual_baseline_parent_generation_required':1403,'cpu_command':'PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest '+' '.join(TESTS)+' -q','full_cache_runtime_qualified':False,'full_model_math_qualified':False,'latency_qualified':False,'physical_residency_qualified':False}
if __name__=='__main__':
 if OUTPUT.exists():raise SystemExit('Preserve frozen existing V5 source ledger')
 OUTPUT.write_text(json.dumps(plan(),indent=2,ensure_ascii=True)+'\n',encoding='ascii')
