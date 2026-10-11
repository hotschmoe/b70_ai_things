"""Immutable V11 source closure; V10 actual failure and all earlier bytes kept."""
import ast,hashlib,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'full-cache-shared-runtime-source-plan-v11.json'
PRIOR=HERE/'full-cache-shared-runtime-source-plan-v10.json'
PRIOR_SHA='7527f8c017050de5deaf002cedb468a3e69e4d9dd3c83d989412eeb5f9654b6c'
NEW=['incremental_native_semantic_cursor_v2.py','test_incremental_native_semantic_cursor_cpu_v2.py','full_cache_semantic_phase_stream_v3.py','test_full_cache_semantic_phase_stream_cpu_v3.py','full_cache_flat_producer_import_v3.py','test_full_cache_flat_producer_import_cpu_v3.py','full_cache_shared_flat_bootstrap_v3.py','test_full_cache_shared_v11_drain_routing_cpu_v1.py','test_full_cache_shared_v11_routing_cpu_v1.py','full-cache-shared-runtime-v11-design.md']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plan():
 if sha(PRIOR)!=PRIOR_SHA:raise ValueError('Frozen V10 source closure changed')
 prior=json.loads(PRIOR.read_bytes());files=dict(prior['files']);files[str(PRIOR.relative_to(ROOT))]=PRIOR_SHA
 for path in sorted(set(HERE.glob('*full_cache_shared*v11.py'))|{HERE/n for n in NEW}|{Path(__file__)}):
  raw=path.read_bytes();raw.decode('ascii')
  if path.suffix=='.py':ast.parse(raw,filename=str(path))
  files[str(path.relative_to(ROOT))]=sha(path)
 tests=[name.replace('_v10','_v11')for name in shlex.split(prior['cpu_command'])[shlex.split(prior['cpu_command']).index('unittest')+1:]if name!='-q']
 tests+=['test_incremental_native_semantic_cursor_cpu_v2','test_full_cache_semantic_phase_stream_cpu_v3','test_full_cache_flat_producer_import_cpu_v3']
 row=dict(prior);row.update(schema=11,status='SOURCE/CPU V11 byte-proof-on-metadata-suspicion successor; actual new GPU run required',files=dict(sorted(files.items())),inherited_source_plan_sha256=PRIOR_SHA,active_consumer_generation=11,cpu_command='PYTHONDONTWRITEBYTECODE=1 B70_CURSOR_TEST_ROOT=/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010 PYTHONPATH=strata/flash-next python3 -m unittest '+' '.join(tests)+' -q',metadata_suspicion_requires_current_consumed_prefix_and_pending_bytes=True,actual_V10_failure_cause_claimed=False,original_V10_failed_receipt_promoted=False)
 return row
if __name__=='__main__':
 if OUTPUT.exists():raise SystemExit('Existing V11 source closure must remain immutable')
 OUTPUT.write_text(json.dumps(plan(),indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii')
