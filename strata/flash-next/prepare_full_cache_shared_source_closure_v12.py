"""Immutable V12 newline association/current173 successor; all failures kept."""
import ast,hashlib,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'full-cache-shared-runtime-source-plan-v12.json'
PRIOR=HERE/'full-cache-shared-runtime-source-plan-v11.json'
PRIOR_SHA='fc8f4009580bcdbef6318fd48461bfbb4e5899903188b0b4d240d7e5d7ec1851'
NEW=['eos_waiter_terminal_source40_v3.py','test_eos_waiter_terminal_source40_cpu_v3.py','registry_c140_chunk173_association_v1.py','test_registry_c140_chunk173_association_cpu_v1.py','full_cache_shared_registry_source_v5.py','full_cache_shared_native_lifetime_v6.py','full_cache_flat_producer_import_v4.py','test_full_cache_flat_producer_import_cpu_v4.py','full_cache_shared_flat_bootstrap_v4.py','test_full_cache_shared_v12_routing_cpu_v1.py','test_full_cache_shared_v12_drain_routing_cpu_v1.py','test_full_cache_shared_v12_terminal_routing_cpu_v1.py','test_full_cache_shared_v12_integration_cpu_v1.py','full-cache-shared-runtime-v12-design.md']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plan():
 if sha(PRIOR)!=PRIOR_SHA:raise ValueError('Frozen V11 source closure changed')
 prior=json.loads(PRIOR.read_bytes());files=dict(prior['files']);files[str(PRIOR.relative_to(ROOT))]=PRIOR_SHA
 for path in sorted(set(HERE.glob('*full_cache_shared*v12.py'))|{HERE/n for n in NEW}|{Path(__file__)}):
  raw=path.read_bytes();raw.decode('ascii')
  if path.suffix=='.py':ast.parse(raw,filename=str(path))
  files[str(path.relative_to(ROOT))]=sha(path)
 for n in('matched-geometry-registry171-snapshot-v1.yaml','prefill-chunk64-512-registry-proposal-v1.yaml'):files[str((HERE/n).relative_to(ROOT))]=sha(HERE/n)
 tokens=shlex.split(prior['cpu_command']);tests=[name.replace('_v11','_v12').replace('test_full_cache_shared_v7_integration_cpu_v1','test_full_cache_shared_v12_integration_cpu_v1')for name in tokens[tokens.index('unittest')+1:]if name!='-q'];tests+=['test_eos_waiter_terminal_source40_cpu_v3','test_registry_c140_chunk173_association_cpu_v1','test_full_cache_shared_v12_terminal_routing_cpu_v1','test_full_cache_flat_producer_import_cpu_v4']
 row=dict(prior);row.update(schema=12,status='SOURCE/CPU V12 exact source-readline LF association/current173 successor; new actual runtime required',files=dict(sorted(files.items())),inherited_source_plan_sha256=PRIOR_SHA,active_consumer_generation=12,actual_prepared_registry_models_required='171 or173 actual C140 metadata plus explicit exact173 preserved-entry association',actual_current_registry_models_required=173,cpu_command='PYTHONDONTWRITEBYTECODE=1 B70_CURSOR_TEST_ROOT=/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010 PYTHONPATH=strata/flash-next python3 -m unittest '+' '.join(tests)+' -q',single_readline_LF_is_explicitly_source_bound=True,original_V11_failed_receipt_promoted=False,source41_math_or_ABI_changed=False,old171_current_global_registry_gate_passed=False)
 return row
if __name__=='__main__':
 if OUTPUT.exists():raise SystemExit('Existing V12 source closure must remain immutable')
 OUTPUT.write_text(json.dumps(plan(),indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii')
