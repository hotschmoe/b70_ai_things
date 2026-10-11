"""Fresh current171 C1401403 baseline; no historical169 receipt view."""
from pathlib import Path
from c140_baseline_admission_v3 import finalized_binding as original
from registry_c140_shared_association_v3 import association,C140
from strict_registry_yaml_v3 import parse
from full_cache_shared_registry_source_v4 import ROOT,REGISTRY,sha

def registry_binding(prepared):
 current=association(ROOT/REGISTRY,True);expected=sha(ROOT/REGISTRY)
 if prepared['registry_sha256']!=expected or prepared['alias']not in [r['served_model_id']for r in parse(C140.read_bytes())]:raise ValueError('Fresh actual current171 C140 prepared digest/entry required; no169 view')
 return {'actual_current_registry':current,'actual_prepared_registry_sha256':prepared['registry_sha256'],'actual_prepared_models':171,'actual_prepared_registry_bytes_changed':False,'historical169_proof_transferred':False,'original_global165_hash_gate_passed':False}
def finalized_binding(root):
 prepared,proof=original(root)
 return prepared,{'actual_original_C1401403_proof':proof,'explicit_fresh_current171_association':registry_binding(prepared),'original_prepared_or_proof_rewritten':False,'old_source37_runtime_transferred':False}
