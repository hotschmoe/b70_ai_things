"""Fresh source40/C140 baseline admission only; no older runtime proof transfer."""
from pathlib import Path
import c1_serve_controller_combined_v140_v1 as c
import qualify_c1_serving_combined_v140_v1 as parent

def finalized_binding(root):
 root=Path(root).resolve();final=c.read(root/'qualification.json');c.require(final.get('c1_parent_generation')==1401,'Actual fresh C140 parent1401 required; source35/37/adjudication cannot transfer')
 prepared=c.validate_prepared(root);proof=parent.validate_final_source_proof(root,prepared)
 return prepared,{'baseline_kind':'actual_fresh_source40_parent1401','baseline_root':str(root),'qualification_sha256':c.sha(root/'qualification.json'),'parent_sha256':c.sha(root/'parent-qualification.json'),'source_proof':proof,'engine_receipt_sha256':prepared['engine_receipt_sha256'],'old_source37_runtime_transferred':False,'full_model_math_qualified':False}
