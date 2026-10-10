"""Fresh source39/C139 baseline admission only; no older runtime proof transfer."""
from pathlib import Path
import c1_serve_controller_combined_v139 as c
import qualify_c1_serving_combined_v139_v1 as parent

def finalized_binding(root):
 root=Path(root).resolve();final=c.read(root/'qualification.json');c.require(final.get('c1_parent_generation')==1391,'Actual fresh C139 parent1391 required; source35/37/adjudication cannot transfer')
 prepared=c.validate_prepared(root);proof=parent.validate_final_source_proof(root,prepared)
 return prepared,{'baseline_kind':'actual_fresh_source39_parent1391','baseline_root':str(root),'qualification_sha256':c.sha(root/'qualification.json'),'parent_sha256':c.sha(root/'parent-qualification.json'),'source_proof':proof,'engine_receipt_sha256':prepared['engine_receipt_sha256'],'old_source37_runtime_transferred':False,'full_model_math_qualified':False}
