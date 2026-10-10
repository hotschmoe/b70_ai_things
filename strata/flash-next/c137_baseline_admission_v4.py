"""Explicit new baseline artifact admission; never rewrite an old failed parent."""
import importlib
from pathlib import Path
import c1_serve_controller_combined_v137 as c
import adjudicate_c137_missing_started_v3_v1 as adjudicator
def admit_adjudication(receipt):
 receipt=Path(receipt).resolve();saved=c.read(receipt)
 c.require(saved.get('adjudication_kind')=='C137_actual90362_missing_started_v3_v1' and not receipt.is_relative_to(adjudicator.RUN),'Explicit new readonly adjudication artifact outside originalrun required')
 actual=adjudicator.finalized_binding(saved['original_run_root']);c.require(saved==actual,'Actual current strict readonly adjudication differs from declared saved artifact')
 return saved
def finalized_binding(root,adjudication_receipt=None):
 root=Path(root).resolve()
 if adjudication_receipt is not None:
  proof=admit_adjudication(adjudication_receipt);c.require(Path(proof['original_run_root']).resolve()==root,'Adjudication belongs to different baseline root');prepared=c.validate_prepared(root)
  return prepared,{'baseline_kind':'explicit_readonly_adjudicated_actual90362','baseline_root':str(root),'adjudication_receipt':str(Path(adjudication_receipt).resolve()),'adjudication_receipt_sha256':c.sha(adjudication_receipt),'adjudication':proof,'original_parent_passed':False,'engine_receipt_sha256':prepared['engine_receipt_sha256']}
 c.require(c.read(root/'qualification.json').get('c1_parent_generation')==1374,'Only currentparent1374 or explicit strict90362 adjudication artifact admitted')
 parent=importlib.import_module('qualify_c1_serving_combined_v137_v4');prepared=c.validate_prepared(root);proof=parent.validate_final_source_proof(root,prepared)
 return prepared,{'baseline_kind':'actual_current_parent1374','baseline_root':str(root),'qualification_sha256':c.sha(root/'qualification.json'),'parent_sha256':c.sha(root/'parent-qualification.json'),'source_proof':proof,'engine_receipt_sha256':prepared['engine_receipt_sha256']}
