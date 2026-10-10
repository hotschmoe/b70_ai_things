# NEW mandatory immutable logical-byte port of c137_baseline_pack49_v1.py
from upload_logical_port_identity_v2 import original_producer_file, original_module_file
from c137_prepared_logical_ports_v2 import validate_prepared
'Explicit new baseline artifact admission; never rewrite an old failed parent.'
import importlib
from pathlib import Path
import c1_serve_controller_combined_v137 as c
import adjudicate_c137_logical_ports_v2 as adjudicator

def admit_adjudication(receipt, *, pack_epoch, sdk_epoch, logical_epoch, logical_roster):
    receipt = Path(receipt).resolve()
    saved = c.read(receipt)
    c.require(saved.get('adjudication_kind') == 'C137_actual90362_missing_started_v3_v1' and (not receipt.is_relative_to(adjudicator.RUN)), 'Explicit new readonly adjudication artifact outside originalrun required')
    actual = adjudicator.finalized_binding(saved['original_run_root'], pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    c.require(saved == actual, 'Actual current strict readonly adjudication differs from declared saved artifact')
    return saved

def finalized_binding(root, adjudication_receipt=None, *, pack_epoch, sdk_epoch, logical_epoch, logical_roster):
    root = Path(root).resolve()
    if adjudication_receipt is not None:
        proof = admit_adjudication(adjudication_receipt, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
        c.require(Path(proof['original_run_root']).resolve() == root, 'Adjudication belongs to different baseline root')
        prepared = validate_prepared(root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
        return (prepared, {'baseline_kind': 'explicit_readonly_adjudicated_actual90362', 'baseline_root': str(root), 'adjudication_receipt': str(Path(adjudication_receipt).resolve()), 'adjudication_receipt_sha256': c.sha(adjudication_receipt), 'adjudication': proof, 'original_parent_passed': False, 'engine_receipt_sha256': prepared['engine_receipt_sha256']})
    c.require(c.read(root / 'qualification.json').get('c1_parent_generation') == 1374, 'Only currentparent1374 or explicit strict90362 adjudication artifact admitted')
    parent = importlib.import_module('qualify_c1_serving_combined_v137_v4')
    prepared = validate_prepared(root, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    proof = parent.validate_final_source_proof(root, prepared)
    return (prepared, {'baseline_kind': 'actual_current_parent1374', 'baseline_root': str(root), 'qualification_sha256': c.sha(root / 'qualification.json'), 'parent_sha256': c.sha(root / 'parent-qualification.json'), 'source_proof': proof, 'engine_receipt_sha256': prepared['engine_receipt_sha256']})
