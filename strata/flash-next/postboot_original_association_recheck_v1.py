"""Recompute the named association from original receipts on every public entry."""
from postboot_original_model_association_v1 import association,require
from serial37_canonical_json_v3 import canonical

def recheck(model_association):
    current=association(model_association['historical_identity_path'],
                        model_association['current_identity_path'],
                        current_expected_sha256=model_association['current_identity_sha256'])
    require(canonical(current)==canonical(model_association),'Current exact association changed')
    return current
