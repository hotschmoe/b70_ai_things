"""Cheap prelease source/shape check; full semantic admission runs once leased.
No digest is cached/transferred; no model/pack payload is read by this check.
"""
from pathlib import Path

def cheap(plan,ctrl):
 ctrl.require(__debug__ and plan.get('schema')==4 and plan.get('harness_generation')==46 and plan.get('slots') in (4,6),'Exact future46 requested4/6 recipe required')
 ctrl.require(plan['kind'] in ('native','serial'),'Unsupported API cache0 mode rejected before exclusion lease; separate API purpose required')
 ctrl.require(plan['driver_sha256']==ctrl.sha(Path(ctrl.__file__)) and set(plan['dependency_sha256'])==set(ctrl.DEPENDENCIES),'Current future46 controller/source roster differs')
 for name,digest in plan['dependency_sha256'].items():ctrl.require(ctrl.sha(ctrl.HERE/name)==digest,'Current prelease source changed '+name)
 ctrl.source_observers_off(plan['env']);return {'cheap_source_shape_checked':True,'full_semantic_admission_performed':False,'model_or_pack_payload_read':False}

def full_leased(plan,ctrl):
 ctrl.c1.leased([0,1]);return ctrl.manifest_binding(plan)
