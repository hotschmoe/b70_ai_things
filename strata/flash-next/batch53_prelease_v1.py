"""Cheap prelease source/shape check; full semantic admission runs once leased.
No digest is cached/transferred; no model/pack payload is read by this check.
"""
from pathlib import Path

def cheap(plan,ctrl):
 ctrl.require(__debug__ and type(plan.get('schema'))is int and plan.get('schema')==4 and type(plan.get('harness_generation'))is int and plan.get('harness_generation')==53 and type(plan.get('slots'))is int and plan.get('slots') in (4,6),'Exact future49 requested4/6 recipe required')
 ctrl.require(plan['kind']=='serial','H50 serial4/6 only; API/serial refused before exclusion lease; separate API purpose required')
 ctrl.require(plan['driver_sha256']==ctrl.sha(Path(ctrl.__file__)) and set(plan['dependency_sha256'])==set(ctrl.DEPENDENCIES),'Current future49 controller/source roster differs')
 for name,digest in plan['dependency_sha256'].items():ctrl.require(ctrl.sha(ctrl.HERE/name)==digest,'Current prelease source changed '+name)
 ctrl.source_observers_off(plan['env']);return {'cheap_source_shape_checked':True,'full_semantic_admission_performed':False,'model_or_pack_payload_read':False}

def full_leased(plan,ctrl,*,pack_epoch=None,sdk_epoch=None,logical_epoch=None,logical_roster=None):
 ctrl.c1.leased([0,1]);return ctrl.manifest_binding(plan,pack_epoch=pack_epoch,sdk_epoch=sdk_epoch,logical_epoch=logical_epoch,logical_roster=logical_roster)
