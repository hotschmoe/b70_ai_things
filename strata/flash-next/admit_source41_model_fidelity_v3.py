#!/usr/bin/env python3
"""Metadata refusal first; valid captures still require complete public admission."""
import argparse,hashlib,importlib,json,time
from pathlib import Path
import native_qsa41_targets_v1 as raw
import native_qsa41_original_localization_v1 as original
from serial37_canonical_json_v3 import canonical
from source41_fidelity_ur_preflight_v2 import preflight
ROOT=Path(__file__).resolve().parents[2];PLAN=Path(__file__).with_name('source41-model-fidelity-admission-source-plan-v3.json');require=raw.require
ACTOR_MODULE='validate_source41_model_cli_v5'
def actor():return importlib.import_module(ACTOR_MODULE)
def source_binding():
 record=raw.read_unique(PLAN)
 for name,want in record['files'].items():require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,'Exact enclosing/preflight source changed '+name)
 raw.source_binding();return hashlib.sha256(PLAN.read_bytes()).hexdigest()
def publish(output,report):
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x',encoding='ascii')as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
def finalized_binding(actor_root,reference_root,output):
 actor_root=Path(actor_root).resolve();reference_root=Path(reference_root).resolve();output=Path(output).resolve()
 require(not output.exists()and output not in(actor_root,reference_root)and actor_root not in output.parents and reference_root not in output.parents,'New external admission file required')
 report={'schema':3,'started_epoch':time.time(),'actor_root':str(actor_root),'reference_root':str(reference_root),'passed':False,'errors':[],'diagnostic_comparison_only':True,'full_model_math_qualified':False,'API_qualified':False,'shelf_qualified':False,'tolerance_fitted':False,'captured_operands_are_reference_inputs':False}
 try:report['metadata_preflight']=preflight(actor_root)
 except BaseException as exc:
  report.update(metadata_only_refusal=True,public_actor_admission_executed=False,original_reference_admission_executed=False,model_payload_read_by_refusal=False,new_closing_model_scan_executed=False,finished_epoch=time.time())
  report['errors'].append(type(exc).__name__+': '+str(exc));report['preflight_producer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();publish(output,report)
  raise ValueError('Metadata-only necessary-condition refusal; original evidence retained')from exc
 source=source_binding();report['source_plan_sha256']=source;before=None;public=actor();public.a.closure()
 try:
  before=public.captured_binding(actor_root);report['actor_before']=before
  require(type(before['materialized_mode'])is int and before['materialized_mode']==1 and before['observer_enabled']is True,'Separate actual41 mode1 observer process required')
  directory=Path(before['capture_root']);native=raw.captured_arm(before,raw.read_unique(directory/'requests.json'));report['original_fidelity']=original.join(reference_root,native)
  require(report['original_fidelity']['full_model_math_qualified']is False,'Comparison cannot qualify fullmath')
 except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
 finally:
  try:
   after=public.captured_binding(actor_root);report['actor_after']=after;require(before is not None and canonical(after)==canonical(before),'Actual41 descriptor changed after comparison')
  except BaseException as exc:report['errors'].append('actor-after: '+type(exc).__name__+': '+str(exc))
  try:require(source_binding()==source and public.a.closure()==before['public_actor_admission']['source_plan_sha256'],'Final enclosing/raw/actor source changed')
  except BaseException as exc:report['errors'].append('source-after: '+type(exc).__name__+': '+str(exc))
  report.update(metadata_only_refusal=False,public_actor_admission_executed=True,finished_epoch=time.time());report['passed']=not report['errors'];publish(output,report)
 require(report['passed'],'Enclosing actual41 fidelity admission failed; originals retained');return report
def main():
 p=argparse.ArgumentParser();p.add_argument('--actor-root',type=Path,required=True);p.add_argument('--original-reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();finalized_binding(a.actor_root,a.original_reference_root,a.output)
if __name__=='__main__':main()
