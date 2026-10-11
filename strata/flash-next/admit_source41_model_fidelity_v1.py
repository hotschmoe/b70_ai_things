#!/usr/bin/env python3
"""Small enclosing actor-V3/original-reference comparison admission."""
import argparse,hashlib,json,time
from pathlib import Path
import validate_source41_model_cli_v3 as actor
import native_qsa41_targets_v1 as raw
import native_qsa41_original_localization_v1 as original
from serial37_canonical_json_v3 import canonical
ROOT=Path(__file__).resolve().parents[2];PLAN=Path(__file__).with_name('source41-model-fidelity-admission-source-plan-v1.json');require=raw.require

def source_binding():
    record=raw.read_unique(PLAN)
    for name,want in record['files'].items():require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,'Exact enclosing fidelity source changed '+name)
    raw.source_binding();actor.a.closure();return hashlib.sha256(PLAN.read_bytes()).hexdigest()

def finalized_binding(actor_root,reference_root,output):
    actor_root=Path(actor_root).resolve();reference_root=Path(reference_root).resolve();output=Path(output).resolve()
    require(not output.exists()and output not in(actor_root,reference_root)and actor_root not in output.parents and reference_root not in output.parents,'New external admission file required')
    source=source_binding();report={'schema':1,'started_epoch':time.time(),'actor_root':str(actor_root),'reference_root':str(reference_root),'source_plan_sha256':source,'passed':False,'errors':[],'diagnostic_comparison_only':True,'full_model_math_qualified':False,'API_qualified':False,'shelf_qualified':False,'tolerance_fitted':False,'captured_operands_are_reference_inputs':False}
    before=None
    try:
        before=actor.captured_binding(actor_root);report['actor_before']=before
        require(type(before['materialized_mode'])is int and before['materialized_mode']==1 and before['observer_enabled']is True,'Separate actual41 mode1 observer-enabled process required')
        directory=Path(before['capture_root']);requests=raw.read_unique(directory/'requests.json');native=raw.captured_arm(before,requests)
        report['original_fidelity']=original.join(reference_root,native)
        require(report['original_fidelity']['full_model_math_qualified']is False,'Descriptive original comparison cannot qualify fullmath')
    except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        # Always re-admit the original actor after raw/reference reads, even on failure.
        try:
            after=actor.captured_binding(actor_root);report['actor_after']=after
            require(before is not None and canonical(after)==canonical(before),'Exact admitted actual41 descriptor changed after comparison')
        except BaseException as exc:report['errors'].append('actor-after: '+type(exc).__name__+': '+str(exc))
        try:require(source_binding()==source,'Final current enclosing/raw/actor source changed')
        except BaseException as exc:report['errors'].append('source-after: '+type(exc).__name__+': '+str(exc))
        report['finished_epoch']=time.time();report['passed']=not report['errors'];output.parent.mkdir(parents=True,exist_ok=True);require(not output.exists(),'New enclosing admission output required');output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='ascii')
    require(report['passed'],'Enclosing actual41 original-fidelity admission failed; original evidence retained')
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--actor-root',type=Path,required=True);p.add_argument('--original-reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();finalized_binding(a.actor_root,a.original_reference_root,a.output)
if __name__=='__main__':main()
