#!/usr/bin/env python3
"""Fresh44 mode1 four prefix1 NUM38 OFF/ON; diagnostic native CLI only."""
import ast,copy,os,sys
from pathlib import Path
import run_source41_model_cli_v5 as original
import source44_num38_intake_v2 as intake
import source44_num38_zero_case_v2 as case
from serial37_canonical_json_v3 import canonical
time=original.time;life=original.life;c=original.c;contract=original.contract;Protocol=original.Protocol
ROOT=original.ROOT;HERE=original.HERE;PLAN=HERE/'source44-num38-model-source-plan-v2.json';require=intake.require

def closure():
    for path,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/path)==want,'Exact NUM38 model source changed '+path)
    from admit_postboot_original_qsa_reference_v4 import source_binding
    source_binding();return c.sha(PLAN)

def admission(plan):
    original_m=c.read(plan['template_path']+'/prepared.json');metadata=life.prepared_ctrl.validate_prepared(Path(plan['template_path']))
    require(canonical(original_m)==canonical(metadata)and c.sha(Path(plan['template_path'])/'prepared.json')==plan['template_prepared_sha256']and c.sha(Path(plan['template_path'])/'server-config.json')==plan['template_config_sha256'],'Exact original software template changed')
    require(type(plan['schema'])is int and plan['schema']==2 and plan['intake_producer_sha256']==c.sha(intake.__file__),'Fresh44 intake producer required')
    proof=intake.sdk_binding();require(canonical(proof)==canonical(plan['source44']),'Fresh current44 SDK changed')
    upload=intake.upload_binding(plan['upload_receipt'],plan['oracle_receipt'],plan['pack_receipt']);require(canonical(upload)==canonical(plan['fresh44_upload']),'Fresh current44 upload changed')
    from qualify_indexer_materialized44_masked_v2 import finalized_binding as masked
    require(canonical(masked(plan['masked_root']))==canonical(plan['fresh44_masked_component']),'Fresh current44 masked component changed')
    require(c.sha(plan['actual_model_identity_path'])==plan['actual_model_identity_sha256']and c.sha(plan['inputs_path'])==plan['inputs_sha256'],'Current full identity/input receipts changed')
    require(c.read(plan['inputs_path'])['short'][0]==c.read(plan['inputs_path'])['prefill_about512'][0]==248045,'Independent literal input association differs')
    return metadata,c.read(Path(plan['template_path'])/'server-config.json')

def phase_recipe(phase,metadata,cfg,inputs,binding):
    require(phase in('source44-num38-OFF','source44-num38-ON'),'Only exact NUM38 diagnostic arms allowed')
    m={k:metadata[k]for k in('cards','pack','runtime','profile')};m.update(source=str(intake.SDK/'source'),executable=str(intake.SDK/'build/strata'),alias='source44-mode1-NUM38-prefix1-CLI-diagnostic')
    return m,case.recipe(metadata,cfg,binding,phase.endswith('-ON'))

def derived():
    nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in('run_phase','main')]
    run=next(n for n in nodes if n.name=='run_phase')
    source=ast.unparse(run).replace("('captures', 'p30', 'ple-input', 'qsa')","('captures', 'p30', 'ple-input', 'qsa', 'layer0')")
    require("'layer0'"in source,'Exact extra NUM38 output directory required')
    main=next(n for n in nodes if n.name=='main')
    # The one old phase-selection branch carrying a source41 OFF prerequisite is
    # removed. Both new arms are actual source44 mode1; only NUM38 differs.
    matches=[n for n in main.body if isinstance(n,ast.If)and ast.unparse(n.test)=="a.phase == 'original-fidelity'"]
    require(len(matches)==1,'Exact obsolete cross-engine prerequisite branch required');main.body.remove(matches[0])
    text=ast.unparse(main).replace("choices=('off-equivalence', 'original-fidelity')","choices=('num38-equivalence',)")
    text=text.replace("phases = ('source40-control', 'source41-off') if a.phase == 'off-equivalence' else ('source41-original-fidelity',)","phases = ('source44-num38-OFF', 'source44-num38-ON')")
    text=text.replace("a.phase == 'off-equivalence'","a.phase == 'num38-equivalence'").replace("'OFF_comparison'","'NUM38_observer_comparison'")
    namespace={**vars(original),**globals()};exec(compile(source+'\n'+text,'[source44-NUM38-explicit-parent-port]','exec'),namespace)
    return namespace

def run_phase(*args,**kwargs):return derived()['run_phase'](*args,**kwargs)
def main():return derived()['main']()
if __name__=='__main__':main()
