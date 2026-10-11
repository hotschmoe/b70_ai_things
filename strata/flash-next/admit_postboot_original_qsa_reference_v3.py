#!/usr/bin/env python3
"""ROOT read-only original QSA evidence plus named currentboot authorization.

Historical mathematics is recollected, never rerun with captured operands.
Current C140 authorizes the separate future native observer workload only.
"""
import argparse,hashlib,json
from pathlib import Path
from postboot_original_model_association_v1 import association,require
from serial37_canonical_json_v3 import canonical
HERE=Path(__file__).resolve().parent
PLAN=HERE/'postboot-original-qsa-reader-source-plan-v3.json'

def source_binding():
    from serial37_canonical_json_v3 import read_unique
    source=read_unique(PLAN);root=HERE.parents[1]
    for name,want in source['files'].items():
        path=Path(name);path=path if path.is_absolute() else root/path
        require(hashlib.sha256(path.read_bytes()).hexdigest()==want,'Explicit postboot reader source changed '+name)
    return hashlib.sha256(PLAN.read_bytes()).hexdigest()

def finalized_binding(original_root,historical_identity,current_identity,current_identity_sha256,
                      current_c140_root):
    source=source_binding()
    proof=association(historical_identity,current_identity,
                      current_expected_sha256=current_identity_sha256)
    from c140_baseline_admission_v3 import finalized_binding as current_baseline
    prepared,current=current_baseline(current_c140_root)
    require(current['baseline_kind']=='actual_fresh_source40_parent1403'
            and current['old_source37_runtime_transferred'] is False,
            'Fresh current source40 baseline required')
    require(len(prepared['model_shards'])==4 and
            [row['path'] for row in prepared['model_shards']]==[row['path'] for row in proof['mapping']],
            'Fresh exact ordered four native model shards required')
    for row in prepared['model_shards']:
        path=Path(row['path']);stat=path.stat();sig=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
        expected=[x for x in proof['mapping'] if x['path']==str(path)]
        require(len(expected)==1 and sig==expected[0]['current_stat5'],
                'Fresh native baseline and current publisher identity differ')
    from postboot_owned_qsa_v4_historical_reader_v3 import finalized_binding as historical
    original=historical(original_root,proof)
    require(original['old_current_stat_gate_passed'] is False
            and original['current_runtime_qualified'] is False,
            'Historical proof cannot authorize current runtime')
    from owned_layer3_qsa_control_v4 import arrays
    arrays(original_root)
    from postboot_original_dispatch_source_v1 import admit as source_admit
    import full48_owned_hc_device_rs_v3 as own_math
    from serial37_canonical_json_v3 import read_unique
    original_report=read_unique(Path(original_root)/'report.json');original_work=read_unique(Path(original_root)/'reference-work/work-report.json')
    first_root=Path(original_report['original_work_config']['first_native_root'])
    first_plan=read_unique(first_root/'child/plan.snapshot.json')
    dispatch_proof=source_admit(original_work['first_source_binding'],own_math.prior.source_binding(Path(first_plan['engine_root'])/'source'),proof)
    from postboot_hc35_bulk_host_v1 import finalized_binding as host_admit
    from serial37_canonical_json_v3 import read_unique
    host_root=read_unique(Path(original_root)/'report.json')['original_work_config']['bulk_root']
    host_proof=host_admit(host_root,proof)
    require(host_proof['old_current_runtime_gate_passed'] is False,'Historical host runtime cannot become old current PASS')
    from postboot_original_association_recheck_v1 import recheck
    require(canonical(recheck(proof))==canonical(proof),'Current bytes association changed')
    _,again=current_baseline(current_c140_root)
    require(canonical(again)==canonical(current),'Fresh native baseline changed during reference admission')
    require(source_binding()==source,'Postboot reader source changed during recollection')
    return {'schema':1,'source_plan_sha256':source,'report_sha256':original['original_binding']['report_sha256'],'original_reference':original,'model_identity_association':proof,
            'fresh_native_runtime_authorization':current,'host_identity_association':host_proof,'dispatch_source_identity_association':dispatch_proof,'original_reports_modified':False,
            'old_current_stat_gate_passed':False,'historical_GPU_health_transferred':False,
            'captured_operands_used':False,'full_model_math_qualified':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--original-root',type=Path,required=True)
    p.add_argument('--historical-identity',type=Path,required=True)
    p.add_argument('--current-identity',type=Path,required=True)
    p.add_argument('--current-identity-sha256',required=True)
    p.add_argument('--current-c140-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'New supplemental evidence output required')
    result=finalized_binding(a.original_root,a.historical_identity,a.current_identity,
                             a.current_identity_sha256,a.current_c140_root)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii')
    print('POSTBOOT_ORIGINAL_QSA_ASSOCIATION scope=historical_reference current_runtime=fresh_C140 fullmath=0')

if __name__=='__main__':main()
