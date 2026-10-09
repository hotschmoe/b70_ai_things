#!/usr/bin/env python3
"""CPU-only full390 and segmented C1 guards, using actual source metadata fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('c1_controller',HERE/'c1_serve_controller.py');c1=importlib.util.module_from_spec(spec);spec.loader.exec_module(c1)


def rejected(call):
    try:call()
    except (ValueError,KeyError):return True
    return False


def main():
    plan=c1.read(HERE/'native-source-upload-plan-full-v6.json');roster={r['name']:r for r in c1.read(plan['source_roster']['receipt'])['RESULT']['rows']}
    inventory=c1.read(plan['inventory']);shapes={t['name']:t['shape_ggml_order'] for f in inventory['files'] if '/UD-Q4_K_XL/' in f['path'] for t in f['tensors']}
    rows=[]
    for name,source in roster.items():
        shape=shapes[name]+[1] if len(shapes[name])==1 else shapes[name]
        rows.append(dict(name=name,shard=source['shard'],type=source['type_id'],ne0=shape[0],ne1=shape[1],bytes=source['bytes'],absolute_offset=source['absolute_offset'],gpu_sha256=source['sha256'],gpu_byte_equal=True))
    stage=dict(lo=0,hi=48,source_rows=rows,hc_images=387,ple_images=3,ordinary_images=300,unique_allocations=690,accounting_equal=True,expected_image_bytes=3821772160,reported_weight_bytes=3821772160)
    device=dict(schema=2,coverage='whole390',source_and_probe_passed=True,all_owners_destructor_returned=True,hc_images=387,ple_images=3,stages=[stage])
    assert len(c1.full_source_case_gate(device,roster,shapes))==390
    controls=[]
    mutations=[('narrow27cannotqualify',lambda d:d.update(hc_images=27)),('missing_source_row',lambda d:d['stages'][0]['source_rows'].pop()),('wrong_gpu_sha',lambda d:d['stages'][0]['source_rows'][0].update(gpu_sha256='0'*64)),('wrong_source_offset',lambda d:d['stages'][0]['source_rows'][0].update(absolute_offset=1)),('ordinary_count_mismatch',lambda d:d['stages'][0].update(ordinary_images=299)),('missing_byte_accounting',lambda d:d['stages'][0].update(reported_weight_bytes=1)),('missing_layer',lambda d:d['stages'][0].update(hi=47)),('destructor_notreturned',lambda d:d.update(all_owners_destructor_returned=False))]
    for name,mutate in mutations:
        bad=copy.deepcopy(device);mutate(bad);assert rejected(lambda:c1.full_source_case_gate(bad,roster,shapes)),name;controls.append(name)
    profile=c1.PROFILES['two-card-segmented'];launch=c1.read(HERE/'native-hc-launch-plan.json');args=launch['engine_arguments_common']+profile['extra'];env={**launch['runtime_environment_common'],**profile['env']};build={'patches':[{'path':'strata/flash-next/patches/'+name} for name in ['0011-sycl-strict-native-omit-legacy-ple-key.patch','0012-sycl-segmented-stage-expert-mirrors.patch']]}
    c1.segmented_profile_gate(profile,build,args,env)
    for name,change in [('missing_segment_flag',lambda e:e.pop('STRATA_STAGE_MIRRORS')),('missing_segment_extent',lambda e:e.update(STRATA_STAGE_MIRROR_SEGMENT_MIB='0')),('incomplete_nohost_guard',lambda e:e.update(STRATA_VERIFY_NO_HOST='0'))]:
        bad=dict(env);change(bad);assert rejected(lambda:c1.segmented_profile_gate(profile,build,args,bad));controls.append(name)
    badargs=args+['--mtp','unrelated'];assert rejected(lambda:c1.segmented_profile_gate(profile,build,badargs,env));controls.append('MTP_route_rejected')
    assert rejected(lambda:c1.segmented_profile_gate(profile,{'patches':[]},args,env));controls.append('unbuilt0011_0012_rejected')
    with tempfile.TemporaryDirectory(prefix='b70-c1-gate-') as tmp:
        p=Path(tmp);(p/'prepared.json').write_text(json.dumps({'launch_allowed':False}));assert rejected(lambda:c1.validate_prepared(p));controls.append('incomplete_draft_rejected')
    print(json.dumps({'scope':'Synthetic GPU report composed from actual390 original source metadata; no device readback claimed','full390_positive':True,'segmented_profile_positive':True,'negative_controls':controls,'passed':True,'gpu_operations':False},sort_keys=True))


if __name__=='__main__':main()
