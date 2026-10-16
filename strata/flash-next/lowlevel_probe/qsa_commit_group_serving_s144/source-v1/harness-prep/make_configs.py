#!/usr/bin/env python3
"""Create fresh S144 same-binary arms after root's actual build audit."""
import argparse, hashlib, json
from pathlib import Path
D = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--root-build-receipt', type=Path, required=True)
p.add_argument('--reused-prehealth', type=Path, required=True)
a = p.parse_args()
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
receipt_path=a.root_build_receipt.resolve()
receipt=json.loads(receipt_path.read_text())
assert receipt['stage']=='S143' and receipt['build_exit']==0 and receipt['tee_exit']==0
exe=Path(receipt['exe']['path']).resolve(); exe_sha=receipt['exe']['sha256']
assert len(exe_sha)==64 and exe.is_file() and sha(exe)==exe_sha
verify_source={'path':str(Path(receipt['verify_source']['path']).resolve()),'sha256':receipt['verify_source']['sha256']}
assert len(verify_source['sha256'])==64 and sha(Path(verify_source['path']))==verify_source['sha256']
assert verify_source['sha256']=='c28de288061beae19949982ff3de7cf0a54fcae4d23f61e3a0374f90aeb435c6','Root actual verify source must match frozen groupLen3 S143 candidate'
health_path=a.reused_prehealth.resolve(); health=json.loads(health_path.read_text())
assert health['root_revalidated_no_intervening_GPU_or_stack_change'] is True
assert health['all_required_health_exit0'] is True and health['stage']=='S142'
assert health['sources']
for item in health['sources']:
    assert sha(Path(item['path']))==item['sha256']
assert not (D/'serving-inputs.json').exists() and not (D/'reused-prehealth.json').exists()
assert all(not (D/f'mode{arm}').exists() for arm in [0,1])
source = D.parent/'hc-lo-once-serving-s51-prep'
basefile = source/'mode1/config.json'
assert sha(basefile) == '959df3d15f689fefd8525079d7be3287939000477a4e84ecc497542402a08321'
base = json.loads(basefile.read_text())
prior = json.loads((source/'serving-inputs.json').read_text())
assert json.loads((source/'comparison.json').read_text())['status'] == 'PASS_MATCHED_DISCRIMINATOR'
assert base['args'][base['args'].index('--native')+1] == prior['native']['path']
assert base['args'][base['args'].index('--ple-gguf')+1] == prior['native']['sibling_path']
assert str(Path(prior['native']['path']).resolve()).startswith('/mnt/cache/')
assert Path(prior['native']['path']).stat().st_size == 47039860096
assert str(Path(prior['native']['sibling_path']).resolve()) == prior['native']['sibling_resolved']
assert Path(prior['native']['sibling_path']).stat().st_size == 28800138432
base['exe'] = str(exe)
assert Path(verify_source['path'])==Path(base['cwd']).resolve()/'sycl/src/core/verify.cpp'
# Fixed work in BOTH arms; this differs from S100 and is not its speed control.
fixed_work = {'--suffix-draft':'0', '--lookup-chain':'0', '--spec':'4', '--mtp-max-t':'4', '--spec-min-p':'0'}
for key, value in fixed_work.items():
    assert base['args'].count(key) <= 1
    if key in base['args']: base['args'][base['args'].index(key)+1] = value
    else: base['args'] += [key, value]

base['env'].pop('STRATA_BATCH_COMMIT_OVERLAP', None)
for key in ['STRATA_HOST_BOUNDARY', 'STRATA_VERIFY_PROFILE', 'STRATA_PREFILL_HOST_BOUNDARY', 'STRATA_PREFILL_TIMING', 'STRATA_PREFILL_SYNC', 'STRATA_VERIFY_TRACE', 'STRATA_VERIFY_EAGER']:
    base['env'].pop(key, None)
base['env'].update(STRATA_MMVQ_WIDE='1', STRATA_MMVQ_WIDE_MIN='0', STRATA_MMVQ_WIDE_MAXC='8', STRATA_Q6_PACKED='0',
                   STRATA_MMVQ_WIDE_K='1', STRATA_MMVQ_WIDE_K_RPW='1',
                   STRATA_HEAD_Q5K_RPW2='0', STRATA_MMVQ_WIDE_RPW='1', STRATA_MMVQ_A2='1', STRATA_MMVQ_PF='0', STRATA_BATCH_MTP_PLE_PREFETCH='0', STRATA_TRACE='1', STRATA_Q6_CENTER='0', STRATA_QSA_PREFILL_TILED='0', STRATA_ATTN_VALUE_PIPELINE='0', STRATA_PROMPT_ATTN_XMX='0', STRATA_KV_ROT='0', STRATA_BATCH_PREFILL_QUANTUM='0', STRATA_PREFILL_FIRST='0', STRATA_EXPERT_SPLIT='0', STRATA_GU_LANES='8', STRATA_IQ2XS_MULTI='0', STRATA_BATCH_MTP_PAIR_GRAPH='0', STRATA_Q6_A4='0', STRATA_BATCH_MTP_PAIR_SKIP_PROB='0', STRATA_MTP_PARALLEL_ARGMAX='0', STRATA_BATCH_MTP_OWNER_QUEUES='0', STRATA_Q5_HEAD_RUNTIME_ONES='0')
assert base['parallel'] == 2 and base['env']['STRATA_BATCH_MTP_PAIR_HEAD'] == '1'
assert base['env']['STRATA_BATCH_MTP_DEPTH'] == '2'
assert base['args'][base['args'].index('--max-context')+1] == '200000'
assert base['args'][base['args'].index('--ple-io')+1] == 'direct'
assert base['args'][base['args'].index('--prefill')+1] == '12288'
assert base['args'][base['args'].index('--conversation-cache-mib')+1] == '32768'
assert base['args'][base['args'].index('--conversation-cache-slots')+1] == '4'
arms = {}
for arm in ['0', '1']:
    out = D/f'mode{arm}'
    out.mkdir(exist_ok=False)
    c = json.loads(json.dumps(base))
    c['log'] = str(out/'engine.log')
    c['env']['STRATA_Q5_HEAD_RUNTIME_ONES_C6'] = '0'
    c['env']['STRATA_QSA_COMMIT_GROUP_STEPS'] = arm
    (out/'config.json').write_text(json.dumps(c, indent=2)+'\n', encoding='ascii')
    arms[arm] = {'exe': str(exe), 'sha256': exe_sha}
inputs = {'arms': arms, 'image': prior['image'], 'same_binary': True,
          'HC_state': 'COMMON_LO_ONCE_S51', 'S49_state': 'OFF_BOTH',
          'native': prior['native'], 'base_config_sha256': sha(basefile),
          'containers': {'0': 'strata-s144-mode0', '1': 'strata-s144-mode1'},
          'warm_pairs': 1, 'timed_fresh_pairs': 3, 'warm_output_tokens_per_stream': 256, 'timed_output_tokens_per_stream': 1500,
          'common_trace': 'STRATA_TRACE1_PROMPT_START_ONLY_CPU_RING_ABSENT',
          'arm_difference': 'STRATA_QSA_COMMIT_GROUP_STEPS 0/1 and arm-local log/container/service names only',
          'capture_scope': 'QSA groupLen3 once eligible HOST selection by warm (startup capture allowed), exact observed new-capture layouts; no all-shape/call count or independent GPU dispatch proof',
          'latency_scope': 'client wall/SSE content-event gaps; matched TRACE1 CPU logging, CPU ring/profile absent; not token ITL or device-only service',
          'candidate_scope': 'default-OFF QSA commit groupLen3 only; all other group lengths use old loop; C2/C6 words0 both; HOST selection only; no serving promotion',
          'fixed_work_args': fixed_work, 'request_spec_min_p': 0, 'S100_speed_comparison_allowed': False,
          'qsa_commit_group_steps_marker_scope': 'once first eligible HOST selection; startup capture permitted; no shape incidence, all launches or independent GPU dispatch'}
inputs['source_binding']={'verify_source':verify_source,'root_build_receipt':{'path':str(receipt_path),'sha256':sha(receipt_path)}}
(D/'reused-prehealth.json').write_text(json.dumps(health,indent=2)+'\n',encoding='ascii')
(D/'serving-inputs.json').write_text(json.dumps(inputs, indent=2)+'\n', encoding='ascii')
print(json.dumps(inputs))
