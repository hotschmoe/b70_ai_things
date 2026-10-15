#!/usr/bin/env python3
"""Create fresh S106 same-binary arms after root's actual build audit."""
import argparse, hashlib, json
from pathlib import Path
D = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--exe', type=Path, required=True)
p.add_argument('--sha256', required=True)
a = p.parse_args()
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
exe = a.exe.resolve()
assert exe.is_file() and len(a.sha256)==64 and sha(exe) == a.sha256
assert not (D/'serving-inputs.json').exists()
assert all(not (D/f'mode{arm}').exists() for arm in [0, 1])
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
                   STRATA_HEAD_Q5K_RPW2='0', STRATA_MMVQ_WIDE_RPW='1', STRATA_MMVQ_A2='1', STRATA_MMVQ_PF='0', STRATA_BATCH_MTP_PLE_PREFETCH='0', STRATA_TRACE='1', STRATA_Q6_CENTER='0', STRATA_QSA_PREFILL_TILED='0', STRATA_ATTN_VALUE_PIPELINE='0', STRATA_PROMPT_ATTN_XMX='0', STRATA_KV_ROT='0', STRATA_BATCH_PREFILL_QUANTUM='0', STRATA_PREFILL_FIRST='0', STRATA_EXPERT_SPLIT='0', STRATA_GU_LANES='8', STRATA_IQ2XS_MULTI='0')
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
    c['env']['STRATA_BATCH_MTP_PAIR_GRAPH'] = arm
    (out/'config.json').write_text(json.dumps(c, indent=2)+'\n', encoding='ascii')
    arms[arm] = {'exe': str(exe), 'sha256': a.sha256}
inputs = {'arms': arms, 'image': prior['image'], 'same_binary': True,
          'HC_state': 'COMMON_LO_ONCE_S51', 'S49_state': 'OFF_BOTH',
          'native': prior['native'], 'base_config_sha256': sha(basefile),
          'containers': {'0': 'strata-s106-mode0', '1': 'strata-s106-mode1'},
          'warm_pairs': 1, 'timed_fresh_pairs': 3, 'warm_output_tokens_per_stream': 256, 'timed_output_tokens_per_stream': 1500,
          'common_trace': 'STRATA_TRACE1_PROMPT_START_ONLY_CPU_RING_ABSENT',
          'arm_difference': 'STRATA_BATCH_MTP_PAIR_GRAPH0/1 and arm-local log/container/service names only',
          'capture_scope': 'source/build/config+first-eligible host marker+successful warm/use/no errors+exact logged new-capture layouts; all-window cache-hit/row-layout/independent GPU-dispatch visibility absent',
          'latency_scope': 'client wall/SSE content-event gaps; matched TRACE1 CPU logging, CPU ring/profile absent; not token ITL or device-only service',
          'candidate_scope': 'default-OFF depth2 exact-key whole-pair rerecord, host selection only until separate main submission probe; no daily promotion',
          'fixed_work_args': fixed_work, 'request_spec_min_p': 0, 'S100_speed_comparison_allowed': False,
          'pair_key_visibility': 'marker observes only first successful eligible T0,T1 capture; all keys/replays are not logged'}
(D/'serving-inputs.json').write_text(json.dumps(inputs, indent=2)+'\n', encoding='ascii')
print(json.dumps(inputs))
