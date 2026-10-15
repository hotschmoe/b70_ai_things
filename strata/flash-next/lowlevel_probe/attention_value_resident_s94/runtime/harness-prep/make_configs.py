#!/usr/bin/env python3
"""Create fresh S94 same-binary prefill arms after root's actual build audit."""
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
assert exe == D.parent/'attn-value-resident-s94-build/build/strata-attn-value-resident-s94', 'Use only the new frozen S94 binary'
assert exe.is_file() and len(a.sha256)==64 and sha(exe) == a.sha256
assert not (D/'serving-inputs.json').exists()
assert all(not (D/f'mode{arm}').exists() for arm in [0, 1])
source = D.parent/'attn-value-main-s93-serving'
basefile = source/'mode0/config.json'
assert sha(basefile) == '0b1b32509cf341f9cdbeb08e4c7a223d48f6299291e0a97f0cac7aa3d5e2fc06'
base = json.loads(basefile.read_text())
prior = json.loads((source/'serving-inputs.json').read_text())
assert base['env']['STRATA_Q6_CENTER']=='0'
assert str(Path(prior['native']['path']).resolve()).startswith('/mnt/cache/')
assert Path(prior['native']['path']).stat().st_size == 47039860096
assert str(Path(prior['native']['sibling_path']).resolve()) == prior['native']['sibling_resolved']
assert Path(prior['native']['sibling_path']).stat().st_size == 28800138432
base['exe'] = str(exe)
base['env'].pop('STRATA_BATCH_COMMIT_OVERLAP', None)
for key in ['STRATA_HOST_BOUNDARY', 'STRATA_VERIFY_PROFILE', 'STRATA_PREFILL_HOST_BOUNDARY', 'STRATA_PREFILL_TIMING', 'STRATA_PREFILL_SYNC', 'STRATA_VERIFY_TRACE', 'STRATA_VERIFY_EAGER', 'STRATA_ATTN_LANECELL']:
    base['env'].pop(key, None)
base['env'].update(STRATA_MMVQ_WIDE='1', STRATA_MMVQ_WIDE_MIN='0', STRATA_MMVQ_WIDE_MAXC='8', STRATA_Q6_PACKED='0',
                   STRATA_MMVQ_WIDE_K='1', STRATA_MMVQ_WIDE_K_RPW='1',
                   STRATA_HEAD_Q5K_RPW2='0', STRATA_MMVQ_WIDE_RPW='1', STRATA_MMVQ_A2='1', STRATA_MMVQ_PF='0', STRATA_BATCH_MTP_PLE_PREFETCH='0', STRATA_TRACE='1', STRATA_Q6_CENTER='0', STRATA_PROMPT_ATTN_XMX='0', STRATA_KV_ROT='0')
assert base['parallel'] == 2 and base['env']['STRATA_BATCH_MTP_PAIR_HEAD'] == '1'
assert base['env']['STRATA_BATCH_MTP_DEPTH'] == '2'
assert base['args'][base['args'].index('--max-context')+1] == '200000'
assert base['args'][base['args'].index('--ple-io')+1] == 'direct'
arms = {}
for arm in ['0', '1']:
    out = D/f'mode{arm}'
    out.mkdir(exist_ok=False)
    c = json.loads(json.dumps(base))
    c['log'] = str(out/'engine.log')
    c['env']['STRATA_QSA_PREFILL_TILED'] = '0'
    c['env']['STRATA_ATTN_VALUE_PIPELINE'] = arm
    (out/'config.json').write_text(json.dumps(c, indent=2)+'\n', encoding='ascii')
    arms[arm] = {'exe': str(exe), 'sha256': a.sha256}
inputs = {'arms': arms, 'image': prior['image'], 'same_binary': True,
          'HC_state': 'COMMON_LO_ONCE_S51', 'S49_state': 'OFF_BOTH',
          'native': prior['native'], 'base_config_sha256': sha(basefile),
          'root_binary_audit_required': True, 'ambient_attention_routes': {'STRATA_PROMPT_ATTN_XMX':'0','STRATA_KV_ROT':'0'}, 'containers': {'0': 'strata-s94-mode0', '1': 'strata-s94-mode1'},
          'warm_pairs': 1, 'timed_fresh_pairs': 0, 'warm_output_tokens_per_stream': 256, 'prefill_cases':['p4096','p32768'], 'prefill_repeats':3, 'prefill_completion_tokens':256,
          'common_trace': 'STRATA_TRACE1_PROMPT_START_ONLY_CPU_RING_ABSENT',
          'arm_difference': 'STRATA_ATTN_VALUE_PIPELINE0/1 and arm-local log/container/service names only',
          'capture_scope': 'source/build/config+observed first-emission request; ON-only value HOST marker once from FIRST4K through all prefills+raw native part headers+successful use/no errors; no independent GPU-dispatch/call-count visibility',
          'latency_scope': 'client wall/SSE content-event gaps; matched TRACE1 CPU logging, CPU ring/profile absent; not token ITL or device-only service',
          'candidate_scope': 'default-OFF explicit resident-mode0/full-width prefill-only nq32/cap2051/INT8/page4 value pipeline; QSA0/Q6center0 fixed; no daily promotion'}
(D/'serving-inputs.json').write_text(json.dumps(inputs, indent=2)+'\n', encoding='ascii')
print(json.dumps(inputs))
