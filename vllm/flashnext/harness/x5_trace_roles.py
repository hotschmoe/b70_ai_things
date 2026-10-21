"""Attribute the control's 390 ordered GEMMs using validated layer counts."""
import collections
import json
import statistics
import sys
from pathlib import Path


events = json.loads(Path(sys.argv[1]).read_text())['traceEvents']
gpu = sorted((e for e in events if e.get('cat') == 'gpu_op' and e.get('ph') == 'X'),
             key=lambda e: e['ts'])
cut, end = gpu[0]['ts'], None
for e in gpu:
    if end is not None and e['ts'] - end > 1e6:
        cut = e['ts']
    end = max(end or 0, e['ts'] + e['dur'])
steps, cur = [], []
for e in [e for e in gpu if e['ts'] >= cut]:
    if cur and e['ts'] - max(p['ts'] + p['dur'] for p in cur[-4:]) > 400:
        steps.append(cur)
        cur = []
    cur.append(e)
steps.append(cur)
plans = lambda row: 'b70_sched_planq8_k' in row['name']
selected = [s for s in steps if sum(plans(e) for e in s) == 40]
assert len(selected) >= 20, len(selected)
roles = collections.defaultdict(list)
examples = {}
for step in selected:
    last = 0
    layer = 0
    for i, event in enumerate(step):
        if not plans(event):
            continue
        gm = [e for e in step[last:i] if 'gemm_kernel' in e['name']]
        gdn = layer % 4 != 3
        names = (['gdn_qkv', 'gdn_z', 'gdn_b', 'gdn_a', 'gdn_out'] if gdn else
                 ['attn_q', 'attn_k', 'attn_v', 'attn_out'])
        names += ['router', 'shared_gate', 'shared_up', 'shared_down', 'shared_input_gate']
        assert len(gm) == len(names), (layer, len(gm), len(names))
        for name, e in zip(names, gm):
            roles[name].append(e['dur'])
            examples.setdefault(name, e['name'])
        last = i + 1
        layer += 1
    assert layer == 40
result = dict(steps=len(selected), provenance='ordered layer attribution; counts verified 10/9 GEMMs',
              roles={name: dict(calls_per_step=len(v)/len(selected),
                                ms_per_step=sum(v)/len(selected)/1000,
                                mean_us=statistics.mean(v), kernel=examples[name])
                     for name, v in roles.items()})
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n')
print(json.dumps(result, ensure_ascii=True))
