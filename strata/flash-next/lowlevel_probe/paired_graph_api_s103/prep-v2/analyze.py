#!/usr/bin/env python3
"""Bounded text-summary + complete-output checks; no raw trace or timing claims."""
from pathlib import Path
import csv, hashlib, json, re, sys

out = Path(sys.argv[1])
mode = int(sys.argv[2]) if len(sys.argv) == 3 else None
sha = lambda b: hashlib.sha256(b).hexdigest()
if mode is None:
    a, b = [json.loads((out / f'mode{m}/closure.json').read_text()) for m in (0, 1)]
    assert all(x['status'] == 'PASS_ONE_MODE_GRAPH_API_ONLY' for x in (a, b))
    assert a['selected_step_counts'] == b['selected_step_counts'] == {f'Step{i}': 3 for i in range(6)}
    assert a['actual_graph_submissions'] == 18 and b['actual_graph_submissions'] == 3
    assert a['common_device_commands'] == b['common_device_commands']
    for phase in range(3):
        x = (out / f'mode0/leaf/phase-{phase}.bin').read_bytes()
        y = (out / f'mode1/leaf/phase-{phase}.bin').read_bytes()
        assert x == y and len(x) == 7704
    v = {'status': 'PASS_SYNTHETIC_GRAPH_API_FRESHNESS_AND_18_VS_3_SUBMISSIONS', 'same_six_operations_and_args': True, 'mapped_A_B_A_full_bytes_equal': True, 'each_step_runs_three_times': True, 'actual_graph_submissions': [18, 3], 'per_replay_submissions': [6, 1], 'common_app_setup_fills': 1, 'common_app_readbacks': 3, 'MTP_numeric_qualified': False, 'speed_qualified': False, 'serving_qualified': False}
    (out / 'comparison.json').write_text(json.dumps(v, indent=2) + '\n', encoding='ascii')
    print(json.dumps(v)); sys.exit(0)
assert mode in (0, 1)
p = out / f'mode{mode}'
result = json.loads((p / 'leaf/result.json').read_text())
assert result['status'] == 'PASS_GRAPH_API_FRESHNESS_ONLY' and result['mode'] == mode
assert result['device_pci'] == '0000:0b:00.0' and result['mapped_A_B_A'] is True
assert result['application_setup_fills'] == 1 and result['application_readbacks'] == 3
assert result['application_graph_submits'] == (3 if mode else 18)
files = sorted(f for f in p.glob('unitrace*') if f.is_file())
assert 1 <= len(files) <= 8, 'Missing/unexpected bounded summary files'
api, commands, seen, pins = {}, {}, set(), []
for f in files:
    raw = f.read_bytes(); assert 0 < len(raw) < 2_000_000
    h = sha(raw); pins.append({'path': str(f), 'bytes': len(raw), 'sha256': h})
    if h in seen: continue
    seen.add(h); header = None
    for line in raw.decode('utf8').splitlines():
        cells = [s.strip() for s in next(csv.reader([line], skipinitialspace=True))]
        if not cells or not cells[0]: header = None; continue
        if cells[0] in ('Function', 'Kernel'):
            header = cells if 'Calls' in cells and 'Time (ns)' in cells else None
            continue
        if header:
            assert len(cells) == len(header), ('HOLD malformed summary table row', f.name, line)
            row = dict(zip(header, cells))
            try: n = int(row['Calls'])
            except ValueError: raise AssertionError(('HOLD malformed Calls field', f.name, line))
            assert n >= 0, ('HOLD negative Calls field', f.name, line)
            target = api if 'Function' in row else commands
            key = row.get('Function', row.get('Kernel'))
            target[key] = target.get(key, 0) + n
assert api and commands, 'Actual host and device summary tables required'
steps = {f'Step{i}': sum(n for name, n in commands.items() if re.search(r'\bs103::Step' + str(i) + r'\b', name)) for i in range(6)}
assert steps == {f'Step{i}': 3 for i in range(6)}, steps
graph_submits = api.get('zeCommandListImmediateAppendCommandListsExp', 0)
assert graph_submits == (3 if mode else 18), ('HOLD unexpected graph API count; do not rerun', graph_submits)
common = {name: n for name, n in commands.items() if not any(re.search(r'\bs103::Step' + str(i) + r'\b', name) for i in range(6))}
copies = {name: n for name, n in common.items() if 'zeCommandListAppendMemoryCopy' in name}
fills = {name: n for name, n in common.items() if 'zeCommandListAppendMemoryFill' in name}
assert sum(copies.values()) == 3 and all('[7704]' in name for name in copies), ('HOLD common readback count', copies)
assert sum(fills.values()) == 1 and all('[7704]' in name for name in fills), ('HOLD common setup fill count', fills)
assert common.keys() == copies.keys() | fills.keys(), ('HOLD unclassified device command', common)
snapshots = []
for phase in range(3):
    raw = (p / f'leaf/phase-{phase}.bin').read_bytes(); assert len(raw) == 7704
    snapshots.append({'phase': phase, 'bytes': len(raw), 'sha256': sha(raw)})
assert snapshots[0]['sha256'] == snapshots[2]['sha256'] != snapshots[1]['sha256']
v = {'status': 'PASS_ONE_MODE_GRAPH_API_ONLY', 'mode': mode, 'leaf': result, 'actual_graph_submissions': graph_submits, 'selected_step_counts': steps, 'common_device_commands': common, 'all_host_API_counts': api, 'all_device_command_counts': commands, 'summary_files': pins, 'snapshots': snapshots, 'capture_calls_do_not_execute_selected_kernels': 'Observed selected kernels total18 equals six operations times3 replays; no eager/warm/capture execution remains unattributed.', 'scope': 'Synthetic graph API/ordering/freshness only; profiling-sensitive times are intentionally not interpreted as model gains.'}
(p / 'closure.json').write_text(json.dumps(v, indent=2) + '\n', encoding='ascii')
print(json.dumps({'status': v['status'], 'mode': mode, 'graph_submissions': graph_submits, 'step_counts': steps}))
