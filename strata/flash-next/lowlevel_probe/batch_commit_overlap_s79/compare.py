#!/usr/bin/env python3
"""Exact aggregate work/output gates for S79; no frame/layout inference."""
import hashlib, json, statistics
from pathlib import Path
D = Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text())
def content(row):
    return ''.join(c.get('choices', [{}])[0].get('delta', {}).get('content') or ''
                   for c in row['chunks'] if c.get('choices'))
inputs = read(D/'serving-inputs.json')
assert inputs['same_binary'] and inputs['arms']['0'] == inputs['arms']['1']
for arm in ['0', '1']:
    with Path(inputs['arms'][arm]['exe']).open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == inputs['arms'][arm]['sha256']
configs = [read(D/f'mode{a}/config.json') for a in range(2)]
normalized = []
for arm, c in enumerate(configs):
    assert c['exe'] == inputs['arms'][str(arm)]['exe']
    assert c['log'] == str(D/f'mode{arm}/engine.log')
    assert c['env']['STRATA_BATCH_COMMIT_OVERLAP'] == str(arm)
    assert c['env']['STRATA_HEAD_Q5K_RPW2'] == '0'
    assert c['env']['STRATA_MMVQ_WIDE_K_RPW'] == c['env']['STRATA_MMVQ_WIDE_K'] == '1'
    assert c['env']['STRATA_BATCH_MTP_PLE_PREFETCH'] == '0' and c['env']['STRATA_TRACE'] == '1'
    assert all(k not in c['env'] for k in ['STRATA_HOST_BOUNDARY', 'STRATA_VERIFY_PROFILE', 'STRATA_PREFILL_HOST_BOUNDARY', 'STRATA_PREFILL_TIMING'])
    n = json.loads(json.dumps(c)); del n['log']; del n['env']['STRATA_BATCH_COMMIT_OVERLAP']; normalized.append(n)
assert normalized[0] == normalized[1]
checks = {'identity_and_image': all(read(D/f'mode{a}/health-ready.json')['model'] == 'hotschmoe-dd'
          and read(D/f'mode{a}/container-ready.json')['Image'] == inputs['image'] for a in range(2))}
results = []
for label in ['warm', 'repeat1', 'repeat2', 'repeat3']:
    rows = [[read(D/f'mode{a}/coding/{label}/coding-{i}.json') for i in range(2)] for a in range(2)]
    pairs = [read(D/f'mode{a}/coding/{label}/pair.json') for a in range(2)]
    c = {'admission_overlap': all(p['admission_gate']['status'] == 'PASS_NATIVE_PROMPT_START_ORDER_AND_CLIENT_OVERLAP' and p['client_overlap_ns'] > 0 for p in pairs),
         'request_equal': all(rows[0][i]['request'] == rows[1][i]['request'] for i in range(2)),
         'full_text_equal': all(rows[0][i]['text'] == rows[1][i]['text'] for i in range(2)),
         'SSE_content_equal': all(content(rows[0][i]) == content(rows[1][i]) == rows[0][i]['text'] for i in range(2)),
         'usage_equal': all(rows[0][i]['usage'] == rows[1][i]['usage'] and rows[0][i]['usage']['completion_tokens'] == 1500 for i in range(2)),
         'native_draft_work_equal': pairs[0]['work_timings'] == pairs[1]['work_timings'],
         'native_window_pair_solo_equal': all(pairs[0][k] == pairs[1][k] for k in ['native_windows', 'paired', 'solo']),
         'activation_and_join': pairs[0]['commit_overlap']['enabled'] == 0 and pairs[1]['commit_overlap']['enabled'] == 1
             and pairs[1]['commit_overlap']['asked'] == pairs[1]['commit_overlap']['deferred'] == pairs[1]['commit_overlap']['joined'] == pairs[1]['native_windows']}
    checks[label] = all(c.values())
    results.append({'label': label, 'matched': checks[label], 'checks': c,
                    'pair_wall_s': [p['wall_s'] for p in pairs],
                    'client_decode_tps': [[r['client_decode_tps'] for r in arm] for arm in rows],
                    'content_event_gap_ms': [[r['content_event_gap_ms'] for r in arm] for arm in rows]})
checks['functionals'] = all(read(D/f'mode{a}/functional/verdict.json')['status'] == 'PASS' for a in range(2))
checks['health_teardown'] = all(read(D/f'mode{a}/normal-stop.json')['lease_wrapper_exit'] == 0
            and all(v['exit'] == 0 for v in read(D/f'mode{a}/post-health.json').values()) for a in range(2))
checks['final_diagnostics_complete'] = all(read(D/f'mode{a}/aggregate-proof.json')['present_once'] for a in range(2))
report = {'status': 'PASS_MATCHED_DISCRIMINATOR' if all(checks.values()) else 'FAILED_OR_UNMATCHED',
          'checks': checks, 'same_binary': True, 'pairs': results,
          'timed_pair_wall_median_s': [statistics.median(r['pair_wall_s'][a] for r in results[1:]) for a in range(2)] if all(checks.values()) else None,
          'latency_scope': 'client wall/SSE content-event gaps, not token ITL or device-only service; BDONE ms sampled before publication',
          'capture_scope': 'no exact per-window row-layout/cache-hit visibility without CPU ring; matched aggregate work only',
          'configuration_normalization': ['STRATA_BATCH_COMMIT_OVERLAP0/1', 'arm-local engine log only'],
          'raw_native_token_ids_observed': False, 'no_promotion_claim': True}
(D/'comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='ascii')
print(json.dumps({'status': report['status'], 'checks': checks}))
raise SystemExit(0 if all(checks.values()) else 2)
