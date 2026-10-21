"""Host-only outcome summary: operational closure and strict parity differ."""

import argparse
import hashlib
import json
from pathlib import Path


def classify(aout, fixtures):
    errors = []
    log = (aout / 'window.log').read_text()
    for marker in ('GREEDY_RC:0', 'BENCH_RC:0', 'QSA_BINDING_RC:0',
                   'SERVE_RC:0', 'ENGINE_ERROR_RC:0', 'HEALTH_AFTER_RC:0',
                   'SOURCE_UNCHANGED', 'WINDOW_DONE'):
        if marker not in log.splitlines():
            errors.append('missing lifecycle marker: ' + marker)
    if 'xpu-health: HEALTHY' not in (aout / 'health-after.txt').read_text():
        errors.append('native post-health did not pass')
    if (aout / 'engine-errors.txt').read_text().strip():
        errors.append('engine error log is nonempty')
    greedy = json.loads((aout / 'greedy.json').read_text())
    if not greedy[0].get('first_ok') or not all(r.get('match') for r in greedy[1:]):
        errors.append('identity or arithmetic failed')
    probe = json.loads((aout / 'x6-probe.json').read_text())
    fixtures = json.loads(fixtures.read_text())
    expected = tuple(f['target_tokens'] for f in fixtures)
    by_target = {f['target_tokens']: f for f in fixtures}
    if len(by_target) != len(fixtures):
        errors.append('duplicate fixture target')
    rows = probe.get('rows', [])
    if tuple(r.get('target_tokens') for r in rows) != expected:
        errors.append('long-context fixture sequence incomplete')
    if not probe.get('identity', {}).get('first_ok') or not probe.get('identity', {}).get('alias_ok'):
        errors.append('long-context identity failed')
    for row in rows:
        fixture = by_target.get(row.get('target_tokens'))
        if fixture is None:
            errors.append('unexpected fixture row')
            continue
        prompt_sha = hashlib.sha256(fixture['prompt'].encode()).hexdigest()
        if prompt_sha != fixture['prompt_sha256'] or row.get('prompt_sha256') != prompt_sha:
            errors.append('fixture prompt hash mismatch')
        streams = row.get('solo', []) + row.get('concurrent', [])
        if len(row.get('solo', [])) != 2 or len(row.get('concurrent', [])) != 2:
            errors.append('incomplete C1/C2 streams')
        for stream in streams:
            usage = stream.get('usage', {})
            if (stream.get('prompt_tokens') != len(fixture['token_ids'])
                    or stream.get('completion_tokens') != fixture['max_tokens']
                    or usage.get('prompt_tokens') != len(fixture['token_ids'])
                    or usage.get('completion_tokens') != fixture['max_tokens']
                    or not isinstance(stream.get('n_chunks'), int)
                    or stream.get('n_chunks', 0) <= 0
                    or stream.get('ttft_ms') is None
                    or not stream.get('text')):
                errors.append('incomplete or invalid content stream')
    repeat, strata = probe.get('repeat_gate'), probe.get('strata_gate')
    if type(repeat) is not bool or type(strata) is not bool:
        errors.append('final strict gate statuses missing')
    # X6 client exits1 for a completed parity reject; other client outcomes
    # must not be silently laundered into a diagnostic success.
    expected_rc = 0 if repeat is True and strata is True else 1
    if f'X6_PROBE_RC:{expected_rc}' not in log.splitlines():
        errors.append('probe exit status does not match final strict gates')
    return dict(operational_complete=not errors, issues=errors,
                strict_repeat=repeat, strict_strata_reference=strata,
                qualification_pass=not errors and repeat is True and strata is True,
                scope='finite diagnostic; operational success does not qualify fidelity/stability')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aout', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = classify(args.aout, args.fixtures)
    except Exception as exc:
        result = dict(operational_complete=False,
                      issues=[f'{type(exc).__name__}: {exc}'], qualification_pass=False)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n')
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result['operational_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
