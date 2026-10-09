#!/usr/bin/env python3
"""Read-only exact cross-topology comparison of finalized numerical v8 all33-field runs."""
import argparse
import hashlib
import json
from pathlib import Path
import layer0_numerical_qualification_v8 as ctrl


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def normalize_args(args, pair):
    result = list(args)
    if pair:
        for option, expected in [('--layer-split', '32'), ('--split-device', '1')]:
            require(result.count(option) == 1, 'Missing/duplicate topology option')
            index = result.index(option)
            require(result[index + 1] == expected, 'Unexpected topology value')
            del result[index:index + 2]
        require(result.count('--trim-stage-weights') == 1, 'Missing split weight trim')
        result.remove('--trim-stage-weights')
    else:
        require(not any(x in result for x in ['--layer-split', '--split-device', '--trim-stage-weights']), 'One-card arm is split')
    return result


def load_run(directory, cards):
    directory = Path(directory).resolve()
    parent = read(directory / 'parent-qualification.json')
    require(parent['passed'] is True and not parent['errors'], 'Parent not finalized PASS')
    child = read(directory / 'child/report.json')
    plan = read(directory / 'child/plan.snapshot.json')
    require(parent['child_return_code'] == 0 and parent['owned_containers_terminal'] is True, 'Child/owned teardown incomplete')
    require(parent['pre_health_passed'] is True and parent['post_health_passed'] is True and parent['kernel_fault_gate_passed'] is True, 'Health/fault gate incomplete')
    require(child['passed'] is True and child['post_health_passed'] is True and child['layer0_logical_lifecycle_qualified'] is True, 'Child qualification incomplete')
    require(child['cards'] == plan['cards'] == cards, 'Unexpected physical card roster')
    require(child['plan_sha256'] == sha(directory / 'child/plan.snapshot.json'), 'Child plan binding differs')
    ctrl.manifest_binding(plan)
    for name in ['post-model-identity.json', 'layer0-logical-lifecycle.json', 'packet-contract.json']:
        require(read(directory / name)['passed'] is True, 'Required finalized receipt failed: ' + name)
    arm = directory / 'child/candidatecombined_on'
    result = read(arm / 'result.json')
    require(result['passed'] is True and result['removed'] is True, 'Capture arm failed')
    require(result['canonical_combined_log_sha256'] == sha(arm / 'engine.combined.log'), 'Canonical log changed')
    rows = read(arm / 'requests.json')
    require([r['prefix'] for r in rows] == [1, 2, 4, 8], 'Exact prefix roster required')
    binding = sha(directory / 'child/plan.snapshot.json')
    for row in rows:
        actual = ctrl.collect_frame(row['raw'], arm, binding)
        require(actual == row['layer0'], 'Captured frame/field provenance changed')
    receipts = {name: sha(directory / name) for name in ['parent-qualification.json', 'child/report.json', 'child/plan.snapshot.json', 'child/candidatecombined_on/requests.json', 'post-model-identity.json', 'layer0-logical-lifecycle.json', 'packet-contract.json']}
    return plan, rows, receipts


def compare(one_directory, pair_directory):
    one, left, one_receipts = load_run(one_directory, [0])
    pair, right, pair_receipts = load_run(pair_directory, [0, 1])
    require(one['engine_receipt_sha256'] == pair['engine_receipt_sha256'], 'Different engine generations')
    require(one['prefixes'] == pair['prefixes'] and one['max_new'] == pair['max_new'] == 1, 'Workload differs')
    require(normalize_args(one['args'], False) == normalize_args(pair['args'], True), 'Non-topology configuration differs')
    results = []
    for a, b in zip(left, right):
        numerical = ctrl.compare_numeric(a, b)
        x = {f['name']: f for f in a['layer0']['observed']}
        y = {f['name']: f for f in b['layer0']['observed']}
        require(set(x) == set(y) and len(x) == 33, 'Observed field roster differs')
        fields = {}
        for name in x:
            require(x[name]['encoding'] == y[name]['encoding'] and x[name]['bytes'] == y[name]['bytes'], 'Field shape differs')
            equal = x[name]['sha256'] == y[name]['sha256']
            fields[name] = {'bitwise_equal': equal, 'bytes': x[name]['bytes'], 'encoding': x[name]['encoding'], 'one_sha256': x[name]['sha256'], 'pair_sha256': y[name]['sha256']}
            require(equal, 'Cross-topology captured field differs: ' + name)
        results.append({'prefix': a['prefix'], 'numerical': numerical, 'layer0_fields': fields})
    return {'passed': True, 'one_receipts_sha256': one_receipts, 'pair_receipts_sha256': pair_receipts, 'results': results, 'scope': 'Matched serial fresh prefixes1/2/4/8 across one card and32/16 split; exact full head/all48 residuals and33 fields (31 source values plus2 DERIVED hidden; raw fused hidden unobserved). No independent model math, own-state reference, API concurrency or latency qualification', 'full_model_math_qualified': False, 'concurrency_qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--one', type=Path, required=True)
    parser.add_argument('--pair', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing evidence')
    result = compare(args.one, args.pair)
    result['checker_sha256'] = sha(Path(__file__))
    result['controller_sha256'] = sha(Path(ctrl.__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='ascii')
    print(json.dumps({'passed': True, 'prefixes': 4, 'observed_fields_compared': 132, 'full_model_math_qualified': False}))


if __name__ == '__main__':
    main()
