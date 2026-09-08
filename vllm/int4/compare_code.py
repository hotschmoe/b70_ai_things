#!/usr/bin/env python3
"""Report paired HumanEval+ changes without treating a score as full quality proof."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fp8'))
from audit_code_outputs import inspect_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    def load(root):
        config = json.loads((root / 'config.json').read_text())
        summary = json.loads((root / 'summary.json').read_text())
        assert not summary.get('error') and not summary.get('skipped'), summary
        corruption = inspect_samples(summary['raw_samples'])
        assert not corruption, ('raw output corruption invalidates quality comparison', str(root), corruption)
        result = json.loads(Path(summary['eval_results']).read_text())
        return config, result
    bc, b = load(args.baseline); cc, c = load(args.candidate)
    for key in ('sampling', 'thinking', 'dataset_sha256', 'grader_image', 'limit'):
        assert bc[key] == cc[key], ('configuration mismatch', key)
    assert len(b['eval']) == len(c['eval']) == 164 and b['eval'].keys() == c['eval'].keys()
    assert b['hash'] == c['hash'], 'grader dataset identity changed'
    report = {'baseline': str(args.baseline), 'candidate': str(args.candidate),
              'baseline_model': bc['model'], 'candidate_model': cc['model'],
              'n': 164, 'metrics': {}, 'quality_scope': 'thinking-off HumanEval+ only'}
    for metric in ('base', 'plus'):
        def passes(rows):
            assert len(rows) == 1
            r = rows[0]
            return r['base_status'] == 'pass' and (metric == 'base' or r['plus_status'] == 'pass')
        bp = {k for k, rows in b['eval'].items() if passes(rows)}
        cp = {k for k, rows in c['eval'].items() if passes(rows)}
        report['metrics'][metric] = {'baseline_pass': len(bp), 'candidate_pass': len(cp),
                                     'additional_failures_net': len(bp) - len(cp),
                                     'new_failures': sorted(bp-cp), 'recovered': sorted(cp-bp)}
    report['score_gate_passed'] = all(r['additional_failures_net'] <= 2 for r in report['metrics'].values())
    report['promotion_qualified'] = False
    report['remaining_review'] = 'Inspect every new failure; require independent state/tool/coherence/speed/lifecycle gates.'
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
