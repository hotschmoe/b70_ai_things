#!/usr/bin/env python3
"""Distribution diagnostics on bound saved original/native heads; no quality gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def distributions(original, native, top_k=10):
    a, b = [np.asarray(x, dtype=np.float64) for x in (original, native)]
    require(a.ndim == 1 and a.shape == b.shape and a.size >= top_k >= 2,
            'Matching full head vectors and bounded top_k required')
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite head')
    def logs(x):
        shifted = x - x.max()
        return shifted - np.log(np.exp(shifted).sum())
    la, lb = logs(a), logs(b)
    pa, pb = np.exp(la), np.exp(lb)
    # Stable sort declares ascending token-ID tie breaking.
    ta, tb = [np.argsort(-x, kind='stable')[:top_k] for x in (a, b)]
    return {'vocabulary': int(a.size), 'top_k': top_k,
            'original_argmax': int(ta[0]), 'native_argmax': int(tb[0]),
            'argmax_equal': bool(ta[0] == tb[0]),
            'top_k_overlap': len(set(ta.tolist()) & set(tb.tolist())),
            'original_top_ids': ta.tolist(), 'native_top_ids': tb.tolist(),
            'original_top_logprobs': la[ta].tolist(),
            'native_top_logprobs': lb[tb].tolist(),
            'KL_original_to_native': float(np.dot(pa, la-lb)),
            'KL_native_to_original': float(np.dot(pb, lb-la)),
            'total_variation': float(np.abs(pa-pb).sum()/2),
            'max_logprob_difference': float(np.abs(la-lb).max()),
            'numeric_pass_claim': False, 'quality_qualified': False,
            'tolerance_gate': None, 'tie_policy': 'ascending token ID'}

def bound_array(field):
    path = Path(field['path'])
    require(not path.is_symlink() and int(field['bytes']) == 993280,
            'Exact saved full vocabulary extent required')
    raw = path.read_bytes()
    require(len(raw) == 993280 and hashlib.sha256(raw).hexdigest() == field['sha256'],
            'Saved head bytes changed')
    return np.frombuffer(raw, dtype='<f4')

def analyze(report_path, report_sha):
    require(digest(report_path) == report_sha, 'Completed original report changed')
    report = json.loads(report_path.read_text())
    require(report['status'] == 'exploratory complete; NO numerical qualification'
            and not report['errors'] and report['native_observation_binding']['prefix'] == 1
            and report['native_observation_binding']['native_inputs_used_for_own_computation'] is False
            and report['post_original_identity']['complete_four_publisher_hashes_verified'],
            'Completed independent original prefix1 provenance required')
    original = report['exploration']['arrays']['head']
    native = report['native_observation_binding']['native_vectors']['head']
    return {'schema': 1, 'source_sha256': digest(__file__),
            'original_report': str(report_path), 'original_report_sha256': report_sha,
            'original_head': original, 'native_head': native,
            'scope': 'Saved prefix1 head distribution diagnostic only; no new model computation',
            'full_model_math_qualified': False, **distributions(bound_array(original), bound_array(native))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-report', type=Path, required=True)
    parser.add_argument('--original-report-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Choose a new evidence output')
    result = analyze(args.original_report, args.original_report_sha256)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='ascii')
    print(json.dumps({k:result[k] for k in ('argmax_equal','top_k_overlap','KL_original_to_native','total_variation')}))

if __name__ == '__main__':
    main()
