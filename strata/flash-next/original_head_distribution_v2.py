#!/usr/bin/env python3
"""Bound route-reference head diagnostics; no numerical or quality qualification."""
import argparse
import json
from pathlib import Path
from original_head_distribution_v1 import distributions, bound_array, digest, require
from explore_full48_original_routes_v1 import dependency_binding

BASE_SHA = '0e64c5b5be3c4042dbba6ce7635737a0056a02ceb076f500c82dbe67b0141fc1'

def analyze(path, expected_sha):
    require(digest(Path(__file__).with_name('original_head_distribution_v1.py')) == BASE_SHA,
            'Frozen distribution mathematics changed')
    dependencies = dependency_binding()
    require(digest(path) == expected_sha, 'Completed original route report changed')
    report = json.loads(path.read_text())
    require(report['status'] == 'exploratory complete; NO numerical qualification'
            and not report['errors'] and report['full_model_math_qualified'] is False,
            'Completed exploratory route reference required')
    require(report['dependency_sha256'] == dependencies, 'Original route dependencies differ')
    binding = report['native_observation_binding']
    prefix = binding['prefix']
    require(prefix in (1, 2, 4, 8) and binding['native_inputs_or_routes_used_for_own_computation'] is False,
            'Independent own-state provenance required')
    require(report['post_original_identity']['complete_four_publisher_hashes_verified']
            and report['post_original_identity']['current_stat_verified']
            and report['post_dispatch_binding'] == report['source_dispatch_binding'],
            'Final source4 and source3 proof required')
    original = report['exploration']['arrays']['head']
    native = report['exploration']['arrays']['native_head']
    require(native['sha256'] == binding['native_vectors']['head']['sha256'],
            'Saved native head does not match admitted capture')
    return {'schema': 2, 'source_sha256': digest(__file__),
            'distribution_math_sha256': BASE_SHA, 'original_report': str(path),
            'original_report_sha256': expected_sha, 'prefix': prefix,
            'original_head': original, 'native_head': native,
            'scope': 'Saved independent route-estimate/native head distributions only',
            'native_window_rounding_qualified': False, 'full_model_math_qualified': False,
            **distributions(bound_array(original), bound_array(native))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-report', type=Path, required=True)
    parser.add_argument('--original-report-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Choose a new evidence output')
    result = analyze(args.original_report, args.original_report_sha256)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='ascii')
    print(json.dumps({key: result[key] for key in
                     ('prefix', 'argmax_equal', 'top_k_overlap', 'KL_original_to_native', 'total_variation')}))

if __name__ == '__main__':
    main()
