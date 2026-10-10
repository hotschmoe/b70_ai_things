"""Admit selected CPU-screen inputs for a future fresh GPU experiment.

This returns authentic input messages/IDs only. CPU output/state and surrounding
observer success are not transferred into GPU correctness or overlap evidence.
"""
import copy
from pathlib import Path
import qualify_api_positive_overlap_cpu_screen_v3 as screen
import produce_api_positive_overlap_corpus_v2 as corpus
from serial37_canonical_json_v3 import canonical


def selected_inputs(binding, fixture, fixture_binding, candidates):
    screen.require(binding['continuation_screen_passed'] is True and
                   type(binding['complete_measurement_rows']) is int and
                   binding['complete_measurement_rows'] == 16,
                   'Complete eligible all16 screen required')
    screen.require(canonical(binding['fixture_binding']) == canonical(fixture_binding),
                   'Actual selected-screen fixture join changed')
    screen.require([c['target_indices'] for c in candidates] == [[0, 1], [2, 3], [4, 5]] and
                   [c['id'] for c in candidates] == ['water-cycle', 'library-returns', 'trail-signs'],
                   'Exact preregistered candidate roster required')
    selected = [c for c in candidates if c['id'] == binding['selected_candidate']]
    screen.require(len(selected) == 1, 'Unique actual selected candidate required')
    warm, targets = fixture['fixtures']['warm'], fixture['fixtures']['target']
    screen.require(len(warm) == 2 and len(targets) == 6,
                   'Complete authentic eight-input corpus required')
    rows = warm + [targets[i] for i in selected[0]['target_indices']]
    for row in rows:
        screen.require(type(row['ids']) is list and row['ids'] and
                       all(type(t) is int and t >= 0 for t in row['ids']) and
                       type(row['rendered']) is str and row['rendered'] and
                       type(row['messages']) is list,
                       'Authentic rendered messages and input IDs required')
    return {'schema': 1, 'selected_candidate': selected[0]['id'],
            'warm': copy.deepcopy(warm),
            'target': copy.deepcopy([targets[i] for i in selected[0]['target_indices']]),
            'CPU_screen_binding': copy.deepcopy(binding),
            'CPU_outputs_or_state_imported': False,
            'CPU_response_lengths_used_as_GPU_expectations': False,
            'surrounding_observer_success_transferred': False,
            'GPU_request_configuration_qualified': False,
            'actual_GPU_positive_overlap_observed': False,
            'actual_full_cache_qualified': False,
            'full_model_math_qualified': False}


def admit(screen_root, *, expected_report_sha256):
    root = Path(screen_root).resolve()
    screen.require(type(expected_report_sha256) is str and
                   len(expected_report_sha256) == 64 and
                   all(c in '0123456789abcdef' for c in expected_report_sha256),
                   'Explicit actual screen report SHA required')
    report_path = root / 'report.json'
    original = report_path.read_bytes()
    screen.require(screen.sha(report_path) == expected_report_sha256,
                   'Actual selected screen report changed')
    binding = screen.finalized_binding(root)
    fixture, fixture_binding = corpus.finalized_binding(binding['fixture_binding']['root'])
    plan = screen.read(root / 'source-plan.snapshot.json')
    result = selected_inputs(binding, fixture, fixture_binding, plan['candidate_order'])
    screen.require(report_path.read_bytes() == original and
                   binding['report_sha256'] == expected_report_sha256,
                   'Actual screen report changed during admission')
    return result
