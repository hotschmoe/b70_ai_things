#!/usr/bin/env python3
"""Negative controls for the mirror report gate, using actual source metadata."""
import copy
import json
from pathlib import Path

from run_stage_mirror_gpu_oracle_v2 import report_gate, source_roster


def main():
    plan = json.loads(Path(__file__).with_name('stage-mirror-gpu-oracle-plan-v4.json').read_text())
    roster = source_roster(plan)
    rows = [{**{k: r[k] for k in ['layer', 'expert', 'gu_type', 'down_type', 'source_sha256', 'raw_bytes']},
             'gpu_vs_float_l1_relative': 0, 'cpu_vs_float_l1_relative': 0, 'gpu_vs_native_cpu_l1_relative': 0,
             'input_q8_bytes_exact': True, 'hidden_q8_bytes_exact': True,
             'raw_fused_hidden_observed': False, 'hidden_reconstruction_observed': True,
             'implementation_metrics': [dict(stage=s, nmse=0, normalized_linf=0) for s in
                 ['gate', 'up', 'silu_reconstruction', 'down_matched_q8']]} for r in roster.values()]
    stages = [dict(lo=lo, hi=hi, device=device, segments=count, experts=count,
                   bytes=sum((r['raw_bytes'] + 255) // 256 * 256 for r in roster.values() if lo <= r['layer'] < hi))
              for lo, hi, device, count in [(0, 32, 0, 6), (32, 48, 1, 4)]]
    report = dict(schema=2, full_serving_qualified=False, source_and_arithmetic_passed=True,
                  all_owners_release_returned=True, partial_read_failure_cleanup_returned=True,
                  expert_cases=10, tokens_per_expert=3, graph_replays_after_source_close=10,
                  mixed_quant_layouts=3, mirrored_padded_bytes=33587200, source_storage_reads=10,
                  mirror_host_reads=20, registered_owner_records=12, missing_entry_plan_negative_checks=10,
                  stages=stages, expert_rows=rows)
    report_gate(report, roster, True)
    controls = [
        lambda r: r.update(schema=1),
        lambda r: r['expert_rows'][0].update(raw_fused_hidden_observed=True),
        lambda r: r['expert_rows'][0].update(hidden_reconstruction_observed=False),
        lambda r: r.update(full_serving_qualified=True),
        lambda r: r.update(all_owners_release_returned=False),
        lambda r: r.update(source_storage_reads=11),
        lambda r: r.update(graph_replays_after_source_close=0),
        lambda r: r['expert_rows'][0].update(source_sha256='0' * 64),
        lambda r: r['expert_rows'][0].update(raw_bytes=r['expert_rows'][0]['raw_bytes'] + 1),
        lambda r: r['expert_rows'][0].update(input_q8_bytes_exact=False),
        lambda r: r['expert_rows'][0].update(gpu_vs_native_cpu_l1_relative=0.03001),
        lambda r: r['expert_rows'][0]['implementation_metrics'][0].update(nmse=1.001e-6),
        lambda r: r['expert_rows'][0]['implementation_metrics'][0].update(normalized_linf=1.001e-4),
        lambda r: r['expert_rows'][0]['implementation_metrics'][0].update(nmse=float('nan')),
        lambda r: r['expert_rows'].pop(),
        lambda r: r['expert_rows'].__setitem__(1, copy.deepcopy(r['expert_rows'][0])),
        lambda r: r['stages'][1].update(device=0),
        lambda r: r['stages'][0].update(bytes=r['stages'][0]['bytes'] + 256),
    ]
    for control in controls:
        bad = copy.deepcopy(report)
        control(bad)
        try:
            report_gate(bad, roster, True)
        except AssertionError:
            continue
        raise RuntimeError('Negative report accepted')
    print(json.dumps(dict(scope='CPU report validator only, no GPU or serving',
                         source_experts=len(roster), negative_controls=len(controls), passed=True)))


if __name__ == '__main__':
    main()
