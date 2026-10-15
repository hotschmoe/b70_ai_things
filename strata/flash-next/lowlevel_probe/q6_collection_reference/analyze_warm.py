"""Summarize S44 batch timings only after the independent CPU diagnosis."""
import hashlib
import json
from pathlib import Path
import statistics

P = Path(__file__).resolve().parent
closure = json.loads((P / 'run-clean/closure.json').read_text())
gate = json.loads((P / 'cpu-full/gate.json').read_text())
assert closure['oracle_exit'] == closure['pre_health_exit'] == closure['post_health_exit'] == 0
assert closure['container_removed'] and closure['snapshot_file_count'] == 96
assert closure['external_profiler'] == 'none'
assert gate['status'] == 'PASS_PROPOSED_CPU_BOUND_FOUR_FIXTURES_ONLY'
assert gate['violations'] == 0 and gate['outputs_checked'] == 2623488
assert gate['identity']['collection_hashes_unchanged']
records = []
contexts = []
for line in (P / 'run-clean/oracle.stderr').read_text().splitlines():
    if not line.startswith('{'):
        continue
    value = json.loads(line)
    if value['type'] == 'q6_microtiming':
        records.append(value)
    elif value['type'] == 'q6_microtiming_context':
        contexts.append(value)
assert len(records) == 24 and len(contexts) == 4
results = []
metrics = ['host_submit_us', 'host_after_submit_to_complete_us',
           'host_complete_us', 'same_queue_barrier_span_us']
for i in range(4):
    fixture = json.loads((P / f'cpu-full/fixture{i}-gate.json').read_text())
    tensor = fixture['tensor']
    context, = [x for x in contexts if x['tensor'] == tensor]
    assert context['C'] == 6 and context['warmup_calls_per_path'] == 16
    assert context['rounds'] == 3 and context['calls_per_batch'] == 16
    assert context['never_registered_control'] and context['own_output_bitmatch']
    rows = [x for x in records if x['tensor'] == tensor]
    assert len(rows) == 6
    paths = {}
    for path in ('original-control', 'packed'):
        samples = sorted((x for x in rows if x['path'] == path), key=lambda x:x['round'])
        assert [x['round'] for x in samples] == [0, 1, 2]
        assert all(x['C'] == 6 and x['calls'] == 16 and not x['begin_event_waited_before_host_t0'] for x in samples)
        paths[path] = {}
        for metric in metrics:
            values = [x[metric] / x['calls'] for x in samples]
            paths[path][metric] = {'batch_mean_us_per_call':values,
                                  'median':statistics.median(values),
                                  'min':min(values), 'max':max(values), 'batches':3}
    paired = []
    for round_index in range(3):
        pair = sorted((x for x in rows if x['round'] == round_index), key=lambda x:x['position'])
        assert [x['path'] for x in pair] == (['packed','original-control'] if round_index == 1 else ['original-control','packed'])
        original, = [x for x in pair if x['path'] == 'original-control']
        packed, = [x for x in pair if x['path'] == 'packed']
        paired.append(100 * (packed['host_complete_us'] / original['host_complete_us'] - 1))
    results.append({'tensor':tensor,'shape':fixture['shape'],'paths':paths,
                    'paired_host_complete_percent_change':paired,
                    'ratio_of_medians_percent_change':100*(paths['packed']['host_complete_us']['median']/paths['original-control']['host_complete_us']['median']-1),
                    'numerical_outputs_checked':fixture['outputs_checked'],
                    'numerical_max_abs_error':max(x['max_abs_error'] for x in fixture['cases'].values()),
                    'numerical_max_error_over_bound':max(x['max_error_over_bound'] for x in fixture['cases'].values())})
inputs = ['run-clean/oracle.stderr','run-clean/closure.json','cpu-full/gate.json']
report = {'status':'MICROPROBE_ONLY_NO_INTEGRATION', 'inputs_sha256':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in inputs},
          'timing_units':'microseconds per call derived from 16-call batch means, 3 batches per path',
          'host_complete_scope':'submission through final completion, same in-order queue',
          'event_scope':'same-queue barrier span includes submission gaps/barrier cost; not isolated kernel service',
          'conditions':'one process/card1, warmed16 each, AB/BA/AB, no unitrace, profiling-enabled queue, no server',
          'fixtures':results, 'verdict':'Do not integrate packed Q6 owner: no demonstrated benefit and three large regressions in this warmed C6 discriminator. Preserve source-only S46. Not a serving regression or all-shape result.'}
(P / 'warm-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='ascii')
for x in results:
    print(x['tensor'],x['shape'],'original_us',round(x['paths']['original-control']['host_complete_us']['median'],3),'packed_us',round(x['paths']['packed']['host_complete_us']['median'],3),'change_percent',round(x['ratio_of_medians_percent_change'],3),'max_error',x['numerical_max_abs_error'])
