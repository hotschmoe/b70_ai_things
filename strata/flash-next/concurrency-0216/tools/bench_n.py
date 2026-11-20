#!/usr/bin/env python3
"""Bounded concurrent OpenAI workload with client-side latency evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path('/mnt/vm_8tb/github/b70_ai_things')
sys.path.insert(0, str(ROOT / 'strata/flash-next'))
import bench_serving as bench


def pct(values, p):
    values = sorted(values)
    if not values:
        return None
    q = (len(values)-1)*p
    i = int(q)
    return values[i] + (values[min(i+1, len(values)-1)]-values[i])*(q-i)


def clock_sample(label):
    a=time.time_ns(); raw=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW); boot=time.clock_gettime_ns(time.CLOCK_BOOTTIME); mono=time.clock_gettime_ns(time.CLOCK_MONOTONIC); b=time.time_ns()
    return {'label':label,'real_before_ns':a,'real_after_ns':b,'raw_ns':raw,'boot_ns':boot,'monotonic_ns':mono}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--concurrency', type=int, required=True)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    base = 'http://127.0.0.1:18134'
    ids = [r['id'] for r in bench.get(base, '/v1/models')['data']]
    alias = 'qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata-0216'
    assert ids[0] == 'hotschmoe-dd' and alias in ids
    assert alias in (ROOT/'evals/configs/models.yaml').read_text()
    slots = bench.get(base, '/slots')
    (a.output/'identity-slots.json').write_text(json.dumps({'ids': ids, 'slots': slots}, indent=2, ensure_ascii=True)+'\n')
    for p, expected in [('Return only the integer: 17 * 19.', '323'),
                        ('Return only the integer: 23 * 29.', '667')]:
        check = bench.send(base, 'hotschmoe-dd', [{'role':'user','content':p}], 32)
        if check['text'].strip() != expected:
            raise RuntimeError('arithmetic coherence failed: ' + check['text'])
    template = json.loads(Path('/mnt/vm_8tb/b70/results/conc-0216/a-baseline-c1/coding-0.json').read_text())['request']['messages'][0]['content'].split('] ',1)[1]
    rounds=[]
    clock_marks=[]
    for repeat in range(a.repeats):
        clock_marks.append(clock_sample('repeat_'+str(repeat)+'_begin'))
        gate = threading.Barrier(a.concurrency)
        def worker(i):
            marks = {}
            content = f'[Flash-Next N={a.concurrency} repeat={repeat} slot={i}] ' + template
            gate.wait(timeout=60)
            marks['start'] = time.perf_counter()
            row=bench.send(base, 'hotschmoe-dd', [{'role':'user','content':content}], a.max_tokens,
                           on_first_content=lambda: marks.update(first=time.perf_counter()))
            marks['end'] = time.perf_counter()
            row['common_clock'] = marks
            row['slot_index'] = i
            return row
        with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            rows = list(pool.map(worker, range(a.concurrency)))
        clock_marks.append(clock_sample('repeat_'+str(repeat)+'_end'))
        (a.output/'clock-marks.json').write_text(json.dumps(clock_marks,indent=2,ensure_ascii=True)+'\n')
        (a.output/f'repeat-{repeat}.json').write_text(json.dumps(rows,indent=2,ensure_ascii=True)+'\n')
        earliest = min(r['common_clock']['first'] for r in rows)
        last = max(r['common_clock']['end'] for r in rows)
        ttft = [r['ttft_s'] for r in rows]
        stats = {'repeat':repeat, 'offered_concurrency':a.concurrency, 'actual_slots':len(slots),
                 'ttft_s':ttft, 'ttft_p50_s':pct(ttft,.5), 'ttft_p95_s':pct(ttft,.95),
                 'per_stream_event_gap_ms':[r['content_event_gap_ms'] for r in rows],
                 'completion_tokens':[r['usage']['completion_tokens'] for r in rows],
                 'native_prompt_ms':[r['timings'].get('prompt_ms') for r in rows],
                 'client_first_minus_native_prompt_s':[r['ttft_s']-r['timings'].get('prompt_ms',0)/1000 for r in rows],
                 'decode_aggregate_tps':sum(r['usage']['completion_tokens']-1 for r in rows)/(last-earliest),
                 'client_decode_tps':[r['client_decode_tps'] for r in rows],
                 'hashes':[r['output_sha256'] for r in rows],
                 'event_gap_note':'SSE content events may contain multiple tokens; event gaps are not per-token ITL.',
                 'admission_note':'TTFT minus native prompt time is a residual including queueing, Python and network, not isolated admission.'}
        rounds.append(stats)
        print(json.dumps(stats,ensure_ascii=True),flush=True)
    (a.output/'summary.json').write_text(json.dumps({'rounds':rounds, 'health':bench.get(base,'/health')},indent=2,ensure_ascii=True)+'\n')


if __name__=='__main__':
    main()
