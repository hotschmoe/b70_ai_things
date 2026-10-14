#!/usr/bin/env python3
"""A fresh 32K prefill arrives after another stream has begun a long decode.

Use the already-leased server. Preserve all SSE arrival times; gaps are content
events, potentially several tokens each. Each arm uses the same fresh prompts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

from bench_serving import get, send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:18085')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rstrip('/')
    identity = get(base, '/v1/models')
    alias = 'qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata'
    ids = [r['id'] for r in identity['data']]
    registry = (Path(__file__).resolve().parents[2]/'evals/configs/models.yaml').read_text()
    if ids[0] != 'hotschmoe-dd' or alias not in ids or alias not in registry:
        raise RuntimeError('Wrong model identity')
    (args.output/'models.json').write_text(json.dumps(identity, indent=2)+'\n')
    source = json.loads(args.template.read_text())['request']
    prompt = source['messages'][0]['content'].split('] ', 1)[1]
    summaries = []
    for repeat in range(args.repeats):
        first = threading.Event()
        epoch = time.perf_counter()
        starts = {}
        def request(label, content, cap, signal=None):
            starts[label] = time.perf_counter()-epoch
            row = send(base, 'hotschmoe-dd', [{'role':'user', 'content':content}], cap,
                       on_first_content=signal)
            row['start_offset_s'] = starts[label]
            if row['timings']['cache_n'] > 64:
                raise RuntimeError('Fresh mixed request reused more than 64 tokens')
            (args.output/f'{label}-{repeat}.json').write_text(json.dumps(row, indent=2)+'\n')
            return row
        with ThreadPoolExecutor(max_workers=2) as pool:
            decoder = pool.submit(request, 'decode', f'[Mixed decode repeat {repeat}] '+prompt,
                                  source['max_tokens'], first.set)
            if not first.wait(timeout=180):
                if decoder.done():
                    decoder.result()
                raise RuntimeError('Decoder did not emit first content within 180 seconds')
            if decoder.done():
                raise RuntimeError('Decoder completed before prefill arrival')
            prefill = pool.submit(request, 'prefill', f'[Mixed prefill repeat {repeat}] '+
                                  'The history of computing is long and full of surprising turns. '*2724+
                                  '\nExplain GPU memory hierarchies in detail.', 256)
            a, b = decoder.result(), prefill.result()
        start_b = starts['prefill']
        if not starts['decode']+a['ttft_s'] < start_b < starts['decode']+a['wall_s']:
            raise RuntimeError('Long prefill did not arrive during decode')
        gaps = a['content_event_gap_ms']
        # Gaps intersecting the prefill request expose the client-visible interruption.
        events = a['content_events']
        overlapping = [(y['elapsed_s']-x['elapsed_s'])*1000
                       for x,y in zip(events,events[1:])
                       if starts['decode']+y['elapsed_s'] >= start_b
                       and starts['decode']+x['elapsed_s'] <= start_b+b['wall_s']]
        summary = dict(repeat=repeat, prefill_arrival_s=start_b,
                       decoder_ttft_s=a['ttft_s'], decoder_wall_s=a['wall_s'],
                       decoder_content_event_gap_ms=gaps,
                       max_decode_gap_overlapping_prefill_ms=max(overlapping) if overlapping else None,
                       prefill_ttft_s=b['ttft_s'], prefill_wall_s=b['wall_s'],
                       timings=[a['timings'],b['timings']], hashes=[a['output_sha256'],b['output_sha256']])
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (args.output/'health-end.json').write_text(json.dumps(get(base,'/health'),indent=2)+'\n')

if __name__ == '__main__':
    main()
