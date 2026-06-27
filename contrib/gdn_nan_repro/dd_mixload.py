#!/usr/bin/env python3
# dd_mixload.py -- reproduce the under-load degenerate-output ("!!!!"/empty) failure by driving
# SUSTAINED MIXED load at ONE replica: long streaming "anchor" decodes running WHILE repeated bursts
# of new prefills are injected (mimics an agent swarm: long parent turns + bursty subagent spawns).
# Classifies every response for the degenerate signature.
#   usage: dd_mixload.py [port] [anchors] [burst] [waves] [interval_s] [burst_maxtok] [anchor_maxtok]
import json, subprocess, sys, urllib.request, threading, time
from collections import Counter

KEY = open('/mnt/vm_8tb/b70/secrets/dd_api_key').read().strip()
PORT     = int(sys.argv[1]) if len(sys.argv) > 1 else 18091
ANCH     = int(sys.argv[2]) if len(sys.argv) > 2 else 4
BURST    = int(sys.argv[3]) if len(sys.argv) > 3 else 6
WAVES    = int(sys.argv[4]) if len(sys.argv) > 4 else 20
INTERVAL = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
BMAX     = int(sys.argv[6]) if len(sys.argv) > 6 else 500
AMAX     = int(sys.argv[7]) if len(sys.argv) > 7 else 2500

import os
BASEFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backhoe_req.json')
base = json.load(open(BASEFILE))
base.pop('stream_options', None); base.pop('store', None)

def make(maxtok, stream):
    r = json.loads(json.dumps(base))                 # deep copy (APC is off -> identical reqs don't share KV)
    r['max_tokens'] = maxtok
    r['stream'] = stream
    return r

def degen(blob, finish):
    s = (blob or '').strip()
    if not s:
        return 'DEGEN_EMPTY(finish=%s)' % finish
    if len(s) >= 16:
        ch, n = Counter(s).most_common(1)[0]
        if n / len(s) >= 0.6:
            return 'GARBAGE(%r x%d%%)' % (ch, int(100 * n / len(s)))
    return None

results = []
lock = threading.Lock()
def rec(label, lat, v):
    with lock:
        results.append((label, lat, v))

def url(req):
    return urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % PORT,
        data=json.dumps(req).encode(),
        headers={'Authorization': 'Bearer %s' % KEY, 'Content-Type': 'application/json'})

def non_stream(label, variant, maxtok):
    t0 = time.time()
    try:
        with urllib.request.urlopen(url(make(maxtok, False)), timeout=300) as resp:
            d = json.load(resp)
        ch = d['choices'][0]; m = ch['message']
        blob = (m.get('content') or '') + (m.get('reasoning_content') or '') + ''.join(
            (t.get('function', {}).get('arguments') or '') for t in (m.get('tool_calls') or []))
        rec(label, time.time() - t0, degen(blob, ch.get('finish_reason')) or 'OK(%d)' % len(blob))
    except Exception as e:
        rec(label, time.time() - t0, 'HTTP_ERR %s' % type(e).__name__)

def stream_anchor(label, variant, maxtok):
    t0 = time.time(); acc = []
    try:
        with urllib.request.urlopen(url(make(maxtok, True)), timeout=300) as resp:
            for line in resp:
                line = line.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                try:
                    delta = json.loads(data)['choices'][0].get('delta', {})
                    acc.append(delta.get('content') or delta.get('reasoning_content') or '')
                except Exception:
                    pass
        blob = ''.join(acc)
        rec(label, time.time() - t0, degen(blob, 'stream') or 'OK(%d)' % len(blob))
    except Exception as e:
        rec(label, time.time() - t0, 'HTTP_ERR %s' % type(e).__name__)

print('MIXLOAD port=%d anchors=%d burst=%dx%d every %.1fs (bmax=%d amax=%d)' % (
    PORT, ANCH, BURST, WAVES, INTERVAL, BMAX, AMAX))
threads = []
for i in range(ANCH):
    t = threading.Thread(target=stream_anchor, args=('anchor%d' % i, 'A%d' % i, AMAX)); t.start(); threads.append(t)
for w in range(WAVES):
    for b in range(BURST):
        t = threading.Thread(target=non_stream, args=('w%02db%d' % (w, b), '%d-%d' % (w, b), BMAX)); t.start(); threads.append(t)
    time.sleep(INTERVAL)
for t in threads:
    t.join()

hist = Counter(v.split('(')[0] for _, _, v in results)
bad = [r for r in results if not r[2].startswith('OK')]
print('--- %d requests; verdicts: %s' % (len(results), dict(hist)))
for label, lat, v in sorted(bad)[:50]:
    print('  BAD %-9s %6.1fs %s' % (label, lat, v))
print('TOTALS  GARBAGE=%d  DEGEN_EMPTY=%d  HTTP_ERR=%d  OK=%d' % (
    sum(v.startswith('GARBAGE') for _, _, v in results),
    sum(v.startswith('DEGEN_EMPTY') for _, _, v in results),
    sum(v.startswith('HTTP_ERR') for _, _, v in results),
    sum(v.startswith('OK') for _, _, v in results)))
