#!/usr/bin/env python3
# dd_loadprobe.py -- ramp ANCH long streaming anchors, then fire P CONCURRENT non-stream probes under
# that load and dump each RAW message, so we see exactly what degenerate-under-load output is.
#   usage: dd_loadprobe.py [port] [anchors] [probes] [probe_maxtok]
import json, subprocess, urllib.request, threading, time, sys
KEY = open('/mnt/vm_8tb/b70/secrets/dd_api_key').read().strip()
PORT  = int(sys.argv[1]) if len(sys.argv) > 1 else 18091
ANCH  = int(sys.argv[2]) if len(sys.argv) > 2 else 8
PROBES = int(sys.argv[3]) if len(sys.argv) > 3 else 12
PMAX  = int(sys.argv[4]) if len(sys.argv) > 4 else 500

import os
BASEFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backhoe_req.json')
base = json.load(open(BASEFILE))
base.pop('stream_options', None); base.pop('store', None)

def req(maxtok, stream):
    r = json.loads(json.dumps(base)); r['max_tokens'] = maxtok; r['stream'] = stream; return r
def U(r):
    return urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % PORT,
        data=json.dumps(r).encode(),
        headers={'Authorization': 'Bearer %s' % KEY, 'Content-Type': 'application/json'})
def anchor():
    try:
        with urllib.request.urlopen(U(req(3000, True)), timeout=300) as resp:
            for _ in resp:
                pass
    except Exception:
        pass

out = {}
lock = threading.Lock()
def probe(i):
    t0 = time.time()
    try:
        with urllib.request.urlopen(U(req(PMAX, False)), timeout=300) as resp:
            d = json.load(resp)
        ch = d['choices'][0]; m = ch['message']
        blob = (m.get('content') or '') + ''.join((t.get('function', {}).get('arguments') or '')
                for t in (m.get('tool_calls') or []))
        with lock:
            out[i] = (time.time() - t0, ch.get('finish_reason'), d.get('usage', {}).get('completion_tokens'),
                      repr((m.get('content') or '')[:120]),
                      repr((m.get('reasoning_content') or '')[:80]),
                      [(t.get('function', {}).get('name'), repr((t.get('function', {}).get('arguments') or '')[:80]))
                       for t in (m.get('tool_calls') or [])],
                      len(blob))
    except Exception as e:
        with lock:
            out[i] = (time.time() - t0, 'ERR', None, str(e), '', [], 0)

print('ramping %d anchors on :%d ...' % (ANCH, PORT))
for _ in range(ANCH):
    threading.Thread(target=anchor, daemon=True).start()
time.sleep(5)
print('firing %d CONCURRENT probes (maxtok=%d) under load ...' % (PROBES, PMAX))
ts = [threading.Thread(target=probe, args=(i,)) for i in range(PROBES)]
for t in ts: t.start()
for t in ts: t.join()
for i in sorted(out):
    lat, fin, ctoks, content, reason, tcs, blen = out[i]
    flag = 'OK' if blen >= 8 else '*** DEGEN ***'
    print('probe %2d %6.1fs finish=%-10s ctoks=%s blob=%d %s' % (i, lat, fin, ctoks, blen, flag))
    print('     content:%s reasoning:%s tools:%s' % (content, reason, tcs))
