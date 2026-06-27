#!/usr/bin/env python3
# dd_rawtokens.py -- under load, fire one probe with logprobs and dump the LITERAL tokens the model
# emitted (bypasses reasoning/tool parsing), to see exactly what the degenerate output is.
import json, urllib.request, threading, time, os, sys
KEY = open('/mnt/vm_8tb/b70/secrets/dd_api_key').read().strip()
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
ANCH = int(sys.argv[2]) if len(sys.argv) > 2 else 8
PMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 40
base = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backhoe_req.json')))
base.pop('stream_options', None); base.pop('store', None)

def req(maxtok, stream, logprobs=False):
    r = json.loads(json.dumps(base)); r['max_tokens'] = maxtok; r['stream'] = stream
    if logprobs:
        r['logprobs'] = True; r['top_logprobs'] = 1
        r.pop('tools', None); r.pop('tool_choice', None)   # tools + logprobs can 400
    return r
def U(r):
    return urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % PORT,
        data=json.dumps(r).encode(), headers={'Authorization': 'Bearer %s' % KEY, 'Content-Type': 'application/json'})
def anchor():
    try:
        with urllib.request.urlopen(U(req(3000, True)), timeout=300) as resp:
            for _ in resp:
                pass
    except Exception:
        pass

print('ramping %d anchors...' % ANCH)
for _ in range(ANCH):
    threading.Thread(target=anchor, daemon=True).start()
time.sleep(5)
print('firing 1 logprobs probe under load...')
try:
    with urllib.request.urlopen(U(req(PMAX, False, logprobs=True)), timeout=400) as resp:
        d = json.load(resp)
except urllib.error.HTTPError as e:
    print('HTTP', e.code, e.read().decode()[:400]); raise SystemExit
ch = d['choices'][0]
print('finish=%s  usage=%s' % (ch.get('finish_reason'), d.get('usage')))
toks = (ch.get('logprobs') or {}).get('content') or []
print('num tokens emitted: %d' % len(toks))
from collections import Counter
ids = [t.get('token') for t in toks]
print('first 40 tokens (repr):')
for t in toks[:40]:
    print('  %r  logprob=%.4f' % (t.get('token'), t.get('logprob')))
if ids:
    c = Counter(ids).most_common(3)
    print('most common tokens:', c)
print('parsed message content repr:', repr((ch['message'].get('content') or '')[:80]))
