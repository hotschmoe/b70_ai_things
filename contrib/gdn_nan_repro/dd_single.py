#!/usr/bin/env python3
# dd_single.py -- fire ONE long decode with NO other load, check it stays coherent (is the bug purely
# a CONCURRENCY/mixing thing, or do long single-stream decodes NaN on their own?).
import json, urllib.request, os, sys, time
KEY = open('/mnt/vm_8tb/b70/secrets/dd_api_key').read().strip()
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
base = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backhoe_req.json')))
base.pop('stream_options', None); base.pop('store', None)
base['max_tokens'] = MAX; base['stream'] = False
r = urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % PORT,
    data=json.dumps(base).encode(), headers={'Authorization': 'Bearer %s' % KEY, 'Content-Type': 'application/json'})
t = time.time()
d = json.load(urllib.request.urlopen(r, timeout=400))
ch = d['choices'][0]; m = ch['message']
blob = (m.get('content') or '') + ''.join((tc.get('function', {}).get('arguments') or '') for tc in (m.get('tool_calls') or []))
print('single %d-tok decode ALONE: finish=%s ctoks=%s blob_len=%d (%.1fs)' % (
    MAX, ch.get('finish_reason'), d.get('usage', {}).get('completion_tokens'), len(blob), time.time() - t))
print('VERDICT:', 'COHERENT' if len(blob.strip()) > 50 else '*** DEGEN ***')
print('tail:', repr(blob[-120:]))
