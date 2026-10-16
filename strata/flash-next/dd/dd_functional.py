#!/usr/bin/env python3
"""hotschmoe-dd functional checks through the frontdoor (API key): thinking, tools round trip, concurrency, long."""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE = 'http://127.0.0.1:18080/v1'
KEY = open('/mnt/vm_8tb/b70/secrets/dd_api_key').read().strip()

def post(body, stream=False):
    req = urllib.request.Request(BASE + '/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY})
    with urllib.request.urlopen(req, timeout=900) as r:
        if not stream: return json.load(r)
        text, n, t0, first = [], 0, time.time(), None
        for line in r:
            if line.startswith(b'data: ') and b'[DONE]' not in line:
                ch = json.loads(line[6:])
                for c in ch.get('choices', []):
                    d = c.get('delta', {}).get('content')
                    if d: text.append(d); first = first or time.time()
                if ch.get('usage'): n = ch['usage']['completion_tokens']
        return ''.join(text), n, (time.time() - (first or t0))

res = {}
# 1 thinking (template default) on a small reasoning problem
r = post(dict(model='hotschmoe-dd', max_tokens=2048, temperature=0.6,
              messages=[{'role': 'user', 'content': 'A train leaves at 9:40 and arrives at 13:05. How many minutes is the trip? Answer with the number only at the end.'}]))
m = r['choices'][0]['message']
res['thinking'] = dict(reasoning_chars=len(m.get('reasoning_content') or ''), answer=(m.get('content') or '')[-40:], ok='205' in (m.get('content') or ''))
# 2 tool call + round trip
tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
msgs = [{'role': 'user', 'content': 'What is the weather in Boise right now? Use the tool.'}]
r = post(dict(model='hotschmoe-dd', max_tokens=1024, temperature=0, tools=tools, messages=msgs,
              chat_template_kwargs={'enable_thinking': False}))
m = r['choices'][0]['message']; tc = m.get('tool_calls') or []
res['tool_call'] = dict(n=len(tc), name=tc[0]['function']['name'] if tc else None, args=tc[0]['function']['arguments'] if tc else None)
if tc:
    msgs += [m, {'role': 'tool', 'tool_call_id': tc[0].get('id', 'call_0'), 'content': json.dumps({'city': 'Boise', 'temp_f': 58, 'sky': 'clear'})}]
    r2 = post(dict(model='hotschmoe-dd', max_tokens=512, temperature=0, tools=tools, messages=msgs, chat_template_kwargs={'enable_thinking': False}))
    c2 = r2['choices'][0]['message'].get('content') or ''
    res['tool_roundtrip'] = dict(ok='58' in c2, text=c2[:120])
# 3 concurrent coherence (4 at once)
qs = [(17, 19, '323'), (23, 29, '667'), (31, 37, '1147'), (41, 43, '1763')]
with ThreadPoolExecutor(4) as ex:
    outs = list(ex.map(lambda q: post(dict(model='hotschmoe-dd', max_tokens=32, temperature=0, chat_template_kwargs={'enable_thinking': False},
                     messages=[{'role': 'user', 'content': f'Return only the integer: {q[0]} * {q[1]}.'}]))['choices'][0]['message']['content'].strip(), qs))
res['concurrent'] = dict(ok=outs == [q[2] for q in qs], outs=outs)
# 4 long streamed answer
text, n, dt = post(dict(model='hotschmoe-dd', max_tokens=2048, temperature=0, stream=True, stream_options={'include_usage': True},
                        chat_template_kwargs={'enable_thinking': False},
                        messages=[{'role': 'user', 'content': 'Write a complete, well-commented Python implementation of an LRU cache with TTL expiry, plus pytest tests.'}]), stream=True)
res['long'] = dict(tokens=n, tok_s=round((n - 1) / dt, 1) if dt > 0 else None, has_code='def ' in text and 'class ' in text)
print(json.dumps(res, indent=1))
