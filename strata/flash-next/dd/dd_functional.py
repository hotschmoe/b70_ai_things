#!/usr/bin/env python3
"""Functional checks for a Strata endpoint: thinking, tools, concurrency, images and long output."""
import argparse, base64, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url', default='http://127.0.0.1:18080/v1')
parser.add_argument('--api-key-file', default='/mnt/vm_8tb/b70/secrets/dd_api_key')
parser.add_argument('--no-api-key', action='store_true')
parser.add_argument('--image-fixture', type=Path, help='PNG with a blue circle, for the vision check')
args = parser.parse_args()
BASE = args.base_url.rstrip('/')
KEY = '' if args.no_api_key else Path(args.api_key_file).read_text().strip()

def post(body, stream=False):
    headers = {'Content-Type': 'application/json'}
    if KEY:
        headers['Authorization'] = 'Bearer ' + KEY
    req = urllib.request.Request(BASE + '/chat/completions', json.dumps(body).encode(), headers)
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
if args.image_fixture:
    url = 'data:image/png;base64,' + base64.b64encode(args.image_fixture.read_bytes()).decode('ascii')
    r = post(dict(model='hotschmoe-dd', max_tokens=32, temperature=0,
                  chat_template_kwargs={'enable_thinking': False},
                  messages=[{'role': 'user', 'content': [
                      {'type': 'text', 'text': 'What color is the circle? Answer with one color word.'},
                      {'type': 'image_url', 'image_url': {'url': url}}]}]))
    answer = r['choices'][0]['message'].get('content') or ''
    res['vision'] = dict(ok='blue' in answer.lower(), answer=answer)
# 4 long streamed answer
text, n, dt = post(dict(model='hotschmoe-dd', max_tokens=2048, temperature=0, stream=True, stream_options={'include_usage': True},
                        chat_template_kwargs={'enable_thinking': False},
                        messages=[{'role': 'user', 'content': 'Write a complete, well-commented Python implementation of an LRU cache with TTL expiry, plus pytest tests.'}]), stream=True)
res['long'] = dict(tokens=n, tok_s=round((n - 1) / dt, 1) if dt > 0 else None, has_code='def ' in text and 'class ' in text)
print(json.dumps(res, indent=1))
if not (res['thinking']['ok'] and res['thinking']['reasoning_chars'] > 0 and
        res['tool_call']['name'] == 'get_weather' and res.get('tool_roundtrip', {}).get('ok') and
        res['concurrent']['ok'] and res['long']['has_code'] and res['long']['tokens'] >= 512 and
        (not args.image_fixture or res['vision']['ok'])):
    raise SystemExit('Functional check failed')
