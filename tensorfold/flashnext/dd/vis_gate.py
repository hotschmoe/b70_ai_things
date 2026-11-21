#!/usr/bin/env python3
"""VIS gates for gpu_window.sh (worker VIS): image input through the serving path on the window server.

  text  URL OUT                 fixed text prompt, greedy, thinking off, 128 tokens -> OUT {"hash", "tokens"}
  same  A B                     two text records' hashes are equal (vision handling leaves text bit-identical)
  image URL FIXTURE LOG OUT     the fixture's circle answers "blue" (streamed: TTFT), twice (second = encoder cache);
                                the server log's encoder lines give image tokens and encode ms
Each prints one GATE line and exits 0 on PASS, 1 on FAIL. Never prints prompt or response text beyond one color word.
"""
import base64, hashlib, json, re, sys, time, urllib.request

MODEL = 'hotschmoe-dd'
TEXT = ('List the planets of the solar system in order from the Sun, one per line, '
        'each with one short fact about it.')


def gate(name, ok, detail):
    print(f"GATE {name} {'PASS' if ok else 'FAIL'} {json.dumps(detail)}")
    sys.exit(0 if ok else 1)


def post(url, body):
    req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t0 = time.time()
    first, parts, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=170) as r:
        for line in r:
            if not line.startswith(b'data: ') or b'[DONE]' in line:
                continue
            ch = json.loads(line[6:])
            for c in ch.get('choices', []):
                d = c.get('delta', {})
                piece = (d.get('content') or '') + (d.get('reasoning_content') or '')
                if piece:
                    first = first or time.time()
                    parts.append(piece)
            if ch.get('usage'):
                usage = ch['usage']
    return ''.join(parts), usage, (first - t0) if first else None, time.time() - t0


def text(url, out):
    body = dict(model=MODEL, max_tokens=128, temperature=0, stream=True, stream_options={'include_usage': True},
                chat_template_kwargs={'enable_thinking': False}, messages=[{'role': 'user', 'content': TEXT}])
    s, usage, ttft, _ = post(url, body)
    rec = {'hash': hashlib.sha256(s.encode()).hexdigest()[:12], 'tokens': usage.get('completion_tokens'),
           'ttft_s': round(ttft, 3) if ttft else None}
    json.dump(rec, open(out, 'w'))
    gate('vis-text', bool(s) and (usage.get('completion_tokens') or 0) > 0, rec)


def same(a, b):
    ra, rb = json.load(open(a)), json.load(open(b))
    gate('vis-text-same', ra['hash'] == rb['hash'] and ra['tokens'] == rb['tokens'], {'before': ra, 'after': rb})


def encoder_lines(log):
    pat = re.compile(r'flashnext vision: image (\d+) tokens \((\d+)x(\d+)\) encoded in (\d+) ms')
    try:
        return [tuple(int(x) for x in m.groups()) for m in pat.finditer(open(log, errors='replace').read())]
    except OSError:
        return []


def image(url, fixture, log, out):
    data = 'data:image/png;base64,' + base64.b64encode(open(fixture, 'rb').read()).decode('ascii')
    body = dict(model=MODEL, max_tokens=32, temperature=0, stream=True, stream_options={'include_usage': True},
                chat_template_kwargs={'enable_thinking': False},
                messages=[{'role': 'user', 'content': [
                    {'type': 'text', 'text': 'What color is the circle? Answer with one color word.'},
                    {'type': 'image_url', 'image_url': {'url': data}}]}])
    before = len(encoder_lines(log))
    runs = []
    for _ in range(2):
        s, usage, ttft, total = post(url, body)
        word = re.sub(r'[^a-z]', '', s.lower())[:16]
        runs.append({'blue': 'blue' in s.lower(), 'word': word, 'prompt_tokens': usage.get('prompt_tokens'),
                     'ttft_s': round(ttft, 3) if ttft else None, 'total_s': round(total, 3)})
    enc = encoder_lines(log)[before:]
    rec = {'runs': runs, 'encoder': [{'tokens': e[0], 'grid': f'{e[1]}x{e[2]}', 'ms': e[3]} for e in enc]}
    json.dump(rec, open(out, 'w'))
    ok = (all(r['blue'] for r in runs) and runs[0]['word'] == runs[1]['word'] and len(enc) == 2 and
          all(e[0] > 0 and e[0] == e[1] * e[2] for e in enc) and
          all((r['prompt_tokens'] or 0) > enc[0][0] for r in runs))
    gate('vis-image', ok, rec)


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    {'text': text, 'same': same, 'image': image}[cmd](*args)
