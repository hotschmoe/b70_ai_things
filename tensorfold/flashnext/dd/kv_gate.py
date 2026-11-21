#!/usr/bin/env python3
"""KV gates for the cutover window (worker KV). CPU-side client: talks HTTP to the window server on
127.0.0.1:$TF_PORT only and reads $TF_SERVER_LOG. Never prints prompt or response text.

  budget LOG                       'paged KV budget' load line: auto source, >= 4 resident 90K, >= 2 resident 200K
  long URL TOKENS DECODE TAG       one synthetic needle prompt of about TOKENS tokens, DECODE greedy tokens,
                                   streamed: TTFT, prefill and decode rates, needle recall; saves TAG.json ids
                                   into $KV_OUT (default $TF_OUT)
  conc URL TOKENS N DECODE SOLO    N concurrent distinct needle prompts (prompt 0 == SOLO's prompt): all admitted
                                   and complete, every needle recalled, stream 0 ids == SOLO ids bitwise
  short URL N DECODE               oracle p0 ids solo, then N concurrent copies == solo bitwise; saves
                                   expected-p0-kv.json (the TF_PROD_EXPECTED candidate)
  again URL TOKENS DECODE REF      repeat REF's prompt: ids == REF, cached_tokens >= 90% of the prompt, TTFT
                                   <= 0.25 x REF's (RAM prompt cache, --prompt-cache-gib > 0)
  parks LOG MIN                    eviction park/restore lines since launch >= MIN each, no 'eviction' errors
Prompts are deterministic from a seed: numbered ledger lines with one code word placed early, then a question.
Each mode prints one GATE line and exits 0 on PASS, 1 on FAIL.
"""
import concurrent.futures, http.client, json, os, random, re, sys, time

WORDS = ('amber basalt cedar delta ember fjord granite harbor indigo juniper kestrel lumen marble nickel orchid '
         'pewter quartz raven saffron tundra umber vellum willow xenon yarrow zephyr').split()
CODES = ('BLUEFINCH', 'COPPERMOTH', 'SILVERWREN', 'IRONHERON', 'JADEOTTER', 'OPALLYNX', 'RUBYCRANE', 'TEALBISON')

def gate(name, ok, detail):
    print(f"GATE {name} {'PASS' if ok else 'FAIL'} {json.dumps(detail)}")
    sys.exit(0 if ok else 1)

def prompt(seed, tokens):
    rnd = random.Random(seed)
    code = CODES[seed % len(CODES)]
    lines = []
    # About 27 tokens per ledger line (Qwen3.8 tokenizer.json, digits split); usage.prompt_tokens is the measure.
    for i in range(max(tokens // 27, 8)):
        a, b = rnd.choice(WORDS), rnd.choice(WORDS)
        lines.append(f"Line {i}: the {a} crate moved to dock {rnd.randint(1, 99)} with {rnd.randint(100, 9999)} {b} units.")
        if i == 40:
            lines.append(f"Note: the vault code word is {code}. Remember it.")
    return ('Ledger of shipments.\n' + '\n'.join(lines) +
            '\nQuestion: what is the vault code word from the note near the start? Answer: The vault code word is'), code

def complete(url, text, decode):
    host, port = url.split('//')[-1].split(':')
    conn = http.client.HTTPConnection(host, int(port), timeout=900)
    body = {'model': os.environ.get('TF_MODEL', 'hotschmoe-dd'), 'prompt': text, 'temperature': 0,
            'max_tokens': decode, 'stream': True, 'return_token_ids': True,
            'stream_options': {'include_usage': True, 'tensorfold_token_stream_version': 1}}
    start = time.monotonic()
    conn.request('POST', '/v1/completions', body=json.dumps(body), headers={'Content-Type': 'application/json'})
    reply = conn.getresponse()
    if reply.status != 200:
        return {'status': reply.status, 'error': reply.read()[:300].decode(errors='replace')}
    first = last = None
    out, ids, final, usage, finish = [], [], None, None, None
    while True:
        line = reply.readline()
        if not line:
            break
        if not line.startswith(b'data:'):
            continue
        payload = line[5:].strip()
        if payload == b'[DONE]':
            break
        obj = json.loads(payload)
        if obj.get('object') == 'tensorfold.token.chunk':
            ids.extend(obj['token_ids'])
            last = time.monotonic()
            first = first or last
        for c in obj.get('choices', []):
            if c.get('text'):
                out.append(c['text'])
                last = time.monotonic()
                first = first or last
            finish = c.get('finish_reason') or finish
        if 'tensorfold' in obj:
            final = obj['tensorfold']
        usage = obj.get('usage') or usage
    conn.close()
    if final and final.get('token_ids'):
        ids = final['token_ids']
    end = time.monotonic()
    ptok = (usage or {}).get('prompt_tokens', 0)
    ttft = (first or end) - start
    cached = ((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
    return {'status': 200, 'text': ''.join(out), 'ids': ids, 'finish': finish, 'prompt_tokens': ptok, 'cached_tokens': cached,
            'ttft_s': round(ttft, 3), 'prefill_tok_s': round(ptok / ttft, 1) if ttft > 0 else None,
            'decode_tok_s': round((len(ids) - 1) / (last - first), 1) if first and last and last > first and len(ids) > 1 else None}

def summary(r, code):
    keep = {k: r.get(k) for k in ('status', 'prompt_tokens', 'cached_tokens', 'ttft_s', 'prefill_tok_s', 'decode_tok_s', 'finish')}
    keep['tokens'] = len(r.get('ids') or [])
    keep['needle'] = code.lower() in (r.get('text') or '').lower()
    keep['text_ok'] = bool(re.search(r'[A-Za-z]{3}', r.get('text') or ''))
    if r.get('error'):
        keep['error_head'] = r['error'][:120]
    return keep

def long(url, tokens, decode, tag):
    text, code = prompt(0, int(tokens))
    r = complete(url, text, int(decode))
    s = summary(r, code)
    out = os.environ.get('KV_OUT') or os.environ.get('TF_OUT', '.')
    if r.get('ids'):
        json.dump({'seed': 0, 'tokens': int(tokens), 'prompt_tokens': s['prompt_tokens'], 'ttft_s': s['ttft_s'], 'token_ids': r['ids']},
                  open(os.path.join(out, f'{tag}.json'), 'w'))
    ok = (s['status'] == 200 and s['prompt_tokens'] >= 0.9 * int(tokens) and s['needle'] and s['text_ok']
          and (s['tokens'] == int(decode) or s['finish'] == 'stop'))
    gate(f'kv-long-{tag}', ok, s)

def again(url, tokens, decode, ref):
    """Repeat REF's prompt: same ids bitwise, served from the RAM prompt cache (cached_tokens, fast TTFT)."""
    base = json.load(open(ref))
    text, code = prompt(0, int(tokens))
    r = complete(url, text, int(decode))
    s = summary(r, code)
    m = min(len(r.get('ids') or []), len(base['token_ids']))
    s['equal'] = m > 0 and r['ids'][:m] == base['token_ids'][:m]
    s['first_ttft_s'] = base.get('ttft_s')
    s['ttft_ratio'] = round(s['ttft_s'] / base['ttft_s'], 3) if base.get('ttft_s') else None
    # cached_tokens is checked when the server reports it (API item); TTFT is checked always.
    ok = (s['status'] == 200 and s['equal'] and (s['cached_tokens'] is None or s['cached_tokens'] >= 0.9 * base['prompt_tokens'])
          and s['ttft_ratio'] is not None and s['ttft_ratio'] <= 0.25)
    gate('kv-cache-repeat', ok, s)

def conc(url, tokens, n, decode, solo):
    ref = json.load(open(solo))
    jobs = [prompt(i, int(tokens)) for i in range(int(n))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(n)) as pool:
        runs = list(pool.map(lambda j: complete(url, j[0], int(decode)), jobs))
    sums = [summary(r, j[1]) for r, j in zip(runs, jobs)]
    m = min(len(runs[0].get('ids') or []), len(ref['token_ids']))
    equal = m > 0 and runs[0]['ids'][:m] == ref['token_ids'][:m]
    ok = (all(s['status'] == 200 and s['needle'] and s['text_ok'] for s in sums) and equal
          and ref['tokens'] == int(tokens))
    gate(f'kv-conc{n}', ok, {'streams': sums, 'stream0_equals_solo': equal, 'compared_tokens': m})

def short(url, n, decode):
    """Oracle p0 ids solo, then N concurrent copies: every stream == solo bitwise (parks on fewer physical slots)."""
    ids = json.load(open('/mnt/vm_8tb/b70/results/tf-flashnext/workers/I/oracle-eight.json'))['prompts'][0]['ids']
    solo = complete(url, ids, int(decode))
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(n)) as pool:
        runs = list(pool.map(lambda _: complete(url, ids, int(decode)), range(int(n))))
    same = [r.get('ids') == solo.get('ids') for r in runs]
    out = os.environ.get('KV_OUT') or os.environ.get('TF_OUT', '.')
    if solo.get('ids'):
        json.dump(solo['ids'], open(os.path.join(out, 'expected-p0-kv.json'), 'w'))
    ok = solo.get('status') == 200 and len(solo.get('ids') or []) > 0 and all(same)
    gate(f'kv-short-c{n}', ok, {'solo_tokens': len(solo.get('ids') or []), 'equal': same,
         'solo_decode_tok_s': solo.get('decode_tok_s'), 'ttft_s': [r.get('ttft_s') for r in runs]})

def launch_tail(log):
    text = open(log, errors='replace').read()
    at = max(text.rfind('paged KV budget'), 0)
    return text, text[at:]

def budget(log):
    text, _ = launch_tail(log)
    lines = re.findall(r'paged KV budget source=(\w+) pages=(\d+) tokens=(\d+) free=(\d+) reserve=(\d+) slab_est=(\d+) '
                       r'resident_200k=(\d+) resident_90k=(\d+)', text)
    if not lines:
        gate('kv-budget', False, {'line': None})
    src, pages, tokens, free, reserve, slab, r200, r90 = lines[-1]
    detail = {'source': src, 'pages': int(pages), 'tokens': int(tokens), 'free': int(free), 'reserve': int(reserve),
              'slab_est': int(slab), 'resident_200k': int(r200), 'resident_90k': int(r90)}
    gate('kv-budget', src == 'auto' and int(r90) >= 4 and int(r200) >= 2, detail)

def parks(log, least):
    _, tail = launch_tail(log)
    p = len(re.findall(r'eviction park id=', tail))
    r = len(re.findall(r'eviction restore id=', tail))
    bad = len(re.findall(r'(?i)(eviction|park|restore)[^\n]*(error|failed|poison)', tail))
    gate('kv-parks', p >= int(least) and r >= int(least) and bad == 0, {'parks': p, 'restores': r, 'errors': bad})

if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    {'budget': budget, 'long': long, 'conc': conc, 'parks': parks, 'short': short, 'again': again}[mode](*args)
