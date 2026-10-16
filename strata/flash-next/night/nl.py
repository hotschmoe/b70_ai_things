#!/usr/bin/env python3
"""Night-lab: lean start / bench / stop loop for Strata Flash-Next experiments.

  nl.py start <name> <config.json>     detached systemd unit + pair lease, waits for /health loaded
  nl.py bench <name> --out DIR [--c 1,2,4] [--tokens 512] [--prefill 4096,32768] [--reps 2]
  nl.py stop <name> [--no-health]      SIGTERM (engine QUIT), wait for container exit, per-card health
Decode numbers: engine timings (predicted_per_second) per request, client per-stream decode rate
(tokens after the first content event / time after it) and aggregate (all completion tokens over the
window from the first request start to the last end, and over first-content..last-end).
GPU busy: samples DRM fdinfo engine cycles of the native engine inside the container (no tracing).
"""
import argparse, hashlib, json, os, statistics, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path('/mnt/vm_8tb/github/b70_ai_things')
HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get('NL_PORT', '18085'))
BASE = f'http://127.0.0.1:{PORT}'
MODEL = 'hotschmoe-dd'

CODING = ("Implement a Python asyncio bounded worker pool for processing file paths. "
          "Requirements: fixed worker count, bounded queue for backpressure, preserve input order "
          "in results, propagate the first worker exception, cancel and await all outstanding tasks "
          "on failure, and handle caller cancellation without leaked tasks. Explain the shutdown "
          "protocol, provide complete typed code using only the standard library, and include "
          "three meaningful tests for ordering, worker failure, and cancellation.")
CODING2 = ("Write a Rust command line tool that reads a large CSV file of (timestamp, sensor_id, value) rows, "
           "computes per-sensor rolling 5-minute averages, min, max and p95 using a streaming algorithm, and "
           "writes the results as JSON lines. Use only the standard library plus serde and csv crates. Give the "
           "complete Cargo.toml and main.rs, explain the memory bounds, and add unit tests for the window logic.")
PROSE = ("Write a detailed explanation of how GPU memory hierarchies work: registers, shared local memory, "
         "L1 and L2 caches, HBM or GDDR, and PCIe host memory. Cover latency, bandwidth and how kernels should "
         "be structured to use each level well. Use concrete numbers for a modern discrete GPU.")
FILLER = "The history of computing is long and full of surprising turns. "


def get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


def send(messages, cap, extra=None):
    body = dict(model=MODEL, messages=messages, max_tokens=cap, temperature=0, stream=True,
                stream_options={"include_usage": True}, chat_template_kwargs={"enable_thinking": False})
    if extra:
        body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start = time.perf_counter(); first = None; events = []; usage = timings = None; text = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            t = time.perf_counter(); ch = json.loads(line[6:])
            if ch.get("error"):
                raise RuntimeError(ch["error"])
            usage = ch.get("usage") or usage; timings = ch.get("timings") or timings
            for c in ch.get("choices", []):
                d = c.get("delta", {}); s = d.get("content") or d.get("reasoning_content")
                if s:
                    text.append(d.get("content") or "")
                    events.append(t)
                    if first is None:
                        first = t
    end = time.perf_counter()
    txt = "".join(text)
    n = (usage or {}).get("completion_tokens", 0)
    dec = (n - 1) / (end - first) if first and end > first and n > 1 else None
    return dict(start=start, first=first, end=end, ttft=(first - start) if first else None, usage=usage,
                timings=timings, text=txt, sha=hashlib.sha256(txt.encode()).hexdigest()[:16],
                client_decode_tps=dec, n_events=len(events))


# ---------------- GPU busy via DRM fdinfo (inside the container's PID namespace) ----------------
SAMPLER = r"""
import os,sys,time,json
exe=sys.argv[1]; dt=float(sys.argv[2]); flag=sys.argv[3]
pid=None
for p in os.listdir('/proc'):
    if p.isdigit():
        try:
            if os.path.basename(os.readlink('/proc/%s/exe'%p)).startswith(exe): pid=p; break
        except Exception: pass
if pid is None: print(json.dumps({'error':'no pid'}),flush=True); sys.exit()
def snap():
    out={}
    for fd in os.listdir('/proc/%s/fdinfo'%pid):
        try: lines=open('/proc/%s/fdinfo/%s'%(pid,fd)).read().splitlines()
        except Exception: continue
        d={}
        for l in lines:
            if l.startswith('drm-') and ':' in l:
                k,v=l.split(':',1); d[k.strip()]=v.strip()
        if 'drm-client-id' not in d: continue
        key=d.get('drm-pdev','?')+'#'+d['drm-client-id']
        out[key]={k:int(v.split()[0]) for k,v in d.items() if k.startswith('drm-cycles-') or k.startswith('drm-total-cycles-')}
    return out
while os.path.exists(flag):
    print(json.dumps([time.time(),snap()]),flush=True); time.sleep(dt)
"""
VENV_PY = '/mnt/vm_8tb/b70/build/strata-ab/venv/bin/python'
FLAGDIR = Path('/mnt/vm_8tb/b70/results/night-20261016/runs')


class Busy:
    """Sample per-client DRM engine cycles of the native engine until stop()."""
    def __init__(self, name, dt=0.05):
        self.flag = FLAGDIR / f'.busy-{name}-{time.time_ns()}'; self.flag.write_text('1')
        self.p = subprocess.Popen(['docker', 'exec', name, VENV_PY, '-c', SAMPLER, 'strata', str(dt), str(self.flag)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def stop(self):
        try: self.flag.unlink()
        except FileNotFoundError: pass
        try: out, _ = self.p.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            self.p.kill(); out, _ = self.p.communicate()
        self.rows = []
        for line in out.splitlines():
            try: self.rows.append(json.loads(line))
            except Exception: pass
        return self

    def summarize(self, t0, t1):
        rows = [r for r in self.rows if isinstance(r, list)]
        acc = {}
        for (ta, a), (tb, b) in zip(rows, rows[1:]):
            if not (t0 <= ta and tb <= t1):
                continue
            per = {}
            for key, vals in b.items():
                if key not in a: continue
                pdev = key.split('#')[0]
                for k, v in vals.items():
                    if not k.startswith('drm-cycles-'): continue
                    cls = k[len('drm-cycles-'):]; tk = 'drm-total-cycles-' + cls
                    if tk not in vals or tk not in a[key]: continue
                    e = per.setdefault(pdev + '/' + cls, [0, 0])
                    e[0] += v - a[key].get(k, 0); e[1] = max(e[1], vals[tk] - a[key][tk])
            for k, (dc, dt) in per.items():
                if dt: acc.setdefault(k, []).append(dc / dt)
        return {k: round(statistics.mean(v), 4) for k, v in acc.items() if statistics.mean(v) > 0.002}


# ------------------------------------------------------------------------------------------------
def health_loaded():
    try:
        h = get('/health', timeout=3)
        return h.get('loaded') is True, h
    except Exception as e:
        return False, repr(e)


def cmd_start(a):
    cfg = Path(a.config).resolve()
    ok, h = health_loaded()
    if ok or 'Connection refused' not in str(h):
        sys.exit(f'port {PORT} busy: {h}')
    log = Path(a.log or (cfg.parent / f'{a.name}.server.log'))
    unit = f'nl-{a.name}'
    cmd = ['systemd-run', '--user', f'--unit={unit}', '--collect', f'--property=WorkingDirectory={REPO}',
           f'--property=StandardOutput=append:{log}', f'--property=StandardError=append:{log}',
           f'--setenv=NL_ENV={os.environ.get("NL_ENV", "")}',
           '/bin/bash', str(HERE / 'nl_launch.sh'), a.name, str(cfg), str(PORT)]
    t = time.time(); subprocess.run(cmd, check=True)
    last = None
    while time.time() - t < 1500:
        ok, h = health_loaded()
        if ok:
            print(json.dumps({'loaded': True, 'elapsed_s': round(time.time() - t, 1), 'health': h})); return
        s = str(h)[:120]
        if s != last:
            print(s, flush=True); last = s
        r = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True)
        if r.stdout.strip() not in ('active', 'activating'):
            sys.exit(f'unit {unit} ended before load; see {log}')
        time.sleep(3)
    sys.exit('timeout waiting for load')


def run_concurrent(prompts, cap, name, busy=False):
    busy = Busy(name) if busy else None
    barrier = threading.Barrier(len(prompts))

    def one(p):
        barrier.wait(timeout=30)
        return send([{"role": "user", "content": p}], cap)
    t0 = time.time(); w0 = time.perf_counter()
    with ThreadPoolExecutor(len(prompts)) as ex:
        rows = list(ex.map(one, prompts))
    w1 = time.perf_counter(); t1 = time.time()
    first_all = max(r['first'] for r in rows); end_min = min(r['end'] for r in rows)
    tot = sum(r['usage']['completion_tokens'] for r in rows)
    out = dict(
        c=len(prompts), cap=cap, wall_s=round(w1 - w0, 3), total_tokens=tot,
        agg_tps_wall=round(tot / (w1 - w0), 2),
        engine_tps=[(r['timings'] or {}).get('predicted_per_second') for r in rows],
        client_decode_tps=[round(r['client_decode_tps'] or 0, 2) for r in rows],
        ttft_s=[round(r['ttft'], 3) for r in rows],
        tokens=[r['usage']['completion_tokens'] for r in rows],
        drafts=[((r['timings'] or {}).get('draft_n'), (r['timings'] or {}).get('draft_n_accepted')) for r in rows],
        sha=[r['sha'] for r in rows])
    # steady overlap window: all streams decoding (after the last first-token, before the first finish)
    if end_min > first_all:
        out['overlap_s'] = round(end_min - first_all, 3)
    if busy:
        busy.stop(); off = t0 - w0  # perf_counter -> time.time()
        out['gpu_busy_overlap'] = busy.summarize(first_all + off, end_min + off)
        out['gpu_busy_prefill'] = busy.summarize(min(r['start'] for r in rows) + off, min(r['first'] for r in rows) + off)
    return out, rows


def cmd_bench(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ids = [m['id'] for m in get('/v1/models')['data']]
    assert ids[0] == MODEL, ids
    tag = a.tag or str(int(time.time()))
    res = {'models': ids, 'tag': tag, 'started': time.strftime('%F %T')}
    # coherence: concurrent deterministic arithmetic
    rows = run_concurrent([f"[{tag} coh {i}] Return only the integer: {x} * {y}." for i, (x, y) in
                           enumerate([(17, 19), (23, 29)])], 32, a.name)[1]
    res['coherence'] = [r['text'].strip() for r in rows]
    res['coherence_ok'] = res['coherence'] == ['323', '667']
    print('coherence', res['coherence'], flush=True)
    if not res['coherence_ok'] and not a.force:
        (out / 'bench.json').write_text(json.dumps(res, indent=2)); sys.exit('COHERENCE FAIL')
    if a.warm:
        run_concurrent([f"[{tag} warm] " + CODING], 64, a.name)
    for c in [int(x) for x in a.c.split(',') if x]:
        for rep in range(a.reps):
            bases = [CODING, CODING2, PROSE, CODING][:c] if a.mix else [CODING] * c
            prompts = [f"[{tag} c{c} r{rep} s{i}] " + p for i, p in enumerate(bases)]
            summ, rows = run_concurrent(prompts, a.tokens, a.name, busy=not a.no_busy)
            summ['rep'] = rep
            res.setdefault('decode', []).append(summ)
            print(json.dumps({k: summ[k] for k in summ if k not in ('sha',)}), flush=True)
            (out / f'decode-c{c}-r{rep}.json').write_text(json.dumps(
                [{k: v for k, v in r.items() if k not in ('start', 'first', 'end')} for r in rows], indent=1))
    if a.long:
        docs = sorted((REPO / 'docs').glob('2026*.md'))
        corpus = ''.join(ch if ord(ch) < 128 else ' ' for p in docs for ch in p.read_text(errors='replace'))
        for c in [int(x) for x in a.long_c.split(',') if x]:
            prompts = []
            for i in range(c):
                off = ((i + 3 * c) * 7919 * 13) % max(1, len(corpus) - a.long * 4)
                ctx = corpus[off:off + a.long * 4]
                prompts.append(f"[{tag} long{a.long} c{c} s{i}] " + ctx + "\n\nUsing the notes above as background, " + CODING)
            if a.long_warm and c > 1:   # prefill each context first so the timed streams decode together from cache
                for p in prompts:
                    send([{"role": "user", "content": p}], 1)
            summ, rows = run_concurrent(prompts, a.tokens, a.name, busy=False)
            summ['long_ctx_tokens'] = [ (r['usage'] or {}).get('prompt_tokens') for r in rows]
            res.setdefault('long', []).append(summ)
            print(json.dumps({k: summ[k] for k in ('c', 'engine_tps', 'client_decode_tps', 'ttft_s', 'long_ctx_tokens', 'drafts')}), flush=True)
    for n in [int(x) for x in a.prefill.split(',') if x]:
        for rep in range(a.prefill_reps):
            p = f"[{tag} prefill {n} r{rep} {time.time_ns()}] " + FILLER * max(1, (n - 60) // 13) + \
                "\nSummarize the above in one sentence."
            summ, rows = run_concurrent([p], 8, a.name, busy=True)
            r = rows[0]; tm = r['timings'] or {}
            row = dict(target=n, prompt_n=tm.get('prompt_n'), cache_n=tm.get('cache_n'),
                       gpu_busy=summ.get('gpu_busy_prefill'),
                       prompt_tps=tm.get('prompt_per_second'), prompt_ms=tm.get('prompt_ms'),
                       client_ttft=round(r['ttft'], 3))
            res.setdefault('prefill', []).append(row); print(json.dumps(row), flush=True)
    res['finished'] = time.strftime('%F %T')
    (out / 'bench.json').write_text(json.dumps(res, indent=2))
    # compact summary
    summ = {}
    for d in res.get('decode', []):
        s = summ.setdefault(f"c{d['c']}", {'agg': [], 'stream': [], 'engine': []})
        s['agg'].append(d['agg_tps_wall']); s['stream'] += d['client_decode_tps']
        s['engine'] += [x for x in d['engine_tps'] if x]
    for k, s in summ.items():
        print(k, 'agg_wall med', statistics.median(s['agg']), 'stream med', statistics.median(s['stream']),
              'engine med', statistics.median(s['engine']) if s['engine'] else None)
    for p in res.get('prefill', []):
        print('prefill', p['target'], p['prompt_tps'])


def cmd_stop(a):
    unit = f'nl-{a.name}'
    r = subprocess.run(['docker', 'kill', '--signal=TERM', a.name], capture_output=True, text=True)
    print('TERM', r.returncode, r.stderr.strip())
    t = time.time()
    while time.time() - t < 120:
        if not subprocess.run(['docker', 'ps', '-q', '--filter', f'name=^{a.name}$'], capture_output=True,
                              text=True).stdout.strip():
            break
        time.sleep(1)
    else:
        print('container still running after 120s; docker stop'); subprocess.run(['docker', 'stop', '-t', '30', a.name])
    for _ in range(30):
        st = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True).stdout.strip()
        if st not in ('active', 'activating', 'deactivating'):
            break
        time.sleep(1)
    print('stopped in', round(time.time() - t, 1), 's; unit', st)
    if not a.no_health:
        h = subprocess.run([str(REPO / 'bin/gpu-run'), str(REPO / 'bin/xpu-health')], capture_output=True, text=True)
        print('xpu-health exit', h.returncode, h.stdout.strip().splitlines()[-1:] if h.stdout else '')
        if h.returncode != 0:
            sys.exit('HEALTH FAIL')


def main():
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest='cmd', required=True)
    x = s.add_parser('start'); x.add_argument('name'); x.add_argument('config'); x.add_argument('--log')
    x = s.add_parser('bench'); x.add_argument('name'); x.add_argument('--out', required=True)
    x.add_argument('--c', default='1,2'); x.add_argument('--tokens', type=int, default=512)
    x.add_argument('--reps', type=int, default=2); x.add_argument('--prefill', default='4096,32768')
    x.add_argument('--prefill-reps', type=int, default=1); x.add_argument('--tag')
    x.add_argument('--mix', action='store_true'); x.add_argument('--warm', action='store_true')
    x.add_argument('--no-busy', action='store_true'); x.add_argument('--force', action='store_true')
    x.add_argument('--long', type=int, default=0, help='approx context tokens for long-context decode')
    x.add_argument('--long-c', default='1,2'); x.add_argument('--long-warm', action='store_true')
    x = s.add_parser('stop'); x.add_argument('name'); x.add_argument('--no-health', action='store_true')
    a = p.parse_args()
    {'start': cmd_start, 'bench': cmd_bench, 'stop': cmd_stop}[a.cmd](a)


if __name__ == '__main__':
    main()
