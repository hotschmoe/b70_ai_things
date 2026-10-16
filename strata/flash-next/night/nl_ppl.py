#!/usr/bin/env python3
"""Night-lab quality gate: teacher-forced NLL of fixed texts through the decode-window (verify) path.

The server config must add `--short-read 8192` and env STRATA_LOGPOS=<tsv> (see make_ppl_config).
  nl_ppl.py config <base.json> <out.json> <tsv>    write a ppl variant of a config
  nl_ppl.py run <tsv> [--n 3]                      send the fixed texts, then summarize the TSV
  nl_ppl.py compare <a.tsv> <b.tsv>                per-token NLL difference (same texts, same tokens)
"""
import json, math, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TEXTS = [REPO / 'docs/P2P_GPU.md', HERE / 'nl.py', REPO / 'docs/20261013_flashnext_iq3_vs_27b_ab_brief.md']


def texts(n, chars=5200):
    out = []
    for p in TEXTS[:n]:
        t = p.read_text(errors='replace')[:chars]
        out.append(''.join(ch if ord(ch) < 128 else ' ' for ch in t))
    return out


def make_config(base, out, tsv):
    c = json.loads(Path(base).read_text())
    c['args'] += ['--short-read', '8192']
    c['env']['STRATA_LOGPOS'] = tsv
    c['log'] = str(Path(tsv).with_suffix('.engine.log'))
    Path(out).write_text(json.dumps(c, indent=1))


def summarize(tsv):
    rows = [l.split('\t') for l in Path(tsv).read_text().splitlines() if l.strip()]
    lp = [float(r[2]) for r in rows]
    return dict(tokens=len(lp), mean_nll=-sum(lp) / max(1, len(lp)), ppl=math.exp(-sum(lp) / max(1, len(lp))))


def main():
    cmd = sys.argv[1]
    if cmd == 'config':
        make_config(*sys.argv[2:5]); return
    if cmd == 'run':
        sys.path.insert(0, str(HERE)); import nl
        tsv = sys.argv[2]; n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 3
        for i, t in enumerate(texts(n)):
            r = nl.send([{"role": "user", "content": t}], 1)
            print('text', i, 'prompt_n', (r['timings'] or {}).get('prompt_n'), flush=True)
        time.sleep(2)
        print(json.dumps(summarize(tsv)))
        return
    if cmd == 'compare':
        a = [l.split('\t') for l in Path(sys.argv[2]).read_text().splitlines() if l.strip()]
        b = [l.split('\t') for l in Path(sys.argv[3]).read_text().splitlines() if l.strip()]
        n = min(len(a), len(b))
        same = sum(1 for x, y in zip(a, b) if x[1] == y[1])
        d = [float(y[2]) - float(x[2]) for x, y in zip(a[:n], b[:n])]
        print(json.dumps(dict(n=n, same_targets=same, a=summarize(sys.argv[2]), b=summarize(sys.argv[3]),
                              mean_dlogp=sum(d) / n, max_abs_dlogp=max(abs(x) for x in d))))


if __name__ == '__main__':
    main()
