#!/usr/bin/env python3
# K10: TTFT / prefill tok/s against a live OpenAI-compatible server.
# Usage: bench_prefill_ttft.py <base_url> <model> [lens=2048,8000] [reps=3]
# max_tokens=1 so wall is prefill + 1. ASCII only.
import json, sys, time, urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
LENS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "2048,8000").split(",") if x.strip()]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
URL = BASE + "/completions"
UNIT = "The capital of France is Paris. "


def one(n_units):
    prompt = UNIT * n_units
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": 1,
        "temperature": 0.0, "ignore_eos": True,
    }).encode()
    t0 = time.time()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        obj = json.load(r)
    wall = time.time() - t0
    u = obj.get("usage") or {}
    pt = u.get("prompt_tokens") or 0
    ct = u.get("completion_tokens") or 0
    txt = (obj.get("choices") or [{}])[0].get("text") or ""
    return dict(wall=wall, pt=pt, ct=ct, tps=(pt / wall if wall else float("nan")),
                text=txt[:40])


def units_for(target):
    # UNIT is ~7-8 tok. Aim high then report actual pt.
    return max(1, target // 7)


if __name__ == "__main__":
    print(f"model={MODEL} lens={LENS} reps={REPS} max_tokens=1 (TTFT~prefill)", flush=True)
    for tgt in LENS:
        n = units_for(tgt)
        rows = []
        for i in range(REPS):
            try:
                r = one(n)
            except Exception as e:
                print(f"  tgt={tgt} FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
                rows = []
                break
            rows.append(r)
            print(f"  tgt~{tgt} rep={i} pt={r['pt']} ct={r['ct']} "
                  f"wall={r['wall']:.3f}s prefill={r['tps']:.1f} tok/s text={r['text']!r}",
                  flush=True)
        if not rows:
            continue
        avg_w = sum(x["wall"] for x in rows) / len(rows)
        avg_t = sum(x["tps"] for x in rows) / len(rows)
        pt = rows[0]["pt"]
        print(f"LEN~{tgt} actual_pt={pt} TTFT_avg={avg_w:.3f}s "
              f"prefill_avg={avg_t:.1f} tok/s n={len(rows)}", flush=True)
