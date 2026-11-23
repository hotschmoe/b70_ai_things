#!/usr/bin/env python3
"""Idle-gap gate (2026-11-23 mirror fix): one request with a fresh prompt of about N tokens (leaves closed KV pages
for the write-behind mirror), then IDLE_S seconds with no traffic, then a second request. Passes when the server
answers both and is still healthy. Before a4b1394c an idle gap over 60 s with a mirror copy pending exited the server
on the next request (FN park FATAL: mirror copy did not drain). Prints only sizes, timings and finish reasons.
Usage: idle_gap_gate.py URL PROMPT_TOKENS IDLE_S
"""
import json, random, sys, time, urllib.request

url, n_tok, idle = sys.argv[1].rstrip('/'), int(sys.argv[2]), float(sys.argv[3])
WORDS = "alpha bravo cedar delta ember fjord grove harbor island jasper kettle lumen meadow nectar orbit prism quartz ridge summit tundra umber valley willow xenon yarrow zephyr".split()


def post(body, timeout=170):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def health():
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            return r.status
    except Exception as e:  # noqa: BLE001
        return repr(e)[:80]


def gate(ok, detail):
    print("GATE idle_gap", "PASS" if ok else "FAIL", json.dumps(detail))
    sys.exit(0 if ok else 1)


rng = random.Random(time.time_ns())
text = " ".join(rng.choice(WORDS) for _ in range(n_tok))  # fresh each run, so no prompt-cache hit
detail = {"prompt_words": n_tok, "idle_s": idle}
try:
    d1, s1 = post({"model": "hotschmoe-dd", "messages": [{"role": "user", "content": text + "\n\nHow many distinct words above? One number."}], "max_tokens": 48, "temperature": 0})
    detail["first"] = {"finish": d1["choices"][0]["finish_reason"], "prompt_tokens": d1.get("usage", {}).get("prompt_tokens"), "s": round(s1, 2)}
except Exception as e:  # noqa: BLE001
    gate(False, {**detail, "first_error": repr(e)[:200]})
time.sleep(idle)
detail["health_after_idle"] = health()
try:
    d2, s2 = post({"model": "hotschmoe-dd", "messages": [{"role": "user", "content": "Reply with the word ready."}], "max_tokens": 48, "temperature": 0})
    detail["second"] = {"finish": d2["choices"][0]["finish_reason"], "s": round(s2, 2)}
except Exception as e:  # noqa: BLE001
    gate(False, {**detail, "second_error": repr(e)[:200]})
detail["health_end"] = health()
gate(detail["health_end"] == 200 and detail["second"]["finish"] in ("stop", "length"), detail)
