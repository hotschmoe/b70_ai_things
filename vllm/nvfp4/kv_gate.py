#!/usr/bin/env python3
"""Coherence + long-context needle gate for the fp8-KV nvfp4 server.
Runs a few factual prompts and a needle-in-haystack recall at NEEDLE_DEPTH tokens.
PROBE_HOST default http://127.0.0.1:8079. Prints PASS/FAIL per check.
"""
import json, os, urllib.request

HOST = os.environ.get("PROBE_HOST", "http://127.0.0.1:8079")
NEEDLE_DEPTH = int(os.environ.get("NEEDLE_DEPTH", "0"))  # approx tokens of filler; 0 disables
KEY = os.environ.get("KEY", "")  # optional API key (Authorization: Bearer) for key-enforced serves


def _hdr():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h


def mid():
    r = urllib.request.Request(HOST + "/v1/models", headers=_hdr())
    return json.load(urllib.request.urlopen(r, timeout=15))["data"][0]["id"]


def gen(m, prompt, maxtok=64, temp=0.0):
    body = {"model": m, "prompt": prompt, "max_tokens": maxtok, "temperature": temp}
    r = urllib.request.Request(HOST + "/v1/completions", data=json.dumps(body).encode(),
                               headers=_hdr())
    with urllib.request.urlopen(r, timeout=600) as x:
        return json.load(x)["choices"][0]["text"]


def main():
    m = mid()
    print("served:", m)
    checks = []
    # 1. capital of France
    t = gen(m, "Question: What is the capital of France?\nAnswer:", 12)
    ok = "paris" in t.lower(); checks.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] capital-of-France -> {t.strip()[:60]!r}")
    # 2. arithmetic with reasoning
    t = gen(m, "What is 17+26? Think step by step, then give the final number.", 120)
    ok = "43" in t; checks.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 17+26=43 -> ...{t.strip()[-70:]!r}")
    # 3. short factual
    t = gen(m, "The chemical symbol for gold is", 6)
    ok = "au" in t.lower(); checks.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] gold=Au -> {t.strip()[:30]!r}")
    # 4. needle in haystack
    if NEEDLE_DEPTH > 0:
        secret = "The launch code for project Bluefinch is 7391-ZULU."
        filler_unit = ("In the archives of the northern library, scholars catalogued "
                       "manuscripts describing agricultural techniques of the river valleys. ")
        n_units = max(1, NEEDLE_DEPTH // 20)
        half = n_units // 2
        doc = filler_unit * half + secret + " " + filler_unit * (n_units - half)
        prompt = ("Read the following document carefully.\n\n" + doc +
                  "\n\nQuestion: What is the launch code for project Bluefinch? Answer with just the code.\nAnswer:")
        approx_tok = len(prompt) // 4
        t = gen(m, prompt, 24)
        ok = "7391" in t and "zulu" in t.lower(); checks.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] needle@~{approx_tok}tok -> {t.strip()[:50]!r}")
    print(f"GATE: {sum(checks)}/{len(checks)} PASS")


if __name__ == "__main__":
    main()
