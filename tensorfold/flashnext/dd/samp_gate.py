#!/usr/bin/env python3
"""SAMP gate (worker SAMP): real sampling on the TensorFold Flash-Next serve, short and stdlib-only.

    samp_gate.py BASE_URL [MODEL]        e.g. samp_gate.py http://127.0.0.1:18152 hotschmoe-dd

Prints one "GATE <name> PASS|FAIL <json>" line per check (no prompt or response text beyond lengths/hashes) and exits
non-zero when a hard check fails. Hard: caps, s1 greedy repeat, s2 top_k=1 == greedy, s3 seed repeat, s4 coherent at
T 0.6 with MTP drafting, s5 penalties served. Soft: s6 seeds differ, s7 drafts on == drafts off (lossless keyed draws).
SAMP_P0_EXPECTED=<token json> (e.g. workers/S/expected-current.json) adds s0: the bench's p0 completion, greedy, is
token-identical to that baseline, and so is the same prompt at temperature 0.6 with top_k 1 (shard merge on 128 rows).
"""
import os
import hashlib
import json
import re
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "hotschmoe-dd"
PROMPT = "Explain in one short paragraph why sea water is salty."
hard_fail = False


def post(path, body, timeout=120):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.read().decode()


def chat(max_tokens=128, **kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(kw)
    t = time.time()
    r = post("/v1/chat/completions", body)
    msg = r["choices"][0]["message"]
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    return text, r.get("usage", {}), time.time() - t


def counters():
    m = get("/metrics")
    def one(name):
        hit = re.search(r"^\S*" + name + r"(?:\{[^}]*\})?\s+([0-9.e+]+)", m, re.M)
        return float(hit.group(1)) if hit else 0.0
    return one("mtp_drafted_total"), one("mtp_accepted_total")


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def gate(name, ok, hard=True, **detail):
    global hard_fail
    print("GATE samp-%s %s %s" % (name, "PASS" if ok else "FAIL", json.dumps(dict(detail, hard=hard))), flush=True)
    if hard and not ok:
        hard_fail = True


def coherent(text):
    words = re.findall(r"[A-Za-z']+", text)
    ascii_ok = sum(c.isascii() for c in text) >= 0.95 * max(1, len(text))
    grams = [" ".join(words[i:i + 4]).lower() for i in range(max(0, len(words) - 3))]
    worst = max((grams.count(g) for g in set(grams)), default=0)
    topical = re.search(r"salt|sodium|mineral|river|rock", text, re.I) is not None
    return ascii_ok and len(words) >= 20 and worst <= 3 and topical, dict(words=len(words), worst_4gram=worst, topical=topical, ascii=ascii_ok)


def p0():
    want = json.load(open(os.environ["SAMP_P0_EXPECTED"]))
    ntok = int(os.environ.get("SAMP_P0_TOKENS", "128"))
    ids = json.load(open("/mnt/vm_8tb/b70/results/tf-flashnext/workers/I/oracle-eight.json"))["prompts"][0]["ids"]
    def run(**kw):
        body = {"model": MODEL, "prompt": ids, "max_tokens": ntok, "return_token_ids": True}
        body.update(kw)
        return post("/v1/completions", body)["tensorfold"]["token_ids"]
    g = run(temperature=0)
    gate("s0-greedy-p0-bitwise", g == want[:ntok], tokens=len(g), first_diff=next((i for i, (a, b) in enumerate(zip(g, want)) if a != b), None))
    k = run(temperature=0.6, top_p=0.95, top_k=1, seed=9, presence_penalty=0)
    gate("s0b-topk1-p0-equals-greedy", k == g, tokens=len(k), first_diff=next((i for i, (a, b) in enumerate(zip(k, g)) if a != b), None))


def main():
    if os.environ.get("SAMP_P0_EXPECTED"):
        p0()
    caps = json.loads(get("/v1/models"))["data"][0].get("capabilities", {})
    gate("caps", caps.get("greedy_only") is False and caps.get("penalties") is True and caps.get("repetition_penalty") is True,
         greedy_only=caps.get("greedy_only"), penalties=caps.get("penalties"), repetition_penalty=caps.get("repetition_penalty"))

    g1, u1, t1 = chat(temperature=0)
    g2, _, _ = chat(temperature=0)
    gate("s1-greedy-repeat", g1 == g2, sha=h(g1), tokens=u1.get("completion_tokens"), s=round(t1, 2))
    k1, _, _ = chat(temperature=0.6, top_p=0.95, top_k=1, seed=7, presence_penalty=0)
    gate("s2-topk1-equals-greedy", k1 == g1, greedy=h(g1), topk1=h(k1))

    d0, a0 = counters()
    s_a, us, ts = chat(temperature=0.6, top_p=0.95, top_k=20, seed=1234)
    d1, a1 = counters()
    s_b, _, _ = chat(temperature=0.6, top_p=0.95, top_k=20, seed=1234)
    gate("s3-seed-repeat", s_a == s_b, sha=h(s_a), again=h(s_b))
    ok, info = coherent(s_a)
    drafted, accepted = d1 - d0, a1 - a0
    gate("s4-t0.6-coherent-mtp", ok and drafted > 0, mtp_drafted=drafted, mtp_accepted=accepted,
         accept_rate=round(accepted / drafted, 3) if drafted else None, tokens=us.get("completion_tokens"),
         tok_s=round(us.get("completion_tokens", 0) / ts, 1) if ts else None, **info)
    if drafted:
        gate("s4b-mtp-accept-floor", accepted / drafted >= 0.25, hard=False, accept_rate=round(accepted / drafted, 3))

    p, _, _ = chat(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, seed=5)
    r, _, _ = chat(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1, frequency_penalty=0.2, seed=5)
    okp, ip = coherent(p)
    okr, ir = coherent(r)
    gate("s5-penalties-served", okp and okr, presence=ip, repetition=ir)

    s_c, _, _ = chat(temperature=0.6, top_p=0.95, top_k=20, seed=4321)
    gate("s6-seeds-differ", s_c != s_a, hard=False, a=h(s_a), b=h(s_c))
    nd, _, _ = chat(temperature=0.6, top_p=0.95, top_k=20, seed=1234, draft=False)
    gate("s7-drafts-lossless", nd == s_a, hard=False, drafts=h(s_a), no_drafts=h(nd))
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
