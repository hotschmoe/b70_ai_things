#!/usr/bin/env python3
"""F1: HTTP probe for the sidecar prefill serve. No GPU lease (host HTTP only).

  identity -> greedy gates (17*19=323, 23*29=667) -> TTFT sweep (each length
  twice: first run carries JIT) -> long-prompt coherence (filler + math) ->
  C1/C2 decode via bench_latency.

Env: F1_BASE (http://127.0.0.1:18140), F1_OUT (json path), F1_LENGTHS
("1024,8192,32768,131072"), F1_COHERENCE_LEN (32768), F1_REPEAT (2),
F1_DECODE (1), F1_ALIAS.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_latency import (  # noqa: E402
    coherence,
    decode_c1,
    decode_c2,
    identity,
    make_prompt,
    post_json,
    stream_completion,
    summarize_itl,
)

BASE = os.environ.get("F1_BASE", "http://127.0.0.1:18140")
OUT = os.environ["F1_OUT"]
MODEL = os.environ.get("F1_MODEL", "hotschmoe-dev")
ALIAS = os.environ.get("F1_ALIAS", "qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-f1-prefill")
LENGTHS = [int(v) for v in os.environ.get("F1_LENGTHS", "1024,8192,32768,131072").split(",") if v.strip()]
COH_LEN = int(os.environ.get("F1_COHERENCE_LEN", "32768"))
REPEAT = int(os.environ.get("F1_REPEAT", "2"))
TIMEOUT = float(os.environ.get("F1_TIMEOUT", "1800"))
report: dict = {"base": BASE, "model": MODEL, "lengths": LENGTHS, "t0": time.strftime("%H:%M:%S")}


def save() -> None:
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")


def log(msg: str) -> None:
    print(f"F1 {time.strftime('%H:%M:%S')} {msg}", flush=True)


def math(a: int, b: int, context: str = "") -> dict:
    body = post_json(
        BASE + "/v1/chat/completions",
        dict(
            model=MODEL,
            messages=[{"role": "user", "content": context + f"\nReturn only the integer: {a} * {b}."}],
            temperature=0,
            max_tokens=16,
            chat_template_kwargs={"enable_thinking": False},
        ),
        TIMEOUT,
    )
    text = body["choices"][0]["message"]["content"].strip()
    return dict(a=a, b=b, text=text, match=text == str(a * b), usage=body.get("usage"))


report["identity"] = identity(BASE, MODEL, ALIAS)
log("identity " + json.dumps(report["identity"]))
report["gates"] = [math(17, 19), math(23, 29)]
log("gates " + json.dumps(report["gates"]))
save()

CTL = os.environ.get("F1_CTL", "")
ARMS = [a for a in os.environ.get("F1_ARMS", "").split(";") if a.strip()] or [""]


def set_arm(arm: str) -> None:
    if not CTL:
        return
    with open(CTL, "w", encoding="utf-8") as fh:
        fh.write(arm + "\n")
    os.utime(CTL, None)
    log(f"ctl -> {arm!r}")


report["prefill"] = []
for arm in ARMS:
  set_arm(arm)
  if arm and arm != ARMS[0]:
    report.setdefault("gates_by_arm", {})[arm] = [math(17, 19), math(23, 29)]
    log(f"gates arm={arm!r} " + json.dumps(report["gates_by_arm"][arm]))
  for n in LENGTHS:
    prompt, built = make_prompt(BASE, MODEL, n, f"f1-{n}-{time.time_ns()}")
    for r in range(REPEAT):
        if r > 0:
            # Fresh nonce: no prefix reuse between repeats (prefix cache is off anyway).
            prompt, built = make_prompt(BASE, MODEL, n, f"f1-{n}-{r}-{time.time_ns()}")
        row = stream_completion(BASE, MODEL, prompt, 1, TIMEOUT)
        s = summarize_itl(row)
        s.update(target=n, built=built, run=r, first_token=row.get("text"), arm=arm)
        report["prefill"].append(s)
        log(f"prefill arm={arm!r} n={n} run={r} prompt_tokens={s.get('prompt_tokens')} ttft_ms={s.get('ttft_ms')} tok_s={s.get('prefill_tok_s')}")
        save()

if COH_LEN > 0:
    prompt, built = make_prompt(BASE, MODEL, COH_LEN, f"f1-coh-{time.time_ns()}")
    report["long_coherence"] = [math(17, 19, prompt), math(23, 29, prompt)]
    report["long_coherence_built"] = built
    log("long_coherence " + json.dumps(report["long_coherence"]))
    save()

if os.environ.get("F1_DECODE", "1") == "1":
    report["coherence"] = coherence(BASE, MODEL, TIMEOUT)
    report["decode_c1"] = decode_c1(BASE, MODEL, 64, TIMEOUT)
    report["decode_c2"] = decode_c2(BASE, MODEL, 64, TIMEOUT)
    c1 = report["decode_c1"].get("itl_ms", {})
    log(f"decode c1 itl p50={c1.get('p50')} coherence_ok={report['coherence'].get('ok')}")
    save()

ok = all(g["match"] for g in report["gates"]) and all(g["match"] for g in report.get("long_coherence", []))
ok = ok and all(g["match"] for arm in report.get("gates_by_arm", {}).values() for g in arm)
report["ok"] = ok
save()
log(f"DONE ok={ok}")
sys.exit(0 if ok else 3)
