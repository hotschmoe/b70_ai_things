#!/usr/bin/env python3
"""G1 coherence probe for Ornith / Qwen 35B NVFP4 serves.

Usage: g1_probe.py [base_url] [model_or_auto]
Prints one JSON object to stdout. Exit 0 = GO (Paris and not bangs).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://192.168.10.5:18080/v1").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "auto"


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def post(path, body, timeout=180):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def is_bangs(text: str) -> bool:
    if not text:
        return True
    if "!!!!" in text:
        return True
    compact = "".join(text.split())
    if len(compact) < 2:
        return True
    bangs = compact.count("!")
    return bangs / len(compact) > 0.4


def one_completion(prompt: str, n: int):
    t0 = time.time()
    obj = post(
        "/completions",
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": n,
            "temperature": 0.0,
        },
        timeout=180,
    )
    wall = time.time() - t0
    txt = (obj["choices"][0].get("text") or "")
    u = obj.get("usage") or {}
    ct = u.get("completion_tokens") or 0
    return {
        "text": txt,
        "wall_s": round(wall, 2),
        "ct": ct,
        "tps": round(ct / wall, 2) if wall and ct else None,
        "bangs": is_bangs(txt),
    }


def one_chat(prompt: str, n: int):
    t0 = time.time()
    obj = post(
        "/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    wall = time.time() - t0
    msg = obj["choices"][0].get("message") or {}
    txt = msg.get("content") or ""
    think = msg.get("reasoning_content") or ""
    u = obj.get("usage") or {}
    ct = u.get("completion_tokens") or 0
    return {
        "text": txt,
        "think": think[:120],
        "wall_s": round(wall, 2),
        "ct": ct,
        "tps": round(ct / wall, 2) if wall and ct else None,
        "bangs": is_bangs(txt or think),
    }


def main():
    global MODEL
    if MODEL in ("", "auto"):
        MODEL = get(BASE + "/models", timeout=10)["data"][0]["id"]
    out = {"model": MODEL, "base": BASE, "ok": False, "err": None, "probes": {}}
    try:
        paris = one_completion("The capital of France is", 16)
        mul = one_completion("17*23=", 8)
        try:
            chat = one_chat("The capital of France is", 16)
        except Exception as e:
            chat = {"text": "", "think": "", "err": f"{type(e).__name__}: {e}",
                    "bangs": False, "tps": None, "ct": 0, "wall_s": None}
        out["probes"] = {"paris": paris, "mul": mul, "chat": chat}
        ptxt = paris["text"]
        mtxt = mul["text"]
        ctxt = chat.get("text") or ""
        out["has_paris"] = "paris" in (ptxt + " " + ctxt).lower()
        out["has_391"] = "391" in mtxt
        out["any_bangs"] = bool(paris["bangs"] or mul["bangs"] or chat.get("bangs"))
        out["ok"] = bool(out["has_paris"] and not out["any_bangs"])
        out["verdict"] = "GO" if out["ok"] else "NO-GO"
    except Exception as e:
        out["err"] = f"{type(e).__name__}: {e}"
        out["verdict"] = "ERR"
        out["ok"] = False
    print(json.dumps(out, ensure_ascii=True))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
