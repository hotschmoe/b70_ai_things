#!/usr/bin/env python3
"""E2b: llama-server SYCL C1 decode on Strata's coding prompt, then a C2 cross-slot check.

C1: coding prompt (Strata layer-mtp4 template, 1500 max tokens) and a distinct sysadmin
prompt, each alone, greedy, streamed (TTFT + client decode tok/s, Strata's metric).
C2: both prompts at once on two slots; each output is compared with its own C1 output
(exact match, common-prefix length, and a topic-leak check).
"""
import json, sys, threading, time, urllib.request

URL = sys.argv[1]; OUT = sys.argv[2]
TEMPLATE = sys.argv[3]
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
coding = json.load(open(TEMPLATE))["request"]["messages"][0]["content"].split("] ", 1)[1]
sysadmin = ("A Debian 12 server intermittently drops its WireGuard tunnel after DHCP renewals on the "
            "uplink. Write a step-by-step diagnosis plan with exact commands (ip, wg, journalctl, "
            "networkctl, tcpdump), likely root causes, and a robust systemd-networkd fix. Include "
            "a verification checklist.")
PROMPTS = {"coding": coding, "sysadmin": sysadmin}


def stream(prompt):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK,
            "temperature": 0, "top_k": 1, "stream": True, "timings_per_token": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = []; n = 0; timings = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("timings"):
                timings = d["timings"]
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.time()
                    text.append(piece); n += 1
    end = time.time()
    t = "".join(text)
    pred = (timings or {}).get("predicted_n") or n
    return {"ttft_s": round(first - t0, 3) if first else None, "wall_s": round(end - t0, 2),
            "client_decode_tps": round((pred - 1) / (end - first), 2) if first and end > first else None,
            "server_timings": timings, "chars": len(t), "text": t}


def main():
    res = {"url": URL, "max_tokens": MAXTOK, "c1": {}, "c2": {}}
    for k, p in PROMPTS.items():
        res["c1"][k] = stream(p)
        print("C1", k, {x: res["c1"][k][x] for x in ("ttft_s", "client_decode_tps", "wall_s")},
              "draft", (res["c1"][k]["server_timings"] or {}).get("draft_n_accepted"), "/",
              (res["c1"][k]["server_timings"] or {}).get("draft_n"), flush=True)
    th = {k: threading.Thread(target=lambda k=k: res["c2"].__setitem__(k, stream(PROMPTS[k]))) for k in PROMPTS}
    [t.start() for t in th.values()]; [t.join() for t in th.values()]
    for k in PROMPTS:
        a, b = res["c1"][k]["text"], res["c2"][k]["text"]
        cp = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        other = "wireguard" if k == "coding" else "asyncio"
        res["c2"][k].update({"equal_c1": a == b, "common_prefix_chars": cp, "c1_chars": len(a),
                             "leak_" + other: other in b.lower() and other not in a.lower()})
        print("C2", k, {x: res["c2"][k][x] for x in ("client_decode_tps", "equal_c1", "common_prefix_chars",
                                                     "c1_chars", "leak_" + other)}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
