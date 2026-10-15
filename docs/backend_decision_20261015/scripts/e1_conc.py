#!/usr/bin/env python3
"""E1b: two slots decoding concurrently with draft-mtp; outputs must equal the sequential MTP control."""
import json, os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "http://127.0.0.1:18190"]
import e1_restore as R
ctrl = {s["step"]: s for s in json.load(open("e1-control-mtp.json"))["steps"]}
res = {}
def run(name, toks, n, slot): res[name] = R.complete(toks, n, slot)
pA, pB = R.chat_prompt("A"), R.chat_prompt("B")
ts = [threading.Thread(target=run, args=("first-A", pA, 32, 0)), threading.Thread(target=run, args=("first-B", pB, 32, 1))]
[t.start() for t in ts]; [t.join() for t in ts]
f = R.tok(R.FOLLOW)
hA, hB = pA + res["first-A"]["tokens"], pB + res["first-B"]["tokens"]
ts = [threading.Thread(target=run, args=("return-A", hA + f, 24, 0)), threading.Thread(target=run, args=("return-B", hB + f, 24, 1))]
[t.start() for t in ts]; [t.join() for t in ts]
out = {}
for k, v in res.items():
    out[k] = {"equal_seq_mtp": v["text"] == ctrl[k]["text"], "cache_n": v["cache_n"], "prompt_n": v["prompt_n"],
              "draft_n": v["draft_n"], "acc": v["draft_n_accepted"], "tps": v["predicted_per_second"], "text": v["text"]}
    print(k, {x: out[k][x] for x in ("equal_seq_mtp", "cache_n", "prompt_n", "draft_n", "acc", "tps")})
json.dump(out, open("e1-conc-mtp.json", "w"), indent=1)
