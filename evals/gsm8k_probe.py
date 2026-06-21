#!/usr/bin/env python3
"""Compact GSM8K correctness probe -- runs on the dev box, hits a vLLM endpoint over the LAN.
Purpose: confirm a TP=2 serve produces CORRECT reasoning (tensor-parallel must not corrupt outputs),
not just fluent tokens. Greedy/deterministic. Stdlib only.
Usage: python3 gsm8k_probe.py <served_model_id> [endpoint] [n]
"""
import sys, json, re, urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen36-27b-int4-tp2"
ENDPOINT = sys.argv[2] if len(sys.argv) > 2 else "http://192.168.10.5:18080/v1"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12

# Fixed GSM8K-style items (question, integer answer). Hand-checked.
ITEMS = [
    ("Natalia sold clips to 48 friends in April, then sold half as many in May. How many did she sell altogether?", 72),
    ("Weng earns $12 an hour for babysitting. Yesterday she babysat for 50 minutes. How much did she earn?", 10),
    ("Betty has half the money she needs for a $100 wallet. Her parents give her $15 and her grandparents twice as much as her parents. How much more does she need?", 5),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", 624),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?", 3),
    ("Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many as Seattle. Seattle has 20. How many sheep do they have together?", 260),
    ("Mark has a garden with 10 yellow flowers, 80% more purple ones, and as many green as 25% of the yellow and purple combined. How many flowers total?", 35),
    ("Ken created a care package. He put a box on a scale, added jelly beans to 2 lbs, tripled the weight with brownies, added 2 lbs more jelly beans, then doubled it with gummy worms. Final weight in lbs?", 16),
    ("A deep-sea monster eats ships every 100 years. Over 300 years it ate 847 people. Each new ship had twice as many people as the last. How many people were on the first ship?", 121),
    ("Tobias buys $95 shoes. He saved 3 months allowance of $5, mowed 4 lawns at $15, shoveled driveways at $7 each, and has $15 change. How many driveways did he shovel?", 5),
    ("There are 15 trees. Workers plant more so there are 21. How many did they plant?", 6),
    ("If there are 3 cars and 2 more arrive, then each car has 4 wheels, how many wheels total?", 20),
]

def last_int(text):
    nums = re.findall(r"-?\d[\d,]*", text.replace(",", ""))
    return int(nums[-1]) if nums else None

def ask(q):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": q + " Give the final answer as a single number on the last line."}],
        "temperature": 0.0, "max_tokens": 1500,
    }).encode()
    req = urllib.request.Request(ENDPOINT.rstrip("/") + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]

ok = 0
items = ITEMS[:N]
for i, (q, ans) in enumerate(items, 1):
    try:
        out = ask(q)
        got = last_int(out)
        good = (got == ans)
        ok += good
        print(f"[{i:2d}] {'OK ' if good else 'XX '} got={got} want={ans}")
    except Exception as e:
        print(f"[{i:2d}] ERR {type(e).__name__}: {str(e)[:120]}")
print(f"\nGSM8K probe: {ok}/{len(items)} correct  ({MODEL} @ {ENDPOINT})")
