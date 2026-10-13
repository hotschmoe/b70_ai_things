#!/usr/bin/env python3
"""Hidden checker for re-proto: replays recorded multi-connection session scripts against the agent's kvd.py."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from mscore import Milestones
from kvsession import replay

SUB, HID = sys.argv[1], os.path.dirname(os.path.abspath(__file__))
CMD = ["python3", f"{SUB}/kvd.py"]
MS = [("m01", "greeting, nonce PRNG, ping echo"), ("m02", "frame errors: magic/version/oversize/partial delivery/pre-auth"),
      ("m03", "auth handshake: keyed hash, strikes, roles"), ("m04", "SET/GET int and string values"),
      ("m05", "lists, DEL, wrong types, argument validation"), ("m06", "INCR semantics and wraparound"),
      ("m07", "role permission matrix"), ("m08", "logical clock: TTL, TICK, EXPIRE"),
      ("m09", "KEYS, MGET, INFO"), ("m10", "capacity limits"), ("m11", "transactions: isolation, commit, rollback"),
      ("m12", "transaction conflicts and state errors"), ("m13", "WATCH notifications fan-out"),
      ("m14", "notifications from TICK expiry and transaction commit"), ("m15", "pipelining and split frames"),
      ("m16", "malformed input matrix")]
cases = json.load(open(f"{HID}/cases.json"))
jobs = list(enumerate(cases))


def one(j):
    i, c = j
    ok, why = replay(CMD, 20000 + i, c["script"], cwd=SUB)
    if not ok and os.environ.get("KV_DEBUG"):
        print("FAIL", c["id"], why, file=sys.stderr)
    return c["ms"], ok


res = {}
with ThreadPoolExecutor(6) as ex:
    for ms, ok in ex.map(one, jobs):
        res.setdefault(ms, []).append(ok)
M = Milestones()
for mid, name in MS:
    M.add(mid, name, res.get(mid, []))
M.emit()
