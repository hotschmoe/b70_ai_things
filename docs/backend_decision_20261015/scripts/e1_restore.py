#!/usr/bin/env python3
"""E1: llama-server in-RAM hybrid-state restore oracle (CPU, no GPU).

Two slots, four distinct chats. Chats A,B fill the slots; C,D force them out
(saved into the --cache-ram prompt cache). Returning to A and B with a short
follow-up must reuse the full held history (cache_n) and evaluate only the new
tokens (prompt_n), and its greedy continuation must equal the continuation of
an uninterrupted control (same follow-up sent immediately, no eviction).

Token arrays are used end to end so re-tokenization cannot shorten the match.
"""
import json, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18190"
OUT = sys.argv[2] if len(sys.argv) > 2 else "e1-result.json"
N_REPLY = int(sys.argv[3]) if len(sys.argv) > 3 else 32
N_CONT = int(sys.argv[4]) if len(sys.argv) > 4 else 24
PHASE = sys.argv[5] if len(sys.argv) > 5 else "evict"   # evict | control


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def tok(text):
    return post("/tokenize", {"content": text})["tokens"]


def complete(tokens, n, slot):
    t0 = time.time()
    r = post("/completion", {"prompt": tokens, "n_predict": n, "temperature": 0,
                             "top_k": 1, "cache_prompt": True, "id_slot": slot,
                             "return_tokens": True, "samplers": ["top_k"]})
    tm = r.get("timings", {})
    return {"tokens": r["tokens"], "text": r["content"], "wall_s": round(time.time() - t0, 2),
            "cache_n": tm.get("cache_n"), "prompt_n": tm.get("prompt_n"),
            "n_input": len(tokens), "slot": slot,
            "draft_n": tm.get("draft_n"), "draft_n_accepted": tm.get("draft_n_accepted"),
            "predicted_per_second": tm.get("predicted_per_second")}


TOPICS = {
    "A": "the history of the Rust borrow checker and why non-lexical lifetimes mattered",
    "B": "how a Linux page cache interacts with mmap, readahead and memory pressure",
    "C": "the design of the Go scheduler: Ms, Ps and Gs and work stealing",
    "D": "Dart isolates, event loops and how they differ from OS threads",
}
FILLER = ("Explain step by step, with concrete examples, edge cases and common "
          "misconceptions. Keep each paragraph focused. ")


def chat_prompt(key):
    body = f"Write a detailed technical essay about {TOPICS[key]}. " + FILLER * 40
    return tok(f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n")


FOLLOW = "<|im_end|>\n<|im_start|>user\nNow summarize that in one sentence.<|im_end|>\n<|im_start|>assistant\n"


def main():
    res = {"url": URL, "phase": PHASE, "n_reply": N_REPLY, "n_cont": N_CONT, "steps": []}
    follow = tok(FOLLOW)
    held = {}
    order = [("A", 0), ("B", 1)] if PHASE == "control" else [("A", 0), ("B", 1), ("C", 0), ("D", 1)]
    for key, slot in order:
        p = chat_prompt(key)
        r = complete(p, N_REPLY, slot)
        held[key] = p + r["tokens"]
        res["steps"].append({"step": f"first-{key}", **{k: v for k, v in r.items() if k != "tokens"}})
    for key, slot in [("A", 0), ("B", 1)]:
        ret = held[key] + follow
        r = complete(ret, N_CONT, slot)
        r["held_tokens"] = len(held[key])
        r["new_tokens"] = len(follow)
        r["replayed"] = (r["prompt_n"] or 0) - len(follow)
        res["steps"].append({"step": f"return-{key}", **r})
    json.dump(res, open(OUT, "w"), indent=1)
    for s in res["steps"]:
        print(s["step"], "in", s["n_input"], "cache_n", s["cache_n"], "prompt_n", s["prompt_n"],
              "replayed", s.get("replayed", "-"), "wall", s["wall_s"])


if __name__ == "__main__":
    main()
