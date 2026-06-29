#!/usr/bin/env python3
# serve-soak.py -- sustained concurrent-load soak for an OpenAI-compatible serve, tuned to surface the
# XPU GDN/Mamba "!!!!" NaN-poison (SHORTCOMINGS.md): the failure mode that needs SUSTAINED concurrent
# prefill+decode co-batching, which a short fixed-length sweep can miss.
#
# It keeps C workers busy for DURATION seconds, each firing /v1/completions with a VARIED prompt length
# (short / medium / long) drawn round-robin so new prefills keep batching against in-flight decodes.
# Greedy (temperature 0) so any degeneracy is deterministic + obvious. Flags a reply as degenerate if:
#   - it is empty / errored, OR
#   - one non-space char is >55% of a >=12-char reply (the gen-probe gate), OR
#   - it contains a run of >=8 identical non-space chars (e.g. "!!!!!!!!").
#
# Usage:
#   python3 bin/serve-soak.py --base-url http://localhost:8000/v1 --model <served-id> \
#       --concurrency 4 --duration 900 [--max-tokens 128]
# Exit 0 = clean soak (0 degenerate). Exit 1 = degeneracy detected (records first-bad time).
import argparse, json, time, threading, urllib.request, urllib.error, collections, sys

PROMPTS = {
    "short":  ["The capital of France is",
               "List three primes.",
               "What color is the sky?",
               "2 + 2 ="],
    "medium": ["Explain in one paragraph why the sky appears blue during the day.",
               "Write a short function in Python that returns the nth Fibonacci number.",
               "Summarize the causes of the French Revolution in a few sentences.",
               "Describe how a hash map works and its average-case complexity."],
    # long prompt: a chunk of repeated context so the prefill is sizeable (forces big prefills to
    # co-batch with running decodes -- the exact GDN/Mamba co-batch the NaN poison needs).
    "long":   ["Here is a long document. " + ("The quick brown fox jumps over the lazy dog. " * 120)
               + "\nGiven the document above, write a detailed multi-paragraph analysis of its style."],
}
ORDER = ["short", "medium", "long", "medium"]  # round-robin bias toward mixed sizes

def is_degenerate(txt):
    if txt is None:
        return "EMPTY/ERROR"
    s = txt.replace(" ", "").replace("\n", "")
    n = len(s)
    if n < 12:
        return None  # too short to judge
    # run of >=8 identical chars
    run = 1
    for i in range(1, n):
        if s[i] == s[i-1]:
            run += 1
            if run >= 8:
                return "RUN(%s)" % s[i]
        else:
            run = 1
    # single-char dominance
    c = collections.Counter(s)
    ch, cnt = c.most_common(1)[0]
    if cnt / n > 0.55:
        return "DOMINANT(%s=%.0f%%)" % (ch, 100*cnt/n)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--duration", type=int, default=900)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/completions"
    stop = time.time() + args.duration
    lock = threading.Lock()
    stats = {"req": 0, "ok": 0, "degen": 0, "err": 0, "out_tok": 0, "first_bad": None,
             "first_bad_t": None, "samples": []}
    t0 = time.time()

    def worker(wid):
        i = wid
        while time.time() < stop:
            kind = ORDER[i % len(ORDER)]
            prompt = PROMPTS[kind][i % len(PROMPTS[kind])]
            i += 1
            body = json.dumps({"model": args.model, "prompt": prompt,
                               "max_tokens": args.max_tokens, "temperature": 0}).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            txt = None
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as r:
                    j = json.loads(r.read().decode())
                    txt = j["choices"][0].get("text", "")
                    ntok = j.get("usage", {}).get("completion_tokens", 0)
            except Exception as e:
                with lock:
                    stats["req"] += 1; stats["err"] += 1
                    if stats["first_bad"] is None:
                        stats["first_bad"] = "ERR:%s" % type(e).__name__
                        stats["first_bad_t"] = time.time() - t0
                continue
            d = is_degenerate(txt)
            with lock:
                stats["req"] += 1; stats["out_tok"] += ntok
                if d:
                    stats["degen"] += 1
                    if stats["first_bad"] is None:
                        stats["first_bad"] = d
                        stats["first_bad_t"] = time.time() - t0
                        stats["samples"].append((kind, d, repr(txt[:100])))
                else:
                    stats["ok"] += 1

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(args.concurrency)]
    for t in threads: t.start()

    # progress line every 30s
    while time.time() < stop:
        time.sleep(30)
        with lock:
            el = time.time() - t0
            tput = stats["out_tok"] / el if el > 0 else 0
            print("[soak %4ds] req=%d ok=%d degen=%d err=%d  agg_out=%.1f tok/s  first_bad=%s@%s"
                  % (el, stats["req"], stats["ok"], stats["degen"], stats["err"], tput,
                     stats["first_bad"], ("%.0fs" % stats["first_bad_t"]) if stats["first_bad_t"] else "-"),
                  flush=True)
    for t in threads: t.join(timeout=args.timeout + 5)

    el = time.time() - t0
    print("\n=== SOAK DONE: c=%d dur=%ds ===" % (args.concurrency, args.duration))
    print("requests=%d ok=%d degenerate=%d errors=%d" % (stats["req"], stats["ok"], stats["degen"], stats["err"]))
    print("aggregate decode throughput = %.1f tok/s" % (stats["out_tok"] / el if el > 0 else 0))
    if stats["samples"]:
        print("first degenerate samples:")
        for kind, d, s in stats["samples"][:5]:
            print("  [%s] %s -> %s" % (kind, d, s))
    bad = stats["degen"] + stats["err"]
    if bad == 0:
        print("VERDICT: CLEAN -- no degeneracy under sustained c%d for %ds." % (args.concurrency, args.duration))
        return 0
    print("VERDICT: FAILED -- %d bad (degen=%d err=%d); first at %s (%s)."
          % (bad, stats["degen"], stats["err"], stats["first_bad"],
             ("%.0fs" % stats["first_bad_t"]) if stats["first_bad_t"] else "?"))
    return 1

if __name__ == "__main__":
    sys.exit(main())
