#!/usr/bin/env python3
"""Suffix-decoding effectiveness probe (vLLM v0.24.0 XPU, method="suffix").

Suffix decoding (arXiv:2411.04975) drafts from a per-request suffix tree built out of the prompt
+ the tokens generated so far -- a CPU-side, training-free, no-kernel drafter that REPLACES MTP.
It SHINES on repetitive/agentic generation (long verbatim runs that echo the prompt or earlier
output); it does ~nothing on fresh, non-repeating prose. This probe measures exactly that gap.

It runs two phases, N turns each, temperature 0 (greedy, so accept is deterministic):
  (A) REPEAT  -- prompt hands the model a code/text block and asks it to reproduce it verbatim
                 (+ a trivial edit). The completion is mostly a verbatim echo of the prompt ->
                 the suffix tree nails long drafts -> high accept length.
  (B) FRESH   -- a "write something novel" prompt with no echo -> low accept length (control).

Around each phase it scrapes the server /metrics spec-decode counters and computes the accept
length from the deltas, and it times streamed decode t/s per turn. Prints a comparison table.

Prometheus counters used (names verified in the image, v1/spec_decode/metrics.py):
  vllm:spec_decode_num_drafts_total          (# of draft steps)
  vllm:spec_decode_num_draft_tokens_total    (# tokens proposed)
  vllm:spec_decode_num_accepted_tokens_total (# tokens accepted)
Mean acceptance length = 1 + accepted/drafts  (the convention incl. the bonus token).

Dependency-free (urllib + threads). Usage:
  suffix_probe.py <base_url> <model> [turns=6] [max_tokens=400] [api_key]
  <base_url> = the OpenAI base, e.g. http://192.168.10.5:18080/v1  (/metrics is derived from it)
  api_key also read from $API_KEY.
"""
import sys, json, os, time, urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 400
API_KEY = sys.argv[5] if len(sys.argv) > 5 else os.environ.get("API_KEY", "")

# /metrics lives at the server root, NOT under /v1 -- strip a trailing /v1 (or /v1/...) off BASE.
_root = BASE
for suf in ("/v1", ):
    if _root.endswith(suf):
        _root = _root[: -len(suf)]
METRICS_URL = _root + "/metrics"
CHAT_URL = BASE + "/chat/completions"

SPEC_METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)

# A block the model is asked to reproduce verbatim -- long verbatim runs are what the suffix tree
# drafts. Deterministic so every REPEAT turn is comparable.
CODE_BLOCK = """
def kadane_max_subarray(nums):
    best = cur = nums[0]
    start = end = temp = 0
    for i in range(1, len(nums)):
        if cur < 0:
            cur = nums[i]
            temp = i
        else:
            cur = cur + nums[i]
        if cur > best:
            best = cur
            start = temp
            end = i
    return best, start, end
"""


def _hdr():
    h = {"content-type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def fetch_metrics():
    """Return summed spec-decode counter values (sum across per-engine label lines)."""
    out = {m: 0.0 for m in SPEC_METRICS}
    try:
        req = urllib.request.Request(METRICS_URL, headers=_hdr())
        with urllib.request.urlopen(req, timeout=30) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line or line.startswith("#"):
                    continue
                name = line.split("{", 1)[0].split(" ", 1)[0]
                if name in out:
                    try:
                        out[name] += float(line.rsplit(" ", 1)[1])
                    except Exception:
                        pass
    except Exception as e:
        print(f"[!] /metrics fetch failed ({METRICS_URL}): {e}", file=sys.stderr)
    return out


def one_turn(prompt):
    """Stream one greedy completion. Return (decode_tps, n_tokens, ttft_s)."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK, "temperature": 0, "stream": True,
    }).encode()
    first = None
    n = 0
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(CHAT_URL, data=body, headers=_hdr())
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    d = json.loads(data)
                except Exception:
                    continue
                dl = d.get("choices", [{}])[0].get("delta", {}) or {}
                delta = dl.get("content") or dl.get("reasoning_content")
                if delta:
                    if first is None:
                        first = time.perf_counter()
                    n += 1
    except Exception as e:
        print(f"[!] turn failed: {e}", file=sys.stderr)
        return (0.0, 0, 0.0)
    if first is None or n < 2:
        return (0.0, n, 0.0)
    dec = (n - 1) / (time.perf_counter() - first)
    return (dec, n, first - t0)


def run_phase(name, prompt_fn):
    before = fetch_metrics()
    tps_list, tok_list, ttft_list = [], [], []
    for i in range(TURNS):
        dec, n, ttft = one_turn(prompt_fn(i))
        tps_list.append(dec)
        tok_list.append(n)
        ttft_list.append(ttft)
        print(f"  [{name}] turn {i+1}/{TURNS}: decode={dec:6.1f} t/s  tok={n:4d}  ttft={ttft*1000:6.0f}ms")
    after = fetch_metrics()
    d_drafts = after[SPEC_METRICS[0]] - before[SPEC_METRICS[0]]
    d_draft_tok = after[SPEC_METRICS[1]] - before[SPEC_METRICS[1]]
    d_accept = after[SPEC_METRICS[2]] - before[SPEC_METRICS[2]]
    accept_len = (1 + d_accept / d_drafts) if d_drafts > 0 else float("nan")
    accept_rate = (100.0 * d_accept / d_draft_tok) if d_draft_tok > 0 else float("nan")
    mean_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
    return {
        "name": name, "accept_len": accept_len, "accept_rate": accept_rate,
        "drafts": d_drafts, "draft_tok": d_draft_tok, "accept_tok": d_accept,
        "mean_tps": mean_tps, "tokens": sum(tok_list),
    }


def repeat_prompt(i):
    # Ask for a verbatim reproduction -> the completion echoes CODE_BLOCK (long verbatim runs that
    # the suffix tree drafts). The trailing counter keeps each turn a distinct request id.
    return (f"Reproduce the following Python function EXACTLY, character for character, inside a "
            f"```python code block, then below it write the same function again with the single "
            f"comment '# variant {i}' added as the first line of the body. Do not change anything "
            f"else.\n\n```python{CODE_BLOCK}```")


def fresh_prompt(i):
    topics = ["ocean tides", "the printing press", "volcanic soil", "jazz improvisation",
              "coral reefs", "the Silk Road", "photosynthesis", "bridge engineering"]
    t = topics[i % len(topics)]
    return f"Write an original, detailed essay about {t}. Do not repeat sentences. Be varied and specific."


def main():
    m0 = fetch_metrics()
    if all(v == 0.0 for v in m0.values()):
        print("[!] spec-decode counters are all zero/absent at /metrics. Either the server is not "
              "running with a --speculative-config, or /metrics is unreachable. Check the base_url "
              "and that the serve uses method=\"suffix\".", file=sys.stderr)
    print(f"== suffix-decoding probe ==  model={MODEL}  turns={TURNS}  max_tokens={MAXTOK}")
    print(f"   chat={CHAT_URL}\n   metrics={METRICS_URL}")
    print("\n-- Phase A: REPEAT (verbatim echo -> suffix decoding should SHINE) --")
    a = run_phase("REPEAT", repeat_prompt)
    print("\n-- Phase B: FRESH (novel prose -> control, low accept) --")
    b = run_phase("FRESH", fresh_prompt)

    print("\n" + "=" * 78)
    print(f"{'phase':<8} {'accept_len':>11} {'accept_rate':>12} {'drafts':>9} "
          f"{'draft_tok':>10} {'accept_tok':>11} {'decode t/s':>11}")
    for r in (a, b):
        print(f"{r['name']:<8} {r['accept_len']:>11.2f} {r['accept_rate']:>11.1f}% "
              f"{int(r['drafts']):>9} {int(r['draft_tok']):>10} {int(r['accept_tok']):>11} "
              f"{r['mean_tps']:>11.1f}")
    print("=" * 78)
    if a["accept_len"] == a["accept_len"] and b["accept_len"] == b["accept_len"]:
        speedup = a["mean_tps"] / b["mean_tps"] if b["mean_tps"] > 0 else float("nan")
        print(f"VERDICT: REPEAT accept_len {a['accept_len']:.2f} vs FRESH {b['accept_len']:.2f}  "
              f"(decode {a['mean_tps']:.1f} vs {b['mean_tps']:.1f} t/s, {speedup:.2f}x). "
              f"Suffix decoding is WORKING if REPEAT accept_len >> FRESH (~1.x).")


if __name__ == "__main__":
    main()
