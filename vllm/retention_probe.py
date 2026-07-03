#!/usr/bin/env python3
"""VLLM_PREFIX_CACHE_RETENTION_INTERVAL validation probe (vLLM v0.24.0 XPU, hybrid GDN Qwen3.6).

WHAT RETENTION DOES (v1/core/kv_cache_coordinator.py + single_type_kv_cache_manager.py):
  Prefix caching on this hybrid model runs mamba "align" mode -- to allow a prefix HIT at a token
  boundary it must keep a full Mamba STATE SNAPSHOT at that boundary. Dense caching keeps a snapshot
  at EVERY 832-token block boundary (block_size=832 on this model; see the serve log "Setting
  attention block size to 832 tokens"), which is very memory-hungry, so under intervening traffic
  the block pool evicts an idle prefix's snapshots quickly -> a warm resend MISSES and re-prefills.
  VLLM_PREFIX_CACHE_RETENTION_INTERVAL=<mult-of-832> keeps only ONE snapshot per interval-sized
  segment (+ the replay-boundary tail), so a cached prefix costs far less and survives longer under
  pressure -> warm resends keep HITTING where the dense default has already evicted them.

HOW THIS PROBE PROVES IT:
  1. Prime a big (~PROMPT_TOKENS) fixed prompt, record its cold TTFT.
  2. Resend immediately -> warm hit, low TTFT (baseline).
  3. Loop: inject increasing UNIQUE large-prompt traffic (fills/evicts the cache), then resend the
     SAME big prompt and record TTFT + the prefix_cache hit-rate delta for that resend.
  Run once with the retention env UNSET (dense default) and once SET, and compare: with retention
  the resend stays warm (low TTFT, hit>0) out to a larger intervening-traffic gap.

Prometheus counters (names verified, v1/metrics/loggers.py):
  vllm:prefix_cache_queries_total, vllm:prefix_cache_hits_total  -> per-resend hit rate = d_hits/d_q.

Dependency-free (urllib). Usage:
  retention_probe.py <base_url> <model> [rounds=5] [prompt_tokens=10000] [api_key]
  <base_url> = OpenAI base, e.g. http://192.168.10.5:18080/v1  (/metrics derived from it)
  api_key also read from $API_KEY.
"""
import sys, json, os, time, random, urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
PROMPT_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
API_KEY = sys.argv[5] if len(sys.argv) > 5 else os.environ.get("API_KEY", "")

_root = BASE
if _root.endswith("/v1"):
    _root = _root[:-3]
METRICS_URL = _root + "/metrics"
COMP_URL = BASE + "/completions"

CACHE_METRICS = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total")

_WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
          "sigma tau upsilon phi chi psi omega vector matrix tensor kernel scalar gradient buffer "
          "pointer register cache latency throughput bandwidth pipeline scheduler quantize decode").split()


def _fixed_text(n_words, seed):
    r = random.Random(seed)
    return " ".join(r.choice(_WORDS) for _ in range(n_words))


# ~PROMPT_TOKENS tokens. ~0.75 tokens/word for this vocab is conservative; use 1.4 words/token target.
BASE_PROMPT = ("Study the following reference log and remember it verbatim.\n"
               + _fixed_text(int(PROMPT_TOKENS / 0.7), seed=1234)
               + "\n\nWhen asked, answer only 'OK'.")


def _hdr():
    h = {"content-type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def fetch_metrics():
    out = {m: 0.0 for m in CACHE_METRICS}
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


def send(prompt, max_tokens=1, stream=True):
    """Send a completion; return TTFT seconds (streamed) or total wall (non-stream)."""
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "stream": stream,
    }).encode()
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(COMP_URL, data=body, headers=_hdr())
        with urllib.request.urlopen(req, timeout=600) as r:
            if not stream:
                r.read()
                return time.perf_counter() - t0
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
                if d.get("choices", [{}])[0].get("text"):
                    return time.perf_counter() - t0
    except Exception as e:
        print(f"[!] send failed: {e}", file=sys.stderr)
    return time.perf_counter() - t0


def resend_measure(tag):
    m0 = fetch_metrics()
    ttft = send(BASE_PROMPT, max_tokens=1, stream=True)
    m1 = fetch_metrics()
    dq = m1[CACHE_METRICS[0]] - m0[CACHE_METRICS[0]]
    dh = m1[CACHE_METRICS[1]] - m0[CACHE_METRICS[1]]
    rate = (100.0 * dh / dq) if dq > 0 else float("nan")
    print(f"  [{tag}] resend TTFT={ttft*1000:7.0f}ms  hit_rate={rate:5.1f}%  (dq={int(dq)} dh={int(dh)})")
    return ttft, rate


def inject_traffic(round_idx, n_prompts):
    """Fire n_prompts UNIQUE large prompts to fill/evict the prefix cache between resends."""
    for j in range(n_prompts):
        junk = ("Analyze this unique log and summarize in one word.\n"
                + _fixed_text(int(PROMPT_TOKENS / 0.7), seed=round_idx * 100000 + j))
        send(junk, max_tokens=1, stream=False)


def main():
    ret = os.environ.get("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "<unset/dense>")
    print(f"== retention probe ==  model={MODEL}")
    print(f"   (server-side) VLLM_PREFIX_CACHE_RETENTION_INTERVAL as reported by THIS client env: {ret}")
    print(f"   NOTE: the value that matters is the SERVER's env; run this probe once against a")
    print(f"   dense-default serve and once against a retention serve, then compare the tables.")
    print(f"   prompt ~{PROMPT_TOKENS} tokens ({len(BASE_PROMPT)} chars)  rounds={ROUNDS}")
    print(f"   comp={COMP_URL}\n   metrics={METRICS_URL}\n")

    print("-- prime (cold) --")
    cold = send(BASE_PROMPT, max_tokens=1, stream=True)
    print(f"  cold TTFT={cold*1000:.0f}ms")
    print("-- immediate resend (should be a warm hit at both settings) --")
    resend_measure("gap=0")

    print("\n-- increasing intervening traffic, resend the SAME prompt after each round --")
    print(f"{'round':<7}{'injected_prompts':>17}{'~injected_tokens':>18}")
    total_junk = 0
    results = []
    for i in range(1, ROUNDS + 1):
        n = i * 2  # 2,4,6,... unique big prompts per round
        inject_traffic(i, n)
        total_junk += n
        ttft, rate = resend_measure(f"round{i} (+{n} prompts, ~{n*PROMPT_TOKENS//1000}k tok)")
        results.append((i, total_junk, ttft, rate))

    print("\n" + "=" * 66)
    print(f"{'round':<7}{'cum_prompts':>12}{'resend_TTFT_ms':>16}{'hit_rate_%':>12}")
    print(f"{'cold':<7}{'-':>12}{cold*1000:>16.0f}{'-':>12}")
    for i, cum, ttft, rate in results:
        print(f"{i:<7}{cum:>12}{ttft*1000:>16.0f}{rate:>12.1f}")
    print("=" * 66)
    print("PASS EVIDENCE: with retention SET, resend TTFT stays LOW (near the gap=0 warm hit) and")
    print("hit_rate stays >0 out to more intervening traffic than the dense-default run, where TTFT")
    print("climbs back toward cold and hit_rate collapses to 0 once the snapshots are evicted.")


if __name__ == "__main__":
    main()
