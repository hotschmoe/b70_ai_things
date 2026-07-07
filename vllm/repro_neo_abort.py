#!/usr/bin/env python3
# Repro driver for the MTP-verify x piecewise-cudagraph NEO linear_stream.h:84 abort.
# Fires forced-decode (ignore_eos) requests at a chosen context length and reports
# tokens-to-crash. Detects engine death via HTTP 500 / connection error. No tokenizer
# dependency: we send raw chars and read back usage.prompt_tokens to confirm ctx.
#
#   PORT=18091 KEY=... PROBE_CTX_CHARS=175000 PROBE_MAXTOK=3000 PROBE_RUNS=6 \
#     python3 vllm/repro_neo_abort.py <label>
#
# Verdict per run: CLEAN (finished forced decode) | CRASH (engine died) | ERR.
import json, os, sys, time, urllib.request, urllib.error

LABEL       = sys.argv[1] if len(sys.argv) > 1 else "cfg"
HOST        = os.environ.get("PROBE_HOST", "192.168.10.5")
PORT        = os.environ.get("PORT", "18091")
KEY         = os.environ.get("KEY", "").strip()
CTX_CHARS   = int(os.environ.get("PROBE_CTX_CHARS", "175000"))   # ~50K tok for code-ish text
MAXTOK      = int(os.environ.get("PROBE_MAXTOK", "3000"))
NRUNS       = int(os.environ.get("PROBE_RUNS", "6"))
TIMEOUT     = int(os.environ.get("PROBE_TIMEOUT", "600"))
BASE        = f"http://{HOST}:{PORT}"

def _hdr():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h

def models():
    r = urllib.request.Request(BASE + "/v1/models", headers=_hdr())
    return json.load(urllib.request.urlopen(r, timeout=20))["data"][0]["id"]

def completion(body):
    r = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                               headers=_hdr())
    with urllib.request.urlopen(r, timeout=TIMEOUT) as x:
        return json.load(x)

# A long, varied, code-ish filler so the prompt is a realistic large agentic context.
_CHUNK = (
    "def process_block_{i}(state, config):\n"
    "    # audit trail entry {i}: reconcile KV blocks against the scheduler output\n"
    "    total = 0\n"
    "    for j, blk in enumerate(state.blocks):\n"
    "        if blk.ref_count > 0 and not blk.is_free:\n"
    "            total += blk.num_tokens * config.head_dim\n"
    "    assert total >= 0, 'block {i} accounting invariant'\n"
    "    return total, state.seq_len + {i}\n\n"
)

def build_prompt(nonce):
    buf = [f"# repro session {nonce}\n# Read the following module carefully.\n\n"]
    n = 0; i = 0
    while n < CTX_CHARS:
        s = _CHUNK.format(i=i); buf.append(s); n += len(s); i += 1
    buf.append("\n\n# TASK: Summarize every process_block function above in extreme, "
               "exhaustive detail, one paragraph each, then propose refactors. Think out loud "
               "at length; do not stop early.\n")
    return "".join(buf)

def alive():
    try:
        urllib.request.urlopen(urllib.request.Request(BASE + "/health", headers=_hdr()), timeout=10)
        return True
    except Exception:
        return False

def main():
    mid = models()
    print(f"[{LABEL}] served={mid} ctx_chars={CTX_CHARS} maxtok={MAXTOK} runs={NRUNS}", flush=True)
    verdicts = []
    for i in range(NRUNS):
        pr = build_prompt(f"{LABEL}-{i}-{int(time.time())}")
        t0 = time.time()
        try:
            d = completion({"model": mid, "prompt": pr, "max_tokens": MAXTOK,
                            "temperature": 0, "ignore_eos": True})
        except urllib.error.HTTPError as e:
            print(f"  run{i}: *** CRASH/HTTP{e.code} after {time.time()-t0:.0f}s "
                  f"(engine likely died); health_alive={alive()}", flush=True)
            verdicts.append("CRASH"); break
        except Exception as e:
            print(f"  run{i}: *** ERR {type(e).__name__}: {e}; health_alive={alive()}", flush=True)
            verdicts.append("CRASH" if not alive() else "ERR"); break
        u = d["usage"]; txt = d["choices"][0]["text"]
        print(f"  run{i}: CLEAN prompt_tok={u['prompt_tokens']} comp_tok={u['completion_tokens']} "
              f"dt={time.time()-t0:.0f}s tail={txt[-60:]!r}", flush=True)
        verdicts.append("CLEAN")
    print(f"[{LABEL}] VERDICTS={verdicts}", flush=True)
    # exit 3 == crash observed (for scripting)
    sys.exit(3 if "CRASH" in verdicts else 0)

if __name__ == "__main__":
    main()
