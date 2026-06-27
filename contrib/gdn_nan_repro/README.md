# gdn_nan_repro -- minimal repro for the qwen3_5 GDN NaN ("!!!!") under load

Root cause + full A/B in JOURNAL.md (2026-06-26 / 2026-06-27 entries). One-liner: vLLM 0.23.0 on
Intel Arc B70 (Xe2/Battlemage) produces **NaN logits** in the Gated-DeltaNet (GDN / linear-attention)
layers of Qwen3.6-27B (`model_type qwen3_5`) under **mixed prefill+decode batching and/or longer
decodes**. The NaN then poisons a shared KV block (vLLM #35219 `0*NaN` propagation) and spreads to
every request -> global "!!!!" until a `docker restart`. Open upstream: vLLM #38994 (our exact 2x B70
box), vllm-xpu-kernels #172. No serve-flag fix on 0.23.0 (enforce-eager, GDN kernels v0.1.10 #411,
--no-enable-chunked-prefill, VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0, concurrency cap -- all ruled out).

## The decisive signal
Request token logprobs while the server is under load. If the logits are NaN, vLLM cannot serialize
them and returns:
```
HTTP 400  {"error":{"message":"Out of range float values are not JSON compliant: nan", ...}}
```
That 400 is unambiguous proof of NaN logits. A healthy server returns valid logprobs.

## Scripts (stdlib python3; edit PORT/host as needed)
- `backhoe_req.json` -- a real captured agentic chat request (Qwen3.6 tool-calling, 2 msgs, 4 tools).
- `dd_loadprobe.py [port] [anchors] [probes] [maxtok]` -- ramp N long streaming "anchor" decodes, then
  fire P concurrent probes under that load; classify each (OK / DEGEN). Reproduces the failure.
- `dd_rawtokens.py [port] [anchors] [maxtok]` -- ramp anchors, fire ONE logprobs probe -> prints the
  `400 nan` (NaN confirmed) or the literal tokens (clean). The cleanest pass/fail.
- `dd_single.py [port] [maxtok]` -- ONE long decode, no concurrency (shows even a single long
  generation degenerates once the serve is in a bad state).
- `dd_mixload.py [port] [anchors] [burst] [waves] [interval] [bmax] [amax]` -- fuller sustained
  mixed-load harness with a verdict histogram.

Auth: the scripts read an API key from `/mnt/vm_8tb/b70/secrets/dd_api_key` and send it as a Bearer.
If the server under test does not enforce a key, the header is ignored (harmless); if the file is
absent, create an empty one or edit the `KEY = ...` line.

## Pass/fail for any candidate fix (new vLLM build, SGLang, P/D disagg, etc.)
Point the scripts at the candidate endpoint and require BOTH:
1. `dd_rawtokens.py <port> 8 800` -> valid logprobs, **no** `400 nan`.
2. `dd_loadprobe.py <port> 8 12 500` -> all probes `OK` (coherent content/tool_calls), zero DEGEN.
Then re-confirm after several minutes of sustained load (the bug is also a slow shared-state poison).
