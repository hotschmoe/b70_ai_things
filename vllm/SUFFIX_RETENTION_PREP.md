# Suffix decoding + prefix-cache retention -- prep + how to test on our stack

Source-read of the two untested v0.24.0 features in `vllm-xpu-env:int8g-v0240`
(vLLM v0.24.0, torch 2.12). Read-only image inspection (no GPU, no `--device`).
Serve entry: `rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh`; plumbing `rdy_to_serve/_common/lib.sh`.
Companion probes: `vllm/suffix_probe.py`, `vllm/retention_probe.py`.

Model facts pinned from a real v0.24.0 serve log of THIS model:
- `block_size = 832 tokens` -- log: "Setting attention block size to 832 tokens to ensure that
  attention page size is >= mamba page size" + "Padding mamba page size by 4.26% ...". So the
  hybrid's scheduler/hash block granularity is **832 tokens** (attention and mamba groups aligned
  to the same page). This is the unit that `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` must be a
  multiple of.
- kv groups: full-attention group + Mamba (GDN linear-attn) group -> retention DOES apply.
- cudagraph_capture_sizes = [1,2,4,6,8], max capture size 8; PIECEWISE; MTP spec=3 today.

---

## TASK 1 -- Suffix decoding

### What it is (verified from source)
- `vllm/v1/spec_decode/suffix_decoding.py::SuffixDecodingProposer`. Wraps
  `arctic_inference.suffix_decoding.SuffixDecodingCache` (**arctic_inference 0.1.1 is already
  installed** in the image; imports cleanly with NO GPU present).
- It is a **CPU-side, training-free, no-kernel drafter**. `propose()` reads
  `input_batch.token_ids_cpu` (host numpy), builds/queries a per-request suffix tree, and returns
  variable-length draft token id lists. There are **no CUDA-isms** -- nothing that breaks on XPU.
  Verify is the normal target forward + `RejectionSampler` (unchanged from MTP).
- **It REPLACES the drafter; it does NOT compose with MTP.** `gpu_model_runner.py` picks exactly
  one drafter by `speculative_config.method`; `method="suffix"` -> `SuffixDecodingProposer`. You
  run EITHER MTP OR suffix, not both. (Conceptually they could stack, but this vLLM build has no
  such path.)
- `num_speculative_tokens` is a **MAX**, not a fixed count. Suffix proposes a *dynamic* number per
  request per step (0..max). If `num_speculative_tokens` is omitted it defaults to
  `suffix_decoding_max_tree_depth` (24).

### Config fields (`config/speculative.py`)
- `suffix_decoding_max_tree_depth` (default 24) -- caps prefix-match + speculation length.
- `suffix_decoding_max_cached_requests` (default 10000) -- global cross-request suffix tree size
  (FIFO evict). 0 disables the global tree (prompt-local trees still used).
- `suffix_decoding_max_spec_factor` (default 1.0) -- max_spec_tokens = factor * prefix_match_len.
- `suffix_decoding_min_token_prob` (default 0.1) -- only speculate tokens with freq-prob >= this.

### GPU-kernel need: NONE. XPU-safety: HIGH.
CPU proposing + the same int8/GDN target forward we already serve. No new device kernels, no new
collectives. The MTP-specific `patches/sitecustomize.py` bits are inert-but-harmless with suffix:
block (1) patches `qwen3_5_mtp` (never instantiated -> no-op); block (2) capture-safe all_gather and
block (4) mamba-ptr fix stay loaded and are orthogonal.

### Graph-capture interaction (the one real caveat)
Suffix drafts are **variable length**, so the target-verify token count per step varies
(1 + draft_len). PIECEWISE cudagraph only REPLAYS for counts in `cudagraph_capture_sizes`
([1,2,4,6,8]); any other size runs the compiled-but-uncaptured path (correct, just no replay speed).
With the default `num_speculative_tokens`=24 most verify steps land >8 -> uncaptured -> you lose the
captured-decode win that makes this box fast.
Mitigation for a first coherent+fast test: **cap `num_speculative_tokens` small (e.g. 7)** so the
common verify size (1+accepted, up to 8) stays inside the captured set, and optionally widen
`CAPSIZES`. Do NOT expect capture to cover the whole variable range.

### Model requirements: NONE (training-free, works on the target as-is).

### EXACT serve command (suffix, A/B vs the MTP daily driver)
Suffix REPLACES MTP, so pass `SPEC` directly (this makes serve.sh ignore `MTPTOK`) and keep the
verify batch inside the capture set. Keep prefix caching ON (independent; helps TTFT not decode).

```
cd /mnt/vm_8tb/github/b70_ai_things
SPEC='{"method":"suffix","num_speculative_tokens":7,"suffix_decoding_max_tree_depth":24,"suffix_decoding_max_spec_factor":1.0,"suffix_decoding_min_token_prob":0.1}' \
CAPSIZES='1,2,4,6,8' \
PREFIXCACHE=1 \
./bin/gpu-run bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh
```
- `num_speculative_tokens`: start at **7** (keeps 1+spec within CAPSIZES=8). Then sweep
  {7 -> 15 -> 24} watching decode t/s: higher tree depth raises accept on repetitive loads but
  pushes verify past the captured sizes. If you go >7, widen `CAPSIZES='1,2,4,6,8,16'`.
- `PREFIXCACHE=1` stays (retention/MTP notes below unaffected). PUSH_AR/vision defaults unchanged.
- To A/B the launch-bound tradeoff cleanly, also capture the MTP baseline (the shelf default) with
  the same probe.

### Metrics used by `vllm/suffix_probe.py`
`vllm:spec_decode_num_drafts_total`, `..._num_draft_tokens_total`, `..._num_accepted_tokens_total`
(verified in `v1/spec_decode/metrics.py`; `/metrics` at the server ROOT, not under /v1).
Accept length = `1 + accepted/drafts`.

Run:
```
API_KEY=<key> python3 vllm/suffix_probe.py http://192.168.10.5:18080/v1 qwen36-27b-w8a8-sqgptq-mtp 6 400
```
### Expected evidence (suffix)
- Phase REPEAT (verbatim echo) accept_len clearly ABOVE Phase FRESH (which should sit ~1.x).
- REPEAT decode t/s > FRESH decode t/s. If REPEAT ~= FRESH ~1.0 accept, suffix isn't drafting
  (check `method`, that arctic_inference imported, that /metrics counters move at all).

---

## TASK 2 -- VLLM_PREFIX_CACHE_RETENTION_INTERVAL

### Semantics (verified: envs.py + kv_cache_coordinator.py + single_type_kv_cache_manager.py)
- Env is `int | None` (unset = None = dense = today's behavior). Units = **tokens**, and it is
  validated to be `>= 0` AND `% scheduler_block_size == 0`. On this model **scheduler_block_size =
  832**, so the value must be a **multiple of 832**.
- It only affects groups with `SlidingWindowSpec` or `MambaSpec`. If a model has neither, init
  RAISES ("no sliding-window or Mamba KV cache group ... Unset it"). Our hybrid HAS a Mamba group,
  so it is accepted. Full-attention groups ignore it (they stay dense/fine-grained).
- `MambaManager.reachable_block_mask` implements it: which mamba state snapshots are kept as
  prefix-cache-hit boundaries.
  - `None` -> dense: a snapshot at every 832-token boundary (default; most memory).
  - `0`    -> keep only the latest replay boundary (most aggressive sparsification).
  - `>0`   -> keep ONE snapshot per interval-sized segment, plus the replay-boundary tail.
- Effect: each cached prefix costs far fewer (large) mamba-state blocks, so idle prefixes SURVIVE
  eviction longer under memory pressure -> warm resends keep hitting where dense had evicted them.
  Trade: hit granularity coarsens to the interval (a resend lands on the nearest interval boundary
  below the match, not the nearest 832 boundary).

### Sensible value for the 248K coding daily driver
Coding sessions grow one long shared prefix and re-send it; the mamba snapshots are the expensive
part. Pick a coarse interval that still lets long prefixes rehit cheaply:
- **`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=8320`** (= 832 * 10, ~10k tokens) as the first value:
  ~10x fewer mamba snapshots per sequence, hits still land every ~10k tokens -- fine for multi-turn
  coding where the reused prefix is tens of thousands of tokens.
- Alternatives to sweep: `4160` (832*5, finer hits, more memory) and `16640` (832*20, cheaper,
  coarser). All must be multiples of 832 or init RAISES.

### Wedge / overflow risk: LOW.
Retention is **scheduler-side CPU block bookkeeping** (which blocks are marked cacheable). It does
NOT touch `MambaSpecDecodeGPUContext.initialize_from_forward_context` or any `data_ptr()` packing,
so it does NOT add a new path for the XPU USM `>=2**63` signed-int64 overflow. That overflow is in
the WORKER and is already handled by `patches/sitecustomize.py` block (4), which runs regardless of
retention. Retention just changes eviction granularity on an already-working prefix-cache serve.
Only init failure mode: value not a multiple of 832 (validation ValueError) -- a clean refuse at
config time, NOT a wedge. Keep the existing PREFIXCACHE=1 shim; nothing new to guard.

### EXACT serve command (retention)
`B70_EXTRA_ENV` injects the env into the container (lib.sh -> `-e`). Keep the daily-driver defaults
(MTP spec=3, PREFIXCACHE=1, MAXLEN 253952, etc.) and just add the retention env:
```
cd /mnt/vm_8tb/github/b70_ai_things
B70_EXTRA_ENV="VLLM_PREFIX_CACHE_RETENTION_INTERVAL=8320" \
PREFIXCACHE=1 MAXLEN=253952 \
./bin/gpu-run bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh
```
(For a quick smoke, drop MAXLEN back to the recipe default; retention still applies. Retention +
suffix can be combined -- both are independent envs.)

### Validation probe: `vllm/retention_probe.py`
Sends a ~10k-token fixed prompt, resends immediately (warm), then after progressively more unique
intervening traffic, reporting resend TTFT + `vllm:prefix_cache_{queries,hits}_total` delta each
time. Run it TWICE against the SAME box:
```
# run 1: dense default (no retention env on the SERVE)
API_KEY=<key> python3 vllm/retention_probe.py http://192.168.10.5:18080/v1 qwen36-27b-w8a8-sqgptq-mtp 5 10000
# restart serve WITH B70_EXTRA_ENV=VLLM_PREFIX_CACHE_RETENTION_INTERVAL=8320, then run 2 (same args)
```
NOTE: the value that matters is the SERVER's env; the probe prints the client-side env only as a
label. You are comparing the two tables.

### Expected evidence (retention)
- **Dense run:** resend TTFT starts warm (~gap=0), then CLIMBS back toward cold TTFT and hit_rate
  collapses to ~0 after a few rounds of intervening traffic (the mamba snapshots got evicted).
- **Retention run:** resend TTFT stays LOW (near the gap=0 warm hit) and hit_rate stays >0 out to
  MORE intervening traffic than dense -- i.e. retention keeps the prefix warm where dense evicted
  it. That divergence is the pass.
- Because block_size=832 and the interval is 8320, the ~10k-token base prompt spans one full
  retention segment + the replay tail, so a retained hit is expected; a <8320-token prompt would
  only ever keep the replay-boundary tail (still hits on exact resend, but is a weaker test) --
  keep the probe prompt >= ~10k tokens.

---

## Blockers / open items
- Neither feature has been RUN on GPU here (source-read + probe-authoring only; GPU was off-limits).
  Both need a real TP=2 serve to confirm.
- Suffix + capture: expect uncaptured verify steps whenever draft_len pushes 1+spec past CAPSIZES.
  The capped `num_speculative_tokens=7` config keeps the common case captured; validate decode t/s
  before trusting a large tree depth.
- arctic_inference 0.1.1 imports without a GPU but was not exercised end-to-end; watch init logs for
  its lazy import on the first `method="suffix"` serve.
