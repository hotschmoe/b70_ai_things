# Engine lessons (running design notes for a future in-house Zig engine)

Owner directive 2026-11-21: the engine architecture is designed separately. This file records what we learn porting
Strata's hotschmoe-dd feature set onto TensorFold Flash-Next (Zig/Level Zero, two B70s). Append dated bullets under the
right heading, with an evidence path. Facts, not opinions; say what it cost.

## TensorFold architecture: what works

- Zig + Level Zero + OpenCL C kernels: a small, readable host path; the critical-path ledger (markers <1% overhead) made
  launch gaps and host stalls visible per op (regen.sh, B/ledger/). Recommend: build the ledger in from day one.
- Descriptor windows (XPU_FN_DESC): one precomputed submission per decode window cut host-side work; worth C4 +14% /
  C8 +21% over the classic path at int-rf2.
- Host unit tests that run without a device (CPU references for kernels, mocked-engine server tests) let five workers
  build features on the CPU while production held both GPUs (2026-11-21).
- 2026-11-21 The server/engine seam (api.Engine + Info capabilities + a scripted fake engine) is the best part: the whole HTTP layer (reasoning split, qwen3-coder tool streaming, usage, errors, queueing) was parity-tested against Strata on CPU in minutes with zig/tests/server/fake_serve and a real tokenizer + chat template, no weights, no GPU. A new engine should keep this seam and the fake. Evidence: tensorfold/flashnext/api_parity/ ; JOURNAL.md [API] 2026-11-21 19:30.
- 2026-11-21 Capability flags (Info.structures, greedy_only, penalties, logprobs) make refusals precise and discoverable; adding a prompt-side fallback for a missing capability was a 100-line change behind one env flag. Evidence: zig/src/server/server.zig fallbackActive, PATCHES.md.

## TensorFold architecture: what hurts

- Feature flags are mutually exclusive in places where they should compose: descriptor windows vs paged int8 KV were
  exclusive, and desc mode preallocated KV for abi.max_rows=16 x --context, capping context at ~20-24K (2026-11-21).
  Lesson: KV memory must be a shared page pool sized by a budget, never rows x max context, and the fast path must be
  built over the paged representation from the start.
- The Flash-Next XPU path shipped greedy-only (server refused temperature > 0); sampling was an afterthought, so the
  first sampled path reads top-64 rows back to the host and runs at ~60 vs ~95 tok/s greedy. Lesson: design the
  sampler (device-side top-k + per-position keyed draw + spec-decode acceptance) into the decode window.
- Request-field allowlist in openai.zig silently dropped unknown sampling fields. Lesson: unknown fields log once;
  supported-field tables live next to the capability flags.
- The release binary embeds the worktree path, so the same commit built elsewhere has a different sha; gates must
  re-check the frozen binary.
- 2026-11-21 Engine limits leak as late 500s: the lane engine refuses prompt + max_tokens + 1 > window, while the server preflight allows prompt + max_tokens == window, so a clamped request failed only on the GPU. One shared admission predicate (owned by the engine, queried by the server) would remove this class. Evidence: workers/API/gpu1/server.log vs gpu2.
- 2026-11-21 Strict env registry: reading a new TF_* variable through sys.environment.value panics at runtime unless registered (and registry tests pin the list); plain environ_map reads work. Env knobs are hard to add cleanly. Evidence: fake_serve panic "unregistered strict environment parser" in workers/API/tf_fake.log history.
- 2026-11-21 A mature server surface (about 400 server files) is also coupling: a change in chat.zig/prompt.zig touches golden fixtures (55 frozen goldens). Good safety net, slow to evolve; budget a golden run per change. Evidence: zig build test-golden.
- 2026-11-21 Per-request inputs (image rows, M-RoPE frame) were first attached to the physical sequence slot; the merged eviction work moves a stream between slots, so they had to become per-prefill-pass inputs set on whatever slot the pass runs on. Lesson: request-scoped data belongs to the stream/pass, never to the KV slot. Evidence: tensorfold/flashnext/JOURNAL.md [VIS] 2026-11-21; b70/dd-vis 2b552f0c (xpu_lane_prompt.zig step).
- 2026-11-21 Classic (non-paged) bf16 KV sized at rows x --context failed engine init with ZE out of device memory at ctx 32768 with weights resident (24.2 GB/rank); 16384 passed. Same lesson as the desc cap above. Evidence: dd-vis-run/gpu1/server.log.

- 2026-11-21 [INT] Integration hot spots: three independent features (sampling, API parity, vision) collided in the same
  three places -- the request field allow-list in server/openai.zig plan(), the Forward state struct in
  families/flashnext/xpu_forward.zig, and the capability block of native/xpu_flashnext.zig open(). A registry (fields,
  per-feature state, capability setters) instead of one hand-edited list each would make features merge without conflicts.
  Evidence: b70/dd-serve dc8370e5, fb0e6006; tensorfold/flashnext/PATCHES.md.
- 2026-11-21 [INT] Build cost: one ReleaseFast xpu-native build is about 6 min at -j4; test-runtime-api (278 s) and
  test-lane-api-host (197 s) mostly recompile the native API module per test root. The binary embeds the worktree path,
  so the same commit built in two worktrees has different sha256 (9423485c vs 7bce9c8e at 372bf1b2), and a frozen
  production build must be re-gated. Want: path-independent (reproducible) builds and a shared, prebuilt test module.
  Evidence: workers/INT/suite.log.

- 2026-11-21 [SAMP] Correction to the "~60 vs ~95" note above: that pair came from a 63-token reply including TTFT. With
  256-token decode at C1, v1 sampling measured 95.0 tok/s at T 0.6 against 101.9 greedy. The real loss was penalties
  (the card's instruct set, presence 1.5): 70.3 tok/s at C1 and 28.7/stream at C4, because penalized rows read whole
  248K-vocab shard rows back (7.6 ms a window against a 16 ms verify). Lesson: measure with fixed-length decode and
  every request shape clients send; penalties need a device path too. Evidence:
  /mnt/vm_8tb/b70/results/tf-flashnext/workers/SAMP/prof-base/prof.log.

- 2026-11-21 [KV] Implicit resource contracts break when features meet: descriptor padding rows leased one real KV page,
  and eviction requires an idle pool at start, so paged desc + eviction failed only at GPU init (PagePoolNotIdle).
  A spare page outside the ledger fixed it. Lesson: scratch/trash memory never comes from an accounted pool.
  Evidence: workers/KV/q1/server.log; b70/dd-kv 0929b00c.
- 2026-11-21 [KV] Quantized-KV dequant cost hides in "cheap" per-element helpers: a software fp16 widen per element
  (needed because IGC folds fpext(fptrunc)) made int8 paged prefill 30% slower than bf16 pages; hoisting the 4+4 group
  scales per key gave +38% prefill, bitwise equal. Lesson: profile the dequant inner loop first; dequant per group.
  Evidence: tensorfold/flashnext/JOURNAL.md [KV] 2026-11-21 21:45; b70/dd-kv f70795f7.
- 2026-11-21 [KV] There is no CPU OpenCL device on this host, so kernel correctness work went into Zig CPU references
  (paged vs dense bitwise, park/restore under pool pressure); they caught layout bugs before any lease.
  Evidence: zig/src/families/flashnext/xpu_dpage_ref.zig (test-fn-dpage-host).
- 2026-11-21 [KV] Gates must send what clients send: raw-completion prompts without the chat template gave false
  "EOS at long context" failures; Strata had no raw endpoint to compare. Evidence: workers/KV/q5, q12, strata-ref.

## Strata features that were hard to port, and why

- Long context + parking (Strata: 200K, int8 KV, --kv-resident 65536 + RAM-backed slots): TensorFold had no paged
  fast path and the eviction work lived on an unmerged branch.
- Vision: Strata's CPU sidecar (strata-vision + mmproj) is reusable as a process; the work is the prompt splice
  (pad-token rows replaced by embeddings on both TP ranks, M-RoPE positions, encoder cache). ~930 ms encode per image.
- API parity: Strata applies model-card sampling defaults when the client sends none, clamps max_tokens to the room
  left instead of erroring, and supports tool_choice required/named; clients poll /slots.
- 2026-11-21 Strata has no token-grammar decoder either: tool_choice required/named is a forced call opening written into the prompt, response_format is a system instruction plus validation. The XPU lane backend has no masked first-token path, so TensorFold structured output could not use libtfgrammar anyway. Porting the Strata behaviour (prompt-side) was cheaper than a device mask. A future engine needs a device-side logit mask if it wants real grammars. Evidence: server/structured.py vs zig/src/server/openai.zig responseFormatInPrompt.
- 2026-11-21 Strata defaults: it samples with the model card (T 1.0, top_p 0.95, top_k 20) when a client sends nothing and clamps over-long max_tokens; TensorFold upstream defaults to refusing (400). Small request-semantic defaults like these are what real clients depend on; list them before porting. Evidence: api_parity/parity.py KNOWN, workers/API/strata_views.json.
- 2026-11-21 Vision: the embedding splice was easy (rows over pad-token embeddings, both ranks, PLE n-gram hashes the pad id as Strata does). The positions were the hard part: every TF attention/indexer kernel derives the rope angle from the implicit row index, so M-RoPE needed four kernel copies taking a position table (fn_mrope.cl). A "shifted frame" (Strata positions + sum(n - max(nx,ny))) keeps relative positions identical and leaves decode, MTP and drafts untouched. Evidence: tensorfold/flashnext/JOURNAL.md [VIS] 2026-11-21; b70/dd-vis 2b552f0c (xpu_mrope.zig).

## Kernel and hardware lessons (B70, see HARDWARE.md)

- Plain read ~604 GB/s; M=1 GEMV reaches 96-97%. Raw GGUF 98-byte q3 blocks get 17-21% of bandwidth vs ~90% for an
  aligned split layout: repack at load. Codebooks in SLM are 2.3x faster than __constant.
- Vector path goes ALU-bound at M >= 8 rows; DPAS holds 87-93%: concurrency needs DPAS decode kernels.
- ~3 us per short kernel, 2.1-2.3 us immediate append: launch count is the C1 currency.
- D2H ~4.5 GB/s (platform limit) vs P2P push ~13 GB/s: avoid host readback on the decode critical path.
- Device memory compression makes memset buffers read at 3.2 TB/s: never benchmark on them.
- 2026-11-21 Pass rope positions as a device table (cell*3 ints) from day one; one kernel then serves text and multimodal. Measured cost of the table path: none (9K-token prompt TTFT 10.45 s text vs 10.47 s with an image, desc+paged int8). Evidence: tensorfold/flashnext/JOURNAL.md [VIS] 2026-11-21; b70/dd-vis 2b552f0c.
- 2026-11-21 Image decoding is a parity surface of its own: Strata relies on Pillow (EXIF transpose, alpha over white, any
  format to PNG) in front of an stb_image encoder; TF pins exact decoder releases (libjpeg-turbo 3.1.4.1, libwebp 1.6.0)
  loaded at runtime and refuses JPEG/WebP when they are absent, so a host without them silently diverged (unturned phone
  photos). Both decoders built from source with zig cc in 32 s. Lesson: ship the pinned decoders with the engine (or link
  them) and fail loudly at startup, not per request. Evidence: tensorfold/flashnext/JOURNAL.md [VIS] JPEG EXIF/WebP entry.

- 2026-11-21 [SAMP] Vocab-split head (124160 per card): a two-stage device top-K is enough to sample without a gather.
  Stage 1 is 256 partitions x K. Stage 2 is one 256-thread work-group per row with a sub-group reduce, so it reads
  K candidates per card. Penalized rows add a gather of the window's logits (<= 128 ids per card), penalized on the
  host. Each window's draw cost fell from 1.47 to 0.21-0.35 ms (T 0.6) and from 7.6 to 0.9 ms (penalized). The
  device merge matched a host merge on 94K candidates. Evidence: workers/SAMP/prof-new, check-new; xpu_sample.zig.

## Scheduler and serving lessons

- Host-bound first (event churn, host MTP), then device-bound: fix host overhead before kernels.
- Recorded command windows hung until signal events + drain were explicit; a recorded path needs its own tests.
- Prefill over paged desc KV measured ~865 tok/s at 9K tokens vs 1560 at 2K on the dense path (2026-11-21, VIS gate):
  long-prompt prefill must be benchmarked at the real prompt sizes (58-90K), not 2K.
- Benchmarks must include the production features (sampling, long context, vision, tools), not only greedy 16K.
- 2026-11-21 Queueing beyond --parallel is the lane host job, not the server: the server never rejects (admission off by default), which matched Strata behaviour; the fake engine needed an explicit lane cap to test it. Real check on 4 lanes: 8 concurrent requests all 200, max 1.5 s. Evidence: workers/API/gpu2/gates.log.
- 2026-11-21 Thinking budget is enforced in the generic lane stream (forced close token), so it worked on XPU with no engine work: budget 48 gave 86 reasoning chars. Put policy like this in the shared scheduler layer, not in per-backend code. Evidence: workers/API/gpu2/gates.log.

- 2026-11-21 [SAMP] Keyed per-position draws make MTP lossless with plain exact-match acceptance: drafts on equal drafts
  off for the same seed. Coupled drafts use the same keyed Gumbel noise in the MTP pick (Strata's STRATA_SPEC_COUPLED
  idea). They gave no acceptance gain here: 0.35-0.38 at T 0.6 against 0.38 greedy at C1. The MTP head's quality is
  the limit, not sampling, so acceptance work belongs in the drafter. Kept as XPU_FN_COUPLED=1, default off.
  Evidence: workers/SAMP/prof-base vs prof-new.

## Backend portability

- (to fill: what in TensorFold is Level Zero specific vs portable; OpenCL C kernels vs SYCL/ESIMD; what Strata's SYCL
  stack gives that L0 does not.)
- 2026-11-21 The CPU fake_serve builds in 7 s (Debug) with no device libraries, while xpu-native ReleaseFast takes 5-7 min under the shared build lock; having a device-free server target is what let parity work proceed while Strata held the cards. Keep the engine behind a vtable with a no-device build target. Evidence: workers/API/build*.log.
- 2026-11-21 The vision encoder as a separate process (Strata's strata-vision over a pipe, SVE1 files) ported unchanged and is backend-agnostic, but it is CPU and serial: 0.93 s for 49 tokens, 3.9 s for 200, 5.8 s for 289 on 8 threads; that dominates image TTFT. A future engine should run the encoder on the GPU or overlap encode with text prefill; keep the sha256 encoder cache (repeat image 0-3 ms). Missing libturbojpeg/libwebp on the host silently limits JPEG/WebP. Evidence: tensorfold/flashnext/JOURNAL.md [VIS] 2026-11-21; b70/dd-vis 2b552f0c.

## Operations

- Five concurrent zig builds (~2.5 GB each) + agents caused a global OOM (2026-11-21 17:59) that killed the user systemd
  manager; enabled units auto-started on its restart. At most two builds host-wide, under cpu-run limits.
- 2026-11-21 [INT] One host-wide cpu-run lease per zig invocation makes a 25-step test suite wait behind every other
  build between steps (8 min waits seen); run all host steps in one zig build call under one lease instead.
  Evidence: workers/INT/host_suite2.sh vs host_suite.sh timings in workers/INT/suite.log.
