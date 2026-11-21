# TensorFold Flash-Next on two B70s (campaign brief, 2026-11-20)

Orchestrator: Claude Opus 5.5 (main session). Leads: two Claude Opus 5.5 (high effort) agents, each overseeing workers
(Claude Sonnet 5.5 subagents via the Agent tool, and/or Codex gpt-6.1-terra xhigh via
`codex exec -m gpt-6.1-terra -c model_reasoning_effort="xhigh" -s danger-full-access -C <dir> --add-dir <dirs> -o <out.md> "<prompt>"`
run in the background with its log polled). Leads own design, review every worker diff, and keep the journal honest.

User direction (2026-11-20): "start patching tensorfold to run flashnext across our two cards and lets chase our high ceiling
numbers! take big items from strata to bring over (calibrated 8-bit kvcache, evict to ram and retrieve without recompute,
prefix caching, etc). remember we want to profile latency not bandwidth. it doesnt matter if we are 80% effective on
bandwidth if the cards are waiting for data! have some fun!"

## Why (evidence)
- TensorFold (Zig, Level Zero, OpenCL C via ocloc, nGEN prompt GEMMs; Apache-2.0) already runs Qwen3.8-27B on ONE of our B70s:
  C1 142, C2 228, C4 304, C8 265 aggregate tok/s, prefill 1.9-2.5K, concurrent==solo bitwise
  (strata/flash-next/concurrency-0216/tensorfold-probe/JOURNAL.md). Strata Flash-Next on TWO cards: C1 133.5, C2 147-158,
  C4 ~148, C8/C16 capped, prefill 0.9-1.2K.
- Flash-Next ceilings on our box (analyst/opus_zig_engine_design.md): C1 ~250, C4 480-670 agg, C16 880-1100 agg,
  prefill 6-12K tok/s. Strata is far below because of launches, host waits, per-row/per-slot serial work, card 1 idle.
- TensorFold PR #3603 (fetched; worktree /mnt/vm_8tb/b70/build/tensorfold/pr3603 at 207aaae) has Flash-Next on XPU with TP over
  Level Zero P2P, host-RAM PLE, confidence MTP, Xe2 DPAS tiles, adaptive solo chunks - but it needs the MLX 4-bit checkpoint
  (~79 GiB on cards, does not fit 2x32 GB) and its two-card tf-xpu-tp-bench HUNG on our host with a GuC job timeout + GT reset
  on card 1 (probe journal step f). zeDeviceCanAccessPeer reports 1 both ways on kernel 7.1.
- Reviews: concurrency-0216/analyst/fable_tensorfold_review.md (TF architecture, lessons), opus_zig_engine_design.md
  (roofline/design), fable_decode_window_walk.md + opus_sched_prefill_restore_walk.md (Strata internals),
  concurrency-0216/LEDGER.md + ledger/ (Strata measured op census), strata/flash-next/update-0216/feature_map.md.

## Fixed facts
- Model: Qwen3.8-Flash-Next ISTA GSQ-RCO IQ3_XXS GGUF: /mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/ (shard 1 =
  weights, shard 2 = PLE table 28.8 GB, keep in host RAM/page cache). MTP drafter: /mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt
  (Strata format; the GGUF/MLX MTP alternatives may exist - check). Tensor quant types include IQ3_XXS, IQ2_XS, IQ2_XXS,
  IQ4_NL, Q2_0 (ISTA custom type 42), Q6_K, Q8_0, BF16, F32 - read from the GGUF header.
- Strata (MIT) is the reference engine and kernel donor: source /mnt/vm_8tb/b70/build/strata-0216/src (branch dd-0.2.16),
  qualified binary /mnt/vm_8tb/b70/build/strata-0216/strata-dd-0216, config strata/flash-next/dd/hotschmoe-dd-0216.json,
  launcher strata/flash-next/dd/nl_launch_0216.sh. Port kernels deliberately and keep the MIT notice. Note TensorFold
  upstream rules (no porting from other engines, no tool co-author trailers): this is OUR fork; nothing goes upstream
  without the user's explicit OK.
- TensorFold toolchain: Zig 0.17.0 at /mnt/vm_8tb/b70/build/tensorfold/toolchain, nGEN headers at .../ngen-src, host ocloc
  26.22, c++ via the zig c++ wrapper (see concurrency-0216/tensorfold-probe/README.md for exact build commands and the
  card-pinning recipe: gpu-run --card N env ZE_AFFINITY_MASK=N ...). Card 0 = PCI 0b:00.0, card 1 = 44:00.0.
  Known flag: tf-xpu-qwen27-means-test fails (5.1e-5 vs 2e-6 bound) on our host - Lead B should run that down early
  (compiler/runtime/wrapper) since numerics matter for parity work.
- Hardware: 2x Arc Pro B70 (Xe2, 32 GB, ~608 GB/s), PCIe Gen3 x16 each, Threadripper 1950X (2 NUMA dies), 121 GB DDR4.

## Code layout
- Integration branch b70/flashnext-2card in a new worktree /mnt/vm_8tb/b70/build/tensorfold/fn-int (from pr3603 HEAD 207aaae).
  Lead A works in /mnt/vm_8tb/b70/build/tensorfold/fn-model (branch b70/fn-model), Lead B in .../fn-runtime (branch
  b70/fn-runtime); merge into b70/flashnext-2card at milestones (Lead A integrates; small, reviewed merges). Commit often
  in these local branches. Never push to GitHub, never comment on TensorFold issues/PRs.
- Campaign notes (tracked): /mnt/vm_8tb/github/b70_ai_things/tensorfold/flashnext/ : JOURNAL.md (append-only,
  CONFIG -> COMMAND -> RESULT -> VERDICT, one entry per stage/major result, both leads, prefix entries [A] or [B]),
  STATUS.md (each lead keeps its own short section current: milestone, last result, blockers, next GPU need),
  GPU.md (GPU claims, see below), design docs as needed. Raw logs/traces: /mnt/vm_8tb/b70/results/tf-flashnext/.
  Do not commit in b70_ai_things (the orchestrator commits). ASCII only.

## Lead A - model path (Flash-Next two-card, our quant)
M-A1 Design (CPU): read PR #3603 Flash-Next XPU (layers, TP layout, MTP, PLE, prompt/decode paths) and Strata's EP layout
  (expert placement per card, push/return/PSUM, PLE, MTP). Choose the two-card split for decode LATENCY on Gen3 (expect
  EP-style: dense/attn/GDN on card 0 or split, routed experts across both; TP all-reduce twice per layer is likely worse) and
  the cross-card transport (coordinate with Lead B's measurements). Write tensorfold/flashnext/DESIGN-A.md.
M-A2 GGUF loader for the ISTA IQ3_XXS pack (all tensor types above) + OpenCL C decode kernels for those quant types (port
  from Strata's SYCL kernels / ISA paths; start correct, then fast), with CPU reference parity tests per type/shape
  (gguf-py dequant as oracle). nGEN/DPAS prompt GEMMs for the IQ types later (prefill).
M-A3 First tokens: Flash-Next forward on two cards, greedy, fixed prompts; oracle = Strata logits for the same prompts
  (STRATA_DUMP_FIRST_LOGITS or a short Strata serve under gpu-run on :18134) and/or TF's own Flash-Next reference
  (CUDA/Metal family code + any Python reference). Gate: top-1 agreement and small KL over a few hundred positions,
  deterministic, identity/coherence on fixed prompts.
M-A4 MTP drafter at batch, PLE from host RAM, multi-stream (TF lanes host), concurrent==solo and drafted==serial gates.
M-A5 Speed: C1/C2/C4/C8/C16 + prefill with Lead B's latency profiler; chase the ceilings.

## Lead B - runtime, latency profiling and Strata features
M-B1 Latency profiler for TF XPU (start NOW on the 27B single-card build, which already runs): per-kernel device timestamps on
  both cards, host submission timeline, event/fence waits, copies; produce per-window critical path and, per card, BUSY vs
  WAITING time attributed to cause (waiting on host submit, on the other card, on a copy, on a dependency, launch gaps).
  The headline metric is time the cards spend waiting for data/work, not bandwidth efficiency.
M-B2 Platform latency constants on our host (microbenches, Zig + Level Zero): kernel launch latency (immediate vs regular
  command lists, graph/replay), event signal -> wait latency same card and cross-card, host-visible memory polling,
  copy-engine latency for small transfers (4 KB - 4 MB) H2D/D2H/P2P, P2P read/write latency and bandwidth by size,
  and a STABLE cross-card transport for per-layer small exchanges (P2P via zeDeviceCanAccessPeer, IPC handles, host-pinned
  staging). Reproduce/contain the tp-bench GuC hang (one bounded attempt, then avoid that path). Feed results to Lead A.
M-B3 Strata features into TF (designed for Flash-Next hybrid state: GDN recurrent state + QSA KV): calibrated int8 KV cache
  (per-position/head scales as Strata does; validate NLL/top-1 vs bf16), KV eviction to host RAM and restore WITHOUT
  recompute (async on copy engines, never block other streams), prefix caching (TF already keeps planned-point states for
  the 27B - generalize), many slots (C8/C16) with RAM tier, and a scheduler that mixes prefill chunks with decode rows
  (cold C4 TTFT target << Strata's 17/39/61/84 s).
Lead B may validate features on the 27B first where that is faster, then port to the Flash-Next path.

## GPU coordination
- Every GPU touch through /mnt/vm_8tb/github/b70_ai_things/bin/gpu-run (--card N for one card + ZE_AFFINITY_MASK=N).
  bin/xpu-health before a lead's first and after its last GPU run of a session; after a hang/DEVICE_LOST: bin/xe-reset then
  xpu-health; stop a path after two consecutive failures.
- Until "STRATA DONE" appears in strata/flash-next/concurrency-0216/ORCHESTRATOR.md, the Codex Strata worker has priority
  (it is wrapping up); keep GPU use to short single-card tests until then.
- tensorfold/flashnext/GPU.md: append a line before each GPU claim (lead, cards, expected minutes) and after release.
  Do not hold a lease idle; no claim over 30 minutes without a note. Card 1 can usually serve Lead B single-card tests
  while Lead A uses card 0, but two-card runs need both.
- Builds/heavy CPU via bin/cpu-run (MAX_JOBS<=4, docker --memory 90g); timed measurement windows under bin/cpu-run --quiet.
  Host /tmp is tmpfs: scratch on disk. Host RAM 121 GB: the PLE table and pinned buffers count.
- Do not start/stop hotschmoe-dd or the frontdoor; never use ports 18080/18124. Test ports 18150-18159.

## Evidence rules
No speed claim without matched config, coherence check, /v1/models identity, health and teardown. Profiled runs are
diagnostic; confirm timing claims unprofiled. Mark inferred vs measured. Light evidence, no verification loops, but
correctness gates (parity, concurrent==solo, drafted==serial) are mandatory before speed numbers count.

## Reporting
Each lead: keep STATUS.md current; append JOURNAL entries; report back to the orchestrator at each milestone (your final
message when you stop) with: what works, numbers, open problems, next milestone and its GPU needs. If you run long, stop at a
milestone boundary with state written down; the orchestrator will resume you.

## USER TARGETS (2026-11-21 04:35 UTC, user's personal decode/prefill targets, two cards, Flash-Next IQ3_XXS)
- Decode aggregate tok/s: C1 >= 100, C2 >= 150, C4 >= 300, C8 >= 500.
- Prefill: >= 1500 tok/s single stream, more in aggregate.
- User: "we have the hardware and zig gives us the tooling, we just need to close the software gap. IT IS DOABLE".
Roofline check (orchestrator, from BYTES-hc2 [I]):
- C1: 5.26 GB/card/window = 8.65 ms ideal at 2.27 tok/window -> 262 tok/s ceiling; 100 needs ~38% of roofline.
- C4: 7.37 GB = 12.1 ms ideal at ~1.76 tok/stream -> ~580 ceiling; 300 needs ~52%.
- C8: ~9-10 GB (expert union ~37/rank/layer) = ~16 ms ideal -> ~880 ceiling; 500 needs ~57%.
Today: C1 82.6 (~32%), C4 151 (~26%), C8 192 (~22%), prefill 503.
=> The concurrency targets need multi-row efficiency about 2x better. Levers: grouped expert kernels at R=8-16 near bandwidth; HC weights read once per window, not per row group, and maybe split across cards instead of replicated (1.33 GB/card); host/launch overhead ~0 at C4/C8 (one submit per window); deeper MTP at C2/C4 when acceptance allows.
