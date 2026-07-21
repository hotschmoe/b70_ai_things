# 2026-07-21 kickoff: headless box, DP=2 NVFP4 daily driver, 4-bit quant research campaign

This is a session-kickoff prompt. Paste it (or "follow docs/20260721_headless_dp2_4bit_kickoff.md")
into a fresh session AFTER the machine reboot. It assumes CLAUDE.md + memory are loaded; facts below
override stale memory where they conflict.

## Mission (3 phases, in order)

1. Post-reboot verification: the display was PHYSICALLY REMOVED -- both B70 cards should now be
   headless. Verify what that changes (card-1 clocks, xe module reload).
2. Stand up the new daily driver: NVFP4 27B, DP=2 (one single-card replica per card, NO TP), with
   CALIBRATED fp8 KV -- soak-gated before it is trusted unattended.
3. Run the 4-bit quant research campaign (W4A8, W4A4, NVFP4 optimization) on card 1 while card 0
   keeps serving. Heavy multi-agent orchestration: parallel agents (opus/grok/codex-class) write,
   compile, and analyze; the COORDINATOR alone touches the GPUs, serially, health-gated.

## State as of handoff (2026-07-21 evening, this repo at HEAD)

- DD right now: W8A8 27B TP=2 `hotschmoe-dd` on :18080, image `int8g-v0251` (vLLM 0.25.1),
  MAXLEN=253952, PIECEWISE+MTP3+PUSH_AR+prefix cache+parsers. It crashed overnight (NEO
  linear_stream.h:84 graph-replay accumulation, results/logs/dd_w8a8_crash_20260721.log) and the
  reclaim shim was PORTED to the W8A8 shelf (sitecustomize block (7), B70_XPU_CG_RECLAIM=1000,
  default ON for GRAPH=1). Relaunched and verified engaged in both workers.
- The linear_stream leak needs a CROSS-DEVICE COLLECTIVE inside the captured graph (TP=1 replays
  300k+ clean) -- this is a core motivation for DP: single-card replicas have no collective, so the
  whole leak class (and every oneCCL/TP wedge) is out of the picture.
- W8A8 shelf gotchas discovered today (both documented in serve.sh):
  - The small-M w8a16 routing (B70_W8A16_M_MAX) DOUBLES int8 weight residency -> ctx-gated
    (ON only at MAXLEN<=8192). Kernel TODO: make int8_gemm_w8a16 consume the s8s8 [K,N] layout.
  - The push-AR block REASSIGNS DOCKER_ENV -- env appends must go BELOW it or they are silently
    dropped. (Bit us once already.)
- bin/xpu-health default image is `vllm-xpu-env:v0230` which NO LONGER EXISTS -> always pass
  `--img vllm-xpu-env:int8g-v0251` (or fix the default in bin/xpu-health as a first task).
- The INSTALLED systemd unit (/etc/systemd/system/b70-daily-driver.service + override.conf drop-in)
  is STALE (points at an older NVFP4 TP=2 config). On boot it will auto-start the WRONG thing.
  FIRST ACTION after reboot: `systemctl status b70-daily-driver`, stop/disable or repoint it
  (needs sudo) before doing anything else on the GPUs.

## Phase 0 -- post-reboot verification (do this before any serve)

1. Deal with the stale systemd autostart (above). Make sure no container holds the cards:
   `docker ps`, `./bin/gpu-run --status`.
2. `./bin/xpu-health --img vllm-xpu-env:int8g-v0251` -> both cards OK.
3. Headless check: is anything still bound to a framebuffer? (`cat /proc/fb`, `lsmod | grep xe`
   refcount vs the old baseline ~5). Then the BIG one: does `modprobe -r xe` now work with all
   containers stopped? If yes, `bin/xe-reset` becomes a REAL non-reboot wedge recovery -- retire the
   "reboot is the only recovery" rule (update AGENTS.md + memory). Test this ONCE, carefully, with
   zero GPU users, and reload after.
4. Card-1 clock check: card 1 was display-attached and DOWNCLOCKED (~2/3 of card 0: 23.5 vs 15.3
   t/s on the same workload). Re-run a quick same-workload A/B (`gpu-run --card N` + any shelf
   single-card serve or a matmul bench) to see if the asymmetry is GONE. This matters for DP=2
   replica balance and for trusting card-1 research numbers. Record in JOURNAL.

## Phase 1 -- new daily driver: NVFP4 27B DP=2 + CALIBRATED fp8 KV

Target: two independent single-card replicas of nvidia/Qwen3.6-27B-NVFP4 (b70_daily_0 on card 0,
b70_daily_1 on card 1) behind the nginx proxy on :18080, served id `hotschmoe-dd`, API key from
/mnt/vm_8tb/b70/secrets/dd_api_key.

Per-replica config (vllm/nvfp4/serve_nvfp4_27b.sh, which defaults TP=1):
- MODE=fused GRAPH=1 + stock MTP (single-card captured+MTP was the box champion: 38.7 t/s stock,
  67 t/s code at MTPTOK=5). Sweep MTPTOK {3,5} per replica under the real-coding harness.
- Hard walls that still apply: UTIL=0.85, CAPSIZES=1,2,4,8.
- Keep CGRECLAIM default ON (belt-and-suspenders; TP=1 should not leak, but it costs nothing).
- PUSH_AR is TP-only -- irrelevant at DP; do not set it.
- fp8 KV: CALIBRATED ONLY. KV_FP8=1 + KV_SCALES=vllm/nvfp4/kv_scales_nvfp4_27b.json (sitecustomize
  block (10) injects the scales). HISTORY, do not repeat it: the 2026-07-06 DD garbage was
  UNCALIBRATED (scale-1.0) fp8 KV at TP2+MTP -> repetition; the user was burned and explicitly
  demands calibrated-only. Single-card fp8 KV with real scales passed a 118,856-token needle and
  2026-07-21 Result 4 passed 4000-tok forced decode + 6-way concurrent -- but a FULL soak has not
  run. GATE before trusting: 27k+ single-stream + 36k concurrent-token soak + 4-gram repetition
  scan + gate_concurrent_coherence 18/18 + a real-coding bench_code.py pass PER REPLICA.
- MAXLEN: 131072 per replica (the proven fp8-KV single-card ctx). We LOSE the 253952 single-session
  ctx of TP=2 -- accepted trade for wedge-immunity + a free research card.
- Agentic parity (MANDATORY, bit us twice): TOOLCALL=1 TOOLPARSER=qwen3_coder REASONPARSER=qwen3
  THINK_BUDGET=4096 OVERRIDE_TEMP=0.6 SERVED_FORCE=hotschmoe-dd. Missing REASONPARSER also breaks
  the think-budget inject (400s on chat).
- Proxy: bin/dp_nginx.conf via vllm/daily_driver_serve.sh DD_REPLICAS=2. Prefer hash/affinity
  routing (prefix-cache warmth) over round-robin if cheap; at minimum least_conn. GOTCHA: editing a
  single-file bind mount needs `docker restart` (inode swap), not nginx reload.
- dd-watchdog: update the watch spec for two replicas; its self-heal (docker restart) is the whole
  point of DP -- soft garbage on one replica must not require a reboot or touch the other card.
- Persistence: update vllm/deploy/b70-daily-driver.service + the /etc override to the DP=2 config
  (DD_MODEL=vllm/qwen36-27b-nvfp4 DD_REPLICAS=2 + env above), `sudo cp` + daemon-reload, and VERIFY
  a systemctl restart brings up the right thing.
- Research mode: `bash daily_driver_serve.sh` (or docker stop b70_daily_1) drops to card-0-only
  serving; card 1 is then free for Phase 2. DP=2 is the DEFAULT when not researching.
- Fallback DD (known-good, commit-pinned): the W8A8 TP=2 shelf serve with reclaim --
  `NAME=b70_daily_0 PORT=18080 TP=2 MAXLEN=253952 SERVED=hotschmoe-dd API_KEY=$(cat
  /mnt/vm_8tb/b70/secrets/dd_api_key) ./bin/gpu-run bash rdy_to_serve/vllm/qwen36-27b-w8a8/serve.sh start`.

Shelf discipline: when the DP replica config is measured coherent+fast (sweep-gated), bake it into
rdy_to_serve/vllm/qwen36-27b-nvfp4 as THE entry (settings, not a sibling dir). Update README table.

## Phase 2 -- 4-bit research campaign on card 1 (card 0 keeps serving)

Standing target reminder (CLAUDE.md): compressed-tensors is the artifact format; GPTQ is the
current default calibration; W8A8/W4A8 exercise the INT8 XMX fast paths. The user has explicitly
green-lit starting W4A4 work now (supersedes the "later frontier" ordering note).

Track A -- W4A8 (highest expected value):
- Checkpoint exists: models/files/qwen3.6-27b/w4a8-sqgptq (compressed-tensors). Single-card fits
  easily (~14 GB weights) -> big KV headroom, no TP.
- Kernels exist on the sglang side: oneDNN int4_gemm_w4a8 (prefill, int8 act) + int4_gemm_w4a16
  (decode, fp16 act) -- sglang/W4A8_BUILD.md, shim sglang/patches/w4a8_shim.py. The vLLM-0.25.1
  path needs the same ops wired into a scaled-mm kernel class (mirror vllm/contrib/vllm_int8_xpu/
  xpu_int8.py; the int4 analog xpu_int4.py already exists in that dir -- start there).
- Goal: single-card W4A8 27B serve on vLLM 0.25.1, captured+MTP, bench_code.py + HumanEval+ vs
  NVFP4 (quality bar: NVFP4 27B = 0.988/0.945) and vs W8A8. Decide if W4A8 challenges NVFP4 as the
  4-bit serve path.
- Reuse today's lesson: watch weight residency when a kernel path keeps extra layouts.

Track B -- NVFP4 further optimization (the incumbent 4-bit):
- Open lever: MTPTOK/accept tuning single-card, prefix-cache-on numbers, fp8-KV headroom (the
  Phase 1 soak doubles as research data).
- Closed NO-GOs -- do NOT re-attempt: int8-XMX prefill (group16 < K32 DPAS, dot32 identity ISA-
  proven dead), capture-the-all_gather, PP=2, host-barrier redirects, decode all-reduce cheap fixes,
  argmax lever. All have hard evidence in JOURNAL 2026-07-16..21.

Track C -- W4A4 (frontier, accuracy-first then kernels):
- Known physics: s4xs4->s32 DPAS is real on Xe2 (dpas.s4.s4 disasm, int-exact) but naive ESIMD caps
  ~64 TOPS vs int8's 367; decode is BW-bound so W4A4's win is PREFILL compute + weight/act traffic.
- The blocker is ACCURACY: W4A4 needs rotation (Hadamard/QuaRot-style) + online FWHT we do not have.
  Phase C1 = quantization-quality research OFF-GPU or on card 1: produce a rotated W4A4
  compressed-tensors 14B (not 27B) checkpoint, eval HumanEval+ degradation vs W4A8/W8A8. Only if
  quality survives does C2 (kernel work: fused rotation + s4 DPAS GEMM) start. Journal a go/no-go.

Eval + identity discipline (all tracks): served ids and output dirs encode method+scheme
(...-W4A8-sqgptq etc.); curl /v1/models + cross-check evals/configs/models.yaml before trusting any
number; every quant must retain the vision tower (sglang/graft_vision.py to graft if dropped);
uniform workload = vllm/nvfp4/bench_code.py + evals/ HumanEval+.

## Orchestration + safety rules for this campaign

- GPU lease: EVERY GPU touch through ./bin/gpu-run (--card 1 for research; the DD holds card 0 --
  never lease card 0 while the DD serves). Coordinator runs GPU commands SERIALLY, health-gated
  (xpu-health between crashy experiments). Agents NEVER touch the GPU directly.
- Agents (opus/grok/codex-class, use them heavily and in parallel): kernel/microbench authoring,
  single-file icpx compiles (low-RAM, safe), quantization scripting, eval harness runs against the
  serving endpoint, doc/analysis writeups, adversarial verification of findings. Use worktrees for
  parallel code edits; sync through the coordinator.
- RAM-bomb rules stand: NO heavy multi-TU docker/image builds while the DD serves (those need DD
  DOWN + MAX_JOBS=4 + docker --memory 90g); never setsid a build. Single-file compiles are fine.
- Wedge posture at DP: expected failure = soft per-replica garbage -> docker restart (watchdog).
  If Phase 0 proved modprobe -r xe works headless, xe-reset is the new hard recovery; else reboot.
  The oneCCL/P2P TP cautions stay in force for any TP experiment (P2PACCESS=1 in serve = forbidden).
- JOURNAL every experiment (config -> command -> result -> verdict), newest at bottom; update
  RESEARCH_TODO.md ordering; commit+push often. Do not rewrite old numbered scripts.

## Success criteria

1. Phase 0 findings journaled (headless effects: clocks, xe reload) same-day.
2. DP=2 NVFP4 DD serving with calibrated fp8 KV, soak-gated, systemd-persistent, watchdog-covered.
3. At least Track A (W4A8 single-card on vLLM) at a measured go/no-go vs NVFP4, and Track C at a
   quality go/no-go, each with JOURNAL entries and committed artifacts.
