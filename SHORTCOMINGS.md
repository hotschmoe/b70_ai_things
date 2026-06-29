# SHORTCOMINGS.md -- open blockers + dead ends

Where the README keeps its headlines, this is where the honest negatives live: what is
blocked, what we tried that did not work, and the upstream bugs we are waiting on. Keeps the
README clean. Positive optimization lessons live in `research/LESSONS.md`; raw measurements in
`FINDINGS.md`; the chronological log in `JOURNAL.md`.

## Open blockers

- **vLLM "!!!!" garbage under concurrency -- vLLM is PAUSED.** Any vLLM serve that batches a
  prefill alongside an in-flight decode eventually emits endless `!!!!` and then poisons a
  shared KV/SSM block so even later single requests return NaN (a `docker restart` clears it).
  Root-caused to the XPU GDN/Mamba recurrent path: a NaN in the GDN forward propagates through
  attention's multiply-by-zero (`0*NaN=NaN`). The CUDA-side fixes are already in vLLM 0.23.0
  (#35219 zero-freed-SSM-blocks, #43961 MLA+linear, #44700 route-to-recurrent); the remaining
  XPU-side defect is **vllm-xpu-kernels #172** ("support fp32 ssm_state in chunk_fwd_o", NOT
  DONE). Tracking issue for our exact box+symptom: **vLLM #38994** (Qwen-3.5 garbled output,
  Intel backend, 2x Arc B70, open since 2026-04). No local fix on 0.23.0 -> we serve on sglang,
  which does not co-batch prefill+decode this way. Resume vLLM when #172 lands.
  (Evidence: JOURNAL 2026-06-27 research+A/B; mechanism verified upstream.)
- **sglang W8A8 MoE (35b-a3b) is blocked.** Two missing pieces, both real projects: (a) sglang
  has no Quark loader (the only on-disk 35b W8A8 is Quark format), and (b) sglang/XPU has no
  int8 fused-MoE kernel -- the 256 experts have no int8 path (`_get_recipe` lacks an int8
  branch; our oneDNN ops are dense-only). The benchable int8 MoE today is the **vLLM Quark**
  entry (true int8 experts via Triton fused-MoE). Unblocking = port vLLM's QuarkMoEMethod +
  build a fused grouped-int8 kernel. See `research/LESSONS.md`.
- **TP=2 BCS/oneCCL hardware wedge (reboot-only).** A TP>1 worker-init crash (or `P2P_ACCESS=1`)
  can corrupt the cross-GPU oneCCL/Level-Zero state so every later TP=2 serve `DEVICE_LOST`s at
  warmup -- and sometimes degrades even single-card ops. Recovery on THIS display-attached box
  is a full reboot (`modprobe -r xe` fails: `xe` drives the framebuffer). Guarded by
  `bin/xpu-health` (pre-flight probe) + `bin/xe-reset`; never chain crash-prone TP=2 starts.
  See `docs/P2P_GPU.md` and the GPU Discipline section of `AGENTS.md`.
- **W4A16 / W4A8 vision+MTP graft is UNVERIFIED on-GPU.** `models/graft_w4_complete.sh` rebuilt
  them mechanically (complete checkpoints), but they have not been coherence-gated yet. The
  W4A16 case in particular got a spliced multimodal config (its quant was authored text-only) --
  serve + a few prompts + one image before trusting. See `models/REFACTOR.md`.

## Dead ends (tried, measured, did not work)

- **XPUGraph multibucket capture for serving.** Custom XPU cuda-graph decode capture is great
  single-stream (int4 bs1/maxreq1 = 23.5 t/s) but COLLAPSES under concurrency: multibucket-nopad
  4.67 t/s, multibucket-pad 7.36, maxreq4 -> 3.55, c4 -> 0.75. Graph capture is a single-stream
  driver only; for concurrent serving, prefer MTP-eager. (scripts/141, 144; `research/LESSONS.md` row 6.)
- **graph + MTP stacked.** The captured speculative-decode forward HANGS under XPU graph capture
  -- the two cannot be combined today. (JOURNAL 2026-06-26/27; scripts 134-144.)
- **sglang's native `--enable-cuda-graph` flag on XPU.** No-op: graph_off 9.54 vs graph_on 9.46
  vs graph_on_si8 9.36 -- it stays eager. Our win came from the custom XPUGraph path, not the flag.
- **"Cheap" MTP flags (continuous steps, overlap schedule).** No single-stream uplift on int4
  (15.30 baseline -> 15.30-15.37); overlap-on actually hurt aggregate + TTFT. (scripts/134.)
- **TP=2 GRAPH=1 for W8A8 decode.** A ceiling, not a lever: decode is all-reduce-bound at TP=2
  (+5% only, prefill/TTFT regress). Eager + MTP is the TP=2 config. (JOURNAL 2026-06-27.)
- **P2P (`CCL_TOPO_P2P_ACCESS=1`) inside a vLLM TP>1 serve.** DANGER -- wedges the multi-GPU
  state box-wide (see Open blockers). Refused unless `I_KNOW_P2P_WEDGES=1`. (`docs/P2P_GPU.md`.)
- **vLLM W4A8 (27B, int8g) at `GRAPH=1`.** Engine init OOMs: the PIECEWISE capture buffers leave
  only 0.32 GiB for KV (needs 0.66 GiB for max-len 8192; est. usable max len 2496). So the W4A8
  vLLM entry has no working captured config -- EAGER (~6 t/s) is its only path. Raise
  `gpu-memory-utilization` or drop `max-model-len` to unblock. (2026-06-29 best-to-best bench.)

## Bench methodology note

`bin/serve-sweep --bench` must run each shelf entry at ITS OWN config -- it now honors each
entry's `GRAPH` default instead of forcing `GRAPH=0`. The old eager-forcing default undercounted
every graph-capture config ~4x (e.g. vLLM 27b-int4 8.4 -> 28.6 t/s, sglang W4A8 10.3 -> 27.3 t/s).
`--smoke` still forces eager (it only checks boot+coherence, where speed does not matter).
