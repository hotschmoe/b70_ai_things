# LocalMaxxing overnight loop -- 27B/35B B70 4-bit + W8A8

**Created:** 2026-08-20
**Status:** LOOPING (30m scheduler)
**Goal:** move measured C1 on 1x or 2x B70 for NVFP4, W4A16, and
W8A8-INT8. Do not chase LocalMaxxing aggregates (1139 C64, 438
concurrency=64, 248 p4096/o32).

This file is the standing prompt. Re-read it every fire. Details:
`docs/20260820_lmx_overnight_loops.md` (handoff),
`docs/20260820_lmx_overnight_deadends.md` (closed paths).

### Living header

| field | value |
|---|---|
| Last loop | 5 (P1b Q8_DOORS A/B RUNNING) |
| Last JOURNAL | `2026-08-20ba` |
| Next pick | **P1b attach** -- parse doors 1 vs 0 if benches exist. Do not start a second serve. |
| Blocked on | none. P1b holds both cards (pid 260538). |
| Hold Ornith NVFP4 GRAPH no-MTP | **34.9** `bench_code` c1, Paris/391. STICKY=0 M1=0. |
| Hold W8A8 3.8 DSpark k1bar | **31.9** `bench_code` c1 @122880. |
| Hold 3.8 GPTQ-Int4 MTP4 (S1) | **47.58** post-first p512/g128, no draft-INT4. |
| Hold 3.8 GPTQ-Int4 MTP4 + draft-INT4 (W1) | **65.08** post-first p512/g128. G1 Paris/391. |
| Hold Pliny OBLITERATED Q8_0 TP2 | **30.89** median / **32.14** warmup g128. G1 Paris/391. |
| DD | PARKED. Do not start. :18080 is research. |

Published LocalMaxxing C1 ceilings (do not treat as our holds):
SergiioB 1x GPTQ-Int4+draft-INT4 **112.7**, Steve 2x AutoRound MTP5
**101.9**, SergiioB 35B GPTQ MTP4 **204.6**, Steve 2x Quark W8A8 35B
**85.9**, 5090 Ornith NVFP4 **258.8**. Our S1 was 47.58 on the same
3.8 GPTQ ckpt without the 2026.08.19 overlay.
W1 with that overlay is **65.08**.

## L.0 Read order (before any edit or GPU)

1. This file, living header + queue.
2. `docs/20260820_lmx_overnight_loops.md` -- last 3 loops and NEXT PICK.
3. `docs/20260820_lmx_overnight_deadends.md`.
4. JOURNAL.md bottom (from `2026-08-20at` / LOOP 63-65).
5. `./bin/gpu-run --status` and `/mnt/vm_8tb/b70/lmx_overnight/STATUS`.
6. One pick. Write the ledger even if you only attach to a running serve.

## L.1 30m check-in, not a hard kill

- A fire is **one verdict** or a RUNNING handoff.
- If card 0/1 is busy with our overnight container, **attach**: G1 if
  not done, phase_bench if G1 GO, parse logs. Do not start a second serve.
- If busy with a foreign holder, CPU-only (patches, kernel source,
  docs). Do not steal the lease.
- If a serve is healthy, leave it up for the next fire unless the next
  pick needs a different checkpoint. Set `STOP=1` only after the bench
  file exists and NEXT PICK names a different model.
- Do not sit idle for 25 minutes. If GPU load will take >30m, start it
  under `gpu-run`, write RUNNING, and return.
- Never two `gpu-run` serves. P2P off. No DD. No emul NVFP4 G1 (D18).
  No `KVDTYPE=bfloat16`. No 51MB GDN-OFF overlay. No D16/101.9 retry.

## L.2 Queue (pick the first unblocked row)

**P1** Pliny OBLITERATUS Qwen3.8-27B Q8_0 (operator love; just landed)
GGUF: `models/files/qwen3.8-27b/obliterated-q8/Qwen3.8-27B-OBLITERATED-Q8_0.gguf`
  (~29 GB, hf file only -- do not fetch the whole repo).
Serve: `./bin/gpu-run bash llamacpp/serve_qwen38_obliterated_q8.sh start`
Sweep: `bash llamacpp/sweep_obliterated_q8.sh`
Image `qwen38-b70:latest`. Q8 fused doors ON, Q4K doors OFF.
Card: temp 0, repeat_penalty 1.15, thinking off, empty system.
2x TP2 first (29 GB weights). Then GPU_COUNT=1 if VRAM allows.
G1 Paris/391. Metric: phase_bench p512/g128 n=5.
Compare to 0xSero Q4_K_M lab doors **43.8** (different quant, same arch)
and S1 GPTQ 47.58. Q8_0 is weight-only, NOT W8A8.
Profile next: Q8_DOORS=0 vs 1; COMM_DIRECT_Q8; 1x vs 2x;
llama-bench tg128/pp512 if the binary exists in-image.
Custom kernel track (from 0xSero/lab map):
- Decode is reorder MMVQ Q8_0 x Q8_1 via `dpct::dp4a`, NOT XMX/DPAS.
- Prefill is dequant->F16 GEMM (MMQ disabled, oneDNN off in this image).
- Enable MMVQ pair/triple/quad; leave Q4K reorder off.
- `GGML_SYCL_COMM_DIRECT_Q8=2` only. `=3` is DEVICE_LOST (lab 2026-08-16).
- Never `CCL_TOPO_P2P_ACCESS=1` (vLLM TP>1 wedge). llama.cpp N=2 is
  a device-0 USM sum kernel, not oneCCL.
- If Q8_0 is ~4x slower than 43.8, first prove we are on reorder MMVQ
  not DMMV (`GGML_SYCL_DEBUG` / `[Q8-DEDUP]`). Next kernel is lab
  DP4A2 + GDN-quad SG24, not vLLM P2P and not enabling MMQ.
Do not demote k1bar 31.9.

**W1** 3.8 W4A16 draft-INT4 (DONE LOOP 1)
G1 Paris/391. Median post-first **65.08** vs S1 47.58.
Do not re-run unless a new overlay lands. Do not
publish as W8A8. Do not demote k1bar 31.9.
Board 112.65 remains the ceiling, not the gate.

**O2** Ornith GRAPH-safe INT4 compute (L65 Half!=BF16)
`int4_gemm_w4a16` is fp16-out. `out.to(x.dtype)` did not satisfy
GRAPH dummy_run. Options: opaque custom op with bf16 schema, or
`--dtype float16` only for the drafter path. Unit test first.
Then GRAPH=1 MTP3 + `B70_DRAFT_MTP_INT4=1` G1. Hold stays 34.9
until it beats 34.9.

**O3** Pack Ornith MTP **routed experts** to INT4 at load_weights
L65 only packed 5 dense linears (78->20 MB). Experts still BF16
Triton (~1.5 GB) so eager INT4 was 4.4 vs 4.3. This is the miss.

**O4** Grouped NVFP4 apply (llm-scaler M=1 1D block_load / tile-map)
Python sticky/M1 was 33.1/33.3 vs 34.9. Needs kernel work, not
another host copy. Can span fires. Unit `test_fused_moe_apply.py`
must stay XPUGraph PASS.

**O1** LocalMaxxing-method C1 of the 34.9 hold (p512/g128 n=5,
cache off) so NVFP4 has a submit-shaped number. Measurement, not
a speed unlock. Do after W1 or when a serve is already the hold.

**W8** W8A8 k1bar 31.9 -- only if W1 and O* are blocked. Do not
regress it.

## L.3 Write order at the end of every fire

1. JOURNAL.md bottom: `### YYYY-MM-DD<letter> - LOOP N: ...`
   CONTEXT / CONFIG / COMMAND / RESULT / VERDICT.
2. Append `## LOOP N` to the overnight ledger. Update NEXT PICK.
3. Dead-end packet if you closed a path.
4. Living header on this file.
5. RESEARCH_TODO overnight blurb if Next pick or a hold moved.
6. Commit and push. ASCII only.

Ledger block:

```
## LOOP N -- YYYY-MM-DDThhmmZ -- <one-line pick>

Picked:
Why this, not the other open row:
GPU:
Command:
Log:
Result:
Verdict: GO / NO-GO / BLOCKED / DEAD-END / RUNNING
Changed beliefs:
Next pick:
Do not:
Restore:
JOURNAL:
```

## L.4 Standing no

- `CCL_TOPO_P2P_ACCESS=1` in TP>1.
- Daily driver / `daily_driver_serve.sh`.
- Emul NVFP4 G1 (D18 bangs).
- Rewrite old `scripts/NN_*.sh`. Copy to a new number or backend root.
- Photocopy 101.9 / 112.65 / 204 onto NVFP4 MoE without measuring.
- Kill a healthy overnight serve at the 30m boundary.
- Start 4x B70 work. 1x or 2x only.
