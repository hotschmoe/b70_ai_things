# Qwen3.8-27B W4A8 full-send -- dead-end packets

Format (copy this). Retry only if the packet's retry condition is now true.

```
## D?? -- YYYY-MM-DD -- <one line>

Closed: <what we will not do again>
Evidence: JOURNAL heading + command + the number or error
Why it is dead: <mechanism, not vibes>
Retry if: <precise condition, or NEVER>
Related: K-id / Path H|X|S
```

Imported from prior W4A8 / 3.8 work so LOOP 1 does not pay the same tuition.

---

## D01 -- 2026-06-20 -- AutoRound cannot export W4A8

Closed: AutoRound / INC as the 3.8 W4A8 producer
Evidence: `research/w4a8/AUTOROUND_W4A8_FEASIBILITY.md`. auto_round 0.13.1
  `formats.py::check_and_reset_format` hard-asserts `bits==8` for int8-dynamic-act
  compressed-tensors export. Native `auto_round` format serves as W4A16 on XPU
  (INC ignores the int8 acts).
Why it is dead: exporter, not a B70 bug
Retry if: a released AutoRound grows a real W4A8 CT export AND
  `XPUW4A8IntLinearKernel` loads it. Re-read the assert; do not assume.
Related: producer = GPTQ/SQ via 151

---

## D02 -- 2026-06-28 -- grouped-128 `_int_mm` in Python is dead

Closed: split K into K/128 chunks and loop `torch._int_mm` in eager Python
Evidence: `sglang/W4A8_PLAN.md` Gate-1. M=2048 = 0.03x bf16 (89.6 ms vs 2.9)
Why it is dead: 136 small GEMMs + launch tax
Retry if: NEVER as an e2e path. A fused grouped oneDNN/ESIMD op is K2, not this.

---

## D03 -- 2026-06-28 -- per-forward int4->int8 materialize is dead

Closed: dequant int4 to a materialized int8 weight each forward, then `_int_mm`
Evidence: unpack microcost 4.26 ms/layer > the whole bf16 matmul. Storing both
  int4+int8 ~= 40 GB > 32 GB card.
Why it is dead: O(weight) every step, or 2x storage
Retry if: NEVER. Unpack must be in-register (Path X) or you stored W8A8.

---

## D04 -- 2026-06-28 -- woqgemm compute_type=int8 joint_matrix gated

Closed: `auto_round_kernel.woqgemm(..., compute_type=int8)` as the W4A8 prefill
  on this box (oneAPI DPC++ 2025.3)
Evidence: M=2048 FAIL "no matrix hardware on the target device, joint_matrix
  is not supported". M=1 ran but was not faster than fp16 woqgemm.
Why it is dead: SYCL joint_matrix runtime gate, not missing weights
Retry if: oneAPI 2026+ or a runtime that advertises joint_matrix on BMG.
  One re-probe is campaign K1-whacky. If it still fails, leave this packet.
Related: oneDNN `int4_gemm_w4a8` is the path that already works

---

## D05 -- 2026-06-28 -- torch.compile of sglang W4A8 act-quant hangs serve

Closed: `torch.compile` on the prefill per-token int8 act-quant in sglang
Evidence: `rdy_to_serve/sglang/qwen36-27b-w4a8/README.md`. Inductor
  async-worker deadlock at startup. Triton AQ (`w4a8_actquant_triton.py`)
  is the working 8.3x vs eager.
Retry if: NEVER on this inductor. Triton (or fusedq C++) only.

---

## D06 -- 2026-06-28 -- 3.6 sqgptq MLP-only as a multimodal serve

Closed: serving `Qwen3.6-27B-W4A8-sqgptq-prepacked` through the VLM path
  without a text-only config overlay
Evidence: `sglang/W4A8_PLAN.md` UPDATE 2026-06-28c. 0 vision tensors, GDN
  left BF16, `Qwen3_5ForCausalLM` / missing lm_head sample() crash.
Why it is dead: artifact + loader, not the kernel
Retry if: NEVER as the 3.8 headline. 151 keeps vision/MTP graft + surgical
  GDN INT8. A text-only overlay is a debug valve only.

---

## D07 -- 2026-06-20 -- 14B W4A8-RTN dominated by W4A16-gptq on decode

Closed: "W4A8 is automatically faster than W4A16 at M=1"
Evidence: `research/w4a8/README.md`. 14B W4A8-RTN decode 16.5 vs W4A16-gptq
  29.0 at the same 9.3 GiB VRAM. Cause: act-quant tax + unfused Path X,
  not packing.
Why it is dead as a belief, not as a scheme
Retry if: Path X fused M=1 beats Path H on 3.8 shapes (K2). Until then
  Path H is the decode default.

---

## D08 -- 2026-07-21 -- W4A4 naive accuracy is broken without rotation

Closed: W4A4 integer serve on 3.8 in this campaign
Evidence: `research/profiling/w4a4_int4_xmx_plan.md`. Fake-quant 3.6
  gate_proj: cosine 0.796 / SNR 4.1 dB. Hadamard lifts to 0.973 / 12.6 dB,
  still below a ~20 dB code-safe bar. Native s4 DPAS is 2.0x s8 MAC
  (`INT4_DPAS_PIONEER.md`) -- that is Path S microbench, not a serve.
Retry if: FlatQuant/QuaRot + an online Hadamard kernel exist AND W4A8
  Path X is already the 3.8 headline. Not before.

---

## D09 -- 2026-08-20 -- D16 GRAPH=1 hang is not the W4A8 recipe

Closed: photocopying Steve INT4-AR GRAPH=1 as the 3.8 W4A8 path
Evidence: overnight / Steve photocopy. GRAPH=1 hung on shm_broadcast.
  101.922 withdrawn (greedy margin). W4A16-autoround is not W4A8.
Retry if: NEVER as this campaign's first serve. Path H GRAPH on the
  *oneDNN* int4 ops is the 3.6-proven capture (different kernel).

---

## D10 -- 2026-08-20 -- NVFP4 is not INT4 XMX

Closed: treating Unsloth / RadixArk / Ornith NVFP4 as the W4A8 XMX path
Evidence: `nvfp4_gemm_w4a16` decompresses E2M1 into a W4A16 matmul.
  Xe2 has no FP4 tensor core.
Retry if: NEVER. `nvfp4_gemm_w4a8` (E2M1 -> s8 then DPAS) is a *different*
  research op; it still starts from an NVFP4 file. This campaign produces
  integer W4A8 via 151.

---

## D11 -- 2026-06-24 -- vLLM P2PACCESS=1 TP>1

Closed: `CCL_TOPO_P2P_ACCESS=1` inside a vLLM TP>1 serve
Evidence: AGENTS.md, `docs/P2P_GPU.md`. Hang at shm_broadcast / worker
  warmup. Raw oneCCL P2P SYCL copy works; vLLM-multiproc does not.
  Production is P2PACCESS=0 + PUSH_AR (~11 GB/s).
Retry if: NEVER in this campaign. K15 is PUSH_AR only.

---

## D12 -- 2026-07-21 -- offline_prepack_w4a8.py on GDN INT8 weights

Closed: running `research/w4a8/offline_prepack_w4a8.py` on a two-group
  W4A8+GDN-INT8 artifact
Evidence: `scripts/149` header. That packer keys on `.weight_scale`
  presence and will pack I8 channelwise GDN as if it were int4.
Retry if: NEVER. 151 Stage B already skips the INT8 group by [N,1] scale.

---

## D13 -- 2026-08-18 -- D18 emul + auto-fp8-KV G1 garbage

Closed: retry of the parked D18 emulation + auto-fp8-KV G1 combo as a
  W4A8 speed trick
Evidence: W8A8 DSpark campaign living header. G1 garbage.
Retry if: NEVER. K14 keeps auto/bf16 KV until a W4A8 c1 exists.

---

## D14 -- 2026-08-21 -- GRAPH=1 MTP3 3.8 W4A8 bench_code !!!! false 61.7

Closed: claiming GRAPH=1 + grafted MTP n=3 as a 3.8 W4A8 speed win
Evidence: LOOP 16 / 2026-08-21p. First Paris/391 completions OK. Then
  chat/code collapsed to "!!!!". bench_code c1 avg=61.7 with SpecDecoding
  100% accept / mean length 4.00 / per-position 1.000,1.000,1.000.
  lib.sh 55% single-char gate: LRU n=256 top='!' frac=1.000 GARBAGE.
  After restore GRAPH=1 B70_NOMTP=1: Paris/391/fib coherent again
  (HEALTHY 56s). Logs: l16_w4a8_gptq_mtp3_20260821T043551Z.log,
  l16b_w4a8_gptq_nomtp_restore_20260821T045155Z.log.
Why it is dead: degenerate target body makes draft==target (the
  documented 3.4x MTP false-win). 3.6 W4A8 shelf MTP=3 is a different
  image/artifact (v0.25.1, GDN BF16).
Retry if: coding dump garbage-test OK AND spec accept < 100% on a
  non-repeated body AND bench_code c1 > 25.0. Do not retry as the next
  pick without a hypothesis (graph capture vs spec batch, GDN INT8
  hidden mismatch, v0.26 MTP). Isolated M=4,8 GEMM (K4 kernel half)
  is still open and is not this packet.
Related: K4 e2e MTP / K16 c>1
