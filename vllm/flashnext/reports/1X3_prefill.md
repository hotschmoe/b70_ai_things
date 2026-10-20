# X3: Flash-Next real-shape prefill budget and GGUF serving dispatch

Work in progress, 2026-10-20. No commit, no DD changes. Raw evidence:
`/mnt/vm_8tb/b70/build/vllm-flashnext/x3/`.

## CONFIG

12288-token chunk. Flash-Next ISTA GSQ-RCO IQ3_XXS, 48 layers:
36 GDN, 12 full attention, hidden 2560, 512 routed experts, top-10,
expert intermediate 640, shared expert, HC=4, PLE at GGUF blk.1.
One-card probes use card 0 with ZE_AFFINITY_MASK=0 via xpu-run.sh;
GPU execution leased, timed windows under cpu-run --quiet. Priority
C7b/H1e pair serves go first. Host torch 2.14.0+xpu, existing campaign
container and host UMD. No driver/runtime change.

C13's 3775.74 ms table mixes measured expert time with scaled projection
estimates and a projected full-chunk head. It omits GDN, attention, HC
pointwise work, PLE, indexer selection and communication. It is not a
measured model prefill rate. In a normal serve the logits processor selects
sampling positions before lm_head; a 12288-row head is not a standard
single-request prefill cost.

## COMMAND

- `harness/x3_prefill_ops.py`: event and wall timings of individual real
  shapes, with actual GGUF projection/expert/HC/PLE weights where specified.
- `scripts/x3_wide_build.sh`: additive C13 wide dequant registration, leaving
  the installed decode library intact. Build under cpu-run, MAX_JOBS=4.
- `harness/x3_serve_probe.py`: matched baseline/new prefill on one loaded
  35B proxy server, diagnostic event spans, identity and coherence.
- CPU `tests/x3_hash_check.py`: 100 ragged hash/history trials.

Claude consultation was attempted concurrently through `claude --bare -p`
with requested models fable-5.1 and claude-5.5. Both returned exit 1,
unrecognized_model and Not logged in. No actual responding model ID or
model advice exists for these calls.

## RESULT

CPU audit proves the GGUF serve bypassed every optimized prefill wrapper:
`gguf/ops_interface.py:linear` unconditionally called `gemv`, whose token
axis is decode-sized, and `moe` used one singleton grouped_gemm entry per
expanded token slot. It did not group tokens by expert for GEMM reuse.
This explains why C12's prefill rate stayed roughly constant at
177/175/173 tok/s as prompt length increased. E2's 1961 tok/s used unpack
plus oneDNN linear, with its remaining cost the per-expert launch loop.
The checkpoints and quant formats differ; that is not a matched speedup.

Added opt-in B70_PREFILL=1 routing to existing prefill.py wrappers. Decode
T<=8 is unchanged. Mixed gate/up types are decoded independently. The
35B Q2_K/Q3_K expert types must use dequant, since the fused prefill kernel
has no Q2_K/Q3_K instantiations. Dense and expert dispatch support that.
C13 smoke .so cannot serve on its own; the additive _C.x3wide.so supplies
only wide dequant after loading a full decode library.

The padded scratch cache previously retained a separate buffer for every
observed expert capacity, which varies per layer. Keys now reuse one
buffer per expert/activation geometry. A Triton cast/multiply/top-k sum
candidate avoids materializing two fp32 slot cubes. PLE host hashing has
an opt-in NumPy uint64 implementation; its 100 ragged CPU trials pass,
including EOS reset, overflow, initial history, empty requests and slot
permutation. GPU measurements and serving qualification follow below.

## VERDICT

No end-to-end 2000 tok/s claim yet. The remaining work is measurement and
serving validation of the opt-in dispatch and the complete budget table.
