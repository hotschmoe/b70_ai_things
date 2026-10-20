# X1 -- Flash-Next MTP on vLLM

Status: implementation and CPU sidecar audit complete; GPU gates in progress.
No promotion, commit, or daily-driver service change.

## Strata findings

The main ISTA IQ3_XXS GGUF does not contain the daily-driver MTP head.
`strata/flash-next/dd/hotschmoe-dd.json:43` selects
`/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt`. This directory contains
116099072-byte dense.bin, dense.txt (29 tensor records), and 707788800-byte
experts.bin. Source pack: `mtp-bf16/mtp-q2_0.gguf` (889014272 tensor bytes).
Its 31 tensors came from the original BF16 HF MTP checkpoint. The experts
are Q2_0. Runtime dense projections are rounded to Q8_0; HC/router stay
BF16 and norms are F32 folded gamma. `src-ep/tools/mtp_rt.py:3` documents
the expert relayout and folded norms. The model-lock json pins the main
ISTA model, not this separate MTP sidecar.

Default solo speculation is four drafts (`--spec 4`, `--mtp-max-t 4`),
with batch MTP depth two for concurrent requests. See
`docs/20261016_night_lab.md:267`. One native MTP decoder layer is reused
across draft steps. Pooled RMS is over all four HC streams, fc_hidden
projects each stream independently, and fc_embedding is broadcast onto
them (`src-ep/src/core/mtp.cpp:906`). Mean-collapse before this projection
is a different computation. The Unsloth eh_proj sidecar is not the
daily-driver draft graph (G3 report).

Acceptance is prompt dependent. Historical engine.log lines 383554 and
383614 report 12526/17457 (71.75%) and 108/159 (67.92%) accepted drafts;
these are examples, not a matched benchmark or an average for the 141
tok/s reference. Strata also uses a restricted draft vocabulary
`src-ep/data/draft_vocab_en.bin`; vLLM currently uses the whole target head.

## Implementation

Fork: `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`, branch b70/flashnext.
Existing dirty changes from other agents are preserved.

- New `vllm/models/qwen4_exp/gguf/mtp_loader.py` isolates draft placement
  from target layers. The target loader dispatches to it for Qwen4ExpMTP
  and Qwen3_5MoeMTP. Shape and payload audits reject incomplete loads.
- Flash-Next selects `B70_MTP_RT=<rt>` plus
  `B70_MTP_GGUF=<mtp-q2_0.gguf>`. Dense weights use the actual rounded rt
  bytes; expert weights use the original GGUF bytes, avoiding the runtime
  code/scale-plane relayout. They remain packed for b70_gguf_xpu.
- `xpu/mtp.py` attaches quant methods to shared vocab and to the two fc
  projections when loading the rt sidecar. It then uses the existing wide
  fc path, bypassing eh_proj. Other quantization paths retain their former
  unquantized vocab construction. No new host reads or synchronization
  are introduced inside forward/capture.
- Proxy nextn block 40 is loaded into the native Qwen3.5 MTP module,
  including Q2_K/Q3_K experts supported by the current GGUF kernels.
  Target blocks never load into draft layer zero.
- `harness/x1_serve_proxy.sh`, `x1_proxy_window.sh`, and
  `x1_compare_greedy.py` provide FULL_DECODE_ONLY on/off qualification,
  exact token-ID comparison, acceptance, C1/C2 timing, health, RAM watchdog,
  and teardown. Raw evidence is in build/vllm-flashnext/x1.

## CPU evidence

CONFIG -> CPU-only torch, Gloo TP=1, meta model constructed from the real
Flash-Next text config. B70_MTP_RT/GGUF as above. No GPU allocation.

COMMAND -> `bin/cpu-run uv ... python harness/x1_check_sidecar.py`.
Also invoked the existing test_mtp_sidecar_every_tensor_once and
test_meta_qwen4exp_constructs directly because pytest is absent.

RESULT -> Existing checks pass (cpu-mapping.log). sidecar-cpu-r2.log:
FLASHNEXT_PAYLOAD_AUDIT_OK and WIDE_FC_REFERENCE_OK [2,4,2560], exit 0,
34 seconds after CPU lease. An initial CPU attempt failed because the
test initialized parallel groups outside set_current_vllm_config; fixed.
sidecar-cpu-r3.log also passes the actual _fc_stage forward against an
independent pooled RMS and Q8 dequant matmul reference, exit 0 in 25 s.

VERDICT -> Correctly sized, complete Flash-Next draft payload placement
is verified on CPU. This does not establish GPU graph or serving behavior.

## GPU evidence

First proxy baseline: card 0 health passed, target GGUF loaded, full graphs
captured, HTTP ready. X1 edited its active shell harness and caused a shell
read-offset failure (line 42 leanup: command not found), followed by clean
server teardown. No greedy or timing requests ran; this is not a baseline.
Preserved as proxy-mtp0-harness-error.log and window-mtp0-harness-error.out.
Frozen-harness baseline retry is in progress.

The harness checks `/v1/models` against diagnostic entries added to
`evals/configs/models.yaml`. The stable first ID is hotschmoe-dd;
secondary IDs encode the model, GGUF method, and MTP depth.

## Review and scope

Both requested Claude CLI consultations (fable-5.1 and claude-5.5)
returned unrecognized_model and Not logged in. No actual responding model
ID or review advice was received. Evidence: x1/fable51.txt, claude55.txt.
No kernels were rebuilt for X1. MAX_JOBS was not used. Existing GGUF
library is pinned to _C.o3.so, sha256 be76661bd120aa2f6c7523ae3c08813ba56d6114660f3d7f31183c32e5e861eb.
