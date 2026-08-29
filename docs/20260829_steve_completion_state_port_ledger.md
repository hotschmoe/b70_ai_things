# Steve completion and state port ledger

Date: 2026-08-29

Status: M01 source audit complete. No source patch is accepted by this ledger.
The August mechanisms remain candidates until their isolated local oracles and
the later BF16, P2P-off model gates pass.

## Scope and identities

This audit compares Steve's qualified 2026-08-28 Qwen3.8 official-FP8 route
with the retained vLLM source. It does not import a binary or use the cleanup
archive.

| Source | Identity | Role |
| --- | --- | --- |
| Steve official-FP8 baseline | vLLM `ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`; XPU kernels `1e90ffa672ba02f17a909da11838a4c55b199783` | Patch base and published FP16-KV, P2P-on control |
| Retained current Steve tree | `/mnt/vm_8tb/b70/steve-s2b/vllm`, detached `44fc8fde09fc311d3099dab10366b672d9142ea4` | Current source comparison target |
| Retained June control | `/mnt/vm_8tb/b70/steve-s2b/vllm-e190`, detached `e190923b32e1b87fe33d08264bff9215fb7770fc` | Qwen3.6 transfer/control source |

The retained current tree and `ac7509e2b` diverge after merge base
`a5bbd81e2e872eba255da9b9f8b86063c0d0cef0`; the August patches must be ported
by behavior, not applied blindly.

The patch text is not retained locally. The immutable public evidence is:

| Patch | SHA256 |
| --- | --- |
| `vllm-qwen38-xpu-deterministic-gdn-ba-state-20260828.patch` | `cda7dd1e42a1e0fed2dd34f3936303cb038852a46d8d00786a1c2ebae326f8eb` |
| `vllm-qwen38-xpu-compiled-gdn-state-ccl-wait-20260828.patch` | `8f8febcd0abc59bc9b69830827cd7607c00870414b17bd02cf32e2d879858ac8` |
| `vllm-qwen38-xpu-gemma-rmsnorm-mtp1-serial-exact-r30-20260828.patch` | `ff5b4f33f5596efbad75112bdbbca2bbf81b6c84688476bfa1c9ec9e546c78c4` |

Public packet:
`https://neural.download/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/`.

## Line-level port ledger

### Explicit collective completion

Steve delta:

- Baseline `ac7509e2b`,
  `vllm/distributed/device_communicators/xpu_communicator.py:48-51`, clones the
  input and calls blocking `dist.all_reduce`.
- Patch `8f8febcd...`, hunk `@@ -47,7 +47,8`, changes the call to
  `async_op=True`, retains the returned `Work`, and calls `Work.wait()` before
  returning the output.

Retained state:

- `44fc8fde0`,
  `vllm/distributed/device_communicators/xpu_communicator.py:60-99`, has three
  compiled routes: static in-place custom op, clone custom op, and blocking
  c10d. None uses the optional asynchronous path.
- The optional `VLLM_XPU_ALLREDUCE_ASYNC_WAIT=1` branch is only at lines
  `101-114`, after the compiled early returns. It therefore covers eager calls
  only.
- Local `bin/xpu-collective-health.py:49-55` uses functional c10d plus
  `wait_tensor`; this is useful precedent but is not the communicator
  `Work.wait()` route.

Disposition: retained-equivalent for eager execution; missing for compiled
execution. First close M02, then implement the M03 blocking-versus-async/wait
micro-oracle. Do not change endpoint code until the P2P-off consumer dependency
is proven.

### Compiler-visible GDN state mutation and cache binding

Steve delta:

- Baseline `ac7509e2b`, `vllm/_xpu_ops.py:116-122,171-201`, takes no state
  tensors in the custom-op schema and retrieves `self.kv_cache[0:2]` inside the
  implementation.
- Baseline registration at `vllm/_xpu_ops.py:1259-1264` declares only
  `core_attn_out` and `z` mutated. Its fake schema at lines `204-210` also omits
  state.
- Patch `cda7dd1e...`, hunks at `_xpu_ops.py:118,179,208,1263` and
  `qwen_gdn_linear_attn.py:977`, adds `conv_state` and `ssm_state` arguments and
  declares both mutated.
- Patch `8f8febcd...`, hunks `@@ -538`, `@@ -557`, and `@@ -1033`, registers
  two non-persistent compiler-visible buffers, aliases them to the allocated
  recurrent cache in `bind_kv_cache`, and passes those stable views to the op.

Retained state:

- `44fc8fde0`, `vllm/_xpu_ops.py:765-771`, still omits state arguments;
  lines `1063-1064` recover state through `self.kv_cache`.
- Registration at `vllm/_xpu_ops.py:2697-2705` still declares only the two
  output tensors mutated. The caller at
  `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2545-2551`
  passes no state tensors.
- The retained cache API now assigns `forward_context[layer_name].kv_cache`
  directly at `vllm/v1/worker/utils.py:514-516`. It no longer invokes the
  baseline layer `bind_kv_cache` hook used by Steve's patch.

Disposition: missing and API-aware port required. Add stable state views at the
current cache-binding boundary or restore a general binding hook deliberately;
then expose those views in the custom-op schema, fake implementation, call, and
mutation list as one coherent change.

### Deterministic 256-row GDN B/A projection

Steve delta:

- Patch `cda7dd1e...`, hunk after
  `qwen_gdn_linear_attn.py:76`, adds a custom op which processes small GDN B/A
  prefills in fixed 256-row blocks and discards zero padding.
- Its forward hunk at baseline lines `966-969` selects the route for at least
  17 tokens and FP16 or BF16 weights. The published motivating shape is
  `M=75,K=5120,N=96`.

Retained state:

- No fixed-256 B/A operator exists in `44fc8fde0`.
- `qwen_gdn_linear_attn.py:2473-2486` has a different diagnostic: it serializes
  four one-row B/A projections only when `VLLM_XPU_GDN_BA_SERIAL_M1=1`.

Disposition: missing. Prove the fixed-256 operator independently for FP16 and
BF16, real prefill row counts, both ranks, and repeat exactness before model
integration. Do not treat the four-row diagnostic as equivalent evidence.

### Exact two-row Gemma RMSNorm replay

Steve delta:

- Baseline `ac7509e2b`, `vllm/model_executor/layers/layernorm.py:142-167`, sends
  the complete input to one RMSNorm or fused-add RMSNorm call.
- Patch `ff5b4f33...`, hunks `@@ -147`, `@@ -154`, and `@@ -166`, gates on
  `VLLM_XPU_QWEN_GEMMA_RMSNORM_PACKED_SERIAL_EXACT=1` and exactly two rows,
  replays each row through the original one-row operator, and explicitly routes
  XPU dispatch through that implementation.
- The published operator proof ran 100 trials across plain, fused-add, and
  residual outputs with zero mismatches. The route still required two fresh
  servers plus deterministic Inductor for model qualification.

Retained state:

- `44fc8fde0`, `vllm/model_executor/layers/layernorm.py:206-237`, contains a
  different `VLLM_XPU_QWEN_GEMMA_RMSNORM_SERIAL_M4` path for exactly four rows.
- There is no two-row environment control or exact publisher-MTP1 replay path.

Disposition: missing. Port the two-row shape as a separate bounded mechanism;
do not generalize the four-row Qwen3.6 research path or claim that it covers
publisher MTP1.

### Deterministic Inductor

Steve evidence:

- The qualified profile requires `TORCHINDUCTOR_DETERMINISTIC=1`.
- The patched but non-deterministic r31 attempt matched only 9 of 12 complete
  target arrays. Two fresh deterministic r32 servers matched 12 of 12 against
  each other and both qualified MTP0 references.

Retained state:

- No tracked local vLLM launcher sets `TORCHINDUCTOR_DETERMINISTIC`.
- This is a launch and evidence requirement, not a substitute for the state,
  collective, B/A, or RMSNorm source repairs.

Disposition: missing from the candidate launch contract. Add it only to the
explicit candidate arm and keep a matched off control; require fresh compile
caches so a prior artifact cannot satisfy the gate accidentally.

## M01 pass criterion and verdict

Roadmap M01 passes when the official async completion, GDN state binding and
mutation, cache lifecycle, deterministic B/A, and RMSNorm replay are mapped to
the retained APIs at line level, with every item labeled equivalent, missing,
or requiring an API-aware port. This document satisfies that source-audit
criterion. It does not qualify or accept a port.

## Recommended port order

1. Close M02 on P2P-off direct, compiled, and replayed all-reduce/all-gather
   shapes with matched rank entry/return and repeated teardown.
2. Close M03 as a micro-oracle comparing source-default completion with
   `async_op=True` plus `Work.wait()`, including an immediate tensor consumer.
3. Port compiler-visible GDN state arguments and current-API cache binding;
   prove state aliasing and mutation before a model serve.
4. Port and qualify deterministic 256-row B/A projection independently.
5. Port and qualify exact two-row RMSNorm independently, including fused-add
   and residual results.
6. Combine the qualified mechanisms with deterministic Inductor in a fresh
   compile cache, then run target-only and MTP1 exactness before performance.

## M02 closure update

M02 passed after this audit. The exact configuration, initial functional
all-gather graph failure, opaque-direct repair, three fresh lifetime results,
and health evidence are in
`docs/20260829_m02_p2p_off_compiled_collective_oracle.md`. The subsequent M03
closure is recorded below; no Steve model patch has been accepted yet.

## M03 closure update

M03 passed after this audit. Blocking c10d and `async_op=True` plus
`Work.wait()` both produced exact results for immediate dependent consumers
at BF16 `[1,5120]` and `[4,5120]` with P2P disabled. Three fresh lifetimes,
matched rank event streams, teardown, and all health gates passed. The
explicit route had higher exploratory host-boundary medians, so the result is
a correctness qualification rather than a speed claim. Exact evidence is in
`docs/20260829_m03_explicit_completion_ab.md`. M04 is next; no Steve model
patch has been accepted.
