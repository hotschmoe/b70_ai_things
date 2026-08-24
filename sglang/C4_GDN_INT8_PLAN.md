# C4 target GDN projection INT8 candidate

Status: corrected GPU mechanism gate PASS. The balanced A-B-B-A gate is pending.
This candidate is default-off and does not change the shelf recipe.

## Why this is the next candidate

The existing target checkpoint already contains per-output-channel symmetric
RTN INT8 weights for the three large GDN projections in all 48 linear-attention
layers:

- `in_proj_qkv.weight`: `[10240,5120]` INT8
- `in_proj_z.weight`: `[6144,5120]` INT8
- `out_proj.weight`: `[5120,6144]` INT8
- one BF16 `[N,1]` scale for every weight

`in_proj_b`, `in_proj_a`, `conv1d`, `A_log`, `dt_bias`, and the recurrent core
stay BF16. The MTP checkpoint also stays BF16. This is intentionally smaller
than quantizing every GDN tensor.

No new kernel is required. The current SGLang compressed-tensors W8A8 shim
already selects:

- M=1: `int8_gemm_w8a16`
- M>1: `dynamic_per_token_int8_quant` plus `int8_gemm_w8a8`

The baked Qwen3.5 model loader already binds packed checkpoint loaders for the
merged `in_proj_qkvz` weight and its scale. The only missing serving artifact is
accurate compressed-tensors metadata. The source checkpoint keeps a stale
blanket GDN ignore because it was originally made for zml.

## Exact baseline census

Source: `results/logs/c4_math_census_replicated_20260824.tsv`, five profiled
decode steps on both ranks. Rank device times agree within 0.5 percent.

| Route | Per-rank shape | Calls | Device time | Candidate action |
|---|---:|---:|---:|---|
| target GDN `in_proj_qkvz` | M11 K5120 N8192 | 48/step | 7.146 ms/step | W8A8 |
| target GDN `out_proj` | M11 K3072 N5120 | 48/step | 4.449 ms/step | W8A8 |
| target GDN `in_proj_ba` | M11 K5120 N48 | 48/step | 0.251 ms/step | keep BF16 |
| target GDN `in_proj_qkvz`, M1 edge traffic | M1 K5120 N8192 | 45/trace | 13.651/trace | W8A16 |
| target GDN `out_proj`, M1 edge traffic | M1 K3072 N5120 | 45/trace | 2.846/trace | W8A16 |

The BF16 M11 K3072 N5120 census has 49 calls/step. Forty-eight are target GDN
`out_proj`; the remaining call is the BF16 MTP projection and must remain.
Therefore the target candidate covers about 11.59 ms of per-step projection
device time per rank in the complete M11 target passes before adding activation
quantization. The trace also contains 45 M1 calls of each large GDN projection,
but they cross the profiler-window edges and are not a whole-layer multiple.
Record them as exact trace totals, not as a steady calls-per-token estimate. The
MTP layer is full attention, so these M1 GDN calls belong to the target model
and will use W8A16 in the candidate.

Expected post-route census at M=11:

- BF16 `mm` K5120 N8192: 48 -> 0 calls/step
- BF16 `mm` K3072 N5120: 49 -> 1 calls/step
- W8A8 K5120 N8192: 0 -> 48 calls/step
- W8A8 K3072 N5120: 16 -> 64 calls/step
- activation quant K5120: 80 -> 128 calls/step
- activation quant K3072: 16 -> 64 calls/step
- BF16 `mm` K5120 N48: stays 48 calls/step
- BF16 M1 K5120 N8192 and K3072 N5120 GDN calls: 45 -> 0 per trace
- W8A16 M1 K5120 N8192 and K3072 N5120 GDN calls: 0 -> 45 per trace

## Storage and lifecycle

Run the fail-closed CPU audit and create a full config overlay:

```bash
out=results/logs/c4_gdn_int8_prepare
python3 sglang/prepare_c4_gdn_int8_candidate.py \
  --overlay-out "$out/config.json" \
  --report-out "$out/checkpoint_audit.json"
```

Measured artifact facts:

- 144 target GDN weights and 144 scales pass exact dtype/shape checks.
- Projection storage falls from 11,072,962,560 bytes (10.312 GiB) to
  5,538,545,664 bytes (5.158 GiB).
- Checkpoint saving is 5,534,416,896 bytes, or 5.154 GiB total.
- Exact TP=2 residency saving is 2,766,962,688 bytes (2.577 GiB) per rank.
  This accounts for the replicated row-parallel `out_proj` scale.
- With correct packed-layer ignores, there is no load-time requantization,
  duplicate BF16 backing, target/draft aliasing, or runtime weight mutation.

The generated `config.json` is a copy of the base full model config with only
`quantization_config` replaced. Its ignore list is exactly:

```text
lm_head
re:.*linear_attn\.in_proj_b$
re:.*linear_attn\.in_proj_a$
re:.*visual.*
re:.*mtp.*
```

Compressed-tensors expands the runtime fused name `in_proj_ba` into checkpoint
leaf names `in_proj_b` and `in_proj_a` before evaluating ignores. Both leaves
must match. A fused-name regex matches neither leaf, while matching only one
leaf raises a mixed-scheme error.

## First mechanism result and correction

Artifact `c4_gdn_int8_mechanism_20260824T141330Z` passed every lifecycle,
capacity, acceptance, determinism, mixed-load, coherence, and health check, but
failed exact route scope on both ranks:

- BF16 M11 K5120 N48 BA: 0, expected 240
- quant K5120: 880, expected 640
- W8A8 M11 K5120 N48 BA: 240, expected 0

This was not harmless extra INT8 coverage. The failed ignore made SGLang create
an INT8 packed BA parameter even though the checkpoint has only BF16
`in_proj_b.weight` and `in_proj_a.weight` and no BA scales. Its packed loader
then used `copy_` to cast those BF16 leaves into INT8 storage. A CPU check on
all 96 BA leaf weights found a global BF16 absolute maximum of 0.333984375, so
all 23,592,960 coefficients cast to zero. The checkpoint contains zero BA scale
tensors, leaving the wrongly created scale parameters without checkpoint data.
Artifact hashes before and after the run were equal; the mutation was runtime
representation, not an on-disk rewrite. Coherent text therefore does not make
the N48 route valid.

The smallest fix is metadata only: ignore both checkpoint leaves as shown
above. The mechanism rerun must restore exactly 240 BF16 BA calls and reduce
quant K5120 to exactly 640 on each rank before any performance claim.

## GPU mechanism gate

The append-only mechanism harness requires and verifies one enclosing dual-card
lease. It keeps endpoints down after every outcome and uses the shelf only as
the stable launcher without changing its defaults:

```bash
./bin/gpu-run bash sglang/01_c4_gdn_int8_mechanism.sh
```

The mechanism gate must prove all of the following before an A/B run:

1. Exact served id, TP=2, PP=1, MTP10/draft11, graph off, P2P access off.
2. The generated config is mounted read-only over the candidate config.
3. Fused W8A8 dispatch installs and model loading has no missing/unexpected
   GDN weight or scale, dtype, shard, device, or shape error.
4. A fresh five-step trace satisfies every expected post-route count above on
   both ranks. This is the runtime dispatch proof.
5. KV capacity is at least the 143,360-token BF16-GDN baseline and the static
   audit still reports the 2.577 GiB/rank saving.
6. Two same-process greedy deterministic corpora are byte-identical.
7. A fixed 640-token request is coherent, speculative acceptance does not
   collapse to zero, and a four-stream mixed gate completes 24/24.
8. A graceful stop leaves both cards healthy and both endpoints down.

## Performance and quality gates

If the mechanism passes, run position-balanced A-B-B-A. A is the current
`w8a8-sqgptq` checkpoint and B is `w8a8-sqgptq-gdnint8` with the generated
config overlay. Keep every other setting and served prompt corpus identical.

Minimum performance gate:

- both phase-decode pairs win
- balanced phase-decode improvement is at least 3 percent
- both long-soak pairs do not regress, with a balanced improvement of at least
  2 percent
- prefill and TTFT remain within 5 percent
- candidate restart spread and within-process CV are at most 5 percent
- candidate outputs are byte-identical across B1/B2 and all mixed gates pass

Only after that gate, run HumanEval+ against the current W8A8 shelf baseline.
Promote only if base and plus pass rates do not regress and the full concurrent
sweep remains green. Otherwise keep this as a measured research artifact.

## Main risk and likely outcome

RTN INT8 GDN projections were already coherent and 7 percent faster in the zml
TP=2 experiment. SGLang is more promising for M=11 because it uses genuine
W8A8 XMX rather than materializing BF16 weights. The main risk is the two extra
activation-quant launches per target GDN layer: `in_proj_qkvz` is already a
fast BF16 GEMM, so its net win may be small. `out_proj` has the clearer kernel
case. If the combined candidate misses the serving gate, split the next test to
`out_proj` only; do not optimize or pad the tiny N=48 `in_proj_ba` path.
