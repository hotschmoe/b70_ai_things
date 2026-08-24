# C4 GDN out_proj-only INT8 candidate

Status: CPU artifact and GPU mechanism gates PASS. Balanced A-B-B-A is a
performance and repeatability NO-GO. This is default-off and does not change
the shelf.

## Why split out_proj

The corrected combined GDN A-B-B-A artifact
`results/logs/c4_gdn_int8_abba_20260824T145553Z` failed qualification:

- balanced phase decode: -17.107 percent
- balanced c1 decode: -8.528 percent
- balanced c4 aggregate: +1.529 percent
- balanced long soak: +2.888 percent, with both pairs nonregressing
- TTFT and prefill: within 3.111 percent
- correctness, identity, hashes, endpoint-down, and health: PASS

This does not prove projection-level causality, but it makes the smallest useful
isolation clear. Keep the large BF16 `in_proj_qkvz` path that dominated the
phase regression candidate and retain only INT8 `out_proj`, the part most
consistent with the aggregate and soak gains.

## Exact artifact contract

Candidate:
`models/files/qwen3.6-27b/w8a8-sqgptq-gdn-out-proj-int8`

- 48 target GDN `out_proj.weight`: INT8 `[5120,6144]`
- 48 matching `weight_scale`: BF16 `[5120,1]`
- all target `in_proj_qkv`, `in_proj_z`, `in_proj_b`, and `in_proj_a`: BF16
- zero scales for those four BF16 projection leaves
- INT8 out_proj tensors and scales are byte-identical to the combined source
- all auxiliary checkpoint/config/tokenizer files are hardlinks to the base
- only `model.safetensors`, its index, and the note are new files

The source `config.json` remains the unchanged base artifact. SGLang must mount
the generated corrected config overlay read-only. Its ignore list names every
unfused checkpoint leaf of both packed BF16 modules:

```text
lm_head
re:.*linear_attn\.in_proj_qkv$
re:.*linear_attn\.in_proj_z$
re:.*linear_attn\.in_proj_b$
re:.*linear_attn\.in_proj_a$
re:.*visual.*
re:.*mtp.*
```

CPU audit command:

```bash
python3 sglang/prepare_c4_gdn_out_proj_candidate.py \
  --overlay-out /tmp/c4_gdn_out_proj_config.json \
  --report-out /tmp/c4_gdn_out_proj_audit.json
```

Measured storage:

- base indexed tensor bytes: 35,928,709,600
- candidate indexed tensor bytes: 34,419,251,680
- checkpoint saving: 1,509,457,920 bytes, or 1.406 GiB
- exact TP=2 saving: 754,483,200 bytes per rank, or 0.703 GiB

The TP=2 calculation accounts for row-parallel K sharding and the replicated
BF16 per-output scale: per layer it replaces 31,457,280 BF16 bytes per rank
with 15,728,640 INT8 weight bytes plus 10,240 scale bytes.

## Exact five-step M11 route contract

The baseline trace and corrected combined mechanism establish the following
per-rank counts. The out_proj-only mechanism must match all of them:

| Route | Baseline | Out-only expected |
|---|---:|---:|
| BF16 qkvz, M11 K5120 N8192 | 240 | 240 |
| W8A8 qkvz, M11 K5120 N8192 | 0 | 0 |
| BF16 BA, M11 K5120 N48 | 240 | 240 |
| BF16 target out_proj, M11 K3072 N5120 | 240 | 0 |
| BF16 MTP out projection, M11 K3072 N5120 | 5 | 5 |
| W8A8 out shape, M11 K3072 N5120 | 80 | 320 |
| activation quant, M11 K5120 | 400 | 400 |
| activation quant, M11 K3072 | 80 | 320 |
| BF16 MTP qkv, M11 K5120 N7168 | 5 | 5 |

At the profiler-window M1 edge, target qkvz remains BF16. The 45 observed
target out_proj calls move from BF16 to W8A16. Treat those as exact trace-edge
totals, not steady calls per token.

## Next GPU mechanism gate

Create one append-only candidate-only mechanism script by narrowing
`sglang/01_c4_gdn_int8_mechanism.sh`; do not reuse its combined-candidate
analyzer unchanged. The intended command is:

```bash
./bin/gpu-run bash sglang/03_c4_gdn_out_proj_int8_mechanism.sh
```

The mechanism must retain the existing fail-closed lease, pre/post health,
exact image/model/served-id/config-mount checks, artifact hashes, capacity,
fixed/deterministic/mixed/soak coherence, and endpoint-down/no-restore policy.
Its unique proof is the exact route table above on both ranks. Do not start an
A-B-B-A performance run until the mechanism proves qkvz and BA stayed BF16.

Mechanism artifact `c4_gdn_out_proj_int8_mechanism_20260824T164538Z` passed
every check. Both ranks matched the route table exactly, capacity increased
143360 -> 164992, deterministic replay was byte-exact, mixed load passed 24/24,
the 1600-token soak was coherent and stable, all artifacts and identities were
exact, and both cards remained healthy with the endpoint down. Proceed to the
same position-balanced A-B-B-A thresholds used for the combined candidate.

## Balanced serving verdict

Command:

```bash
./bin/gpu-run bash sglang/04_c4_gdn_out_proj_int8_abba.sh
```

Artifact: `c4_gdn_out_proj_int8_abba_20260824T170145Z`.

The fail-closed analyzer rejected the candidate after four fresh serve
lifecycles:

- balanced phase decode: -9.439 percent; both pairs lost
- balanced 6400-token soak: -4.956 percent; both pairs lost
- balanced warm c1: +0.611 percent
- balanced c4 aggregate: +1.100 percent
- TTFT and prefill: within 3.151 percent
- B phase restart spread: 9.620 percent; failed the 5 percent gate
- B soak restart spread: 0.121 percent; passed
- deterministic eight-prompt corpus: byte-identical across B restarts
- fixed output: coherent but not byte-identical across B restarts
- mixed load: 96/96 coherent; all health, identity, hash, and endpoint-down
  checks passed

Verdict: out-only is safe enough for mechanism research but is not a serving
optimization. Keep it default-off. Its 240 added K3072 activation-quant calls
and dispatch boundaries erase the out-projection kernel and capacity benefit.
Next prioritize one shared/reused quantized activation for eligible GDN
projections. Use qkvz-only only if a clean K5120-vs-K3072 attribution is needed.
