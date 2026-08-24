# C4 GDN qkvz-only split plan

## Question

The combined target-GDN checkpoint makes `in_proj_qkv`, `in_proj_z`, and
`out_proj` W8A8. The out-only split is already running. The smallest remaining
attribution arm makes only `in_proj_qkv` and `in_proj_z` W8A8 while preserving
target-GDN `out_proj`, `in_proj_b`, and `in_proj_a` as BF16.

This is a checkpoint/config split. It requires no SGLang model or kernel source
change: `qwen3_5.py` already packs the checkpoint qkv and z leaves into the
runtime `in_proj_qkvz` module, and its packed weight loader also handles their
per-channel scales.

## Candidate artifacts

- Directory: `models/files/qwen3.6-27b/w8a8-sqgptq-gdn-qkvz-int8`
- Builder/auditor: `sglang/prepare_c4_gdn_qkvz_candidate.py`
- Config overlay:
  `sglang/configs/qwen36_w8a8_sqgptq_gdn_qkvz_rtn_quantization.json`
- Source of unchanged tensors: `w8a8-sqgptq`
- Source of 96 INT8 weights and 96 BF16 scales:
  `w8a8-sqgptq-gdnint8`
- Exact checkpoint saving: 4,024,958,976 bytes
- Exact TP=2 saving per rank: 2,012,479,488 bytes
- Exact candidate index total: 31,903,750,624 bytes

Materialize only after the current campaign releases the host I/O and GPU
lease:

```bash
python3 sglang/prepare_c4_gdn_qkvz_candidate.py \
  --materialize \
  --overlay-out /tmp/c4-gdn-qkvz-config.json \
  --report-out /tmp/c4-gdn-qkvz-audit.json
```

The builder refuses to overwrite an existing candidate. Its audit compares
every copied qkv/z payload with the combined checkpoint, checks all excluded
GDN leaves are BF16, verifies tensor-name/index contracts, and requires the
unchanged auxiliary artifacts to remain hardlinked to the base.

## Activation-quant result

The candidate only partially avoids the overhead implicated by the combined
GDN arm. `CompressedTensorsW8A8Int8.apply_weights` dynamically quantizes every
M>1 input immediately before its own W8A8 GEMM. Decode uses M=11 because MTP
draft tokens are processed together.

For five profiled M=11 decode steps, each rank should show:

| Route | Base | Combined | Out-only | Qkvz-only expected |
| --- | ---: | ---: | ---: | ---: |
| W8A8 qkvz | 0 | 240 | 0 | 240 |
| BF16 qkvz | 240 | 0 | 240 | 0 |
| W8A8 shape `(11,3072)x(3072,5120)` | 80 | 320 | 320 | 80 |
| BF16 same shape | 245 | 5 | 5 | 245 |
| dynamic quant `(11,5120)` | 400 | 640 | 400 | 640 |
| dynamic quant `(11,3072)` | 80 | 320 | 320 | 80 |
| BF16 GDN b/a | 240 | 240 | 240 | 240 |
| BF16 MTP qkv | 5 | 5 | 5 | 5 |

The combined and out-only columns are measured except the base counts, which
are inferred by subtracting the exact GDN route census. The qkvz-only column is
the fail-closed mechanism expectation to measure. Shape-only traces combine 16
full-attention `o_proj` calls per step with the 48 GDN `out_proj` calls, so the
80/245 counts, not names, distinguish the routes.

Therefore qkvz-only removes all 240 added K=3072 activation-quant calls from
combined, but retains all 240 added K=5120 activation-quant calls. It does not
avoid activation quantization as a class. Together, combined, out-only, and
qkvz-only identify whether the cost is primarily the qkvz side, out side, or
the sum of both.

## Append-only live gates

After the out-only campaign finishes, copy rather than rewrite its scripts:

- `05_c4_gdn_qkvz_int8_mechanism.sh`
- `analyze_c4_gdn_qkvz_int8_mechanism.py`
- `06_c4_gdn_qkvz_int8_abba.sh`
- `analyze_c4_gdn_qkvz_int8_abba.py`

The mechanism gate should be identical to experiment 03 except for candidate
paths, served identity, config overlay, exact checkpoint counts/savings, and
the route census above. Preserve TP=2, PP=1, context 131072, max requests 4,
MTP 10/11, replicated MTP embedding, push-all, C3b off, LM-head INT8 off,
P2P off, graph/radix off, two rank traces, fixed generation, speculative
acceptance, coherence, soak, fatal markers, restart=no, external gpu-run lease,
and endpoint-down cleanup.

The ABBA gate should be experiment 04 with B changed to qkvz-only. Preserve
A-B-B-A, deterministic reporting, phase c1, perf, prefill c1/c4,
24-stream mixed coherence, soak, source/SO hashes, per-arm health/log/inspect,
and endpoint down between arms and after the campaign. Use the same conservative
promotion gate: both phase pairs win, balanced phase at least 3 percent, both
soak pairs nonregress, balanced soak at least 2 percent, TTFT/prefill within 5
percent, coherence, exact restart outputs, and bounded CV/restart spread.

Do not promote qkvz-only merely because it saves more memory. It must beat the
current base and should be compared directly with the completed out-only result
to select the faster coherent split.
