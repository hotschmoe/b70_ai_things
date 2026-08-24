# Qwen3.8 UD-Q4_K_XL matched profiling campaign

Status: matched GPU campaign passed without HumanEval+ on 2026-08-24. XL MTP3
was byte-exact on the deterministic corpus and materially faster; full
HumanEval+ remains the production-promotion gate.

## Question

Replace the stock ggml-org Q4_K_M serving artifact with Unsloth's
UD-Q4_K_XL artifact without confusing a model-quality change with a kernel,
topology, or benchmark-method change. Determine which existing B70 custom
operations still apply, measure the changed weight-streaming path, test the
embedded NEXTN layer, and keep an exact Q4_K_M rollback.

## Pinned artifacts

| Arm | File | Bytes | SHA256 |
|---|---|---:|---|
| Q4_K_M | `Qwen3.8-27B-Q4_K_M.gguf` | 18973870432 | `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| UD-Q4_K_XL | `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17559178144 | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` |

The campaign checks the host size and SHA, the model file and SHA passed into
the live container, TP=2, native 262144 context, MTP state, lab-door state,
`CCL_TOPO_P2P_ACCESS=0`, and the live served id. The stable public alias alone
is not accepted as model identity.

## Quant-mix finding

The name UD-Q4_K_XL does not mean a uniform Q4_K artifact. The GPU-free GGUF
header audit found:

| Type | Q4_K_M tensors | UD-Q4_K_XL tensors |
|---|---:|---:|
| F32 | 353 | 360 |
| IQ3_S | 0 | 1 |
| IQ4_NL | 0 | 6 |
| IQ4_XS | 0 | 70 |
| Q3_K | 0 | 3 |
| Q4_K | 193 | 69 |
| Q5_K | 0 | 191 |
| Q6_K | 17 | 56 |
| Q8_0 | 288 | 110 |
| Total | 851 | 866 |

Stored tensor data averages 5.640 bits/element for Q4_K_M and 5.138 for XL.
The XL file also contains block 64 and four explicit `blk.64.nextn.*` tensors,
so its embedded NEXTN path can be tested without a sidecar draft model.

The B70 Q4_K reordered gate/up/SwiGLU operation requires both MLP matrices to
be Q4_K. It covers all 64 Q4_K_M target blocks but only XL blocks 21, 35, and
47: 3 complete pairs out of 65 target-plus-NEXTN blocks. Therefore:

- TP collectives, fused all-reduce consumers, Q8 activation handoffs, GDN
  fusions, and other activation/communication work remain structurally valid.
- The Q4_K-only reordered MMVQ/SwiGLU operation is mostly inapplicable to XL.
- Production `LAB_DOORS=0` already disables that experimental Q4_K-only path,
  so the artifact replacement does not invalidate the currently served path.
- XL must be treated as a different weight-dequant/MMVQ workload. Q5_K,
  IQ4_XS, Q6_K, and the remaining types need profiling and eventually broader
  fused-kernel coverage if they dominate decode.

The complete machine-readable mix, including bytes by type and category, is
written to `gguf_quant_mix.json` by the campaign.

## Matched measurements

`llamacpp/profile_qwen38_api.py` uses the same fixed prompts and order for every
arm. It records client post-first decode throughput, TTFT, prefill proxy, server
token counts, finish reason, output hashes, and all individual samples. Timed
arms keep verbose and per-token census logging off.

The three timed regimes are:

1. Native decode: one fixed short prompt, forced 256-token decode, five runs.
2. Coding decode: three fixed Python workloads cycled over five runs, forced
   256-token decode.
3. Cold prefill: five fixed but distinct approximately 2048-token prompts with
   eight output tokens.

The existing usage-based `bench_code.py` c1 format is also captured so the new
run remains comparable with historical B70 campaign logs.

Deterministic checks use the existing seven-case qualification tool. A minimum
of six correct and seven nonempty outputs is only a coherence floor. It is not
a quality-promotion result. When full HumanEval+ is requested, XL must remain
within one base problem and two plus problems of the pinned Q4_K_M result:
base at least 0.963 and plus at least 0.915. The reference is the exact
Q4_K_M 0.970/0.927 run already on disk.

The optional MTP arm uses embedded `draft-mtp`, draft max 3, F16 draft KV, and
the same target artifact. Greedy deterministic responses must be byte-exact to
XL MTP-off. Performance is reported separately and does not override an output
mismatch.

## Kernel evidence

The optional evidence arm is excluded from timing. It enables llama.cpp verbose
logging, meta all-reduce census, and SYCL fusion census with the production Q4
lab doors still off. The graceful stop captures:

- GGUF tensor type counts;
- resolved SYCL door settings;
- meta all-reduce and fused-consumer counts;
- available `[FUSE-EXT]` and `[Q8-DEDUP]` exit counters;
- native llama.cpp prompt-eval and eval timing lines.

The current patched binary has no exact per-quant MMVQ dispatch counter. GGUF
mix plus the available fusion counters are therefore the current evidence
ceiling. If XL weight-streaming optimization becomes the next lever, add
per-type MMVQ/MMQ dispatch and device-time counters before changing kernels.

## Run order

GPU-free identity and quant audit:

```bash
bash llamacpp/campaign_qwen38_ud_q4k_xl.sh metadata
```

Complete matched campaign, including MTP and verbose evidence but excluding the
long code-quality run:

```bash
./bin/gpu-run env RUN_MTP=1 RUN_EVIDENCE=1 RUN_HEPLUS=0 \
  bash llamacpp/campaign_qwen38_ud_q4k_xl.sh full
```

Full promotion-quality campaign:

```bash
./bin/gpu-run env RUN_MTP=1 RUN_EVIDENCE=1 RUN_HEPLUS=1 \
  bash llamacpp/campaign_qwen38_ud_q4k_xl.sh full
```

All real GPU work is inside one two-card lease. The endpoint is intentionally
left down between arms, so no intermediate production reload contaminates the
campaign or adds avoidable model loads. On successful completion it restores
XL MTP-off exactly once. If the campaign exits early, it stops the active test
container and leaves the endpoint down rather than hiding the failure with an
automatic restore.

Explicit recovery commands, both under the lease:

```bash
./bin/gpu-run bash llamacpp/campaign_qwen38_ud_q4k_xl.sh restore-xl
./bin/gpu-run bash llamacpp/campaign_qwen38_ud_q4k_xl.sh restore-q4km
```

`FINAL_RESTORE=q4km` makes a completed campaign roll back to the exact stock
artifact; `FINAL_RESTORE=none` leaves the endpoint down. The campaign does not
delete either artifact or rewrite the old serving engine.

Results live under
`results/logs/qwen38_ud_q4k_xl_campaign_<UTC>/`. The authoritative summary is
`analysis.json`; raw identities, samples, responses, container inspection,
server logs, health checks, and quality output remain beside it.

## Measured result - 2026-08-24

The non-HumanEval campaign at
`results/logs/qwen38_ud_q4k_xl_campaign_20260824T060230Z/` passed every enabled
hard gate. Median post-first-token throughput was:

| Arm | Native decode | Coding decode | Prefill TTFT |
|---|---:|---:|---:|
| Q4_K_M, MTP off | 35.57 tok/s | 35.67 tok/s | 4.814 s |
| XL, MTP off | 34.44 tok/s | 34.04 tok/s | 5.553 s |
| XL, embedded MTP3 | 43.19 tok/s | 53.87 tok/s | 5.799 s |

XL MTP-off changed native/coding decode by -3.18%/-4.56% versus Q4_K_M.
Embedded MTP3 improved XL native/coding decode by +25.41%/+58.26% and was
byte-exact to XL MTP-off on all seven deterministic responses. Q4_K_M and XL
also shared the same six-of-seven canary result and exact response hashes, so
the modular miss is not evidence of an XL regression. Full HumanEval+ with MTP
enabled is still required before production promotion.
