# Dense MMVQ source attribution, 2026-10-16

S121 ranks retained operation records and identifies one input-only integer preparation hypothesis. It supplies no new kernel, compiled optimization proof or latency result.

CONFIG -> Historical S95 small859-row operation CSV, primary device0, operation name contains MMVQ, excluding WideQ5K grid62080 vocabulary head. Quantized dense nonvocabulary MMVQ is separate from BF16 MMVF/GEMM, HC and routed GU/down. Current source and retained tensor-header descriptors supply role sets. No model payload, large trace, ELF/archive read, new build/decoder/GPU or production edit.

COMMAND -> derive.py joins that retained census to exact source dispatch/callers and S41/S57/S63 descriptor summaries. Exact six frozen payloads preserve ranking strings/counts, source excerpts, pins and one audit plan. Native source SHAae47a632d0cb3a0fc5a992c335bdd1145d4bbb9210c5b481dd45e53cc3835252; source/role joins are not per-event tensor/layer/phase joins.

RESULT -> Quantized dense nonvocabulary ranking totals102 operation rows/9600 calls/361.772043ms across the WHOLE instrumented collection, including retained capture/cold/other phases. It is NOT a per-cycle wall budget: do not divide by a guessed cycle count, add it to CPU/peer time, or treat it as removable latency. Leading family collection totals are Q6_K111.049ms, Q4_K64.939ms, IQ4_XS60.430ms, IQ3_S56.396ms and Q5_K28.147ms.

| Largest C6 operation row | Calls | Collection service ms | Retained source/descriptor role set |
| --- | ---: | ---: | --- |
| Q6_K grid640 | 696 | 43.823245 | K6144/M2560;10 attention-output +19 SSM-output tensors |
| Q6_K grid2560 | 336 | 28.765068 | K2560/M10240;14 attention-QKV tensors |
| IQ3_S grid1536 | 240 | 21.133954 | K2560/M6144;10 attention-gate tensors |
| Q4_K grid2560 | 192 | 20.115752 | K2560/M10240;8 attention-QKV tensors |
| IQ4_XS grid2560 | 264 | 17.762699 | K2560/M10240;11 attention-QKV tensors |

C6 mapping supplies M=4*grid here; retained headers supply K and possible roles. Grid alone does not identify K or an individual tensor. The Q6 M2560 descriptor set resolves to29 K6144 matrices, not a shared-down K640 ambiguity. These are source-conditioned sets, not observed routing attribution.

The two leading Q6 shapes already received S104/S107 A4 tests; the bounded serving observation was+0.427673%, defaultOFF. Do not repeat it as a new lead. Q6 source already prepares weight decode outside the active-column loop and shares input preparation across RPW rows; emitted CSE/load behavior is not established by that source. IQ4_XS RPW2 was tiny/inconclusive, Q4/Q5 C6 RPW2 has reported spill/scratch liabilities, IQ3_S SWAR was mixed and the S75 SG16/DPAS mapping was7.7-7.9x slower. None is selected again.

The distinct source lead is Q4/Q5's affine-minimum correction. WideQ4K::apply1954..1960 calculates TWO raw signed16-byte sums of the canonical stored Q8 bytes using eight logical DP4A calls per lane/column/kbx, independent of weight row. WideQ5K forwards to that helper. A narrow quantizer adjunct might prepare those exact integers once after rounding/clamping/encoding, while leaving ALL36-byte Q8_1 blocks and the original FP products/subtract/add/XOR order intact. Signed-int8 sum bounds[-2048,2032] fit int16. Existing ds[1] is the FP16 FULL32 raw-activation sum and cannot substitute for either encoded-q half sum.

Logical adjunct size is4*(N/32)*C:1920B for N2560/C6 or640B/C2. These are storage sizes, not measured traffic or savings. Producer work, extra loads, owner scratch/stride/lifetime/capture/readiness costs may outweigh reuse. RPW1 rows execute in separate subgroups/workgroups, so compiler CSE cannot share sums across them; columns use distinct inputs. Actual lowering or local strength reduction remains UNKNOWN. Check that before building a candidate.

VERDICT -> ONE source-selected existing-image audit: Q5_K full-head C2/RPW1/N2560/M248320. Its EXCLUDED S95 head row is42 calls/50.470608ms, median1201.562us; this context does not inflate the nonvocabulary ranking. Bind the actual host branch/Info/cache/guard/name/image first, then inspect emitted input-only half sums and exact FP/data dependencies. Ambiguous binding or already-folded/cheap correction closes the lead. Actual source reviews by claude-opus-5-5 (46.035s) and claude-fable-5-1 (66.380s), both exit0/root full-read, support this one-image CPU audit. S122 preparation is separate; no compiled audit result is claimed here. No kernel, build, GPU, format sweep, speed forecast or serving approval follows from S121.

[Exact six source payloads and absolute-path/SHA copy index](../strata/flash-next/lowlevel_probe/dense_mmvq_source_s121/source-copy-index.json), [findings](../strata/flash-next/lowlevel_probe/dense_mmvq_source_s121/findings.md), [one bounded audit plan](../strata/flash-next/lowlevel_probe/dense_mmvq_source_s121/audit-plan.md). Source manifest bad4f4f97ff1a301b872bdc3372f7849b34b147a053de4c24f737d28edd4c686. The existing curated CSV is referenced without duplication.

Root accepts the bounded audit and rejects broader inferences: exact integer algebra alone does not prove compiled output identity; encoded weight bytes divided by recorded service do not measure bandwidth; static instruction proportions do not establish dynamic bottlenecks. The reviewers disagree about bandwidth dominance, so it remains unresolved. [Actual answers and root decision](../strata/flash-next/lowlevel_probe/dense_mmvq_source_s121/reviews/root-decision.json). Frozen source payloads retain their original wording; these corrections govern the next step.
