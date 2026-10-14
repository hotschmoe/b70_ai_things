# Clean upstream Intel versus selected source40

CONFIG

Readonly upstream /mnt/vm_8tb/github/strata is HEAD
fb58e0dbc8399662c0e47c76578c6e878b14f6cf; tracked status is clean.
No checkout, download, pack, build, GPU or registry change was performed.
INTEL.md SHA a394b896a24582227dad7fbd971771d6a6467ad3e4c8f89ab3361d6a0c625e8a;
UNSLOTH_Q4.md SHA e334f33312835cb91c7f7ea0ade110afcbf58bb295624a46e758f19e5ea6af3d.
Selected target remains the exact original four UD-Q4_K_XL shards, tokenizer,
template, current source40 native HC/PLE math and no MTP.

COMMAND

Do not run setup.sh or iq_pack.py to manufacture a fast control: upstream's
documented --compat-bf16 rounds195 small Q8 HC projections. iq_pack.py75..80
expects BF16 HC/PLE copies. Clean SYCL fused_gr.dp.cpp supplies no q8_down/up/
inject computation; line1882 explicitly reports Q8 reads unbuilt. Upload option
STRATA_HC_Q8 is present in native_dense.cpp but does not repair that missing
compute path. Current native_hc_policy.hpp and verify.cpp994..1030 use the NEW
STRATA_SYCL_NATIVE_HC=1 source-preserving Q8_0/F32 path, with old HC_Q8/QFUSE
flags0. Unchanged upstream can serve the same filenames with compatibility
rounding; it cannot presently be the requested unchanged-arithmetic control.
Passing the new flag to upstream does not add its absent implementation.

First executable same-artifact control should use current source40, singleGPU0,
ctx2048, fp16 KV, PLE mmap65536, segmented mirror1024/cap65536, expert cacheauto,
device-plan/no-host1, same54/524 token prompts, seed1/temp0, EOS honored and192
max output. All four prior speed requests actually finished length at192; these
are bounded diagnostics, not the separate natural-completion distribution goal.
Change ONLY prefill64 to512 in genuinely new signed configurations.
Use the bounded speed client and original owned lifecycle/fresh full4 gates.
Do not change frozen configs or original C140 profiles in place. A separate
truthful two-alias proposal is pending current171 consumers; no registry write.

The native CLI expression for a separately owned diagnostic input is:

```
STRATA_SYCL_NATIVE_HC=1 STRATA_HC_Q8=0 STRATA_HC_Q8_INJECT=0 STRATA_QFUSE=0 STRATA_GR_V3=0 SYCL_CACHE_PERSISTENT=0 ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 STRATA_STAGE_MIRRORS=1 STRATA_STAGE_MIRROR_SEGMENT_MIB=1024 STRATA_MIRROR_MIB=65536 /build/strata --pack /pack --native /model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --ple-gguf /model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf --expert-profile /src/data/expert-profile.bin --expert-cache auto --stream-experts --prefill 512 --spec 2 --suffix-draft 0 --lookup-chain 0 --max-context 2048 --kv fp16 --ple-io mmap --ple-row-cache 65536 --pcie-frac 1 --vram-reserve-mib 2048 --prompt-cache 0 --adapt-swaps 0 --batch 0 --adapt-every 0 --no-prefill-borrow --tokens-file /inputs/ids-524.txt --max-new 192 --greedy --stop-eos --eos-ids 248046
```

This is the command INSIDE a future root-owned, pinnedC388 container, not a host
launch authorization. It needs exact readonly source/build/pack/model/input
mounts, GPU0 pin/card lease, strict+compiled health, current byte identity,
bounded resources, normal owner/EOF/removal, source/current postfull4. Extract
ids-524 from original speed engine_begin submitted_ids; verify source SHA and
rendered/submitted digest before replay. Prefer the same API client for direct
comparison; standalone startup differs from the warmed API process.

RESULT

Published Intel IQ3_S section730..776 is a different artifact:46GB experts,
~11.5k resident slots, int8 KV/32K, PLE RAM, prefill4096, --spec4 +MTP. Its code
decode40.8 tok/s had90% drafts accepted and3.65 tokens/round; prose31.4 had1.98.
Our actual source40 one-card later native-host decode8.7/8.8, long prompt28.4/33.2
is54/524 prompts,192 outputs, fp16/2048, chunk64, PLE mmap and ALL drafts0/0.
The source40 pair is12.1/12.2 decode and56.4/80.2 prefill, but8192/128/PLE1048576
and different524-prompt outputs; not hardware-only matched evidence.

Actual single startup log396..482:8092 of24576 experts resident,16484 mirrored,
51,660,083,200 mirror bytes,23.62GiB resident cache, ~1900MiB free, own prompt
buffers because too few slots to borrow, captured1/2-token windows, no drafts.
These are startup logical/allocation observations, NOT measured transfer bytes,
page-fault counts or evidence that RAM spill dominates. CPU no-AVX512/threadpool
messages do not imply CPU expert compute while device-plan/no-host are active.

Upstream supports expert Q4_K/Q5_K/Q5_1/Q8_0 (native_mmvq.dp.cpp and CMake161).
SYCL prompt experts use dequant FP16+GEMM (prefill.cpp3966..4094); CUDA MMQ is not
built (INTEL.md674). Upstream mirror code generate.cpp4687..4742 requires every
miss covered for NO_HOST1, otherwise refuses. Its single pinned allocation has
no demonstrated51.66GB qualification; our segmented/stage-owned mirrors are
extra code. Intel wrapper sets device-plan/no-host1 and stager threads12; our
speed cfg leaves stager override absent (prefill.cpp984..992 chooses by source).
Do not attribute a measured gap to these differences without a matched trace.

Both upstream and current support suffix drafts and upstream supports MTP with
a separate draft artifact. Upstream suffix default is3; ours explicitly0.
Our native-HC prefill.cpp1302 rejects MTP prefill as unintegrated. Neither a
silent draft enable nor downloading another model is this no-MTP control.
Upstream UNSLOTH CUDA/Windows same-GGUF table reports7..8.5 decode at40GiB RAM,
not Intel; its IQ3 and dense27B throughput are not selected-artifact baselines.

VERDICT

Run one-variable chunk64/512 first; admit workspace/cache/mirror coverage from
actual startup and reject fallback. Keep PLE RAM, KV quantization, resident
budget, draft length and math flags unchanged. Larger4096 follows only a fresh
memory/budget and context declaration; do not bundle it with the first A/B.
Output IDs/finish/coherence are a gate, not a cosmetic check: larger chunks can
change reduction/window arithmetic and earlier long outputs already differed.

Critical-path next collection: actual per-window routed experts classified by
VRAM/mirror source plus logical selected blob bytes; existing host prep/submit/
wait/sample/stream intervals; retained graph/queue event timings only where
genuinely exposed and clock calibrated. Startup mirror size and DONE offload0
are not dynamic traffic counters. Physical PCIe/device residency/page faults
need actual runtime evidence. Existing H36 proposal identifies precise hooks;
default-OFF/no new waits plus OFF/ON output equivalence remains required. Upload
ue=0 stubs and stage timestamps that read zero are excluded. No causal speed
claim, whole-model fidelity authority or weight/quant substitution is made.

Evidence: upstream generate.cpp SHA ac294c880aaa3641c7df815855b5e4df1123a9cfe6c2c36dedeb998621188141;
verify.cpp fadbf789353c67e9efb18eb7d37940ec12c2a4fd0ea067e290b531f152f6c6b0;
fused_gr.dp.cpp 0b6272f7b9d54dc7889ab162a5d36c4b7fe95b14968d254e6d0567ad540073cf;
iq_pack.py 81b59dd1d71484106cddec9418ef933a2e55a1fd28a57b955715cf0e2e9e4878.
Original speed client report SHA90cadc19a16eb41c5a5f680fe418671a88990cccfbfadf76da3c9edd1c0d87aa
(one),58f6c835fec43493506560d5b291786616da0d7e59b42fc1bb8511d09f2dd397(pair).
