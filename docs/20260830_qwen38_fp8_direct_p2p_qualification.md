# Qwen3.8 FP8 direct-P2P qualification

Date: 2026-08-30 UTC

## Scope and identity

CONFIG -> Official Qwen3.8-27B-FP8 weights, W8A16 runtime, FP16 KV,
TP2, the corrected mixed-path GDN kernel, packed serial RMSNorm, deterministic
Inductor, FlashAttention, XPU graph disabled, and one MTP draft token. The
container was
`neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-mixed-gdn-f05c-local`
with image ID
`8e0e3deb0dbddfc7b2ca24cc06f5077a61d3b00c059d3bd0b963685acbd91b81`.
The host ran kernel `7.1.0-070100-generic`; the image carried PyTorch
2.13.0+xpu, vLLM `0.27.2rc1.dev77+gac7509e2b`, corrected `libccl.so.1.0`
SHA256 `733980ab...`, and oneCCL `kernels.spv` SHA256 `0d549c35...`.

CONFIG -> The direct route used `FI_PROVIDER=tcp`, `FI_TCP_IFACE=lo`,
`CCL_ZE_IPC_EXCHANGE=pidfd`, `CCL_SEND=direct`, `CCL_RECV=direct`,
`CCL_TOPO_P2P_ACCESS=1`, and all three oneCCL SYCL simple thresholds at
`4294967296`. Every accepted GPU transaction held the whole-box lease. Every
risky transaction required pre-health, bounded execution, teardown, and
P2P-off post-health. Failed direct transactions were configured to invoke Xe
rebind recovery before post-health.

RESULT -> One F06d preflight mistakenly invoked a leased-only helper outside
the lease. It reached card health and interrupted a P2P-off collective before
any model or P2P1 server started. The process and container were stopped, Xe
rebind recovery restored both cards and the compiled P2P-off collective
without reboot, and that partial result was marked aborted. The accepted F06d
run and every later transaction used the normal whole-box lease.

## Staged boundary results

| Gate | New boundary | Result |
|---|---|---|
| F06a | Exact-image eager plus compiled two-rank oneCCL at 4x5120 BF16 | Pass; direct P2P and post-health passed |
| F06b | vLLM `XpuCommunicator`, registered custom op, immediate consumer, 40 compiled calls from 1 to 2048 rows | Pass; both ranks returned every call with zero error |
| F06c | Full weights, MTP0, 1K envelope, one completion | Pass; `READY`, normal stop, clean teardown |
| F06d | Same model boundary with MTP1 | Pass; target and drafter served and tore down cleanly |
| F06e | MTP1, four slots, 32 concurrent exact-answer/isolation requests | Pass; all 32 requests passed |
| F06f | Two fresh 32K/C4 lifetimes, full serial and concurrent qualification | Pass |

COMMAND -> Run the staged scripts
`run_qwen38_fp8_p2p_f06a_collective.sh` through
`run_qwen38_fp8_p2p_f06e_mtp1_c4_canary.sh`, then run
`qualify_qwen38_fp8_neural_f06f_p2p.sh`. Result roots are under
`/mnt/vm_8tb/b70/results/f06[a-f]_qwen38_fp8_neural_p2p/`; F06f is
`/mnt/vm_8tb/b70/results/f06f_qwen38_fp8_neural_p2p/20260830T130000Z/`.

RESULT -> The old loaded-vLLM queue-handoff failure did not reproduce at any
stage. F06b specifically exercised vLLM's clone, asynchronous direct
all-reduce, `Work.wait()`, custom-op dispatch, compiled caller, immediate
consumer, and matched per-rank entry/return evidence. F06f also crossed the
real 32,768-row profile shape in two fresh server processes.

## Full F06f result

CONFIG -> Two fresh MTP1/P2P1 server processes, 32,768 model and batched-token
limits, four service slots, a shared compiler cache, no prefix cache, and a
32 GiB container RAM/no-extra-swap boundary. Each lifetime ran the 12-prompt
512-token suite, independent canaries, four serial long streams, two C4
batches of four 2K-prompt/512-output streams, and 32 concurrent exact-answer
and isolation requests.

RESULT -> Serial rates were 18.297860 and 18.377703 tok/s, centered at
18.337782 tok/s with 0.436 percent spread. F05d's matched P2P-off center was
17.538941 tok/s. Direct P2P therefore improved this single-stream metric by
4.554670 percent. It did not reproduce the publisher's 51.918757 tok/s
headline, so direct P2P alone does not explain the remaining speed gap.

RESULT -> Direct-P2P C4 aggregate post-first-token rates were 71.223289,
70.946863, 70.647401, and 66.240527 tok/s. Their 69.764520 mean and
70.797132 median improved on F05d P2P-off's 58.500306 mean and 58.628042
median by 19.254966 and 20.756433 percent respectively. P2P is therefore a
larger concurrency/prefill lever than a single-stream decode lever on this
route.

RESULT -> The two direct lifetimes matched one another 12/12 and each matched
both corrected-kernel P2P-off references 12/12. Both independent canary sets
passed. All 16 long C4 streams returned all 512 requested tokens, and all 64
concurrent exact-answer/isolation requests passed. As in F05d, asynchronous
C4 raw token arrays are batch-history dependent and are not a correctness
gate.

RESULT -> Both lifetimes gracefully tore down. All card checks and compiled
P2P-off collective checks passed. No matching Xe fault/reset/wedge event was
found. Peak observed container RAM was 8.591 GiB, minimum host MemAvailable
was 110,729,476 KiB, and maximum global swap use was the pre-existing
340,152 KiB. The regenerated summary SHA256 is
`a838f76e750b822b3f80b306559fca8f80e60f28771d688edfe950edc5f82c69`.

## Operational verdict and next work

VERDICT -> Direct oneCCL P2P is qualified for this exact Qwen3.8 FP8,
corrected-kernel, MTP1, graph-off, TP2 route on the current kernel/runtime.
This does not authorize arbitrary P2P1 use by other vLLM images, models,
communicators, or graph modes. Keep the launcher default at P2P off and
require the explicit wedge-risk guard for P2P on.

VERDICT -> Direct P2P is worth retaining for this route: it preserved the
corrected deterministic target and improved serial decode modestly and C4
aggregate throughput materially. The next speed investigation should compare
the publisher process/profile shape, collective counts, target/drafter
execution, and kernel selection against F06f. The custom push all-reduce is a
separate comparison arm only after its loaded-context oracle returns; its
current first-submission stall remains unresolved. Long growing-agent and
thinking-cap quality qualification also remains separate from this runtime
qualification.
