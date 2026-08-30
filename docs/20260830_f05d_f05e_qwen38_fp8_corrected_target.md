# Qwen3.8 FP8 corrected-kernel target and concurrency qualification

Date: 2026-08-30 UTC

## F05d: corrected kernel passes C4 but selects a new C1 target

CONFIG -> Official Qwen3.8-27B-FP8, W8A16, FP16 KV, TP2, P2P off,
MTP1, graph off, deterministic Inductor, 32,768 model/batch limits, four
service slots, packed serial RMSNorm, persistent GDN scratch, and the pinned
1e90 mixed-path kernel. Image ID was
`8e0e3deb0dbddfc7b2ca24cc06f5077a61d3b00c059d3bd0b963685acbd91b81`.
Result root was
`/mnt/vm_8tb/b70/results/f05d_qwen38_fp8_neural/20260830T075200Z/`.

COMMAND -> Under `bin/gpu-run`, compile two independent empty caches. In each
fresh lifetime, run the 12-prompt deterministic C1 suite, independent
canaries, two synchronized C4 batches with four 2K-prompt/512-output streams
each, and eight rounds of four concurrent exact-answer/isolation canaries.
Then tear down and run per-card plus compiled P2P-off collective health.

RESULT -> Both lifetimes matched one another 12/12. Their class-balanced C1
rates were 17.574570 and 17.503311 tok/s, for a 17.538941 tok/s median and
0.406 percent spread. Target/draft AOT keys were `80de0121...` and
`be175b50...`, with zero `.best_config` files.

RESULT -> All 16 long concurrent streams returned all 512 requested tokens.
The four C4 batch aggregate post-first-token rates were 66.376149, 50.879935,
67.294763, and 49.450377 tok/s. All 64 exact-answer/isolation requests passed.
No prompt used cached tokens. Async C4 output bytes remain batch-history
dependent and are not used as a correctness gate.

RESULT -> Both corrected-kernel attempts matched the old-kernel F05a target
only 10/12. The same two prompts changed stably: `customer-email` first
changed at zero-based token 124 and `technical-guide` at token 160. The
generic analyzer therefore correctly rejected old-reference promotion even
though restart exactness and concurrent semantics passed.

RESULT -> All teardowns and card/collective checks passed. Peak container RAM
was 9.088 GiB, minimum host MemAvailable was 111,507,896 KiB, and global swap
stayed at its preexisting 28,652 KiB baseline. Summary SHA256 was
`373f32462f63db25a540c60e1e54afface84cece11585180358cfb0c4ef10f76`.

VERDICT -> F05d qualifies complete and semantically isolated C4 serving for
the corrected kernel, but it does not preserve the old-kernel deterministic
target. Run the MTP0 F05e control against the corrected MTP1 references to
separate a kernel target change from a speculative-decoding target change.

## F05e: MTP0 causal control

CONFIG -> Keep the exact F05d image, corrected kernel, P2P-off topology,
graph-off deterministic compiler controls, FP16 KV, and 32K capacity. Change
only speculative tokens from one to zero and reduce service slots to one.
Use a shared compiler cache across two fresh server processes and require
exactness to both corrected-kernel F05d MTP1 references. Result root is
`/mnt/vm_8tb/b70/results/f05e_qwen38_fp8_neural/20260830T083000Z/`.

COMMAND -> Run `vllm/fp8/qualify_qwen38_fp8_neural_f05e.sh` under the
whole-box lease. Execute the 12-prompt suite and canaries in two fresh server
lifetimes, compare all raw token arrays to one another and both F05d
references, then require clean teardown, card health, and compiled P2P-off
collective health.

RESULT -> Both MTP0 attempts matched one another and both corrected-kernel
F05d MTP1 references 12/12. They each matched the old-kernel F05a target
10/12, with the same prompt and first-token differences as F05d. Their
class-balanced rates were 11.327250 and 11.742236 tok/s, for an 11.534743
tok/s diagnostic median. Attempt 1 compiled target AOT key `560096c7...` in
97.66 seconds; attempt 2 directly loaded the same AOT artifact. The shared
cache held 1,747 files and zero `.best_config` files.

RESULT -> Both canary sets, teardowns, card checks, and compiled P2P-off
collective checks passed. Peak observed container RAM was 7.755 GiB, minimum
host MemAvailable was 112,371,368 KiB, and global swap stayed fixed at the
preexisting 28,652 KiB baseline across all 291 samples. Summary SHA256 was
`a7385835dad957e386203465add4550ea4f5d57cd5f7af4972600bf1e62c9fe7`.

VERDICT -> F05e passes. Disabling MTP does not restore the old target: the
stable two-prompt target shift belongs to the corrected GDN kernel rather
than speculative acceptance. F05d plus F05e establish a corrected-kernel
deterministic target and qualify its C4 completion and semantic-isolation
behavior under P2P off. Long growing-agent qualification and shelf promotion
remain separate gates.

## Interpretation

The local safe route does not consist entirely of sub-15 tok/s runs. MTP0 is
the target-only control and measures about 11.3 to 11.5 tok/s. Qualified MTP1
measures about 17.54 to 17.65 tok/s with P2P off. The publisher's roughly
51.9 tok/s strict profile additionally uses direct oneCCL P2P, which remains
blocked for a full local vLLM TP2 serve by the documented queue-handoff
failure.

The corrected 1e90 kernel is required for concurrent MTP engine survival, but
it also selects a stable new deterministic target for two of 12 prompts. F05e
proves that target belongs to the corrected kernel regardless of MTP.
Neither the bounded 512-token performance suite nor the synthetic 4K output
can answer reasoning-quality questions. After this runtime identity gate, use
the growing-agent policy ladder: true thinking off at 8,192 max output first,
then native thinking at 16,384 max output with an explicit private-thinking
cap, increasing that cap only in a matched quality experiment.
