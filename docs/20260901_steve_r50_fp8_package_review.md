# Steve R50 Qwen3.8 FP8 package review

Date: 2026-09-01 UTC

## Source identity

CONFIG -> Review the September 1 public package and reproduction sources at
`b70-optimization-lab` commit
`6adab048f80c4f1161fb812e0387b124a9624494`, without changing the preserved
external worktree or touching either GPU.

COMMAND -> Fetch `origin/main`, inspect
`packages/qwen38-27b-fp8-tp2-b70/{README.md,package.json}`, inspect the R50
builders, launch wrappers, image-contract verifier, strict/depth/concurrency
results, and compare the pinned vLLM prefix-cache behavior with current
upstream vLLM.

RESULT -> The September 1 package is a useful machine-readable front door and
closes the public source gap that existed in the August 31 R31 audit. It pins:

- model `Qwen/Qwen3.8-27B-FP8` revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`;
- official base image digest
  `f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`;
- XPU kernel commit `1e90ffa672ba02f17a909da11838a4c55b199783`;
- the complete kernel R13 -> deterministic MTP0 R15 -> packed RMS R31 ->
  serial FlashAttention R49 -> rebuilt split-GDN R50 build order;
- the strict launcher, model verifier, source-closure verifier, installed-file
  image contract, and strict/depth/concurrency benchmark entrypoints.

The clean source rebuild image measured 51.579521 tok/s and matched all 12
complete arrays against both the qualified MTP1 and matched-image MTP0
references. The package remains a candidate because an independent clean-host
driver and Docker installation replay is still missing.

VERDICT -> Use the September 1 package plus the reproduction guide as the
authoritative transfer source. The old preserved R31 checkout is historical
evidence and is not the final performance image.

## Profile selection

CONFIG -> Compare the published Qwen3.8 choices for a high-quality daily
driver with repeated agent prefixes, c2/c4 work, and a required long-context
decode floor of 30 tok/s.

COMMAND -> Audit the strict R54/R55C result, natural-content R56 depth matrix,
R58 graph screen, R63 concurrency failure, and R77-R82 localization work.

RESULT -> Official FP8 W8A16 plus static MTP1 is the best primary target:

| Profile | Short c1 | Natural-content 32K c1 | Identity boundary |
| --- | ---: | ---: | --- |
| R50 MTP0 | 33.733520 tok/s | 30.330705 tok/s | qualified control |
| R50 MTP1 | 51.808087 tok/s | 50.087665 tok/s | c1 exact vs MTP0 |

The R56 natural-content matrix used unrepeated technical prose, Python, and
structured documents. MTP1 stayed between 49.990 and 53.134 tok/s from 2K to
32K and matched 18/18 complete MTP0 arrays. It is diagnostic rather than a
promoted public curve because the boot contained an earlier GPU reset. No
published result proves the 30 tok/s floor above 32K or at 262K.

R58 enabled XPU Graph on the R50 route and measured 51.229844 tok/s, 1.116
percent below the graph-off incumbent, while retaining exact output. Graph is
not needed to recover Steve's rate. The final route deliberately keeps graph
replay off, PIECEWISE size-one compilation metadata, default FlashAttention,
deterministic Inductor controls, packed RMS serial replay, persistent GDN
scratch, and direct oneCCL P2P.

Q4 and Q8 llama.cpp profiles trade away precision and do not provide this
source-closed XPU serving path. The 101 tok/s INT4 MTP5 entry is experimental
and lacks the required host/context/concurrency qualification. Draft-only INT4
R62 is a later optional optimization, not the first reproduction target.

VERDICT -> Focus on R50 FP8 W8A16 MTP1. Keep same-image MTP0 as the exact
control and operational fallback. Do not promote MTP8, MTP5, Q4, Q8, or
draft-only INT4 ahead of the R50 long-context and concurrency gates.

## Prefix cache boundary

CONFIG -> Require automatic prefix caching for repeated and growing agent
sessions.

COMMAND -> Inspect the exact vLLM base commit
`ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`, its Qwen3Next cache policy, the
R50 launch wrapper, and current upstream vLLM `73723b707fe4`.

RESULT -> The R50 launcher explicitly passes
`--no-enable-prefix-caching`, and every publisher performance result requires
`cached_tokens=0`. Therefore Steve has published no cache-on FP8 performance or
correctness result. This is an intentional benchmark isolation policy, not a
claim that Qwen3.8 cannot cache prefixes.

The pinned vLLM source already supports hybrid-model prefix caching. Enabling
it selects Qwen3Next `mamba_cache_mode=align`, requires chunked prefill, and
uses the normal attention KV cache together with block-aligned GDN/Mamba state
checkpoints. Qwen3Next rejects `mamba_cache_mode=all`. Current upstream retains
that Qwen3Next boundary and has additional hybrid-cache fixes and internal
checkpoint work, but changing the vLLM base would be a separate ABI and
qualification lane.

VERDICT -> First reproduce R50 cache-off exactly. Then change only the prefix
cache setting on the same R50 image and qualify `align` mode with cold, exact
repeat, and growing-session requests. Require nonzero cached-token evidence,
complete output parity versus cache-off, stable MTP acceptance, TTFT benefit,
VRAM capacity, c2/c4 behavior, teardown, and post-health. Cache-on is required
for the final daily-driver promotion but must not be mixed into the initial
R50 reproduction.

## Concurrency boundary

CONFIG -> Require real c2 and c4 scaling without changing greedy output versus
each request's sequential c1 oracle.

COMMAND -> Inspect the R63 FP16-draft control and draft-INT4 candidate, then
the R64-R82 repair and trace chain.

RESULT -> The high aggregate curve is not output-identity-qualified. The
matched FP16-draft control measured 70.020 tok/s at c2, 178.386 at c4, and
1061.646 at c64, but matched only 1/2, 3/4, and 54/64 sequential-oracle arrays.
The draft-INT4 candidate behaved similarly and did not create the defect.
R77 localized the first meaningful c1-versus-c2 difference to
`gdn_attention_core_xpu` in decoder layer 1. R80 and R81 showed that serializing
the convolution and delta stages, separately or together, did not repair it.
R82 is preregistered to trace scheduler request boundaries, packed-token
ownership, state-cache columns, and accepted-prefix selection.

The older 1,091.642 tok/s c64 result passed semantic and isolation canaries but
explicitly allows batch-shape-dependent greedy tokens. It is useful throughput
shape evidence, not exact c2/c4 evidence. Our local F09f 32/32 semantic canary
has the same limitation and must be upgraded to sequential-oracle array
comparison.

VERDICT -> Do not claim that Steve has already solved exact scaled MTP1
concurrency. Reproduce c1 first, then run c2 and c4 against frozen sequential
oracles. If MTP1 differs, retain MTP0 as the concurrency fallback while the
GDN metadata/state mapping is repaired. Do not spend time on c64 before c2 and
c4 are exact.

## Proposed qualification order

1. Build and content-verify the complete R50 chain from the September 1
   package. Reproduce two MTP1 and two MTP0 1K strict attempts with empty
   compile caches. Target at least 46.63 tok/s for MTP1, which is within 10
   percent of 51.808087, while requiring all cross-arm arrays to match.
2. Repeat the R56 natural-content 2K-32K MTP0/MTP1 matrix after clean health.
3. Extend one variable at a time to 64K, 128K, and 262K. Require at least 30
   tok/s decode at every measured depth and record TTFT, acceptance, VRAM,
   host RAM/swap, teardown, and post-health. A 262K maximum allocation is not
   evidence of 262K prompt performance.
4. Enable R50 `align` prefix caching. Qualify exact repeated and growing-agent
   cache hits at representative 2K, 32K, 64K, and near-window depths.
5. Run c2/c4 with unique natural prompts and growing cached sessions. Require
   exact sequential-oracle arrays as well as aggregate throughput and
   per-stream latency. Full 262K per agent is not a realistic c2/c4 capacity
   target on two 32 GiB cards; qualify c1 at 262K and c2/c4 at explicit smaller
   active-context budgets.
6. Only after those gates, add tool parsing, xhigh reasoning defaults, LAN
   authentication, Open WebUI/Pi integration, and shelf promotion.

The currently promoted FULL/Triton service remains stopped. No GPU was touched
during this review.

## Primary sources

- https://github.com/steveseguin/b70-optimization-lab/blob/main/packages/qwen38-27b-fp8-tp2-b70/README.md
- https://github.com/steveseguin/b70-optimization-lab/blob/main/packages/qwen38-27b-fp8-tp2-b70/package.json
- https://github.com/steveseguin/b70-optimization-lab/blob/main/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/README.md
- https://neural.download/models/qwen38-27b-fp8-vllm-tp2-asrock-b70.html
