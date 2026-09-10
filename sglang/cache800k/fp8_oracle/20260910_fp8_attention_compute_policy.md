# FP8 cache versus FP8 attention arithmetic

CONFIG -> Same exact current-main SGLang source and bdc51 image as the failed scalar-corrected oracle. Synthetic FP16 Q/K/V, FP8 cache, grouped24/4 heads, dimension256, sequence67, query1. Strict byte gates passed; first actual attention case failed48/6144 elements, maxabs .03187394142150879 at rtol.03/atol.015.

COMMAND -> CPU-only cpu_policy.py reproduces the source arithmetic: grouped decode splits64+3,32-token online-softmax tiles, raw FP8 K/V with scales, Q and unnormalized P converted to FP8, float32 accumulation and FP16 final output. CPU container has no devices/network. See cpu-policy.json. Patch source syntax parsed and git apply --check succeeded; no image build or GPU run.

RESULT -> Stock Q_FP8/P_FP8 CPU model reproduces the exact observed48 failed elements and maxabs .03187394142150879. It is not a Triton execution, but agreement identifies a concrete arithmetic policy explaining the observed error. Keeping Q_FP8 and P_FP32 still fails21 elements. Keeping Q/P_FP16 reduces maxabs to .00028014183044433594 and fails0 elements. Kx2 and Vx2 Q/P-half variants likewise fail0, at maxabs .00092149/.00056028. No thresholds were relaxed.

Source: decode_attention.py:623-633 converts Q to cache dtype;660-664 performs QK;731-735 converts P to V dtype. Cache FP8 therefore implicitly requests extra Q/P quantization. Standard extend does the same on cached prefix (extend_attention.py:566-569 and659-663); fresh suffix is FP16. AMD gfx1250 already has a different FP32 probability workaround, but this candidate does not reuse that compute policy.

Candidate xpu-fp8-cache-half-compute.patch adds a separate constexpr selected only for XPU, FP16 Q and e4m3fn K/V cache; grouped decode also requires non-MLA. It converts loaded K/V to Q's FP16 dtype and uses FP16 P for dots, preserving FP8 cache storage and original FP32 scale factors. Non-XPU dispatch and existing gfx1250 behavior remain unchanged. Candidate source and base/candidate SHA256 identities are adjacent.

Scope limits: grouped non-MLA decode and standard extend only. Lean decode (HIP-gated in this source), normal/non-grouped decode, unified/split extend, dense/specialized paths, BF16 Q, other FP8 formats, graphs, prefix-wrapper integration and MTP integration are unqualified. Direct-oracle success will not establish general SGLang feature parity. Actual installed-source checks must point at the candidate source when a candidate image is tested.

The oracle now collects every numerical close/sensitivity result for all nine cases and exits nonzero after printing JSON if any fail. Store exact gates remain fail-fast; kernel exceptions still fail execution. First run this instrumented oracle on the unchanged image to preserve full baseline evidence, then compare the source-patched candidate under the same lease/health/configuration.

VERDICT -> Explained lossy compute policy, separate from earlier device scalar rounding and unrelated to proving historical Pi bang causality. Candidate is a precision-policy change requiring an actual leased XPU qualification. No assertion tolerance changes, GPU launch or image build performed here.

CPU all-nine extension: cpu-all-nine-policy.json models decode q1, extend q1 and extend q4 with correct/Kx2/Vx2 scales. Stock predicted failing-element counts are48/1006/499,45/1039/488,137/3165/1778 respectively. All nine Q/P-half variants have zero failures, worst maxabs .0009214878. These remain CPU arithmetic predictions until matched XPU execution. The tracked cpu_compute_policy.py checks all nine half-policy gates and pins the observed stock first-case failure.
