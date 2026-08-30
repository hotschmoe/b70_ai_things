# F04 Qwen3.8 official-FP8 MTP1 target negative

Date: 2026-08-30

Status: failed the frozen-target gate. Two MTP1 processes sharing one compiled
cache were exact with one another, but only 5/12 natural-prompt token arrays
matched the frozen F03a MTP0 target.

## CONFIG

- Git harness identity: `dfe7ffd`.
- Model: official `Qwen/Qwen3.8-27B-FP8` revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Runtime: vLLM `0.27.2rc1.dev77+gac7509e2b`, PyTorch `2.13.0+xpu`, and
  local image
  `sha256:338ef5e2c956471a195a0644e2acb38288ac6837ebf5282671794be5329a7e2b`.
- W8A16 FP8 runtime, TP2, P2P off, MTP1, XPU Graph off, deterministic
  Inductor, FP16 target, automatic FP16 KV, one request, 1,024 context,
  prefix caching off, packed serial RMSNorm, and persistent GDN scratch.
- Lifetime 1 created one cache and lifetime 2 reused it. Both containers had a
  32 GiB memory limit and no additional swap allowance.
- Frozen reference: both mutually exact F03a MTP0 attempts under the same
  P2P-off target policy.
- Result directory:
  `/mnt/vm_8tb/b70/results/f04_qwen38_fp8_neural/20260830T012500Z/`.

## COMMAND

```text
STAMP=20260830T012500Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f04.sh
```

The whole-box lease covered model and runtime verification, pre-health, both
server lifetimes, raw-token workloads, independent canaries, graceful
teardown, inter-lifetime health, and final health.

## RESULT

The two MTP1 processes matched all 12/12 complete token arrays. Lifetime 1
reported 141.48 seconds for target compilation. Lifetime 2 reconstructed 21
standalone target artifacts and 65 submodules per rank, directly loaded target
AOT key `ed4b9708...`, and reported 1.92 seconds. Its draft-head graph still
needed 15.11 seconds to compile under a distinct key.

The required MTP0 comparison failed identically in both attempts: 5/12 arrays
were exact. The seven mismatches began at zero-based tokens 392, 303, 124, 7,
511, 169, and 77 for `incident-retrospective`, `code-review`,
`customer-email`, `architecture-tradeoff`, `technical-guide`,
`performance-hypotheses`, and `decision-memo`, respectively. This is a target
failure, so no speed attribution or promotion is authorized.

The diagnostic rates were 18.076070 and 18.410930 tok/s, with an 18.243500
tok/s median and 1.836 percent spread. That is 56.029 percent above the F03a
diagnostic median, but it is not an MTP speed claim because the required target
arrays differ. Reported MTP acceptance commonly ranged from about 64 to 94
percent over ten-second windows.

Both canary files passed and had SHA256
`f234e605954b061e7f902eb92dd96739722df5437cadd9b2aceed79b976e45f8`.
All card and compiled P2P-off collective checks passed before, between, and
after serving, and teardown was clean. Model loading reported 14.07 GiB per
rank; the runtime accounting later reported about 14.59 GiB consumed and 8.3
GiB KV cache per card. Container host-RAM use was 8.325 to 8.514 GiB. Swap
remained zero, minimum host MemAvailable was 112,478,424 KiB or 107.268 GiB,
and memory PSI `some`/`full` totals moved by only 480.143/476.803 milliseconds.
No configured kernel or server fault marker appeared.

The 367 MiB shared cache contained 3,081 files. Its sorted relative-path and
content manifest is `cache-files.sha256`, whose SHA256 is
`8d85d9cc5e9f5d271048c0bd32863a489fe2e20c55dfd2e3d6f97c6a8a417e3f`.
The corrected result summary SHA256 is
`4a0a2b38cd04691690729e71cb5fe1c2b7201fe02c3a57f49f540330065b042c`.

## VERDICT

F04 closes negatively at the frozen-target gate. Shared-cache restart
coherence is real, and MTP1 has a strong internally coherent speed signal, but
that signal cannot be attributed while target identity differs.

The result does not by itself prove that accepted draft tokens caused the
difference. F03a already showed that a newly compiled artifact selects a
target, and MTP changes the compiled target verification shape. Proceed only
to F04a: copy the exact F04 cache and force all MTP drafts to reject while
retaining the two-row MTP target verification path. Require direct reuse of
the same target AOT key. Long-context, concurrency, P2P-on serving, and shelf
promotion remain blocked.
