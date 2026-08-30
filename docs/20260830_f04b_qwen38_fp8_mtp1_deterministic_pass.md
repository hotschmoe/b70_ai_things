# F04b Qwen3.8 official-FP8 deterministic MTP1 pass

Date: 2026-08-30

Status: research-qualified for the bounded single-stream 1K configuration.
MTP1 was exact to the frozen deterministic MTP0 target and delivered a matched
53.890 percent decode improvement under the local P2P-off safety port.

## CONFIG

- Harness commit: `59f72c0`.
- Official FP8 W8A16 checkpoint; local MTP-capable packed-RMS image
  `sha256:338ef5e2c956471a195a0644e2acb38288ac6837ebf5282671794be5329a7e2b`.
- TP2, P2P-off, MTP1, persistent GDN draft scratch, FP16 target/KV,
  graph-off, one request, 1,024 context, and prefix caching off.
- Two lifetimes used separate empty caches with explicit deterministic
  compiler controls and no autotune selection surface.
- Both F02g MTP0 attempts were required frozen references.
- Result directory:
  `/mnt/vm_8tb/b70/results/f04b_qwen38_fp8_neural/20260830T053900Z/`.

## COMMAND

```text
STAMP=20260830T053900Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f04b.sh
```

## RESULT

Both fresh MTP1 compiles matched one another 12/12 and each matched both F02g
MTP0 references 12/12. Target AOT key `57e8f544...` and draft AOT key
`fe3112d...` were identical across caches. Target compilation took 96.08 and
96.28 seconds; draft compilation took 9.67 and 9.55 seconds. Both 976-file
caches contained zero `.best_config` files.

Class-balanced rates were 17.648289 and 17.650913 tok/s, with a 17.649601
tok/s median and 0.015 percent spread. Against F02g's matched 11.468959 tok/s
MTP0 median, MTP1 improved decode by 53.890 percent. This attribution is valid
for the bounded fixed corpus because configuration, output arrays, identity,
health, and teardown all matched. Acceptance commonly ranged from about 65 to
93 percent in ten-second windows.

Both canaries, graceful teardowns, card health, and compiled P2P-off
collective health passed. Container host RAM peaked at 8.399 GiB; minimum
MemAvailable was 112,819,476 KiB or 107.593 GiB; swap remained zero. Device
accounting reported 14.59 GiB weights plus non-Torch, 1.20 GiB peak
activation, and 8.45 GiB KV cache per card.

The cache-manifest SHA256 values were
`991754783fac88e890060c1d8a1e056f46cc1d31d121b013fff8fdcae3ad2e7a`
and `47114dc0892f1e2762ec2c450e959818569471705332eaddf22d0f9b5cd354e3`.
The summary SHA256 was
`4c7a689698e32bd3865f6e3147637ada3eb8a040556c9ed2706a0c6cdaa8963e`.

## VERDICT

F04b passes bounded target, speed, restart, teardown, and health gates. It is
the current official-FP8 local candidate, but it is not agent- or
shelf-qualified. Proceed to F05a at 32K configured context with actual growing
prefill shapes, a forced 4,096-token decode, two fresh lifetimes, and the same
P2P-off target and health gates. P2P-on full serving remains blocked.
