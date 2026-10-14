# Actual paired copy-engine overlap, 2026-10-12

CONFIG -> Original SDK52/main Unsloth UD-Q4_K_XL, pair24/24, prefill512,
context4096, noMTP/mirror0. Device-only PTI: paused P2048 GEN8 warmup and
selected fresh P2048 GEN8. Compare COPY_ENGINE0 versus1. GPU UUID/timer
geometry, all source/model/profile/input identities match. Native/tool argv
match after replacing only owner session IDs; environment matches except copy
flag and diagnostic owner/binding IDs. Both warm/selected output arrays match.

COMMAND -> OFF is the original complete V5 ArmA component independently
admitted by scoped95911, retaining failed wholeparent. ON is independent
single-arm producer59431 PASS421s and reader63007 PASS. Both original full4,
source/all8, health/kernel, owned EOF/removal, clock controls and individual
1GiB bounds pass. Observed inspectors70782/43083 bind exact raw clock/Chrome
instances to each selected emission sequence; no old failure rewritten.

RESULT -> OFF operations use engine0,0 with zero observed copy/compute
intersection. ON copies additionally use copy-only engine1,0 (7087/5821
observed operations). Explicit payload bytes match by direction on each card.

| Card | Copy union ms OFF -> ON | Compute union ms OFF -> ON | Copy/compute overlap ms OFF -> ON | Explicit H2D GB |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1613.432 -> 1620.213 | 6826.083 -> 6999.780 | 0.000 -> 461.433 | 22.100 |
| 1 | 1324.293 -> 1328.801 | 6647.171 -> 6732.729 | 0.000 -> 369.167 | 18.359 |

Total H2D payload40.4581888GB is command payload, not physical PCIe traffic;
mapped-RAM loads inside kernels remain unmeasured. Per-device interval unions
are not additive critical-path accounting, and cross-card unions must not be
summed into utilization. Selected slices include prefill+8decode; no proven
phase split. Host/device calibration, complete operation coverage and physical
idle are unqualified. Instrumented request times are not serving speed.

VERDICT -> Actual dedicated-copy route overlaps compute, supporting the
mechanism behind the separate clean COPY0/1 prefill gain of about5%. The route
moves the same explicit payload; it does not remove repeated expert streaming.
HC projection dominates observed named compute and is the next concrete kernel
candidate. No2000/50, fullmodelmath/API/concurrency/shelf qualification.

Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/

- OFF scoped component proof SHA256
  be61d764340d57b1c26b7f506645d35cb5ae34154bd1f4761cecbd4875588e76
- ON producer report SHA256
  208b720757bdb60c7f44253795bc82307c35f11ef89806914d13b6c563996142
- ON public proof SHA256
  c0bcc1a742d82c0e0f51d0e547edd0a4179c8710af84f5896ef24c2b23877d08
- ON observed JSON SHA256
  6845ebe83a046f2fea28d37c9ba07045a472782116edefd9ff052ffd38742c6a
