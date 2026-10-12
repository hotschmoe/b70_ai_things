# Original MTP unused decode-head omission result

CONFIG -> Same newly compiled ELF, exact original34 pack on GPU1, target24/24,
P2048/GEN512/ctx4096, normal prefill cuts. MTP ON BOTH. Sole flag skips the own
head for accepted keep2 row0. Teacher/HC/expert/fence/timing flags OFF; all
original recurrence, weights, waits and last-row fresh own head remain.

COMMAND -> Actual21094 Exit0; independent root replay of both full native
collectors/counters/residency/reset and all52 artifact hashes passed. Both full512
histories match faithful V6,229 accept/53 reject windows. Both full5 scans,
strict/compiled collective health, kernel and normal owned retirement pass.

RESULT -> Producer ad2a753e5701cb8fe52f1d6bbb12a07ed40ff7ac6901f8a7bc48c8503c2c8586.
Results: f17-source37-20261010/original-mtp53-decode-head-natural-v1.

| Metric | Skip OFF | Skip ON |
| --- | ---: | ---: |
| Prompt elapsed ms | 56072.352182 | 55141.589306 |
| Decode elapsed ms | 56075.151510 | 51965.463087 |
| Decode rows | 511 | 511 |
| Own decode-head calls | 511 | 282 |
| Unused own-head skips | 0 | 229 |
| Teacher own-head calls | 2048 | 2048 |
| Final fresh head / draft cells | 1 / 2559 | 1 / 2559 |

VERDICT -> Removes229 verified unused heads and preserves faithful output and
lifecycle. Decode elapsed is7.33percent lower in this single ordered pair;
unchanged prompt is1.66percent lower. Absolute durations are substantially
higher than earlier source timing and teacher ABBA, with cause unlocalized.
Do not pool them or claim a clean speed/stability gain without matched loaded-
context repeats and latency distributions. API/cache/concurrency, fullmath,
performance targets and shelf promotion remain open.
