# 0.2.0-beta re-validation of the upstream Strata PRs (2026-10-19)

Scripts and specs used to rebase-check the hotschmoe PRs on Niko1221/Strata 0.2.0-beta (main e88ee39d)
on two Arc Pro B70s. Results live in /mnt/vm_8tb/b70/results/pr-020 (scratch, not in git).

- build_all.sh: every PR engine + parity test, PR_BUILD_IMAGE=strata-sycl-dev:020 (IGC 2.41.5; the
  older IGC 2.36.3 of our pinned build image ICEs on 0.2.0's int8 prompt-expert kernels, upstream #3219).
- parity.sh: the six kernel parity self-tests on card 1 in the runtime image.
- serve.spec (+ configs/pr-020-layer-c1.json): main vs flag-off vs flag-on, one stream, layer split 30, 65536 ctx.
- phase2.sh: hcq8.spec at 32768 ctx (the Q8 copies do not fit at 65536), dac2.spec (two streams,
  configs/pr-020-layer-c2-16k.json), ep.spec (configs/pr-020-ep-c1.json), then oracle.sh.
- oracle.sh + slot_oracle.py + oracle-configs/: #2526 slot parking (no MTP) and #2511 draft accounting vs main.
- summ.py: per-arm tok/s, tokens per verify window, ms per window, output SHA, prompt tok/s.
