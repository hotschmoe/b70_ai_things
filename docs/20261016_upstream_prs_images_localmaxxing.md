# 2026-10-16: night-lab changes upstreamed, images on the daily driver, localmaxxing runs

Follow-up to docs/20261016_night_lab.md. Orchestrated by Claude Opus 5.5 (claude-opus-5-5) with Claude Opus 5.5 and
Claude Fable 5.1 (claude-fable-5-1) subagents porting the PRs.

## Upstream Strata (Niko1221/Strata): what was filed

Every change is one commit (HC fused read: two, stacked) on upstream main 296b59d2, SYCL-only (sycl/** + docs/INTEL.md),
behind a default-off flag, with a parity self-test. Branches are on the public fork hotschmoe/Strata.

| PR | issue | flag | what | measured on main (below) |
|---|---|---|---|---|
| #2804 | #2803 | STRATA_MMVQ_KREG=1 | KREG small-batch MMVQ, Q4_K/Q5_K/Q6_K/IQ4_XS 1-8 columns | C1 +19.5 % (verify window -12 %), not bitwise |
| #2802 | #2801 | STRATA_IQ_DECODE_FAST=1 | ISA-guided IQ expert decode | C1 +7.7 %, 40.8K-ctx +7.1 %, bitwise |
| #2806 | #2805 | STRATA_BATCH_QSA_ATTN, _QSA_TOPK_SCAN, _QSA_SCORES_TRANSPOSED, _ATTN_MERGE_ACTIVE | decode QSA selection/attention in fewer launches | C1 +0.6 %, C2 +2 %, bitwise; kernels 1.3-4x |
| #2808 | #2807 | STRATA_PROMPT_ATTN_UNION=1 | XMX prompt attention over 8-position selection unions | prefill 4K +16 %, 32K +15 %, 40.8K TTFT -12 % |
| #2810 | #2809 (bug) | STRATA_HC_Q8=1 (fix) | the SYCL HC read never used the Q8_0 copies; now it does | verify window -5.3 %, +0.64 GiB VRAM |
| #2812 | #2811 | STRATA_GR_FUSED=1 | two-kernel fused BF16 HC read (depends on #2810) | verify window -2.5 % |
| - | #2800 (bug) | --pipeline-windows 2 | first verify window fails, UR_RESULT_ERROR_UNINITIALIZED | reproduced on main |

Not filed: the EP-specific work (needs the unmerged EP PRs #1798/#2205), batch-MTP pairs/depth, D2D admission, the
fused shared expert / BF16 KREG GEMV (stack on #2804), router/GDN tweaks. llama.cpp: not filed (its CONTRIBUTING
forbids AI-written posts).

## Validation on upstream main (CONFIG -> COMMAND -> RESULT -> VERDICT)

CONFIG -> builds of main 296b59d2 and each branch (pr_build.sh: one shared build tree src-pr, incremental, flock);
strata/flash-next/night/configs/pr-main-layer-c1.json: 2x B70 `--layer-split 30 --prefill 8192 --max-context 65536
--kv int8 --spec 4 --mtp <BF16> --mtp-max-t 4`, one slot, fixed prompts (nl.py --tag nl). 200000 context does not fit
all 24,576 experts on a plain layer split of main (710 refused); HC Q8 (+0.64 GiB) ran at 32768 / --vram-reserve-mib
512; C2 at 16384 / 512 with `--batch-groups 1` (main refuses --batch-mtp on a layer split, so C2 ran without MTP).
COMMAND -> pr_validate.sh phases A-E (results /mnt/vm_8tb/b70/results/pr-20261016/{phaseA..E}.log, runs/), parity
self-tests on card 1, NLL gates (nl_ppl.py), kernel benches; the timed phases under cpu-run --quiet.
RESULT ->
- main ran in three processes: identical outputs (2 C1 prompts + a 40,796-token turn), 79.5 tok/s median C1.
- flag unset, all six branches: output SHAs and draft counts identical to main.
- parity: KREG 160/160 rows (multi-column bitwise to 1-column), IQ 15/15 bitwise, decode-attn 45 hashes reproduced
  per switch, prompt union 5/5 + 9 timing cases, HC Q8 33/33, HC fused 33/33.
- NLL (no degradation; not claimed as gains): prompt union 2.6234 -> 2.5506 (control 2.6321; bootstrap 95 % interval
  includes zero, same cells attended - code review), HC Q8 3.1974 -> 3.1699.
- --pipeline-windows 2: all 16 graphs capture, the first verify fails, UR_RESULT_ERROR_UNINITIALIZED; card health ok.
VERDICT -> six PRs and two bug reports filed; daily driver unaffected (separate binary).

Harness pitfalls this stage: `pgrep -f <script>` inside a wait loop matches the loop's own command line (never ends);
`grep -h` over several logs - read which file a block came from before concluding (MERGE_ACTIVE was first misread as a
regression); upstream main's VRAM fit differs from the EP stack (refusals at 200K); a native IQ pack needs `--spec >= 2`.

## Daily driver: images on

CONFIG -> strata/flash-next/dd/hotschmoe-dd.json now carries a "vision" section (CPU strata-vision encoder built from
the Strata tree's tools/vision against the pinned llama.cpp, /mnt/vm_8tb/b70/build/strata-ab/build-vision-cpu/bin,
8 threads; mmproj-Qwen3.8-Flash-Next-BF16.gguf; `--vision`). Text-only fallback: hotschmoe-dd-text.json.
RESULT -> encoder 1.2 s for a 256x256 image on 8 CPU threads, no VRAM; quadrant-colour test correct through the API,
the frontdoor and concurrently with a code request; C1 140.9 / 140.5 tok/s (unchanged), prefill 32K 2158; C2 84.5
per stream median, below the ~92 at deploy but with a wide spread (one slow stream on rep 1) - needs a matched A/B;
dd_functional.py passes (thinking, tools, round trip, concurrency, 2048-token reply 152.4 tok/s). open-webui shows the
image button by default (capabilities?.vision ?? true). Upstream docs/INTEL.md still lists images as "not yet" for the
SYCL port; the CPU encoder path works on it.
VERDICT -> deployed. Follow-ups: matched C2 A/B text vs vision; an upstream docs/setup note for images on SYCL.

## open-webui

max_tokens default 8192 -> 32768 (the user's reply was cut while thinking); the stale Pliny preset deleted (exported to
/mnt/vm_8tb/b70/backups/), the connection filtered to hotschmoe-dd, hotschmoe-dd the default/pinned model. Still set
from that preset: temperature 0 and repeat_penalty 1.15 in the user/global defaults (the penalty hurts code; left for
the user to decide).

## localmaxxing (strata/flash-next/dd/localmaxxing/)

lmx v0.1.54 has no --canonical-prompt, so lmx_verified.py measures with the same remote method and records Strata's
per-request timings (draft_n / draft_n_accepted) for the Verified badge.

| run | tok/s | status |
|---|---|---|
| MTP, reasoning-v1 | 122.2 | APPROVED, Verified |
| MTP, code-v1 | 117.9 | APPROVED, not verified: the site assumes a fixed draft width (accepted + drafts/4 = 460 vs 512); Strata drafts 2.99 per window here |
| no speculation, reasoning-v1 | 66.2 | APPROVED, Verified |
| no speculation, code-v1 | 65.9 | APPROVED, Verified |

No-speculation runs: the native IQ pack needs `--spec >= 2`, so `--spec 2` with no drafter (no MTP, suffix/lookup
off): 0 drafts proposed. The engine source is public: hotschmoe/Strata branch b70-night-20261016 (1c360526 = the
deployed binary).

## b70-launcher: Strata engine (SergiioB/b70-launcher#4)

At the maintainer's request (he asked for our recipe in the launcher), a `strata` engine kind + builder + the
`qwen38-flashnext-strata` recipe (our exact flags and STRATA_* env), weights repo ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
added to the download allow-list, docs and 25 new tests (340 pass on Python 3.9/3.12/3.14). Prepared by a Claude Opus 5.5
subagent. The engine source it points at is the public branch hotschmoe/Strata b70-night-20261016 (1c360526).
CONFIG -> launcher branch feat/strata-engine on the maintainer's 0.6.14 (06bf7af), a fresh clone of the Strata branch,
our release binary, pack and MTP layer, isolated launcher state; both B70s, hotschmoe-dd stopped 20:33:52-20:36:53 UTC.
RESULT -> first live attempt (20:22) reported "exited" after 5 s: a launcher race (a container created but not yet
started reads as Running false / ExitCode 0 for 1-3 s with a 14.6 GB image), fixed in shared runtime code with a test;
the same rendered command run by hand loaded fine. Second attempt on 7381fb3: ready in 32 s, /v1/models ok, b70 test
ok, 512-token reply 135 tok/s incl. prompt (one request), metrics read, clean stop in 4 s, health before/after ok.
VERDICT -> PR #4 opened with nine open questions for the maintainer (entry layout, fork branch until #1798/#2205 land,
stop timeout, images not in the recipe).
