# Flash-Next on vLLM: session handoff (2026-10-21)

Written after an office power outage, around 15:38 UTC on 2026-10-21. The
host rebooted. All agents were killed mid-run: grok, Codex, Opus, Fable.
Nothing was lost on disk except /tmp, which is tmpfs, so any agent scripts
kept under /tmp are gone.

Read this file first. Then read BRIEF.md (the rules for agents) and the
last ~40 entries of JOURNAL.md.

## 1. Goal and standing user direction

- Serve Qwen3.8-Flash-Next (ISTA GSQ-RCO IQ3_XXS GGUF) on vLLM across two
  Intel Arc B70 cards. The aim is to beat our own Strata stack, which is
  the current daily driver, on the same model and cards.
- Latency is the top metric. Motto: make it work, make it right, make it
  fast. Mindset: there is no software moat; build any kernel or feature
  that is missing.
- Targets: prefill 2000+ tok/s, decode 150+ tok/s per stream, 2x200K
  context, KV evicted to host RAM and restored without recompute.
- The localmaxxing leaderboard ranks only NON-speculative runs. No-MTP
  decode is therefore the headline number.
- Keep the vLLM fork thin and rebaseable on upstream.
- The Qwen3.6-35B-A3B GGUF is a proxy for testing single pieces.
  Flash-Next is the real target.
- Ornith-1.5-122B is PARKED. Revisit it only if it beats Flash-Next on
  terminal benchmarks AND survives 3-bit quantization. See
  reports/2S1_ornith122b_fit.md.
- The user approved localmaxxing no-spec submissions for three configs:
  Flash-Next, 35B Q4_K_M and 35B UD-IQ3_XXS. MTP submissions are NOT
  approved yet. Ask the user before any other external submission.

## 2. Bars to beat (Strata, same model, 2x B70)

| metric | Strata |
|---|---|
| decode C1 no-spec | 66.2 / 65.9 tok/s on lmx reasoning-v1 / code-v1 (~15.1 ms) |
| decode C1 with MTP | 141 tok/s (7.09 ms) |
| decode C2 with MTP | 92 tok/s per stream (10.87 ms) |
| prefill | ~2200 tok/s |

## 3. Where vLLM stands (all measured; evidence in JOURNAL and reports)

| item | vLLM now | notes |
|---|---|---|
| Flash-Next coherent serve | yes | greedy 17*19 -> 323 and 23*29 -> 667 match Strata |
| Flash-Next decode C1 no-spec, 64-token bench | 15.20 ms (65.8 tok/s) | merged recipe, max_len 131072 |
| Flash-Next on the lmx suites (512 tokens) | 61.3 / 61.5 tok/s | NOT submitted; trails Strata's 66.2 |
| Flash-Next decode C2 no-spec | 19.5 ms per stream | |
| Flash-Next MTP | k=3 C1 9.19-9.28 ms; k=2 C2 12.89 ms per stream | O5; k=2 on/off parity passed |
| Flash-Next prefill (sidecar) | 8K 2448, 12K 2397, 32K 2349, 128K ~1587 tok/s | F1; 128K pays dense attention growth |
| KV capacity | ~356K tokens on card 0 in the merged recipe | X2: 2x200K bf16 fits |
| MoE decode kernel | 0.53-0.58x Strata R2 layer time | O3; we are faster |
| KV parking to RAM | 200063-token restore in 0.30 s (13.9 GB/s) | Ornith stand-in, not yet Flash-Next |
| 35B proxy, localmaxxing (SUBMITTED, verified) | UD-IQ3_XXS 120.9 / 122.1 tok/s; Q4_K_M 97.0 / 97.2 tok/s | IDs in localmaxxing/submitted-20261021/SUBMITTED.txt |

The layout of record is the EP sidecar. It runs as one process with world
size 1:
- Card 0 runs the trunk, the shared expert and about 138 experts per layer.
- Card 1 holds about 374 experts per layer (plugin b70_ep_xpu).

The DP2/EP2 layout is retired.

## 4. Recipes (start here)

- Merged best recipe: decode from O4 plus prefill from F1.
  - harness/o4_recipe.env
  - harness/f1_prefill_delta.env
  - harness/f1_merged_serve.sh (max_model_len 131072, chunk 12288)
  - The full flag list is in reports/2O4_flashnext_decode.md and
    reports/2F1_prefill.md.
- Decode-only recipe (O4 s4): harness/o4_serve.sh, port 18147. It waits for
  its own alias and for free RAM.
- MTP: harness/o5_window.sh and reports/2O5_mtp.md.
- 35B proxy: reports/1X5_proxy35b.md (X5 recipes) and o3/serve_35b.sh.
- localmaxxing tooling:
  - localmaxxing/ (lmx_vllm.py, run_dry.sh, validate_dry.sh)
  - L1 scripts in localmaxxing/l1/
  - report reports/2L1_localmaxxing.md
- Promotion gate: harness/promo_gate.sh, from X4 (reports/1X4_gate_lmx.md).
  It checks identity, the 323/667 gates, exact repeat, and 8K/32K needles.

Costs:
- The pinned PLE table uses 26.8 GiB of host RAM per Flash-Next process and
  takes about 90-120 s to load.
- Run only one Flash-Next serve at a time.

## 5. Code state (IMPORTANT)

- Campaign repo: master at 2d5ff2a1 plus this handoff commit. About 25
  campaign files written after 2d5ff2a1 were committed with the handoff.
- vLLM fork: /mnt/vm_8tb/b70/build/vllm-flashnext/vllm, branch
  b70/flashnext, tip 0a6170da8c.
  - It carries about 20 UNCOMMITTED files from many agents.
  - The 16:00 snapshot ref is b70/wip-snapshot-20261021-1600. It is a
    stash-style commit made with `git stash create`; the working tree was
    left untouched.
  - Earlier snapshots: b70/wip-snapshot-20261020-1940 and
    b70/wip-snapshot-20261021-0200.
  - Frozen localmaxxing engine: branch b70/lmx-20261021, commit cfd2b6ff,
    worktree /mnt/vm_8tb/b70/build/vllm-flashnext/vllm-lmx-20261021.
  - TODO: split the WIP into small, upstream-rebaseable commits per
    feature. All new behaviour sits behind default-off B70_* flags.
- Plugins live in vllm/flashnext/plugins:
  - b70_gguf_xpu: the current _C.so / _C.graph.so is O3's o3g build, sha
    38f7176a. O4 has its own variants under build/vllm-flashnext/o4/.
  - b70_ep_xpu: the F1 build is f1c, sha 249cf94e.
  - b70_qsa_xpu, b70_lowbit_xpu, b70_kvpark, b70_o4_xpu.
- The venv, wheel and build outputs are under /mnt/vm_8tb/b70/build/vllm-flashnext/.
- The vLLM base is upstream main 5fd62688 (0.32.1rc1.dev94), XPU build.

## 6. Open threads, in priority order

1. Beat 66.2 tok/s on the lmx suites (O4 owned this).
   - Per-token time grows inside the decode graph by ~1.6 us per KV token
     per step (15.3 -> 16.2 ms over 512 tokens). The FA split count is NOT
     the cause (JOURNAL O4 04:52).
   - O4 kept route + fused Q8 HC, which saves 0.54 ms (lmx-shape mean
     15.70 ms).
   - Next was "Ld": a unitrace over one 512-token lmx request to find the
     kernel(s) that grow with KV length.
   - Other queued levers: EP dispatch/top-k fusion (0.2-0.3 ms), streamed
     HC GEMVs (~0.5 ms), the remaining PLE kernels.
   - Once a dry run on the merged recipe beats 66.2, submit Flash-Next.
     Use the L1 procedure: freeze the engine, run the promo gate, dry-run,
     submit.
2. Nondeterminism (O5, JOURNAL 05:00). The GDN b|a BF16 F.linear in
   prefill returns different bits for the same input. It runs on oneDNN at
   M=20; decode at T<=8 uses bf_gemv. The fix is a batch-invariant GEMV/GEMM
   for that projection, like O4's bf_gemv. This blocks strict exact-repeat
   gates. O5 also found that dynamic MTP k (k=3 at C1, k=2 at C2) gives the
   best of both.
3. Wire QSA into a real serve (X6, JOURNAL 04:33-05:23).
   - The QSA paged KV spec now binds.
   - The component graph gate passed on card 1 (X6_QSA_RUNNER_GRAPH_OK).
   - Next: a one-card synthetic integration test, then a full QSA serve
     compared against Strata long-context references.
   - Dense FA matches QSA only up to ~2051 tokens, so this matters for
     200K fidelity. It also matters for 128K prefill, since dense attention
     grows from 1.9 to 7.9 s per chunk.
   - Generating Strata references needs hotschmoe-dd or Strata on the GPUs.
     Coordinate this; agents must not start the daily driver themselves.
4. Card-1 prefill experts (F1). They take 36-42 ms per layer against an
   estimated 10-15 ms floor. The stage-profile build f1/_C.f1bucket2.so is
   ready. Its run q2 was interrupted by the outage.
5. KV parking on Flash-Next (G4 spec, reports/2O2 and 1G4). It is proven
   on Ornith but not yet on the Flash-Next sidecar. Budget host RAM:
   26.8 GiB PLE pin plus a 32 GiB parking pool.
6. MTP to beat 141 tok/s. Today 9.2 ms is about 108 tok/s. The host gap is
   ~1 ms per step; the verify graph at T=3 is ~19.8 ms. MTP submission
   needs user approval.
7. Housekeeping:
   - Split the fork WIP into commits (section 5).
   - Retire agent-specific .so copies once the recipes pin the final builds.

## 7. Agents at the outage (all dead now; none need killing)

| agent | model | owned | last state |
|---|---|---|---|
| O4 | Opus | no-MTP decode | lmx-shape "Ld" unitrace queued |
| O5 | Opus | MTP | nondeterminism located; dynamic k measured |
| F1 | Fable | prefill | q2 card-1 stage profile starting |
| X6 | Codex | QSA serve | synthetic integration test prepared |
| G2b | grok | auditor | pass 34 |
| L1 | Opus | localmaxxing | standing by for Flash-Next |

Finished earlier: O3 (MoE kernels), X1-X5, C7b, H1e, all of wave 2.

The Opus/Fable agents and their transcripts belong to the old session and
cannot be resumed from a new one. Relaunch fresh agents with a pointer to
their report and to the JOURNAL. Launchers:
- vllm/flashnext/grok-agent.sh NAME PROMPT. It now passes
  --background-wait-timeout 86400; the 600 s default killed H1d.
- vllm/flashnext/codex-agent.sh NAME PROMPT. It runs gpt-6.1-sol at high
  effort with sandbox bypass.
- Prompts live in vllm/flashnext/prompts/.

Lessons learned:
- Give one owner per area. Write orchestrator directives as ORCHESTRATOR
  entries in the JOURNAL, because Codex agents cannot take messages
  mid-run.
- Set an explicit card priority. Last order: L1 > O4 > O5 > F1 > X6.
- Check claims against raw JSON/logs. The G2b auditor helps here.
- Never edit a window script while its shell is still running it.
- Codex can briefly hold .git/index.lock. Retry the commit.
- `pkill -f PATTERN` can match your own shell. Kill by pid.

## 8. First steps for the new session

1. hotschmoe-dd (Strata) and its frontdoor AUTO-STARTED after the reboot
   and are active. Before any GPU research, run
   `systemctl --user stop hotschmoe-dd`.
2. Run bin/xpu-health and bin/xpu-collective-health first. This is a fresh
   boot after a power loss.
3. Requalify the merged recipe: run harness/f1_merged_serve.sh, then the
   gates and the bench. Confirm C1 is ~15.2 ms and 8K prefill ~2450 tok/s.
4. Relaunch owners for threads 1-4 in section 6, Opus or Fable for the
   critical ones. Point each at its report and the last JOURNAL entries.
5. Restart hotschmoe-dd when the GPU session ends.

## 9. Resumed session (2026-10-21, after 16:00 UTC)

The user resumed this campaign with Sol 6.1 agents and explicitly authorized
startup serving down for research. AGENTS.md now reflects active vLLM work.
Owners are D1 decode, N1 GDN determinism, and Q1 QSA. Their current reports
are reports/3D1_decode.md, reports/3N1_determinism.md, reports/3Q1_qsa.md;
orchestrator recovery evidence is reports/3R1_restart.md. Consult the latest
campaign JOURNAL entries for live GPU ownership and subsequent results.

Post-outage native and compiled collective health passed. The unchanged
merged recipe reproduced C1 15.17 ms and C2 about 19.6 ms, passed identity,
arithmetic and 8K/32K needles, and failed strict exact repeat. No promotion.

Several runs marked pending above actually finished before the outage:
Ld's trace and F1 q2 stage profile completed with post-health; the first
QSA tiny engine test failed, then worker-config binding was repaired at
05:28. The resumed Q1 tiny test passes all 12 real pooled cache bindings,
FULL graph initialization and C1/C2 generation. Full-model QSA remains
unqualified. Read the new reports before repeating any recovered work.

The merged serve harness now keeps hotschmoe-dd first, defaults to maxlen
131072, supports optional promotion gates and an integrated pre-teardown
callback, and propagates failed checks. For canonical suites, use the
metadata shim (harness/d1_api.py); ordinary vllm serve does not expose the
collector's required /server_info endpoint on this pinned version.

Resumed matched control: 61.642817 / 61.800831 tok/s on the two 512-token
suites, no MTP, complete health/source/teardown evidence, strict-repeat
FAIL. D1's compact-plan16 component is numerically valid but its corrected
full serve (r2) failed with level_zero OUT_OF_RESOURCES40 before canonical
measurement. Keep it disabled; the PLE traceback is a surfacing site, not
a proven cause. Post-health recovered cleanly. No new submission or speed
promotion. N1 full-model validation uses the original attention path.
