# research_moe_optimizations.md -- MoE-specific multi-GPU optimization ideas (try later)

Status: idea/scratch doc (started 2026-06-24). Captures reasoning from the J.18-J.21 PP/TP/push-ar
work about why MoE behaves differently from dense under multi-GPU, and what is worth trying next.
Nothing here is measured yet unless it says MEASURED. Cross-refs: docs/P2P_GPU.md (J.2 push/pull,
J.13 PP-vs-TP, J.18-J.21 PP results), JOURNAL Lever B/C, README shelf table.

Model in scope: Qwen3.6-35B-A3B quark-W8A8 (~3B active, 256 routed experts), served TP=2 today.


## 0. Background: collectives are a FAMILY, and push-ar only does one of them

Tensor-parallel (TP) serving combines partial results across cards with collective ops, and there
is more than one:

- **all-reduce** -- everyone ends with the SUM of all cards' copies. The dense-TP workhorse (one per
  attention / dense-MLP layer). This is the ONLY primitive `contrib/vllm_push_allreduce` reimplements
  (our custom L0-IPC "push my partial sum to the peer and add").
- **reduce_scatter / all_gather** -- sum-but-keep-a-slice / gather-the-slices. (Their composition is
  an all-reduce.)
- **all-to-all** -- each card sends a different chunk to every other card. The natural primitive for
  MoE expert routing (dispatch each token to the card holding its chosen expert, then combine back).

push-ar accelerates ONLY all-reduce. So it helps dense TP (attention + dense MLP) and does nothing for
the MoE expert path, which (per the contrib README + vLLM MoE-TP) rides reduce_scatter/all_gather and
all-to-all-style routing. That is why the README calls push-ar "dense-only": not a policy, a primitive
mismatch. See the J.21 discussion.

Secondary reasons push-ar is a poor fit for THIS MoE even on prefill: (a) the MoE's dominant cross-card
cost is the routing/combine, not the attention all-reduce push-ar could touch; (b) ~3B active params =
compute-light, not very collective-bandwidth-bound; (c) its real bottleneck was per-token op-launch
overhead, already fixed by graph capture (Lever B, 8.7x decode), a different disease.


## 1. THE IDEA: PP=2 should suit MoE EVEN BETTER than dense (hypothesis)

Pipeline parallel (PP=2) puts a contiguous block of LAYERS on each card and runs each stage at TP=1.
Consequence for MoE: **every layer's full set of 256 experts lives whole on one card**, so top-k
routing + dispatch + combine for that layer is ENTIRELY ON-CARD. There is no cross-card all-to-all /
reduce_scatter / all_gather at all. The only thing crossing the card boundary is the inter-stage
activation handoff: ONE point-to-point push per microbatch (the primitive our cross-die / push-fast,
read-slow fabric is fastest at -- J.2: 11.3 GB/s push vs 3.24 GB/s pull).

Why this could beat the dense PP win:
- The MoE collective PP eliminates (all-to-all routing) is NASTIER on our fabric than the dense
  all-reduce -- a full crossbar of cross-die transfers vs a single reduction. Removing a worse baseline
  is a bigger relative win.
- TP-for-MoE (a.k.a. expert/tensor sharding) PAYS that routing tax on every MoE layer; PP pays a single
  activation push per microbatch instead. Same shape as the dense J.13 bet (128 allreduces -> 1 push),
  but the per-layer cost it removes is larger.
- Capacity is fine: PP splits by layers, so each card holds ~half the total expert weights (35B / 2
  ~= 17.5 GB), well inside 32 GB.

This is the standard PP-vs-EP tradeoff from the literature: PP trades the expert all-to-all for a
pipeline handoff + a startup bubble. On a fabric where all-to-all is expensive, PP can win.


## 2. Why MoE DODGES the two blockers that killed dense PP (J.19)

Dense PP=2 was parked on two blockers; neither necessarily applies to MoE:

1. **MTP + PP = NotImplementedError (drafter lacks SupportsPP).** Irrelevant for MoE: the 35B-MoE-int4
   MTP run was NET-NEGATIVE at concurrency (Lever C -- ~3B active makes decode already fast, MTP's
   verify overhead does not pay off). So MoE does not WANT MTP, and the MTP+PP incompatibility is moot.
   We would run PP=2 MoE without MTP, which is the configuration we want anyway.
2. **Captured PP=2 (GRAPH=1) returned EMPTY output (J.19).** That was on the 27B **W8A8 + GDN** hybrid
   (gated-delta-net + int8 captured pieces), the SAME family as that model's documented captured-TP=2
   bug. The MoE quark-W8A8 is a DIFFERENT architecture and captures fine under TP (Lever B works), so
   captured PP for the MoE might just work. UNTESTED -- this is the main thing to check.


## 3. Open questions to answer (in order)

1. Does the MoE model class implement vLLM's `SupportsPP`? (Cheap: try `--pipeline-parallel-size 2
   --tensor-parallel-size 1`; a config-time NotImplementedError means no, like the MTP drafter did.)
2. Does PP=2 MoE serve COHERENTLY eager? (Compare to the dense J.18 eager PP=2, which worked.)
3. Does PP=2 MoE + GRAPH=1 capture COHERENTLY? (The dense hybrid failed here; the MoE may not.)
4. PP=2 vs TP=2 head-to-head on the 35B-quark, matched config (eager first, then captured if #3 holds):
   decode t/s, TTFT, and especially aggregate at concurrency 4/8 where the pipeline fills and the
   bubble hides. Baseline = the README TP=2 captured row (43.1 t/s c1 / 53.2 agg c4).
5. Bubble cost at c=1 (single stream pays the handoff latency with no overlap) -- expect PP to trail TP
   at c=1 and catch/pass at concurrency.

Tooling already exists: `scripts/110_serve_pp2_graph_mtp.sh` + lib.sh `PP` support (b70_multicard) from
J.19. A MoE PP variant is mostly: point CKPT/IMG at the 35B-quark recipe, set PP=2/TP=1, drop MTP.


## 4. Operational cautions (do NOT skip)

- Captured-PP on the dense hybrid CORRUPTED the multi-GPU collective state (J.20): a later TP=2 serve
  went empty + DEVICE_LOST + card-1 hang. If MoE captured-PP misbehaves the same way, expect a wedge.
- **xe-reset CANNOT recover this box -- REBOOT ONLY** (J.20): `xe` drives the console/display, so
  `modprobe -r xe` always fails "in use". Budget a reboot between risky PP capture attempts; run with
  the guard (`B70_AUTO_RESET=1` will DETECT but cannot self-heal here).
- The single-card pre-flight probe does NOT catch collective-state degradation (guard gap, J.20). Treat
  a coherence-gated gen probe returning EMPTY as a red flag even if /health is green.


## 5. Adjacent idea: push-based reduce_scatter / all_gather (make TP-MoE faster instead)

If PP turns out not to work for MoE (capture broken, or bubble too costly), the OTHER route is to give
push-ar siblings: custom L0-IPC push implementations of `reduce_scatter` and `all_gather` (and maybe an
all-to-all), so the TP-MoE path also rides the fast push transport instead of oneCCL. This is a real
kernel project (separate ops, different data-movement than the all-reduce push), and is strictly more
work than trying PP first. Try PP first; keep this as the fallback for accelerating TP-MoE in place.


## 6. Summary

- push-ar is all-reduce-only -> helps dense TP, not the MoE expert path (primitive mismatch). [reasoned]
- PP=2 should suit MoE even better than dense: it keeps each layer's experts whole on one card and
  removes the expensive routing all-to-all entirely, leaving one push/microbatch. [hypothesis]
- MoE dodges both dense-PP blockers (does not want MTP; the captured-PP bug was a GDN-hybrid artifact).
  [reasoned; capture-coherence UNTESTED]
- Next experiment: PP=2 on the 35B-quark (eager -> captured), head-to-head vs the TP=2 captured baseline,
  run guarded with a reboot budget. [TODO]
