# Flash-Next goals and measured status

Set by the user on 2026-10-13. One or two users. Backend focus: Strata.
vLLM parked, ready to port. 27B INT4 AutoRound on vLLM is the fallback.

| Goal | Target | Measured (2026-10-14, bounded screens) | Gap |
|---|---|---|---|
| Prefill | 2000+ tok/s | median 1952-2004 tok/s at 32k, 1260-1337 at 4k across same-behavior screens (IQ3_XXS, v0.1.44.1, split27, spec4; 3 prompts each) | 32k near target, not a stable 2000+ floor; short prompts below |
| Decode | 150+ tok/s per stream | median 83.8 coding, 64.5-65.7 after 4k, 64.9-65.0 after 32k, spec4 (at most 3 drafts) | about 2x short |
| Context | 200k per stream, 2 streams | IQ3_XXS: 2 x 200000 slots with spec4 (at most 3 MTP drafts); both retrieved three needles in 199501-token prompts | capacity screen passed; sustained load remains |
| KV eviction to RAM | restore without recompute under load | both GPU slots overwritten, then concurrent returns reused 199555/199582 tokens; 54/27 new tokens read. RAM restore 430/422 ms, client TTFT 1.97/4.40 s (parking16384 MiB/4 entries) | bounded forced-eviction test passed; sustained use remains |

Quality: on 2026-10-14 Flash-Next IQ3_XXS beat the 27B INT4 8-0 on paired
langs + sysadmin tasks (41/43 vs 33/43); see the A/B brief, Outcome 2.

Current serving config: strata/flash-next/serve-iq3xxs-200k.json, clean
upstream 296b59d2/v0.1.44.1. Peak sampled resident VRAM 28.22/29.27 GiB;
all 24576 main experts in VRAM. Health, concurrent coherence, reasoning,
tool round trip and clean teardown passed. The soak observer is stopped at user request; multi-day stability is
still unqualified. Intake observed uptime was 10.7 h on the old config.

Matched concurrent coding screens: layer spec3/spec4 total63.56/63.94
output tok/s, EP spec3 69.99 (35.8-37.1 per stream). Three two-request
windows per arm,107 input/1500 output each; total includes prefill and
admission. All six outputs match. Every overlap response offers ZERO MTP
drafts. Upstream --batch-mtp is default-off and rejects layer splits;
its EP combination remains untested. This limits the value of depth tuning
for two users. Single-stream quiet layer spec3 is77.7 coding/64.7 prose;
spec4/.5 is83.8/65.7, and cutoff0.3/0.7 gives no broad gain. Retain spec4,
which includes the seed and offers at most three MTP drafts.

EP #1798/#2205 on head296b59d2 records91.9 coding and71-72 prose decode
with spec4, but only578-579 prefill at4K/32K, so it is not promoted.
A4K trace puts52.8% of prefill in combine (peer wait/row collection included).
Docker IPC launcher fix: https://github.com/Niko1221/Strata/pull/2468.
A transfer microprobe crashed before large timing; rebind/per-card/compiled
collective health recovered, and no native copy-path patch was promoted.
A new tool continuation fails on EP/spec3 and passes on layer/spec4;
remaining functional2x2 arms are needed to isolate its cause. This further
blocks EP/spec3 promotion. The original EP/spec4 tool probe did pass.
IQ3_S, peer-row prefill collection and concurrent MTP remain open.
Final layer spec4 is serving :18085, observer off, final functional gates
and post-experiment GPU health pass. The only config update is log path.

Evidence: docs/20261014_flashnext_strata_optimization.md, JOURNAL.md S1/S2/S3/S4
entries dated 2026-10-14, and raw results under
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_opt_20261014.
EP/MTP raw results: /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_ep_mtp_20261014.
The original 2026-10-13 measurements remain in the A/B armB report and
JOURNAL.md.
