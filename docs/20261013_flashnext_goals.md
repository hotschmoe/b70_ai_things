# Flash-Next goals and measured status

Set by the user on 2026-10-13. One or two users. Backend focus: Strata.
vLLM parked, ready to port. 27B INT4 AutoRound on vLLM is the fallback.

| Goal | Target | Measured (2026-10-14, bounded screens) | Gap |
|---|---|---|---|
| Prefill | 2000+ tok/s | median 2004.2 tok/s at 32k, 1337.2 at 4k (IQ3_XXS, Strata v0.1.44.1, split27, MTP4; 3 prompts each) | long-prompt screen at target; short prompts below |
| Decode | 150+ tok/s per stream | median 83.8 coding, 64.5 after 4k and 65.0 after 32k, MTP4 | about 2x short |
| Context | 200k per stream, 2 streams | IQ3_XXS: 2 x 200000 slots with MTP4; both retrieved three needles in 199501-token prompts | capacity screen passed; sustained load remains |
| KV eviction to RAM | restore without recompute under load | both GPU slots overwritten, then concurrent returns reused 199555/199582 tokens; 54/27 new tokens read. RAM restore 430/422 ms, client TTFT 1.97/4.40 s (parking16384 MiB/4 entries) | bounded forced-eviction test passed; sustained use remains |

Quality: on 2026-10-14 Flash-Next IQ3_XXS beat the 27B INT4 8-0 on paired
langs + sysadmin tasks (41/43 vs 33/43); see the A/B brief, Outcome 2.

Current serving config: strata/flash-next/serve-iq3xxs-200k.json, clean
upstream 296b59d2/v0.1.44.1. Peak sampled resident VRAM 28.22/29.27 GiB;
all 24576 main experts in VRAM. Health, concurrent coherence, reasoning,
tool round trip and clean teardown passed. The final same-config restart
starts a once-per-minute real-use soak observer; multi-day stability is
still unqualified. Intake observed uptime was 10.7 h on the old config.

Two-stream aggregate decode has no established improvement: two 1500-token
displacement answers took 54.51 s including prefill (55.0 output tok/s
overall). IQ3_S with a rebalanced split, expert-parallel #1798/#2205,
MTP depth tuning and short-prompt prefill tuning remain untested here.

Evidence: docs/20261014_flashnext_strata_optimization.md, JOURNAL.md S1/S2
entries dated 2026-10-14, and raw results under
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_opt_20261014.
The original 2026-10-13 measurements remain in the A/B armB report and
JOURNAL.md.
