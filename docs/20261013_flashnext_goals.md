# Flash-Next goals and measured status

Set by the user on 2026-10-13. One or two users. Backend focus: Strata.
vLLM parked, ready to port. 27B INT4 AutoRound on vLLM is the fallback.

| Goal | Target | Measured (2026-10-13, single measurements) | Gap |
|---|---|---|---|
| Prefill | 2000+ tok/s | 1902-2052 tok/s at 32k prompt, 1260-1277 at 4k (IQ3_XXS, Strata v0.1.43.1) | at target for long prompts, short prompts below |
| Decode | 150+ tok/s per stream | 65-81 tok/s with MTP4, 40-45 without | about 2x short |
| Context | 200k per stream, 2 streams | IQ3_S: 1 x 200k (no MTP). IQ3_XXS: 2 x 131072 with MTP4 | 2 x 200k untested |
| KV eviction to RAM | restore without recompute under load | conversation parking (8192 MiB, 4 entries) restores a 30k conversation in 0.54-0.60 s | works at 30k; not tested at 200k or under sustained load |

Not yet shown: multi-day stability on Strata (longest clean run about 3 h),
two-stream aggregate decode gain (54 tok/s total at 2 streams, same as 1).

Evidence: JOURNAL.md entries dated 2026-10-13 and
/mnt/vm_8tb/b70/results/flashnext_iq3_ab_20261013/armB/report.md.
