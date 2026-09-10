# Next optional SGLang FP8/cache control

CONFIG

Independent backend control after current vLLM TP2 diagnostics fully tear down.
vLLM native repair remains preferred. No quarantine dependencies or image change.
Image14ee7d0112b4 is pinned main2f7393f0 with source-only loader/read-policy fixes,
INT4 GPTQ model, TP1/card1, Triton, FP8 be02, eager MTP0, prefix extra_buffer,
context8192, chunk512, c4, memory0.90, text-only skip-server-warmup. This does
not test graph, MTP, long context, multimodal native execution or TP2 collectives.

COMMAND

The original plan is preserved and must not be launched: its copied tool probe
misses imported sibling kv_campaign_probe.py. New plan copies that exact known
dependency (SHA3edcbcffd914) into a separate directory, preserving all feature
settings and original strict gate code. After the strict26 text +32 concurrent
360-record tool gate succeeds, a separately recorded exact session0 serial
cold/reuse diagnostic runs. Its result is not substituted for the strict gate.

Prepared, not launched:

    python3 sglang/refresh/20260910_fp8_read_policy/feature_gate.py /mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang14ee-fp8-cache-plan-v2/plan.json --run

Do not add an outer gpu-run: tp1_viability acquires card1 itself and verifies
the inherited lease/device pin. Parent must allocate card1 after clean TP2
teardown and strict both-card/compiled-pair post-health.

RESULT

CPU gate validation passes; all recorded source hashes match, output is fresh,
and actual lifecycle/tiny/tool/serial CLI imports and --help pass. Prior image
matched pair-health PASS and all-nine attention numeric result PASS (including
strict pre/post health) are retained. This plan performs fresh selected-card
strict pre/post health, owned container cleanup and defers any pair reset until
the selected lease is released and both cards are acquired.

The strict model gate requires one target16-layer calibrated loader receipt,
26 text checks, 32/32 tool checks without retries/bangs, and a positive measured
cache-token delta. Feature-enabled flags alone do not count as cache coverage.

Pinned serving_chat.py:1085 rejects return_token_ids with streaming; its stream
schema uses matched_stop rather than vLLM stop_reason. Therefore the follow-up
uses the existing serial client and original payload without raw-token metadata.
It checks stable hotschmoe-dd identity and independent semantic/cache outcomes.
Prompt rendering, special tokens and backend cache metrics can differ from
vLLM: compare measured prompt tokens, outputs and actual hits, not assumed
identical internal token sequences. The 8K context may also reject an expanded
rendered prompt plus4096 output reservation; preserve that as a capacity/config
result, not semantic failure. No feature expansion is authorized by this plan.

VERDICT

Ready for review, not GPU execution. Estimated practical slot10-25 minutes is
an unmeasured planning estimate; cold compilation/CPU contention may exceed it.
Explicit startup bound20 minutes and combined job bound4000 seconds plus strict
health/teardown remain the hard operational limits. Full-model FP8/cache is still
unqualified until this run completes. No SGLang feature parity or promotion claim.
