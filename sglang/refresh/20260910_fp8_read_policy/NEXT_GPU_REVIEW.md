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


## Measured v2 startup failure and prepared v3

CONFIG -> Exact14ee v2 prefix extra_buffer, FP8/Triton/eager/MTP0, card1.
COMMAND -> Parent allocated card1; existing feature_gate and selected lifecycle
ran after all frozen/prior health/numeric prerequisites passed.
RESULT -> Strict card1 pre-health passed. Before model loading, installed
mamba_hook rejected extra_buffer for Qwen3_5ForConditionalGeneration. The
current pinned overrides.py:499-505 rejects extra_buffer unconditionally on
XPU. Normal owned cleanup completed, container absent, strict card1 post-health
passed, parent and lifecycle returned1. No model/token/feature result occurred.
VERDICT -> Configuration failure, not FP8 arithmetic failure. Original v2 and
its pre-edit lifecycle source are preserved. No GPU retry was performed.

A separate v3 plan changes only the requested cache strategy to no_buffer and
adds its required page_size1 companion. Existing overlap-disabled/Triton setup
already meets other validators. Source MambaRadixCache remains enabled and
inserts both KV indices and matching Mamba states at finished-request prefix
boundaries; no_buffer uses copy_from rather than extra-buffer ping-pong. This
is genuine hybrid prefix caching, subject to measured positive cache-hit gates.
The strategy itself does not establish graph or MTP correctness; both stay off.
Actual-source CPU validators and default-versus-candidate argv tests pass.
All26+32+serial gates remain and the new output is fresh. Parent review and
allocation are required before the separately prepared plan-v3 launch.


## Measured v3 FP8/cache and independent serial outcomes

CONFIG -> Same14ee TP1/card1 FP8/be02/Triton/eager/MTP0; explicit no_buffer
and required page1, overlap disabled. V2 extra_buffer failure stays preserved.
COMMAND -> Root allocated card1 after prior TP2 teardown/strict health. Ran
exact frozen plan-v3 through feature_gate; no outer lease and no public serve.
RESULT -> Strict pre-health passed. Startup resolved the supported policy and
loaded target16-layer calibrated scales. Cold compilation completed. All26 text
checks and32 concurrent tool checks passed,32 attempts, zero bangs/retries.
The strict cache counter increased6 ->126304 (delta126298); live c4 observed.
This is the gate's aggregate metric delta, not an independent unique-token count.

Both separate session0 serial requests returned prompt4045/completion2058,
finish stop, raw matched_stop248046, text ending433comma. Their visible output
is byte-identical to each other and the vLLM raw EOS diagnostic, SHA256
4d85ef8116e01fee788f0acaaac2506029606c51820f781742a95fd39dc72311.
Raw durations200.735s/196.316s are observation bounds, not matched speed claims.
SGLang usage returned prompt_tokens_details=null; the unchanged serial client
raised AttributeError while reading cache fields. Preserve its rc1 and missing
metadata separately from actual cache reuse. The incomplete array is also a
real independent semantic negative; raw matching EOS remains available.

Normal cleanup completed, owned container absent, strict card1 post-health
passed. Parent/lifecycle/job rc1 reflect the separate serial diagnostic failure;
fp8-cache-stage-outcome.json remains strict26+32 PASS. No reset or further GPU
retry occurred. Card1 was returned to parent allocation.
VERDICT -> Bounded full-model FP8/no_buffer cache gate passes, but not whole-job
semantic success, graph/MTP/TP2/long-context feature parity, speed or promotion.
Cross-backend exact EOS output is evidence separating the early-stop case from
the repaired vLLM native cache-contract corruption. Raw output:
/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang14ee-fp8-prefix-eager-mtp0-card1-v3/
