# Residual early-stop token observation

CONFIG

Exact candidate d55637b3353e, native6717e3ec7fe3. Read-only installed Python
source exported through CPU-only containers, no devices/network, 8GiB bound.
Raw source identities and prepared one-request plan:
/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/early-stop-token-audit

COMMAND

probe_stop_tokens.py reads a reconstructed session0 wire request. The recorded
request file precedes the frozen stream helper, which adds stream=true and
stream_options.include_usage=true at kv_campaign_agent_probe.py:34. The raw
wire-provenance.json records this reconstruction and both source hashes.
Default is a CPU
dry run. --run is only for a later parent-owned controlled endpoint. The sole
payload change is return_token_ids=true. All model, message, sampling, cache
salt, stream and stop fields remain identical. No HTTP requests were sent.
test_stop_tokens.py verifies this delta and synthetic EOS/stop/truncation SSE.

RESULT

Installed entrypoints/openai/chat_completion/protocol.py:415 supports
return_token_ids; streaming emits prompt IDs once and per-choice delta token
IDs. This differs from return_tokens_as_token_ids, which needs logprobs.
No logprobs, ignore_eos, min_tokens or stop-policy change is needed initially.

chat_completion/serving.py:688-728 includes output.token_ids on both ordinary
and finishing chunks and forwards stop_reason. Its lines643-646 suppress
metadata when include_reasoning=false and a parser exists. Do not toggle that
setting silently if the selected request hides metadata. A missing token ID
does not establish that EOS was absent.

v1/core/sched/utils.py:103-115 distinguishes primary EOS (finish stop, typically
stop_reason null) from an additional stop token (finish stop, integer reason).
Length termination is separate. sampling_params.py:659-685 merges model
generation_config EOS IDs into stop_token_ids even without user stop fields.
The local model generation_config lists 248046 and 248044; effective primary
EOS must be checked against the active tokenizer/config, not inferred solely
from this list. Integer stop_reason may therefore indicate configured EOS.

v1/engine/detokenizer.py:108-127 removes a terminal stop token from text
detokenization but appends it to retained token IDs. output_processor.py:398-431
keeps token IDs and stop reason; its lines670-674 set a matched string stop as
the reason. EOS omission from visible text is expected and is not evidence of
API corruption. Raw metadata should identify the terminal token in this route.

The probe retains complete raw SSE bytes, terminal empty deltas, token IDs,
stop_reason, response identity, usage and text separately. Diagnostic exit0
means a complete observation, never semantic correctness. A truncated stream
or absent token metadata stays distinguishable rather than being called EOS.

VERDICT

Prepared only. If terminal ID is an effective EOS and stop_reason is consistent,
the model/sampler produced a terminating token; this does not by itself prove
model correctness. A string stop indicates detokenizer policy. Missing IDs or
conflicting terminal metadata is inconclusive and motivates a narrower server
trace. ignore_eos would be a separate intervention after this unchanged-policy
observation, not a replacement for the strict early-stop failure.
