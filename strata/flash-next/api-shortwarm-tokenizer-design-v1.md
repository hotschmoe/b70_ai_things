# Authentic shortwarm tokenizer fixture

CONFIG -> Only the reviewed warm messages change. The old authentic target
messages, rendered strings and both 235-token ID arrays remain exact. Source35
tokenizer/frontend, original tokenizer files and c388 image are pinned. The
container has one CPU, 512MiB with no additional swap, read-only root, no network
and no devices. Its sole writable mount is generated/, which cannot write the
seed, renderer or source snapshots through another path. No GGUF is mounted.

COMMAND -> Root runs produce_api_shortwarm_tokenizer_fixture_v1.py --output NEW.
Then generate_api_shortwarm_positive_case_v1.py --fixture-root NEW --output CASE
--port PORT. The first command runs Docker CPU tokenization only. Public
finalized_binding(root) verifies owner/image/entrypoint/actual mounts/resources,
terminal exit/removal, original logs and source/tokenizer/data hashes before and
after recollection. The case generator requires this authentic public admission.

RESULT -> CPU8 synthetic controls pass, including actual recipe-derived Docker
inspection, foreign ownership, devices/network/memory/mount/entrypoint refusal,
target ID/rendered/literal/type mutation and checkpoint-prefix collision. These
controls do not execute Docker, tokenizer or model payloads. Root must run the
actual container before new warm IDs exist. No fabricated positive receipt or
token IDs are in this source generation.

VERDICT -> Separate authentic fixture and case, not a buffer-only A/B claim.
Existing V2/V4/V5 and their evidence remain unchanged. The generated case needs
a separately admitted shortwarm runtime successor; existing V2 strict case
admission deliberately rejects it. Real warm two-row observation, positive
handoff counters/raw49, fresh numerical comparison, healthy ownership teardown
and current all4 source identity remain mandatory and unqualified here.
