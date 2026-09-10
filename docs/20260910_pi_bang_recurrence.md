# Pi bang recurrence on the patched day trial

The September 10 current-world-5 capture confirms continuing generated bangs
and non-bang degeneration on the deployed GDN-phase + MRV1 repaired image.
The repairs passed their controlled reproductions; they did not complete the
overall corruption repair. This is new workload evidence, not a replay of the
original world's bang messages. No service changes or GPU experiments were
performed during this read-only investigation.

## Evidence and scope

CONFIG -> User-supplied archive:

    /mnt/storage/StrongSync/junk/agent-world/citizen-pi-sessions-20260910T172335Z.tar.gz

SHA256:

    e1848e97c91533d2224a0903ba3e419bc6b1216413e7f7a293716be7452b3578

Raw evidence root:

    /mnt/vm_8tb/b70/results/bang_recurrence_20260910T172335Z

COMMAND -> Validate archive paths/types before extraction; check supplied file
hashes; independently audit Pi sessions and RPC/gateway events; inspect the
actual running container's source, arguments, scale receipts and preserved
server logs. Diagnostic commands embedded in sessions were not executed.

RESULT -> All 136 supplied file hashes match. The archive contains 271 members
and 684,474,713 expanded bytes. Separate audits retain current-world-5 apart
from historical worlds 1-4. Files were captured individually while the world
ran, so boundaries are not a simultaneous snapshot. Travelers were excluded;
citizen sessions are not a complete record of every endpoint caller.

VERDICT -> Suitable for identifying new generated failures and correlating
citizen request lifetimes. It is not exact HTTP request/raw SSE capture and
does not establish all server-side scheduling or sampling inputs.

## Deployment really contained both fixes

CONFIG -> hotschmoe-dd.service, current trial 20260910T072531Z; container started
07:26:38 UTC. Actual image:

    sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1

TP2/P2P0, MTP3, FULL_DECODE_ONLY graphs, prefix cache, FP8 E4M3 KV, context
200000, maximum concurrent sequences 4, batch 32768. Primary served alias
remains hotschmoe-dd. The actual resolved cache block size is 1600.

COMMAND -> Read inside-container source hashes without importing the GPU
runtime; preserve identity/configuration, all 34 scale receipts, timestamped
logs and metrics in runtime/.

RESULT -> Actual GDN phase source and MRV1 runner match the patched image.
Runner SHA256 is
7299b4cfabc447b15b66a5fcaf5bb858a0783fc46340d561f292f5f96e99e789.
All scale receipts match fresh artifact
be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062.
Preserved post-opening logs through approximately 17:33 UTC contain no engine
errors, traceback, preemption or NaN reports. Metrics report zero preemptions;
maximum logged KV usage is 27.6% of the 861428-token logical pool. Existing
host tracing covers startup, not the later failing generations.

VERDICT -> This is recurrence with the repairs actually loaded. Neither an
unpatched deployment nor cache exhaustion is supported by these observations.
HTTP 200, transport completion and quiet engine logs do not prove coherent
generation. These observations cannot exclude silent numerical/state defects.

## New session outcomes

CONFIG -> 19 current-world-5 sessions, September 10 07:40:07 through 17:23:16
UTC. Session message timestamps generally identify request starts; RPC observed
times and gateway ends identify later completion/abort events.

COMMAND -> Count assistant records once, inspect generated content by field,
separate tool/user quotations, and review earliest histories manually.

RESULT -> 1256 assistant attempts partition as follows:

| Recorded outcome | Count |
| --- | ---: |
| toolUse | 584 |
| stop | 254 |
| length | 5 |
| aborted, each containing 32 generated bangs | 188 |
| local per-agent 429 error | 206 |
| stream ended without finish_reason | 19 |

The 188 bang-bearing messages occur in thinking (155), text (6), or tool
arguments (27). No user or toolResult message contains a qualifying bang span.
All observed bang spans end at 32, consistent with the guard threshold.
These are assistant generations, not repeated counts of their RPC updates.

A conservative heuristic additionally finds 36 non-bang messages with at least
1024 repeated non-whitespace characters or 32 identical nonempty lines.
Short garbling occurs outside that heuristic, including normal stop responses.
The 843 completed responses report prompts from 1107 to 151287 tokens.
Zero usage on aborted/error responses means unavailable accounting, not zero
prompt tokens or absence of cache reuse. Counts are not a general failure-rate
estimate: retries and damaged continuations are not independent trials.

VERDICT -> The guard limits the bang stream but does not establish coherent
output or catch all degeneration. Historical world-4's 41 bangs are excluded
from the 188 new occurrences.

## Earliest useful comparison cases

| Request start UTC | Citizen / session message | Observation |
| --- | --- | --- |
| 07:40:09.791 | agent_1 / 43e5b657 | Partial multi-tool response; gateway TimeoutError after 183.243 seconds; Pi reports missing finish_reason. |
| 07:44:11.734 | agent_1 / 339ac4f7 | Earliest clear manually observed garbled thinking, normal stop after 75 output tokens; prompt 9544, including 6400 cache-read tokens. |
| 08:28:34.846 | agent_2 / 43ba437e | First current-world bang; thinking includes 1566 zeros before bangs; earlier non-bang garbling already exists in this session. |
| 08:42:32.579 | agent_3 / 35f26d26 | First bang for agent_3, after earlier short garbling; no overlapping admitted citizen request in its gateway interval. |

The earliest substantive garble has response ID
chatcmpl-aw-c5b6a71c2829ae886e2769e4. Its gateway interval is
07:44:11.756-07:44:18.026, with no overlapping admitted citizen request.
Preceding completed assistant outputs appear coherent, although the earlier
transport failure makes this a cancellation/recovery history, not a pristine
fresh-engine control. Lack of citizen overlap does not exclude other endpoint
callers or previously retained backend state. In fact the backend reports two
running requests at 07:44:15, inside this garble interval, with cache usage
6.3% and no waiting requests. Thus this case is NOT demonstrated to be a
server-isolated request. Sparse 10-second statistics cannot identify the other
request or reconstruct exact execution; preserve both observations.
The distinctive garbled fragments are absent from preceding session records;
the prior task/tool outputs do not ask for this garble.

The first bang's response ID is
chatcmpl-aw-6f24e162e1dcf235f2a28419; its abort was observed at 08:29:21.958.
Agent_3's first bang is chatcmpl-aw-7764163f49e3081bba7d8d86, ending at
08:42:40.462. Exact source paths, lines and history limitations are retained
under session-audit/ and gateway-audit/.

These cases refute the assumption that only old Pi bang history or near-200K
contexts explain the recurrence. They do not yet identify a kernel defect,
prove TP2 is necessary, or prove cancellation caused later corruption.

## Separate gateway/recovery track

CONFIG -> Current-world citizen gateway log and RPC message-end records.

COMMAND -> Join Pi/RPC response IDs to gateway request IDs; inspect admitted
intervals and distinguish model generation from transport/admission outcomes.

RESULT -> Gateway has 1410 starts, 1408 ends and 367 rejected requests; peak
overlapping admitted citizen intervals is 2. End statuses include 1162
transport_complete, 226 client_disconnected, 19 TimeoutError and 1 cancelled.
All 188 bang aborts join to client_disconnected. Pi records 206 per-agent 429
errors; the gateway's larger rejection count is a different population.
Unlike the original bundle's immediate recovery failures, none of the 188
new bang disconnects has a same-unit gateway 429 within one second before or
after release. Only two have one within 30 seconds. Do not carry the original
immediate retry-race conclusion into this capture without new evidence.
Two starts lack ends at the capture boundary. No HTTP request bodies, complete
sampling settings or actual system/tool schema are recorded in this bundle;
upstream_request_id is null or absent. RPC deltas are not raw upstream SSE.

VERDICT -> Transport/admission trouble is real and must be investigated
separately. Do not treat transport_complete as model correctness, infer every
disconnect is a bang, or count every gateway rejection as a Pi recovery.
Twelve of the 19 Pi stream errors ID-join to gateway TimeoutError; seven lack
response IDs. Equal aggregate counts alone do not establish all 19 pairings.

## Updated isolation order

1. Preserve exact wire JSON and raw SSE for the earliest reproducible new
   failure, with authorization credentials removed but prompt/tool content and
   sampling settings intact. Keep pre-failure and later damaged histories
   separate. Reconstruction must explicitly label missing transformations.
2. At directed downtime, reproduce the earliest short garble on the pinned
   live configuration. Compare fresh lifecycle against a matched sequence
   containing timeout/cancellation, drain and cached continuation. Assert
   actual cache hits and evaluate thinking/text/tool JSON, not just bangs.
3. Hold payload, weights, scales and sampling fixed. Independently disable
   prefix reuse, MTP, then graphs; compare FP16 versus FP8 with matched actual
   reuse and block geometry. Preserve failed results and change one feature
   per comparison. The serial case deserves coverage as well as concurrency.
4. Use both cards as independent TP1 targets for paired controls and cross
   over cards when outcomes differ. Then compare the informative arm at TP2.
   Existing TP1 bang reproductions already show TP2 is not a universal cause.
5. Continue the prepared FP8 geometry and Mamba-copy oracles and review GDN
   state lifecycle around restored prefixes and interrupted MTP. They remain
   candidate mechanisms, not findings established by this capture.
6. Investigate gateway retry/admission ordering separately using source and
   lifecycle events. The new 429s are not the original immediate-release
   pattern; determine which operations overlap before proposing a race fix.
7. Retain the paused SGLang qualification as an alternative backend campaign.
   This recurrence supplies no evidence that SGLang with the same features
   avoids it; full-model FP8/cache/graph/MTP parity is still unqualified.

No new GPU jobs, endpoint restart, live patches or guard installation were
performed. The user-authorized day trial remains serving; quality qualification
is contradicted by these failures and no shelf promotion is warranted.
The user explicitly requires notification of the concrete proposed patch/test
or backend change, followed by confirmation that clients are paused, BEFORE
taking down the vLLM endpoint. Read-only preparation may continue meanwhile.
The earlier [campaign handoff](20260910_bang_campaign_resume.md) retains exact
prepared commands and health/lease requirements; prioritize these recurrence
cases before completing the older synthetic qualification checklist.

Portable CPU auditors and frozen case metadata are under
[vllm/int4/pi_recurrence](../vllm/int4/pi_recurrence/README.md). Their replay
against this frozen bundle reproduces the saved audit outputs. These are
analysis checks, not new inference correctness tests.
