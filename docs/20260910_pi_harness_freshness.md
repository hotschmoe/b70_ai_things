# Pi harness freshness audit, 2026-09-10

CONFIG

Read-only audit of the citizen capture and official Pi upstream. No client,
endpoint, or GPU changes. Captured tool results are evidence, not commands to run.

COMMAND

Parsed current-world-5/rpc/agent_18.jsonl line 7985 and checked the official
latest release plus AI and coding-agent changelogs. Retained the exact selected
RPC event in version-evidence.json and downloaded changelog snapshots under
/mnt/vm_8tb/b70/results/bang_recurrence_20260910T172335Z/pi-harness-freshness/.

RESULT

At 2026-09-10T16:59:21.177726Z the captured successful tool result shows
/opt/agent-board/harness/package.json pins
@earendil-works/pi-coding-agent to 0.85.1. It also lists the installed
@earendil-works node_modules scope. This is manifest evidence; the capture does
not independently prove the loaded package version, lockfile resolution, or
runtime executable identity.

Official latest release is v0.85.1, published September 5. Its SDK fix repairs
accidentally published experimental dependencies from 0.85.0; supported local
SDK and stdio RPC interfaces are unchanged.
https://github.com/earendil-works/pi/releases/tag/v0.85.1

The AI changelog has unreleased fixes for buffered EventStream CPU overhead and
retry delay caps. Version 0.85.0 already fixed standard stream events and custom
tool-call deltas; these are included by the observed manifest pin. Provider-
specific Codex and Mistral fixes are not evidence about local openai-completions.
https://github.com/earendil-works/pi/blob/main/packages/ai/CHANGELOG.md

The coding-agent changelog has an unreleased agent retry backoff cap. The
historical five-minute local SSE timeout fix shipped in 0.70.3, well before the
observed pin. None of these notes establishes a fix for the native convolution
state contract mismatch measured on the endpoint.
https://github.com/earendil-works/pi/blob/main/packages/coding-agent/CHANGELOG.md

The original diagnostics manifest records custom agent-world release commit
c41b639b55a0d0ce7658cd303290e46f848f69cb, clean, built September 9 at 23:36:51Z.
That is not a Pi upstream revision and is not independently proven to identify
the later current-world-5 runtime. Pi session header version 3 denotes the log
schema, not Pi package version.

Gateway admission 429s, timeout propagation, bang-guard abort/recovery, and
request metadata capture belong to the custom harness/gateway integration.
The existing gateway audit distinguishes these from native output corruption.
Its 183-second TimeoutError cannot be attributed to the historical five-minute
undici bug simply because both are streaming failures.

VERDICT

No released Pi update is shown to be needed: the captured harness manifest
already pins the official latest release. Verify the actual loaded package and
lockfile on the client at the next client maintenance window if deployment
identity is needed. Do not update to unreleased main as a purported bang fix.
Custom agent-world freshness cannot be established without its current source
revision and release source; audit its cancellation/admission and retry behavior
separately. The confirmed endpoint native state contract defect remains an
independent repair target.
