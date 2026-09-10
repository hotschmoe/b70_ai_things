# Agent2 reconstructed input audit

CONFIG -> Audit the supplied agent2 session and the frozen64-message target
payload used by the Pi replay. Source session SHA256
ad32738546762298540971e7854059766771179e2ed3adba0fe9268f2cff5fd4.
The reconstruction stops before the first32-bang response; it does not filter
all forms of earlier degeneration. No GPU, LAN write or tool execution.

COMMAND -> Independently reproduce the conversion and require exact equality
with agent_2-target.json. Map every retained message to the original JSONL
line, ID and stop status; check tool-call/result pairing. Raw audit and script:
results/bang_isolation_20260910/pi-input-audit/{audit.py,agent2.json}.

RESULT -> The first bang, source line67/idfbf61443, timestamp
1788999884886, stopReason=aborted, is excluded. No retained message contains
32 consecutive bangs. Before that first bang, no assistant message was skipped
by the status filter. All27 tool calls have27 matching results: zero orphan
results, duplicate call IDs or unanswered calls. The independent conversion
exactly matches all64 frozen messages.

Earlier non-bang degeneration does remain:

| Target index | Source line / ID | Original status | Retained evidence |
| --- | --- | --- | --- |
| 18 | 21 / 736482f0 | stop | 1660 characters, including1509 consecutive zeros and garbled prose |
| 26 | 29 / ab204ed8 | stop | 89 characters of garbled prose |
| 35 | 38 / 8bed4acb | stop | 168 characters of garbled prose |
| 51 | 54 / 4fb77362 | toolUse |1194-character Python command references undefined SCHEMA |
| 52 | 55 / 73d9b444 | toolResult, isError=true | Recorded NameError for SCHEMA |
| 55 | 58 / a488bee0 | toolUse |6668-character,1582-line heredoc containing long repeated In/2 lines, garbled tail and no closing EOF |
| 56 | 59 / c2a8a413 | toolResult, isError=false | Shell warning that heredoc reached end-of-file without EOF |

The statuses stop and toolUse indicate completion shape, not semantic quality.
Even a tool result marked isError=false can report a shell warning. The replay
keeps such original tool results as data; it never executes their commands.
The future manifest wording now states first *bang* excluded and earlier
non-bang degeneration may remain. Historical manifests, payloads and frozen
source copies are preserved unchanged.

This is reconstructed history, not captured Pi wire input. The bundle omits
the exact system prompt, complete tool schemas and forwarded request bodies.
The replay substitutes a system prompt and minimal tools, omits past thinking,
and uses manifest sampling/seed settings whose actual forwarding is unverified.
Pi extensions may transform stored history before transmission. Therefore a
session JSONL record is not proof that the LAN client sent that exact history.

The new fixed-image failure must remain separate: its b:,.: the/python: suffix
is absent from the input, and prior phase/old-FP8, stock/old-FP8 MTP3 and MTP0,
and phase/FP16 runs produced valid JSON on the exact same reconstructed request.
Earlier malformed history is present, but its presence does not establish that
it caused the new output-only garbling. That strict Pi quality arm remains
failed. Independent comparison lives beside its preserved raw output.

## Guard source availability

The local guard source is /mnt/vm_8tb/github/bang-guard/src/bang-guard.ts at
commit d341fe6c9a3b53061e9936b0cb78b6f08e6f6438 (package0.3.0).
BangDetector tracks32 consecutive exclamations across content deltas and JSON
unicode escapes. hasBangs and cleanContext cover bangs and timestamps tainted
by a detected trip; they do not generally reject long zero/line repetition or
short gibberish. The bundle itself contains no guard source, and the LAN
installed version is not verified.

A source-pinned, locally staged opt-in repetition detector patch is feasible
without editing this external checkout or writing to the LAN. It should cover
text, thinking and tool deltas, preserve completed tool actions, and use the
same context-cleaning criterion on recovery. Tests must include split chunks,
JSON escapes, benign code/repeated data and completed tool-result pairing.
Long numeric runs and repeated lines can be legitimate, so broader detection
needs explicit thresholds and false-positive review. It would mitigate obvious
repetition, not reliably classify all short gibberish or repair server state.
No guard patch or external deployment was performed by this audit.

VERDICT -> We did not feed the first bang response back into this replay. We
retained earlier malformed outputs carrying normal completion statuses.
Describe that precisely; neither ignore the retained degeneration nor attribute
the new runtime failure to it without a controlled comparison.
