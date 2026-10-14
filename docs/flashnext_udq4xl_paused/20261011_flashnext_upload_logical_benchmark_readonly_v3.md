CONFIG
Original corrected benchmark report172ff115 remains immutable. Root subsequently
added supplemental timing summaryc59e0a52 inside its run directory. Reader V2's
exact nine-file guard correctly refused that addition. V2 and its frozen source
plan are preserved. Reader V3 declares precisely the resulting ten-file tree.

COMMAND
validate_upload_logical_benchmark_v3.py --root ORIGINAL --output NEW_FILE
It retains V2 current bytes, worker provenance, source/result, negative and full
root-only payload checks. It additionally pins the exact supplemental summary
and independently recomputes medians, ratios and complete-operation reduction
from the eight original trial records. Unknown files remain rejected.

RESULT
Three tiny reader CPU tests pass. Actual V3 metadata-only recollection passed,
including all eight trial hashes, exact ABBA order and result equality, saved
workers5 executions/5 positive/15 negative scans per B trial, original A zero
parser counters, current log/runtime/source bytes, and unchanged ten-file tree.
The independently recomputed timing summary matches exact recorded values.
The metadata-only receipt explicitly leaves current SDK/pack payload edges and
current live upload predicate/negative reexecution pending root full mode.

VERDICT
This is a read-only diagnostic CPU admission gate recollection. Original pack
byte proofs have semantic_digest_calls0; no semantic pack-use claim is made.
No GPU/model execution, full model quality, full preparation speed or serving
latency qualification. First calls are slower in B; repeated-gate and complete
operation measurements have separate scopes. OS-cold state is not claimed.
