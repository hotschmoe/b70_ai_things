# PTI census failure evidence retention

CONFIG -> Preserve frozen PTI43 actor/source212 and its actual failed run.
The old exec helper required returncode0 before saving stdout/owner-after, so
nonzero helper errors were discarded. An empty after_READY directory locates
failure after mkdir, but does not identify the cause.

COMMAND -> New owned exec V3 always retires/drains the actual CLI, then saves
exact command, combined stdout, returncode, original owner-after or its error,
and a typed failed/successful diagnostic receipt before rejecting. It retains
unknown values as null; normal native cleanup/lease ownership is unchanged.
PTI actor/reader V2 select that helper only, with reader strictsuccess joins.
Root uses the original PTI command with run_source43_pti_latency_v2.py and
validate_source43_pti_latency_v2.py in new output directories after review.

RESULT -> Three actual tiny CPU processes exercise rc7, post-launch interruption
and failed owner-after inspection; complete output/terminal/failure records
survive. No Docker/GPU/model payload was touched by agents.

Possible source seams: helper capture creates output first, reads /proc/1/stat,
then /proc/1/maps; maps may refuse nondumpable/credential/ptrace policy access.
Alternatively mapselection requires the exact /pti/lib/libunitrace_tool.so
name/path and an executable mapping, then exact backing dev/inode/file SHA.
The empty actual directory is consistent with any earlier refusal. Root's
small NoDri reproduction and retained original stdout are needed to distinguish
these possibilities; no permission, PID, path or digest gate is relaxed.

VERDICT -> Observability correction only; no causal claim or incomplete census
qualification. Pending live census/model coverage still needs an actual run.
