# E1-E4 source-audit interpretation follow-up (2026-10-15)

Authors: Codex agents `/root/ple_upstream` and `/root` (root independently
read and accepted the source findings). Source audit only; no new benchmark,
build, GPU run or model consultation.

This note qualifies strong claims in the original
[E1-E4 report](README.md). Its scripts, raw rows and original verdict text
remain preserved as historical evidence.

E3's [chain](scripts/e3_xccl_sync.py) repeats `x *= 0.5` followed by two-rank
SUM all-reduce. Both ranks receive the same seed, first 1.0 and then 3.0,
and the replay checks expect that seed back. Each complete pair is an identity
on these inputs. Consequently a wholly empty replay, or omitted complete
multiply/reduce pairs, also passes both checks. The statement that these
checks pass "only if every reduction executes" is unsupported. They do not
prove that all 96 captured collectives, or the intended captured workload,
executed. This is a control limitation, not evidence that work was omitted.

The raw measured chain medians remain facts: eager/captured 9030.1/1213.6 us
at one row and 9402.1/2094.2 us at four rows. The recorded value-check
sentinels are true, and post-card and compiled collective health passed.
The reported "per-sync p50" is the whole-chain median divided by 96, not a
distribution of individually timed collectives. These timings include the
dependent multiplies, submission/replay and final synchronization. They do
not isolate PCIe transfer, establish a topology latency floor, prove complete
captured TP correctness, or supply a budget for Strata's different EP/MTP path.
P2P was disabled in this experiment.

E2 measured the reported llama.cpp layer/tensor and MTP results, but the
Strata numbers are historical references from different engine execution and
speculation configurations, without a contemporaneous matched Strata arm.
Comparable acceptance and a shared coding template do not attribute the
entire speed gap to kernels/execution or exclude speculation and state effects.

E4 measured approximately 33 ms per eager step and capture failure on a
four-layer dummy-weight model. It did not isolate pure host overhead or
measure linear scaling to the full 48-layer model. Neither a real-weight
accuracy result nor a full-model latency projection follows from it.

These qualifications leave the user's Strata focus unchanged. They constrain
causal and hardware claims; no new experiment or backend migration is proposed.
