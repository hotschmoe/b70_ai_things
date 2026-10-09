# Numerical v2: one producer-merged chronological log

CONFIG -> Actual v1 run F08/layer0-numerical-onecard-v1 preserved. Its frozen20
versus21-off and SAME21 off/on four-prefix heads/all48 residuals, four current
26-field frames and12 actual native packets passed. Its parent lifecycle failed:
public UR allocation/free records went to stdout, owning L0 markers to stderr,
while the parent checked only stderr. Separate reader arrival timestamps or
concatenation cannot establish the missing cross-stream chronology.

COMMAND -> New source-only v2 drivers and real CPU subprocess fixture:
python3 strata/flash-next/test_merged_numerical_protocol_cpu_v2.py
python3 strata/flash-next/test_layer0_numerical_qualification_cpu_v2.py
python3 strata/flash-next/test_qualify_layer0_numerical_cpu_v2.py

RESULT -> PASS: actual bounded producer duplicates FD2 onto FD1 before ANY output,
then writes UR allocation/free on originalstdout calls, L0 owner markers and
request diagnostics on originalstderr calls, plus READY/GEN/T/LP/DONE. One OS pipe
and one reader preserve emitted order in engine.combined.log. Protocol lines are
classified for the existing GEN transport; PCL/PREFIX/SFD/L0/UR and all other
nonprotocol lines remain in request diagnostic records. Synthetic reversed
allocation/owner order and missing free fail the same logical lifecycle checker.
Actual21 source, layout/nonce negatives and new genuine one/pair prepared-chain
controls pass. No GPU or model run by this agent.

VERDICT -> READY frozen v2 host orchestration only. Source20/21 SDK/weights/math,
v1 drivers/results and existing CPU/GPU evidence remain unchanged. Current
capture scope is26of33; full_model_math_qualified remains false. Runtime graph
handle retirement is explicitly unobserved, target owner logical free remains a
real required gate, all7023104B including56 controls/constants are checked.

## Producer merge and canonical provenance

The container launch command begins exec2>&1 BEFORE engine exec, dynamic loader
or model constructor can emit. Thus the producer itself has one write pipe,
not merely a Docker client whose two producer streams were separately captured.
The parent process also uses stderr=STDOUT and a single pump. No two-reader
sorting, copied stderr labels or fabricated metric channel exists. The combined
file is canonical; engine.protocol.log is only a derived protocol subset. Each
arm result binds the canonical combined-log SHA and producer merge declaration.
The parent audits candidate21_on/engine.combined.log after terminal collection.

Genuine new plans reuse the actual qualified source21/C1/full390/current model
identity and matched finalized20 reference, because only the host logging driver
changed. They are newly generated, not edited v1 JSON:

- F08/layer0-numerical-onecard-prepared-v2/plan.json, SHA
  88d3f7cb3bbb3f88a78bf445456ca552ff6e538e1dfd99427b078fd0f2b082e7.
- F08/layer0-numerical-twocard-prepared-v2/plan.json, SHA
  f299f1dc7eef6e49c6a379b926b880228d2dcea7fda22b486e5c18511876ac00.

```
python3 strata/flash-next/qualify_layer0_numerical_v2.py \
  --plan /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f08-20261009/layer0-numerical-onecard-prepared-v2/plan.json \
  --output NEW_V2_ONECARD_PARENT_DIRECTORY
```

Run only after parent CPU retest/commit. Old source21 engine is unchanged; actual
full390/source identity gates are not replaced. Parent retains both GPU leases,
fd8/9, exact source/runtime/nonce/field checks, bounded child and owned cleanup,
source sentinel watch, pre/post health/fault refusal and post-terminal full4hash.
The same21 off/on and20-versus21-off head/residual comparisons and actual12packet
checks remain mandatory. Pair requires finalized same21 v2 one-card receipt.
V1 cannot become a pass by relabeling its disconnected log streams.

All synthetic CPU trace tests retain their synthetic scope. Actual chronological
UR/owner evidence must come from a NEW GPU run; numerical/model state/whole math
and API/natural completion/latency are not qualified by this host logging fix.
Sourceplan/freeze: layer0-numerical-log-order-plan-v2.json.
