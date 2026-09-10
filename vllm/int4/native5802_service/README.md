# Qualification-gated native5802 service candidate

CONFIG -> Exact d55637 image/native6717, current verified AutoRound model,
freshbe02 FP8 scales, TP2/P2P0/MTP3/prefixON/FULL_DECODE_ONLY,200K,c4,batch32768.
Public authenticated18080 -> loopback18124; hotschmoe-dd remains the first model
ID and the finalized200K research alias remains second. Existing key-file path
and frontdoor authentication semantics are retained. No secret was read.

COMMAND -> python3 -m unittest discover -s vllm/int4/native5802_service -p test_serve.py
RESULT -> Eleven CPU tests pass using temporary files and mocked processes:
feature/image/legacy-adapter rejection, old trial/incomplete evidence refusal,
exact command adaptation, unique34rank-layer scale coverage, actual receipt
file semantic failure, quality-limitation retention, PID/socket readiness,
auth source with a temporary fixture key, signals/teardown ordering and failed
teardown preservation. Actual serve.py check rejects unfinalized200K inputs.
VERDICT -> Source preparation only. No qualification.json exists; no systemd
file installed, service started, GPU used or old day-trial record changed.

The implementation deliberately reuses service_candidate's lease/frontdoor/
readiness lifecycle in a separate namespace. Qualification now requires both
NEW backend arms, hashes their actual results and strict semantic gates, actual
image/config/source identity, calibrated rank/layer coverage, host profile,
strict both-card+compiled collective pre/post health, normal teardown and exact
owned-container absence. A quality-negative sidecar remains false and binds the
independent CPU/GPU EOS evidence; it is not a general exception for malformed
outputs or new degeneration. Metadata states backend_configuration_qualified,
unrestricted_model_quality_qualified=false, known_numeric_array_quality_passed=false.

Inputs and qualification stages (not executed):

1. Once native5802-backend100k-v2 completes, run serve.py audit100k. It independently
   checks the exact16-job scope,24tiny/3early/192strict tools, full model/source
   identity, actual lifecycle/profile/health and quality-negative evidence, then
   creates qualification-receipt.json with exclusive creation. No receipt is
   emitted for a partial/failed run. The failed original100K plan is rejected.
2. The separate200K prerequisite consumer validates that immutable receipt and
   all backing hashes before its newly frozen arm can launch. The planned root
   is native5802-backend200k. A template alone cannot satisfy service readiness.
3. After both plans are frozen, prepare_inputs.py binds sources, plans, model/
   native receipts and known negative in inputs.json. Current inputs explicitly
   say inputs_finalized=false; there is no placeholder qualification marker.
4. Only after both complete arms pass, serve.py freeze creates a new scoped
   qualification.json. serve.py check revalidates full evidence and model files.
5. The repo-only90-native5802-qualified.conf may then be reviewed and installed
   by the parent. See MIGRATION_PLAN.md. Never blindly start the currently
   installed unit: its old override still points to7b day_trial.

At startup the wrapper reacquires both leases, verifies prior qualification,
starts private18124 with the exact qualified200K settings, checks image and
calibrated scales, runs pinned tiny24 plus deterministic and sampled32-check
rounds with fresh namespaces, then verifies ordered model IDs before opening
public18080. STOP/signals terminate the frontdoor and wait for backend teardown/
post-health while retaining leases. Restart=no is preserved.

Pending: final200K plan and reviewed inputs, completed100K/200K evidence,
independent final source review and eventual parent-controlled installation.
No production or full model-quality qualification is claimed by these tests.
