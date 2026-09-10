# Native5802 service migration preparation

CONFIG

Read-only inspection on2026-09-10. The installed hotschmoe-dd.service is inactive,
MainPID0, Restart=no, KillMode=mixed, stop timeout900seconds. Its active override
is /etc/systemd/system/hotschmoe-dd.service.d/20260910-day-trial.conf and points to
vllm/int4/day_trial/serve.py start --context200k. That launcher pins old image7b
and partial trial evidence. It must not be blindly restarted for this migration.
No service, public endpoint, GPU, secret value or old trial artifact was changed.

COMMAND

Inspect systemd unit text/status, day_trial/serve.py and inputs-200k.json,
service_candidate/serve.py, kv_campaign_server.py and openai_key_frontdoor.py.
Read only credential path references; never open or print the credential file.

RESULT

The existing public frontdoor listens on0.0.0.0:18080 and proxies to loopback
127.0.0.1:18124. It reads and strips the first line (rejecting an empty result) from
/mnt/vm_8tb/b70/secrets/dd_api_key. It accepts Authorization:Bearer or X-API-Key
using constant-time comparisons, strips those credentials before forwarding,
and currently exempts /health and /metrics plus OPTIONS. Preserve these exact
semantics; do not copy, rotate, log or embed the key in systemd/environment values.
The Docker backend port is published to127.0.0.1 only. Startup checks both stable
and research model IDs before opening the frontdoor.

The day-trial launcher takes both leases through bin/gpu-run, runs the server
with --leased, retains leases through teardown/post-health, and handles
SIGTERM/SIGINT/SIGHUP by writing its owned STOP marker. Its READY marker links
systemd's lease-shell MAINPID to the child owner and the owned frontdoor socket.
These lifecycle boundaries should be retained. Its prerequisite accepts the
old partial100K day trial and tiny-only startup, so changing only its image
constant is insufficient and would misrepresent the new qualification.

VERDICT / MINIMAL IMPLEMENTATION

Create a separate vllm/int4/native5802_service/serve.py derived from the existing
qualification-gated service_candidate/serve.py. Do not edit day_trial records
or reuse either old wrapper's qualification marker. Reuse the existing server,
frontdoor, lease and health helpers rather than introducing new lifecycle code.

1. Pin image
   sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067,
   native6717e3ec7fe3e7cdc26cd66b84aabdd076f211e5bed62b8711b7377fca8668d9,
   calibrated scale artifact
   be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062,
   and the current verified17-file model manifest. The source/driver/UMD/native
   identity ledger belongs in inputs. Reject the old7b image and legacy native
  271db0. Require the diagnostic conv-copy adapter flag absent/off; the rebuilt
   native implements the current rolling-state contract directly.
2. Bind the NEW complete native5802-backend100k/plan.json and forthcoming NEW
   native5802 backend200K plan, their frozen dependencies and backing output
   evidence. The100K plan being prepared uses tiny24, early3, three deterministic
   and three sampled records360 rounds (seeds42/43/44), and host trace. The200K
   plan must add its separately defined actual long-context, reuse, concurrency
   and recovery gates. Old partial trial/Pi manifests cannot satisfy this gate.
3. Require every mandatory job and semantic/reuse gate, actual executed image
   and configuration,34 unique rank/layer calibrated-scale receipts, paired
   profile entry/return review, strict pre/post both-card and compiled collective
   health, normal teardown, exit0 and owned-container removal. Hash the actual
   results/SSE/gate files, not only an overall marker. Verify model hashes and
   exact command equality between qualification and service, excluding only
   deliberate service port/name/output substitutions.
4. Preserve failed numeric-array evidence as a named limitation sidecar. The
   captured433-comma stop is also preferred by independent same-quant FP32 CPU
   scoring; this does not make its exact-array semantic check pass. Qualification
   should say backend_configuration_qualified with an explicit workload scope
   and known limitations, not unrestricted model-quality qualification. Never
   convert arbitrary future EOS, malformedJSON or degeneration into an allowed
   failure merely because this one case is explained.
5. Derive the service command from the qualified200K plan: TP2, P2P0, MTP3,
   FP8be02, prefixON, FULL_DECODE_ONLY, context200000, c4 and batch32768 if those
   exact settings are what the new arm qualifies. Require served-model-name
   starts with hotschmoe-dd and the final200K research alias is registered.
   Keep private18124/public18080, authenticated frontdoor and existing key path.
6. Use fresh run pointer/results under hotschmoe-dd-native5802-current and
   qwen38_native5802_service. Startup must verify the live model IDs, graph mode,
   all calibrated scales, new native identity and a frozen bounded coherent
   startup workload/reuse gate before opening the public frontdoor. Prior full
   qualification remains prerequisite. Choose startup jobs from the new plan;
   do not silently substitute the old trial's tiny-only path.
7. Keep PID/socket-owned READY, Restart=no and900-second systemd stop budget.
   STOP must close the frontdoor, drain the backend through existing lifecycle
   and retain leases until strict post-health completes. Preserve failures as
   failures; no automatic restart/reset retry loop.

SOURCE / GUARD TESTS REQUIRED BEFORE IMPLEMENTATION IS ACCEPTED

- Reject missing/new-incomplete100K or200K evidence, old7b/partial-day-trial
  markers, changed source/model/scale files, mismatched executed command and
  missing/duplicate rank-layer scale coverage.
- Reject wrong context, P2P1, MTP0, prefixOFF, eager, wrong graph mode, old native,
  enabled legacy copy adapter and reversed/missing stable model alias.
- Prove service command differs from qualified200K only by approved service
  port/name/output/lease adaptations. Confirm localhost-only backend binding.
- Keep limitation sidecar separate: the known array failure is retained and
  any new malformed output still fails mandatory semantic gates.
- Reuse readiness tests for stale markers, unrelated18080 listeners, exited
  MAINPID, wrong parent/socket ownership and startup identity/scale failure.
- CPU mock lifecycle tests verify STOP on signals, frontdoor termination,
  backend teardown completion before lease release, failure preservation and
  refusal to start public access before gates. Use a temporary test credential
  only for frontdoor auth tests; never the live key.

CONCRETE ADMINISTRATIVE RECIPE (PREPARED ONLY)

After both NEW full backend qualification arms and the wrapper checks pass:

1. Run the new wrapper's freeze/check commands against exact immutable inputs;
   review its recorded qualification scope and known limitation sidecar.
2. Verify hotschmoe-dd.service inactive and the current GPU test owners drained
   with required strict post-health. Do not start the old unit during this gap.
3. Install a NEW90-native5802-qualified.conf override, retaining the old day
   trial file as historical configuration. Its directives clear ExecStart,
   ExecStop and ExecStartPost, point all three to native5802_service/serve.py,
   unset B70_PRIOR_VALIDATION, and keep Restart=no/TimeoutStopSec900. Its lexical
   ordering must override the old20260910-day-trial.conf.
4. daemon-reload, inspect the effective ExecStart/ExecStop/ExecStartPost paths
   and settings, then start only the new checked wrapper. Startup's private
   readiness and semantic gates complete before public18080 opens.
5. Validate live first model ID,200K advertised context, authorized/unauthorized
   API behavior using the existing key without printing it, bounded coherent
   client response, and service-owned readiness. Record source/image/config/
   health identity. If migration fails, stop; never blindly fall back to7b.

No wrapper, override or qualification.json has been installed or activated by
this preparation. Exact new plan hashes and final200K alias are intentionally
pending the agents preparing those plans; their absence is a deployment gate,
not permission to reuse historical evidence.


NEW100K PLAN RECEIVED

/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/native5802-backend100k/plan.json
SHA256:50d977c22663f5139dd397832cd4adbc985021ef940f486effeddc2645faf7d8.
It has16 jobs covering24tiny,3early,192strict tool checks and final host trace.
Its frozen.json and known-quality-negative.json belong in the eventual service
inputs/evidence gate. Preparation is not execution success;200K remains pending.
