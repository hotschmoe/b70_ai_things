# Candidate service source - not installed or qualified

CONFIG -> Image7b107d0e phase+MRV1 fix, freshbe02 scales, exact prepared200K TP2/P2P0/MTP3/FP8/FULL_DECODE_ONLY/prefix configuration. The registered research alias is taken unchanged from the long200K plan; hotschmoe-dd stays primary. Original full Pi100K review failed malformed/garbled tool arguments; it is preserved and is not counted as a quality pass.
COMMAND -> python3 -m unittest discover -s vllm/int4/service_candidate -p test_serve.py
RESULT -> Eight CPU tests pass: unqualified/missing workload/changed source/wrong image/unsafe pointer and unowned frontdoor refusals, bounded startup job/namespace selection and exact200K server-command equivalence except name, output and loopback port. No GPU or systemd actions run. inputs.json freezes source/config/scales/model-manifest hashes and the replacement clean100K prerequisite. qualification.json deliberately does not exist.
VERDICT -> Reviewable implementation only. Successful clean100K and200K evidence, checked startup/recovery behavior and final review remain prerequisites. Frozen criteria provide bounded qualification, not a claim that the original Pi replay or all production workloads are correct.

The wrapper holds both leases through existing strict preflight, serving, shutdown and post-health. It verifies model content against the current publisher-verified17-file manifest, hashes source/config/scales, and requires actual executed plan/config/image,34 calibrated attention records, every planned job, lifecycle0, per-card health, compiled pair health and no remaining-EngineCore force-kill in both qualification runs. It refuses old trial calibration and the stale prior-validation shortcut.

Commands, only after the required runs actually finish:

```sh
python3 vllm/int4/service_candidate/serve.py freeze
python3 vllm/int4/service_candidate/serve.py check
```

Freeze uses exclusive creation. It never runs GPU work or installs/starts the service. A source/config change requires a deliberate reviewed input refresh and requalification; do not edit hashes merely to silence a failure. No default is weakened. Start rechecks the complete frozen evidence before obtaining the lease, then runs pinned clean100K tiny24 plus one greedy and one temperature0.7 records360/four-session/four-turn round on the new backend. Each round has a distinct fresh cache namespace and its pinned strict gate requires all32 exact tool/answer checks plus positive reported cache reuse in every session. These five jobs use a new output tree and loopback18124 before opening the authenticated frontdoor. The serving configuration remains exactly the qualified200K configuration. The conservative scoped12000-second start timeout remains unchanged; no restart latency is claimed. Existing key-file authorization and public health/metrics behavior are retained without copying or logging the key.

Once the wrapper, frozen qualification and actual readiness have been reviewed, the administrative switch is:

```sh
sudo install -d -m0755 /etc/systemd/system/hotschmoe-dd.service.d
sudo install -m0644 vllm/int4/service_candidate/20260910-mrv1fix.conf /etc/systemd/system/hotschmoe-dd.service.d/20260910-mrv1fix.conf
sudo systemctl daemon-reload
sudo systemctl start hotschmoe-dd.service
```

NOT EXECUTED. The root-owned unit directory requires sudo; model/source preparation does not. No automatic restart loop is configured. A failed candidate must remain stopped with its teardown evidence retained; rollback must not automatically restart the known-defective old baseline. The old service remains enabled/inactive until an explicit switch is performed.

Independent source review tightened three gates: both nested qualification manifests now validate every file/external dependency; evidence hashes include actual decision/summary/results/SSE files; readiness requires the current main PID and the frontdoor child owning the listening18080 socket. A pre-existing unrelated health endpoint cannot satisfy readiness.

CONFIG -> Bounded restart workload replaces repeated long-context testing; full successful clean100K and200K frozen qualification/hash/model/lifecycle/health requirements remain unchanged.
COMMAND -> Same CPU unittest command above; inspect generated startup job argv against pinned clean100K plan.
RESULT -> Eight CPU tests pass. Startup selects tiny24, T0 records360 and T0.7 records360, with a strict32-case semantics/positive-hit gate after each round. It preserves the registered alias,34 calibrated scale-load gate, P2P0,200K serving configuration and owned frontdoor readiness. No GPU, qualification freeze, installation or service start performed.
VERDICT -> Restart checks no longer repeat185K-context retrieval/guides/cancellation; those remain mandatory prior qualification evidence. Passing this bounded restart gate does not re-establish long-context correctness or resolve the separate Pi garble investigation. No direct restoration until the investigation and all required quality gates are reviewed.
