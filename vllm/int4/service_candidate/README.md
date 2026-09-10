# Candidate service source - not installed or qualified

CONFIG -> Image7b107d0e phase+MRV1 fix, freshbe02 scales, exact prepared200K TP2/P2P0/MTP3/FP8/FULL_DECODE_ONLY/prefix configuration. The registered research alias is taken unchanged from the long200K plan; hotschmoe-dd stays primary. Original full Pi100K review failed malformed/garbled tool arguments; it is preserved and is not counted as a quality pass.
COMMAND -> python3 -m unittest discover -s vllm/int4/service_candidate -p test_serve.py
RESULT -> Six CPU tests pass: unqualified/missing workload/changed source/wrong image/unsafe pointer refusals and exact200K server-command equivalence except name, output and loopback port. No GPU or systemd actions run. inputs.json freezes source/config/scales/model-manifest hashes and the replacement clean100K prerequisite. qualification.json deliberately does not exist.
VERDICT -> Reviewable implementation only. Successful clean100K and200K evidence, checked startup/recovery behavior and final review remain prerequisites. Frozen criteria provide bounded qualification, not a claim that the original Pi replay or all production workloads are correct.

The wrapper holds both leases through existing strict preflight, serving, shutdown and post-health. It verifies model content against the current publisher-verified17-file manifest, hashes source/config/scales, and requires actual executed plan/config/image,34 calibrated attention records, every planned job, lifecycle0, per-card health, compiled pair health and no remaining-EngineCore force-kill in both qualification runs. It refuses old trial calibration and the stale prior-validation shortcut.

Commands, only after the required runs actually finish:

```sh
python3 vllm/int4/service_candidate/serve.py freeze
python3 vllm/int4/service_candidate/serve.py check
```

Freeze uses exclusive creation. It never runs GPU work or installs/starts the service. A source/config change requires a deliberate reviewed input refresh and requalification; do not edit hashes merely to silence a failure. No default is weakened. Start rechecks frozen evidence before obtaining the lease, then replays the exact200K job list on the new backend before opening the authenticated frontdoor. Those jobs use a new output tree and loopback18124. This can take substantially longer than a health-only startup, so the candidate drop-in has a scoped12000-second start timeout. Existing key-file authorization and public health/metrics behavior are retained without copying or logging the key.

Once the wrapper, frozen qualification and actual readiness have been reviewed, the administrative switch is:

```sh
sudo install -d -m0755 /etc/systemd/system/hotschmoe-dd.service.d
sudo install -m0644 vllm/int4/service_candidate/20260910-mrv1fix.conf /etc/systemd/system/hotschmoe-dd.service.d/20260910-mrv1fix.conf
sudo systemctl daemon-reload
sudo systemctl start hotschmoe-dd.service
```

NOT EXECUTED. The root-owned unit directory requires sudo; model/source preparation does not. No automatic restart loop is configured. A failed candidate must remain stopped with its teardown evidence retained; rollback must not automatically restart the known-defective old baseline. The old service remains enabled/inactive until an explicit switch is performed.
