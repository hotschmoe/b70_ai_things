# User-authorized patched vLLM day trial

CONFIG -> User explicitly requested pausing research and serving patched vLLM for a day, selected200K context, and reiterated primary model hotschmoe-dd. This is a trial, NOT a fabricated full qualification pass. Image7b107d0e phase+MRV1 accepted-count fix, freshbe02 calibratedFP8, TP2/P2P0/MTP3/prefix/FULL_DECODE_ONLY, exact prepared200K model/config/mount settings. Ports/auth unchanged: authenticated0.0.0.0:18080 -> loopback18124, existing key-file path, hotschmoe-dd first plus registered detailed day-trial alias.
COMMAND -> python3 -m unittest discover -s vllm/int4/day_trial -p test_serve.py
RESULT -> Four CPU tests pass, including100K/200K config/primary alias/P2P0, no fake qualification claim, tiny-only startup and missing prerequisite refusal. No GPU/install/start by this preparation. User selected200K is explicit in the drop-in and default;100K is only an explicit fallback option.
VERDICT -> User-authorized day trial may proceed after current testing drains and required strict post-health completes. Full100K/200K/Pi quality qualification remains incomplete. Research resumes at next downtime. No additional long test or32-case round is queued at startup.

The wrapper requires already completed tiny24, early3 plus strict early review, threeT0/oneT0.7 records360 rounds with all128 strict tool checks, and the previous run's lifecycle0/strict pre/postbothcard+compiledpairhealth. It then owns both GPU leases, preserves existing strict server pre/posthealth, checks34freshcalibrated scale receipts, runs only tiny24 and verifies live stable+research aliases before opening the unchanged authenticated frontdoor. Metadata explicitly records user_authorized_day_trial=true, production_qualified=false, full_qualification_incomplete=true. No old qualification.json is consulted. Current source/config/scales hashes are pinned and checked.

After the previous run finishes:

```sh
python3 vllm/int4/day_trial/serve.py check --context 200k
```

Exact administrative switch (prepared, NOT executed):

```sh
sudo install -d -m0755 /etc/systemd/system/hotschmoe-dd.service.d
sudo install -m0644 /mnt/vm_8tb/github/b70_ai_things/vllm/int4/day_trial/20260910-day-trial.conf /etc/systemd/system/hotschmoe-dd.service.d/20260910-day-trial.conf
sudo systemctl daemon-reload
sudo systemctl start hotschmoe-dd.service
```

The unit override is root-owned, so these installation/service-control steps require sudo. The service runs as hotschmoe, keeps Restart=no, retains stop timeout900 and uses owned PID/socket readiness. Existing credentials are not copied, printed or rotated. Stop creates the owned STOP marker and waits for server teardown/post-health while the parent retains its leases. No active configuration, service or GPU was changed by preparation.
