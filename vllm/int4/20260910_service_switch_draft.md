# Candidate service restoration draft - NOT DEPLOYED OR QUALIFIED

CONFIG -> Read-only audit of stopped hotschmoe-dd.service on 2026-09-10. Candidate is phase plus MRV1 accepted-count image sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1, fresh calibrated FP8 artifact SHA256 be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062, TP2/P2P0/MTP3/FULL_DECODE_ONLY/prefix caching, requested context200000. TP2 100K/200K qualification is pending; no promotion or restart is authorized by this draft.
COMMAND -> Inspect nonsecret service properties, source launch chain, path ownership and readiness/validation code. No service changes, GPU calls, secret reads, deployment or process launch.
RESULT -> Active unit is /etc/systemd/system/hotschmoe-dd.service, root:root0644, parent root:root0755; neither is writable by hotschmoe. Service runs as hotschmoe:hotschmoe, Type=simple, Restart=no, KillMode=mixed, TimeoutStartSec9000, TimeoutStopSec900. It is enabled but inactive/dead, prior stop exit0. ExecStart is /usr/bin/python3 /mnt/vm_8tb/github/b70_ai_things/vllm/cache800k/serve_fp8_trial.py start; ExecStop uses the same wrapper with stop; ExecStartPost invokes vllm/cache800k/trial_ready.py ${MAINPID}. Environment keys are PATH, SERVED and B70_PRIOR_VALIDATION; values were not printed. No drop-ins currently exist.
RESULT -> Actual chain is systemd -> serve_fp8_trial.py -> bin/gpu-run (both leases) -> kv_campaign_server.py --leased -> Docker loopback18124; after startup validation, existing openai_key_frontdoor.py exposes0.0.0.0:18080. It reads /mnt/vm_8tb/b70/secrets/dd_api_key (hotschmoe:hotschmoe0600) without logging the value. Authorization accepts Bearer or X-API-Key with constant-time comparison; /health and /metrics remain public. Preserve these controls and existing network/firewall state. Backend remains bound127.0.0.1:18124. The current lifecycle pointer is /mnt/vm_8tb/b70/run/hotschmoe-dd-fp8-trial-current -> /mnt/vm_8tb/b70/results/qwen38_int4_fp8kv_trial/20260909T224703Z.
VERDICT -> A root-owned systemd override is needed to switch to a separate reviewed candidate wrapper without editing the existing trial or shared lifecycle files. Source/config/artifact preparation is user-writable and needs no sudo. Do not restart the old unit as a shortcut: it pins old image521e, old calibration artifact, old configuration and --health-p2p-check. Its startup full_feature_validate.py also hardcodes minimum800000 logical KV capacity and records180 tool fixture, which did not prove FP8 reuse. The current prior-validation environment references old evidence and must not be inherited by the candidate.

## Concrete candidate recipe to finalize after qualification

1. Freeze the successfully qualified200K Config.json/Mounts.json, model file manifest, source hashes, image ID and be02 scales into a new immutable candidate directory under /mnt/vm_8tb/b70. Preserve exact model mount models/files/qwen3.8-27b/int4-autoround-gptq-relabel-r212. Explicitly distinguish max-model-len200000 from actual logical KV pool capacity: do not set an artificial200K pool or carry the old800K gate. Keep the measured GPU memory utilization, max sequences, prefill budget and shutdown-timeout from the qualified200K configuration. Proposed30-second shutdown grace is eligible only if measured successfully.
2. Write a separate vllm/int4/serve_qwen38_mrv1fix_candidate.py wrapper following the existing parent-owned lease and finally/STOP/wait lifecycle. Its launch must refuse an absent or nonpassing qualification manifest, wrong image/config/mount/model/scales/source hashes, or missing100K/200K identity, numeric cache, concurrent coherence, teardown and health evidence. Do not adapt the old trial by merely changing its constants or relaxing its800K gate. Use a new result root/current pointer so historic trial evidence remains intact.
3. Candidate backend argv uses existing kv_campaign_server.py with --image sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1 --served-model hotschmoe-dd --served-alias qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276phase-mrv1fix-tp2-mtp3-fp8kv-freshcal-prefixon-ctx200k --tensor-parallel-size2 --p2p0 --mtp3 --kv-dtype fp8_e4m3 --hook load --scales <frozen-be02-file> --port18124 --health-probe vllm/int4/diagnostics/xpu_health_strict.sh --leased. Flags above are shown compactly in prose: actual argv must use separate flag/value elements. No --health-p2p-check, no P2P1, no new native binaries, no sampling overlay. Parent owns both GPU leases until backend teardown and strict post-health finish. Register the detailed alias before use; hotschmoe-dd remains first.
4. Before opening front18080, verify /v1/models stable+research identity and exact loaded manifest, calibrated layer coverage17perrank/34total with be02 hash, graph capture, resolved attention block/layout and actual capacity. Run the already qualified bounded startup fixtures through the leased job queue, including records360 concurrent tools and positive reported cache hits in each of four sessions. Reuse only frozen matching long-context evidence, never the old B70_PRIOR_VALIDATION path. If any check fails, keep the frontdoor closed and tear down under the lease.
5. Start the unmodified authenticated frontdoor only after the startup gate. Preserve FRONTDOOR_HOST0.0.0.0, PORT18080, BACKEND_URLhttp://127.0.0.1:18124 and API_KEY_FILE path. Keep API key values out of argv, copied manifests and logs. Retain the existing key file and authorization semantics. Test unauthenticated rejection and authenticated models/streaming using the existing frontdoor smoke tools without printing the key.
6. CPU-test the new wrapper before installation: refusal for missing/altered qualification, key-path existence only, primary alias ordering, P2P0, failed startup never exposing frontdoor, frontdoor failure stopping backend, and stop retaining the lease until post-health. The wrapper is not yet written because final200K configuration/evidence is pending. This draft is not an executable service candidate.

## Precise privileged switch after review

Create a reviewed drop-in at a user-writable path first. It must contain:

```ini
[Service]
UnsetEnvironment=B70_PRIOR_VALIDATION
ExecStart=
ExecStart=/usr/bin/python3 /mnt/vm_8tb/github/b70_ai_things/vllm/int4/serve_qwen38_mrv1fix_candidate.py start
ExecStop=
ExecStop=/usr/bin/python3 /mnt/vm_8tb/github/b70_ai_things/vllm/int4/serve_qwen38_mrv1fix_candidate.py stop
```

Retain ExecStartPost only if the new wrapper exposes /health exclusively after its own matching startup gate, as the existing trial_ready.py assumes. Retain Restart=no and stop timeout900; do not launch an automatic TP2 crash/restart loop. Once the concrete wrapper, qualification and drop-in are reviewed, the only required administrative writes/actions are:

```sh
sudo install -d -m0755 /etc/systemd/system/hotschmoe-dd.service.d
sudo install -m0644 <reviewed-user-owned-drop-in> /etc/systemd/system/hotschmoe-dd.service.d/20260910-mrv1fix.conf
sudo systemctl daemon-reload
sudo systemctl start hotschmoe-dd.service
```

These commands have NOT been executed. Root permissions are needed for the unit directory/file and system service control, not for reading or rotating API credentials. No key rotation is proposed. If candidate startup fails, stop it and preserve its owned recovery/health evidence. Removing the override restores the old configuration on disk but must not automatically restart the known-defective baseline. The enabled unit means an unintended reboot could still start the old configuration until the approved switch; this audit makes no change to that state.
