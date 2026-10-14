# Daily-driver service, repetition investigation, and source preservation

Date: 2026-09-07 UTC

## Service identity and boot configuration

CONFIG -> Live Qwen3.8 official FP8 W8A16, R187 whole-graph compile,
R156 image, MTP3, TP2, FP16 activations/KV, prefix caching enabled,
200000 maximum context, four sequences, 32768 batched tokens. Launcher:
`vllm/fp8/serve_qwen38_fp8_steve_r187_mtp3_daily.sh` in tmux session
`qwen38_mtp3_daily`. Container: `qwen38-fp8-steve-mtp3-daily-r187`.

COMMAND -> Inspect systemctl, container identity, live /v1/models, launcher,
and current logs; prepare `vllm/fp8/hotschmoe-dd.service` and its readiness,
stop, and installation helpers. Run bash syntax and systemd unit validation.

RESULT -> The enabled `/etc/systemd/system/b70-daily-driver.service` is
failed and still references the retired Qwen3.6 NVFP4 launch. The separate
FP8 shelf systemd template also references a different old image and is not
the current R187 launch. The prepared new unit uses the actual MTP3 script,
names the container hotschmoe-dd, and preserves the descriptive served ID:
`qwen3.8-27b-FP8-official-W8A16-mtp3-r187-xpugraph-cacheon-ctx200000-daily`.
No matching ID was found in `evals/configs/models.yaml`; no evaluation was
run or trusted during this investigation. Shell syntax and unit validation
passed; systemd printed unrelated existing-unit warnings. No GPU test or
runtime restart was performed. Root installation is pending: `sudo -n true`
requires interactive authentication; NOPASSWD grants cover GPU reset only.

VERDICT -> Prepared, not installed or boot-tested. From the repository run:

```bash
sudo bash vllm/fp8/install-hotschmoe-dd-systemd.sh
```

The installer backs up existing units, enables the new unit, then disables
the retired boot entry. It leaves the active tmux lifecycle alone. For live
cutover, drain clients, stop the tracked tmux launcher, wait for its post-card
and compiled collective health, and only then start:

```bash
sudo systemctl start hotschmoe-dd.service
systemctl status hotschmoe-dd.service --no-pager
```

Do not simply rename the running Docker container: the current shell retains
its old NAME and uses it for cleanup. Do not run both services simultaneously.
The new unit waits for exact backend identity and frontdoor health and waits
for the leased launcher's normal teardown before systemd kills remaining
processes. Interrupted startup and full shutdown remain live validation work.

## Repetition investigation

CONFIG -> Same uninterrupted lifecycle, started 2026-09-05. Evidence root:
`/mnt/vm_8tb/b70/results/qwen38_fp8_steve_mtp3_daily_r187/20260905T051117Z/`.
The public frontdoor is port 18080; backend is loopback port 18124.

COMMAND -> Read server.log, kernel journal, live metrics and Prometheus
preemption history at five-minute resolution. Read installed cache-manager
Python source using docker exec with shell tools only (no GPU execution).
Inspect primary upstream issue and pull-request reports.

RESULT -> Shared cache capacity is 292968 tokens, 1.46 times a full 200K
request. The lifetime preemption counter is 412. All increases occurred on
September 5 between approximately 06:16 and 13:21 UTC; none afterward through
the inspected September 7 snapshot. Large bursts occurred around 10:21-10:36
and 12:01-12:06. This does not establish a cause for a later episode.
No matching Xe fault/reset/OOM appeared in the inspected kernel journal.
The server has nine template errors: eight unsupported `minimal` reasoning
efforts and one unsupported `high`; this template accepts low, medium, xhigh.
Eight requests used the stale MTP5 model ID and received 404. These are
request errors, not evidence of the punctuation-generation cause.

Response text is not retained in server.log or the frontdoor access log.
Zero engine error/repetition finish counters do not demonstrate output
coherence. The local tokenizer maps `!` to token 0; that is a diagnostic clue,
not proof of NaNs. An episode timestamp or client transcript is still needed.

Upstream [issue 53912](https://github.com/vllm-project/vllm/issues/53912)
reports the same punctuation symptom with MTP plus prefix caching on a hybrid
GDN model, despite normal responses and server health. The installed
`MambaManager.find_longest_cache_hit` accepts but never uses
`drop_eagle_block`; the installed coordinator also excludes Mamba from its
extra lookahead margin. This matches the reported suspect code pattern.
This is a strong hypothesis, not a reproduced local causal result.

The proposed cache-read fix
[PR 48375](https://github.com/vllm-project/vllm/pull/48375) remains open and
unmerged at inspection. Our installed finder has an additional fine-grained
partial-hit branch; any port must cover that branch and its coordinator.
The scheduler-side [PR 51113](https://github.com/vllm-project/vllm/pull/51113)
is merged, but is a different fix. Do not assume a generic upgrade fixes the
reported cache-read issue. Also inspect the separate async accepted-count
ownership report [51571](https://github.com/vllm-project/vllm/issues/51571)
if request churn remains a trigger after isolating prefix reuse.

VERDICT -> First controlled mitigation experiment: MTP3 with prefix caching
disabled, otherwise matched configuration and clean lifecycle. Compare
repeated agent/tool turns and mixed-length concurrent requests, not just
short canaries. Expect slower repeated long-context prefill. MTP1 can reduce
speculation overhead and state allocation but retains the suspected feature
interaction. MTP0 is the separate speculation-off control. Lower admission
or compact contexts before the shared 293K-token budget to address the
independently observed pressure. Do not present these as measured fixes.
No live decoding parameter or patch was changed by this investigation.

## Steve update review

CONFIG -> Running source checkout is clean, full-history, pinned at
`8319e0964df12a1f0bc920301efc662ac49a949e`. Repository:
https://github.com/steveseguin/b70-optimization-lab

COMMAND -> Fetch origin/main without moving HEAD; diff the FP8 package and
review subsequent relevant experiment commits. Observed upstream snapshot:
`8121d831bc4f3410cf624937980d465eafc009b6`.

RESULT -> Four files changed in the FP8 repro directory: run-server.sh,
run-w8a16-mtp1-server.sh, run-w8a16-mtp1-strict-server.sh, and
verify-model-direct.sh. Changes parameterize quantization/model manifests,
forward INT4 determinism/GDN/runner/graph options, and allow overriding three
serial GDN flags whose defaults remain zero. The package image builders and
FP8 kernel chain are unchanged. Later experiments are predominantly INT4;
the R252 FP8 graph experiment reports no gain on Steve's measured shape.

VERDICT -> No reviewed Steve update is a demonstrated correction for this
FP8 repetition issue. Preserve the serving pin. Prioritize a deliberate
review/port of the upstream cache correctness work, with matched validation,
instead of importing the INT4 experiment defaults into this server.

## Preservation and independent reproduction process

Existing local assets already permit launching without contacting Steve:
the full source checkout, model weights, cached image, and launcher are here.
The pinned image ID is
`sha256:f46780e1a72c506248e3240eae1b470b39743dffbc17524c7248b9b3f63fb152`.
It occupies about 5.79 GB by Docker image accounting (archive size may differ).
Current model files total about 30.87 GB. A tag alone is not a backup.

The September 4 reconstruction used downloaded release binaries in the
R13 -> R15 -> R31 -> R49 -> R55C -> R62 -> R139 -> R156 chain. It was an
independent image reconstruction, not a local source compilation of every
native library. A full Git clone does not include release assets or images.

The concrete preservation scope is:

1. Save the exact working image using `docker image save`, record its image
   ID and archive SHA-256, and test `docker image load` on a separate Docker
   store. Save the pinned base image as well:
   `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
   Preserve any intermediate images used as build inputs.
2. Mirror the full lab repository with `git clone --mirror`, fetch all tags,
   create a `git bundle create ... --all` backup, and verify it with
   `git bundle verify`. Keep the exact running commit as a retained ref.
   Preserve submodule objects and LFS objects separately if present. Do not
   use shallow/partial clones as the only offline copy.
3. Copy and hash the assets for releases
   `qwen38-fp8-kernel-1e90-20260825`,
   `qwen38-fp8-tp2-r55c-20260901`, and
   `qwen38-fp8-tp2-r139-20260902`, including wheels, libraries, manifests,
   patches, and build logs. Verify against publication-manifest.json. The
   existing build tree is `/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260904-r187/`;
   inventory it before redownloading. Keep binary backups tied to this ABI.
4. Mirror the actual dependency repositories and pinned commits:
   vllm-project/vllm at `ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`,
   vllm-project/vllm-xpu-kernels at
   `1e90ffa672ba02f17a909da11838a4c55b199783`,
   uxlfoundation/oneDNN at `0e2a5bfeef1bfbffc3137464606540233086ce9b`,
   intel/sycl-tla at `cd763790ad2f74d7294435ecf77682bac0062c3a`.
   Recursively resolve build-time submodules/FetchContent and package inputs;
   these four pins alone do not establish complete offline closure.
5. Preserve the Intel oneAPI 2026.1 compiler/toolchain and exact host/container
   package inventories, wheel/deb caches, Dockerfiles, patches, licenses,
   weight/tokenizer/chat-template files and hashes. Back up the off-repo API
   key through private backup handling, never through Git or public logs.
6. Port builders to explicitly selected local source mirrors and local asset
   directories. Several builders hardcode GitHub URLs; copying the files
   without changing their resolution will still attempt network access.
   Preserve original source history and add local changes as tracked commits.
7. Run source-closure checks, rebuild in an isolated clean build root with
   external network unavailable, and inventory any missing input. Even the
   named full-source stack helper starts from a released R13 wheel, so source
   compilation of that initial stage must be included for an all-source claim.
   The R139 source builder uses host oneAPI 2026.1 and bmg-g21-a0 AOT. Compare
   installed library/content hashes, not independently generated image IDs.
8. Test restored launch plus matched identity, deterministic coherence,
   long-context concurrent agent traffic, performance, teardown, and post-health
   under bin/gpu-run. Store a second backup on another physical host/device.

VERDICT -> Immediate serving independence largely exists on this host;
durable disaster recovery and proven offline source-build closure do not yet.
No preservation archive or offline rebuild was created in this review.
Saving the exact image protects restoration of the current implementation;
equivalent performance still requires matched host/runtime and qualification.

## Stable client model alias (2026-09-07 22:10 UTC)

CONFIG -> User explicitly requests the model ID `hotschmoe-dd` to survive
physical model changes across Pi/OMP installations. This supersedes the
repository's descriptive served-ID convention for this client endpoint;
the launcher's config still records the pinned model/source/image and shape.
Installed `hotschmoe-dd.service` is enabled and directly executes the editable
MTP3 launcher, with only `NAME=hotschmoe-dd` in its environment.

COMMAND -> Change the launcher SERVED default and readiness helper default to
`hotschmoe-dd`; make readiness compare against SERVED; add an explicit
`Environment=SERVED=hotschmoe-dd` to the source systemd unit. Update installer
instructions to cover an already-running systemd service. Run bash syntax,
launcher --print-config, systemd-analyze verify, authenticated public models
and health requests, and a noninteractive systemctl restart attempt.

RESULT -> Static checks passed (systemd reported unrelated installed-unit
warnings). Public /health and /v1/models return HTTP 200. The running process
still advertises the previous long Qwen ID. Both sudo and systemctl require
interactive authentication, so the restart did not occur. The installed unit
has no SERVED override and will use the edited default on its next boot/start.
No inference, GPU workload, runtime change, or teardown was performed.

VERDICT -> Boot launcher and readiness are prepared for the stable alias;
live activation and live alias verification await the following privileged
commands. The installer backs up existing units and keeps serving running;
restart uses the normal leased launcher and its health/teardown lifecycle.

```bash
sudo bash /mnt/vm_8tb/github/b70_ai_things/vllm/fp8/install-hotschmoe-dd-systemd.sh
sudo systemctl restart hotschmoe-dd.service
```

After readiness completes, clients use base URL `http://192.168.10.5:18080/v1`
and model ID `hotschmoe-dd`, with the existing API key. Future backend swaps
must retain `SERVED=hotschmoe-dd` in the unit and pass it as the backend served
model name. A changing physical model may still require client capability or
context-limit updates. No fresh stability/performance qualification is claimed.

## Stable model alias live verification (2026-09-07 22:18 UTC)

CONFIG -> Same pinned Qwen3.8 FP8 R187 MTP3 service, stable served ID
`hotschmoe-dd`. A new lifecycle was observed starting at 22:11:58 UTC under
leased systemd MainPID 17224. Evidence root:
`/mnt/vm_8tb/b70/results/qwen38_fp8_steve_mtp3_daily_r187/20260907T221158Z/`.

COMMAND -> Read systemd state and lifecycle logs while startup completed;
query backend models/OpenAPI and authenticated public /health and /v1/models.
No further restart or inference request was issued for this verification.

RESULT -> Startup pre-card/compiled collective health passed; weight integrity
verification completed, followed by model initialization and kernel warmup.
Service entered active/running at 22:17:23 UTC and remains enabled for boot.
Both public endpoints returned HTTP 200 and public/backend model lists
advertise exactly `hotschmoe-dd`. The installed unit environment still contains
only NAME, so its referenced launcher uses the updated SERVED default; the
source unit additionally pins SERVED explicitly for future installations.
Live OpenAPI includes `cache_salt` in both ChatCompletionRequest and
CompletionRequest: nullable string, minimum length 1, documented prefix-cache
salting. This verifies schema support, not end-to-end cache recovery behavior.

VERDICT -> Stable client model ID is now live and persists through the current
boot launcher. Pi/OMP clients can use `http://192.168.10.5:18080/v1`, model
`hotschmoe-dd`, and the existing API key. No inference/coherence, performance,
or repetition-recovery qualification is claimed by these read-only checks.
