# vLLM localmaxxing preparation (X4)

Default operation is a dry-run plan. No script contains a submission command.
Ranking comparisons use no-spec runs; MTP runs are separate diagnostics.

Canonical reasoning-v1 and code-v1 text/hashes were fetched with lmx context
on 2026-10-20 and match the Strata payloads. Files have no added newline.
CLI inspected: v0.1.54, commit 8bb36c3eb8fcd95d363beb44725d1992e80fe404.
hardware_card1.json describes a one-card proxy; hardware_pair.json describes
the two-card Flash-Next layout. ramGb is nominal installed RAM, as in the
Strata submissions; Linux reports about 121.4 GiB usable.

From the repository root, prepare the exact server launch command in a text
file. It must include the model, quantization, dtype, graphs, kernel library
and plugin settings. Pass that file with --command-file. A dirty fork needs
its source/config hashes retained beside the run; a Git commit alone does
not identify these campaign builds.

Example (no HTTP requests and no speed measurements):

```bash
L=vllm/flashnext/localmaxxing
B=/mnt/vm_8tb/b70/build/vllm-flashnext/x4
"$L/run_dry.sh" --base-url http://127.0.0.1:18125 \
  --served-model hotschmoe-dd --alias qwen3.6-35b-a3b-ud-iq3_xxs-x5-mtp0 \
  --hf-id unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
  --quantization Unsloth-Dynamic-IQ3_XXS --hardware "$L/hardware_card1.json" \
  --engine-version 0.32.1rc1.dev94+g5fd62688f.xpu \
  --engine-commit 0a6170da8ca9991972322fa1ca1d27674f0e75cd \
  --engine-build b70-flashnext-shared-dirty-plan \
  --command-file "$B/proxy-launch-plan.txt" --mode no-spec --out "$B/plans"
```

For MTP use --mode mtp --spec-tokens 2, the mtp2 alias and the MTP launch
command file. These flags describe an already configured server: a client
flag cannot switch MTP on or off. Use --suite to select one canonical
prompt; otherwise both run. For Flash-Next select the actual research alias,
ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ3_XXS and hardware_pair.json.
Do not relabel the proxy as Flash-Next.

The future local measurement command is python3 "$L/lmx_vllm.py" with the
same arguments and --measure-local instead of --dry-run. It never submits.
Only the orchestrator should execute it in an agreed exclusive request
window, under bin/cpu-run --quiet. The server keeps its existing GPU lease;
the HTTP client does not acquire a second one. No measurement was executed
by X4: this campaign stage is dry-run only.

The local collector records one warmup and three measured streams per
suite, all at temperature 0, with a new leading nonce. Thinking uses the
server default, and reasoning text counts as output. Stream usage is
authoritative. There is no character-based or chunk-count token estimate.
TTFT starts at HTTP dispatch and ends at the first nonempty reasoning or
content delta. Decode tok/s = (completion_tokens - 1) / (last content arrival
- first content arrival), matching lmx and Strata lmx_verified.py. MTP can
emit several tokens per chunk: chunk intervals are saved separately and
must not be reported as per-token ITL. Effective ITL is 1000 / decode tok/s.

payload.json selects the median decode-rate iteration and keeps that
iteration's correlated TTFT, hashes, usage and acceptance, like Strata's
verified client. validation.json also records the CLI's independently
computed metric medians, which can select a different TTFT iteration.
No engine throughput or engine prefill timing is inferred from HTTP time.

vLLM normally omits Strata's per-request timings. engineTimingsRaw contains
actual server usage, optional native timings (null when absent), and
isolated Prometheus deltas with explicit provenance. Full request bodies,
stream events/arrival offsets and metrics snapshots live beside the payload.
Global counters are attributed only if success delta = 1, generated-token
delta = usage, busy gauges are zero, no counter resets occur, and no cache
hits occur. Export lag is polled outside the measured window (20 s bound).
MTP acceptance is accepted_tokens / draft_tokens; mean accepted length is
1 + accepted_tokens / num_drafts. Live /server_info structured config must
agree with the declared speculative method/depth; no-spec requires null.
Identity and config are rechecked after each suite. These checks detect
contamination; they do not replace an agreed exclusive request window.
The declared engine version must match /version; do not assume that wheel
metadata equals the imported fork version. Config credential fields are
redacted and the server environment is omitted from stored evidence.

After a real local measurement, validate without submission:

```bash
"$L/validate_dry.sh" /path/to/no-spec-reasoning-v1/payload.json
```

This requires validation.json valid=true, runs lmx validate-local, then
calls only lmx speed-test dry-run. Authentication stays in lmx's saved
configuration. X4 did not read, copy or print the key. Actual LocalMaxxing
verification/approval remains untested until measured payloads exist.

See ../reports/1X4_gate_lmx.md for commands and observed gate results.
