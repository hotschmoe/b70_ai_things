# Same-AutoRound CPU reference preparation

CONFIG

Exact native5802 image d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067,
CPU only, no devices/network, 8 CPUs and 12 GiB hard memory limit. Model is the
live int4-autoround-gptq-relabel-r212 artifact. No complete model forward yet.
FP16-rounded dequantized weights are widened to FP32; activation and accumulation
precision therefore differ from FP16 serving even when weights match.

COMMAND

Run prepare_probe.py, then validate_layers.py, then test_streamed.py in the
bounded container. The preparation script requires --model, --bf16, --raw,
--source and --output. Layer validation requires --model, --source and --output.
Source is the exact installed Transformers modeling_qwen3_5.py, hash pinned in
streamed.py. Raw evidence is under:

/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/reference-feasibility/

RESULT

The bounded projection measured 255-285 GFLOP/s and 1.66 GiB peak RSS. All 64
text layers' names and shapes map exactly with no unhandled tensors. Real isolated
layers 0 (GDN) and 3 (full attention) load/dequantize in 5.40/5.09 seconds. Their
65-token outputs are finite, and changing tokens 32 onward leaves earlier output
exactly unchanged. Two forwards take 0.75/0.68 seconds. Peak RSS is 3.15 GiB;
FP32 parameter storage is 1.53/1.49 GB. These are isolated layers, not chained.
A separate dense versus blocked causal GQA fixture differs by at most 2.98e-8.

VERDICT

The loader and single-layer execution are feasible within the cap. A 6K full
pass remains unexecuted and needs source review plus explicit scheduling by the
parent. The rough dense arithmetic estimate is 18-20 minutes before recurrence,
attention and streaming overhead; it is not a measured full-pass runtime.

Implementation boundaries:

- streamed.py extracts exact original Torch model functions/classes and strips
  integration decorators, preserving static/class methods. No FLA/hub/XPU
  dispatch is allowed. Attention registry accepts only the explicit CPU blocked
  causal implementation. No padding, cache, batch>1 or dropout is accepted.
- Symmetric GPTQ nibble decoding uses zero eight, input-axis packing, group128,
  FP16 scales and FP16 weight rounding. Runtime ignores qzeros in this route;
  its shape remains validated. Other quantization layouts fail closed.
- Only one layer is materialized. Embeddings use selected rows; lm_head uses
  vocabulary chunks and at most16 selected positions. A small synthetic test
  validates these paths; full real vocabulary scoring is not yet executed.
- Exact canonical prompt IDs are 4045. Visible incomplete outputs re-encode to
  2054/2057 IDs against reported completions2055/2058. Original sampled IDs were
  not captured. The gap is consistent with a hidden stop token, not proof of ID.
- Full artifact identity must use the existing verified model manifest and
  pre/post immutable-file checks before an eventual full reference run. Current
  scripts validate index completeness, names, shapes and source identity; they
  do not themselves hash every large weight shard or authorize a full run.

## Full-length preparation checkpoint

The independent source review is retained in raw
reference-feasibility/INDEPENDENT_REVIEW.md. Its direct-weight dtype correction
is applied: dense BF16 tensors round through endpoint FP16 before widening;
A_log alone remains FP32. Exact GemmaRMSNorm source computes weight.float()+1,
so the plus-one operation stays FP32. Gated norm weight uses default FP16 dtype.
Earlier short-layer receipts predate this correction and remain unchanged.

CONFIG -> COMMAND -> RESULT -> VERDICT:

Same capped CPU container -> validate_long_layer.py for isolated layers0 and3
on actual 6102-token prefix embeddings, with600-second alarms -> GDN forward
14.68s and peak4.62GiB; attention forward15.12s and peak3.99GiB; both finite ->
full-length representative memory fits the cap. These are independent isolated
layers, not the actual later-layer hidden states or a whole-model pass.
Raw long-layer0-v1/result.json and long-layer3-v1/result.json retain receipts.
Three updated CPU tests pass, including the A_log/default-dtype distinction and
causal next-token indexing. Full vocabulary scoring remains unexecuted.

New GPU metadata independently captured the actual sampled tokens. They exactly
match the previously frozen canonical prefix:4045prompt+2057visibleoutput, followed
by token248046. The proposed full plan now uses these actual IDs and excludes
that terminal token. This resolves the re-encoding ambiguity for this current
case only; it does not reconstruct missing original Pi wire requests.

run_reference.py implements the full orchestrator but has NOT been executed.
It verifies full model hashes before and after computation, file stats, source,
input and tokenizer provenance; releases every layer; applies final RMSNorm;
and scores the last8 causal positions, including6098/6101, for EOS248046/248044
and the exact expected next token. launch_reference.py owns a no-device,
network-disabled container with8CPU/12GiB caps,3600-second deadline, actual image
and container inspect receipts, actual exit code and verified removal.

The immutable proposed snapshot is raw reference-feasibility/full-plan-v1.
Its PREPARED.json records exact invocation and hashes. It passed a no-execution
wrapper check and awaits parent authorization. The layer timings extrapolate to
about21minutes for64load/forward steps, excluding final head, hashes and other
overhead. FP32 compute differences remain explicit; no endpoint qualification
or EOS diagnosis is claimed before the reference actually runs.


CONFIG -> Frozen full-plan-v1, exact sampled6102-token prefix, same artifact
and reviewed dtype policy,8CPU/12GiB/no-device/no-network.
COMMAND -> Parent authorized and launched the exact frozen launch_reference.py
wrapper after both full-length layer checks and independent driver review.
RESULT -> Actual image/isolation inspect passes; initial sequential layer
progress is recorded under full-reference-v1/result/layers.jsonl.
VERDICT -> Full comparison is now running, not completed. Preserve the earlier
preparation evidence above; no EOS or whole-model verdict follows yet.

## Completed independent reference

CONFIG -> COMMAND -> RESULT -> VERDICT:

Parent authorized exact full-plan-v1 -> owned CPU wrapper ran the64-layer
teacher-forced reference -> all layers finite, full17-file pre/post hashes and
stats match, computation exit0,22.44minutes, peak4.88GiB -> independent numerical
result available; this is not endpoint or model-quality qualification.

At captured input length6102, immediately after `433,`, EOS248046 ranks first:
logit27.20612 versus expected space220 at26.93143, a0.27469-logit margin.
At length6099, after `432, 4`, expected digit3 ranks first, with EOS second and
only0.19073 logits behind. Thus this same-quant FP32 CPU implementation also
prefers stopping at the433-comma boundary, without executing vLLM or XPU model
kernels. Earlier stop-position differences are consistent with a narrow ranking
margin, but their precise numerical cause is not established. This result does
not distinguish original-model behavior from quantization effects.

Raw result: reference-feasibility/full-reference-v1/result/result.json.
Compact audit: reference-feasibility/full-reference-v1-review.json.
The original lifecycle receipt reported container_removed=false because the
frozen wrapper searched for uppercase `No such` while Docker returned lowercase
`no such object`. Independent exact-ID inspect verifies absence; correction is
recorded separately in full-reference-v1-independent-cleanup.json. Original
frozen sources and receipts remain unchanged. The tracked future wrapper now
uses a case-insensitive check; missing-container and daemon-error negative
fixtures pass. No extra model inference was performed for this correction.
