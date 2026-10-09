# Native HC composition numerical fixture preregistration

CONFIG -> Source patches 0001 and 0002; original Q8_0 down/up and F32
norm/injection; selected geometry D=10240, N=2560, H=4, L=320. No model
callsites are qualified by this fixture. GPU execution remains a leased parent
operation and requires per-card health and teardown evidence.

COMMAND -> Build native_hc_composition_gpu_fixture.cpp with the two patched
SYCL translation units, overlay headers first, in runtime image39992d70.
The optional sole argument is a JSONL receipt path. The fixture uses a GPU
selector and an in-order queue; pin its device with the parent's lease and
Level Zero selector. A CPU_REFERENCE_ONLY build is available with
-DHC_FIXTURE_CPU_ONLY and the composition header; it performs no SYCL discovery.

RESULT -> Preregistered coverage is48 cases: tokens1/2/4/8, pending write
absent/out-of-place/in-place, injection absent/present, and mixed/epsilon-sensitive
inputs. This yields344 stage rows: six unconditional stages, pending_R when
apply=true, and injection when a source injection tensor exists.

The independent CPU reference uses FP64 accumulation and exact FP16-scale
signed-code Q8_0 decoding. All input/norm/injection source values are the original
F32 values. The reference applies the following equations without BF16 storage:

- pending_R = R + bo_prev * 2 * sigmoid(inj_prev /4), when apply=true;
- rs per stream = 1 / sqrt(mean(pending_R^2) + original F32 eps);
- xn = pending_R * norm * rs, separately for each2560-element stream;
- down_pre_silu = source_Q8_down * xn;
- lo_post_silu = SiLU(down_pre_silu /4);
- up_raw_gate = source_Q8_up * lo_post_silu;
- injection = source_F32_injection * xn, when present;
- mixed = mean over4 streams of xn * sigmoid(up_raw_gate).

Each exposed stage must be finite and meet NMSE <=1e-6 and normalized maximum
error <=1e-4, where the latter is max_abs_error / max(1e-6,max_abs_reference).
These gates deliberately retain the primitive tolerances for complete composed
stages, with no tolerance increase based on observed GPU results. The two input
profiles use finite bounded activations and nonsaturating/saturating gate values,
and the epsilon-sensitive profile puts residual magnitudes near the RMS epsilon
scale. The gates measure numerical agreement with continuous FP64 source math;
they do not require bitwise equality with CPU serial summation. The pre-SiLU down
projection is recomputed independently on the composed xn because the composed
workspace exposes only post-SiLU lo.

Every stage has a host negative control perturbing one output by1 percent of
max(1e-6,max_abs_reference); accepting this perturbation aborts the fixture.
The48 cases also require exact output-byte repeats after an independent buffer
set intervenes, exact multi-token versus concatenated single-token output bytes,
128-byte guards on every allocation, unchanged original tensor and input bytes,
and numerical validation of the explicitly permitted in-place residual write.
In-place repeat and single-token runs restore original residual inputs first.
The comparison includes every exposed intermediate and pending residual image.

VERDICT -> CPU-only build and reference round-trip validation pass48 cases and
344 stage/negative-control checks. This is fixture-construction evidence only;
CPU mode reports CPU_REFERENCE_ONLY and supplies reference-rounded outputs.
No GPU arithmetic, composed performance, model coherence, callsite coverage,
request-state isolation, two-device behavior, or serving qualification is proven.
A GPU pass must report GPU_COMPOSITION_NUMERICAL,48 cases,344 stages, zero failed
cases and344 numeric negative controls. Parent receipt validation must confirm
complete unique case/stage coverage and every numerical/repeat/guard/source gate.

## Parent GPU follow-up

CONFIG -> Patches0001+0002, same pinned source/runtime; synthetic fixtures,
no model routes. COMMAND -> run_native_q8_hc_gpu.py --composition, after
passing projection receipt. RESULT ->48/48 cases and344/344 intermediate and
negative-control checks per card pass; exact repeat/batch bytes, guards and
unchanged sources pass. Worst NMSE2.8520817958328582e-11 and normalized maximum
error3.533520650739953e-5, under frozen1e-6/1e-4 gates. Normal exit/removal,
strict per-card and compiled P2P0 pre/post-health pass; no selected GPU faults.
VERDICT -> Synthetic composed math qualified on each card; model source
upload, complete routes, shared write, history, split/concurrency/latency remain
unqualified. Receipt:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009/native-hc-composition-v1/receipt.json.
