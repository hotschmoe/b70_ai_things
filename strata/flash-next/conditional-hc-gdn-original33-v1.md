# Conditional original HC/GDN consumers for33-field captures

CONFIG -> frozen first-layer consumer math and tolerances NMSE<=1e-6,
max_normalized<=1e-4, normalizer floor1e-6. Exact prefix1/2/4/8 candidate-on
frames, 31 actual source fields plus two explicitly DERIVED hidden fields.
All33 fields and actual producer/live-blob bindings are admitted by frozen
verify_layer0_ffn_original_v1.load_frame. The new checker reloads its required
HC/GDN bytes with exact SHA/size/encoding checks and verifies both actual GDN
Q8_1 packet encodings against their supplied F32 source bytes.

COMMAND -> python3 -m unittest discover -s strata/flash-next -p
 'test_verify_layer0_hc_gdn_original33_v1.py'

RESULT -> 20 CPU controls passed using bounded synthetic captures, receipt and
scalar-math fixtures. The existing GDN18 CPU controls also passed unchanged.
No original model payloads, GPU work or real-run arithmetic were executed.

VERDICT -> CPU source READY; actual commands below remain conditional on root
permission and terminal source/health gates. A new explicit33 admission path
calls frozen HC check and GDN conditional convolution/state/projection functions
directly. It does not filter33 frames to26 fields, override old admission or
change scalar equations/tolerances. Frozen26 admission remains unchanged and
rejects the synthetic33 frame.

The required finalized parent/child, actual engine receipt, capture-plan binding
and postidentity associations are checked before original GGUF consumption.
The original lazy provider checks publisher hashes, canonical five-element
current stats, pinned inventory/header/lock and original tensor byte extents.
That admission uses the supplied completed post-run identity; it does not
perform a new whole103GB hash scan. Source health recurrence checks and exclusion
of concurrent model/GPU activity remain parent responsibilities.

The admitted original HC roles are Q8_0 down/up and F32 norm/injection, exactly
for both layer0 attention and FFN halves. GDN uses original Q8_0 qkv/gate/out,
F32 convolution[4,10240] and F32 norm[128]. Q8_0 projections use verified supplied
Q8_1 d*codes; stored Q8_1 sum is unused and this rule is not generalized to affine
weights. All13 original roles retain source offsets, sizes, shapes and stats.

The verdict is conditional on supplied residuals, incoming physical recurrent
state[128,48,128], convolution history, normalized qkv, decay/beta, z and block
outputs. The state is mapped to logical[48,128,128] with transpose(1,0,2).
No independent incoming state, alpha/beta projection, upstream/prefill, FFN
consumer, complete layer/model, raw fused hidden or lifecycle proof is inferred.
DERIVED hidden validation does not establish raw fused-hidden equality.

Once root permits read-only original payload consumption, run each command
serially while no conflicting GPU/model activity is live. Use a new output path;
existing output evidence is never overwritten:

```sh
python3 strata/flash-next/verify_layer0_hc_gdn_original33_v1.py --run-root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f11-20261009/layer0-numerical-onecard-v8 --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f11-20261009/layer0-numerical-onecard-v8/conditional-original33-hc-gdn-v1.json
python3 strata/flash-next/verify_layer0_hc_gdn_original33_v1.py --run-root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f11-20261009/layer0-numerical-twocard-v8 --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f11-20261009/layer0-numerical-twocard-v8/conditional-original33-hc-gdn-v1.json
```

These commands consume child/candidatecombined_on/requests.json and the matching
post-model-identity.json within each supplied finalized run directory. Metadata
inspection established their schema/count and chronology shape; actual capture
binary fields and original weight blocks remain unevaluated by this CPU draft.
