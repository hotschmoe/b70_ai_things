# S85 private Q6 byte-centering records (2026-10-15)

Exact SMALL records; record-copies.json binds original/resolved paths, bytes
and SHA256. Large148binary artifacts13343616B, executable/object/image wrappers,
archives, complete ASM, model slices and full linker logs remain external.
Do not execute these copies in place: absolute paths point to completed raw
runs and this directory supplies no executable or fresh root build closure.

CONFIG -> Two real Q6 weight slices: blk.3.attn_output K6144/M2560 and
blk.4.attn_qkv K2560/M10240; C6/RPW1/A2/SG32, exact S44 saved canonical Q8
initial/changed inputs. Activations were synthetic, not captured production;
FP32 buffer is a read-only witness, NOT kernel input. No new quantizer, layout,
expanded plane, global dispatch policy or model substitution.

COMMAND -> Existing six-bit byteu is0..63. Packed (u+0x60606060)^0x80808080
matches subtract32 in each byte without carry/saturation. Private original
retains native load/dot/FP statement/reduction ledger; candidate changes only
centering expressions. V1 CPU build failed16s on missing strata::dp4a include;
v2 adds strata/sycl_math.hpp plus versioned paths/pins, leaf unchanged. Exact
textual delta and original failure are retained, no failed GPU retry.
V2 offline build24s reuses frozen common8f and native/core archives; no native
main TU/common AOT rebuild. ISA audit11s indexes only new adapter targets and
binds both whole target images bytewise into actual final executable. Source
and actual compiler/build hashes retained; final binary
 a0a1756b2670422caab13da80cc7e45a7beabd1fc694d684b0e34789b80112e8.

RESULT -> Static original32 word-minus32/32overflow-mask/32compare sites ->
candidate0, with8packedadd+8XOR sites. Both48staticDP4A, no call sites, SIMD32/
GRF128. Text16576->10880B; scratch/spill fields absent, not measuredzero or
occupancy. Source/ISA reports retain historical runtime-pending language;
subsequent runtime comparison and root closure below close those gates.

One physical-card0 CPUquiet window41s/leaf12s. Public matches S44, private
original matches public, candidate fullbits match for initial/changed/restore/
repeat and own graphs. All146strict snapshots,148binaries13343616B,10DOTs/
86nodes/76edges, guards/tails/read-only/source+fixture pins PASS. Comparison
explicitly validates public/single/batch20 DOT names and pointer resources;
reviewer statements that public DOT was unvalidated are corrected by root.
Pre/post health0, full ownCID+NAME absent, wrapper exits0, one process/no retry.

Three warm20-call graphs/arm, then8OCCO/COOC rounds/shape, four submissions per
round. Completed host wall/20calls is primary; reset/reads and canonical Q8
quantization outside timed graphs. All outputs checked. Raw64timing samples
and16nested paired means remain in leaf.log/comparison/root-runtime-decision.

| Shape | Original median us/call | Candidate | Median paired change |
| --- | ---: | ---: | ---: |
| out | 63.1721 | 48.7256 | -22.7841% |
| qkv | 73.8172 | 57.4807 | -22.0951% |

All16nested paired means improve. This is two shapes, repeated-weight primitive
regime, one process; no significance/p95, isolated ALU cost, physical cache
residency, bandwidth bound, occupancy, end-to-end speed or serving claim.
Source actual Opus40.131s/Fable74.111s and post Opus15.159s/Fable27.485s are
preserved with responding IDs/session metadata and root dispositions. Models
read supplied packets with tools disabled; no independent raw-file audit.
Source consultation also closed S84 private-long-lead disposition, copied here
as root-s84-postdecision.json without modifying completed S84 evidence.

VERDICT -> Root accepts SMALL default-OFF source preparation only for exact
C6/RPW1/A2/SG32/PF0/two tested shapes with unchanged fallbacks. No production
integration, serving run or promotion follows from this primitive. Later matched
full-model comparison must use BOTH-card native EP, not one-card/substitute
runs suggested by reviewers. No extra streaming micro, format/knob sweep or
full-cache recovery rerun is selected here. Actual S86 work is separate.

Completed whole evidence323files86839920B, manifest SHA
 dfb4b31b54bc8b43fdd3da04de8c5636721e8a2aef2630c1d373a6c397d3111f.
Raw root:
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/
with q6-center-s85-prep[-v2]/, q6-center-s85-isa/,
q6-center-source-reviews-s85-20261015/, q6-center-postreviews-s85-20261015/
and q6-center-s85-runtime-decision.json.
