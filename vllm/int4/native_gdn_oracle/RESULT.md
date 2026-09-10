# Confirmed native/worker convolution-state contract mismatch

CONFIG

Measured image:
sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1.
This already contains the earlier GDN phase and MRV1 accepted-count fixes.
Installed native _xpu_C SHA256:
271db0d4882124e21ac6a4d080bfeab303fbb08b9ec10e11f21d10fb0723998f.
Worker mamba_utils SHA256:
936d65f7e9e85e0c67ada8210ce26108cf7868c4f991f76d0ad3bc125c07dc3a.

Single physicalcard0; TP1 localK16/V48,dim128; FP16projections/conv, FP32SSM,
MTPq4 and allocatedsix-historyconv. Actual FP8hybrid page3276800bytes, natural
conv122880+SSM3145728=3268608bytes. Physicalspecstate slots[7,3,9,2]. Production
expandable_segments allocator, pinned image/source, selectedlease, strictpre/
posthealth, freshoutput and ownedcleanup. No full model or TPcollective is
inside these numerical probes. The parent launched them; preparation was CPU-only.

COMMAND

The original rolling-contract oracle and the separate publication/read probe
were run through their frozen owned lifecycle plans. Raw evidence:

- bang_recurrence_20260910T172335Z/native-gdn-tp1-fp8-card0/
- bang_recurrence_20260910T172335Z/native-gdn-publication-card0/
- plan directories native-gdn-tp1-plan-card0/ and
  native-gdn-publication-plan-card0/ under the same runtime results root.

Both actual copy-oracle prerequisites passed: 12precopy cases with production
allocator and12targetedpostprocess cases including selfcopybias3, backwardcopy,
partial acceptance, accepted-count reset and wholeallocation preservation.

RESULT

The original native oracle FAILED atstep0 because conv_window_exact and
inactive_conv_exact werefalse. Its first-step independent math passed:
output maxabs4.76837e-7, SSM maxabs1.01258e-6. Packed-versus-hybrid output,
z, conv and SSM were all byte-exact. This failure remains preserved; its
expected rolling contract was not rewritten to make the installedbinary pass.

The diagnostic then directly observed allfour native prefixslots. Every slot
contains its corresponding chronological three-token convolution history in
rows0..2. Allfour slots' rows3..5 remain untouched. The native kernel therefore
publishes per-prefix three-row histories, not one rolling six-row history in
column0. The72rowidentity map and CPUtensor snapshots preserve exact writes.

Next-call read controls restore the same actual cache snapshot independently
before each accepted count, with identical nextinputs. The kernel follows the
per-prefix state column's firstthree rows. The worker rolling-window reference
fails for accepted2/3/4:

| Accepted | Per-prefix output relativeL2 | Rolling output relativeL2 | Per-prefix SSM relativeL2 | Rolling SSM relativeL2 |
| --- | --- | --- | --- | --- |
| 1 | 0.000014607 | 0.000014607 | 0.000002552 | 0.000002552 |
| 2 | 0.000032004 | 0.721966 | 0.000004983 | 0.274440 |
| 3 | 0.000005856 | 0.942327 | 0.000000113 | 0.419952 |
| 4 | 0.000030896 | 1.065170 | 0.000010082 | 0.523763 |

Accepted1 is intentionally a nondiscriminating control: both references use
column0's firstthree rows. Counts2/3/4 discriminate the contracts. Every output
and selectedSSM remained finite; z and pagepadding checks passed. Publication
probe lifecycleexit0 and strictpre/posthealth0 mean the observations completed
cleanly, not that the rollingcontract or backend passed.

Concrete producer/consumer disagreement

Let the history before nativeq4 be [h-2,h-1,h0], and new projectedQKV rows be
[x1,x2,x3,x4]. The native producer publishes:

| Speccolumn | Convrows0..2 | Convrows3..5 |
| --- | --- | --- |
| 0 | h-1,h0,x1 | untouched |
| 1 | h0,x1,x2 | untouched |
| 2 | x1,x2,x3 | untouched |
| 3 | x2,x3,x4 | untouched |

Native continuation with acceptedN selects columnN-1, rows0..2. That is
internally consistent until a worker copy applies a different representation.
The worker's get_conv_copy_spec / _copy_mamba_state_block consumes convolution
from sourceCOLUMN0 starting at ROWbias=accepted-1; temporalSSM instead comes
from sourceCOLUMN+bias. For accepted4 migration/publication, the worker thus
copies convolution rows3..5 that native left untouched, together with the
correct SSM from speculativecolumn3. It resets accepted to1 where appropriate,
so the next nativecall consumes this wrong convolution history as valid.

For accepted2/3 the selected window includes one/two untouched rows. This is
not a bytecopy implementation error: the bytecopy oracles proved that the copy
routines implement their own specification. The native producer and worker
consumer implement incompatible specifications. "Untouched" can mean stale
or uninitialized runtime data; the synthetic test initialized it deliberately,
so it does not establish the original runtime bytes' contents.

Source identity corroboration

The independent runtime-source audit reconstructed shipped kernel1e90ffa plus
R35/R50 patches. Steve's clean-clone replay record identifies the exactsame
271db0d4 native SHA from that source/build chain. It is retained under
bang_recurrence_testing_20260910/native-contract-provenance/. The shipped
causal_conv1d.hpp reads columnaccepted-1 firstthree rows (:688-724) and writes
per-prefix firstthree rows (:805-823), matching the measured binary.

Upstream5802a414d47855b01b63121bce3655795ef8dfa8 contains the token-indexed
convolution fix (#544) and stride-length fix (#545); it is absent from the
shipped1e90 source. This supplies a concrete source repair direction. The
previously inspected a397 compatible source already implemented the corrected
rolling representation and was not the shipped binary's source; its source
behavior must not be attributed to native271db0.

VERDICT

The incompatible native/worker convolution-state representation is confirmed
in the deployed patchedimage. It explains a concrete corruption mechanism
when MTPaccepted counts>1 reach cache-state copy/publication boundaries and
fits the observed model failures near multiples of block1600. It also explains
why direct nativecalls without state migration can look numerically correct,
why accepted1 controls agree, and why fixing the earlier phase/row-order bugs
was insufficient.

This is a confirmed contract defect, not yet a qualified fix or proof that
every historical bang has this cause. A source-only worker adapter or rebuilt
native fix must pass composed copy/native numerical tests, the unchanged
rolling-contract oracle when applicable, full-model boundary reproductions,
concurrent serving, cleanup and health before promotion. No fixed-image
correctness, speed or stability claim is made here.
