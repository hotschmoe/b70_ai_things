# First HC localization: stored activation scale boundary

CONFIG -> Original source35 NUM10 prefix1 targets, independently owned original
embedding/weights and frozen prior/new HC profiles. Separate conditional GDN
uses native HC mixed values with independently owned zero state; this is
explicit captured-input localization, not independent whole-model math.

COMMAND -> Tracked39fa4be explore_owned_first_hc_fidelity_v2.py with
--conditional-gdn-seam; actual73859 terminal0, errors[], final computation
timestamp after both HC and GDN, new completefour publisher hashes and pages.

RESULT -> Prior normalized10240 floats and full2880-byte Q8_1 input packet
match native BITWISE. Candidate normalized NMSE1.3274626896906163e-14 and
mixed NMSE1.5771766141996228e-14; candidate packet differs in exactlyone byte.

The difference is byte2124, the first scale-D byte in block59, not a code.
Candidate header432b56c4 versus native/prior442b56c4. All32 quantized codes and
stored sum-S are identical. Half scale-D is0.056732177734375 versus native
0.0567626953125. Their midpoint is0.0567474365234375. Candidate F32 scale
0.0567474327981472 falls below it; prior scale0.0567474439740181 falls above.
The maximum magnitude occurs at column1903: candidate7.206923961639404 versus
prior7.206925392150879. Small HC arithmetic differences therefore cross this
stored-scale rounding boundary. No backend contract defect is inferred.

Conditional native mixed -> original GDN output NMSE6.003467811343193e-15,
max normalized7.876110669436327e-8; still not BITWISE. Host expf and reciprocal
sqrtf remain unqualified device hypotheses; actual native RMS/rsqrt evidence
is required before changing any backend mathematics.

Report:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/owned-first-hc-fidelity-v2-conditional-run1/report.json

SHA076c1e1069de311ee3effa39988f3a78875d8d618804bedc404d31196cec8081.
Post-original model identity SHA2a22dade02def8d7f4048afe69eef3abf6143d3a2487a98bcc3cb41917456d55.

VERDICT -> Concrete first-HC stored-scale localization. This explains an early
packet disagreement; causal whole-prefix/head attribution remains unproved.
No tolerance/PASS assigned, no device/whole-model/cache/latency qualification.
Preserve both references and qualify the next RMS/rsqrt control separately.
