CONFIG -> Localize genuine original48 HC candidate regression. Same P30 prefix4
native targets: old p0L0attention NMSE1.3210388e-13, new3.9609885e-7;
old head .00932149853, new .01330947575. These are exploratory observations,
not thresholds. First row input is bitwise equal in both original computations.
Source35 verify.cpp986-1018 calls nativeHC one row per T2 row; do not invent a
separate batched HC arithmetic route to explain this difference. GDN T2
recurrent arithmetic remains a separate window seam.

COMMAND -> Root-only firstHC payload computation:

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 strata/flash-next/explore_owned_first_hc_fidelity_v1.py --native-run-root NUM10_FINAL --model-identity NUM10_POST_FULL4 --bulk-build-root /mnt/vm_8tb/b70/build/hc35-host-bulk-fma-v1-20261010 --output NEW_OUTPUT

Optional --conditional-gdn-seam feeds only actual native prefix1 HC mixed to
an explicitly conditional original GDN with owned zero state. This captured
input lane is labeled separately and never enters independent original48.
It does not establish native/model arithmetic authority. Original GDN remains
unchanged candidate math; its mismatch can include non-HC arithmetic seams.

RESULT -> Pending root computation. Produce complete owned firstHC normalized,
down, low SiLU, gate, inject and mixed vectors, full owned Q8_1 packet, and
compare both frozen prior HC and refined candidate to NUM10 prefix1 actual
fields. Actual native residual input must equal original decoded embedding
broadcast; native fields otherwise are output targets only. Prefix1 T1 target
has the same initial token as prefix4 first row, but cannot qualify T2 GDN.
Readonly trajectory helper compares all577 saved prior/candidate phases against
identical captured native target bindings, preserving each original report.

VERDICT -> Find first arithmetic/packet mismatch, do not tune tolerances.
No concrete new source-contract defect has yet been demonstrated. Host expf
and sqrtf reciprocal remain explicit candidates. Producer retains NUM10 final
source/metadata/raw admission, source closure before original provider, current
bulk prerequisite before/after, model identity/pages before/after and NEW
postCPU full4 scan. No GPU/backend/model execution by the source agent.
