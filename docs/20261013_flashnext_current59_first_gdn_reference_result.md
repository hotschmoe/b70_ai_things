# Current59 first-GDN independent reference observation

CONFIG -> Exact original UD-Q4_K_XL, current59 mode0, fresh prefix1 [248045].
Four admitted native observations; one independently computed original
embedding/HC/zero-state GDN. Native operands are comparison targets only.
Q8_0/Q8_1 integer half16 dot/F32 lane reduction and BF16 group8 FMA paths
are explicit source-expression candidates. Device intrinsics/contraction and
whole48 propagation are not qualified.

COMMAND -> Root CPU60623 run_current59_first_gdn_reference_v1.py, single BLAS
thread, after source61 profiling model exit. Actual original provider reads
closed before new postCPU pages/full4/pages identity. Root independently
verified all38 saved artifact hashes and identical comparisons across four
fresh native targets.

RESULT -> Report620a4ec5f3c8a1bf053afe8529658eb6e02a01a792d1ea403b0013cc6c32a504,
errors[], fresh closing full4 PASS. Report PASS covers execution/evidence
closure; numeric_pass_assigned=false and full_model_math_qualified=false.

| Field | Observed NMSE |
| --- | --- |
| Original embedding/residual input | 0, bitwise equal |
| HC mixed input | 1.6191e-14 |
| GDN QKV projection | 3.3934e-9 |
| GDN gated output | 5.2588e-9 |
| GDN block output | 4.9880e-7 |
| Residual after attention | 3.9610e-7 |

The input Q8_1 packet has80 blocks: all integer codes and stored sums agree,
but one half scale differs. Block59 has independent HC absmax7.206923961639404
versus native7.206924915313721, a difference9.5367431640625e-7 (two F32 ULP
at this magnitude). Its scales are0.056732177734375 and0.0567626953125.
Their half midpoint0.0567474365234375 lies between the two absmax/127 values.
This supports a quantization discontinuity from a small upstream HC difference;
it does not establish the remaining HC or compiler arithmetic contract.
The output packet has192 blocks:18 scales,49 stored sums and16 integer codes
differ, with maximum integer-code difference1.

VERDICT -> Current-mode own-state comparison now exists and localizes the
first scale divergence. No fitted tolerance, quantization patch, native-input
primary reference, full-model fidelity, speed or shelf claim follows. Continue
the HC/rounding contract audit and actual grouped/shared consumer reference.

Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/current59-own-first-GDN-wide32-reference-v1

Admitted native comparison component:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source59-mode0-fidelity-v3-recovered-component-v1.json
