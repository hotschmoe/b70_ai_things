# Original48 HC FMA refinement, prefix4 result

CONFIG -> Pinned original UD-Q4_K_XL, source35 onecard P30V4 native comparison,
independently owned prefix4 inputs/state. NEW HC-only source profile uses
source32-lane square/projection FMA and XOR reductions, F32 stores, ordered
fused mixing and explicit fused residual writes. GDN/QSA/FFN/PLE/head/state
evolution remain inherited. Host expf and sqrtf-reciprocal are candidates,
not qualified device intrinsics. Native arrays are comparison targets only.

COMMAND -> Tracked f1c58d7 runner explore_full48_original_hc_fma_v1.py, prefix4,
qualified bulk root /mnt/vm_8tb/b70/build/hc35-host-bulk-fma-v1-20261010.
Actual session51315 terminal0; new complete4 publisher hashes and known pages.
Root readonly12605 independently recomputed every saved comparison and checked
post-source identity; original output tree SHA/stat5 unchanged.

RESULT -> Exploratory execution complete, errors[], all577 comparisons.

| Metric | Original route reference V2 | HC-only refinement V1 |
| --- | --- | --- |
| Full-head NMSE | 0.009321498530172364 | 0.01330947575246414 |
| Full-head max normalized difference | 0.06275426615732327 | 0.07123261814215877 |

First bitwise difference remains p0_l0_attention: NMSE3.9609884988946576e-7,
max normalized4.2647784379233616e-4. All four token rows,48 layers and three
phases plus the full248320-logit head are compared. No numerical tolerance or
PASS threshold is assigned. Synthetic/scalar helper agreement does not prove
whole-model agreement. The larger head difference is not an optimization win.

Report path:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/original48-hc-fma-prefix4-v1/report.json

Report SHA dce363ee8e533595ba42733bbc0e55c423fdd85f1e5a24954d78404fc4f5b27e.
Post-model identity SHA05e1059d1bcc0c9ea2cdc11c037d1c33619b7e6476ad7cc02884e47031cff52b.

VERDICT -> HC-only host arithmetic refinement does not close the original48
fidelity gap and worsens this head metric. Localize layer0 attention/HC/GDN
seams and rounding/window effects using independently owned operands. Preserve
this result and the older reference. Full fidelity, declared tolerance, broad
quality, complete cache/API concurrency, profiling and matched latency remain
open; no shelf, serving speed or model numerical qualification is claimed.
