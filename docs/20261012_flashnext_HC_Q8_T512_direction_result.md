# HC Q8 T512 direction operator result, 2026-10-12

CONFIG -> Exact existing source55 Q8 kernels, synthetic dense Q8 rows,
T512, tile2/4 for downK10240/M320 and upK320/M10240. Card1 lease and
LevelZero affinity1, full device-node exposure; runtime checked PCI
0000:44:00.0 before allocation/submission. No model data.

COMMAND -> NoDri34964 PASS24s; GPU71991 PASS41s. Initial outputs and
changed-input captured graphs for both tiles/directions are byte exact.
Nine alternating samples per tile/direction, three warm rounds; timings
include host submit and original queue wait, exclude readback.

RESULT -> Median down tile2 5322.06us ->tile4 4472.59us (15.96% less);
up tile2 5324.53us ->tile4 4008.56us (24.72% less).
Both T512 directions improve; the earlier T8 down regression does not
transfer to this shape. Perdirection weight3481600B, down input20971520B
/output655360B, up reversed. Exact graph/direct output and invalidtile
refusal pass. Normalexit0/ownedremove/posthealth/kernel pass.

VERDICT -> Operator evidence supports a default-OFF prefill-only tile4
model candidate. No decode tile change, device-only duration, actual
weight bandwidth, whole HC chain or model speed claim. Card0's independent
MTP diagnostic was active during this single-card operator check.

F17 hc-q8-t512-direction-operator-v1/report.json SHA844f5772b8926496ba64cc5223c4151b1e957e0941dfa2d4e9b24528873cdc5d.
ELFffdc536c4877d2e400fb9a654b1580f59fe52385c215316f3ab261f5ec9f2c9f.
