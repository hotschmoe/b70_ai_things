# Device-only paired profile V5: aggregate failure and observed OFF arm

CONFIG -> Original SDK52/mainUDQ4XL pair24/24, P2048/pref512, GEN8 paused
warmup/selected, device-only Chrome plus admitted clock, copy0/1, H36OFF.
Whole-experiment1GiB cap unchanged; per-arm clock512MiB cap unchanged.

COMMAND -> Actual96965 producer is in failed aggregate cleanup. First arm
completed warmup/selected, explicit finalizer0 and per-arm boundary/raw checks.
Read-only inspector47823 matched581094 Clock/Chrome instances within actual
selected emission bounds20..583306. This is explicitly UNADMITTED diagnostic
analysis: it does not turn the failed two-arm experiment into PASS.

RESULT -> First arm raw postwrite810159793B. Both observed native devices use
engine group/index0,0. Explicit H2D payload22099527680B and18358661120B
(~40.46GB total); observed copy union1613.43/1324.29ms, compute union6826.08/
6647.17ms, copy-compute overlap0 on each device. Small D2D/D2H/M2D commands
remain in the full JSON. No direct mapped-RAM kernel traffic is measured.
RAW-to-MONOTONIC offset range3.220ms; host/device alignment is unqualified.

The second arm pushes the aggregate past the cap (peak1571860968B at current
failure observation). Both raw sets remain untouched. Failed/partial temporal
controls cannot certify the ON arm or yield a matched overlap result.

VERDICT -> Concrete observed OFF copies serialize on compute. Total physical
PCIe/VRAM/RAM bandwidth, complete operation coverage, true GPU idle and joined
host/device criticalpath are unproved. Next capture must use separate fully
owned single-arm experiments, each preserving its1GiB cap, then a matched
readonly comparison after both pass. No larger budget or filtered failed PASS.

Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/
source52-v5-armA-observed-unadmitted-v1.json
SHA256 f63275fcbeddaa21d7af1ef6764661119159bbd99663a382e5e0fee74bc94a6b
