# Paired P2048 profiler V4 budget failure, 2026-10-12

CONFIG -> Original SDK52/main UD-Q4_K_XL, pair24/24, prefill512/context4096,
mirror0/copy0 first arm. H36 off. One paused warmup and one selected fresh
P2048 GEN8; admitted PTI512 sampling plus Chrome call/kernel/device logging.
Aggregate apparent-file cap1GiB, clock cap512MiB unchanged.

COMMAND -> bin/gpu-run producer13066 terminal FAILED545s. No second arm
started. First selected request emitted8 and DONE, but no completed selected-
close control was admitted. Raw-budget watcher triggered controlled parent
interruption, engine/container exit137. Owned removal, post-health, kernel
window, originalfull4 and source checks passed; no recovery used.

RESULT -> Clock472861176B and Chrome720237386B. Postwrite aggregate1202054447B,
128312623B overshoot. This is a periodic apparent-file bound, not a hard disk
quota; the overshoot remains evidence and the capture remains failed.

Chrome category bytes/counts:
- CPU257448562B /1729816 events.
- Flow300424245B /2304194 events.
- GPU162362172B /578827 events.
Clock581094 instance normalizations account472122552B; query records are
comparatively small. This volume audit reads failed artifacts only; it is not
an admitted operation timeline, bandwidth measurement or overlap result.

VERDICT -> Preserve failure. Source inspection shows kernel logging also enables
CPU callbacks, so removing call logging alone is insufficient. The supported
Chrome device-only path uses the same GPU callback/IDs/copy-byte names without
the CPU callback. Next capture retains full P2048/model work and caps, omits
CPU host calls, and leaves host preparation/calibration/full model coverage
explicitly incomplete. No cap increase or filtered failed-capture PASS.

Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/

- source52-p2048-copy-profile-v4-run-v1/report.json SHA256
  a13fe69ed372c51cf098926caf8b82106934d414e386957f24a223a80901bda3
- source52-profile-v4-failed-category-volume-v1.json contains the compact
  category audit. Original failed clock/Chrome files remain untouched.
