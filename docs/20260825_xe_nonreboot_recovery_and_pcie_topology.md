# 2026-08-25 B70 non-reboot recovery and PCIe topology

## Outcome

The previous reboot-only diagnosis was wrong. Neither B70 has an attached or
active display, and `xe` is unloadable after both B70 PCI endpoints are first
unbound. On boot ID `e2d5777d-f6bb-4d92-a718-0fb07ae17919`, rebind, full
`xe` unload/reload, and endpoint FLR each completed without reboot and passed
both per-card and two-rank compiled-collective health checks.

This proves the recovery mechanics on a healthy system. Clearance of a future
naturally occurring deep wedge still needs to be recorded. Reboot remains the
last fallback if unbind hangs or the complete non-reboot ladder fails.

## Display diagnosis

Config -> kernel 7.1.0-070100; two Intel `8086:e223` display functions at
`0000:0b:00.0` and `0000:44:00.0`; no display cable attached.

Command -> inspect every DRM connector's `status` and `enabled`, `/proc/fb`,
`/sys/class/vtconsole/vtcon*/name`, `/dev/dri` holders, `boot_vga`, `lsmod`,
and the xe auxiliary-driver links.

Result -> all 16 DP/HDMI connectors were `disconnected` and `disabled`;
`/proc/fb` was empty; the only VT console was `(S) dummy device`; no process
held a DRM node. Card 0 had `boot_vga=1`, but that is only the firmware-primary
PCI marker. The four apparent `xe` users were two `xe.mei-gscfi.*` children
bound to `mei_gsc` and two `xe.nvm.*` children bound to `mtd_intel_dg`.

Verdict -> neither card is display-held. The old unload attempt failed because
both GPU functions, and therefore their four auxiliary children, were still
bound to `xe`.

## Recovery implementation

`bin/xe-reset` now acquires both GPU leases, stops only running containers that
expose `/dev/dri` or are privileged, and hard-fails if any process still holds
a DRM node. Privileged sysfs operations go through the installed root-owned
`/usr/local/sbin/b70-xe-reset-helper`. The helper dynamically identifies only
Intel B70 display functions by `vendor:device:class =
0x8086:0xe223:0x030000`, so a later slot move can change BDFs without granting
generic sysfs write access.

The default command runs the least disruptive recovery first and escalates:

```text
./bin/xe-reset
  1. unbind both, then bind both
  2. unbind both, unload/reload xe, then bind both
  3. unbind both, FLR both endpoints, then bind both
  4. reboot only if all non-reboot stages fail
```

Explicit stages are available for diagnosis:

```bash
./bin/xe-reset --method rebind
./bin/xe-reset --method reload
./bin/xe-reset --method flr
```

The tool verifies both driver links, both PCI-qualified render paths, xe
auxiliary bindings, per-card matmuls, and `bin/xpu-collective-health`. The
collective probe uses two XCCL ranks, an eager all-reduce, and ten
`torch.compile` functional all-reduces at `[4,5120]` BF16. Routine health uses
`CCL_TOPO_P2P_ACCESS=0`; direct P2P remains an explicitly guarded experiment.
The shared lib-based multi-card serve guard and both production sglang TP=2
shelf launchers run per-card plus collective health before launch and after
teardown. Set `B70_COLLECTIVE_HEALTH=0` only for an explicit lib-based
diagnostic that cannot use the pinned oracle image.

Whole-card secondary-bus reset is not automated. Each B70 exposes a sibling
HDA function behind its on-card bridge, so a bridge reset needs a separate
audited procedure that also handles HDA.

## Validation record

| Stage | Command | Driver result | Card health | Collective health | Boot ID |
|---|---|---|---|---|---|
| baseline | `gpu-run xpu-collective-health --p2p 0` | unchanged | not rerun | pass | `e2d5777d-...` |
| rebind | `xe-reset --method rebind` | both unbound, both rebound | pass/pass | pass | unchanged |
| reload | `xe-reset --method reload` | refcount 0, unload/reload passed, both reprobed | pass/pass | pass | unchanged |
| FLR | `xe-reset --method flr` | both unbound, both FLR, both rebound | pass/pass | pass | unchanged |

The initial collective-probe development attempt lacked the proven container
`SYS_PTRACE`/seccomp permissions and failed DRM-FD exchange before a
collective. After matching the established oracle container contract, the
baseline and all three post-reset runs passed. This was a probe configuration
error, not a device-health result.

## Current PCIe placement

The user reports card 0 in the top full-length slot and card 1 in the third
two-slot position, leaving a two-slot cooling gap. The motherboard firmware
does not expose reliable ACPI physical-slot labels, so the exact `PCIE3` versus
`PCIE4` silkscreen name must be confirmed visually before moving hardware.

The OS topology is unambiguous:

```text
card 0  pci0000:00 -> 0000:00:03.1 -> 0000:09:00.0
        -> 0000:0a:01.0 -> 0000:0b:00.0

card 1  pci0000:40 -> 0000:40:03.1 -> 0000:42:00.0
        -> 0000:43:01.0 -> 0000:44:00.0
```

The cards therefore sit under different Threadripper root domains/dies. Linux
reports `numa_node=-1` and `local_cpulist=0-31` for both because the BIOS is in
UMA mode; that does not make the physical PCIe root paths identical. The B70
GPU-adjacent functions' 2.5 GT/s x1 reading is the known on-card-bridge
artifact. The slot-facing upstream links have previously been verified at the
1950X platform maximum, Gen3 x16.

## Recommended slot A/B

Do not move a card merely to correct link width; both current slot uplinks are
already full Gen3 x16. A move is still useful as a controlled topology test:

1. Record the current BDF paths, slot-facing link width/speed, temperatures,
   `xpu-collective-health --p2p 0`, the direct-P2P oracle, and the exact serve.
2. Power off fully. Move card 1 to the unused full-length slot nearest card 0
   that the motherboard manual maps to the same CPU root domain. Confirm the
   board silkscreen first; do not infer it from Linux card numbering.
3. Boot and require both GPU paths to begin with `pci0000:00`. If they do not,
   the move did not achieve the experimental condition.
4. Reinstall nothing: the helper discovers the new BDF. Re-record link width,
   temperatures, card health, compiled collective health, guarded direct P2P,
   and matched serving metrics.
5. Keep the move only if coherence is unchanged and the measured collective or
   serving result improves enough to justify the lost cooling gap and any x8
   slot-width tradeoff.

Raw oneCCL direct P2P and XPUGraph already work across the current two root
domains. Recent exact vLLM work has also crossed this topology coherently.
Therefore the slot move is an A/B for latency, stability, and integration
behavior, not an assumed fix.

## Four-card note

The ASRock X399 board exposes four full-length PCIe positions with a nominal
x16/x8/x16/x8 lane arrangement. Four dual-slot B70s would consume the entire
slot area and change cooling, power, BDF count, lease coverage, and rank
mapping. Before adding cards, extend `bin/gpu-run`, `bin/xpu-health`, and the
collective probe beyond two cards; verify the actual lane/root-domain mapping;
and qualify thermals and PSU/cabling independently. The reset helper already
accepts up to four matching B70 endpoints, but the rest of the operational
stack is currently two-card-only.

## References

- ASRock Fatal1ty X399 Professional Gaming manual:
  https://download.asrock.com/Manual/Fatal1ty%20X399%20Professional%20Gaming.pdf
- Linux xe device documentation:
  https://docs.kernel.org/gpu/xe/xe_device.html
- Linux PCI sysfs reset interface:
  https://www.kernel.org/doc/html/latest/PCI/sysfs-pci.html
