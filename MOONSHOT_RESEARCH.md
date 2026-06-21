# MOONSHOT_RESEARCH.md -- wild B70 interconnect / fabric ideas

Status: living idea-lab (started 2026-06-22). Home for the high-risk / high-ceiling architecture
bets that don't belong in the measured docs yet. Each idea carries a maturity tag:

  [PROVEN]  works in the wild, just not on our box yet
  [BUYABLE] real shipping hardware/software, untested by us
  [RESEARCH] plausible but nobody has shown it on discrete Battlemage
  [MOONSHOT] far out; bring-up cost is the whole project

Companion to [docs/P2P_GPU.md](docs/P2P_GPU.md) (the measured/near-term P2P picture). This doc is the
"what if we stop accepting the 1950X's topology as fixed" side.

Our villain, restated: the two B70s hang off SEPARATE 1950X dies (cross-die P2P over Infinity Fabric,
worst case), and the host root is PCIe **Gen3**. Real per-card link is Gen3 x16 (~15.8 GB/s). Every
idea below is ultimately about retiring one or both of those constraints.


## 1. The one distinction that governs everything: bifurcation vs. switch

A recurring trap. "Put both GPUs on one slot with an x16->2x x8 cable" can mean two completely
different things with opposite P2P behavior.

```
BIFURCATION (passive x16 -> x8+x8)          ACTIVE SWITCH (Broadcom PEX / Microchip Switchtec)

   die0 root complex                            die0 root complex
    |        |                                         | rootport x16 (Gen3)
 rootport  rootport   <- TWO separate            +-----+-----+
  x8 Gen3   x8 Gen3      bifurcated ports         |  SWITCH   |  one chip; upstream = host Gen3
    |        |                                    +--+-----+--+
  GPU0     GPU1                                      |     |
                                                   GPU0   GPU1   downstream = Gen5 x16 EACH
 P2P: GPU0 -> rootport -> *CPU ROOT               P2P: GPU0 -> SWITCH -> GPU1
 COMPLEX* (hairpin in the SoC) -> GPU1.           (hairpins INSIDE the switch; never
 Capped at host Gen3, x8 per card.                 touches the CPU). Gen5 even though
                                                   the host link is only Gen3.
```

| property                          | passive bifurcation        | active PCIe switch                |
|-----------------------------------|----------------------------|-----------------------------------|
| switch chip                       | none                       | yes (PEX89000 / Switchtec)        |
| GPU<->GPU P2P path                 | up to CPU root complex     | forwarded inside the switch       |
| "hairpin behind the slot"?        | NO (hairpins in the CPU)   | YES (hairpins in the switch)      |
| peer bandwidth                    | host gen, x8/card          | switch gen (Gen5 x16) -- DECOUPLED|
| needs mobo bifurcation support    | yes                        | no                                |
| pci_p2pdma classification         | different rootports        | "behind the same switch" (clean)  |
| cost                              | ~$20-60 cable/riser        | ~$200-1500 switch board           |
| on our 1950X                      | Gen3 x8/card + CPU hairpin | Gen3 host uplink, Gen5 peer        |

**Key insight (the whole reason a switch wins): generation decoupling.** A switch's upstream link
trains to the host's gen (Gen3 here); its downstream links train to the switch's + device's native gen.
Our B70 onboard uplink already advertises **Gen5 x16 (32 GT/s)** -- it only runs Gen3 because the 1950X
root caps it. On a Gen5 switch, GPU<->GPU becomes **Gen5 x16 (~64 GB/s)** while the host stays Gen3,
because peer traffic never touches the host link. That is exactly the "fast TP behind the switch, only
CPU<->GPU on the slow link" architecture -- and it's the thing bifurcation cannot do. Bifurcation
actually makes per-card host bandwidth WORSE (Gen3 x8 ~= 7.9 GB/s vs our current Gen3 x16 ~= 15.8).

So: **bifurcation is not the essence of the Gen5-switch; the active switch chip is.** Bifurcation only
co-locates the cards (which would fix cross-die if both land on one die) -- it does not give local
switch-forwarded P2P or Gen-decoupled peer bandwidth. "Gen5" on a passive cable is wasted on a Gen3
host; "Gen5" only pays off when a Gen5 switch terminates both ends.


## 2. The architecture ladder (cheap -> deluxe)

### T0 -- Software only, current box  [PROVEN, our control]
No hardware change. Steal Seguin's allreduce/graph-fusion env vars, A/B `CCL_TOPO_P2P_ACCESS=1` vs `=0`.
Attacks the FIRST-order bottleneck (graph breaks around the collective). Must be done first -- it tells
us whether transport even matters on our fabric before we spend a dollar. See docs/P2P_GPU.md F.1.

### T1 -- Passive bifurcation x8+x8, both cards on ONE die  [BUYABLE, ~$40]
Riser/cable + mobo BIOS x8x8 bifurcation on a die0 slot. Fixes cross-die (both cards intra-die, P2P
hairpins in ONE die's root complex, not over Infinity Fabric). Cost: each card drops to Gen3 x8
(~7.9 GB/s host bandwidth). Net win ONLY if intra-die P2P working beats the lane halving. Cheap enough
to just try. Does NOT give switch-local P2P or >Gen3 peer.

### T2 -- Active PCIe switch card, both B70s behind one switch  [BUYABLE, ~$200-1500] *** the pragmatic moonshot ***
A Gen5 switch board (Broadcom PEX89000 / Atlas2 family; Microchip Switchtec PFX/PSX; host cards like the
Serial Cables Gen5 x16 board carry a real Atlas2 switch). Both B70s become downstream ports of one
switch -> "behind the same switch" clean P2P + local hairpin + **Gen5 x16 peer bandwidth behind a Gen3
host**. This is how DGX-class boxes wire P2P. Also a strong candidate to FIX the Puget RxErr faulting
(that was riser signal integrity; a retimed Gen5 switch is the cure). Biggest bang-for-complexity. The
one to price out if T0 shows P2P actually moves our number.

### T3 -- External PCIe expansion chassis  [BUYABLE]
Host-bus-adapter in a host slot -> cable -> external box with the switch backplane + both GPUs (OSS /
One Stop Systems, c-Payne switch/redriver boards, H3 Platform). Still pure PCIe end-to-end, so P2P +
transparency just work. Buys physical disaggregation (thermals, power, slot count) on top of T2. The
"GPU brick" that can re-home onto any future host.

### T4 -- Composable PCIe fabric + NTB  [BUYABLE, enterprise $$]
GigaIO FabreX / Liqid: PCIe as a switched fabric across hosts, dynamic compose of N GPUs to a host with
GPU-to-GPU P2P across the fabric. Dolphin PXH-class NTB cards give a raw shared-memory window over a
PCIe cable -- a substrate to prototype OUR OWN GPU<->GPU protocol, sidestepping the host entirely. Most
"invent our own interconnect" energy of the buyable tier.

### T5 -- 400Gb network-tunneled / DPU emulation / CXL 3.0  [MOONSHOT]
The original sketch: CPU -> VF-spoofing card -> 400Gb link -> card -> Gen5 switch -> GPUs. Honest
decomposition:
- "SR-IOV makes it transparent" is really **NTB or DPU device-emulation**, not SR-IOV. SR-IOV slices one
  LOCAL device into VFs; it does not relocate a device across a wire. Transparency needs either an NTB
  mapping the remote PCIe space into the host's, or a DPU (BlueField/SNAP) EMULATING a PCIe device to
  the host while proxying over the net (done for NVMe/virtio; brutal for a 32GB-BAR latency-sensitive GPU).
- If the "400Gb link" is PCIe-over-cable (GigaIO/Liqid/Dolphin/Serial-Cables QSFP-DD), it stays PCIe ->
  transparent for free; this is just T3/T4 with optical cabling. A Gen5 x16 cable ~= 64 GB/s (~512 Gb/s),
  so "400Gb" ~= Gen5 x8.
- If it is literally PCIe-tunneled-over-Ethernet/IB, that's the real moonshot -- TLPs don't packetize
  well without hardware terminating PCIe both ends; you'd be doing hardware bring-up, not inference.
- CXL 3.0 horizon: coherent memory pooling -- tier host DRAM as KV-cache, share VRAM coherently. We're
  early; T2/T3 is the on-ramp.


## 3. Software NVSwitch -- present 2x B70 as ONE logical GPU  [RESEARCH] *** the real prize ***
The most interesting bet, and it's software, not exotic hardware. Build a shim that presents the two
B70s to the inference stack as a single logical device, hiding the TP sharding -- "software NVSwitch for
Battlemage." Realistic layer is NOT DPU hardware emulation; it's the RUNTIME (ZML's device-mesh / SPMD
path, or a vLLM-XPU platform shim exposing one logical XPU and sharding underneath). This composes with
the compiler-fused-collectives bet (docs/P2P_GPU.md F.5): own both the matmul and the collective, present
one device. Nobody has shipped this for B70 TP -> genuinely publishable.


## 4. SR-IOV reality on our cards
The B70 functions report a placeholder Gen1 x1 link + a "Virtual Resizable BAR" cap -> SR-IOV
fingerprint. Battlemage/Flex parts DO support SR-IOV (VFs). Two consequences: (a) we may already be
looking at VFs, not raw PFs; (b) VF-to-VF P2P across two physical cards on two dies is the hardest P2P
case of all. If we go the switch route, confirm whether we're binding PFs or VFs and whether P2P is
even exposed to VFs. (Open question -- needs a booted 7.x + ze_peer to settle.)


## 5. 1950X reality check (don't let "Gen5" fool us)
- The host root is **Gen3**. A Gen5 cable/switch upstream-to-host trains to Gen3. Gen5 only helps on the
  switch<->GPU segments (peer traffic), never on the host uplink.
- CPU<->GPU (weight load, KV spill, sampling) stays Gen3-bound no matter what we do downstream. For
  decode that is mostly fine (weights resident); for prefill/large-batch the host link matters more.
- Therefore the switch's value is specifically: make GPU<->GPU TP fast (Gen5 peer) while accepting a
  Gen3 host uplink. If our workload's TP allreduce volume is small (Seguin: 15-17us raw), the peer-gen
  win is second-order -- which loops back to T0 being the right first move.


## 6. Next experiments (ordered)
1. [T0] Seguin env vars + P2P on/off A/B, TP=2, on the current box (gpu-run lock). Decide if transport matters.
2. [kernel] Boot 7.x (live USB, zero disk risk): does `pci_p2pdma_distance` permit our pair under
   iommu=pt? does `ze_peer` report peer access + what bandwidth? does enabling it fault (Puget RxErr)?
3. [T2 decision] If T0 says transport matters AND ze_peer shows peer access works, price a Gen5 switch
   board (PEX89000/Switchtec) and model: does Gen5 x16 peer behind a Gen3 host beat current cross-die?
4. [T1 cheap probe] Optionally try x8x8 bifurcation onto one die first ($40) to isolate "cross-die" as
   the culprit before buying a switch.
5. [F.5 / sec 3] Prototype compiler-fused collectives / single-logical-GPU runtime shim -- the contribution.


## 7. References
- Broadcom PEX89000 Gen5 switch product brief: https://docs.broadcom.com/docs/PEX89000-Managed-PCI-Express-5.0-Switches
- Broadcom PEX89072 (72-lane Gen5): https://www.broadcom.com/products/pcie-switches-retimers/expressfabric/gen5/pex89072
- Serial Cables Gen5 x16 QSFP-DD host card w/ Atlas2 switch: https://serialcables.com/product/pcie-gen5/serial-cables-pcie-gen5-x16-qsfp-dd-host-card-with-broadcom-atlas2-production-level-llc-pcie-switch-skupci5-ad-x16he-bg5-qdd-id76
- H3 Platform composable w/ PEX89144: https://www.h3platform.com/newsroom/press-release-detail/67
- Level1Techs, PCIe bifurcation and GPU P2P: https://forum.level1techs.com/t/pcie-lane-bifurcation-and-gpu-p2p/211380
- (verify part numbers/prices/availability before purchase -- Gen5 switch boards are enterprise-priced
  and homelab Gen5 switch boards e.g. c-Payne were historically Gen4-class; confirm current options.)
