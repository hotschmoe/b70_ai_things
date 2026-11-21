# HARDWARE.md -- measured Intel Arc Pro B70 (Xe2 BMG-G31) facts for the TensorFold Flash-Next port

Author: O-HW, model claude-opus-5-5 (Claude Opus 5.5). Measured 2026-11-21 05:29-06:58 UTC.

CONFIG: both B70 cards (card 0 = 0000:0b:00.0 under root pci0000:00 / CPU die 0 = CPUs 0-7,16-23; card 1 =
0000:44:00.0 under pci0000:40 / die 1 = CPUs 8-15,24-31), kernel 7.1.0-070100, compute-runtime / ocloc 26.22.38646.4,
Level Zero loader 1.28.2. Kernels: OpenCL C, ocloc -device bmg, CL3.0, SIMD16, 128-GRF unless noted (256-GRF variants
also run). Host: C + Level Zero (zig cc). One process per card, ZE_AFFINITY_MASK=N, taskset to a die-local CPU.
Every GPU touch under bin/gpu-run --card N, <= 175 s per lease, logged in GPU.md as [O-HW].
COMMAND: hwprobe {info,alu,bw,lat,gather,ramp,launch,host,xfer,gemv256,sustain_bw,sustain_dpas,p2p}.
Source: /mnt/vm_8tb/b70/build/tensorfold/hwprobe/ (hwprobe.c, hwprobe.cl, run-lease.sh, in-*.sh).
Raw JSONL: /mnt/vm_8tb/b70/results/tf-flashnext/hw/ (main-c{0,1}, remote-c{0,1}, sustain-c{0,1}, p2p, hostatom-c1;
main-c1.v1-constfill.jsonl is the superseded first run whose bandwidth numbers were inflated by compression, see 1.3).
Timing: device kernel timestamps (19.2 MHz timer, 52 ns tick); every bandwidth/ALU kernel runs >= 100 us; medians of
5-7 reps after a warm-up; latency kernels unroll 4-16 dependent ops per loop iteration. Data: synthetic, random
(xorshift) for every bandwidth/GEMV number unless stated; GEMV outputs are not numerically verified (speed only).
Card 0 and card 1 agree within 2% on every device number; card-1 values are shown unless noted.
Not re-measured here (see TRANSPORT-B.md, B/latbench, orch-numa): cross-card push/pull, events across cards,
host-USM flag wake-up, idle-append-to-start.

VERDICT summary: DRAM is reachable (604 GB/s = 99% of 608 with a simple read kernel, 96% with a well-formed GEMV at
M=1). The engine-style IQ3_XXS GEMV on GGUF array-of-structs (98 B blocks, 2-byte aligned) is load-instruction bound
at 17-20% of roofline; the same math on an aligned split layout reaches 90% at M=1. Above M=4 the vector dp4a path is
ALU bound and DPAS is needed. Short kernels pay ~3 us each; immediate-list appends cost ~2.1 us of host time each.

## 1. Measured numbers

### 1.1 Device configuration (zeDeviceGet*Properties)
| Quantity | Value | Method |
|---|---|---|
| Xe-cores / EUs per core / HW threads per EU | 32 / 8 / 8 = 2048 SIMD16 threads (subgroups) at 128 GRF, 1024 at 256 GRF | device props |
| Core clock max | 2800 MHz (sysfs rp0 2800, rpn 400); idle act_freq 0 within 500 ms (RC6) | props + sysfs |
| L2 (last-level) cache | 24 MiB | zeDeviceGetCacheProperties |
| SLM max per work-group | 128 KiB; max WG 1024; subgroup sizes 16, 32 | compute props |
| VRAM | 32.5 GB, "DDR" 2800 MHz x 64-bit reported (GDDR6 256-bit, 608 GB/s nominal) | memory props |
| Queue groups | ordinal 0 compute+copy (1 queue), ordinal 1 copy (1 queue); no graph/record-replay extension | props |
| Host USM capabilities | RW + ATOMIC; device memory RW + ATOMIC; shared-cross-device 0 | memory access props |
| Peer (0<->1) | P2P ACCESS yes, P2P ATOMICS NOT advertised, canAccessPeer 1 both ways | zeDeviceGetP2PProperties |
| PCIe uplinks | 8.0 GT/s x16 both cards (upstream ports 09:00.0, 42:00.0) | sysfs |

### 1.2 Compute peaks (full device, 4096 subgroups unless noted)
| Quantity | Value | Method |
|---|---:|---|
| DPAS bf16 / fp16 (M8 N16 K16) | 183.3 TFLOPS | 8 independent DPAS chains per subgroup, 2048-8192 SG identical |
| DPAS int8 (M8 N16 K32) | 366.6 TOPS | same |
| DPAS int4 (intel_sub_group_i4_i4_matrix_mad_k64, OpenCL C) | 733.2 TOPS | same; int4 DPAS IS exposed in OpenCL on this driver |
| DPAS tf32 (M8 K8) | 91.7 TFLOPS | same |
| DPAS instruction rate M8 / M4 / M1 | 44.75 / 72.0 / 74.0 per ns (device) | M4 = 80% of M8 TOPS, M1 = 20.7% (bf16 M1 37.9 TFLOPS, int8 M1 75.9 TOPS) |
| DPAS dependent latency (bf16 M8) | 12.3 ns (~34 clk) | single chain, 16 per loop iteration |
| FP32 FMA (vector) | 22.05 TFLOPS (96% of 32x128 lanes x 2 x 2.8 GHz) | 64 independent SIMD16 mad per iteration (checked in IGC asm) |
| FP16 FMA (vector) | 20.3-21.4 TFLOPS (no 2x over fp32; IGC emits SIMD16 :hf mad) | same |
| dp4a int8 (dot_acc_sat_4x8packed) | 84.9 TOPS = 10.6 T dp4a/s | 16 independent chains |
| int8->float convert (convert_float4(as_char4)) | 3.43 T conversions/s | all components consumed |
| FMA dependent latency | 2.79 ns (~7.8 clk); empty loop iteration 0.007 ns (unrolled) | single subgroup |
| Sustained clock | 2800 MHz streaming (no throttle, 64 s); 2733-2767 MHz under DPAS (PL2, card power 230 W = power1_cap) | sysfs act_freq / throttle reasons, 1 s samples |

### 1.3 Memory bandwidth (1 GiB buffer, random data unless noted)
| Quantity | Value | Method |
|---|---:|---|
| Streaming read, best (32 B/lane or 4x256 B block reads in flight) | 603.7 GB/s (99.3%) | rd_u8 / rd_b8 / rd_b4x4 |
| Read by load width per lane: 4 B / 8 B / 16 B / 32 B | 296 / 447 / 589 / 604 GB/s | rd_u1/u2/u4/u8, grid-stride, 4096 SG |
| Subgroup block read 64 B / 256 B / 512 B per op | 302 / 590 / 604 GB/s | intel_sub_group_block_read{,4,8} |
| Occupancy, 1 x 256 B block read in flight per SG | 172 / 295 / 443 / 590 GB/s at 256 / 512 / 1024 / 2048 SG (8/16/32/64 per Xe-core) | rd_b4, WG 16-1024 identical |
| Occupancy, 4 x 256 B in flight per SG | 427 / 590 / 604 GB/s at 256 / 512 / 1024 SG | rd_b4x4 |
| Write, incompressible / pattern | 471 / 605-642 GB/s | wr_rand (hashed) / wr_u4,wr_b4 |
| Copy kernel read+write (random) | 464-472 GB/s total (232 each way) | cp_u4 |
| Lossless compression effect on reads | zero 3198 / constant 1556 / random 604 GB/s | rd_b4x4 after zero, 0x01010101 or random fill |
| Working set: L1 / L2 / DRAM | 13.6-14.2 TB/s (<=64 KiB) / 9.0-11.4 TB/s (256 KiB-16 MiB) / 3.4 at 32 MiB, 582 GB/s >= 256 MiB | rd_ws, 2048 SG x 512 x 256 B, own sequential window per SG |
| Dependent-load latency L1 / L2 / DRAM | 22.6 ns / 84.6 ns / 424-451 ns (256-512 MiB random, incl. TLB) | single-lane pointer chase, 64 B lines, random start per rep |
| Host-USM dependent read over PCIe | 1126-1129 ns | chase in zeMemAllocHost buffer |
| Store then load, device / host USM | 90 ns / 1129-1302 ns | store_lat |
| Random row gather (rows at random row-aligned offsets of 1 GiB) | 64 B 115, 128 B 224, 256 B 391-450, 512 B 465, 1 KiB 470, 2-64 KiB 475-505 GB/s | gather1/gather4, 4096 SG, 256 MiB touched |
| Same rows, sequential order | 64 B 166, 128 B 256, 256 B 395-576, 512 B 548-597, 1-2 KiB 590-600 GB/s | identity index |
| First kernel after 1-1000 ms idle | +8% (485 vs 448 us for 256 MiB); 1 ms idle: +13% | ramp |

### 1.4 SLM, subgroup ops, barriers, atomics (full device = 4096 SG, WG 256)
| Quantity | Value | Method |
|---|---:|---|
| SLM read, 16 B/lane (WG >= 256) | 17.7-17.9 TB/s (~200 B/clk/Xe-core); WG 64: 6.3 TB/s | slm_rd4, 32 KiB per WG |
| SLM read, 4 B/lane | 6.07 TB/s; strided lanes (stride >= 8 words): -22% (837 -> 652 Gloads/s) | slm_rd1, slm_str |
| Random 256-entry table lookup: none / SLM / __constant | 1914 / 915 / 404 G lookups/s | lut_* (divergent index per lane) |
| sub_group_reduce_add (float) | 30.3 ns dependent; 64 G subgroup-ops/s device | 8 per loop iteration, each + 1 fmul |
| sub_group_shuffle (variable index) / shuffle_xor (const) / broadcast | 24.5 / 11.4 / 4.9 ns dependent; 20.8 / 114 / 403 G/s device | same; plain fmul+add: 3.4 ns, 442 G/s |
| Work-group barrier (pair, incl. SLM store+load) | 72 ns (WG 16-64), 84 ns (WG 256), 153 ns (WG 1024) | one WG |
| Global atomic_add dependent latency | 89.7 ns (= L2 round trip) | 1 lane |
| SLM atomic_add dependent latency | 23.1 ns | 1 lane |
| Global atomics throughput: same address / distinct 64 B lines | 22.3 / 43 G/s | all lanes |
| SLM atomics throughput (distinct) | 268 G/s | |
| Host-USM device atomic_add dependent latency | 1225 ns; 3200/3200 increments landed (correct) | card 1, 800 per run, bounded |
| Cross-card atomics | not advertised (P2P ATOMICS=0); not run. Prior: in-kernel peer flags hang (TRANSPORT-B 1) | props |

### 1.5 GEMV (K = 2560, N rows chosen for ~128 MiB weights, 2 matrices alternated -> DRAM-cold; % of 608 GB/s)
Vector kernels: one subgroup per RPS rows, lanes stride over 32-element blocks, dp4a with int8 (q8_1-style)
activations; bf16 = bf16 weights x fp32 activations. DPAS kernels: one subgroup per 16 output rows, B operand from a
repacked tile layout (512 B per 16x32 int8 tile). Best of grids {1024, 2048, 4096, 8192 SG} and RPS {1,2,4}; 128 GRF.
| Format (bytes/weight) | M=1 | M=4 | M=8 | M=16 |
|---|---:|---:|---:|---:|
| q8 vector dp4a (1.0625) | 589 (97%) | 576 (95%) | 563 (93%) | 541 (89%); RPS 4 spills |
| q8 DPAS (s8 x s8) | 562 (93%) | 554 (91%) | 539 (89%) | 528 (87%) |
| q4 vector (0.5625, u8 x s8 dp4a + zero-point) | 587 (97%) | 562 (92%) | 544 (89%) | 467 (77%, RPS 2) |
| q4 DPAS (u8 nibbles -> int8 B) | 542 (89%) | 525 (86%) | 507 (83%) | 507 (83%) |
| q3 IQ3_XXS-like, GGUF AoS 98 B blocks, table in __constant | 119 (20%) | 117 (19%) | 114 (19%) | 100 (16%) |
| q3 same AoS, table in SLM | 122 (20%) | 122 (20%) | 127 (21%) | 109 (18%) |
| q3 ALIGNED split layout (uint2 idx + uint signs + half d), SLM table | 545 (90%) | 457 (75%) | 367 (60%) | 246 (40%) |
| q3 aligned, no table (ALU ceiling w/o lookups) | 566 (93%) | 479 (79%) | 362 (60%) | 252 (42%) |
| q3 DPAS tile (12 B/lane), SLM table | 497 (82%) | 441 (73%) | 401 (66%) | 310 (51%) |
| q3 DPAS tile, __constant table | 330 (54%) | 320 (53%) | 300 (49%) | 268 (44%) |
| bf16 vector (2.0) | 581 (96%) | 570 (94%) | 533 (88%) | 320 (53%) |
| bf16 DPAS | 568 (93%) | 569 (94%) | 564 (93%) | 556 (91%) |
256-GRF mode: halves occupancy; never better than 128 GRF except where 128 GRF spills (q8/q4/bf16 M16 RPS 4).
Small shapes (q8 vector M=1, L2 flushed before every rep): 42 KiB 3.1 us, 255 KiB 3.9 us, 1 MiB 4.2 us
(251 GB/s), 4 MiB 8.9 us (472 GB/s), 16 MiB 30.5 us (549 GB/s): t ~= 3 us + bytes / 590 GB/s.
q3 aligned M=1: 1 MiB 7.0 us, 4 MiB 18.5 us, 16 MiB 43 us. DPAS GEMV without split-K at N <= 1536: 30-56 us
regardless of N (one subgroup walks all 80 K-blocks serially, ~0.4 us per block step).

### 1.6 Kernel launch and command lists (device timestamps of the first and last of 256 chained kernels)
| Quantity | Value | Method |
|---|---:|---|
| Empty kernel duration (own timestamp), 1 WG / full grid | 0.62 / 0.52 us | spin_k iters 0 |
| Back-to-back per-kernel time, regular in-order list, empty kernels | 0.63 us (1 WG), 0.73 us (2048 SG) | 256-kernel chain |
| Extra gap after a real full-grid kernel, regular in-order / OOO+barrier | +0.20-0.40 us / +0.05-0.25 us per kernel | spin 16-4096 iters |
| Immediate in-order list, short kernels | 1.9-2.3 us per kernel (host-append bound; device idles) | same chain, appended live |
| Immediate list, kernels >= ~5 us | overhead 0.03-0.23 us (append hidden) | |
| Barrier / per-kernel signal event / wait-on-previous-event in an in-order list | no measurable device cost beyond the in-order gap (<= 0.15 us) | variants |
| OOO immediate list, 16 independent 1-WG kernels | run concurrently: 114 us vs 1233 us in-order (0.093x) | concurrency |

### 1.7 Host-side Level Zero costs (ns, p50 of 7 x N calls; local-die CPU; remote-die in brackets)
| Call | card 1 | card 0 |
|---|---:|---:|
| zeKernelSetArgumentValue / zeKernelSetGroupSize | 7 / 4 | 8 / 5 |
| regular list append: launch / + signal event / + 1 wait event | 449 [493] / 724 [798] / 492 | 486 [465] / 744 [853] / 545 |
| regular append barrier / signal / wait / event reset / 4 KB copy | 8 / 162 / 86 / 315 / 1103 | 9 / 140 / 83 / 319 / 1193 |
| zeCommandListClose / Reset (256 kernels) | 90 / 991 [1213] | 90 / 1122 |
| zeCommandQueueExecuteCommandLists (256 kernels) | 2094 [2194] | 1813 [2043] |
| execute + queue sync, 256 empty kernels (wall) | 132 us [183 us] | 132 us [228 us] |
| immediate append launch / + signal event / + host sync round trip | 2181 [2261] / 2075 / 6659 [5922] | 2106 [2318] / 2218 / 5929 |
| zeEventHostSignal / HostReset / QueryStatus signaled / not signaled | 201 / 216 / 1525 / 3290 | 212 / 223 / 1486 / 3340 |
| zeEventCreate+Destroy / zeEventPoolCreate+Destroy (256) | 167 / 393 us | 170 / 397 us |
| zeMemAllocDevice+Free 1 MiB / zeMemAllocHost+Free 4 MiB | 403 / 283 | 410 / 298 |
Card-1 main runs held a quiet CPU; card-0 runs did not (die-local CPU still pinned). Remote die costs 5-40% more on
submission calls; execute+sync wall rises 40-70% from the remote die. Observed in the sustain loop (not isolated):
host-resetting 8192 used KERNEL_TIMESTAMP events took ~5 s (~0.6 ms each) versus 0.22 us for a plain host event.

### 1.8 Host <-> device transfers (both cards identical; 256 MiB unless noted)
| Path | Value | Method |
|---|---:|---|
| H2D copy engine / compute engine | 14.3 / 13.9 GB/s (64 KiB: 5.1-5.4) | append + sync |
| D2H copy engine / compute engine | 4.65 / 4.5 GB/s on BOTH cards (64 KiB: 3.0-3.4) | append + sync |
| EU kernel reading host USM / writing host USM | 13.95 / 1.4 GB/s | rd_b4x4 / wr_b4 on zeMemAllocHost |
| D2D copy engine / compute-engine copy | 424 / 900 GB/s | appendMemoryCopy |
| DRAM read kernel during 4x256 MiB copy-engine D2H / H2D | 601-603 / 573 GB/s (alone 604) | concurrent lists |

### 1.9 Sustained load, clocks, power, thermals (65 s each, both cards)
| Load | Clock | Card / pkg power | Temps pkg / VRAM / mem-ctrl (C) | Throttle | Rate |
|---|---|---|---|---|---|
| DRAM streaming (rd_b4x4 1 GiB) | 2800 MHz flat | 191-201 W / 113-123 W | 60-74 / 62-80 / 45-55 | none | 601.5-603.6 GB/s, flat |
| DPAS bf16 | 2733-2767 MHz | 224-231 W / 165-177 W | 74-80 / 68-82 / 55-59 | PL2 (power1_cap 230 W) whole run | 175.5-181 TFLOPS (-2%) |
Fans 840-2330 rpm. No thermal or VRAM throttle observed. Decode is far from both limits.

## 2. Design rules (each follows from the table rows above)

Memory and GEMV
1. Keep >= ~512 KiB of reads in flight device-wide: >= 16 subgroups per Xe-core with 4 x 256 B outstanding, or the
   full 64 per Xe-core with 1 x 256 B. A kernel with <= 1024 subgroups and one load in flight caps at ~73% of DRAM.
2. Every lane must load >= 16 B per instruction (32 B or block_read4/8 is best). 4 B/lane loads cap at 49%, 8 B at 74%.
3. Repack every quantized weight that is read in a hot loop into aligned, split arrays (codes, signs/scales, d).
   IQ3_XXS read as GGUF 98-byte AoS runs at 17-21% of roofline at every M, independent of table placement; the same
   bits aligned reach 90% at M=1. This matches the 22-26% multi-row efficiency in the ledger and is the first fix.
4. Codebook/grid tables: SLM or registers, never __constant with divergent indices (2.3x slower than SLM; the DPAS q3
   kernel loses 33 points of roofline with a __constant table).
5. Above M=4 the vector dp4a path becomes ALU-bound (q3 aligned: 75% at M=4, 60% at M=8, 40% at M=16; q4 M16 77%;
   bf16 M16 53%). For C4/C8 verify rows (M >= 8) use DPAS: bf16 and q8 DPAS hold 87-93% through M=16. Dequant work must
   be done once per weight and shared across all M rows (DPAS does this structurally).
6. DPAS at M=1 costs as much issue time as M=4 (74 vs 72 instr/ns): pad decode rows to 4 or 8 when using DPAS; never
   use DPAS M8 for M=1 expecting a speedup (M1 TOPS = 21% of M8, still far above what a GEMV needs).
7. Small matrices are latency-bound: t ~= 3 us + bytes/590 GB/s. A 1 MiB matrix (one expert gate+up slice) gets 250
   GB/s alone. Batch all experts of a layer (and gate+up) into one launch so each launch moves >= 8-16 MiB.
8. A DPAS or row-per-subgroup kernel must launch >= ~1024 subgroups; when N/16 tiles are fewer, split K (each serial
   K-block step costs ~0.4 us; a K=2560 tile walked by one subgroup takes ~30 us). HC down (320 rows x 10240) needs
   split-K under either scheme.
9. Random expert rows: rows < 256 B waste 26-81% of bandwidth (64 B: 19%, 128 B: 37%, 256 B: 64-74%); rows >= 512 B
   cost ~20% versus sequential (465-505 GB/s). Store each expert matrix contiguously and read whole experts in order.
10. Device memory is losslessly compressed: zero pages read at 3.2 TB/s, constant data at 1.5 TB/s. Never benchmark
    on zero or memset buffers. Zero-filled KV/state padding is cheaper to read than it looks in the byte model.
11. Writes: 471 GB/s for incompressible data (78%); budget KV/state writes at ~470 GB/s, not 608.
12. L2 is 24 MiB (9-11 TB/s): per-window weights (GBs) never stay resident, but per-layer activations, the router
    weights (2.7 MB), HC matrices of one sublayer and KV pages reused across verify rows can. L1 holds <= ~256 KiB.

Launch and command lists
13. Device floor per kernel in a recorded in-order list: ~0.65 us (empty) and +0.2-0.4 us after a real full-grid kernel.
    1000 kernels per window cost >= 0.7-1.0 ms of device time even when fully recorded.
14. Immediate lists are host-bound for kernels shorter than ~2.2 us: each append costs 2.1-2.3 us on the host.
    Record windows into regular lists (append 0.45 us, execute 1.8-2.1 us per list) and replay them.
15. Fuse kernels shorter than ~5 us: below that, the ~3 us start/ramp/drain cost of a DRAM-touching kernel and the
    0.2-0.7 us inter-kernel gap exceed 30-60% of the work.
16. Barriers and in-list event signal/wait are nearly free on the device (<= 0.15 us); the cost is host-side append time
    (signal event +0.28 us, event reset 0.32 us). Prefer in-order lists without per-kernel events.
17. Independent small kernels in an out-of-order list (no barrier) run concurrently (16 one-WG kernels: 0.09x the
    serial time). Use OOO segments only for genuinely independent low-occupancy work; otherwise fuse.
18. Do not create/destroy event pools inside a window (0.4 ms each). Do not poll with zeEventQueryStatus in tight loops
    (1.5-3.3 us per call); a host-USM flag written by the GPU is cheaper (1.2 us, TRANSPORT-B 4). Do not host-reset
    kernel-timestamp events in bulk on the hot path.

Host and CPU placement
19. Pin each card's submission thread to its own die: card 0 -> CPUs 0-7/16-23, card 1 -> 8-15/24-31. Remote-die
    submission is 5-40% slower per call and 40-70% slower for execute+sync.
20. A host round trip (append + host sync) costs ~6-7 us plus ~8 us to restart an idle device (TRANSPORT-B 3); keep
    zero host syncs inside a window.

Transfers and power
21. D2H is 4.5-4.7 GB/s on BOTH cards with both engines (H2D 14 GB/s); EU stores to host USM are worse (1.4 GB/s).
    This is a platform limit (X399/1950X, UMA-interleaved memory), not card 1. KV eviction of 1 GiB costs ~0.22 s;
    budget eviction from this number and restore at 14 GB/s.
22. Copy-engine D2H does not slow device DRAM kernels (601 vs 604 GB/s); H2D costs ~5% (573). Overlap eviction with
    decode on the copy engine freely.
23. Decode-like streaming runs at 2800 MHz and ~200 W without throttling; only dense DPAS hits the 230 W PL2 cap
    (-2% clock). Clocks are not a decode limiter; no warm-up is needed beyond the first kernel (+8-13% after idle).

Other
24. sub_group_reduce_add costs ~30 ns latency / ~7x an FMA in throughput; generic sub_group_shuffle ~21x. Reduce once
    per output row, use shuffle_xor or broadcast, and pass partial sums through SLM (atomics 23 ns, barrier 36-77 ns)
    only at the end of a kernel.
25. Cross-card: no peer atomics (not advertised). Keep the push + kernel-boundary event rules in TRANSPORT-B 4c.
    Host-USM device atomics do work (1.2 us each, verified counts) if a host-visible counter is ever needed.

## 3. Limits of this record
- GEMV kernels are speed probes on random data, not verified numerics; the q3 kernels mirror IQ3_XXS load and lookup
  structure (98 B blocks, 256-entry grid, 7-bit sign index, 4-bit scale) but use a synthetic table.
- Bandwidth rows assume one kernel owns the card. Concurrency between kernels on one card was only tested for 1-WG
  kernels in OOO lists.
- fp16 vector rate equals fp32 here because IGC emitted SIMD16 hf mad; packed SIMD32 fp16 was not attempted.
- Not measured: cross-card atomics (not advertised), DRAM bandwidth with both cards streaming while pushing over PCIe,
  PCIe AtomicOps capability bits (lspci needs root).
