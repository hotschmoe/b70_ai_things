# Handoff: decode-side PUSH all-reduce -- research + implementation session

This is a self-contained briefing for a FRESH session (assumes zero context). Goal: extend our custom
PUSH all-reduce from prefill-only to DECODE by making it SYCL-graph-capturable. Created 2026-06-24.

REPO: `/mnt/vm_8tb/github/b70_ai_things` (run LOCALLY on the box `b70s4dayz` as user `hotschmoe`, NOT over
SSH). Read `AGENTS.md` first for standing rules + GPU discipline. Runtime data root is `/mnt/vm_8tb/b70`
(models, caches, `gpu.lock`). Default serve image `vllm-xpu-env:int8g` for W8A8.

## Mission
Make the custom "push" all-reduce GRAPH-CAPTURABLE so the per-token DECODE all-reduces also use the fast
posted-write transport, instead of falling back to oneCCL inside the SYCL graph. Today push-ar is
capture-gated to PREFILL only; decode is the remaining headroom.

## Background (read docs/P2P_GPU.md J.2, J.7-J.12, J.14, J.17, J.21; contrib/vllm_push_allreduce/README.md)
- On our cross-die PCIe-Gen3 dual-B70 box, peer PUSH (posted write) = ~11 GB/s vs peer PULL = ~3.24 GB/s
  (J.2). We hand-rolled a 2-rank PUSH all-reduce (Level-Zero IPC, `P2PACCESS=0`, so it dodges the H.13
  oneCCL P2P wedge) that BEATS oneCCL: decode all-reduce ~34-45 us vs ~85 us; prefill ~10 vs ~9.4 GB/s.
- It monkeypatches `XpuCommunicator.all_reduce` (`contrib/vllm_push_allreduce/_push_ar_patch.py`). The
  C++/SYCL op is `scripts/106_xpu_push_ar_torch.cpp` -> `libxpu_push_ar_torch.so` (prebuilt in
  `contrib/vllm_push_allreduce/prebuilt/`; C-ABI `ar_setup_torch/ar_exchange/ar_allreduce_ptr_dt/
  ar_teardown`; runs in torch's L0 context via `torch.xpu.current_stream().sycl_queue`, operates on tensor
  `data_ptr()`s).
- PROVEN WIN, prefill only: 27B-W8A8 TP=2 GRAPH=1 with `PUSH_AR_MIN_NUMEL=65536` (capture-gated, prefill-only
  push; decode all-reduces stay on oneCCL inside the graph) = 3.8x prefill TTFT, +80-109% agg throughput vs
  oneCCL, decode UNCHANGED (~25 t/s). That decode parity is the gap this session attacks.

## The core problem (this is the whole task)
The push op is NOT SYCL-graph-capturable because its rank-sync uses a HOST barrier: a CPU spin barrier +
host `.wait()` + a ctypes call (see `_push_ar_patch.py` lines ~13-14, and the barrier in `106_*.cpp`). Under
GRAPH=1 PIECEWISE capture, host-side ops can't be recorded, so the captured decode region can't contain the
push -- hence the `PUSH_AR_MIN_NUMEL` gate that routes small (decode-sized) all-reduces back to oneCCL. To
accelerate decode you must replace the host barrier with a DEVICE-SIDE / graph-recordable sync that records
cleanly into the XPU graph and is correct on Xe2.

## Known dead-end (do not repeat without a new idea -- docs/P2P_GPU.md J.9)
The naive on-device barrier (device-flag fusion: peer-write a flag, spin-wait on it inside the kernel) HANGS
on Xe2: a peer write issued from WITHIN a running kernel is NOT visible to a spinning kernel on the other
device ("Xe mid-kernel peer-write invisibility"). J.9 also tried cross-queue L0 events and got a 1.36x decode
latency improvement (~44 us) -- a partial result worth re-examining as a capture-safe signal.

## Directions to explore (triage; verify each in a microbench BEFORE touching the serve)
1. Cross-queue Level-Zero events as the rank sync (extend J.9's 44us result) -- can a zeEvent signal/wait pair
   be recorded into the SYCL graph and survive replay? Most promising lead.
2. Restructure so the barrier falls on a graph PIECE boundary: split the all-reduce into capturable compute
   pieces with the (small, cheap) sync ejected to a splitting_op, so decode still gets push bandwidth for the
   data movement even if the signal is eager. Measure whether that nets a decode win.
3. A SYCL-graph-native barrier / command-buffer signal primitive, if Xe2/Level-Zero exposes one the
   torch.compile XPU-graph path can record.
4. If fully-capturable is impossible, quantify the best partial: push for the data copy + a minimal
   capturable handshake, vs the oneCCL decode baseline.

## How to build / run
- Rebuild the `.so` if you touch the C++: see `scripts/108_serve_push_ar_ab.sh` (builds in the int8g image
  via `icpx -fsycl ... -lze_loader -lrt`), or `REBUILD_SO=1`. Keep `contrib/.../prebuilt/` in sync if changed.
- Microbench the collective in isolation FIRST (the J.7-J.12 work used standalone 2-process scripts; look for
  `scripts/` numbered around those entries). Do not debug capture inside a full serve.
- End-to-end A/B serve: `scripts/108_serve_push_ar_ab.sh` (27B-W8A8 TP=2, push vs oneCCL). The shelf opt-in is
  `PUSH_AR=1` on `rdy_to_serve/qwen36-27b-w8a8-sqgptq-mtp/serve.sh`. Success target config is GRAPH=1 with
  `PUSH_AR_MIN_NUMEL=0` (push ALL all-reduces incl. decode) serving COHERENTLY.
- Note (J.15): loading the `.so` at interpreter startup inits Level-Zero before vLLM's XPU init and breaks
  GRAPH=1 model load -- the patch DEFERS the dlopen to the first all_reduce. Preserve that.

## Guard + GPU discipline + wedge reality (critical -- AGENTS.md "GPU Discipline" + P2P_GPU.md J.19/J.20)
- ALL GPU work goes through `./bin/gpu-run` (locks both cards for TP=2).
- A wedge guard wraps the TP>1 serve path: `bin/xpu-health` (per-card probe), `bin/xe-reset`, and lib.sh
  layers (pre-flight probe, graceful teardown, post-verdict). Set `B70_AUTO_RESET=1`.
- THIS BOX CANNOT BE RECOVERED BY xe-reset -- `xe` is the display driver, so `modprobe -r xe` always fails
  "in use". A wedge needs a HUMAN REBOOT (`sudo reboot`). So: chaining crash-prone TP=2/capture starts risks a
  wedge that costs a reboot. Run carefully, one capture experiment at a time, watch for empty-output serves
  (the captured-numerics-broken signature), and budget reboots.
- A coherence-gated gen probe returning EMPTY while /health is green = numerically broken capture. The
  single-card xpu-health probe does NOT catch collective-state degradation (known guard gap).

## Success criteria
- A microbench showing a capture-safe (graph-recordable) 2-rank push all-reduce, correct under replay.
- Then: 27B-W8A8 TP=2 GRAPH=1 `PUSH_AR_MIN_NUMEL=0` serves COHERENTLY (coherence gate OK, not empty) and
  improves DECODE t/s vs the oneCCL-decode baseline (~25 t/s at IN=2048 c1), ideally toward the ~2x per-op
  latency edge (34-45us vs 85us). Bench A/B at IN=512 and IN=2048, CONC 1/2/4/8.
- Log every experiment to JOURNAL.md (config -> command -> result -> verdict) and a new P2P_GPU.md J.xx.

## Do NOT
- Do not run `CCL_TOPO_P2P_ACCESS=1` in a TP>1 serve (wedges the box; reboot-only recovery).
- Do not chain multiple crash-prone capture starts without confirming the box is healthy between them.
- Do not just eject the all_reduce to a splitting_op and call it done -- that runs it EAGER and defeats the
  decode-capture goal; it is only acceptable as an explicitly-measured partial.
