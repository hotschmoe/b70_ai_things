# 03 - Parallel-agent GPU kernel dev on the Arc Pro B70

Status: PROPOSAL for review. Decides how 3-4 agents share the single B70 while
doing kernel work (e.g. the fused INT4 MoE XPU kernel). Wires `scripts/gpu-run`.

## TL;DR

- Agents edit + compile in parallel (CPU-bound, no GPU contention). The GPU is
  idle most of the time; only short, occasional test/bench runs touch it.
- So: SHARE the full card, and serialize ONLY the GPU run via a one-line `flock`
  lease (`scripts/gpu-run`). Do NOT statically partition with SR-IOV for this.
- SR-IOV is available (4 VFs) but is the wrong tradeoff here -- see "When to flip".

## What the host actually reports (verified 2026-06-20 on the host; pre-migration kernel 6.18-Unraid)

```
Card     : Intel Battlemage G31 [Arc Pro B70]  (PCI 0000:44:00.0, 8086:e223)
Driver   : xe   (new driver, not legacy i915)
Kernel   : 6.18.33-Unraid
SR-IOV   : sriov_totalvfs = 4    (up to 4 virtual functions supported)
           sriov_numvfs   = 0    (OFF -- running as one full 24GB card)
Devices  : /dev/dri/card0 + /dev/dri/renderD128   (single PF, no VFs)
```

Sharing != partitioning. One GPU shows up as `/dev/dri/renderD128`; many
processes/containers can open it at once and the Level Zero / OpenCL runtime
time-slices between contexts. Multiple agents already run kernels on the same
card with `--device /dev/dri` and ZERO special setup. SR-IOV is about hard
isolation + static VRAM carve-outs for VMs, not about enabling sharing.

## Topology

```
                                          +-------------------------+
  agent-A (worktree A) --[edit/compile]-->|                         |
  agent-B (worktree B) --[edit/compile]-->|   shared /dev/dri       |
  agent-C (worktree C) --[edit/compile]-->|   [ Arc Pro B70 24GB ]  |
  agent-D (worktree D) --[edit/compile]-->|                         |
        |                                 +-----------+-------------+
        |  GPU test/bench run only:                   ^
        +----> gpu-run = flock(gpu.lock) -------------+
               (exactly one holder at a time, ~seconds)
```

## Why serialize the GPU run (and only that)

```
TIMELINE  (edit/compile = parallel & free;  TEST = takes the lease via gpu-run)
  time --------------------------------------------------------->
  A:  [edit][compile........][TEST]            [edit][compile..][TEST]
  B:  [edit..][compile...........][TEST]  [edit][compile...][TEST]
  C:  [compile......][TEST]   [edit][compile.........][TEST]
  D:       [edit][compile........][wait!][TEST]  [edit][compile...]
                                   ^
  lease owner:  ... C ...  A   B   D ...           A      B
                most of the time the lease is FREE; D waited a few s behind B
```

The `wait!` is the entire cost of contention: a few seconds, rare. In exchange
the lease buys two things a shared free-for-all does not:

1. Clean perf numbers. Two kernels timed at once => both numbers are junk
   (noisy neighbor). The lease guarantees exclusive timing for any recorded run.
2. Blast-radius control. A WIP kernel that wedges the device does so while it
   holds the lease, not on top of another agent's run.

## scripts/gpu-run

Wrap every GPU touch:

```
scripts/gpu-run python test_int4_moe.py        # correctness: lease guards a hang
scripts/gpu-run ./bench_fused_moe --iters 200  # perf: lease => exclusive timing
scripts/gpu-run --status                        # who holds the GPU now (or "free")
```

Env knobs (defaults shown):

```
B70_GPU_LOCK          /mnt/vm_8tb/b70/gpu.lock   # SHARED host path, bind-mount into every container
B70_GPU_LOCK_TIMEOUT  600                        # max seconds to wait; 0 = wait forever; exit 124 on timeout
B70_AGENT             $HOSTNAME                  # label shown in logs / --status
```

Lockfile is wired on the host: `/mnt/vm_8tb/b70/gpu.lock` (created 2026-06-20).
flock is advisory on the inode, so all containers that bind-mount the SAME file
honor the same lease.

## Per-agent container launch (each its own worktree, all share card + lease)

```
docker run --rm -it \
  --device /dev/dri:/dev/dri --group-add video \
  -e B70_AGENT=agent-A \
  -v /mnt/vm_8tb/b70/worktrees/agent-A:/work \
  -v /mnt/vm_8tb/b70/gpu.lock:/mnt/vm_8tb/b70/gpu.lock \
  vllm-xpu-env:int8
```

Notes:
- `--group-add video` so the container user can open the render node.
- Bind-mount the lockfile (not just its dir) so the inode is shared.
- Give each agent a separate worktree dir to avoid build-cache / file clobber.
  Compile caches (CCACHE_DIR / TMPDIR) should be per-agent OR a shared ccache --
  decide during review.

## Escape hatch: a buggy kernel hung the card

A GPU hang can wedge the device for everyone (shared OR SR-IOV -- a VF hang may
still force a PF reset). Recover on the host by rebinding the PCI device:

```
echo 0000:44:00.0 > /sys/bus/pci/drivers/xe/unbind
echo 0000:44:00.0 > /sys/bus/pci/drivers/xe/bind
```

(Any in-flight container GPU contexts die; restart affected agents' runs.)

## When to flip to SR-IOV (4 VFs are available)

```
  shared + gpu-run lease   <-- DEFAULT for this project's profile
        |                       (mostly-idle GPU, short rare tests, full 24GB per job)
        |  flip ONLY if the profile changes to:
        v
  4x SR-IOV VFs            <-- agents run GPU work CONTINUOUSLY & concurrently (real
                               thrash), OR you need hard per-agent VRAM isolation.
```

Costs of flipping:
- Static VRAM: 24GB / 4 = ~6GB per VF. Fine for tiny arithmetic, but no single
  job can grab the full card, and a real multi-expert MoE test may exceed 6GB.
- Does NOT help compile parallelism (already free) and may not survive a hang.

Enable / disable (host):

```
echo 4 > /sys/bus/pci/devices/0000:44:00.0/sriov_numvfs    # spin up 4 VFs
echo 0 > /sys/bus/pci/devices/0000:44:00.0/sriov_numvfs    # tear down
```

After enabling, new `/dev/dri/renderD129..` nodes appear; assign one VF per
container instead of sharing renderD128.

## Open questions for review

- ccache: per-agent dirs vs one shared cache (shared = faster cold builds but
  possible lock contention in the cache itself)?
- Should correctness-only tests skip the lease to cut even the rare wait, and
  reserve `gpu-run` for recorded perf runs? (Leaning no -- the hang-isolation
  benefit applies to WIP kernels too, which is exactly correctness testing.)
- Lease timeout default 600s -- is any single kernel microbench longer than that?
```
