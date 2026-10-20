# 2O2 -- KV parking to host RAM without recompute: Strata vs vLLM, G1 review, spec

Author: Opus agent (2O2), 2026-10-20. CPU/source only. No GPU, no commits,
no services touched. Tags: [V] = read in code or log, [I] = inferred.
Source trees:
- Strata: /mnt/vm_8tb/b70/build/strata-ab/src-ep (night-20261016). On the
  SYCL build `sycl/src/...` overrides `src/...` (sycl/CMakeLists.txt:93-105,257).
  G = sycl/src/program/generate.cpp, CS = sycl/src/core/conversation_state.cpp,
  SNAP = sycl/src/core/conversation_snapshot.cpp,
  CCH = include/strata/core/conversation_cache.hpp.
- vLLM: /mnt/vm_8tb/b70/build/vllm-flashnext/vllm (b70/flashnext, upstream
  base 2296c12e, 9 fork commits). OFS = vllm/distributed/kv_transfer/
  kv_connector/v1/offloading/scheduler.py, STM = vllm/v1/core/
  single_type_kv_cache_manager.py.
- vllm-xpu-kernels: /mnt/vm_8tb/b70/build/vllm-flashnext/vllm-xpu-kernels
  (19c6ddc; installed wheel 0.1.15.4).

## 0. Verdict (short)

1. G1's run at 07:35 is a real, valid first step. Ornith, one card,
   PIECEWISE: a 30464-token prompt was evicted from GPU and restored from
   host RAM with greedy text equal to the control. TTFT was 0.73 s restored,
   0.27 s resident and 5.18 s cold. 29376 tokens came from the host and 1088
   were recomputed (logs/g1-evict.json) [V].
2. That test is a resend of the same prompt. It is not a conversation. On a
   real multi-turn conversation, upstream vLLM recomputes the whole previous
   assistant reply plus up to one block:
   - `offload_prompt_only` defaults to True (vllm/v1/kv_offload/base.py:714).
   - Align-mode Mamba never records a GDN state at a decode position
     (STM:2080-2097).
   - The finish-time partial-tail hand-off only fires when the request ends
     exactly at its prompt boundary (STM:1931-1945).
   Strata replays 0 tokens of the prior reply.
3. The XPU copy path is about 5x slower than Strata per byte. G1 loaded
   666,009,600 B in 0.461 s (1.44 GB/s). Strata restores 3.17 GB in about
   0.415 s (about 7.6 GB/s). The cause [V] is that `xpuAsyncMemcpyBatch`
   stages every H2D copy through a single-threaded CPU memcpy and does a
   synchronous `.wait()` per fragment for D2H to non-USM memory
   (vllm-xpu-kernels csrc/utils/mem_cpy.cpp:182-292). A 200K Flash-Next
   restore at that rate takes about 3.7 s with bf16 KV, against Strata's
   0.42 s.
4. Missing for Strata parity. Each item is a work item with a size, not a
   blocker:
   - a. Snapshot the end-of-request state: GDN plus the partial last
     attention block plus the MTP block plus PLE, at `num_computed_tokens`.
   - b. An exact-tail index and an exact (non-aligned) external hit.
   - c. Eviction at conversation granularity (4 slots) instead of per block.
   - d. A direct-DMA XPU copy op on a copy-engine queue.
   - e. Register the Flash-Next PLE conv state as a Mamba-style KV group
     (today it is a private side tensor, so any prefix hit is wrong for
     Flash-Next).
   - f. int8 KV on the XPU attention backend, to halve the bytes.
   Sections 5-7 give the spec.

## 1. Strata conversation cache (what we must match)

### 1.1 Topology and slots [V]

- The production engine runs `--batch 2`: two slot sessions on CUDA0, 2.82
  GiB each, plus the main prompt-reading session (engine.log lines 1,
  14-15).
- Card 1 holds only routed experts ("card 1 holds 17952 experts"). All
  KV and state live on card 0. There is no per-card split.
- Batch-MTP is on, so each slot has its own draft KV (G:4024).
- `--kv-grow` turns itself off under `--batch` (G:6446). Elastic KV is
  still a plan (docs/20261016_next_investigations.md:10-52).
- Flags: `--conversation-cache-mib 32768 --conversation-cache-slots 4`
  (strata/flash-next/dd/hotschmoe-dd.json). Budget and slots are wired at
  G:7772-7774.

### 1.2 What one parked conversation holds

`SavedConversation` (CCH:155-191) holds `live` (CCH:23-50), `kv`, and
`checkpoints` [V]:

| piece | layout | bytes |
|---|---|---|
| token ids (int32) + image keys | `ids`; position = ids.size() | 4/token |
| GDN running state, 36 layers | fp32 recurrence 48 x 128 x 128 + conv history 10240 x 3 (CS:119-141) | 3,268,608/layer; 117,669,888 total |
| PLE/HC n-gram history | NG_HIST 9 x NG_HC_DIM 10240, fp32 (CS:129) | 368,640 |
| QSA indexer tail, per layer x 12 | 3 x 128 fp32 tail + 512 B pooled row + block_pos (CS:130-134) | about 2 KB/layer |
| full-attn KV, 12 layers | int8 K and V, 2 heads x 256, plus 2-byte scale per 64 elements (SNAP:30-74, 56-58) | 1056 B/token/layer |
| QSA pooled indexer keys | fp32, (n/4+1) x 128 (SNAP:48,59) | about 128 B/token/layer |
| MTP draft layer KV | int8, no pooled rows (CS:359) | 1056 B/token |
| ple_prev (last 2 ids) | rebuilt from ids on restore (CS:202-204) | 0 |

- Total running state is about 118 MB (comment at G:1406).
- Per token: 12 x (1056+128) + 1056 = **15,264 B/token**. This is
  confirmed by the log slope between restores of 9155 and 9535 tokens.
- A 200K image is 3.05 GB of KV plus 118 MB = **3.17 GB**. Night lab
  measured payloads of 3.16-3.17 GB (docs/20261016_night_lab.md:254).
- Checkpoints are extra copies of the running state, about 118 MB each.
  The main session keeps 6 (`--prompt-cache` default 6, G:684). A slot
  keeps 1 (G:12300-12305).
- Not saved: sampler state and the MTP hidden state. The draft KV is
  rewound with `mtp.kv_restore(resume)` (G:10578).

### 1.3 When it parks

- Accounting is oldest-first eviction over a deque, skipping pinned
  entries, under both a byte budget and a slot count (CCH:226-338, 400).
- `put()` drops superseded copies of the same conversation first
  (CCH:353-395).
- A physical-RAM gate checks MemAvailable minus
  `--conversation-cache-min-free-mib` (default 2560) before and after the
  capture (G:7944-7988).
- Triggers:
  - a. `park_current`: a new request does not continue the main
    session's live state (G:10395-10400).
  - b. `park_slot(b)`: an admission is about to overwrite an idle slot
    that still holds a conversation (G:9394-9443, called at G:12264). This
    saves the decoded reply. It is the reason replay is 0 (G:9394-9401;
    before it, 255 of a 256-token reply were replayed,
    docs/20261013_flashnext_goals.md:201).
  - c. BYIELD preemption of a long prompt read (G:10965-10990).
- Never parked: a request that has just ended. Its slot stays resident
  until it is needed (G:9149-9152). A running slot is never evicted
  (G:9404).
- Slot choice is longest held prefix, else empty, else LRU
  (serve/server.py:1810-1840).

### 1.4 Copy path [V]

- Host memory is plain pageable `std::vector<uint8_t>` in 16 MiB segments
  (conversation_buffer.hpp:16-49), with glibc hugetlb tunable on. It is
  not pinned and not USM.
- Each segment, part and layer gets one `in_order_queue().memcpy(...).wait()`
  on the default compute queue (SNAP:114-129, 181-190, 222-225; CS:42-52,
  167-206). There is a device-wide wait before every snapshot (CS:32-40).
- There is no overlap. Park and restore run inline on the serve thread
  [I from control flow].
- Incremental park: after a restore the host buffers are retained, and
  only pages from the first rewritten token on are copied again
  (SNAP:166-191; log `reused_kv_bytes`).
- `STRATA_NIGHT_PARK_MOVE=1` moves checkpoints instead of copying them
  (G:7956-7967). Park dropped from 190 ms to 49 ms (night_lab:198-199).
- `STRATA_NIGHT_ADMIT_D2D=1` copies device-to-device between the main
  session and a slot (G:9196-9241). Admission dropped from 230 ms to 1.0 ms
  (night_lab:163). A 174K-token slot copy takes 10.6 ms.

### 1.5 Restore matching [V]

- Matching is exact token equality, with no hashing (CCH:211-224). The
  parked ids (live, or any checkpoint) must equal the prompt's leading
  ids, with `n < prompt.size()`. Image keys and the steering flag must
  also match.
- The longest match wins. A parked image wins only if it is strictly
  longer than the resident candidates (G:10336-10370).
- No mid-image truncation: GDN is valid only at the live end or at a
  checkpoint. A checkpoint hit restores the whole image, then the
  checkpoint's running state on top (CS:435-446; G:10497-10570).
- The engine then reads `[resume, n)` and logs `RESUME`/`fresh`
  (G:10594, 12238).
- "Replay" is a harness metric: `max(0, prev_total - 1 - cache_n)`
  (strata/flash-next/bench_context_four.py:198-205). The final emitted
  token is never fed, which is the same rule as vLLM's `num_tokens - 1`.

### 1.6 Measured numbers

- Native full restores took 410.7-422.3 ms for 3.16-3.17 GB payloads, with
  max prior-tail replay 0 and peak RSS 17.19 GB. TTFT p95 was 2.99 s
  resident and 4.67 s after a RAM restore (night_lab:254-256). That is
  about 7.6 GB/s, against an 8.5 GB/s H2D probe.
- From the live engine.log (589 parks):
  - median park 68.7 ms
  - 174K-token slot park 972 ms
  - 174K-token main park 1588 ms (includes 6 checkpoints)
  - restore of 16,176 tokens in 61 ms
  - restore of 82,624 tokens in 233 ms
- The ~0.4 s restore comes from three things. There is no recompute of
  the prior reply. Payloads are dense int8 KV (half of bf16). Large
  synchronous memcpys stay close to link rate even from pageable memory.
  Strata is not faster because of any overlap trick.

## 2. vLLM main equivalent machinery

### 2.1 Connector API (kv_connector/v1/base.py) [V]

- Scheduler side:
  - `get_num_new_matched_tokens` :516 (`(None, _)` means ask again;
    True means async)
  - `update_state_after_alloc` :569, `build_connector_meta` :595,
    `request_finished` :631
  - `SupportsHMA.request_finished_all_groups` :108
  - `register_finished_partial_tail` :652 (called from
    scheduler.py:3107-3134 only when `is_kv_producer`)
- Worker side: `register_kv_caches` :292, `start_load_kv` :333,
  `wait_for_save` :386, `get_finished` :398, `handle_preemptions` :326.
- `supports_divergent_local_hybrid_hits` :189 defaults to False.

### 2.2 OffloadingConnector + vllm/v1/kv_offload [V]

- **Groups.** It offloads every `prefix_cacheable` group
  (kv_cache_interface.py:1528-1536). MambaSpec is prefix-cacheable, so
  GDN goes along (offloading/config.py:36-38). Hybrid needs prefix
  caching plus `mamba_cache_mode=align` (config.py:91-95; cache.py:63 now
  only allows align or none).
- **Keys.** `make_offload_key(block_hash, group_idx)` per chunk
  (OFS:420-437). One CPU slot is `worker_kv_bytes_per_block`
  (config.py:141-154).
- **Stores.** Stores run every step. They are queued in `wait_for_save`
  and submitted at the next step's `start_kv_transfers`
  (offloading_connector.py:134-138, worker.py:246-271).
  - Consequence: the last store of a finished request waits for the next
    engine step. G1's test needed "flush" requests (evict_test.py).
  - Finished requests are clamped to `max(num_prompt, num_tokens-1)`
    (OFS:1673).
  - `offload_prompt_only` defaults to True (kv_offload/base.py:710-716).
- **Mamba handling.**
  - An align-mode Mamba group is treated as a one-chunk window
    (OFS:148-150, 288-291) and skipped by the positional store (OFS:1741).
  - GDN state is stored only from `boundary_state_offloads` hand-offs:
    aligned boundaries (OFS:1415) and the prompt-tail partial (OFS:1487),
    which come from `MambaManager.cache_blocks` (STM:1885-2018).
  - G1's metrics fit "27 attention blocks + 1 GDN page" per prompt: the
    load was exactly 27 x 22,282,240 + 64,389,120 = 666,009,600 B.
- **Partial tails.** Disabled whenever an EAGLE/MTP group exists, when
  `blocks_per_chunk > 1`, or when DCP is on (OFS:300-315). Hits are capped
  at `num_prompt_tokens - 1` (OFS:1035-1094).
- **Hit window.** Rounded down to the chunk and capped at num_tokens-1
  (OFS:844-853). Loads are always async and load exactly one GDN state
  block (OFS:1220-1361).
- **CPU backend.**
  - `cpu_bytes_to_use` is required (cpu/spec.py:100-104).
  - The policy is per-key lru or arc (policies/factory.py:87-90).
  - On XPU the pool is a /dev/shm mmap (cpu/spec.py:171-176), registered
    with `syclex::prepare_for_device_copy` (gpu_worker.py:203-268;
    host_register.py:69-72).
  - The G1 log shows a 17.18 GB mmap inside a container with
    `--shm-size 16g`.
- **Copy worker.** Each transfer gets a pooled stream that waits on
  compute and calls `ops.swap_blocks_batch` (gpu_worker.py:39-65,
  674-716).

### 2.3 XPU swap path [V]

- `swap_blocks_batch` calls `xpuAsyncMemcpyBatch` (cache.cpp:1373;
  mem_cpy.cpp:182-292).
- H2D: if the source is `host` or `unknown`, everything is first copied
  by one CPU `std::memcpy` into a staging buffer from the torch host
  allocator, then DMA'd. This applies even when the source is already
  USM host (mem_cpy.cpp:205-273).
- D2H: if the destination is `unknown` (the mmap+register region is not
  USM [I]), each fragment does `.wait()`, synchronously on the current
  stream's queue (mem_cpy.cpp:276-289; utils.h:38-42).
- Measured on G1:
  - Load 1.44 GB/s (666 MB / 0.461 s).
  - Store about 4.3 GB/s (864 MB / 0.200 s, metrics.py:108 log lines). The
    store number is a per-interval counter.

### 2.4 Hybrid prefix caching

- **Hit search.** MambaManager requires align mode (STM:1437-1440).
  `find_longest_cache_hit` searches right to left for one state block
  (STM:1463-1542). The hybrid coordinator runs a fixed-point loop over
  groups (kv_cache_coordinator.py:796-931).
- **What align mode caches.** It caches the state at `i*block_size` and
  at the last token of each step (cache.py:189-196). Chunks are forced
  onto block boundaries plus a prompt-tail checkpoint
  (scheduler.py:536-579).
- **Checkpoint position.** `get_mamba_prefill_checkpoint_position`
  (kv_cache_interface.py:1150-1162) gives `floor((n-1)/h)*h`, or
  `floor(n/h)*h - h` with the EAGLE drop.
- **Decode is excluded.** A decode-position state is never offered: "during
  decode the target is the running state block, mutated in place"
  (STM:2080-2097).
- **Block sizing.** The attention block grows until one attention page
  covers one GDN page, rounded to 64 on XPU (platforms/interface.py:
  1060-1132; platforms/xpu.py:453-500).
  - Ornith fp16: 1088 tokens (G1 log line 98).
  - Flash-Next, computed from hf-flashnext config.json. One GDN page is
    3 x 10240 x 2 + 48 x 128 x 128 x 4 = 3,207,168 B.
    - bf16 KV (2 x 256 x 2 x 2 = 2048 B/token/layer): 64 x cdiv(3207168,
      131072) = **1600 tokens** [I, arithmetic].
    - int8_per_token_head (about 1040 B): **3136 tokens** [I].
  - Without an end-of-request snapshot, every turn replays up to one block
    (1600-3136 tokens) **plus** the whole previous reply.

### 2.5 Upstream status [V by absence in tree and git log]

None of issue 52773 or PRs 58413, 61019 and 61282 is merged
(reports/04_ecosystem.md:135-146):

| item | what it addresses |
|---|---|
| issue 52773 | divergent hit: full-attention KV outlives the GDN boundary, no backfill |
| PR 58413 | Mamba state at hash granularity under MTP |
| PR 61019 | partial tails with EAGLE/MTP groups |
| PR 61282 | keep the prompt-boundary state until the request ends |

Merged and relevant: #50507 and #49502 (partial-tail reuse and offload),
#44599 (Mamba CPU offload), #51468 and #50344 (divergent hits for capable
connectors), and #52832 and #60695 (Mooncake finish-time partial tail;
mooncake/store/scheduler.py:600 is the reference implementation of
`register_finished_partial_tail`).

None of the four addresses the decode-end state. Our gap (2) is not
covered upstream.

### 2.6 Other upstream facts

- **SimpleCPUOffloadConnector** is HMA-capable, but it is CUDA/ROCm only
  (torch.cuda.Stream, cudaHostRegister, cuMemcpyBatchAsync;
  simple_kv_offload/worker.py:170, cuda_mem_ops.py:31-130). It is not
  usable on XPU.
- **Graph capture.**
  - Connector load and save run on the host around `_model_forward`
    (kv_connector_model_runner_mixin.py:67-104; V2 runner gpu/kv_connector.py:80-170).
  - Per-layer hooks are no-ops in OffloadingConnector
    (offloading_connector.py:122-132). It does not force PIECEWISE. FULL
    and FULL_DECODE_ONLY are compatible [V/I].
  - Transfers order against compute with `wait_stream`.
- **int8 KV.**
  - `int8_per_token_head` stores fp32 scales inline in the page, so a
    raw page copy carries them (triton_attn.py:306-317, 412-473).
  - It is supported only by TRITON_ATTN. FLASH_ATTN on XPU supports only
    auto/fp16/bf16/fp8 (flash_attn.py:288-294).
  - Per-tensor fp8 scales are per-layer weights (attention.py:136-139)
    and need no copy.

### 2.7 Flash-Next on the XPU fork: state outside the KV manager [V]

- `vllm/models/qwen4_exp/xpu/model.py:278-303, 380-403` keeps the PLE
  dilated-conv history (`_conv_state[num_slots, 9, 10240]` bf16) and the
  n-gram token history (`_token_hist[num_slots, 2]`, CPU) as private
  module tensors. They are indexed by the **GDN layer's** state slot ids
  and `has_initial_state` (model.py:419-430).
- The KV manager does not know about them:
  - align-mode Mamba copies, CoW, and the offload connector do not move them;
  - a prefix hit (GPU or host) resumes GDN from a cached block, but reads
    PLE history from whatever last wrote that slot index.
- **Every prefix hit on Flash-Next XPU therefore has wrong PLE state**
  [I, by construction]. It also does host syncs (`.to("cpu")`,
  model.py:432-435) that block FULL capture of prefill.
- Upstream AMD and NVIDIA variants register PLE as a short-conv
  MambaSpec (amd/ple_layer.py:443-520; amd/model.py:730-750) and QSA as
  FullAttentionSpec plus a CircularBufferSpec compressor
  (amd/qsa.py:323-331; common/qsa_cache.py).
- The XPU model uses dense attention with no indexer cache (model.py:16,
  546-564, 829-830), so there is no QSA state to park yet.
- Once QSA lands on XPU, its compressor ring (CircularBufferSpec, not
  prefix-cacheable) is boundary state like GDN. Strata saves it (1.2).

## 3. Review of G1 (vllm/flashnext/kvoffload)

What works [V]:
- The hybrid contract check (connector.py).
- The byte model matches the serve log: GDN page 2,146,304 B, block 1088
  (layout.py).
- The eviction proof with greedy equality and host-source counters
  passed (logs/g1-evict.json, serve log logs/g1-serve.log).

Bugs and gaps, ordered by impact:

1. **The test does not exercise a conversation.**
   - evict_test.py resends the same token ids. A real next turn is
     prompt + reply + new user turn.
   - Two settings cap such a turn at the prompt-tail boundary, so the
     whole reply is recomputed:
     - `offload_prompt_only` is not overridden (serve_ornith.sh KV_JSON
       has only `cpu_bytes_to_use`);
     - no decode GDN state is stored (2.4).
   - The test cannot see this.
2. **The pass criterion accepts recompute.**
   - `tail_ok` allows `computed <= 2*block`, which is 2176 tokens.
   - The measured `restored_computed` was 1088, because a block-multiple
     prompt loses its last block (checkpoint_boundary).
   - Strata's metric is replay 0 of the prior prompt and reply.
3. **Copy path at 1.44 GB/s** (2.3). Not flagged in the journal. A 200K
   Flash-Next restore extrapolates to about 3.7 s with bf16 and about 2 s
   with int8.
4. **`Park` is not wired.**
   - park.py is a CPU model only.
   - The live pool is upstream per-key LRU over blocks. Under pressure it
     can evict the middle blocks or the single GDN page of a parked
     conversation and keep its other blocks. The leftovers waste RAM and
     produce short or zero hits.
   - There is no 4-slot rule and no pinning.
5. **Deferred final store.** The final store is submitted at the next
   engine step (2.2). On an idle server the reply is not parked until the
   next request arrives, and then it races that request's lookup. G1
   works around this with flush requests.
6. **shm sizing.** `CPU_BYTES` = 17,179,869,184 equals `--shm-size 16g`
   (xpu-run.sh:28) exactly. Any other /dev/shm user (IPC, torch) can
   SIGBUS. The 32 GiB target needs `--shm-size` of at least 40g and a
   container `--memory` above the pool plus RSS (90g today).
7. **The byte model does not carry over to Flash-Next.**
   - `upstream_pool_bytes` assumes a GDN page per block. That is
     pessimistic: the measured store has one per prompt tail.
   - fp16 KV is hard-coded. There is no MTP group, no PLE group and no
     int8 per-token-head layout.
   - The 4 x 200K Flash-Next budget is never computed (5.3).
8. **The contract check is fragile.**
   - It matches by type name (`"MambaSpec"`, `"GDN" in mamba_type`).
   - It does not require the EAGLE/MTP group (or a future PLE short-conv
     group) to be in the offload set. On Flash-Next a missing draft or PLE
     group would pass the check and resume with stale state.
9. **Docstring mismatch.** connector.py says `_max_loadable_tokens` bounds
   the hit to the shallowest group. That is right for a full eviction.
   The issue-52773 backfill matters once decode GDN states exist, because
   a GPU-resident attention prefix can then outlive the GDN boundary.
10. **Minor items.**
    - `block_size_from_log` takes the last match. A future
      "kv lcm block sizes" line with several numbers would parse as the
      first one.
    - The unittest transcript is not saved (1G2_audit.md F15).
    - `--max-num-seqs 2` with `NUM_GPU_BLOCKS 45` does not test
      concurrency with a parked restore.

## 4. What exactly is missing (work items, sized)

| # | item | where | size |
|---|---|---|---|
| W1 | End-of-request snapshot: offer the GDN running-state block at `num_computed_tokens` when the request finishes with `num_in_flight_tokens == 0` | fork: STM `MambaManager.finalize_partial_tail_offload` (STM:1931-1945) + flag | S (~40 lines) |
| W2 | `register_finished_partial_tail` for OffloadingConnector: pin the GDN block, the partial last attention block of every attention group (incl. MTP), and the PLE block; store as one atomic "tail record" | plugin (our connector) | M (~250 lines) |
| W3 | Exact-tail index + lookup: key = (hash of last full block, tail ids); hit = n_full + len(tail) when the prompt extends it | plugin | M (~150 lines) |
| W4 | Scheduler accepts a non-aligned external hit for an align-mode hybrid (state block = loaded tail GDN; next chunk realigns) | fork: kv_cache_coordinator / MambaManager allocate path | M (~80 lines + test); verify first, #50507 may cover it |
| W5 | Producer gate: make `finalize_partial_tail_offloads` run for `kv_both` (scheduler.py:3109) | fork, 1 line, or check is_kv_producer(kv_both) | XS |
| W6 | Conversation-granular eviction policy (4 slots, byte budget, pin while loading) | plugin: custom policy via policies/factory.py | M (~150 lines) |
| W7 | Idle flush of deferred stores (submit at finish, not next step) | plugin worker: submit inside `get_finished` when idle, or fork 5-line change in worker.py:246-271 | S |
| W8 | XPU copy op: direct H2D DMA from USM-host (no staging), async D2H to USM-host, on a dedicated copy-engine queue | b70 plugin custom op (or vllm-xpu-kernels patch) + pool allocated with sycl::malloc_host | M (~200 lines C++/SYCL) |
| W9 | Flash-Next PLE conv/token history as a MambaSpec short-conv group (port AMD ple_layer pattern); token history derived from ids | fork model file vllm/models/qwen4_exp/xpu/model.py | M |
| W10 | int8_per_token_head KV on the XPU attention path used by Flash-Next | separate track (kernel); halves park bytes and VRAM | L |
| W11 | MTP block drop: with W1-W3, set `disable_eagle_block_drop` or snapshot the draft group to `num_computed_tokens-1`; verify acceptance unchanged | config + test | S |

## 5. Implementation spec

### 5.1 Principle

Reuse upstream OffloadingConnector for every **full** block, keyed by hash,
which is already proven by G1. Add one **tail record** per finished
request. The tail record holds everything that is valid only at the exact
end position E = `num_computed_tokens` (= prompt + reply - 1, the vLLM and
Strata commit point):

- every attention group's partial last block (tokens
  `[floor(E/B)*B, E)`), including the MTP draft group;
- the GDN state block (conv + ssm, 36 layers on Flash-Next, 30 on Ornith);
- the PLE short-conv block (Flash-Next, after W9);
- later, QSA compressor rings.

A conversation is "parked" when its full blocks and its tail record are
both in host RAM. A restore hit equals E exactly. Strata's equivalent is
`live` (1.2).

Optional checkpoints (Strata keeps 1-6) are the aligned GDN pages upstream
already stores at prompt tails. They let a client that edits the last
turn fall back to an earlier boundary. Keep them, but count them in the
budget.

### 5.2 Data structures (plugin, vllm/flashnext/kvoffload/)

```
TailKey   = (parent_hash: bytes|None, tail_ids: tuple[int, ...])  # len < B
TailRecord:
    key: TailKey
    conv_id: int            # root block hash of the conversation
    end: int                # E
    full_keys: list[OffloadKey]   # chain of full-block keys (all groups)
    tail_slots: dict[group_id, CPUSlot]  # partial attn pages, GDN, PLE, MTP
    nbytes: int
    pinned: int             # >0 while a load is in flight
ConversationEntry:
    conv_id, last_used, records: list[TailRecord] (latest first),
    full_keys: set[OffloadKey], nbytes
ParkIndex:
    by_tail: dict[TailKey, TailRecord]
    by_conv: OrderedDict[conv_id, ConversationEntry]   # LRU, <= slots
    budget_bytes, slots (=4), resident_reserve
```

- The tail slots come from the same CPU pool (fixed-size slots of
  `worker_kv_bytes_per_block`). The page geometry is unchanged, so the
  copy worker needs no new layout.
- `conv_id` is the hash of block 0 (or the first tail id). Turns of one
  conversation share it, so the eviction unit is the conversation, like
  Strata's superseded-copy drop (CCH:353-395). A new tail record for the
  same conv_id supersedes the older one. The older one's full keys are a
  prefix and stay.

### 5.3 Hooks

Scheduler side (plugin `ParkingOffloadConnector(OffloadingConnector, SupportsHMA)`):

- `__init__`:
  - force `offload_prompt_only=False`;
  - require prefix caching + align;
  - require every prefix-cacheable group to be in the offload set (by
    spec class, not by name), including EAGLE/MTP and PLE groups;
  - log the group table.
- `get_num_new_matched_tokens(req, n_local)`:
  1. Call the parent. It returns full-block hits (n_full; GDN only
     loadable at a stored boundary).
  2. Walk `req.block_hashes` for the longest full-block chain present
     in the pool: P = k*B.
  3. Look up `by_tail[(hash_k, ids[P:P+t])]` for t = len(record tail),
     scanning the conv's records (at most a few). Accept if
     `P + t < len(prompt)` and the ids match.
  4. Return `max(parent, P+t) - n_local`, async=True. The full-block
     chain must be contiguous from n_local. Otherwise fall back to the
     parent.
- `update_state_after_alloc`: build one load job of full blocks
  `[n_local, P)` for attention groups, plus the tail slots into:
  - the request's partial last attention block,
  - its GDN state block,
  - its PLE block,
  - its MTP partial block.
  Pin the record.
- `register_finished_partial_tail(req, block_ids, offloads)` (W2): for
  E = req.num_computed_tokens:
  - build a store job copying GDN/PLE state blocks and the partial
    attention/MTP blocks into fresh tail slots;
  - return True so the blocks stay alive until `get_finished` reports the
    job done (the Mooncake pattern, mooncake/store/scheduler.py:600);
  - insert the TailRecord;
  - evict whole conversations (LRU, skip pinned) until it fits
    `slots` and `budget_bytes`.
- Full blocks of decode tokens are stored by the parent once
  `offload_prompt_only=False`.

Fork commits on b70/flashnext (each small, new flag default-off, rebase-clean):

- **F-A (W1+W5).**
  - Make `finalize_partial_tail_offload` return
    `(group, running_state_block, E)` when the cache config flag
    `park_finished_tail` is set and `num_in_flight_tokens == 0`. It
    currently returns None unless E equals the prompt boundary.
  - Do the same for the PLE short-conv manager (same class).
  - Let `kv_both` count as a producer for this call.
- **F-B (W4).** Allow an external hit at a non-aligned E for align-mode
  groups:
  - the state block of the request is the loaded tail;
  - the attention partial block is a private block;
  - the next prefill chunk ends at the next aligned boundary (the
    existing chunk forcing).
  First check whether the merged partial-tail machinery (#50507) already
  admits non-hash-aligned external counts. If it does, F-B shrinks to a
  test.
- **F-C (W7)** if needed: in kv_offload worker, submit pending store jobs
  in `get_finished` when `wait_for_save` saw no new step (idle flush).
- **F-D (W9).** Flash-Next XPU PLE as a short-conv MambaSpec layer (port
  of amd/ple_layer.py `get_state_shape`/`get_state_dtype` usage). Drop the
  private `_conv_state`/`_token_hist`. Derive the token history from
  `input_ids` and positions on device.

Worker side (plugin, same connector module):

- Allocate the CPU pool with `sycl::malloc_host` (USM host) through the
  b70 plugin op. Do not use mmap + `prepare_for_device_copy`. Keep the
  upstream shared-region path as a fallback flag.
- Replace `ops.swap_blocks_batch` in our worker subclass with
  `torch.ops.b70_kvpark.copy_batch(src_ptrs, dst_ptrs, sizes, dir)` (W8):
  - submit `memcpy` per contiguous run (merge adjacent block ranges
    first; a 200K conversation is about 125 runs per layer tensor at
    B=1600, or one run per layer if the pool keeps a conversation
    contiguous);
  - use a dedicated in-order queue on the copy-engine group, the same
    group Strata uses for expert copies (queue group 1). Return a
    sycl::event wrapped as a torch XPU event so the compute stream can
    `wait_event` only where a loaded block is consumed;
  - H2D: no staging. D2H: async into USM host, completion via event,
    never `.wait()` on the compute queue.
  - Target at least 7.5 GB/s per direction (Strata 7.6, probe 8.5).

### 5.4 Policy and budget

Flash-Next per-conversation bytes on vLLM [I, arithmetic from config.json]:

| piece | bf16 KV (today) | int8_per_token_head (W10) |
|---|---|---|
| 12 attn layers / token | 24,576 B | about 12,480 B |
| MTP layer / token | 2,048 B | about 1,040 B |
| GDN end state (36 layers, fp32 ssm) | 115.5 MB | 115.5 MB |
| PLE history | 0.18 MB | 0.18 MB |
| per 200K conversation | about 5.44 GB | about 2.82 GB |
| 4 parked + 1 aligned checkpoint each | about 22.2 GB | about 11.7 GB |

- Strata's 200K image is 3.17 GB (1.2), close to the int8 column.
- Host has 121 GB, with 106 GB available at 2026-10-20 07:50 (`free -g`).
- Settings:
  - `cpu_bytes_to_use`: 32 GiB, the same as Strata;
  - `slots`: 4;
  - `min_free`: 2.5 GiB, checked against MemAvailable, the same as Strata
    (G:7944-7988);
  - container `--shm-size` 0 (USM host pool) or at least 40g (mmap
    fallback);
  - `--memory` at least 32 GiB + about 30 GiB RSS.
- GPU side: 2 active x 200K must fit in the GPU pool (bf16 about 10.9 GB
  on the card holding attention). That is VRAM planning for the serve
  topology and is outside this spec. int8 (W10) is the lever, as in
  Strata (2.82 GiB per slot).
- Admission:
  - a request that hits a parked record loads it (async). The running
    stream keeps decoding, because loads run on the copy queue;
  - GPU pressure uses vLLM's normal preemption/LRU of free cached blocks.
    Those blocks are already in host RAM, so preemption costs no
    recompute;
  - never evict a running request's state, the same rule as Strata
    G:9404.

### 5.5 Graph capture

- All copies are host-scheduled outside `_model_forward`
  (kv_connector_model_runner_mixin.py:67-104).
- Loads complete before the step that consumes them. Upstream defers the
  request until `get_finished` reports the load. FULL_DECODE_ONLY and
  PIECEWISE are unaffected.
- The restored request's first step is a prefill of the new-turn tokens,
  which is not a captured graph. Decode replays then read the same
  block-table and state-index tensors.
- Requirements:
  - no copy may target a buffer a captured graph holds by pointer (the
    pool is separate; GDN state blocks are ordinary cache pages);
  - `B70_XPU_CG_RECLAIM` is unaffected;
  - F-D removes the PLE host syncs that today block prefill capture.

## 6. Test plan (proves no recompute)

All on one card first (Ornith GPTQ or Qwen3.6-35B-A3B; qwen3_5_moe, same
GDN family), then Flash-Next. GPU only via bin/gpu-run, with
bin/xpu-health before and after. Identity via /v1/models (hotschmoe-dev
first).

**T0 CPU unit.**
- TailKey/ParkIndex: lookup, supersede, conversation LRU, pin, budget.
- Byte model for both models and both KV dtypes.
- Save the transcript next to the journal entry.

**T1 resend (G1's test, tightened).**
- PASS requires `restored_computed == 1`: the final prompt token is
  always read, the same as Strata's `n < prompt.size()`.
- Use a non-block-multiple prompt length.

**T2 multi-turn, the real one.** This replaces evict_test as the gate.
- Build the conversation at the token level with `/v1/completions`, so
  the template cannot rewrite history:
  - turn 1 = prompt P1, greedy reply R1 (64-256 tokens);
  - turn 2 = P1 + R1 + U2.
- Control: the same turn 2 on a fresh server, or with no eviction.
- Evict A with B and C (`NUM_GPU_BLOCKS` cap), then send turn 2.
- PASS conditions:
  - a. `prompt_tokens - cached_tokens == len(U2)`, i.e. replay of
    P1+R1 is 0 with the Strata replay formula;
  - b. `vllm:prompt_tokens_cached_by_source{source="host"}` delta equals
    `len(P1)+len(R1)-1` (minus any device part);
  - c. greedy text of turn 2 equals the never-evicted control for 64
    tokens;
  - d. TTFT is below the cold TTFT and recorded alongside the restore
    bytes and ms from the `kv_offload_load_*` metrics.
- Run once with MTP off and once with MTP on. With MTP on, also check
  that the acceptance rate is within noise of the control (draft-group
  tail, W11).

**T3 concurrency.**
- Two streams decoding (C2) while a third conversation is restored.
- Record C2 ITL p50/p99 during the restore window against without it. The
  copy queue must not stall decode by more than 10 percent.
- Then 4 parked + 2 active, cycling 8 turns. Every restore must pass
  T2 a-c.

**T4 scale.**
- Flash-Next at 200K (`--max-model-len 204800`, both cards per serve
  topology), 4 parked slots, 32 GiB.
- PASS:
  - restore of 200K in at most 0.6 s (target 0.42 s, parity with
    Strata);
  - replay 0;
  - greedy equality;
  - peak RSS recorded;
  - MemAvailable floor respected.
- Run the Strata harness for matched numbers:
  strata/flash-next/bench_context_four.py with the
  `prior_tail_replayed_tokens` definition.

**T5 negative.**
- Edit the last user turn (divergence inside R1). Expect a fallback to the
  aligned checkpoint, with replay equal to the distance from the
  checkpoint and the text still equal to the control.
- Corrupt one tail slot (debug flag) and expect the
  `kv_load_failure_policy` path, not wrong text.

**T6 Flash-Next PLE.**
- Before any parking test, run a GPU prefix hit (resident, no offload)
  on Flash-Next. Compare greedy text against no-prefix-cache. This is
  expected to FAIL today (2.7) and to pass after F-D.

Order:
1. T0.
2. W8 copy op microbench on CPU-quiet card: GB/s H2D/D2H for 16 MiB-1 GiB
   runs, against swap_blocks_batch.
3. F-A/F-B + W2/W3 on Ornith: T1 and T2.
4. W6/W7: T3.
5. F-D on Flash-Next: T6.
6. T2 and T4 on Flash-Next.

## 7. Requests to other agents

- G1:
  - adopt T2 as the gate;
  - set `offload_prompt_only=false`;
  - size `--shm-size` above `cpu_bytes`;
  - stop counting a one-block recompute as a pass.
- A1/C7 (fork owner): F-A, F-B, F-C as three isolated commits behind a
  default-off flag. F-D in the qwen4_exp XPU model file.
- A2/B2 (plugins): host the `b70_kvpark.copy_batch` op and the USM-host
  pool allocator (W8).
