# G1 -- KV eviction to host RAM without recompute

One-card proof on Ornith GPTQ int4 (Qwen3.6-35B-A3B hybrid). Full-attention
KV and the GDN boundary state both come back from pinned host memory.
The prefix is not recomputed. No fork edit. Copy path is upstream
OffloadingConnector plus XPU swap_blocks_batch.

## Verdict

Pass. Conversation A (30464 tokens) was evicted by B and C, then restored.
The restore cached 29376 tokens from host and recomputed 1088, which is
the same align-mode tail a GPU-resident replay recomputes. Device delta
on that request was 0. Greedy text matched the never-evicted control.
Host load was 666009600 bytes in 0.461 s. That byte count is the
full-attention KV for 29376 tokens plus one unpadded GDN snapshot of all
30 linear layers.

## Design

HybridCPUOffloadConnector (vllm/flashnext/kvoffload/connector.py)
subclasses OffloadingConnector. The parent already offloads every
prefix-cacheable group, including MambaSpec in align mode, into a pinned
host pool and loads it on a prefix hit. On XPU the copy is
ops.swap_blocks_batch (vllm/v1/kv_offload/cpu/gpu_worker.py). The subclass
refuses to start unless the offload set contains both full_attention and
gdn. supports_divergent_local_hybrid_hits stays false: turning it on
without a scheduler backfill drops the hit (vLLM issue 52773). Full
eviction does not need that flag. The scheduler asks from the common
boundary, which is zero once A's GPU blocks are gone, and the parent
loads both groups.

Own code besides the subclass:

- layout.py: Ornith page sizes and the XPU block-scale formula
  (alignment * cdiv(mamba_page, alignment * attn_bytes)).
- park.py: the contract. A slot is a hit only when both pieces exist.
  4-slot LRU under a byte budget. The live server does not call Park;
  the connector is the copy engine. Park is the rule the copies must meet.
- serve_ornith.sh: one-card serve. Does not edit harness/.
- evict_test.py: the A/B/C proof.

CPU checks (no GPU, no vLLM import): 16 unittest cases OK.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=vllm/flashnext python3 -m unittest
kvoffload.tests.test_layout_park kvoffload.tests.test_metrics
Scaled block 1088, GDN page 2146304 bytes/layer, attention 20480
bytes/token. Checkpoint of an exact multiple of the block drops one
block, matching get_mamba_prefill_checkpoint_position.

## GPU

CONFIG: card 0, host UMD (no --umd), TP=1, port 18111, container
vllm-fn-g1-kv-c0. Model
models/files/ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2.
Served names: hotschmoe-dev, then
ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.
--dtype float16 --language-model-only --max-model-len 32768
--gpu-memory-utilization 0.90 --num-gpu-blocks-override 45
--max-num-seqs 2 --enable-prefix-caching --mamba-cache-mode align
--enable-prompt-tokens-details --no-async-scheduling.
KV transfer: HybridCPUOffloadConnector, kv_both, module
kvoffload.connector, cpu_bytes_to_use 17179869184.
Graphs: compilation mode 0, cudagraph_mode PIECEWISE, capture sizes
1,2,3,4,8, VLLM_USE_BREAKABLE_CUDAGRAPH=1, B70_XPU_CG_RECLAIM=500.
Engine v0.32.1rc1.dev94+g5fd62688f. Quant auto_gptq.

The override of 45 blocks (of 1088) is what makes B and C evict A.
Without it the planner wanted 245 blocks (266560 tokens), which holds
three 30K prompts. With it the server reported 39852 KV tokens and
max concurrency 1.22x at 32768, so two 30464 prompts cannot both stay
resident. Natural pool on this card is one ~200K stream, not two.

COMMAND: CARD=0 PORT=18111 B70_AGENT=g1-kv
vllm/flashnext/kvoffload/serve_ornith.sh
Log: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/g1-serve.log
Then, inside the container:
python evict_test.py --base-url http://127.0.0.1:18111
--serve-log .../g1-serve.log
--tokenizer <the GPTQ dir>
--out .../logs/g1-evict.json
Sequence: A cold, 32-token flush, A while still resident, B, flush, C,
flush, A again. Temperature 0, seed 1, 16 greedy tokens. Stores are
deferred to the next engine step, so each long prompt is followed by a
flush. The client runs in the container because the venv python is a
symlink to /usr/bin/python3.12 and does not run on the host.

A first serve with PIECEWISE and inductor left on died in 114s:
InductorError, ocloc not in the container PATH.
Log: .../logs/g1-serve-inductor-fail.log
The relaunch uses D1's breakable path (mode 0, no inductor).

RESULT: Startup logged attention block size 1088, mamba page padded
3.82% to match, model 20.18 GiB, PIECEWISE capture 5 graphs in 2s then
5 in 1s, KV override 245 -> 45, CPUOffloadingSpec, mmap
/dev/shm/vllm_offload_*.mmap 17.18 GB, 771 CPU chunks.
gpu-run hold 330s, exit 0. Shutdown was a drain SIGTERM.
XPUWorker shutdown done. Post-health:
bin/gpu-run --card 0 bin/xpu-health --card 0
card 0 OK, HEALTHY, 16s. Card 1 was not probed (E1 held it).

evict_test passed (g1-evict.json):

```
block 1088, prompt 30464, checkpoint 29376
A-cold      ttft 5.177s  cached 0      prompt 30464
A-resident  ttft 0.267s  cached 29376  prompt 30464
B           ttft 5.114s  cached 0
C           ttft 5.116s  cached 0
A-restored  ttft 0.732s  cached 29376  computed 1088
host delta 29376, device delta 0, external hits 29376, gpu hits 0
```

Cold prefill 30464/5.177s = 5884 tok/s.
Greedy text, all three A calls, identical:
"Note A. The quick brown fox keeps talking about topic A. Note A"
B and C produced the same sentence with their own marker, cached 0,
so they did not reuse A's prefix.

Prometheus after the run
(.../logs/g1-metrics.txt, and the 07:35:37 serve-log line):

```
kv_offload_load_bytes_total   666009600
kv_offload_load_time_total    0.460751927
kv_offload_load_size_count    1
kv_offload_store_bytes_total  2064875520
kv_offload_store_time_total   0.474663697
kv_offload_store_size_count   93
external_prefix_cache_hits    29376
prefix_cache_hits (device)    29376
source=host 29376, source=device 29376
```

The single load is the A restore. 29376 * 20480 = 601620480 bytes of
full-attention KV (10 layers, 2 kv heads, head_dim 256, fp16).
30 * 2146304 = 64389120 bytes, one unpadded GDN snapshot (conv state
for kernel 4 plus the fp32 recurrent state). Sum 666009600. The host
copy that came back is both pieces, once, not a padded GDN page per
block. The 93 stores cover A, B, and C (about 0.69 GB each) plus the
short flushes, in 0.475 s.

The 0.465 s gap between restored TTFT (0.732 s) and the GPU-resident
replay (0.267 s) matches the 0.461 s load. Both replays recompute the
same 1088-token tail. Eviction added no prefill tokens.

The kvoffload.connector logger did not reach the vLLM handler on this
run, so the "G1 hybrid offload groups" line is absent. The server
still constructed HybridCPUOffloadConnector (it is the named class;
a missing gdn group raises before listen) and created CPUOffloadingSpec.
The logger name is now vllm.kvoffload.hybrid so a later serve prints it.
That change was not part of the measured process.

VERDICT: Evict-and-restore works on one B70 for this hybrid model.
Prefix replay of the parked tokens is 0. The align-mode tail of one
block is recomputed either way. Greedy identity holds. No C7 fork
commit is required for the copy path.

## Requests

C7, not blocking: issue 52773 / a scheduler hook such as
get_hybrid_backfill_tokens, for the case where full-attention blocks
are still on the GPU and only the GDN state was evicted. Do not flip
supports_divergent_local_hybrid_hits until that hook exists. This
test did not hit that case (device delta on the restore was 0).

No other fork edit. Next measurement, not this gate: 2 streams x 200K.
One card's uncapped pool is 245 blocks, about one 200K stream. Two
streams need a second card or a larger KV budget, with the same
connector. A 200K prefix is about 4.10 GB of KV plus 64 MB of GDN.
Two of those fit in the 17.18 GB host mmap; four are tight against it.
Strata's night-final restore was 411-422 ms for a ~3.16 GB payload.
This restore is 461 ms for 0.666 GB (~1.45 GB/s). Correctness first.
The copy bandwidth is the make-it-fast item.
