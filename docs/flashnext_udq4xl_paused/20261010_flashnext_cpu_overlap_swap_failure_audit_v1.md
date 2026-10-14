# CPU overlap screen: global swap guard failure, attribution unresolved

CONFIG

Actual api-positive-overlap-cpu-screen-v1-run stopped in case0/repeat0 before any
response. Eight preregistered inputs/two fresh repeats, temp0/seed1234/natural64,
CPU V3 build, mmap/lazy/no-repack, batch64/ubatch64/thread8, no GPU devices,
116GiB cgroup/no swap and host-global no-swap-increase guard. No guard is relaxed.
No agent model payload read or runtime execution; audit uses saved metadata/logs.

COMMAND

Read report/memory samples/server log and compare saved successful same235 CPU
reference samples. Read actual CPU mul_mat_id and graph reservation source only.
Report SHA57cc382f25e7d365367f2bfa8b617614c1593c78e879f27a6010b38666060989.
Server log SHAb932a3d54019e0015a9510f953c2f820ae1fd9fe681ffad4efc57a2229eab402.
Prior same235 report SHA519f73be9d17572826f5587b19db2fcae93f58471aa0c7dada123cbce110c0be.

RESULT

Twenty-two owned samples have VmSwap=0, cgroup memory.swap.current=0,
memory.swap.max=0 and no max/oom/oom_kill events. The first failing sample at
1791638029.1637707 has global SwapFree6670303232 versus baseline6680895488:
10,592,256 bytes new global swap use. MemAvailable124967022592 bytes (~116.4GiB),
cgroup current2871312384/peak2871611392, process RSS70925021184/VmSwap0.
No host reserve or owned cgroup cap was exhausted in the saved observations.
The large RSS is consistent with mmap file-backed weights already charged outside
the container; exact anonymous/file breakdown was not sampled, so this remains
an inference. Peak process RSS alone is not owned anonymous or whole-host pressure.

Log shows prompt progress46/50 before stop, no completion. Owned stop led to
RemoteDisconnected and eventual exit137, not OOMKilled; normal-lifecycle proof
correctly remains FAIL. Postterminal full4 passed. All16/selection are unobserved.

The successful same235 reference had maxRSS71559028736/71596216320 and peak
cgroup8141488128/791666688 across its two cases, minMemAvailable124888829952/
124848459776. Both are comparable or larger than this failed case; global swap
free increased450560 bytes relative to its own baseline. This does not support
an established owned-memory-peak threshold causing the new failure.

Actual source: ggml/src/ggml-cpu/ggml-cpu.c1554-1685 (SHA45b22f8c83b2cbccabfc2f81a38ef8a5810223e1ce366b7c6f77d42faa5cc6f3)
allocates converted operand/workspace and expert row-count/mapping/per-thread
scratch, then dispatches only experts with nonzero selected rows. It references
mapped expert matrices; this path does not copy the whole expert weight union
into a newly allocated host array. Source llama-context.cpp638-751
(SHA578e4b2cdcf3c2230bc6081bbe33f18a935c277903675c2f81efc6145c2dc61c)
reserves prompt graph buffers using n_ubatch. A smaller ubatch can change workspace
and mapped expert access pattern, but actual selected expert union/anonymous
allocation pressure was not captured. It also changes arithmetic scheduling and
must be a deliberate new recipe, not a silent numerical/reference adjustment.

VERDICT

Global swap-increase failure is real and retained. Owned workload swap or causal
memory-pressure/expert-union attribution is not established; no lower-peak recipe
is justified by these samples yet. Next narrowly declared diagnostic should keep
all16/corpus/seed/cap/guards unchanged and sample host vmstat pswpin/pswpout plus
owned smaps_rollup and cgroup memory.stat (anon/file/kernel) and bounded host
per-PID VmSwap deltas. Require a stable pre-model global-swap observation window,
but do not claim it authorizes inference by elapsed time or waive a later increase.
Only evidence of owned anonymous/workspace pressure would justify preregistering
an otherwise matched smaller-ubatch experiment. Source/build/model/failed records
remain unchanged; no quality, fullmath, stability or latency qualification.
