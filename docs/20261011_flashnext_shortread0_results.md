# Batched short-prefill diagnostic

CONFIG -> Exact FlashNext UD-Q4_K_XL, SDK43 mode0/GPU0, ctx2048,
prefill64/PLE65536, MTP0/cacheOFF, LP omitted. Same54 input IDs and two fresh
GEN4 warmup/selected requests. Sole native argv delta: --short-read0.
Current174 registry preserves all173 original rows.

COMMAND -> run_source43_shortread0_v2.py under paired gpu-run lease; reused
closed historical warmV3 OFF component without another baseline launch.
Output: F17/source43-shortread0-v2-run-v1.

RESULT -> Producer PASS, exit0 in508 seconds, repeat IDs760/2919/10458/369
match baseline. Arm health, complete four-shard identity and normal teardown
passed. Report SHA256: f0b3a67de9bf2601d680b95969a4999de4d1e06e7f3822bd8e1fd6c072446fb9.

| Selected warm observation | Default short-read64 | Short-read0 |
|---|---:|---:|
| Native prompt interval |4297.0ms|1516.2ms|
| Native four-output interval |374.3ms|374.6ms|
| Output IDs |760,2919,10458,369|760,2919,10458,369|

Prompt interval decreased by64.7percent (2.83x). This is a historical bounded
diagnostic pair, not interleaved clean-serving speed qualification. Cold
short-read0 prompt16272.4ms and four outputs3045.4ms expose substantial warmup;
report both. Four outputs are too few for sustained decode qualification.
Original F16-prefill versus Q81-verifier arithmetic differs; full head/residual
equivalence is unobserved. Short-read0 batching cannot explain long524 speed
because that prompt already uses Prefill::run. Independent public reader PASS, including observed prompt Verify count0
and fresh closing identity; public SHA256: fb3bd7cd14f94872b37994233881c443b98c4d92d717c0711e196192c8e2dc54.

VERDICT -> Useful prefill candidate; no decode benefit in this sample. User
targets2000input tok/s and50output tok/s remain unmet. Next actual default8
versus32 native expert lane control keeps short-read0 unchanged. Original
model fidelity, all-state cache, active-stream latency/fairness and shelf
qualification remain open.
