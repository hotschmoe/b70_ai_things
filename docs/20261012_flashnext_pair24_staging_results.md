# Two-GPU mapped expert staging control

CONFIG -> Original Unsloth UD-Q4_K_XL, source50 SDK2977a, balanced24/24
layers across two B70s plus RAM. Context2048/prefill64/short-read0/spec2,
noMTP/cacheOFF/traceOFF/groupedF160; two fresh short54/GEN64 per process.
Only STRATA_VERIFY_MIRROR_STAGE_T1 changes0->1. Per-verifier16-blob staging
buffers already exist in both arms. This remains serial layer split with
host bridging; no custom push allreduce or new P2P transport is introduced.

COMMAND -> Actual48803 run_source50_pair24_mirror_t1_model_v1.py via
bin/gpu-run; independent73212 validate_source50_pair24_mirror_t1_model_v1.py.
Result root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source50-pair24-mirror-t1-model-v1-run-v1.

RESULT -> ProducerPASS610s, report59e6f6a17aef6ef4cb49f38c0c705e159bad24a0f37eef308d10acdeba38fbf3.
IndependentreaderPASS8d853b4c79cd5c8d421820ec2cedd4e9e972d7c3b217b013e88a97708d331eda.
Complete output IDs/finish reasons repeat and match arms. Source/all8 binary,
full4 publisher identity, strict/compiled health, kernel and owned normal
retirement/removal gates pass.

| Warm native observation | OFF | ON |
| --- | --- | --- |
| Prompt54 milliseconds | 937.0 | 946.5 |
| Decode64 milliseconds | 4727.8 | 4479.3 |
| Useful output tokens/second | 13.54 | 14.29 |

Decode elapsed fell5.3percent in this matched single comparison. Prefill has
no useful gain; the flag applies to singleton verifier windows. Startup
residency and logical mirrored bytes do not establish selected-expert hit
rate, real PCIe traffic, physical bandwidth or GPU idle.

VERDICT -> Promising small decode change, well below50 outputtok/s and2000
inputtok/s. This capped64-token diagnostic is not natural>=256-output
qualification or interleaved repeated latency/fairness distributions. No
original full-model math/cache/concurrent/API/shelf promotion. DirectXMX51
long524 OFF/ON is running separately; naturalP2048/GEN512 inputs are ready.
