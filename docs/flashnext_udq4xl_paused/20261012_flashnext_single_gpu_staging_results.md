# Single-GPU mapped expert staging control

CONFIG -> Exact Unsloth UD-Q4_K_XL, source50 SDK2977a, one B70 plus RAM,
ctx2048/pref64/short-read0/spec2/noMTP/cacheOFF/traceOFF/groupedF160.
Two fresh short54/GEN64 requests per process. Only
STRATA_VERIFY_MIRROR_STAGE_T1 changes0->1. Spec2 reserves capacity; actual
64 decode windows have T1. Existing16-blob staging allocation is unchanged.

COMMAND -> run_source50_mirror_t1_model_v1.py through bin/gpu-run, actual93054;
validate_source50_mirror_t1_model_v1.py, actual21063. Results at
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source50-mirror-t1-model-v1-run-v1.

RESULT -> Producer PASS958s, report
3e25b0c43054bf8ff05364f561d734a300d86168e9d17c002ba30fd66fd8c603.
Independent reader PASS, report
2e3f60942328307eb00965853ac9c644ce59a948ba1867865d1ff4fd838e0429.
Complete generated IDs/finish reasons match across repeat requests and arms.
Fresh all-source/all8 binary, publisher full4, per-card/compiled health,
kernel-fault and owned normal teardown/removal gates pass.

Warm selected observations:

| Metric | OFF | ON |
| --- | --- | --- |
| Prefill54 native milliseconds | 1537.9 | 1529.1 |
| Decode64 native milliseconds | 7071.3 | 6420.7 |
| Output tokens/second from native elapsed | 9.05 | 9.97 |

Decode elapsed fell9.2percent in this single matched comparison. Prefill is
unchanged for practical purposes: staging applies only to actual singleton
verifier windows. Both processes report8092 resident experts of24576 and
24183MiB expert cache. Startup mirrored bytes and logical blob sizes do not
measure physical PCIe traffic, hit rate, bandwidth or idle time.

VERDICT -> Useful narrow decode result, far below50 outputtok/s. This is a
bounded64-token cap diagnostic, not the required natural >=256-output
qualification or repeated/interleaved latency distribution. No original
full-model numerical, cache, concurrent fairness, API or shelf promotion.
Balanced24/24 source50 OFF/ON is next; direct XMX source51 prefill is separate.
