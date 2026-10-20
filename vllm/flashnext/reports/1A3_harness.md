# A3 runtime, harness, downloads

Agent A3. Card 1 is the smoke card. Card 0 stays with A2/B2.
ASCII only. Times are local.

## GPU and CPU window

- Eager Ornith smoke finished and the card was released. Post-health
  was HEALTHY. B1 holds card 1 since 2026-10-20 00:54:50 for AutoRound
  (`run_quant.py`, layers 0-39, 200 iters). At 01:56 the log
  `/mnt/cache/b70/devmodel/quant/_work/quant3.log` was on layer 14,
  about 252 s/layer, so the rest of the quant is on the order of
  2 hours plus the eval that follows. The MTP serve (`a3-serve-mtp`,
  port 18081, pid 2825445) is queued behind that lease and will not
  preempt B1. Graph is after MTP.
- Card 0 is free.
- The CPU lock is free. The download finished at 00:52:53.

## 1. xpu-run

CONFIG: image `vllm-flashnext-xpu:py312` (ubuntu 24.04 + python3.12,
no GPU UMD in the image). Venv `/mnt/vm_8tb/b70/build/vllm-flashnext/venv`
mounted at `/opt/venv` and at the host path (console shebangs are
`#!/opt/venv/bin/python`; the venv python symlink points at
`/usr/bin/python3.12`, which the host does not have). Host loader
`libze_loader.so.1.28.2` (1773224 bytes), `libze_intel_gpu.so.1.15.38646`,
`libigdgmm.so.12.10.0`, and `intel-opencl/libigdrcl.so` bind-mounted at
`/host-umd` and (except the OpenCL ICD) at
`/usr/lib/x86_64-linux-gnu`. Host IGC 2.36.3 from `/usr/local/lib`
(`libigc.so.2` 95845480 bytes, `libigdfcl.so.2`, `libiga64.so.2`,
`libopencl-clang2.so.17`) bind-mounted at `/host-umd` and
`/usr/local/lib`. `LD_LIBRARY_PATH` is `/host-umd:/usr/local/lib` then
the venv. `ZE_AFFINITY_MASK` from `--card`. `ONEAPI_DEVICE_SELECTOR`
is left unset. Lease via `bin/gpu-run`.

COMMAND: `vllm/flashnext/harness/xpu-run.sh --check`

RESULT: exit 0. `ldd` of `/host-umd/libze_loader.so.1.28.2` resolves
only libstdc++/libgcc/libc/libm. `ldd` of `libze_intel_gpu.so.1`
resolves `libigdgmm.so.12 => /host-umd/libigdgmm.so.12`. dlopen maps:
`libze_loader` `/host-umd/libze_loader.so.1`, `libze_intel_gpu`
`/host-umd/libze_intel_gpu.so.1`, `libigdgmm` `/host-umd/libigdgmm.so.12`,
`libigc.so` `/host-umd/libigc.so.2` (95845480 bytes), `libigdfcl` and
`libopencl-clang` also under `/host-umd`. No pip `libze_loader` copy.

VERDICT: other agents can use this script. Usage is the header of
`vllm/flashnext/harness/xpu-run.sh`.

## 2. First allocation (the IGC gap)

CONFIG: same container, `--card 1`, `ZE_AFFINITY_MASK=1`. Before the
IGC mounts, the Level Zero driver loaded from `/host-umd` and
`torch.xpu.device_count()` was 1, then the first device allocation
aborted.

COMMAND: `xpu-run.sh --card 1 -- python -c` a 4x4 `torch.ones` matmul.
Log `/mnt/vm_8tb/b70/build/vllm-flashnext/logs/a3-torch3.log`.

RESULT: before the IGC mounts, exit 134,
`Abort was called at 15 line in file: ../../neo/shared/source/gmm_helper/resource_info.cpp`.
`libze_intel_gpu` does not NEEDED `libigc.so.2`; NEO dlopens it, and
the string `/usr/local/lib` is inside the host UMD. The container did
not have the host IGC stack, so that dlopen could not see it. After
the mounts: `cuda False`, `count 1`, name `Intel(R) Arc(TM) Pro B70
Graphics`, matmul sum `64.0`, `ALLOC_OK`, exit 0, 20s. Card released.

Health around that work: `B70_AGENT=a3-health bin/gpu-run bin/xpu-health`
-> `HEALTHY (cards 0 1)` in 34s, exit 0, after the earlier aborts and
before the successful alloc. Image `vllm-xpu-env:int8g-v0251` (its own
UMD 26.18). That is a different stack from this venv.

VERDICT: the abort was the missing host IGC (2.36.3) next to UMD
26.22, not a wedged card. xpu-run now refuses to start if `libigc.so.2`
does not resolve to the host mount.

## 3. Downloads (verified)

CONFIG: `bin/cpu-run` + `vllm/flashnext/harness/download_devmodel.sh`.
HF cache on `/mnt/cache/b70/devmodel/.hf-home`. `hf` 1.22.0, no token
(unauthenticated rate-limit warning, not a failure).

Pins (Hugging Face API, recorded before the bytes were checked locally):

- `Qwen/Qwen3.6-35B-A3B` revision `995ad96eacd98c81ed38be0c5b274b04031597b0`
  -> `/mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B`
- `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`
  -> `/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF`
- Remote GGUF sizes, all under 25 GB, so all three are in the job:
  `UD-IQ3_XXS` 14069266720, `UD-IQ4_XS` 18209036576, `UD-Q4_K_M` 22663387424.
- IQ3_XXS choice: Unsloth Dynamic (`UD-`) from the MTP GGUF repo.
  Bartowski was not selected.

COMMAND: `B70_AGENT=a3-download B70_CPU_LOCK_TIMEOUT=0 bin/cpu-run bash vllm/flashnext/harness/download_devmodel.sh`

RESULT: status DONE 2026-10-20T00:52:53, cpu-run exit 0, 2625s.
Manifest `/mnt/cache/b70/devmodel/MANIFEST.txt`.
Official: 26 safetensors, 71903776776 bytes. Index weight map has
mtp/nextn keys (`OFFICIAL_INDEX mtp_or_nextn=True`).
GGUF local bytes match the remote sizes exactly:
`UD-IQ3_XXS` 14069266720, `UD-IQ4_XS` 18209036576,
`UD-Q4_K_M` 22663387424.

VERDICT: official BF16 with MTP, and the three comparison GGUFs, are
verified. CPU lock released.

## 3b. Triton JIT

CONFIG: same container after adding gcc 13.3 / g++ / python3.12-dev to
the image, libze-dev 1.28.2 headers at /opt/level-zero, the unversioned
libze_loader.so name, and oneAPI 2026.1 icpx on PATH
(steve-repro qwen38 oneapi-root compiler/2026.1/bin). icpx is not on
LD_LIBRARY_PATH.

COMMAND: card 1, `python .../a3_triton_smoke.py` (128-element add-one).
Log `a3-triton-smoke.log`.

RESULT: sum 256.0, `TRITON_OK`, exit 0, 33s. Lease released.

VERDICT: the earlier engine failure (`Failed to find C compiler` in
`triton/runtime/build.py` during the dummy run, after
`XPUExpertsWNA16` was selected) is fixed for a real Triton kernel.
Log of that failure: `a3-serve-eager-v3.log`.

## 4. Ornith smoke (in progress)

CONFIG: `serve_ornith_gptq.sh` card 1, port 18081, `GRAPH=0` (`--enforce-eager`),
`MTP=0`, `--dtype float16`, `--language-model-only`, `--gpu-memory-utilization 0.90`,
`--max-num-seqs 4`, `--max-model-len 33792` (so a 32768-token prompt plus
one generated token fits). Served names: `hotschmoe-dev` then
`ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2`. Model
`models/files/ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2` (23G on disk).
Log `/mnt/vm_8tb/b70/build/vllm-flashnext/logs/a3-serve-eager-v4.log`.

An earlier launch with max-model-len 32768 was stopped during import so
the 32K TTFT point would fit. It never served a request. The first
"serve" in `a3-serve-eager.log` was a harness bug (the user command was
not passed into `docker run`); that is fixed and the torch alloc above
is the proof the command now runs.

COMMAND: serve_ornith_gptq.sh as above. Then
`bench_latency.py --tag eager-c1` against http://127.0.0.1:18081.
JSON: `/mnt/vm_8tb/b70/build/vllm-flashnext/logs/a3-bench-eager.json`.

RESULT: `/v1/models` lists `hotschmoe-dev` first, then
`ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2`, max_model_len 33792.
Startup log: XPU WNA16 MoE, XPUExpertsWNA16, KV cache 218758 tokens
(6.47x concurrency at 33792), inductor disabled by enforce-eager.
Coherence: two temp-0 streams identical, both mention Paris, verdict OK.
Text was the repeated sentence "Paris. The capital of France is Paris."
Decode C1 ITL p50/p90/p99 74.78/75.36/76.40 ms, 13.57 tok/s, n=64.
Decode C2 streams 13.52 and 13.74 tok/s, ITL p50 73.89 and 73.88 ms
(concurrent HTTP, ITL did not grow). Prefill unique-nonce prompts:
1017 tok TTFT 169 ms (6001 tok/s), 8185 tok TTFT 1038 ms (7888 tok/s),
32762 tok TTFT 5121 ms (6398 tok/s). MTP metrics absent.
Strata DD reference (different model, 2 cards, MTP): C1 141 tok/s
(ITL 7.09 ms), C2 92/stream (ITL 10.87 ms), prefill 4K 2200 and 32K
2150 tok/s. This eager run is about 10x slower on decode. Prefill
tok/s is higher than that DD figure on this smaller GPTQ model.
C1 TTFT in the decode section is after a same-prompt warmup, so it
is a cached-prefix TTFT, not the prefill sweep.

VERDICT: eager smoke serves and is coherent. Decode is far from the
DD latency target. Teardown, MTP, and graph are next.

## 5. Latency harness

`vllm/flashnext/harness/bench_latency.py` ran once, tag `eager-c1`.
It checks `/v1/models` (first id `hotschmoe-dev`, alias present), two
temperature-0 streams of "The capital of France is", C1 and C2 decode
ITL (p50/p90/p99 ms/token) from SSE gaps, TTFT and prefill tok/s at
1K/8K/32K, and `vllm:spec_decode_*_total` when present. JSON plus an
ASCII summary. Strata DD reference embedded from
`strata/flash-next/dd/README.md`: C1 141 tok/s, C2 92/stream, prefill
4K 2200 and 32K 2150. Different model, two cards, MTP on. Comparison
only. Eager numbers are in section 4.

## Consults

Tried at the Triton-vs-ITL decision. No model answered.

- `claude --bare -p --model fable-5.1` exited with
  `unrecognized_model` for `fable-5.1`.
- `claude --bare -p --model claude-5.5` exited with
  `unrecognized_model` for `claude-5.5`.
- `claude --bare -p` with no model returned
  `Not logged in · Please run /login`. No model id was served.

The latency harness still reports SSE-gap ITL and token-count tok/s
separately, because an MTP chunk can carry more than one token.

## Requests

- A2/B2: the CPU lock is free.
- B1: card 1 is B1's quant again. A3 MTP serve is queued behind it.
- A1: no fork commit yet. The IGC mount is a harness bind, not a vLLM
  source change. `harness/patches/sitecustomize.py` is still a no-op.
