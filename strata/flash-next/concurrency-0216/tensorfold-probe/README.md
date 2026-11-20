# TensorFold probe: CPU-only preparation (2026-11-20)

Prepared by a CPU-only agent (Claude Sonnet 5.5, model ID claude-sonnet-5-5). No GPU was touched: no /dev/dri,
no gpu-run, no Level Zero call. Nothing in this file was benchmarked; every speed number lives in TensorFold's docs
(their hardware) and is not ours. Format: CONFIG -> COMMAND -> RESULT -> VERDICT where it matters.

Files next to this README (all CPU-only scripts, plus the one GPU-side helper that has NOT been run):
- build_tensorfold.sh        zig builds of main and pr3603 (run under cpu-run)
- merge_lora.py              streaming per-tensor LoRA merge (bf16 base -> bf16 merged)
- verify_merge.py            merge verification; output in verify_merge_output.txt
- make_4bit_variants.sh      rotate_w4.py runs: rot / norot RTN int4 g128, base and pi-adapted
- check_variants.py          sanity of those variants (structure, RTN error, pass-through tensors)
- make_prompt_ids.py         real-text token-id prompt files for bench_cache.py
- ze_p2p_precheck.py         Level Zero zeDeviceCanAccessPeer both directions (NOT RUN; needs both cards)

## 1. Toolchain identities

| Item | Value |
|---|---|
| Zig | 0.17.0 official x86_64-linux tarball, /mnt/vm_8tb/b70/build/tensorfold/toolchain/zig-0.17.0/zig (host zig 0.16.0 at ~/.local/bin untouched) |
| Zig tarball sha256 | 1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026 (equals ziglang.org/download/index.json shasum for 0.17.0) |
| Zig minisign | VERIFIED (ed25519 over blake2b-512, key RWSGOq2NVecA2UPNdBUZykf1CCb147pkmdtYxgb3Ti+JO/wCYvhbAb/U, trusted comment timestamp:1790965431 file:zig-x86_64-linux-0.17.0.tar.xz); verifier script /mnt/vm_8tb/b70/build/tensorfold/dl/verify_minisign.py (python cryptography; no minisign binary on host) |
| nGEN | oneDNN tag v3.14.2 (commit 4db284662a37cf46401655b5ee99391d9de52c00, shallow clone), third_party/ngen copied to /mnt/vm_8tb/b70/build/tensorfold/ngen-src/ (ngen.hpp there; COPYRIGHT kept, ONEDNN_LICENSE added) |
| ocloc | host /usr/bin/ocloc 26.22.38646.4 (docker image not needed) |
| C++17 compiler | host has NO g++/clang. TensorFold's build runs the system command `c++ -std=c++17 ...` for zig/tools/xpu/w4_ngen.cpp, so toolchain/bin/c++ is a 2-line wrapper that execs `zig c++` (clang 22.1.8 bundled with zig 0.17.0, libc++). Build scripts put toolchain/bin first on PATH. Worked first try. Docker (strata-sycl-dev:020) was therefore not needed. |
| python | 3.14.4 venv at /mnt/vm_8tb/b70/build/tensorfold/venv: torch 2.15.0+cpu, safetensors 0.8.0, huggingface_hub 2.4.0, tokenizers 0.23.3, numpy, cryptography. HF_HOME=/mnt/vm_8tb/b70/hf_cache |
| zig caches | ZIG_GLOBAL_CACHE_DIR=/mnt/vm_8tb/b70/build/tensorfold/zig-global-cache, ZIG_LOCAL_CACHE_DIR=.../zig-local-cache (kept off the boot drive) |

Env to rebuild: `export PATH=/mnt/vm_8tb/b70/build/tensorfold/toolchain/bin:/mnt/vm_8tb/b70/build/tensorfold/toolchain/zig-0.17.0:$PATH ZIG_GLOBAL_CACHE_DIR=/mnt/vm_8tb/b70/build/tensorfold/zig-global-cache ZIG_LOCAL_CACHE_DIR=/mnt/vm_8tb/b70/build/tensorfold/zig-local-cache`

## 2. Worktrees (git worktree of /mnt/cache/b70/refs/TensorFold; its own working tree is unmodified)

| Tree | Commit |
|---|---|
| /mnt/vm_8tb/b70/build/tensorfold/main | 9e2a9bd7a93d3c6cd333d6db10a1bd05d7478145 (main, "changelog: #3863 under Unreleased", TensorFold 1.3.0) |
| /mnt/vm_8tb/b70/build/tensorfold/pr3603 | 207aaae1694e178aba453517da828d9683636acf (refs/pr/3603, "xpu(flashnext): Xe2 DPAS tile kernels, adaptive solo chunks, and confidence MTP"; stacked on #3585 -> #3586 -> #3587; its base is older than main, so main and pr3603 differ in ~950 files) |

## 3. Builds (zig 0.17.0, -Doptimize=ReleaseFast -Dngen-include=ngen-src -j4, under bin/cpu-run + systemd scope MemoryMax=90G)

All steps exit 0, zero compile errors, zero warnings in the captured tails. Wall time on this host: main xpu-native 5m55s, xpu-qwen27 30s, xpu-qwen27-tests 44s; pr3603 xpu-native 4m24s, xpu-qwen27 30s, xpu-qwen27-tests 47s, xpu-tp-bench 17s, xpu-flashnext 28s. Log: /mnt/vm_8tb/b70/build/tensorfold/build.log (tails only).

Note: the xpu-qwen27-tests step also RUNS a host unit-test binary (22 tests, "All 22 tests passed"). It prints a
diagnostic ULP table and a line "failed command: .../test --listen=-" that looks like a failure in the captured
tail but is not: re-running the same binary directly exits 0 (log main_unit_test_run.log). strace of that binary
shows zero opens of /dev/dri or render nodes; "Level Zero : 0x1" in its output is a test-double fault injection
message from ze.zig, not a device call.

| Tree | Binary (zig-out/...) | sha256 |
|---|---|---|
| main | xpu/bin/tensorfold-native (49.8 MB) | e6f10e1dcd2fe8fae86fa19ac80a0cead9a0c65ecb5d2658009c489e22bc1d2a |
| main | bin/xpu-qwen27 | a264d34ae17e341dda1c41acdc9484c23510ecd0b860af4246e770f7e9bd399e |
| main | bin/tf-xpu-qwen27-means-test | b7a5a9ee8ab1df7ad0eaf98805e2d73de9814e712270b66d26d3e479ddf83517 |
| main | bin/tf-xpu-qwen27-seam-test | 7cbec3a2ec9d3761f17b78b2a426acd1beffd4dda79c657c4131a146e2c272a2 |
| main | bin/tf-xpu-qwen27-close-test | e3481692a49cef24c5cc7d4e06ec27d38a677a6db7eb67cd955c70bb7d318b51 |
| main | bin/tf-xpu-qwen27-tail-test | 2915d19c80e63312fa5f14aa6bd1e03f8ea4d438b846a9b1e2b6096af5200225 |
| main | bin/tf-xpu-tp-bench (built afterwards, main has this step too) | 1743a1929122fa7a7de070fdfda2bbc2844a1c83445c748dcf43fe56d068e032 |
| pr3603 | xpu/bin/tensorfold-native (54.1 MB) | bb433044e47e2de2b545a785b8e01a32be4715a55820ad275a47d4632a2cb196 |
| pr3603 | bin/xpu-qwen27 | 8c1b4da0db80198c1ffff456facb599447a1a17bb151831108351340acb537ee |
| pr3603 | bin/xpu-flashnext | 7f47206761b82e1f824b3b62ceee016b4c597c0f0aa77dfbd319eaa4adda6aec |
| pr3603 | bin/tf-xpu-tp-bench | 475d95e6ae4c68722caa2a01530c924ee249e0e526c91636d6a6f4ff009ce7e0 |
| pr3603 | bin/tf-xpu-qwen27-means-test | 58970e25b1a9ef915a33f05cea9d687cbf47f4bdde2d53619696d9eca92a1145 |
| pr3603 | bin/tf-xpu-qwen27-seam-test | 53cd2fb50fe64da21f4c4c278bf6386b1579211dc84b19179c3266669d07c7bb |
| pr3603 | bin/tf-xpu-qwen27-close-test | ab04c418d5274545d866ffc4688ff18a970b9090de8879db23b56e56ec0511c3 |
| pr3603 | bin/tf-xpu-qwen27-tail-test | 6ac245b4f358a630d2e697caed0214b768a64d612be8b80be92681a34b078e49 |

Not built (not requested): libtfgrammar.so (structured output, needs xgrammar v0.2.8; the server refuses structured requests without it, everything else works), xpu-gemma4, xpu-kernel-contract-*.
Build errors and fixes: none besides the missing host C++ compiler, solved by the zig c++ wrapper above. The reference clone had to gain two worktree entries under its .git/worktrees (git worktree add); its working tree is untouched.

## 4. Models (all under /mnt/cache/b70/tensorfold-models, NVMe)

| Dir | Content | Revision / source | Size | Verification |
|---|---|---|---|---|
| Qwen3.8-27B-GPTQ-g128-sergiiob-9d189a60 | target GPTQ int4 g128 sym, MTP head bf16 | SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 @ 9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e | 19,579,716,068 B (18.2 GiB), 5 shards | Original at /mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-27b/gptq-int4-mtp-bf16-9d189a60: every file size equals the HF file list at that revision (only .gitattributes absent); all 6 LFS-or-large files (5 shards + tokenizer.json) sha256 equal HF's lfs sha256 (dl/gptq_sha_check.txt). Copied to NVMe (cp -r, diff -rq identical). Original left in place. |
| Qwen3.8-27B-DFlash2 | drafter bf16 incl. candidate_selector | z-lab/Qwen3.8-27B-DFlash2 @ 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 | 3,848,825,995 B (3.6 GiB) | model.safetensors sha256 equals HF lfs sha256; config.json + README.md downloaded (figure png skipped) |
| pi-adapter | harness-hub/Qwen3.8-27B-pi-adapted LoRA | @ 80d05c9051dfd2ae464e1daf4329cb15524caf60 | 657,634,760 B | adapter_model.safetensors and tokenizer.json sha256 equal HF; adapter_config, chat_template.jinja, tokenizer_config.json, README.md also fetched (banner/gif/png skipped) |
| qwen3.8-27b-pi-adapted-bf16 | merged bf16 (see section 5) | base + pi adapter | 55,593,344,361 B (51.8 GiB), 18 shards | verify_merge_output.txt |
| qwen3.8-27b-pi-adapted-rot-rtn4 | hadamard128-rotated RTN int4 g128 (marker file hadamard128) | rotate_w4.py rot, bf16 = merged pi | 19,579,713,655 B | check_variants.py |
| qwen3.8-27b-pi-adapted-norot-rtn4 | plain RTN int4 g128 | rotate_w4.py norot | 19,579,713,598 B | check_variants.py |
| qwen3.8-27b-base-rot-rtn4 | same from the plain base bf16 | rotate_w4.py rot | 19,572,549,169 B | check_variants.py |
| qwen3.8-27b-base-norot-rtn4 | same from the plain base bf16 | rotate_w4.py norot | 19,572,549,112 B | check_variants.py |

NVMe free after everything: ~274 GB of 954 GB (418 GB at start). Load-time note: the GPTQ original sits on SATA (/mnt/vm_8tb); use the NVMe copy for serving (tensorfold mmaps the shards, first load is read-bound). Base bf16 stays on SATA at models/files/qwen3.8-27b/bf16.

### Base model identity
/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-27b/bf16 is Qwen/Qwen3.8-27B (config Qwen3_5ForConditionalGeneration, 64 layers, 18 shards, 1199 tensors in the index). All 18 shard sha256 equal the HF lfs sha256 at HF main = 37561816e66b5b9153b1b64c10968c18c1e6f4a0 (checked 2026-11-20). Three small files differ from HF main by size (README.md, chat_template.jinja 8952 vs 9004 bytes, tokenizer_config.json 17928 vs 9984): the local snapshot has older text files than HF main; the weights are identical. No revision pin file exists locally, so the weights revision is "HF main as of 2026-11-20, weights unchanged vs the local snapshot".

### Tokenizers differ, record this before comparing
- Official base tokenizer.json and the pi adapter's tokenizer.json both carry \p{M} in the pre-tokenizer regex ([\p{L}\p{M}]+); SergiioB's does not (re-saved with transformers 5.14.1; TensorFold's recipe section 3 documents this). Three different sha256 for the three tokenizer.json files.
- pi chat_template.jinja (9004 B, sha256 prefix af16f654) differs from the local base's (8952 B, c3cf9e34), which is also SergiioB's.
- Choices made: pi variants (merged bf16 and both pi rtn4) use the adapter's tokenizer.json, tokenizer_config.json, chat_template.jinja. Base rtn4 variants use the official base snapshot's three files. Template copied from rotate_w4.py (SergiioB's files) was overwritten in all four variants. The original GPTQ dir keeps SergiioB's files as published.

## 5. LoRA merge (pi adapter -> bf16 base)

CONFIG: adapter_config.json: peft 0.21.2, peft_type LORA, r=32, lora_alpha=64, use_rslora=false, use_dora=false, use_qalora=false, lora_bias=false, bias none, no rank/alpha patterns, no modules_to_save, target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj, base_model_name_or_path Qwen/Qwen3.8-27B. Nothing unusual, so no stop. scale = alpha/r = 2.0.
Adapter: 512 tensors fp32 = 256 modules: 64 layers x (gate, up, down) + 16 full-attention layers x (q, k, v, o). The 48 GDN layers get MLP deltas only (their linear_attn projections are not targets); the MTP head and the vision tower are untouched.
COMMAND: bin/cpu-run systemd-run --user --scope -p MemoryMax=60G -- venv/bin/python -I merge_lora.py BASE ADAPTER OUT (8m, 435 s of merge work after the lease; RAM stayed under the 60 GB scope cap, one shard at a time; peak not sampled).
Method: per shard, per tensor: W' = bf16(fp32(W) + 2.0 * fp32(B) @ fp32(A)); all other tensors copied unchanged, same shard layout and index as the base; config, tokenizer, chat template (adapter's), preprocessor, merges, vocab, LICENSE copied. Per-module stats in merge_stats.json inside the output dir.
RESULT (verify_merge_output.txt, VERIFY OK):
- 256 of 256 adapter modules matched a base tensor (script stops with exit 3 on any miss; none).
- 1199 tensors, 0 shape mismatches, 0 dtype mismatches vs base; index weight_map identical.
- 943 untouched tensors, 40 random samples bit-identical to base; 333 vision-tower tensors present and untouched.
- Delta recheck on 12 random modules recomputed independently: ||merged - base|| / ||B@A*2|| median 0.99 over all 256 (range of the 12 sampled: 0.86 to 1.03), cosine between realized and ideal delta 0.67 to 0.88.
- FINDING: the LoRA delta is tiny (median ||delta||/||W|| = 0.18 percent), at or below bf16 resolution, so the bf16 merge keeps the delta only statistically: only ~38 percent of weights of a target tensor change at all and the realized delta carries rounding noise of the same order as the delta (median error/delta norm 0.62). This is the standard bf16 merge_and_unload outcome, not a bug; it is why "ratio" is ~1 while "cosine" is only ~0.8. If the pi-adapted behaviour looks degraded in bf16 evals, the cheap alternative is to add the fp32 delta inside the quantization script instead of rounding to bf16 first (a ~5 line change to a copy of rotate_w4.py's bf16() loader).

## 6. 4-bit variants (rotate_w4.py, CPU only)

CONFIG: TensorFold main 9e2a9bd7 zig/tools/xpu/rotate_w4.py, template = SergiioB GPTQ NVMe copy. Runs on CPU with torch, no GPU needed (it never touches a device). ~5.5 min per variant under cpu-run (1295 s for all four), systemd MemoryMax=60G.
COMMAND: bin/cpu-run make_4bit_variants.sh (calls: rotate_w4.py BF16_DIR GPTQ_TEMPLATE OUT rot|norot).
RESULT (check_variants.py): index equals the GPTQ template (2399 tensors; 400 quantized projections); hadamard128 marker present only in rot dirs; 60 sampled pass-through tensors (embed, head, norms, MTP, a/b gates for norot) bit-identical to the template; RTN int4 relative weight error 0.111 (rot, vs rotated source) and 0.112-0.116 (norot) vs 0.134-0.151 for SergiioB's GPTQ vs plain W (GPTQ minimizes output error not weight error, so this is not a quality ranking).
FINDING: RTN of the merged pi weights vs RTN of the base weights changes only ~0.55 percent of the int4 codes (layer 11 up_proj sampled), because the LoRA delta is ~0.2 percent of ||W|| and one int4 step is ~7 percent of a group's max. RTN preserves the delta in expectation but mostly erases it per element; do not conclude anything about pi quality from RTN until it is measured on pi-style tasks. A pi-calibrated GPTQ is the principled 4-bit path (section 8).
VERDICT: variants are structurally valid and loadable by the engine in principle; engine load and quality are untested (needs GPU). The rot variants are readable by TensorFold only (recipe section 5 warning); keep the marker file.

Naming: qwen3.8-27b-{base,pi-adapted}-{rot,norot}-rtn4. PROVENANCE.txt in each dir records source dirs and script. README.md and quant_log.csv copied by rotate_w4.py from the GPTQ template were deleted from the variants (they describe the GPTQ checkpoint).

## 7. Run commands for later (GPU steps; nothing here was run)

Common: stop the daily driver first (`systemctl --user stop hotschmoe-dd`, and hotschmoe-dd-frontdoor if that is up) - this prep did not. Run `bin/xpu-health` before and after. Hold `bin/cpu-run --quiet` outside `bin/gpu-run` only for timed windows.
```sh
R=/mnt/vm_8tb/github/b70_ai_things
W=/mnt/vm_8tb/b70/build/tensorfold
M=/mnt/cache/b70/tensorfold-models
GPTQ=$M/Qwen3.8-27B-GPTQ-g128-sergiiob-9d189a60
P=$R/strata/flash-next/concurrency-0216/tensorfold-probe
DRAFT=$M/Qwen3.8-27B-DFlash2
TFN=$W/main/zig-out/xpu/bin/tensorfold-native           # or $W/pr3603/zig-out/xpu/bin/tensorfold-native for the A/B
```

### 7.0 Device selection
TensorFold opens "the first Level Zero device whose PCI device id is 0xe223" (zig/src/xpu/ze.zig openPeers; qwen27.zig). There is no --device for the XPU server (tensorfold-native serve --device is CUDA). The supported way to pick a card is the Level Zero loader variable ZE_AFFINITY_MASK=N (their own receipt docs/receipts/xpu-qwen27-logprobs uses it), which makes card N the only visible device, hence "first". Pair it with the lease: `bin/gpu-run --card N env ZE_AFFINITY_MASK=N ...` (same convention as bin/xpu-health). Check on the first run that the lease card index and ZE index agree (look at the card in xe sysfs/xpu-smi while it loads); if the order differs, add ZE_ENABLE_PCI_ID_DEVICE_ORDER=1.

### 7.1 Pre-flight (one card, no model needed for the first two)
```sh
$R/bin/xpu-health
$R/bin/gpu-run --card 0 env ZE_AFFINITY_MASK=0 $W/main/zig-out/bin/tf-xpu-qwen27-means-test          # group means vs FP64
$R/bin/gpu-run --card 0 env ZE_AFFINITY_MASK=0 $W/main/zig-out/bin/tf-xpu-qwen27-close-test $GPTQ    # 3 opens/closes give the card back
$R/bin/gpu-run --card 0 env ZE_AFFINITY_MASK=0 $W/main/zig-out/bin/tf-xpu-qwen27-seam-test $GPTQ     # prompt cut at 64/1024/whole = same logits
# real-token oracle-style checks (ids.txt from the checkpoint tokenizer):
python3 -I -c "import json;print(' '.join(map(str,json.load(open('$W/prompts/prompt_1024.json')))))" > $W/prompts/ids_1024.txt
$R/bin/gpu-run --card 0 env ZE_AFFINITY_MASK=0 $W/main/zig-out/bin/xpu-qwen27 $GPTQ $W/prompts/ids_1024.txt 32
$R/bin/gpu-run --card 0 env ZE_AFFINITY_MASK=0 XPU_VERIFY_TEST=1 $W/main/zig-out/bin/xpu-qwen27 $GPTQ $W/prompts/ids_1024.txt 32
$R/bin/gpu-run --card 0 env ZE_AFFINITY_MASK=0 XPU_DRAFTER=$DRAFT XPU_TREE=15 $W/main/zig-out/bin/xpu-qwen27 $GPTQ $W/prompts/ids_1024.txt 64
$R/bin/gpu-run --card 0 env ZE_AFFINITY_MASK=0 XPU_MULTI_TEST=8 $W/main/zig-out/bin/xpu-qwen27 $GPTQ $W/prompts/ids_1024.txt 32
$R/bin/gpu-run --card 0 env ZE_AFFINITY_MASK=0 XPU_ROWS_PROBE=1 $W/main/zig-out/bin/xpu-qwen27 $GPTQ $W/prompts/ids_1024.txt 8     # window ms by rows 1..16
```
Prompt id files (real corpus text tokenized with the pi tokenizer, 1024/4096/8192/12288 tokens, JSON arrays) are in $W/prompts/prompt_N.json (made by make_prompt_ids.py from repo docs). Each `xpu-qwen27` check prints "equal to serial: true/false"; any false is a bug.

### 7.2 (a) Single-card 27B serve on port 18144
```sh
$R/bin/gpu-run --card 0 env ZE_AFFINITY_MASK=0 $TFN serve $GPTQ \
  --drafter $DRAFT --parallel 8 --context 16384 --host 127.0.0.1 --port 18144 \
  --name tf-qwen38-27b-gptq-int4-g128-dflash2 --temperature 0 --no-update-check
curl -s http://127.0.0.1:18144/v1/models      # confirm the served id before any benchmark
```
The log prints how many streams the card admitted ("slotsThatFit"; a stream that does not fit is refused at startup). Memory per slot: 2 x 151 MB GDN state + 64 KB per --context token of bf16 K/V; DFlash2 context ~42 MB per slot; prompt cache 2 GiB default (--prompt-cache-gib N, 0 disables). Option envs from docs/xpu/README.md: XPU_KV_INT8=1 (half KV), XPU_PREFILL_INT8=1 (W4A8 prompt GEMMs, GPTQ only), XPU_PREFILL_INT8=1 XPU_PREFILL_INT4=1 on a rot variant (W4A4, refuses without hadamard128). For the pi variants swap $GPTQ for e.g. $M/qwen3.8-27b-pi-adapted-rot-rtn4 and keep --name distinct (model + method + scheme in the id). For 200K context at concurrency 2 the 27B needs 12.5 GiB bf16 KV per stream (6.25 with XPU_KV_INT8=1) on top of ~15 GiB weights: not expected to fit one card at 2 streams; --context 16384 is the validated shape.

### 7.3 (b) Two single-card replicas, 16 streams (ports 18144 / 18145)
```sh
# terminal A
$R/bin/gpu-run --card 0 env ZE_AFFINITY_MASK=0 $TFN serve $GPTQ --drafter $DRAFT --parallel 8 --context 16384 --host 127.0.0.1 --port 18144 --name tf-qwen38-27b-gptq-int4-g128-dflash2-r0 --temperature 0 --no-update-check
# terminal B
$R/bin/gpu-run --card 1 env ZE_AFFINITY_MASK=1 $TFN serve $GPTQ --drafter $DRAFT --parallel 8 --context 16384 --host 127.0.0.1 --port 18145 --name tf-qwen38-27b-gptq-int4-g128-dflash2-r1 --temperature 0 --no-update-check
```
Two independent processes (the engine is single-card for the 27B; no TP). Drive 16 streams as two concurrent client processes of 8 each (section 7.4) and sum, or put a least_conn nginx in front as the DD did (not done here). Each card is a separate lease, so card 1 can run while another agent uses card 0 only if they truly use different cards.

### 7.4 (c) Client benches (stdlib Python, 3.11+; copy from main tools/)
```sh
cd $W/main/tools
M1=tf-qwen38-27b-gptq-int4-g128-dflash2
python3 bench_openai.py http://127.0.0.1:18144 $M1 --tokens 256 --reps 5 --output $W/out_openai_1.json            # single-stream decode, code + chat, T=1 and T=0
for n in 1 2 4 8; do
  python3 bench_concurrent.py http://127.0.0.1:18144 $M1 --levels $n --tokens 256 --reps 3 --alone --serial --output $W/out_conc_$n.json
done                                                                                                               # 1/2/4/8 streams on one card, with the concurrent==alone and drafted==serial token hashes
python3 bench_concurrent.py http://127.0.0.1:18144 $M1 --levels 8 --prompts bench_prompts.json --output $W/out_conc_distinct8.json   # distinct prompts (the default cell sends the same prompt to every stream)
# 16 streams over two replicas: start both at the same time and add the aggregates (bench window overlap matters; use --reps 1 first)
python3 bench_concurrent.py http://127.0.0.1:18144 tf-qwen38-27b-gptq-int4-g128-dflash2-r0 --levels 8 --tokens 256 --reps 3 --output $W/out_r0_8.json &
python3 bench_concurrent.py http://127.0.0.1:18145 tf-qwen38-27b-gptq-int4-g128-dflash2-r1 --levels 8 --tokens 256 --reps 3 --output $W/out_r1_8.json &
wait
# cold prefill (tokens over TTFT of a cold first request) with real token ids, no cache hit expected:
for n in 1024 4096 8192 12288; do python3 bench_cache.py http://127.0.0.1:18144 $M1 --prompt-file $W/prompts/prompt_$n.json --expected-prompt-tokens $n --tokens 16 --require-cold --output $W/out_cache_$n.json; done
# prefill-while-decoding gap (p95 inter-token gap of a running stream while a long prompt fills):
python3 bench_fill.py http://127.0.0.1:18144 $M1 --corpus $W/corpus.txt --long 12288 --reply 512 --output $W/out_fill.json
```
Levels 1/2/4/8 as requested; 16 only via two replicas (MAX_SLOTS per process is 16 but the 32 GB card admits fewer at 16k context). Record per run: engine commit, binary sha256 (section 3), checkpoint dir, card power limit (README numbers are at 230 W), DD state, post-health. The tokenizer for token-id prompts must be the served checkpoint's own (make_prompt_ids.py takes it as an argument).

### 7.5 (d) pr3603 Flash-Next TP bench on two cards
- tf-xpu-tp-bench needs no checkpoint: P2P TP=2 and TP=4 collective suite (bit-exactness vs rank-ordered reference and latency up to 40 MB). It opens peers through zeDeviceCanAccessPeer and errors out with "skipped TP=2: ..." if the pair is not peer accessible; TP=4 prints "skipped" with two cards. Both cards must be visible (no ZE_AFFINITY_MASK). Note the repo rule: do not run arbitrary CCL_TOPO_P2P_ACCESS=1 vLLM TP>1 serves; this is a different (Level Zero direct) path but the same hardware risk class, so run bin/xpu-health and bin/xpu-collective-health around it and bin/xe-reset on failure.
```sh
$R/bin/xpu-health && $R/bin/xpu-collective-health
python3 -I $P/ze_p2p_precheck.py                         # see 7.6 first, under gpu-run (both cards)
$R/bin/gpu-run $W/pr3603/zig-out/bin/tf-xpu-tp-bench     # whole-box lease; same binary exists for main
```
- xpu-flashnext (pr3603 only) is "TP={1,2,4} forward and MTP runner": `XPU_TP=2 XPU_CTX=16384 xpu-flashnext MODEL_DIR (--bench=2048,8192 | ids.txt) [STEPS]` (env: XPU_TP default 4, XPU_CTX, XPU_MTP, XPU_MTP_DEPTH 6, XPU_MTP_CONF -0.4, XPU_CHUNK, XPU_VERIFY_TEST). Its checkpoint is TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP (their receipts: revision 2b170fa6309d5d1ee380b35636075fac7945f286; docs/supported-models.md: 73.3 GiB uploaded). Footprint on cards per their numbers: ~79.2 GiB total = 19.8 GiB per card at TP=4 (plus 23.8-24.4 GiB per card at the full 262,144 window), PLE ~29.8 GiB in host RAM. Two B70s hold 64 GB, so TP=2 (about 39.6 GiB per card) does NOT fit; this is arithmetic from their figures, not a load attempt. NO Flash-Next checkpoint was downloaded. xpu-flashnext on 2 cards is therefore only useful with a smaller/sub-4-bit checkpoint that does not exist yet. The 27B (xpu-qwen27/serve) path is the thing to measure on our hardware; the pr3603 build also contains it (their PR note: xpu-qwen27 2,048-token prefill 1,970 tok/s, decode 32.8 tok/s serial, exact tokens) and is the A/B partner of main.
- Also available in pr3603 (not run): `zig build xpu-kernel-contract-limits` (no GPU, audits launch limits against BMG) and `xpu-contract-run` (gates 1-4 on a card).

### 7.6 (e) Level Zero P2P precheck (not run)
ze_p2p_precheck.py in this directory: zeInit(GPU only), enumerate devices per driver, print deviceId/name, then zeDeviceCanAccessPeer for each ordered pair. No context, memory or kernel. TensorFold's openPeers requires BOTH directions to be 1 before it will open TP>1 at all, and returns ZeFailed otherwise.
```sh
$R/bin/gpu-run python3 -I $P/ze_p2p_precheck.py      # whole-box lease, both cards visible
# expected from docs/P2P_GPU.md H.11: canAccessPeer True on kernel 7.0 with IOMMU off; re-measure on 7.1.
```
(There is no Level Zero header on the host; the script uses ctypes against /usr/lib/x86_64-linux-gnu/libze_loader.so.1 (1.28.2) with the stable ze_device_properties_t layout: stype@0, deviceId@24, name@112.)

## 8. GPTQ calibration of the pi model (later, needs a GPU; not started)

What the SergiioB checkpoint used (recipe + its quantize_config.json): gptqmodel 7.3.2 (PyPI sdist only, 1.0 MB; requires torch>=2.15 on python>=3.11, transformers>=5.22.2, accelerate>=1.15, device-smi, tokenicer, logbar, defuser, torchao, datasets, ninja, pypcre ...), bits 4, group 128, sym true, desc_act false, damp_percent 0.05 with damp_auto_increment 0.01, true_sequential true, act_group_aware true, fallback RTN at threshold 0.5 percent, pack_dtype int32, dynamic {"-:.*mtp.*": {}} (MTP kept bf16), lm_head false. Calibration dataset, sample count and sequence length are NOT recorded anywhere (recipe says TODO). Its quant_log.csv: 400 modules, summed per-module time 46.3 min (device unknown) on top of calibration forwards.
What it would take for pi (feasibility estimates, labelled as estimates, nothing measured):
- Input: qwen3.8-27b-pi-adapted-bf16 (52 GB, plus the fp32 delta alternative of section 5 if bf16 merge noise matters). Outputs must satisfy the engine requirements (recipe section 2): qweight int32 [K/8,N], scales fp16 [K/128,N], embed/lm_head/in_proj_a/in_proj_b stored FP16 (gptqmodel loading with dtype float16 does this), no g_idx, MTP bf16, tokenizer = the pi files.
- Hardware: layer-wise GPTQ needs only one decoder layer (~0.85 GB bf16) plus Hessians (largest down_proj 17408^2 fp32 = 1.2 GB) plus calibration activations (e.g. 128 x 2048 x 5120 x 2 B = 2.7 GB) on the device, so it fits one 32 GB B70 if gptqmodel's torch.xpu path works with torch 2.15 XPU wheels (unverified here); otherwise CPU: weight-time per module is seconds, but the forward through the 48 GDN layers on CPU uses the torch chunk fallback of gated-delta-rule (no fla kernels) and is the slow part. Estimate: ~1-2 h on XPU/GPU, ~10-24 h on 32 CPU cores. Calibration set: for a pi model use pi-style agent trajectories (tool calls rendered with the pi chat_template) mixed with some code and general text, 128-512 samples x 2048-4096 tokens; document the exact set (the upstream recipe does not).
- Check the result with the same gates as the SergiioB checkpoint: loads in tensorfold (GPTQ detection via in_proj_qkv.qweight), oracle/drafted==serial/multi==serial tests, perplexity on raw text vs the RTN and base GPTQ variants, then pi-style task evals. Check DFlash2 acceptance on the pi model: the drafter was trained on base-model hidden states (taps layers 5,19,33,47,61); the LoRA changes those states slightly, so acceptance may fall; drafts never change the reply, only speed. The adapter card says it is used with thinking off.

## 9. Open items for the GPU stage
1. Stop hotschmoe-dd (this prep did not). Run xpu-health, then 7.6 P2P precheck, then 7.1, then 7.2.
2. Confirm ZE_AFFINITY_MASK index == gpu-run card index.
3. Decide main vs pr3603 as the primary binary; run both on the same checkpoint for the A/B (identical GPTQ dir, same --parallel/--context).
4. Real-model load of the new variants (rot/norot, base/pi) is untested; the rot variants need the engine to find the hadamard128 marker file in the model dir.
