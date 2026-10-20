# G2 audit

CPU only. No GPU, no edits outside this file and JOURNAL.md.
Each finding: claim, evidence, verdict, suggested fix.
Newest pass at the bottom.

## Pass 1 -- 2026-10-20 07:00 UTC

Cursor: JOURNAL.md through the 06:51 E1 mixed-bits entry (1206 lines).
Bench JSON under /mnt/vm_8tb/b70/build/vllm-flashnext/logs.
C2 logs under /mnt/vm_8tb/b70/build/c2-prefill-gpu-*.log.
C5 logs under /mnt/vm_8tb/b70/build/vllm-flashnext/c5/.
Fork /mnt/vm_8tb/b70/build/vllm-flashnext/vllm branch b70/flashnext.

### F1. Serve bench numbers match the raw JSON

Claim: JOURNAL C1/C2 ITL, tok/s, TTFT, prefill, and MTP acceptance
for A3 eager, C3 int4, C3 int3, C4 outfix, D1 eager/MTP/graph arms,
and E1 int3 graph match the bench files.
Evidence: parsed every *.json in the logs dir. Rounded journal
figures match the JSON to the printed precision. Examples:
a3-bench-eager.json C1 74.78185/75.36336/76.39513 ms, 13.56867 tok/s,
n=64; prefill 169.48/1037.71/5120.60 ms.
d1-bench-graph-fa.json C1 12.81/12.99/13.12 ms, 78.23 tok/s.
d1-bench-graph-fa-mtp.json C1 18.81 ms, 93.88 tok/s, accept 85/222
and after 95/266.
e1-int3-graph-bench.json C1 11.29/11.52/11.97 ms, 90.34 tok/s, n=57,
prefill 99.33/112.32/115.78 tok/s.
d1-step-timer-mtp.log tokens=3 execute wall_ms p50 is 3.74 (66 samples
under 20 ms; journal said n=65). replay_device_ms=2.275 is in that log.
d1-step-timer-sync.log replay_device_ms=11.436.
d2-gemv-floor.log fp16 stack M=1 7.153 ms, 542.0 GB/s; int4 qkvz
0.1689 ms, lm_head 1.1909 ms, replay_max_abs 0.
e1-int3-ppl.json nll 2.21045595, ppl 9.11987, windows 1.81316/2.44070/
2.21457/2.37339, tokens 2044. exp(nll) matches ppl.
Verdict: OK.
Suggested fix: none.

### F2. Identity and the coherence gate did run

Claim: served benches queried /v1/models and ran two temperature-0
completions, with first id hotschmoe-dev.
Evidence: every bench JSON has identity.first_ok true and ids[0]
hotschmoe-dev. coherence.ok, same, mentions_paris, and both texts are
stored. Ornith GPTQ arms (A3, C3 int4, all D1, F1 host and F1 FA)
repeat " Paris.\nThe capital of France is Paris.\nThe capital of France is".
C3 int3 eager and E1 int3 graph repeat the same different string,
" Paris, a city renowned for its rich history...", on the Qwen3.6
checkpoint. C4 outfix repeats the Ornith-style Paris string.
Verdict: OK.
Suggested fix: none for these arms.

### F3. UMD 26.27 Triton greedy text diverged

Claim: f1-bench-2627-triton.json sets coherence.ok true. No JOURNAL
result entry has claimed this arm yet (06:20 F1 entry says the
campaign was starting). The JSON is still a recorded claim.
Evidence: serve log line 1 is Ornith GPTQ int4, card 0, port 18091,
FULL_DECODE_ONLY, TRITON_ATTN, gdn=triton, mtp=0, loader_bytes
1036264, igc_bytes 96015048 (the 26.27 sizes in the 06:20 entry).
Both samples match each other and contain "Paris", so the harness
gate passes. The text is " Paris.\nA. True\nB. False\nThe capital of
France is". Matched arms on the same checkpoint, prompt, and
temperature 0 do not: f1-bench-host-triton.json, f1-bench-host-fa.json,
f1-bench-2627-fa.json, f1-bench-2635-fa.json, and d1-bench-graph-fa.json
all produce the capital-of-France continuation. C1 ITL on this arm is
13.18 ms, close to host Triton 13.13 ms, so the speed is similar and
the tokens are not.
bench_latency.py coherence() only requires the two texts to match,
the word Paris, and a non-GARBAGE verdict (harness/bench_latency.py:244).
Verdict: WRONG if this arm is later called coherent with the Ornith
control. The gate result is internally consistent and too weak.
Suggested fix: for this prompt at temperature 0, compare the completion
to the frozen host string. Do not treat mentions_paris as a match
across a UMD or attention change. Keep the 26.27 Triton arm out of
any "same quality" claim until the string matches or the divergence
is explained.

### F4. F1 26.35 Triton bench is not in the JSON set yet

Claim: none in JOURNAL. Raw serve log exists.
Evidence: f1-serve-2635-triton.log was still in weight load at
06:54:46 (model load 36.9 s, no bench JSON beside it). Other F1
health logs say HEALTHY (cards 0) after host, 26.27, and 26.35 FA.
f1-driver.log has no apt or dpkg lines.
Verdict: UNVERIFIED. In progress, not a false claim.
Suggested fix: when the bench JSON lands, check the greedy text
against the host string before writing the arm up as coherent.

### F5. Health and matched-config disclosures

Claim: C2 and C5 ran xpu-health before and after. C3 int4 and int3
say post-health did not run. E1 int3 graph vs C3 eager is the same
checkpoint. C3 int3 vs A3 Ornith is not. C4 C2 is not real
concurrency 2.
Evidence: c2-prefill-gpu-bench2.log and c2-prefill-gpu-correct.log
print HEALTHY before and after, failures=0. C5 pre-health.log,
post-health.log, pre/post-health-020.log, pre/post-health-ab.log
print HEALTHY (cards 0). C3 journal text matches the missing
post-health. e1 and c3-int3 JSON share alias qwen36-w3a16-b70-lowbit
and the same prefill rates (98.98/112.72/115.83 eager vs
99.33/112.32/115.78 graph). C4 JSON C2 TTFT is 3198 ms vs 302 ms
with max_num_seqs 1, as the journal says.
D1 05:07 "about 5x" compares Triton-graph 13.20 ms to eager profile
71.85 ms. That eager arm is FlashAttention plus SYCL GDN. The same
entry says so. The later FA graph is 12.81 ms against that eager
71.85 ms. f1-bench-host-fa.json later reads 12.80 ms on the same
shape. Not a contradiction.
Verdict: OK. C3 card-0 post-health stays UNVERIFIED because it was
not run.
Suggested fix: run bin/xpu-health --card 0 on the next free window
and append the line. The owed check is already written down.

### F6. C2 and C5 kernel numbers

Claim: C2 layer times, the 344 us grouped regression, and C5 routed
microseconds match the logs.
Evidence: c2-prefill-gpu-bench2.log: T=512 13884.8 us (768.2 tok/s),
T=4096 27598.9 us (3091.9), T=12288 66129.1 us (3871.2), bf16 T=512
24708.5, int8 T=512 29103.5, IQ4_NL M=128 344.4 us. A2's 136.0 us
baseline is reports/1A2_gguf.md:128. That raw A2 bench log was not
in the logs dir this pass.
C5 repro_ocloc_iq3s_c1c0.log: R2 all 96.87, IQR2 all 72.64, bitwise
yes (153600, 0). meas_jit_strata_oldrt.log: R2 93.47, IQR2 69.26.
meas_ours_plugin_jit.log: graph 198.39 us. meas_ours_aot_only.log:
s20 graph 366.61 us.
Verdict: OK for the C2 and C5 measurements. The A2 136 us comparison
baseline is UNVERIFIED against a raw log this pass (report only).
Suggested fix: cite the A2 raw log path next to the 136 us figure.

### F7. E1 grouped probe matches; mixed-bits run has no log

Claim: grouped probe PASS with the stated errors, and test_mixed_cpu.py
printed MIXED_BITS_OK with hist {2: 7573, 3: 2667}.
Evidence: e1-grouped-probe.log lines 41-49 match the journal,
including b2 rel_l2 6.7185e-09, max_abs 0.125, 11.94 vs 29.19 ms,
GROUPED_PROBE_OK, probe_rc=0.
test_mixed_cpu.py encodes those asserts (lines 75, 89, 111, 139).
No stdout log containing MIXED_BITS_OK was found under
vllm-flashnext/logs, /mnt/vm_8tb/b70/logs, or the home log tree.
Verdict: grouped probe OK. Mixed-bits result UNVERIFIED.
Suggested fix: keep the docker stdout at
logs/e1-mixed-cpu.log and cite it.

### F8. Repo hygiene and host stack

Claim: no secrets and no large binaries staged. F1 did not change
the host driver. The --umd path is opt-in.
Evidence: git diff --cached is empty. vllm/flashnext/.gitignore has
*.so. Six plugin .so files are ignored (largest _C.so 6391032 bytes),
not in git ls-files. No secret-like names in git status.
vllm/contrib/.../libxpu_push_ar_graph.so is a tracked 208312-byte
file, modified, not staged, mtime 2026-08-25, outside this campaign.
xpu-entrypoint.sh check_loader still requires
libze_loader.so.1.28.2 under /host-umd and fails a resolved path
containing "1.32" or "26.27" (lines 157-250). --umd is in HEAD
(commit febd6141) and only swaps mounts when UMD_DIR is set
(xpu-run.sh:381). Fork status is clean, b70/flashnext ahead 7,
tip e3f784ebb2, which matches the C4 journal tip.
Verdict: OK.
Suggested fix: leave the August push-allreduce .so unstaged unless
that change is intentional. Do not git add the plugin .so files.

### F9. Code review, plugins and the D2 patches

Claim: none numeric. Review asked for races, strides, bounds, and
host syncs inside graph-captured ops.
Evidence:
- b70_lowbit ops.py and moe.py decode (T<=8) have no .item(), .cpu(),
  or synchronize. moe.py:418 counts.tolist() is the prefill host sync
  the journal already reports. It is not on the T<=8 graph path.
- lowbit_kernels.cpp checks bits, K%128, T<=8, expert id range on the
  indexed GEMV, and H/I<=2560. GroupedPrefillKernel returns before
  its barrier only when the whole work-group has e>=E or count<=0,
  so the barrier is not split. The slot loaded from order[] is not
  range-checked before the x load (lowbit_kernels.cpp:1444).
- kreg.hpp b70_device_grow does not queue.wait. It throws if the
  queue state is recording and the block must grow, and it leaks the
  old block instead of freeing it (lines 312-339). Scratch pointers
  are function-local statics with no lock. grouped_prefill.cpp
  q.wait() runs only when B70_PREFILL_SPLIT=1 (split_on, line 38).
- d2_int4_linear.py synchronize is in _pack_model (load), not in
  _try_gemv. d2_mtp_shisa.py .cpu() is in apply() cosine/fingerprint
  at load, not in a replay forward. d1_step_timer.py and d2_profile.py
  syncs are measurement hooks.
Verdict: OK for the decode graph path as written. Two latent items,
neither shown failing in a log: the grouped prefill slot is unchecked,
and the recording-state check plus the static scratch are unproven
under a real vLLM XPU capture (GGUF graph serve has not run).
Suggested fix: clamp the grouped slot before the load. Call
b70_reserve_decode before capture, and on the first GGUF graph serve
confirm ext_oneapi_get_state reports recording. Give each stream its
own scratch if two queues share a process.

### F10. Raw F1 numbers, not yet journaled

Claim: none yet. Recorded so the next pass can check the writeup.
Evidence, C1 ITL p50 / tok/s from the JSON:
host FA 12.80 ms / 78.27; host Triton 13.13 / 76.34;
26.27 FA 12.77 / 78.40; 26.27 Triton 13.18 / 75.96 (text diverged, F3);
26.35 FA 12.60 / 79.36. 26.35 Triton bench JSON absent (F4).
Identity first id is hotschmoe-dev on every completed F1 JSON.
Verdict: OK as a ledger. Not a speed claim by F1.
Suggested fix: when F1 writes the journal entry, diff it against this
table and against F3.

## Pass 2 -- 2026-10-20 07:35 UTC

Cursor: JOURNAL from the 07:08 C7 entry through the 07:29 G3 entry.
New files: c7-bench-graph.json, c7-greedy.json, c7-serve-graph.log,
c7-graph-probe.log, c7-cpu-mapping.log, f1-bench-2635-triton.json,
e1-int4-graph-bench.json, e1-int4-ppl.json, e1-int4-serve.log,
reports/1F1_umd.md, flashnext_ref/last_compare.json.

### F11. F1 latency table matches; "all six coherent" does not

Claim: JOURNAL 07:08 and reports/1F1_umd.md. All six serves coherent
(Paris, both strings identical, hotschmoe-dev first). FlashAttention
C1 ITL p50/p99 and TTFT: host 12.80/13.13 and 84.68; 26.27 12.77/13.25
and 89.67; 26.35 12.60/13.00 and 92.07. Prefill 32k tok/s 6182/6171/6193.
Triton C1 p50 13.13/13.18/12.97. Triton 32k 1224/1868/1705. GEMV Q4_K
129.4/132.3/128.9 us, Q5_K 810.6/809.5/835.7 us. moe_layer T=8 behind
1408.5/1424.8/1441 us. T=1 370/589/254 us.
Evidence: those latency figures match f1-bench-*-fa.json,
f1-bench-*-triton.json, f1-gguf-host.log, f1-gguf-2627.log, and
f1-gguf-2635-retry.log, including the report's 1k/8k columns.
Identity first id is hotschmoe-dev on all six JSON files.
The four FlashAttention texts and the host Triton text are the
capital-of-France continuation. f1-bench-2627-triton.json and
f1-bench-2635-triton.json are both
" Paris.\nA. True\nB. False\nThe capital of France is", identical
within each arm, and coherence.ok is true only because the harness
gate checks the word Paris. The report's FlashAttention section quotes
the capital string and does not claim the Triton text. The journal
sentence covers all six.
Verdict: WRONG on "all six serves coherent" as a matched greedy
result. Latency, identity, and the microbench numbers are OK.
Suggested fix: split the sentence. FlashAttention arms and host
Triton match the frozen string. 26.27 Triton and 26.35 Triton match
each other and do not match the host. Do not rank those two arms as
quality-neutral. The shared wrong string points at non-host IGC plus
Triton, not at a one-off sample.

### F12. C7 GGUF graph bench matches the JSON

Claim: GRAPH_PROBE_OK, max_abs 0, 4 FULL graphs, capture 1 s, pool
0.02 GiB, KV 8.92 GiB / 381493 tokens / 11.29x, placed 733, both
greedy texts the C4 Paris string, C1 ITL 11.11/12.41/13.46 ms,
90.16 tok/s, TTFT 174.36 ms, n=64. C2 16.51 ms / 61.34 tok/s / n=46
and 16.65 ms / 62.01 tok/s / n=40. Prefill 177.01 / 175.58 / 173.23
tok/s at TTFT 5746 / (8185) / 189124 ms. CPU suite ALL_DONE 0.
Fork tip at that write was 2583e27239. _C.graph.so is 6987192 bytes.
Evidence: c7-graph-probe.log lines 40-42. c7-serve-graph.log: placed
733, reserve_decode on xpu, capture progress 2 then 4 FULL graphs,
"finished in 1 secs, took 0.02 GiB", KV line 381,493 tokens, 11.29x.
Serve command is host UMD (libze_loader.so.1.28.2, IGC 2.36.3),
FULL_DECODE_ONLY, FLASH_ATTN, B70_GGUF_SO=_C.graph.so.
c7-bench-graph.json and c7-greedy.json match the ITL, token counts,
and both texts. c7-cpu-mapping.log: N 15, ALL_DONE 0, exit 0, 91 s.
test_linear_and_moe_call_ops passed; that test asserts one moe_layer
call and id -1 (fork test_b70_gguf_mapping.py:562). .so is gitignored
and 6987192 bytes, mtime 06:59. Fork log shows 2583e27239 at 07:08:38.
A later commit 4d73572e14 landed at 07:31:07, after this measurement.
No c7 xpu-health log was found under logs/.
Verdict: OK for the bench, capture, identity, and CPU suite.
Post-health is UNVERIFIED. The measured serve is not the fork tip
anymore.
Suggested fix: cite the health command's log. Any later GGUF serve
should name 4d73572e14 or say it is still on 2583e27239.

### F13. E1 int4 graph and perplexity

Claim: C1 ITL 10.23/10.37/10.48 ms, mean 10.229, 99.59 tok/s, TTFT
2533.19 ms, n=56. C2 13.65 and 13.61 ms, 75.25 and 76.22 tok/s.
Prefill 96.90/108.74/112.99 tok/s. Texts start " Paris, a city
renowned for its rich history, art, and culture. Sit". Ppl 8.8582,
nll 2.1813, windows 1.7888/2.4407/2.0974/2.3985, tokens 2044.
Int4 expert bytes 519045120 vs int3 393216000 is 32 percent more,
and the step is 1.06 ms faster than int3's 11.29 ms.
Evidence: e1-int4-graph-bench.json matches p50/p90/p99, tok/s, TTFT,
n, C2, prefill, and the text. first_ok true, alias
qwen36-w4a16-b70-lowbit. itl_mean_ms prints as 10.227, not 10.229.
e1-int4-ppl.json: ppl 8.85816, nll 2.18134, windows 1.78885/2.44069/
2.09735/2.39847, tokens 2044. e1-int4-serve.log repack bits=4,
plane shapes (256, 1024, 512) and (256, 2048, 128).
519045120/393216000 = 1.320. 11.29-10.23 = 1.06. Byte/time is about
51 and 35 GB/s. The absolute byte totals were not rebuilt from the
checkpoint tensors this pass.
Verdict: OK. The mean-ITL digit 10.229 is a 0.002 ms slip.
Suggested fix: quote mean 10.227. Show the byte product (slots,
layers, plane sizes) next to 519045120.

### F14. G3 CPU compare file matches the journal

Claim: vocab sum 320001446, eos 248044 differs from 248046 in 32 of
400 rows, strata vs XPU at 248044 is 0 of 400, HC mix max_abs
1.311e-6 against (1+w) and 1.278 against the raw gamma, PLE block
max_abs 4.77e-7, a second +1 moves it by 1.46, IQ4_NL bit_mismatches
0, wide vs eh_proj max_abs 3.97.
Evidence: flashnext_ref/last_compare.json has those fields:
vocab_sum 320001446, strata_vs_xpu_248044 n_diff 0 of 400,
xpu_248044_vs_248046 n_diff 32, mix max_abs 1.311302e-6 and
1.277879, PLE max_abs 4.76837e-7, stored-gamma delta 1.46408,
bit_mismatches 0, eh_proj max_abs 3.97262.
Verdict: OK.
Suggested fix: none.

### F15. G1 contract numbers are in the tests; the run log is not

Claim: 16 tests OK. GDN page 2146304 bytes/layer, attention 20480
bytes/token, scaled block 1088. A 28*1088 checkpoint keeps 27*1088.
Evidence: kvoffload/tests has 16 test methods. test_layout_park.py
asserts 2146304, 1088, 20480, and 28*1088 -> 27*1088. No unittest
stdout was found under logs/ or b70/logs.
Verdict: UNVERIFIED that the process exited 0. The expected numbers
match the test source.
Suggested fix: save the unittest transcript next to the journal entry.

## Pass 3 -- 2026-10-20 08:16 UTC

Cursor: JOURNAL.md through the 08:11 C7 payload entry (1635 lines).
New since pass 2: E1 mixed probe, G1 KV restore, E1 mixed graph,
C8 prefill, G4 park index, C7 payload copy. Fork tip moved from
4d73572e14 to 2a50cfe2a7 during this pass.
D2 shisa JSON and e1-grp-serve.log have no journal claim yet.

### F16. E1 mixed probe matches the log

Claim: moe_gemv_mixed matches templated moe_gemv on uniform int3
(rel_l2 0) and matches prefill within rel_l2 0.02 and cos 0.999.
probe_rc=0. Library libb70_lowbit_xpu_e1mix.so is 2280328 bytes.
Evidence: logs/e1-mixed-probe.log. uniform3 vs moe_gemv
rel_l2=0 cos=1 max_abs=0. uniform3 vs prefill rel_l2=4.2257e-03.
mix-small rel_l2=1.5849e-03. mix-23 full K rel_l2=1.0336e-03
cos=0.999999 max_abs=1024. mix-234 rel_l2=3.9934e-03 cos=0.999992
max_abs=2048. MIXED_PROBE_OK. gpu-run done in 18s exit 0.
probe_rc=0. The .so path in the log is 2280328 bytes.
No xpu-health line in that log.
Verdict: OK. Health before and after was not recorded.
Suggested fix: none for the numbers. Print xpu-health on the next
GPU probe.

### F17. G1 host restore matches the evict JSON and the serve log

Claim: Ornith PIECEWISE serve on card 0 restores a fully evicted
prefix. Block 1088. KV 245 -> 45 (39852 tokens, 1.22x). Cold TTFT
5.177s (5884 tok/s), resident 0.267s, restore 0.732s with cached
29376, computed 1088, host delta 29376, device delta 0. One load
of 666009600 bytes in 0.461s. Stores 2064875520 bytes, 93 jobs,
0.475s. Greedy texts match. Hold 330s exit 0. Post-health HEALTHY
in 16s. Earlier inductor serve died on missing ocloc, 114s, exit 1.
Evidence: g1-evict.json has those fields (restored_ttft_s 0.73167,
cold_ttft_s 5.17721, 30464/5.17721 = 5884 tok/s). All three texts
are "Note A. The quick brown fox keeps talking about topic A. Note A".
passed true. g1-metrics.txt load 6.660096e+08 in 0.46075s, one load,
store 2.06487552e+09 in 0.47466s, count 93. 29376*20480 + 30*2146304
= 666009600. g1-serve.log: served-model-name hotschmoe-dev first,
cpu_bytes 17179869184, two PIECEWISE captures of 5/5, override
245 -> 45, 39,852 tokens, 1.22x, done in 330s exit 0. Prometheus
model_name is hotschmoe-dev. g1-health-after.log HEALTHY cards 0,
done in 16s exit 0. g1-serve-inductor-fail.log FileNotFoundError
ocloc, done in 114s exit 1. The "card busy" lines are flock waits
in bin/gpu-run, then a later acquire. No /v1/models body was saved.
No g1-health-before.log.
Verdict: OK for the restore numbers and the post-health. This run
used its own greedy prompt, not the Paris harness. Pre-health was
not saved.
Suggested fix: save /v1/models next to g1-evict.json.

### F18. E1 mixed graph bench, perplexity, and serve log match

Claim: qwen36-mixed FULL_DECODE_ONLY, C1 ITL 10.63/10.96/11.12 ms,
95.54 tok/s, TTFT 2123.48 ms, n=64. C2 p50 14.26 and 13.97. Prefill
112.47/126.94/129.10 tok/s. Ppl 10.1590, nll 2.3184, windows
1.9115/2.5400/2.3456/2.4763, tokens 2044. Identity hotschmoe-dev,
both texts the capital-of-France string. bits=3, 40 layers, shapes
(256, 192, 1024) and (256, 48, 2048). Two captures of 1 s. Weight
14.38 GiB in 522 s. KV 10.78 GiB. init engine 18.07 s. Exit 0.
Design-mix bytes 368050176. Count-weighted average 300158976.
Evidence: e1-mix-graph-bench.json first_ok and alias_ok. C1 p50
10.629, p90 10.958, p99 11.117, mean 10.636, tok/s 95.539, TTFT
2123.475, n=64. C2 streams match the printed p50/p90/p99 and tok/s
(p99 21.745 and 14.916). Prefill rows match. Both texts are
" Paris.\nThe capital of France is Paris.\nThe capital of France is".
e1-mix-ppl.json nll 2.318358, ppl 10.15898, windows 1.911537,
2.539995, 2.345628, 2.476273. e1-mix-serve.log: FLASH_ATTN,
FULL_DECODE_ONLY, B70_LOWBIT_MOE_LAUNCHES=2, B70_LOWBIT_SO=
libb70_lowbit_xpu_e1mix.so, B70_LOWBIT_PREFILL absent,
B70_XPU_CG_RECLAIM=500. Forty w13 repacks at bits=3 with those
shapes. Model loading took 14.38 GiB and 522.401 s. Available KV
10.78 GiB. Two FULL captures finished in 1 s each (sizes 2 then 4).
init engine 18.07 s. gpu-run done in 981s exit 0.
0.8*393216000+0.2*267386880 = 368050176.
(7573*267386880 + 2667*393216000)/10240 = 300158976, using the
hist asserted in test_mixed_cpu.py. int4 10.23 / int3 11.29 are
the previously checked benches. 11.29-10.63 = 0.66 ms. No HEALTH
line in the serve log. Checkpoint file size on disk is 14.23 GiB;
14.38 GiB is the device-memory line.
Verdict: OK. Different checkpoint from the int3 and int4 arms, and
the journal quotes this arm's own greedy string. Health before and
after was not recorded.
Suggested fix: record xpu-health for this serve. The prefill path
is still the per-expert loop, which the entry already says.

### F19. C8 prefill times match the build logs

Claim: dense Q4_K reuse-rm64 1227.1/10243.9/30689.3 us
(26.25/25.16/25.19 TF) at M=512/4096/12288. torch.mm
323.0/1855.5/5481.5 us. C2 was 5296/42016/126036 us. Experts
9716.9, 28335.7, 76670.8 us. Down at T=12288 is 19.2 ms vs C2
11.7 ms. Decode behind_us merged 367.3/315.0/515.1/815.1 vs
04:32 472.7/417.0/713.9/1383.3. compare_cpu 13/13 mismatches=0.
HEALTH_BEFORE/AFTER 0. Full link 89s exit 0. _C.so sha
086595a0... GEMM-only projection 7265 ms, 1691 tok/s. Dense is
4.1x C2 and 5.6x behind torch at M=12288.
Evidence: /mnt/vm_8tb/b70/build/c8-prefill-c8bench2.log has those
dense, torch, and layer lines, HEALTH_BEFORE 0, HEALTH_AFTER 0,
failures=0. Down split at T=12288 s64-mt16 is 19165.4 us.
c2-prefill-gpu-bench2.log dense 5295.8/42015.7/126036.2 us and
down=11710.2 us. Expert rows 13884.8/768.2, 27598.9/3091.9,
66129.1/3871.2. c8-task3.log has both behind_us sets, TASK3_FAIL:0,
HEALTH_BEFORE 0 and HEALTH_AFTER 0. c8-compare-cpu.log every row
mismatches=0 and VERDICT failures=0 types=13/13.
c8-full-build.log cpu-run 89s exit 0. sha256 of _C.so and
_C.c8full.so is 086595a01babbc366b3d5366147464ebd24022fba0b65d83b1ff530489c70d55.
_C.c8.so sha starts 82f8e4296849c1c3. 2*12288*12288*2560/30.6893e-3/1e12
= 25.19 TF. 126036/30689 = 4.11. 30689/5482 = 5.60.
48*76670.8 us = 3680.2 ms. The report's piece table sums to 7265.0 ms.
12288/7.265 s = 1691 tok/s. That projection is not a served prefill.
_C.graph.so is still 6987192 bytes, mtime 06:59, and was not the
library under test. *.so matches vllm/flashnext/.gitignore line 2
and is not staged.
Verdict: OK.
Suggested fix: none. Keep the 1691 tok/s figure labeled as a GEMM
projection when it is quoted against the 2000 tok/s target.

### F20. G4 byte model and the 28-test transcript match

Claim: 28 tests OK in 0.003s. Flash-Next 200k conversation 5.440 GB,
4 parked plus one checkpoint each 22.224 GB, int8 2.820 and 11.740 GB.
Ornith fp16 attention 20480 B/token, int8 10240, GDN page 2146304.
Host reserve 2684354560. A tail with end == prompt length is a miss.
Evidence: logs/g4-t0-park-index.txt lists 28 tests and "OK" after
0.003s. FlashNextTextLayout().conversation_bytes(200000) is
5440442368 (5.440 GB). parked_with_one_checkpoint_each(4, 200000)
is 22223601664 (22.224 GB). int8 pair is 2819642368 and
11740401664 (2.820 and 11.740 GB). HOST_RESERVE_BYTES is 5<<29.
test_park_index.py asserts 20480, 10240, 2146304, and the reserve.
index.py lookup_key and find return no hit when end >= prompt_len.
The 16 tests from F15 are in this run and passed. The 07:12 process
stdout is still absent.
Verdict: OK.
Suggested fix: none. F15's missing 07:12 transcript stays a gap for
that process only.

### F21. C7 fork identity matches; the payload run has no transcript

Claim: fork tip 2a50cfe2a7, parent c38d0a6197 (park_finished_tail,
default off). PAYLOAD_TEST_OK in 25s. A 35B quantized tensor matches
the mmap, add-1 leaves the mmap, and a non-contiguous slice does the
same. _C.graph.so still 6987192 bytes, mtime 06:59. Commit not pushed.
Each DP worker copies the PLE table once.
Evidence: git log on b70/flashnext shows 2a50cfe2a7 parent
c38d0a6197, ahead 11 of origin/main. cache.py park_finished_tail
defaults False. _C.graph.so stat is 6987192 bytes at 06:59:04 UTC.
The new test asserts the three aliasing checks. It does not print
PAYLOAD_TEST_OK. That string is only in JOURNAL.md. No transcript
under logs/ or /mnt/vm_8tb/b70/logs. The commit message says a
contiguous memmap was copied twice. The test does not count copies
and does not start two workers.
Verdict: UNVERIFIED that the 25s run exited 0. Commit, parent,
default-off flag, and the serve library stat are OK. The DP-worker
sentence is the commit message, not a measured two-worker result.
Suggested fix: save the cpu-run stdout. Say the test checked
aliasing.

### F22. Code spot check and repo hygiene

Claim: none filed by other agents. This pass looked for a new
graph-capture host sync, a bad expert index, and a staged binary.
Evidence: lowbit_kernels.cpp GEMV writes zero and returns when
e < 0 or e >= E, before the weight load. gguf csrc has no
synchronize, .item, .cpu, or tolist; kreg.hpp still says capture
must not queue.wait. E1's mixed serve left B70_LOWBIT_PREFILL
unset, so the already-filed per-expert prefill host sync is still
that serve's prefill path, and the journal says so. git diff
--cached is empty. Campaign *.so files are ignored. The unstaged
vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so
is the pre-existing file from pass 1, not a new campaign binary.
Verdict: OK.
Suggested fix: none.

## Pass 4 -- 2026-10-20 08:55 UTC

Cursor: JOURNAL.md through the 08:53 G4 CLI entry. G2b, CPU only.
New since pass 3: E1 grouped serve, grouped-time, matmul probe,
matmul serve, prefill split; C7 three DP attempts; C10 and H1
links; G4 parking library and T2 launch; G5 CPU QSA. Fork tip
2fec939431.

### F23. E1 grouped int3 serve matches the JSON

Claim: JOURNAL 08:15. hotschmoe-dev first, alias ok, both texts
the int3 landmarks string. C1 ITL 11.29/11.49/11.61 ms, 90.40
tok/s, TTFT 810.34 ms, n=57. C2 p50 15.59 and 15.66, TTFT
1157.17 and 1156.72. Prefill 165.05/173.56/175.14 tok/s at TTFT
6162/47158/187062 ms. Same checkpoint as the per-expert int3
graph arm (99.33/112.32/115.78). Grouped is 1.55x at 8k. Decode
ITL matches that arm (11.29 ms). gpu-run exit 0.
Evidence: e1-grp-bench.json matches those fields (p50 11.289 and
11.288 on the graph arm). 173.564/112.320 = 1.545. Serve log:
same model path as e1-int3-graph-serve.log, FULL_DECODE_ONLY,
FLASH_ATTN, B70_LOWBIT_MOE_LAUNCHES=2, B70_LOWBIT_PREFILL=grouped,
B70_LOWBIT_SO=libb70_lowbit_xpu_e1.so, B70_XPU_CG_RECLAIM=500,
bits=3, shapes (256, 192, 1024) and (256, 48, 2048). e1.so is
1696504 bytes, mtime 06:35, and was not overwritten. gpu-run
done in 1207s exit 0. The graph arm's docker line does not set
B70_LOWBIT_SO. ops.py still defaults to libb70_lowbit_xpu.so,
mtime 04:05. No HEALTH line.
Verdict: OK for the printed rates, identity, and text. The two
decode ITLs are the same at 0.01 ms. They are not the same
library env.
Suggested fix: name both SO paths next to the 11.29 ms sentence.
Record xpu-health around the serve.

### F24. Grouped-time and matmul-probe logs match

Claim: JOURNAL 08:20 and 08:26. moe_prefill 63.14/96.05/324.10 ms.
Forty layers at 8192 would be 13.0 s, about 632 tok/s. Served
grouped prefill 47.2 s (174 tok/s). One expert wna16 1.118, wxa8
1.156, matmul 0.113, loop 15.06 ms. Matmul probe: e1mm.so
2345192 bytes, probe_rc=0, small linear rel_l2 0, 2.3e-7, 3.8e-6
for bits 2/3/4, large unpack and linear rel_l2 0, 72.659 vs
0.575 ms (126x). Small M=32 is 0.14 vs 0.23-0.30 ms. The 04:05
and 06:35 libraries were not overwritten.
Evidence: e1-grouped-time.log lines 40-46 match, including
tok/s_if_40layers=631.9 and GROUPED_TIME_OK, exit 0. 40*324.10 ms
= 12.96 s. Served 8k TTFT is 47158 ms (173.56 tok/s). 1.118/0.113
= 9.89. e1-matmul-probe.log: bit3 linear rel_l2=0, bit2
2.3351e-07, bit4 3.8139e-06. Large shape 72.659 and 0.575
(126.4x). Small unpack+matmul is 0.140-0.141 against wna16
0.231/0.266/0.299. probe_rc=0, MATMUL_PROBE_OK, exit 0.
e1-matmul-build.log: 2345192 bytes, cpu-run 65s exit 0. mtimes
of libb70_lowbit_xpu.so and e1.so are still 04:05 and 06:35.
No HEALTH line in either probe.
Verdict: OK. The small-linear list is not in bits 2, 3, 4 order.
Bit3 is the exact 0.
Suggested fix: write bit3=0, bit2=2.3e-7, bit4=3.8e-6.

### F25. C7 DP crashes: second and third match; the first log is gone

Claim: JOURNAL 08:22, 08:25, 08:28. First serve segfaulted in
allgatherv_ll_ring bf16 Rt64_128_PCIE from a one-element
dist.all_gather. all_reduce 6.460 and 6.118 ms, barrier 0.609
and 166.562 ms, load about 75 s, collective health 35s.
Second serve, fork 1fcc7bbd1d, all_gather_into_tensor, same
segfault inside PersistentDeviceCodeCache::getItemFromDisc.
all_reduce 0.625 and 0.935, barrier 163.434 and 0.701. Load 49 s.
Third serve, SYCL_CACHE_PERSISTENT=0, did not segfault.
is_padding size mismatch, expected 4096. pid 424 coll_ms
162.166, pid 423 coll_ms 1.542. Load 35 s. SERVE_RC 1. No
greedy, no ITL. placed 733, 11.64 GiB, shards=1 tensors=753
ep=0/2 and ep=1/2.
Evidence: c7-ep-dp-serve.log and c7-ep-dp-serve-nocache.log are
the same sha256 (ba16de95). c7-coll.log matches c7-coll-nocache.log.
c7-dp-window.out only contains LAUNCH 08:26:10, HEALTH_RC 0,
SERVE_RC 1, SCRIPT_EXIT 0. The strings 6.460, 6.118, 0.609, and
166.562 are not in the logs dir. c7-coll-health.txt is the
08:26 run: COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120
compiled_iterations=10 p2p=0, HEALTHY, done in 34s exit 0.
c7-coll-into.log: pid 422 all_reduce=2/0.625ms barrier=2/163.434ms,
pid 423 all_reduce=2/0.935ms barrier=2/0.701ms, tokens=2048, no
execute line. c7-ep-dp-serve-into.log: placed 733 skipped_mtp 20
q8_dequant 7, 11.64 GiB, load 48.728 and 49.027 s, two
all_gather_into_tensor deprecation warnings, two segfaults in
allgatherv_ll_ring<bfloat16, Rt64_128_PCIE> via
PersistentDeviceCodeCache::getItemFromDisc and
all_gather_single. c7-coll-nocache.log: pid 424 coll_ms=162.166
all_reduce=2/0.624ms barrier=2/161.543ms, pid 423 coll_ms=1.542.
Nocache serve: segfault count 0, RuntimeError is_padding size
mismatch expected 4096, in-container echo
SYCL_CACHE_PERSISTENT=0 and SYCL_CACHE_DIR=.../sycl-c7-dp
(the earlier -e PERSISTENT=1 is overridden). Load 34.653 and
35.047 s. topk.cpp:880 is the TORCH_CHECK against num_tokens.
No xe-reset line. No /v1/models body: the serve never became
ready. Fork 1fcc7bbd1d parent is 2a50cfe2a7, dated 08:22:17.
Verdict: second and third crashes OK. First-crash timings and
the one-element stack are UNVERIFIED because that log was
replaced. Pre-health for the first two windows is also gone.
Post-health was not run after the segfault.
Suggested fix: copy the serve log before the next truncation,
under a name that includes the attempt. Run xpu-health, and
xe-reset if that health fails, once the cards are free. The
08:25 entry already says the one-element list was not the cause.

### F26. Padding-mask commit matches the diff; py_compile has no log

Claim: JOURNAL 08:35. Fork 2fec939431, moe_runner.py only.
PY_COMPILE_OK. The local mask is an extra all_gatherv tensor,
installed for top-k, and restored after the layer. A shorter
mask is dropped for that forward.
Evidence: git show 2fec939431 is that file only, parent
1fcc7bbd1d, 08:35:37, not pushed. The diff matches the
description. all2all.py dispatch_router_logits extends
tensors_to_gather with extra_tensors and returns the tail.
No py_compile transcript. The run is unmeasured, as the entry
says. xpu_communicator.py:198 still says the one-element list
was the 08:19 segfault. The into log shows all_gather_into_tensor
faulted in the same kernel, and the fresh-cache run returned.
Verdict: commit contents OK. PY_COMPILE_OK is UNVERIFIED.
The comment overclaims a cause the next serve rejected.
Suggested fix: point the comment at the fresh-cache result.
Save the py_compile stdout.

### F27. C10 and H1 link hashes match

Claim: JOURNAL 08:30 C10 and 08:36 H1. C10 cpu-run 97s exit 0,
_C.c10.so 8294752 bytes sha 55c154ee..., _C.so sha still
086595a0..., 120 warnings, no errors. H1 first link failed in
48s (cannot find -lze_loader). Second cpu-run 159s exit 0.
b70_ep _C.so 391992 bytes sha b0f8376f..., NEEDED
libze_loader.so.1 and libsycl.so.9. _C.ep.so same bytes as
_C.c10.so. nm exports b70_moe_layer and b70_reserve_decode_queue.
_C.graph.so 6987192 bytes, mtime 06:59, sha 2315bd45....
Evidence: those sha256 values match. cmp of _C.c10.so and
_C.ep.so is identical. readelf NEEDED includes both libraries
plus libtorch. nm defines both symbols. c10-build.log: "120
warnings generated", error count 0, done in 97s exit 0.
h1-build.log: done in 159s exit 0. The 48s -lze_loader failure
is not in that log or the other h1 logs. _C.graph.so stat is
6987192 bytes at 06:59:04. *.so is gitignored. git diff --cached
is empty. The August push-allreduce .so is still the only
unstaged binary, same as pass 1.
Verdict: OK for the hashes and the successful links. The 48s
failure is UNVERIFIED.
Suggested fix: keep the failed link log under a second name.

### F28. G5 CPU QSA numbers reproduce; the agent log was not saved

Claim: JOURNAL 08:30 and reports/1G5_qsa.md. QSA_LAYER_OK.
Radix top-k matches lexsort through n=2100 width 2051. Park
slope 15264. rint 0.5->0, 1.5->2, -1.5->-2. Marginal block
Strata [4,5,6,7,12,13,14], vLLM expand [4,5,6,7]. Pooled keys
max_abs 1.19e-7. Text mRoPE equals 1D. Spare key ignores
pos_base. No GPU number.
Evidence: no g5 transcript under logs/. This pass re-ran
python3 -m flashnext_ref.test_qsa_layer with bytecode writes
off. Exit 0. The print is max_abs 1.1920928955078125e-07, park
bytes 15264, rint [0, 2, -2], and those two id lists, then
QSA_LAYER_OK. The report says the XPU plugin is not measured.
No QSA_XPU_OK line is in the journal.
Verdict: OK on the CPU contract. Strata file:line citations in
1G5_qsa.md were not re-opened this pass.
Suggested fix: keep the test stdout next to the journal entry.

### F29. G4 library, import, and the 29-test count

Claim: JOURNAL 08:31 and 08:46. 29 tests OK in 0.003s, including
ParkIndex.discard. IMPORT_OK. libb70_kvpark.so 90904 bytes,
compile rc=0 in 45s, ldd clean, schema alloc_host, copy_batch,
engine_ordinal. Container cap 80g, host pool 32 GiB, port
18111, four slots. F-A is c38d0a6197. C7 tip at 08:31 was
1fcc7bbd1d. F-B is the existing ext_tokens assign, with a DSV41
round-down. turn_test does not send extra flush prompts.
Evidence: this pass re-ran the three unittest modules: 29 tests,
OK, 0.001s. The 08:31 0.003s transcript is still absent.
g4-import.txt: IMPORT_OK, cpu-run 20s exit 0. g4-kvpark-build.txt:
compile rc=0, 90904 bytes, cpu-run 45s exit 0. Host copy is
90904 bytes. ldd reports 0 "not found". strings include
alloc_host(int nbytes), copy_batch, and engine_ordinal.
CPU_BYTES default 34359738368. run_t2.sh passes MEMORY=80g.
serve_ornith.sh still defaults MEMORY to 120g. The 08:49 serve
log shows memory=80g. turn_test.py runs A, salted control, B,
and C, and the comment says a fifth conversation would evict A.
scheduler.py:1064 subtracts ext_tokens % block_size when
prefix_replay_tokens is set. c38d0a6197 and 1fcc7bbd1d match
the fork log. The 08:46 verdict sentence says card 0; the next
line and the launch are card 1.
Verdict: OK. The 0.003s figure is the unsaved run. The card-0
sentence is the slip.
Suggested fix: say card 1 in that verdict sentence.

### F30. G4 T2 health matches; the serve then died on the flag

Claim: JOURNAL 08:49. xpu-health card 1 OK in 24s. The serve
found card 1 held by H1 and would exit rather than take it.
No T2 measurement.
Evidence: g4-t2-health-before-c1.txt: HEALTHY cards 1, done in
24s exit 0. g4-t2-serve-c1.txt starts with the busy wait on
vllm-fn-h1-m0, then the container starts (mem=80g, port 18111,
--park-finished-tail, hotschmoe-dev first in the argv). vLLM
then prints unrecognized arguments: --park-finished-tail and
gpu-run done in 27s exit 2. Fork CacheConfig.park_finished_tail
exists (cache.py:197) and the scheduler copies it
(scheduler.py:348). The process that ran did not accept the
CLI flag.
Verdict: WRONG as a description of the finished process. Health
is OK. There is still no T2 measurement. The process did take
card 1 after H1 and exited 2 before a model load.
Suggested fix: append the exit 2. Check that the container's
vllm import is the fork (vllm.__file__ and hasattr
CacheConfig.park_finished_tail) before the next T2. Do not
describe this launch as a yield.

### F31. E1 matmul serve matches; 3.3x is 3.26x

Claim: JOURNAL 08:45. Same int3 checkpoint, e1mm.so,
PREFILL=matmul. C1 ITL 11.34/11.55/11.63 ms, 89.94 tok/s,
TTFT 1875.76 ms, n=57. C2 p50 15.84 and 15.84, TTFT 2046.96
and 2046.12. Prefill 287.93/366.37/385.20 tok/s, TTFT
3532/22341/85051 ms. Texts are the landmarks string. Grouped
8k was 173.56, per-expert was 112.32. 3.3x the original 8k
and 13x short of 5000. One q-proj was 0.575 ms, so 22 s remains.
gpu-run exit 0.
Evidence: e1-mm-bench.json and e1-mm-serve.log. Identity first
id hotschmoe-dev, alias qwen36-w3a16-matmul. Both texts match.
C1 p50/p90/p99 11.342/11.554/11.631, tok/s 89.943, TTFT
1875.763, n=57. C2 and prefill rows match the printed digits.
Serve env is e1mm.so, PREFILL=matmul, LAUNCHES=2, RECLAIM=500,
FULL_DECODE_ONLY, FLASH_ATTN, same checkpoint path. bits=3.
gpu-run done in 1022s exit 0. 366.368/112.320 = 3.26.
5000/366.37 = 13.6. 8k TTFT 22341 ms. moe.py _prefill_matmul
counts.tolist() is the prefill histogram read. Decode stays
on gemv when T<=8, and this serve's graph mode is
FULL_DECODE_ONLY, so that host read is outside the capture.
No HEALTH line.
Verdict: OK. The 3.3x and 13x sentences are rounded.
Suggested fix: quote 3.26x and 13.6x, or show the division.
Record xpu-health.

### F32. E1 8k layer split matches the log

Claim: JOURNAL 08:48. split_rc=0. w13 unpack 12.93 ms. Expert
layer 144.64 ms, times 40 is 5.79 s. Dense GEMM 0.655 ms, times
160 is 0.10 s. moe_gemv 0.168 ms (6.72 ms times 40). gemv 0.098
ms (15.76 ms times 160). Served 8k prefill was 22.3 s. Kernels
account for about 5.9 s.
Evidence: e1-prefill-split.log lines 39-46 are those strings,
gpu-run done in 24s exit 0, split_rc=0. 5.79+0.10 = 5.89.
22341 ms is the F31 TTFT. No HEALTH line. The docker line is
the E1 prefill-split container.
Verdict: OK. This is a random-plane microbench, which the
entry already says.
Suggested fix: none for the numbers. Print xpu-health on the
next GPU probe.

### F33. H1 M0 lost the device; that result is not in the journal yet

Claim: JOURNAL 08:36 only says the window was requested. No
success number was claimed.
Evidence: h1-m0-window.out records xpu-health HEALTHY cards
0 and 1, done in 28s exit 0, before the probe. h1-m0-probe.log:
PROBE5 zeMemOpenIpcHandle failed (2013265924), same_ctx=1,
ipc_store=0, same_store=1, path=2, spin=200000, then
UR_RESULT_ERROR_DEVICE_LOST (20) in m0_prep. h1-m0-health-before.txt
and h1-m0-health-after.txt are 0 bytes. probe.cpp m0_prep fills
the remote region with a parallel_for and then queue.wait.
Those waits are before begin_recording. m0_card1_replay does
not wait. kreg.hpp b70_device_slot stores two devices in
unlocked function-local statics and returns slot 0 when a third
device appears. b70_device_grow still leaks the old block and
throws if the queue is recording.
Verdict: UNVERIFIED as a journal claim, because none was written.
The raw probe failed and post-health is empty.
Suggested fix: append the DEVICE_LOST line. Run xpu-health and,
if it fails, bin/xe-reset before another two-card probe. Do not
treat the scalar same-context store as proof the parallel_for
poison fill is safe.

### F34. Repo hygiene

Claim: none new. Checked again.
Evidence: git diff --cached is empty. vllm/flashnext/.gitignore
line 2 ignores *.so, and check-ignore matches the new _C.c10.so,
_C.ep.so, and b70_ep _C.so. No secret-like names in git status.
No apt or driver change is in these serve logs. They mount host
libze 1.28.2 and IGC 2.36.3. Fork status is one dirty file,
vllm/models/qwen4_exp/xpu/model.py, which the C7 and G4 entries
say they left untouched. This pass did not review that diff.
Verdict: OK.
Suggested fix: none.

### F35. The 08:53 entry corrects F30

Claim: JOURNAL 08:53. vllm serve exited in 27s on unrecognized
--park-finished-tail. The wheel lacks the field. Fork
e442b4b53f registers the flag and passes it into CacheConfig.
Default off. Not pushed. qwen4_exp model.py stayed unstaged.
Evidence: g4-t2-serve-c1.txt is the same exit 2 and the same
error string as F30, gpu-run 27s. git show e442b4b53f is
arg_utils.py only, parent 2fec939431, 08:53:41, five lines:
the field defaults to CacheConfig.park_finished_tail (False in
cache.py), add_argument("--park-finished-tail"), and the
CacheConfig constructor keyword. Branch is ahead 14 of
origin/main. model.py is still the only dirty fork file.
Verdict: OK. This supersedes the "still waiting" sentence in
the 08:49 entry. The relaunch itself is not measured yet.
Suggested fix: none for the writeup. The next T2 log should
show the flag accepted before any latency number.

## Pass 5 -- 2026-10-20 09:35 UTC

Cursor: journal through the 09:35 orchestrator entry (file was 2300
lines). Pass 4 (F23-F35) is unchanged. CPU only. No GPU, no commit,
no other agents' files. Wall clock at the start of the write was
09:34 UTC; the 09:35 orchestrator heading was already in the file.

### F36 -- C7 DP=2 eager bench, identity, and coherence

Claim (09:01): fork e442b4b53f, eager DP=2 TP=1 EP, max_model_len
2048, max_num_seqs 2, UD-IQ3_XXS, no P2P, no park flag. HEALTH_RC 0.
placed 733, skipped_mtp 20, 11.64 GiB, linear EP 128/256. READY
09:00:32. Both greedy texts are the C4 Paris string, match_c4 true.
Identity hotschmoe-dev then qwen3.6-35b-a3b-ud-iq3-xxs. C1 ITL p50
121.40 / p90 143.80 / p99 153.09 ms, TTFT 462.42 ms, 7.86 tok/s,
n=32. C2 p50 120.73 and 120.79 ms, TTFT 523.45 and 522.66 ms. These
times are not the one-card FULL_DECODE_ONLY 11.11 ms. SERVE_RC 0,
docker stop 09:00:52. No crash, no xe-reset. Post-health was not run.

Evidence: c7-bench-ep-dp.json and c7-ep-dp-greedy.json. first_ok and
alias_ok true, ids in that order, mtp enabled false, port is in the
serve argv as 18104. C1 measured TTFT 462.4195, p50 121.4036, p90
143.7986, p99 153.0859, decode 7.8599 tok/s, n_chunks 32. C2 streams
TTFT 523.448 and 522.664, p50 120.729 and 120.788. Both greedy texts
are exactly " Paris.\nThe capital of France is Paris.\nThe capital of France is",
errors [], match_c4 true against that same ref. That is the short
capital string, not the int3 landmarks string. c7-ep-dp-serve-ok.log
startup args: enforce_eager, data_parallel_size 2, tensor_parallel_size
1, enable_expert_parallel, served_model_name hotschmoe-dev first,
GGUF path UD-IQ3_XXS, load 11.64 GiB, placed 733 skipped_mtp 20,
"Local/global number of experts: 128/256", linear placement. READY
and stop 09:00:52 are in c7-dp-window-ok.out. SERVE_RC 0. BENCH_RC 0.
Collective health file: COLLECTIVE_HEALTH_OK p2p=0, xpu-collective-health
HEALTHY, gpu-run 27s exit 0. The serve argv does not pass
--park-finished-tail. PYTHONPATH starts with the fork tree
/mnt/vm_8tb/b70/build/vllm-flashnext/vllm. SYCL_CACHE_PERSISTENT is
set twice on the docker line, first 1 then 0; the later value is the
journal's 0, and SYCL_CACHE_DIR is sycl-c7-dp. CCL_TOPO_P2P_ACCESS=1
is not in the serve argv. 11.11 ms is the printed C1 p50 in
c7-bench-graph.txt (90.16 tok/s, n=64), a different mode. The 08:58
heading still says fork 2fec939431; the 09:01 heading and the tip
are e442b4b53f. No post-health line and no xe-reset line in the
window. The journal says that gap outright.

Verdict: OK. Identity and coherence ran. The 11.11 ms sentence is a
disclaimer, and that number is in the earlier graph bench text.

Suggested fix: none for the numbers. Keep the short Paris string
labeled as the C4 ref that was actually compared.

### F37 -- C7 collective medians

Claim: steady tokens=1 execute, 159 steps, both pids: 160 collectives,
120 all_gather_into_tensor median 15.01 ms, 40 reduce_scatter_tensor
median 5.62 ms, coll median 20.61 ms, wall median 119.06 ms. One
pid-423 warmup coll 643.27 ms. No all_reduce on the decode step.
40 MoE layers times 3 all-gathers plus 1 reduce-scatter.

Evidence: recomputed from c7-coll-ok.log. 160 lines match
tag=execute tokens=1 with the into_tensor names. Dropping the single
coll_ms 643.273 line leaves 159. Pooled medians: wall 119.057, coll
20.607, all-gather 15.011, reduce-scatter 5.624. Every one of those
lines has coll=160, all_gather=120, reduce_scatter=40. Zero
tokens=1 execute lines contain all_reduce. 40*(3+1)=160. c7-coll-ok.log
and c7-coll.log are the same bytes (64604). The -ok serve and window
copies match the unsuffixed files too.

Verdict: OK.

Suggested fix: none.

### F38 -- G4 CLI parse transcript

Claim (09:02): PARK_ON True, PARK_OFF False, CLI_PARSE_OK, cpu-run
exit 0 in 21s. Image vllm-flashnext-xpu:py312, VLLM_TARGET_DEVICE=cpu,
ONEAPI_DEVICE_SELECTOR=opencl:cpu, memory 24g. MemAvailable about
76 GiB. No T2 number.

Evidence: logs/g4-cli-parse.txt is 428 bytes. It records the docker
argv (network none, cpus 4, memory 24g, those two env vars,
entrypoint /usr/bin/python3.12, image vllm-flashnext-xpu:py312) and
"cpu-run: [g4] done in 18s (exit 0)". The python stdout strings
PARK_ON, PARK_OFF, and CLI_PARSE_OK are not in the file. The argv
ends in `python3.12 -`, not `g4_cli_parse.py`. The script at
logs/g4_cli_parse.py would print those three lines if it ran.
MemAvailable is not in the file. No T2 bench was added for this entry.

Verdict: WRONG on the 21s duration (the transcript says 18s).
UNVERIFIED on PARK_ON / PARK_OFF / CLI_PARSE_OK and on 76 GiB.
Exit 0 is in the transcript.

Suggested fix: tee the python stdout into g4-cli-parse.txt and quote
the printed lines. Re-run is CPU-only.

### F39 -- H1 first M0 window, now journaled

Claim (09:05): health before HEALTHY cards 0 and 1 in 28s. zeMemOpenIpcHandle
failed 2013265924 = ZE_RESULT_ERROR_INVALID_ARGUMENT. same_ctx=1,
ipc_store=0, same_store=1, path=2, peer atomics 0, bdf 0000:0b:00.0
and 0000:44:00.0. m0_prep raised UR_RESULT_ERROR_DEVICE_LOST (20)
after a range fill that includes 153600 words of the return buffer.
gpu-run exit 1 in 81s at 08:50:34. Health after HEALTHY in 30s inside
h1-m0-window.out. The tee file is empty. No xe-reset.

Evidence: h1-m0-window.out lines match: health before "done in 28s
(exit 0)" and "HEALTHY (cards 0 1)"; PROBE5 line with 2013265924 and
the same same_ctx / path / bdf / atomics fields; RuntimeError 20
UR_RESULT_ERROR_DEVICE_LOST; "done in 81s (exit 1)"; "H1 M0 probe
rc=1 2026-10-20 08:50:34"; health after "done in 30s (exit 0)" and
HEALTHY. h1-m0-health-before.txt and h1-m0-health-after.txt are 0
bytes. level-zero ze_api.h in the flashnext SDK defines
ZE_RESULT_ERROR_INVALID_ARGUMENT = 0x78000004, which is 2013265924.
probe.cpp static_asserts kRetB == 614400; 614400/4 = 153600. The
failure log does not print 153600; that count is the source shape.
No xe-reset string in the window. This is the run F33 saw before it
had a journal entry.

Verdict: OK. The empty tee files are still empty; the HEALTHY lines
are in the window log.

Suggested fix: tee health to a non-empty file. The next binary's
result is F42, not a claim that this run passed.

### F40 -- C7 TP=2 lock timeout

Claim (09:07): HEALTH_RC 0 at 09:06:02. Serve at 09:06:30 found card 0
held by C9 pid 3152937. gpu-run waited 30s and exited 124. vLLM did
not start. RAISE 0. SERVE_RC 124 is the lock timeout. No xe-reset.
Card 1 was free again by 09:07.

Evidence: c7-tp-window-race.out is the whole story: HEALTH_RC:0 under
"collective health 09:06:02", SERVE_RC:124, RAISE:0, WINDOW_DONE.
c7-coll-health-tp.txt: COLLECTIVE_HEALTH_OK p2p=0, HEALTHY, gpu-run
28s exit 0. c7-ep-tp-serve-race.log: "card 0 busy (held by: C9
pid=3152937 since=2026-10-20 09:06:30" and "timed out after 30s on
card 0". No vLLM log lines. The two TP window files match, and the
two serve files match. Nothing in those files says card 1 was free.

Verdict: OK on the timeout, the pid, RAISE 0, and "not a sharding
refusal". UNVERIFIED on the card-1 sentence.

Suggested fix: print both owner files when a two-card lease times out.

### F41 -- H1b rebuilt sidecar hash

Claim (09:09): b70_ep_xpu/_C.so sha256
2248124379c09191a4c77aaff2854b49ab59b6e743a90f2f802eae1e70189fe2,
445392 bytes. _C.ep.so unchanged. cpu-run exit 0 in 40s. Window was
only requested here; card 0 was busy at 09:08.

Evidence: sha256 of
plugins/b70_ep_xpu/b70_ep_xpu/_C.so is that digest, 445392 bytes,
mtime 09:06:39 UTC. _C.ep.so and _C.c10.so are still
55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0,
8294752 bytes. _C.c10.so mtime 08:28:33 UTC. h1b-build.log: "done in
40s (exit 0)" after waiting on D3's CPU lock. git check-ignore matches
vllm/flashnext/.gitignore line `*.so` for all three. h1b-m0-window.out
opens with card 0 busy (C9). The 09:09 heading does not claim a
passing oracle; that claim is the 09:35 entry (F42).

Verdict: OK.

Suggested fix: none.

### F42 -- M0 64/65 transport, and the "only gap" sentence

Claim (09:35 orchestrator): one-way flag 1.82/1.77 us p50 (0 to 1 /
1 to 0); push 4 KB 0.83 us, 64 KB 5.83 us (11.2 GB/s, near PCIe 3.0
x16); preflight to 614 KB ok; eager oracle 64/65 (first-use sample
only: stale push, spin timeout); graph samples pass; health clean
after. Verdict sentence: transport beats Strata's own 3-5 us
handshake, and first-use warm-up is the only M0 gap.

Evidence: h1b-m0-probe.log. PROBE1 p50 1.823 us (0to1) and 1.771 us
(1to0), n=200. PROBE2 bytes=4096 groups=256 us_p50=0.833 (groups=4,
16, and 64 are the same 0.833; groups=1 is 1.146). bytes=65536
groups=16 us_p50=5.833 GBps=11.235 (groups=256 is 5.834 / 11.233;
groups=1 is 6.979 / 9.390). PREFLIGHT pass, including dir=1to0
bytes=614400 ok=1. preflight_push in probe.cpp compares only the
first and last word after a host memcpy wait. EAGER fails=1: sample 0
first_use=1 push_bad=4380 err1=11, samples 1 through 64 push_bad=0.
That is 64/65. GRAPH fails=0 on the replay counter. The capture check
printed GRAPH capture_executed=0 after a sample with push_bad=4380,
push_guard_bad=1024, ret_bad=153600, ret_guard_bad=1024. The probe
then replayed from sample 0 and those replays were clean. MIRROR
fails=3, labeled "not a gate" in m0_probe.py: samples 1, 2, and 3
with first_use=0 and push_bad 1746, 1536, and 1514. The process
printed "FAIL eager=1 graph=0" and h1b-m0-window.out says gpu-run
done in 14s (exit 1). Health after is in that same window: xpu-health
HEALTHY (cards 0 1), gpu-run done in 33s (exit 0). h1b-m0-health-after.txt
is 0 bytes. Health before in the window is HEALTHY, done in 26s.
The 3-5 us figure is the design note in reports/2O1_ep_design.md
("device-side flag (3-5 us)", and "about 35 us per handshake versus
about 3-5 us"). It is not a Strata timestamp in h1b-m0-probe.log.
PCIe width is not printed. 65536/5.833e-6 = 11.23e9 B/s, matching
the log's GBps field.

Verdict: OK on the flag times, the 4 KB and 64 KB plateau times,
11.2 GB/s, preflight ok=1, eager 64/65, GRAPH fails=0, and health
after inside the window log. WRONG that first-use is the only M0
gap: the same log has three mirror misses on non-first-use samples,
and the capture check itself was dirty (capture_executed=0).
UNVERIFIED that 1.82 us beats a measured Strata handshake from this
host; 3-5 us is the design doc's stated range. The exit code is 1,
which the orchestrator does not mention.

Suggested fix: put MIRROR fails and capture_executed in the journal
line. Make preflight compare the whole buffer, or a checksum, before
calling 614 KB ok. Keep the health tee non-empty.

### F43 -- C10b dequant prefill bench and the wired policy

Claim (09:16): gpu-run exit 0 in 43s, python quiet 12s, HEALTH_BEFORE
0, HEALTH_AFTER 0. .so not rebuilt (sha 55c154ee, 08:28). Dequant 0
mismatches, 13 types. dense mm_max_abs=0, bf16w ulps bad=0,
vs_f32_max_rel 23.5. MoE bmm and loop max abs 0 and within 8 ulps of
bf16. Q4/IQ4 K=2560 dq wins from M=512 to 12288 (M=12288: 7.93 ms /
97.5 TFLOPS vs fused 30.6 ms / 25.3). K=640 fused wins at M=512
(238 vs 356 us wall); dq wins at 4096 and 12288. Experts: fused
T=512 10.82 ms and T=4096 26.76 ms; bmm T=12288 66.97 vs 74.34 ms.
Dispatch: dense M>=4096 or (M>=512 and K>640); moe T>=12288; grouped
GEMM stays fused. GEMM-only budget T=12288 is 4256 ms, 2887 tok/s,
vs C8 7265 ms / 1691. 1.89 s of the 6.14 s 2000-tok/s budget is left.
End-to-end 2000 tok/s is not claimed. CORRECT_AND_WIRED.

Evidence: c10b-dq-bench.log. gpu-run done in 43s exit 0. cpu-run quiet
done in 12s exit 0. HEALTH_BEFORE:0 and HEALTH_AFTER:0, each next to
"xpu-health: HEALTHY (cards 0)". Card 0 only. At acquire, card 0 was
held by H1 pid 3158233 running xpu-health (09:09:46), so the wait was
behind H1's health probe. The 09:08 heading's "card 0 was C9 at 09:06"
matches the TP window, which is earlier. Thirteen dequant type names
with bf16_mismatches=0. dense_dq Q4_K line: mm_ulps_bad=0,
mm_max_abs=0, bf16w_ulps_bad=0, vs_f32_max_rel=2.353e+01,
vs_f32_max_abs=7.085e+01, max_abs_vs_bf16w=6.337e+01. 2.353e+01 is
23.5. moe bmm_vs_loop max_abs=0 and bf16_ulps_bad=0; vs_f32_max_rel
1.777e+01. M=12288 K=2560 Q4_K event times: dq 7928.2 us / 97.51 TF,
fused 30581.1 us / 25.28 TF. K=640 M=512 wall: fused 238.1 vs dq
355.5. K=640 dq wall is faster at M=4096 (463.0 vs 682.6) and
M=12288 (971.1 vs 1808.9). Expert event ms: fused 10.820 and 26.761;
at T=12288 bmm 66.969 vs fused 74.337. prefill_dq.py constants:
DQ_M_MIN 512, DQ_M_SHORT 4096, DQ_K_SHORT 640, DQ_T_MIN 12288,
DQ_GROUP_M_MIN 10**9. dense_wants_dq and moe_wants_dq match the
journal, and prefill.py calls them. The 8 ulp bound is the bench's
bf16_ulps_bad counter, which is 0 on those lines; the log does not
print the raw ulp distance. Report 1C10_prefill_dq.md GEMM sum at
T=12288 is 4255.71 ms and 2887 tok/s. 12288/4.25571 = 2887.
12288/2.000 = 6.144 s, and 6.144-4.256 = 1.888 s. The routed-expert
piece 3214.53 ms is 48 * 66.969 ms. The head at M=12288 in that sum
is marked projected in the report (measured head stops at M=4096).
1C8_prefill3.md table: GEMM sum 7265.0 ms and 1691 tok/s. No
/v1/models call and no coherence in this log. _C.c10.so hash is the
F41 hash, mtime 08:28:33.

Verdict: OK on the measured bench lines, the health return codes,
the dispatch source, and the arithmetic of 4256 ms / 2887 tok/s /
1.89 s. The 4256 ms figure is the report's projected sum, and the
journal does not say the head term is projected. It does say
end-to-end 2000 tok/s is not claimed. CORRECT_AND_WIRED means the
plugin policy, not a served model.

Suggested fix: label the 4256 ms column as projected when it is
quoted next to a tok/s number. Keep max_num_batched_tokens=12288 as
a recommendation until a serve measures it.

### F44 -- G5 QSA XPU run transcript is missing

Claim (09:10): libb70_qsa_xpu.so 369040 bytes, built 09:07. An earlier
sycl::half scale path exited 1 in 23s. Device scale bits 1015215198
and 1018736225; CPU float16 bits 1015218176 and 1018732544;
k_codes[0,0,0,20] was -103 vs -102. Health after that fail HEALTHY.
Rebuild uses f32_to_f16/f16_to_f32 RTNE. Rerun: QSA_XPU_OK, gpu-run
16s exit 0, xpu-health card 1 HEALTHY. Graph replay pos 4/5/7 bitwise.
Residuals head_dim 256: pooled 2.384e-07, scores 9.537e-07, attn
7.153e-07, gated 2.441e-04. Not a latency number. Hook not served.

Evidence: the .so is 369040 bytes, mtime 09:07:16 UTC, sha256
72e68a7181222410bd6a0e702a96fb48142551ea21ba2990959affc30880ccc0.
qsa_kernels.cpp f32_to_f16 is an explicit round-to-nearest-even bit
path, and the scale store uses f16_to_f32(f32_to_f16(amax/127)).
No synchronize, .item, or .cpu in that csrc file. test_qsa_xpu.py
synchronizes after the `with torch.xpu.graph` block and after
replay, not inside the capture. The string QSA_XPU_OK is not in
b70_qsa_xpu/ or in logs/. run_gpu.sh prints to its own stdout and
does not tee a file. build_qsa.err is 0 bytes. Fork model.py is
still the only dirty fork file (uncommitted hook).

Verdict: OK on the library size, mtime, and the RTNE source. 
UNVERIFIED on QSA_XPU_OK, the 16s/23s exits, the bit patterns, the
residuals, and both HEALTHY lines. Those numbers exist in
reports/1G5_qsa.md and the journal, which repeat each other, and
not in a saved run log.

Suggested fix: redirect run_gpu.sh stdout to
logs/g5-qsa-xpu.txt on the next card-1 run.

### F45 -- D3 fused GDN microbench, and the first-serve cause

Claim (09:23): SSM fp32 [32,128,128] = 2 MiB, conv fp16 [3,8192] =
48 KiB, T=1 traffic 4.40 MB, roof 7.24 us. Stock conv+delta graph
0.0640 ms, fused 0.0411 ms. Core max_abs 7.6e-6, cosine 1. Post-norm
max_abs 3.9e-3 is one fp16 ulp at magnitude 4. T=3 fused 0.0144 vs
stock 0.0170 ms. T=3 conv-state vs stock cosine 0.15 because stock
does not write conv state when num_prefills==0 and T>1. Spec T=3
fused 0.0161 vs stock 0.0178, core cosine 1. FA auto split-k at
S=200000 is 1.13 ms (363 GB/s); custom paged kernel cosine 1 and
about 18x slower. B70_D3_ATTN stays off. First serve fell back
because pure decode sets has_initial_state to None. Second serve
skipped on the padded int8 page view. Both torn down before bench.
30-layer fusion upper bound about 0.69 ms on the 11.4 ms step.
No end-to-end number. Claude model calls exited 1 and were not retried.

Evidence: d3-probe-gdn.json. traffic 4399232 B = 4.399 MB, roof_us
7.24, conv_state 49152, ssm 2097152. T=1 stock 0.0640 ms, fused
0.0411 ms, core max_abs 7.6294e-06, cos 1.0, out max_abs 3.9062e-03.
T=3 stock 0.0170, fused 0.0144, conv cosine 0.152118. Spec stock
0.0178, fused 0.0161, core cos 1.0. (0.0640-0.0411)*30 = 0.687.
11.4 ms is D1's earlier graph-replay figure (JOURNAL 05:17, report
1D2), not a time in this probe. d3-probe.json fa_serve S=200000
block=64 T=1 is 1.1269 ms and 363.5 GB/s. Custom splits=32 on that
row is 20.4631 ms, which is 18.2x. splits=1 is 655.8 ms. Cosine of
ours_vs_fa is 1.000000. Torch line in the probe: 2.14.0+xpu, vLLM
0.32.1rc1.dev94+g5fd62688f. No health line in the probe json.
d3-serve-gdn.log traceback is AttributeError: 'QwenGatedDeltaNetAttention'
object has no attribute 'kv_cache', at d3_gdn.py line 106
`if self.kv_cache is None`, then "b70_d3 gdn skip: decode metadata
missing" during capture. The current source uses getattr and, when
has_initial_state is None, substitutes a ones buffer; "decode
metadata missing" is now printed only when cu or slot is None. The
saved traceback is the kv_cache attribute, not a has_initial_state
None branch. d3-serve-gdn2.log prints "b70_d3 gdn skip layout
conv=(4, 3, 8192) ssm=(4, 32, 128, 128) ...". The non-contiguous
page explanation is the comment now in gdn_decode.cpp, which checks
stride(-1)==1 and packed inner strides and does not require the
slot stride to be contiguous. Both of those serves reached
"Application startup complete" and gpu-run exit 0 (420s and 209s).
No bench JSON is timestamped between them. I did not re-check the
Claude CLI exit; the journal is the only record of that.

Verdict: OK on the microbench numbers, the 0.69 ms arithmetic, the
11.4 ms being the prior D1 replay, the FA 1.13 ms / 363 GB/s row,
and the second serve's layout skip. WRONG on the stated cause of
the first-serve fallback: the log's exception is a missing kv_cache
attribute. Health before and after the probe was not logged.

Suggested fix: quote the traceback line in the journal. The current
getattr path is already a different source than the one that threw.

### F46 -- D3 third serve bench is real and unjournaled

Claim: the 09:23 entry says both serves were torn down before bench
and that there is no end-to-end number. No later heading through
09:35 claims a D3 latency number.

Evidence: logs/d3-bench-gdn.json and d3-bench-gdn.log, mtime 09:26,
after the 09:23 heading and before the 09:35 heading. tag
d3-gdn-fused, base http://127.0.0.1:18081. Identity ids
hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2,
first_ok and alias_ok true. Coherence ok, same, mentions_paris,
both texts the short Paris string, verdicts OK OK. C1 ITL p50
12.603 ms, p90 12.833, p99 13.071, 79.39 tok/s, TTFT 94.75 ms,
n=64. C2 p50 13.502 and 13.484 ms, TTFT 179.68 and 186.46 ms.
Prefill tok/s 6075.8 / 7882.7 / 6415.2 at built prompts 1017 / 8185
/ 32762. cpu-run quiet done in 11s exit 0. d3-serve-gdn3.log is the
serve alive at that time: Ornith GPTQ int4, card 1, port 18081,
FULL_DECODE_ONLY, FLASH_ATTN, max_model_len 33792, mtp 0. It logs
"b70_d3 gdn fused call" with capturing=0 and capturing=1, then
gpu-run done in 244s exit 0. No xpu-health HEALTHY line in that
serve log. This is not the IQ3 GGUF DP run and not the 11.11 ms or
11.4 ms baseline. d3-serve-shisa.log (09:28) is also unjournaled;
it only shows "d2_shisa_applied copied=16 skipped=2" in the lines
searched. d3-gdn-build2.log: compile rc=0, libb70_gdn_xpu.so 322256
bytes, cpu-run 44s exit 0, at 09:23, which matches the journal's
"the .so rebuilt" sentence and is not itself a latency claim.

Verdict: the 09:23 "no end-to-end" sentence was true of the first
two serves. The 09:26 bench is a separate result with identity and
coherence in the JSON. UNVERIFIED as a journal claim because no
heading has owned the numbers yet. Do not treat 12.60 ms as a match
against 11.4 ms or against C7's 121 ms.

Suggested fix: a D3 heading that cites d3-bench-gdn.json, says the
fused op ran during capture, and records health before and after
the card-1 serve.

### F47 -- code read this pass, and repo hygiene

Claim: new plugin and fork code should not hide a race, a bad stride,
a missing bound, or a host sync inside a captured op. The repo should
not gain secrets or staged binaries. Host UMD was not changed.

Evidence: lowbit_kernels.cpp GateUpRowsKernel and DownRowsKernel
replace `if (row >= limit) continue;` with `if (row < limit) { body }`.
The bodies are the same stores and the same index math. The file
header still says no queue.wait, and the diff does not add one.
moe.py _group_slots still does active.max().item() to size the bmm
pad. That is a host sync on the prefill grouping path. fwd_split.py
installs only when B70_LOWBIT_FWD_SPLIT=1 and wraps synchronize for
a prefill log; its header says decode steps are not synchronized
there. qsa capture has no sync inside the graph block (F44).
gdn_decode.cpp recordStream's the inputs, refuses a norm cast with
TORCH_CHECK so capture cannot malloc, and uses slot.stride(0) only
for the spec slot. ep preflight .wait() and host memcpy run before
the eager/graph oracle, not inside m0_submit_card0. The preflight
pass bit is still endpoints only (F42). git diff --cached is empty.
git status still shows the pre-existing modified
vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so
and no secret-like names. New *.so files under flashnext plugins are
ignored by `*.so`. Untracked vllm/int4/guard_candidate is 16K and
vllm/w8a8/ornith15_s2b is 44K. Fork tip is still e442b4b53f, ahead
14, only dirty file vllm/models/qwen4_exp/xpu/model.py. h1 and h1b
probe logs still resolve host libze_loader.so.1.28.2. No journal
heading in this window claims an apt or driver change.

Verdict: OK on hygiene and on the lowbit brace edit. No new host
sync inside a captured op in the files read this pass. The open
correctness gap is the M0 mirror misses in F42, not a stride bug
in the GDN page view.

Suggested fix: none for the lowbit brace change. Keep
active.max().item() off any graph-captured path.

## Pass 6 -- 2026-10-20 10:16 UTC

Cursor at write: JOURNAL.md still ended at the 10:05 C11 heading
(line 2611). Entries below that time are owned by headings 09:42
through 10:05. F46's unjournaled D3 bench is now the 09:50 D3
heading. F42's mirror misses are now disclosed by the 09:51 H1c
heading. G4 CLI stdout and G5 QSA_XPU_OK are still absent.

### F48 -- H1c M0 warmup, 65/65, mirror disclosed

Claim: the 09:38 rebuild plus the 09:51 heading. _C.so sha256
bd4f4aa010c718da, 476912 bytes, 09:38. _C.ep.so still 55c154ee,
8294752 bytes, 08:35. Init warmup clears the scored first-use miss.
Eager 65/65 and graph 65/65. Mirror fails=3 is not a gate. Flag
p50 is 1.771 us (heading "1.8 us"). Health before and after
HEALTHY, no xe-reset.

Evidence: logs/h1c-m0-probe.log and h1c-m0-window.out. PREFLIGHT
64/4096/17520/65536 and 614400 ok. WARMUP touch inbox=1 parts=1
flag1=1 flag0=1, exchange attempt=1 err0=0 err1=0, reset ok.
Eager s=0 first_use push_bad=0, EAGER fails=0. Capture s=0
capture_executed=0 push_bad=4380 ret_bad=153600, replay push_bad=0,
GRAPH fails=0. MIRROR fails=3, push_bad 1700/1596/1566, err=0.
PROBE5 zeMemOpenIpcHandle 2013265924 (ZE_RESULT_ERROR_INVALID_ARGUMENT
0x78000004), same_ctx=1 path=2, bdf 0000:0b:00.0 and 0000:44:00.0.
PROBE1 both ways n=200 min 1.667 p50 1.771 max 1.980. PROBE2
65536 groups=64 us_p50 5.834 GBps 11.233. PROBE3 card0 5.834 card1
5.833 host wall 26.631. PROBE4 plain_bad=0 uncached_bad=0. PASS
eager 65/65 graph 65/65. PROBE_RC=0 at 09:49:45. Health before rc=0
at 09:46:03, health after rc=0 at 09:50:40. No xe-reset in the window.

Verdict: OK. The heading is honest about the mirror arm and the
capture poison. "1.8 us" is the rounded 1.771 p50.

Suggested fix: none for the scored gate. Keep the mirror arm out of
the pass line until those three samples are explained.

### F49 -- C9 grouped schedule table

Claim: the 09:42 C9 heading. Strata whole-layer and plugin graph
times, cosine, eager A/B, gates, and "1.65x to 2.04x slower than
R2" / "1.23x to 1.46x" / "up to 1.63x eager".

Evidence: c9/fair_all5.log, fair_ctrl.log, c5_anchor.log,
ours6_sched1.log, ours6_sched0.log, run_ours7.outer.log (44s exit 0),
health_ours_pre.log and health_ours_post.log HEALTHY card 0 rc 0.
Control 60.14 / 105.06 / R2 92.39. C5 anchor R2 92.55, IQR2 69.69,
gu 60.03, all 103.24. Fair rows match the journal table, including
r2_iqr2_diff=0. Plugin sched=1 and sched=0 us_graph lists match.
Eager re-read this continuation: sched1 moe_layer a2_us T=1/2/4/8
230.7/289.9/492.7/877.3, sched0 234.7/397.6/735.6/1426.8, both
VERDICT failures=0. 1426.8/877.3 = 1.626. Gates: gpu_check.log
VERDICT failures=0, moe_layer_bench.log VERDICT failures=0,
compare_cpu.log VERDICT failures=0 types=13/13 and mismatches=0 on
every OK row, graph_probe.log GRAPH_PROBE_OK. No /v1/models and no
35B serve in this stage. Library hashes match the heading
(8a76f559 and 086595a0, both 7574840 bytes).

The 1.65x-2.04x sentence divides plugin XPUGraph medians by Strata
part-sums (186.32/112.89 = 1.651, 534.72/261.64 = 2.044). Those are
different clocks: three waited eager parts versus one graph of 48
calls. The arithmetic is right. The printed 5e-2 and 8e-2 tolerances
are not in the gate logs. moe_layer_bench prints OK with max_rel up
to 2.2e2, so that file did not enforce an 8e-2 relative cap.
UD-IQ3_XXS "753 tensors, shared Q6_K 120, Q8_0 3" is stated in
reports/1C9_moe_sched.md as a header scan. This pass did not open
the GGUF.

Verdict: OK on the times, the failures=0 lines, and the speedup
arithmetic. UNVERIFIED the tensor census and the 5e-2/8e-2 constants.

Suggested fix: cite the tolerance from the checker source next to
the log, and attach the GGUF header count command to the census
sentence. Keep the Strata comparison labeled as part-sum versus graph.

### F50 -- E2 layer probes, 34 ms, GEMV mismatch

Claim: the 09:42 and 09:51 E2 headings. Clean bmm 39.28 / 54.44 ms,
layout and fast unpack, one layer 34.22 ms, rel about 1.8e-4, GEMV
rows=2 max_abs 1.0, fast probe exited 1. The 22 s reconstruction is
4*40*140 ms. Not a serve tok/s.

Evidence: e2-bmm-probe.log, e2-grouped-probe.log, e2-layout-probe.log,
e2-fast-probe.log, e2-layer-probe.log. Numbers match the headings,
including SMALL_BMM_OK, E2_GROUPED_PROBE_OK, E2_LAYOUT_PROBE_OK,
E2_LAYER_PROBE_OK, ESIMD max_abs 0, w13 3.78 vs 12.94, w2 1.99 vs
6.53, bmm view 2.94, gather 0.52, index_add 7.03/7.66, layer spans
that sum near 34.22, T8192_rel 1.770498e-04. e2-fast-probe.log ends
at GEMV_MISMATCH. The process exit code is not in that log.
int3-int4 0.0973-0.0724 = 0.0249 ms, times 40 is about 1.0 ms.
40*54.44 ms = 2.18 s (heading uses 54 ms and 2.16 s). 40*34.22 ms
= 1.37 s. 8185/5000 = 1.637 s. The 140 ms chunk is not a printed
span (printed loops are 115.1 and 119.2). e1mm.so and e2.so mtimes
were not moved by the 09:51 probes. No identity and no health claim
on these probes. The served ITL gap 11.29 vs 10.23 is a prior journal
pair, not re-benched here.

Verdict: OK on the probe numbers. The 22 s line is an estimate.
UNVERIFIED the fast-probe process rc. The GEMV mismatch line is real.

Suggested fix: keep B70_LOWBIT_GEMV_ROWS unset. Record the fast-probe
exit code in the log. Do not quote 22 s as a measured serve time.

### F51 -- C11 build, SGN bench, and the nan probe

Claim: 09:50 build in flight, 09:49 _C.c11.so built, 10:05 SGN bench.
SGN 8/16/40 match SGN=4 on the small layer (max_abs 0). SGN=40 does
not launch at T=4096 or 12288. Layer ms match the heading. SGN=16 is
not faster than C10 bmm at T=12288. failures=1 inside the python.
Health before and after 0. A dtype probe is "queued".

Evidence: c11-build.log ends C11_BUILD_OK, 48 warnings, cpu-run 60s
exit 0. _C.c11.so 1591952 bytes at 09:48. _C.c10.so and the default
_C.so were not replaced. c11-bench2.log: waited on H1 xe-reset pid
3201092 since 09:56:28, HEALTH_BEFORE 0, small-layer max_abs 0,
FAIL sgn=40 vs bf16-weight max_abs 2.319e+01 max_rel 1.941e+00,
ROUTE empty=0, event_us match the heading's ms (T=12288
77920.4/76416.0/76137.6). SGN=40 at T=4096 and T=12288 raises
RuntimeError, work-group above 512. Dequant gate/up/down
7807.0/7814.6/7407.0 us (256.0/255.8/258.4 GB/s). Copy gate bf16
6375.8 us / 526.3 GB/s. oneDNN TypeError on torch.nn.functional
shadowed as F. BUDGET best_sgn 4/8/16. VERDICT failures=1, cpu-run
exit 0, gpu-run exit 0, HEALTH_AFTER 0. The heading's "held since
09:59" is the state after this bench; the bench itself waited on
the 09:56 xe-reset.

c11-bench.log (09:51) is an earlier run the 10:05 heading does not
use: CHECK finite=False, VERDICT failures=4, gpu-run exit 1, health
after HEALTHY. c11-nan.log (09:55) already finished: y_randn all NaN
on both _C.c11.so and _C.c10.so, gate_bf16 finite max_abs 480,
HEALTH_BEFORE and HEALTH_AFTER 0, C11_NAN_RC 0. c11-build-b.log ends
C11_BUILD_OK, _C.c11b.so 1760272 bytes at 10:04, cpu-run 70s exit 0.
c11-xslm.log and c11-dnn.log are only the lease wait on H1 xpu-health
since 09:59:25. No dispatch change is claimed, and none is shown.

Verdict: OK on the 10:05 bench numbers and on the build. The sentence
"a dtype probe is queued" is WRONG as of 10:05: c11-nan.log had
already finished at 09:55 and the outputs are nonfinite. The oneDNN
wait was still true.

Suggested fix: cite c11-nan.log in the C11 heading. Refuse SGN that
makes the MT=16 group larger than 512 before submit, instead of
raising from the runtime.

### F52 -- D3 no-MTP and Shisa e2e

Claim: the 09:50 D3 heading. No-MTP C1 ITL 12.60/12.83/13.07, 79.39
tok/s, a 0.20 ms win versus D1 12.81/12.99/13.12 and 78.23. Shisa
spec=2 C1 18.67/19.15/19.41, 121.67 tok/s, acceptance 113/170, versus
D2 18.72/18.97/19.26 and 121.73 with the same 113/170. Identity and
coherence pass. First Shisa attempt lost the device on ragged slot
width 3. Second attempt skipped ragged and benched clean. Card 1
HEALTHY. B70_D3_GDN_SPEC stays off in the verdict.

Evidence: d3-bench-gdn.json matches the no-MTP numbers (F46's parse,
now owned). d1-bench-graph-fa.json C1 p50/p90/p99 12.807/12.995/13.121,
78.228 tok/s, C2 p50 13.749 and 13.748, prefill 6029.7/7888.3/6414.4.
Raw delta versus D3 p50 12.603 is 0.204 ms, which rounds to 0.20.
d2-bench-shisa-spec2.json C1 18.725/18.967/19.256, 121.734 tok/s,
accepted 113, draft 170. d3-bench-shisa.json C1 18.674/19.154/19.414,
121.669 tok/s, accepted 113 of 170. Identity on all three JSON files
is hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2,
first_ok and alias_ok true, short Paris coherence OK. D1, D2, and D3
serves are the same checkpoint, FULL_DECODE_ONLY, FLASH_ATTN, max len
33792, port 18081. D2's banner is gdn=default. D3 Shisa sets
B70_D3_GDN=1 and B70_D3_GDN_SPEC=1, so the 0.05 ms pair is not a
single-variable ablation.

d3-serve-shisa.log is the second attempt (startup 09:46, gpu-run done
in 186s exit 0). It logs "b70_d3 gdn skip spec ragged ntok=4
slot=(2, 3)" and then uniform captures. GET /v1/models 200 and
completion 200. Shutdown is SIGTERM, clean. The file contains no
DEVICE_LOST string. reports/1D3_gdn_attn.md cites that same filename
for a 09:29 first attempt. That text is gone. No d3* log contains
HEALTHY or xe-reset. The spec bench ran with B70_D3_GDN_SPEC=1 even
though the verdict says that flag stays off.

Verdict: OK on the JSON numbers, the 0.204 ms rounding, the ragged
skip, and the clean second shutdown. UNVERIFIED the first-attempt
device loss and the card-1 HEALTHY line, because the cited log was
overwritten and no health file was saved.

Suggested fix: keep the first-attempt log under a distinct name.
Save the xpu-health transcript next to the serve log. Say in the
verdict that the spec bench turned B70_D3_GDN_SPEC on, and that the
flag stays off for the daily path.

### F53 -- G4 T2 abort on RequestOffloadState

Claim: the 09:55 G4 heading. Health before OK. Ready 09:53:38.
slots=4 chunks=1542 exact_tail=True, mmap 34.36 GB, the four TTFT
lines, host HIT, then parking.py:578 AttributeError on
_g4_tail_key, empty restored completion, cold filler HTTP 500, no
T2_RESULT, health after HEALTHY. Not a pass.

Evidence: g4-t2-driver-c1-r1.txt and r2.txt are identical. Ready
09:53:38. Identity list hotschmoe-dev then the Ornith GPTQ alias.
turn1 ttft 0.531s computed 3000, control 0.443s computed 3081,
B 5.032s, C 5.023s, restored ttft 0.013s out=0 text empty, then
HTTP Error 500. Serve log g4-t2-serve-c1-r1.txt: command line matches
the heading (card 1, port 18111, 45 blocks, cpu_bytes 34359738368,
shm 40g, memory 80g, PIECEWISE mode 0, park-finished-tail, fork
serve path). Parking line at 09:53:32 matches slots/chunks/groups.
mmap 34.36 GB. Queued tails end=3063/3144/30464/30464. Traceback
at 09:54:06 is parking.py:578 `req_status._g4_tail_key = record.key`,
AttributeError, EngineDeadError. Driver health after is xpu-health
HEALTHY card 1, gpu-run exit 0, ATTEMPT_RC 1, T2_DRIVER_EXIT 1.
No T2_RESULT. The log does not print the token HIT. The string
swap_blocks_batch was not in the lines read.

The file parking.py on disk now has no _g4_tail_key. Line 578 is
`if record is None:`. The hit path stores
`self._tail_choice[request.request_id]`. That edit is not in a
journal heading yet. g4-t2-driver-c1-r3.txt (10:02) is a later
attempt: SERVE_DIED before ready, card 1 held by H1 xpu-health since
09:58:21, gpu-run timeout 180s, rc=124. g4-t2-health-after-c1.txt
is that later wait (90s timeout, card held by E2), not the abort
run's HEALTHY line. g4-t2-health-before-c1.txt is the 09:58 retry
and is HEALTHY card 1.

Verdict: OK on the abort numbers, the traceback, identity, and the
driver's health-after. The HIT word is the code path, not a log
token. The r3 timeout is unjournaled. The parking edit is unproven.

Suggested fix: journal the r3 lease timeout separately. Re-run T2
only after the _tail_choice path is the one loaded, and keep a
T2_RESULT only if the restored completion is non-empty.

### F54 -- H1c M1 r1, numeric miss, xe-reset

Claim: the 09:59 H1c heading. Warmup, attempt 1 err1=11 then attempt
2 clean, own_max=0 at every T, the full_max/elem_rel table, TIME
T=1 580 vs 411 us and T=8 591 vs 816 us, M1_RC=1, xe-reset, health
after HEALTHY at 09:58:47. The T=4096 miss is the relative
definition. The gate is not described as passed.

Evidence: h1c-m1-layer-r1.log (same bytes as h1c-m1-layer.log).
qtypes 22/22/21, E=256. WARMUP touch all 1, exchange attempt=1
err1=11, attempt=2 err0=0 err1=0. OK T=1/2/4/8 own_max=0 bitwise=1.
full_max/full_rel: T=1 3.725e-09 / 8.720e-06, T=2 7.451e-09 /
4.310e-04, T=4 1.490e-08 / 4.585e-04, T=8 1.490e-08 / 4.493e-04,
T=4096 FAIL full_max 2.980e-08 full_rel 7.515e-02 cos 1. TIME
579.6/410.9 and 591.1/816.1. FAIL layers=1. gpu-run done in 47s
exit 1. h1c-m1-window-r1.out: health before HEALTHY cards 0 1,
M1_RC=1 at 09:56:28, xe-reset unbind/bind both BDFs, collective
HEALTHY, RECOVERED method=rebind, HEALTH_AFTER_RC=0 at 09:58:47.
Host loader in the same log is libze_loader.so.1.28.2. The 09:52
heading is only the window open and has no result.

Verdict: OK. The printed table matches, with T=4's 4.585e-04 rounded
to 4.5e-4 in the heading. M1_RC=1 means the process failed its own
gate. "xe-reset on a numeric miss will not run again" is a promise,
not a measurement.

Suggested fix: score max|a-b| / max|b| in the gate that sets the
process rc, and do not xe-reset on that numeric miss. Keep the
elem_rel print.

### F55 -- unjournaled E2 serve death and side logs

Claim: no heading through 10:05 owns a lowbit e2 serve result.
The 09:51 E2 heading says the next serve is still ahead.

Evidence: logs/e2-serve.log, card 1, port 18084, int3 checkpoint
qwen36-w3a16-g128-experts-w4a16-trunk, served-model-name
hotschmoe-dev then qwen36-w3a16-b70-lowbit, FULL_DECODE_ONLY,
FLASH_ATTN, max len 33792, max-num-batched-tokens 8192,
B70_LOWBIT_SO=libb70_lowbit_xpu_e2u.so. Application startup complete.
GET /v1/models 200 twice. The response body is not in the log, and
there is no coherence text. At 10:14:26 a completion with
prompt_token_ids_len=8204 and num_computed_tokens=7616 died in
moe.py expert_prefill_bmm at torch.bmm:
UR_RESULT_ERROR_OUT_OF_RESOURCES. HTTP 500. gpu-run done in 946s
exit 0. No xpu-health line and no xe-reset in that log.
e2-prefill-mode.txt contains "matmul". e2-fwd-split.log says
fwd_split installed. No bench JSON and no tok/s.

Verdict: UNVERIFIED as a journal claim because no heading owns it.
The OOR and the engine death are in the raw log. Do not treat exit 0
as a healthy serve.

Suggested fix: a heading that records the OOR, whether /v1/models
returned hotschmoe-dev first, and health after. Run xe-reset if the
card is not HEALTHY before the next GPU job.

### F56 -- code read this pass

Claim: new plugin and parking code should not hide a bad bound or a
host sync inside a captured op.

Evidence: sched_moe.hpp b70_sched_gu checks g >= plan->ng and
row_wg * kSchedRows >= 2 * I, and launches with
reqd_sub_group_size(kSchedSg). No queue.wait in that launch. The
grid size uses the host integer host_groups while the kernel also
reads plan->ng. d3_gdn.py refuses spec when ntok != nseq * nwidth
or the index is short, and the Shisa log shows that skip. The C11
SGN=40 path does not pre-check the 512 work-item cap; the bench
raises from the runtime (F51). ep probe.cpp and ep_kernels.cpp still
call memcpy().wait() on host reads. The M0/M1 logs place warmup
before the scored rows. This pass did not show those waits inside a
replayed graph. moe.py expert_prefill_bmm is the OOR site (F55) and
still uses a host .item() on the prefill pad path (F47). parking.py
no longer writes _g4_tail_key; the crash traceback is from the code
that ran at 09:54. The replacement _tail_choice path has not been
re-measured. Plugin diff since the last status check is 33 lines in
types.py, decode.hpp, dequant_fast.cpp, iq_dot.hpp, kernels.cpp,
row_moe.hpp, and compare_cpu.py, with no synchronize or .item() added.

Verdict: OK on the sched bounds and the ragged GDN guard. The C11
work-group reject is a hard error, not a silent skip. UNVERIFIED
whether any ep memcpy.wait sits inside capture. The parking fix is
unproven.

Suggested fix: size the sched grid from plan->ng. Pre-check C11
work-group size. Keep ep host waits out of the captured submit.
Re-bench T2 on the _tail_choice build.

### F57 -- repo hygiene at 10:14 UTC

Claim: no secrets, no newly staged binaries, no host UMD change, fork
not pushed.

Evidence: git diff --cached is empty. git status --short has 18 lines.
The only binary is the pre-existing modified
vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so.
vllm/flashnext/.gitignore `*.so` ignores _C.c11.so and the ep _C.so.
No secret-like names. Fork b70/flashnext tip moved from e442b4b53f
to 9761b7b48a, "[XPU] Attach the default-off QSA plugin when
B70_QSA=1", ahead 15 of origin/main, working tree clean. That commit
does not supply the missing QSA_XPU_OK transcript (F44). Serve logs
this pass still resolve host libze_loader.so.1.28.2 and IGC 2.36.3.
No heading claims an apt or driver change.

Verdict: OK.

Suggested fix: none. Leave the pre-existing .so unstaged. Do not push.

