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

