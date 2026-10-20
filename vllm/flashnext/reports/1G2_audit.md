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

## Pass 7 -- 2026-10-20 10:55 UTC

Cursor at write: JOURNAL.md ended at the 10:50 G4 heading (line 2824).
Headings checked: 10:21 E2 through 10:50 G4. The 10:41 and 10:48 C7
entries are window announcements only.

### F58 -- E2 first bmm OOR

Claim: the 10:21 E2 heading. Identity hotschmoe-dev first. Coherence
is the landmarks sentence. MATMUL_1 TTFT 33212 ms, 247 tok/s.
MATMUL_2 TTFT 4311 ms, 1903 tok/s, with a 46.4 percent prefix-cache
hit, so it is not a full 8k prefill. BMM then dies in
expert_prefill_bmm. The dumped step has 588 scheduled tokens and
7616 prefix hits. No post-crash health, because G4 took card 1.

Evidence: logs/e2-driver.log lines 8-15. Identity first_ok true,
alias qwen36-w3a16-b70-lowbit. Both coherence texts start
" Paris, a city renowned for its rich history, culture, and iconic
landmarks." MATMUL_1 33212.096 ms and 247.018 tok/s. MATMUL_2
4310.564 ms and 1903.231 tok/s. BMM_1 is empty and the client raises
HTTP 500 at 10:14:26. Pass 6 had already read the then-current
e2-serve.log dump: prompt_token_ids_len=8204 and
num_computed_tokens=7616, dying in moe.py:232 torch.bmm with
UR_RESULT_ERROR_OUT_OF_RESOURCES. 588 is 8204-7616. That dump is
no longer in the file. The live e2-serve.log is a later serve (one
Application startup, one OOR at 10:39). The string 46.4 is in
neither the driver nor the current serve log.

Verdict: OK on identity, coherence, the two matmul times, and the
HTTP 500. The 588 figure matches the overwritten dump's arithmetic.
UNVERIFIED the 46.4 percent hit rate.

Suggested fix: keep the first serve log under its own name. Cite the
metrics line that prints the hit rate.

### F59 -- E2 second serve, unjournaled

Claim: no heading through 10:50 owns a second lowbit serve.

Evidence: the same e2-driver.log continues at 10:22. Identity and
the landmarks coherence pass again. MATMUL_1 and MATMUL_2 on 8186
tokens are 31352 ms / 261.1 tok/s and 31299 ms / 261.5 tok/s.
BMM_1 returns 35950 ms / 227.7 tok/s. BMM_2 is empty. The live
e2-serve.log OOR at 10:39:48 has total_num_scheduled_tokens=8186
and num_computed_tokens=0, same torch.bmm error 40. e2-driver.status
is FAILED bench_rc=1 health_rc=0. e2-health.log is HEALTHY card 1,
gpu-run exit 0. Mode file is the four bytes "bmm". No bench JSON.

Verdict: UNVERIFIED as a journal claim. The second bmm prefill did
return a TTFT, and the following request killed the engine. Do not
read 228 tok/s as a prefill win. Health after this second death is
HEALTHY.

Suggested fix: a heading for the second serve that separates BMM_1
from the BMM_2 OOR.

### F60 -- H1c M1 r2 passes, health closed later

Claim: the 10:22 heading, closed by the 10:42 health line. Gate is
bitwise, cosine, and max|dy| / max|y| < 1e-2. full_rel T=1/2/4/8/4096
is 5.9e-8, 6.6e-8, 1.4e-7, 1.2e-7, 8.9e-8. T=4096 abs max 3.0e-8.
elem_rel 7.5e-2 is not the gate. TIME 500 vs 410 us and 592 vs 818 us.
M1_RC=0. No xe-reset. Health after rc 0 at 10:40:23.

Evidence: h1c-m1-layer-r2.log. Warmup exchange attempt=1 err0=0
err1=0. Printed full_rel 5.878e-08, 6.550e-08, 1.439e-07, 1.182e-07,
8.923e-08. full_max at T=4096 is 2.980e-08. elem_rel 7.515e-02.
TIME 500.4/410.0 and 592.3/817.6. PASS m1. h1c-m1-window.out:
health-before rc=0 at 10:15:00, M1_RC=0 at 10:21:40, no xe-reset
line, HEALTH_AFTER_RC=0 at 10:40:23, both cards HEALTHY. The wait
was E2's second card-1 serve.

Verdict: OK. The 10:22 sentence that health-after was still waiting
was true at that time. The 10:42 heading's close matches the window.

Suggested fix: none for the gate. Keep quoting full_rel, not elem_rel.

### F61 -- C11 stays slower than C10 bmm

Claim: the 10:26 C11 heading. SGN and XSLM match SGN=4. bf16 ref
fails. Empty experts 0. The event-ms table, XSLM 14.80/28.97/70.02,
down 18.98 to 11.94, C10 bmm 66.97, dequant about 7.8 ms at
255-259 GB/s, copy 6.35 ms at 529 GB/s, uniform bmm 4.72/7.86/4.65 ms,
grouped_mm 21.53 ms, dense ceiling 3.24 ms, two streams 636/124/727 us.
BUDGET best is xslm512 at 70.02 ms and 2791 tok/s, slower than C10.
Health 0. No xe-reset. Dispatch unchanged.

Evidence: c11-xslm.log. SGN vs SGN=4 max_abs 0, xslm vs sgn=4 max_abs 0.
ROUTE empty=0 at T=512/4096/12288. Event us matches the heading's ms
(T=512 sgn 9998.5/10117.2/10001.7, T=4096 28261.5/27872.6/27980.3,
T=12288 77775.2/76539.9/76282.2, xslm 14798.9/28974.0/70018.8).
SGN=40 at T=512 is 14759.7 us. T=12288 sgn=4 down_us=18976.1 and
xslm down_us=11936.3. c10_ms=66.97 is printed. BUDGET line matches.
HEALTH_BEFORE and HEALTH_AFTER 0, C11_GPU_RC 0. c11-dnn.log REF bf16
max_abs 2.5277e+01 max_rel 1.9815 frac_gate 0.9368; fp32 and fp16 also
miss. Dequant 7827.5/7820.2/7396.9 us at 255.4/255.6/258.7 GB/s.
Copy 6347.6 us at 528.6 GB/s. Uniform bmm T=12288 gate/gateup/down
4719.2/7860.8/4649.4 us. grouped_mm gate T=12288 21525.6 us and
T=4096 21979.7 us. Health 0 on that run too. The strings for a
3.24 ms dense ceiling and for 636/124/727 us streams are not in the
c11 logs. Kernel-sum arithmetic 7.83+7.82+7.40+7.86+4.65 = 35.56
(heading 35.55) and the T=4096 sum 33.12 matches the same dequant
plus the T=4096 bmm rows. Those sums are not a single layer launch.

Verdict: OK on the bench table and on "XSLM is slower than C10 bmm".
UNVERIFIED the dense ceiling and the two-stream lines.

Suggested fix: point those two sentences at the log line, or drop them.
Keep dispatch on the C10 split.

### F62 -- G4 T2 host restore passed

Claim: the 10:28 G4 heading. Attempt 1 rc 124. Attempt 2 ready
10:20:05, pass=True, cached=3063, computed=18, host_delta=3063,
device_delta=0, restored TTFT 0.1753 s, control 0.4166 s, cold
4.934 s, load 131235840 bytes in 0.063618 s = 2.063 GB/s, first
store 391249920 bytes in 0.089957 s = 4.35 GB/s, both copies
swap_blocks_batch. Health after HEALTHY. No xe-reset.

Evidence: g4-t2-driver-c1-r3.txt still starts with attempt 1 rc=124
(card held by H1, timeout 180s) and then attempt 2. Ready 10:20:05.
Identity hotschmoe-dev then the Ornith alias. Restored text matches
the control text, out=64, cached=3063, computed=18, ttft 0.175 s.
Control ttft 0.417 s. Cold 4.934 s. pass=True. JSON agrees:
host_delta 3063, device_delta 0, load_bytes 131235840,
load_time_s 0.063618073, greedy_match true, restored ttft
0.175267 s, control 0.416552 s. Serve log: mmap 34.36 GB, slots=4
chunks=1542, "G4 load tail end=3063 local=0 keys=6". Metrics:
store_bytes 391249920, store_time 0.08995698, then load_bytes
131235840 and load_time 0.063618073. 131235840/0.063618073 = 2.063
GB/s. 391249920/0.08995698 = 4.35 GB/s. Health after in the driver
is HEALTHY card 1, 26 s, exit 0. T2_DRIVER_EXIT 0. No xe-reset.
The string swap_blocks_batch is not in the g4 logs searched.

Verdict: OK on the gate numbers, identity, greedy match, and health.
UNVERIFIED the swap_blocks_batch op name. This is not a Paris
coherence run. The 10:50 heading's repeat of 2.063 GB/s matches.

Suggested fix: log the copy kernel name next to the byte counters.

### F63 -- G4 F-B CPU test

Claim: the 10:38 G4 heading. Fork 765993c197, CPU only, F_B_OK,
block=16, external hit=23, prompt=64, scheduled=41, computed after
schedule is 64.

Evidence: g4-fb-test-r3.txt. docker run with VLLM_TARGET_DEVICE=cpu,
HF_HUB_OFFLINE=1, --network none, --cpus 4, --memory 16g. Printed
"scheduled=41 computed=64 hits=23" and F_B_OK. cpu-run done in 29s
exit 0. No /dev/dri in the command. The log does not print block=16
or prompt=64. Fork tip is 765993c197, parent 9761b7b48a, ahead 16,
not pushed.

Verdict: OK on F_B_OK and the 41/64/23 line. UNVERIFIED block=16
from this log.

Suggested fix: print block and prompt on the F_B_OK line.

### F64 -- C12 host dequant, then an unjournaled gate run

Claim: the 10:39 and 10:43 C12 headings. _C.c12.so sha256
d46f80175c67a106, 10120096 bytes, 10:28. compare_cpu 14/14 including
Q8_0 mismatches=0. First link failed in 0.3 s. The gate script was
mode 644, exit 126, then requeued. At 10:43 device gates had not
started. Crop probe five CROP_OK and Q8_0 mismatches=0.

Evidence: sha256 and size match the file on disk at 10:28:57.
c12/compare_cpu.log VERDICT failures=0 types=14/14, Q8_0
blk.39.attn_k.weight mismatches=0. c12/build.log is the successful
link, cpu-run 125s exit 0. The 0.3 s failure is not in that log.
c12/crop_probe.log has five CROP_OK lines and Q8_0 mismatches=0.
run_gates.sh is now mode 775. No EACCES or exit 126 string remains
under c12/. c12/gates.log, after the 10:43 heading, acquired card 0,
health before HEALTHY, three VERDICT failures=0 lines (the third
covered=12), C9_FAIR cosine 0.999992 to 0.999999 with max_rel_bulk
up to 1.1534, then C12_PHASE. The first phase rows before that
verdict include gu=2464520 us. Later T=1 rows are about 180-220 us.
GATES_DONE, gpu-run done in 150s exit 0, health after HEALTHY.
No journal heading owns those device numbers. No serve and no ITL.

Verdict: OK on the hash and the host compare. UNVERIFIED the 0.3 s
link failure and the EACCES exit. The device gate completion is real
and unjournaled. Do not quote the 2.5 s phase row as steady state.

Suggested fix: a heading that cites gates.log, separates the warmup
phase row, and keeps the bulk-rel miss next to the cosine.

### F65 -- G5b hook commit, batch result not in the heading

Claim: the 10:38 G5b heading. Commit 9761b7b48a, model.py clean.
QSA_SPEC_OK and QSA_SERVE_OK. Library 817928 bytes at 10:16.
Card 0 health-before HEALTHY. No QSA_BATCH_OK yet. No xe-reset.

Evidence: fork log shows 9761b7b48a with that subject, and the tree
was clean at the pass-6 check. libb70_qsa_xpu.so is 817928 bytes,
mtime 10:16:22. g5b-qsa-batch.log health-before is HEALTHY card 0.
QSA_SPEC_OK and QSA_SERVE_OK are not in the logs or the flashnext
tree search. g5b-qsa-batch-r2.log, after the heading, is card 1:
health before HEALTHY, QSA_BATCH_OK, gpu-run done in 24s exit 0,
health after HEALTHY. No xe-reset line in that tail.

Verdict: OK on the commit, the library size, and the card-0
health-before. UNVERIFIED the two CPU ok-tokens. The card-1
QSA_BATCH_OK run is unjournaled. It is not a serve and not an ITL.

Suggested fix: save the CPU test stdout next to the GPU log, and
journal the card-1 QSA_BATCH_OK line with health before and after.

### F66 -- H1 M2 died after the window heading

Claim: the 10:42 heading says M2 is waiting. It does not claim a
greedy result.

Evidence: h1c-m2-window.out got the lease after C7, health-before
rc=0 at 10:42:32, both cards HEALTHY, READY 10:45:37. Greedy JSON
texts are empty, both errors HTTP 500, same=false, match_c4=false.
The window still prints GREEDY_RC=0 and BENCH_RC=1. h1c-m2-serve.log
serves hotschmoe-dev then qwen3.6-35b-a3b-ud-iq3-xxs, eager, port
18114, both cards. At 10:45:38 the engine dies with
"RuntimeError: ep flag timeout on a previous call" and both
completion posts return 500. No health-after line in the window
head that was read.

Verdict: the 10:42 "window open" sentence was true when written.
The later greedy failure is unjournaled. GREEDY_RC=0 does not match
the empty HTTP 500 JSON.

Suggested fix: set GREEDY_RC from match_c4, and journal the flag
timeout with health after. This is not a coherence pass.

### F67 -- G4 W8 pointer overflow

Claim: the 10:50 heading. COPY_START 10:40:42, ValueError overflow,
no GB/s, gpu-run exit 1 in 15s, health after 10:41:34 HEALTHY,
no xe-reset. libb70_kvpark.so 90904 bytes from 08:41.
libb70_kvpark_ordered.so 95512 bytes at 10:47, not loaded.

Evidence: g4-copy-driver.txt COPY_START 10:40:42 card 1, malloc_host
type=host, ValueError at bench_copy.py:38 torch.tensor([dst],
dtype=torch.int64), gpu-run done in 15s exit 1, COPY_DRIVER_EXIT 1.
Health-after file is HEALTHY card 1, and the driver records rc=0 at
10:41:34. No COPY_BENCH line. The two libraries are 90904 bytes at
08:41:34 and 95512 bytes at 10:47:48. g4-copy-driver-r2.txt only
shows a later launch line at 10:50:12 naming the ordered .so. No
result in that file yet.

Verdict: OK. No bandwidth number.

Suggested fix: none beyond the uint64 pack the heading already
describes. Do not quote a GB/s until COPY_BENCH prints one.

### F68 -- hygiene at 10:55 UTC

Claim: no new staged binary, fork not pushed, no host UMD change
in this window.

Evidence: git diff --cached is empty. git status still flags only
the pre-existing modified
vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so
as a binary. Fork b70/flashnext is 765993c197, ahead 16 of
origin/main. Serve and probe logs in this window still show host
libze_loader.so.1.28.2. No heading claims an apt or driver change.
c7-ep-tp-serve.log from the 10:41 window was not scored.

Verdict: OK on staging and the fork tip.

Suggested fix: none. Leave the .so unstaged. Do not push.

Cursor extended before the pass-7 journal line. Headings after the
10:50 G4 entry are F69 through F76. F58-F68 are unchanged.

### F69 -- E2 8k split, second crash is F.sigmoid

Claim: the 10:52 E2 heading. Identity hotschmoe-dev first. Coherence
TTFT 92 ms and 71 ms, landmarks sentence. MATMUL 31352 ms / 261 tok/s
and 31299 ms / 262 tok/s. Warm step 2: forward 31263, linear_attn
19519 (n=30), self_attn 5312 (n=10), mlp 6395, experts 6390,
shared_expert 2272 (n=80), router 4.4, norm 22, host_sync item 7,
tolist 44, synchronize 1. BMM_1 35950 ms / 228 tok/s. Step 3: forward
35754, linear_attn 19592, self_attn 5329, experts 10794, group 3687,
bmm_w13 1692, bmm_w2 1281, pad_x 441, reduce 372, unpack_w13 135,
unpack_w2 68. BMM_2 empty. Engine died at 10:39:48 on F.sigmoid of
the shared-expert gate, error 40. Diag returned 0, then bench
HTTP 500. Health rc 0 at 10:40:41. Dense time is WNA16 because
linear.py unpacked only for env exactly matmul. Tiled bmm unpacked
all 256 experts (~1.5 GB). Not a 5000 tok/s result.

Evidence: e2-driver.log lines 78-91 and 130-135. Identity first_ok.
COHERENCE ttft 92.205 and 71.437, both landmarks sentences.
MATMUL_1 31351.952 / 261.100, MATMUL_2 31298.939 / 261.542,
BMM_1 35950.258 / 227.704, BMM_2 empty. OFF_FILE_SET. PHASE
10:39:48 diag rc 0, split log lines 50, bench rc 1. PHASE 10:40:41
health rc 0. e2-health.log HEALTHY card 1, gpu-run exit 0.
e2-bench.json was not written. The 50-line split file was read
before the 10:52:15 serve truncated it to the two install lines.
Step 2 and step 3 matched the heading's rounded milliseconds,
including host_sync item 7 tolist 44 synchronize 1. prefill_flag
on that file was the string matmul (printed once at install).
The serve log of that run, read before the same overwrite, died
at 10:39:48 in qwen2_moe.py:121
`out = F.sigmoid(self.expert_gate(x)[0]) * out`, UR error 40.
F59's sentence that this second OOR is torch.bmm is wrong. The
first crash (F58) remains the overwritten torch.bmm dump.
The second serve command omitted --gdn-prefill-backend. On XPU,
qwen_gdn_linear_attn.py:274 logs "Using {backend} GDN prefill
kernel" and returns, so a missing "Using SYCL GDN" line is the
CUDA label, not proof of the backend. linear.py now returns true
for both matmul and bmm, and its comment says env bmm used to
fall through to prefill_wna16. Step 3 linear_attn stayed 19591.56
ms, the same class as step 2. The 1.5 GB / 256-expert unpack size
is not in the split log. E1's 0.1 s dense figure was not re-opened.
A third serve started at 10:52:15 with --gdn-prefill-backend sycl
and was still in weight load at 10:57. Not scored.

Verdict: OK on identity, the short coherence TTFTs, the rounded
step times, diag rc 0, the sigmoid OOR, health, and "not 5000
tok/s". UNVERIFIED the 1.5 GB figure. The live split and serve
logs no longer contain this run.

Suggested fix: keep each serve's split log and traceback under a
run id. Correct any later cite of the second crash to F.sigmoid.
Do not quote 228 tok/s as a prefill win: bench_rc=1.

### F70 -- H1 M2 r1 flag timeout, health closed

Claim: the 10:53 H1c heading. Health before HEALTHY. Warmup attempt
1. All 40 MoE layers split 128/128. Init 11.56 s. Greedy died in
method.py apply with "EP previous error layer=1 err0=0 err1=11"
and ep flag timeout. Layer 0 had no stored error. Connection reset.
Health after HEALTHY rc 0 at 10:48:44. No xe-reset. New _C.so
sha256 7f0981fd20a2375965fd89612bb32969a71ca6906ef94bfa9a6d1d0e4926f010,
476912 bytes. Eager ep_moe host-waits both queues. Capture still
records the pipelined chain.

Evidence: h1c-m2-window-r1.out is the 10:42 window. Health-before
rc 0 at 10:42:32, both cards HEALTHY. READY 10:45:37. Greedy texts
empty, both HTTP 500, match_c4 false, BENCH_RC=1, GREEDY_RC=0.
ConnectionResetError 104. SERVE_RC=0 at 10:45:43 (the stop wait).
Window done rc=1. HEALTH_AFTER_RC=0 at 10:48:44, HEALTHY both
cards. No xe-reset string. h1c-m2-serve-r1.log: hotschmoe-dev then
qwen3.6-35b-a3b-ud-iq3-xxs, eager, port 18114, B70_EP=1.
EP init nlayers=64 spin=200000. Warmup exchange and moe attempt=1
err0=0 err1=0. Split lines are layers 0-39, each n0=128.
Init engine took 11.56 s. Trace ends at method.py:125
`y = torch.ops.b70_ep.moe_ep(...)` and core.py RuntimeError ep
flag timeout. The only "EP previous error" line is layer=1
err0=0 err1=11. _C.so sha256 and size match the file on disk.
moe_ep.cpp:408 prints that line when the layer's saved flag is
set. ep_wait for card 1's flag_in passes error code 11.
enqueue returns before any host wait when cap_on is set
(moe_ep.cpp:414). The eager path then calls q1->wait() and
q0().wait() (lines 421-422). That host sync is outside capture.

Verdict: OK. GREEDY_RC=0 still does not match the empty HTTP 500
body. SERVE_RC=0 is the container stop, and the window rc is 1.
This is not a coherence pass. r1 text was not measured.

Suggested fix: set GREEDY_RC from match_c4. The eager host wait
matches the failure mode. Keep it off the capture path.

### F71 -- C7 TP=2 loader refusal

Claim: the 10:43 C7 heading. Fork e442b4b53f. HEALTH_RC 0. Both
workers raised at serve_loader.py:337, packed sharding not
implemented. RAISE:1 at 10:43:25. SERVE_RC 137. No xe-reset.
Later commit 35da272b61 does not change this raise.

Evidence: c7-tp-window-raise.out. Collective-health label,
HEALTH_RC:0, RAISE_SEEN 10:43:25, SERVE_RC:137, RAISE:1. Both
worker tracebacks at 10:43:23 carry the packed-attention
RuntimeError. At e442 the f-string is line 338, so the raise
statement is the line above it. 35da272b61 touches only
_check_linear and adds _check_split_shards. The raise text is
unchanged. The window has no /v1/models line and no greedy.
The HEALTHY banner is not in the window file, only HEALTH_RC:0.
The 09:01 DP=2 measurement was not re-opened.

Verdict: OK on the refusal, both workers, and SERVE_RC 137.
UNVERIFIED the 09:01 citation this pass. Not a speed result.

Suggested fix: none for the refusal. Keep TP=1 for this packed
GGUF until a shard exists.

### F72 -- C7 Flash-Next load IndexError

Claim: the 10:51 C7 heading. HEALTH_RC 0. Shards=2, tensors=1224,
root Qwen4ExpForCausalLM, EP 256/512, qk_rows=4096, placed 100.
Both workers IndexError in _check_linear on
sizes[_SHARD_INDEX[shard]]. Next tensor is
blk.3.indexer.k_proj.weight, ggml (2560, 128). SERVE_RC 1.
No greedy. No xe-reset. G4 took card 1 at 10:51:26.

Evidence: c7-fn-window-index.out HEALTH_RC:0, serve exited before
ready, stop 10:51:11, SERVE_RC:1. c7-fn-serve-index.log: DP=2
TP=1, eager, max_seq_len 2048, model hf-flashnext. Load line
shards=2 tensors=1224 root=Qwen4ExpForCausalLM ep=0/2 and ep=1/2,
qk_rows=4096. expert_map_manager prints local/global experts
256/512. placed 100 tensors. Both workers IndexError at the
pre-commit line `expect = sizes[_SHARD_INDEX[shard]]`
(serve_loader.py:260 in that build). The log does not name
blk.3.indexer.k_proj.weight or the ggml shape. G4's copy driver
prints COPY bench start 10:51:25, then the card-1 acquire.
No greedy line. No xe-reset string.

Verdict: OK on the load counters, the IndexError site, SERVE_RC 1,
and health rc 0. UNVERIFIED the tensor name and the ggml shape.
The server never reached /v1/models.

Suggested fix: the 10:55 commit is the proposed guard. Do not
call this load a collective failure.

### F73 -- C7 indexer commit is unmeasured

Claim: the 10:55 C7 heading. Fork 35da272b61, serve_loader.py
only, not pushed. PY_COMPILE_OK. Q rows 512, K rows 128, hidden
2560, indexer heads 4 and 1, head dim 128. 512+128 equals the
ReplicatedLinear output. Unmeasured.

Evidence: fork tip is 35da272b61, ahead 17 of origin/main, status
shows ahead and not a push. The commit adds 37 lines to
serve_loader.py. A one-partition module records shard rows and
_check_split_shards compares the sum to b70_output_sizes.
py_compile of that file in this pass returned rc 0. No saved
PY_COMPILE_OK line and no saved GGUF metadata dump were found.
The 512 and 128 figures are the commit message.

Verdict: OK that the commit exists, is loader-only, and is
unmeasured. UNVERIFIED the metadata row counts against the GGUF.
The heading's own verdict says unmeasured.

Suggested fix: save the metadata print next to the next window.
Do not treat py_compile as a serve.

### F74 -- G4 ordered copy, D2H misses 7.5 GB/s

Claim: the 10:56 G4 heading. bad=0. H2D 13.344 / 14.031 / 14.223 /
14.037. D2H 4.636 / 4.583 / 4.659 / 4.670. Fragments about 13.4.
swap H2D 8.853 / 8.110 / 6.042 / 4.119. swap D2H about 4.66.
worst 4.583. Exit 1 because the gate wants 7.5. Health after
10:52:09 HEALTHY. dev_ptr 0xffffd556b7400000. ordered=1.
No xe-reset.

Evidence: g4-copy-driver-r2.txt. ordered=1, malloc_host type=host.
The four byte rows match those GB/s figures and bad=0. Fragments
13.372 / 13.469 / 13.488 / 13.442. swap D2H 4.661 / 4.677 /
4.680 / 4.656. worst_one_buffer=4.583 COPY_BENCH. gpu-run exit 1
in 28s. bench_copy.py returns 1 when worst < 7.5. Health-after
file is HEALTHY card 1, and the driver records rc 0 at 10:52:09.
dev_ptr on the 16 MiB row is 0xffffd556b7400000. No xe-reset
string. Library size 95512 was checked at the 10:50 entry.
The 8.5 GB/s Strata probe was not re-opened. B70_KVPARK_COPY is
not printed. This supersedes F67's "r2 has no result yet".

Verdict: OK. H2D is above 7.5 on these four sizes. D2H and the
worst one-buffer number are not. bad=0. Not a serve and not an ITL.

Suggested fix: keep quoting worst_one_buffer next to the gate.
Do not wire the copy in until D2H is explained.

### F75 -- G5b card 0 graph ids fail, card 1 batch passes

Claim: the 10:55 G5b heading. Same 817928-byte library. Card 0
QSA_XPU_OK, eager batch sections pass, union attn max_abs
7.007e-4, then FAIL graph ids mismatch. RUN_EXIT 1. Health after
HEALTHY. Card 1 health before HEALTHY (27s). QSA_XPU_OK, graph
ids exact at widths 5, 6, and 7. QSA_BATCH_OK. Graph pos 4 and
pos 5 exact, slot-3 seq 0 attn and gated max_abs 0, inactive
pages exact. gpu-run 24s exit 0. RUN_EXIT 0. Pooled inside 1e-5,
scores inside 1e-4, attn inside 2e-3. Not a speed claim.

Evidence: g5b-qsa-batch.log. Health before HEALTHY card 0.
Sections include block 4, block 8, pos_base, mrope, inactive,
chunked prefill, union attend (union q1 attn max_abs=7.007e-04),
draft, export import. QSA_XPU_OK at line 700, before the graph
replay that prints FAIL graph ids mismatch. gpu-run exit 1 in
28s. Health after HEALTHY card 0, rc 0, RUN_EXIT:1. No xe-reset
string. g5b-qsa-batch-r2.log. Health before HEALTHY card 1,
gpu-run 27s exit 0. Single-seq graph: "graph replay pos 4 ids
exact (5,)", pos 5 ids exact (6,), pos 7 ids exact (7,), with
attn and gated max_abs 0. Batch graph: pages, tail, dead,
block_pos, length, attn, gated exact, then "graph pos 4 ids
exact" and "graph pos 5 ids exact". Slot-3 lines: graph seq0
pages exact, ids exact (7,), attn max_abs 0, gated max_abs 0,
inactive pages exact, inactive tail and dead max_abs 0.
QSA_BATCH_OK. gpu-run 24s exit 0. Health after HEALTHY.
RUN_EXIT:0. Printed atols are 1e-5 pooled, 1e-4 scores, 2e-3
attn, and QSA_BATCH_OK means those checks passed. test_qsa_xpu.py
now fill_(-1) and compares ids[:width]. The comment in
test_qsa_batch.py says topk does not write ids past width.
F65's missing QSA_XPU_OK is this card-0 line. Not a serve and
not an ITL. The libb70_qsa_xpu_next.so compile and the MTP
mark_draft note were not checked.

Verdict: OK on both GPU logs, including card 0's graph ids miss
and card 1's pass. UNVERIFIED the XMX next-library sentence.

Suggested fix: keep the prefix compare. Card 0's RUN_EXIT 1
stays the record of the old whole-buffer check.

### F76 -- code and hygiene at 11:00 UTC

Claim: no new staged binary. Host waits added for the eager EP
path stay off capture. The ordered copy wait is gated by the
sync flag. No host UMD change.

Evidence: git diff --cached is empty. git status still flags only
the pre-existing modified
vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so
as a binary. New untracked paths checked this pass are small
source files (qsa_xmx.cpp 20703 bytes, the largest). Fork
b70/flashnext is 35da272b61, ahead 17, not pushed. g4-copy r2
still resolves libze_loader.so.1.28.2. moe_ep.cpp returns before
q1->wait() and q0().wait() when cap_on is set. copy_batch.cpp
calls last.wait() only when sync is true. qsa_xmx.cpp has no
synchronize, .item, or wait. moe.py's synchronize mentions are
the diagnostic-span comments, not a new capture sync. linear.py
now treats prefill flag bmm like matmul.

Verdict: OK on staging and on the capture guard. The copy wait
is safe only while graph replay calls it with sync false. That
caller was not traced this pass.

Suggested fix: none for staging. Leave the .so unstaged. Do not
push. If a serve path passes sync true inside capture, drop the
wait on that path.

## Pass 8 -- 2026-10-20 11:40 UTC

CPU only. Cursor is the 11:35 E2 heading (JOURNAL line 3368) plus
the H1 M3 window, which finished at 11:38:08 and is not in the
journal yet. F1-F76 are unchanged.

### F77 -- C12 11:02 graph times match, bulk error does not

Claim: the 11:02 C12 heading. Health before and after HEALTHY
card 0. Three VERDICT failures=0, the third covered=12. Fair
us_graph gu=21 dt=42 is 186.95 / 277.77 / 398.45 / 535.64 and
gu=17 dt=20 is 198.51 / 288.44 / 414.10 / 571.82. Cosine 0.99999.
Strata 112.89 / 148.10 / 202.93 / 261.64, ratio about 1.66x and
2.05x. Second binary 457a262d, 11266352 bytes, not measured yet.

Evidence: c12/gates.log lines 7 and 251 are HEALTHY card 0.
VERDICT failures=0 at lines 150, 167, and 187 (covered=12).
OURS lines 189-196 match those microseconds. Cosine on the lines
is 0.999992 to 0.999999. gu=17 T=1 max_abs is 1.4664e+01.
gu=17 T=2 max_rel_bulk is 1.1534. The OURS lines do not name
IQ3_S, Q2_0, IQ2_XS, or IQ4_NL. 186.95/112.89 is 1.656 and
535.64/261.64 is 2.047. The Strata row is the citation already
in JOURNAL line 2403. Measured copy
c12/_C.c12.measured.so is 10120096 bytes, sha256
d46f80175c67a10681e09e7642b5293eedec0d3f98517b5fe7ba4c9662b70112.
c12/_C.c12.ec3.so is 11266352 bytes, sha256
457a262da74b9477ebf34922be18d481c8c5fb224db5b50f2d3277f014d4762f.
Fair2 later ran that second binary and it did not capture (F79).

Verdict: OK on the microseconds, the health lines, and the
verdict lines. The cosine heading is rounding. The gu/dt type
names are UNVERIFIED. max_rel_bulk is not an 8e-2 pass. Not an
ITL.

Suggested fix: print the ggml type names on the OURS line. Keep
max_rel_bulk next to the microsecond row.

### F78 -- G4 flags probe, D2H stays under 7.5

Claim: the 11:10 G4 heading. 256 MiB. Compute 13.812 / 4.687.
Cached 14.205 / 4.690. Uncached 14.014 / 4.663. Write-combined
14.165 / 4.649. bad=0. Health after 11:08:52 HEALTHY.
FLAGS_DRIVER_EXIT 0 is the probe return, not a 7.5 GB/s pass.
Library 106584 bytes.

Evidence: logs/g4-flags-driver.txt contains those six GB/s
figures, bad=0, FLAGS_DRIVER_EXIT, and the health-after rc.
libb70_kvpark_flags.so is 106584 bytes. bench_copy.py returns 1
only when worst_one_buffer is under 7.5. This driver exited 0,
so the script did not apply that gate. D2H on every arm is about
4.65 to 4.69. No serve. The 11:00:56 compile clock was not
re-stated. No xe-reset string in the extract.

Verdict: OK. H2D clears 7.5. D2H does not. Exit 0 is not a pass
of the 7.5 gate.

Suggested fix: keep printing the gate next to FLAGS_DRIVER_EXIT.

### F79 -- C12 EC-split graph does not instantiate

Claim: the 11:12 C12 fair2 heading. Eager failures=0 covered=12,
Q8_0 mismatches=0, cropped max_abs under 2e-3. capture_end
RuntimeError Failed to instantiate native UR executable graph.
gpu-run 83s exit 1. No us_graph. No xe-reset. Health before
HEALTHY. Binary 457a262d, 11266352 bytes.

Evidence: c12/fair2.log. Health before HEALTHY card 0. VERDICT
failures=0 and VERDICT failures=0 covered=12. Q8_0 mismatches=0
on blk.39.attn_k.weight. Cropped rows max_abs 1.936e-03,
9.458e-04, 2.194e-04, 3.684e-04, 1.059e-04. moe_layer max_abs is
4.673e-03, 1.314e-02, and 9.073e-03, so those rows are not the
cropped cases. graphs.py capture_end raises the UR instantiate
error. gpu-run done in 83s exit 1. No second HEALTHY line.
c12/_C.c12.ec3.so matches the sha256 and size. 48*11=528 is
arithmetic, not a log line.

Verdict: OK that eager cropped checks passed and the graph did
not instantiate. No speed result. No health after.

Suggested fix: record health after an exit-1 graph run. Do not
quote this binary as a graph candidate.

### F80 -- C7 11:12 window was only an announcement

Claim: the 11:12 C7 heading says the Flash-Next DP=2 window is
open. Leases requested. No result in that heading.

Evidence: the heading stops at VERDICT: window open. The window
file later ended SERVE_RC 1. That result is the 11:32 heading,
checked in F90. No tok/s in the 11:12 entry.

Verdict: OK as a window announcement. It is not a measurement.

Suggested fix: none. The result belongs on the 11:32 entry.

### F81 -- G4 mmap H2D 9.435, D2H 4.451

Claim: the 11:14 G4 heading. h2d=9.435 d2h=4.451 bad=0. Health
after 11:12:37 HEALTHY. MMAP_DRIVER_EXIT 0. C7 held both cards
from 11:13:08.

Evidence: logs/g4-mmap-driver.txt contains h2d=9.435, d2h=4.451,
bad=0, MMAP_DRIVER_EXIT, and the health-after rc. c12/fair3.log
line 1 says card 0 was held by C7 since 11:13:08, name
vllm-fn-flashnext-gguf-ep. D2H 4.451 is under 7.5. H2D 9.435 is
over 7.5. Not a serve. B70_KVPARK_COPY is not printed.

Verdict: OK. The pool pointer clears 7.5 only in the H2D
direction.

Suggested fix: same as F78. Do not treat MMAP_DRIVER_EXIT 0 as
the bandwidth gate.

### F82 -- E2 third serve TTFTs match; split table is gone

Claim: the 11:15 E2 heading. Server up 11:06:45. Identity
hotschmoe-dev. Coherence TTFT 88 and 69 ms. MATMUL_1 4906 ms
and 1668 tok/s. MATMUL_2 4555 ms and 1797 tok/s. Warm split
step 2: forward 4521, linear_attn 242, experts 4110. BMM_1
10648 ms and 769 tok/s. BMM_2 died at torch.bmm, error 40.
Diag rc 1. No health after.

Evidence: logs/e2-driver.log. PHASE 11:06:45 server up.
first_ok true, ids hotschmoe-dev then qwen36-w3a16-b70-lowbit.
COHERENCE 88.331 and 69.121. MATMUL_1 4906.205 / 1668.499.
MATMUL_2 4554.704 / 1797.263. BMM_1 10648.473 / 768.749.
BMM_2 EMPTY. PHASE 11:07:11 diag rc 1, split log lines 50,
FAILED. The saved traceback in the driver is EngineDeadError
from async_llm.py and serving.py. e2-fwd-split.log is 0 bytes
now; a later serve truncated it. No e2 log still contains
"4521", "linear_attn", or "torch.bmm". No health phase between
FAILED and the 11:14:55 restart.

Verdict: OK on identity, the coherence TTFTs, the matmul and
bmm TTFTs, and diag rc 1. UNVERIFIED the internal millisecond
table and the torch.bmm / error-40 site. 1797 tok/s is not
5000, and the run died. Not a qualified prefill result.

Suggested fix: copy the split log aside before the next serve
truncates it. Keep the worker frame, not only EngineDeadError.

### F83 -- G5b XMX union matches within the stated atols

Claim: the 11:17 G5b heading. Library 962040 bytes. Health
before and after HEALTHY. QSA_XPU_OK, QSA_BATCH_OK, PASS xmx
refuse, xmx ok=1 union_cells=58, vs SIMT 9.710e-04, vs CPU
5.405e-04, PASS xmx, QSA_XMX_OK, RUN_EXIT 0. Not an ITL.

Evidence: logs/g5b-qsa-xmx.log contains those strings, gpu-run
done in 32s exit 0, and HEALTHY card 0 before and after.
libb70_qsa_xpu.so and libb70_qsa_xpu_next.so are both 962040
bytes, sha256
2ee734458f9ed0ac0ebd827c409c3771ba315fab15dfdf004bd19da24f67bf71.
The SIMT backup libb70_qsa_xpu.so.simt-817928 is 817928 bytes.
qsa_xmx.cpp has no synchronize, .item, or wait. Not a serve.

Verdict: OK. The atols are 2e-2 and 5e-2. This is not a decode
ITL.

Suggested fix: none for the numeric lines. Do not quote it as
serve latency.

### F84 -- C13 library linked

Claim: the 11:19 C13 heading. cpu-run 139s exit 0.
_C.c13.so is 2635416 bytes. _C.so stays 7574840.
C13_BUILD_OK. Warnings only. Measurement not started.

Evidence: logs/c13-build.log ends with C13_BUILD_OK and cpu-run
done in 139s exit 0. _C.c13.so is 2635416 bytes, sha256
0efe261a8cc200d9d2b2dabf47b5a7008c848dec565b0a41040699cee04a1510.
_C.so is 7574840 bytes. The heading's _C.c12.so size 10169704
matches c12/_C.c12.fused.so. The live plugin _C.c12.so is now
the later 10165216-byte link (F93). dequant_wide.cpp and
c13_bind.cpp have no synchronize, .item, or wait.

Verdict: OK that the library linked. The bench in this log is
not a result. The later bench is F89.

Suggested fix: none for the link.

### F85 -- H1 M2 eager ITL is a real serve

Claim: the 11:12 H1c heading. Identity first hotschmoe-dev.
Greedy texts equal the C4 loop string. match_c4 is false only
because the script still compared the landmarks sentence.
Coherence OK. C1 ITL p50 81.84 / p90 83.19 / p99 83.78, 12.60
tok/s, TTFT 217.93, n=32. C2 both streams p50 87.33. BENCH_RC 0,
GREEDY_RC 0, SERVE_RC 0. Health before 11:07:45 and after
11:12:23.

Evidence: logs/h1c-m2-bench.json. identity first_ok true,
ids hotschmoe-dev then qwen3.6-35b-a3b-ud-iq3-xxs. model field
hotschmoe-dev. Coherence texts are both the C4 loop string,
ok true, same true. C1 itl_p50_ms 81.8448, p90 83.1874, p99
83.7825, decode_tok_s 12.6037, ttft_ms 217.932, completion
tokens 32. C2 itl_p50_ms 87.3274 and 87.3306. The window file
from this pass's earlier read has READY 11:10:45, health-before
rc 0 at 11:07:45, health-after rc 0 at 11:12:23, and SERVE_RC 0.
The r1 file is a different run and still has an empty HTTP 500.
C4's 87.62 ms citation was not re-opened.

Verdict: OK on identity, the texts, coherence, the ITL, and
health. GREEDY_RC 0 here is a text match. The landmarks
comparator is a script bug and is not the text. Not a graph
result. The C4 comparison is UNVERIFIED this pass.

Suggested fix: point match_c4 at the C4 loop string, as the M3
script later did.

### F86 -- H1 M3 window was open, then finished unjournaled

Claim: the 11:20 H1c heading. Window queued. Card 0 held by
G5b. Not measured yet.

Evidence: that heading has no bench numbers. logs/h1c-m3-window.out
later reached READY. The finished window is F94, and it is not
in JOURNAL.md. The 11:20 text was true when written.

Verdict: OK as an announcement. Do not read it as an ITL.

Suggested fix: append the finished window before quoting it.

### F87 -- C12 fused graph also fails instantiate

Claim: the 11:21 C12 fair3 heading. Same UR instantiate error.
Eager failures=0 covered=12. gpu-run 93s exit 1. No us_graph.
Health before HEALTHY. Binary sha256 6a2e412b, 10169704 bytes.

Evidence: c12/fair3.log contains the UR instantiate error and
gpu-run exit 1. c12/_C.c12.fused.so is 10169704 bytes, sha256
6a2e412bfde900fc086d3fc6b1b7bf46a4fd4c36db8527efe0afe285046a285c.
No us_graph line in the spot check. The use_root_sync diagnosis
is the heading's reading, not a log field. No health-after line
was part of the captured tail.

Verdict: OK that this binary failed graph instantiate after
eager checks. Not a speed result.

Suggested fix: same as F79.

### F88 -- G5b prefill numbers match; some prose is not in the log

Claim: the 11:21 G5b heading. PASS prefill tiles, gated worst
1.953e-03, XMX did not launch. PASS prefill xmx, 16 appended
cells, 10 queries, gated vs CPU 9.766e-04. QSA_XMX_OK.
RUN_EXIT 0. Health before and after. 26s exit 0.

Evidence: logs/g5b-qsa-prefill.log contains PASS prefill tiles,
1.953e-03, PASS prefill xmx, prefill xmx gated vs cpu
max_abs=9.766e-04, QSA_XMX_OK, QSA_BATCH_OK, xmx vs simt
9.710e-04, xmx vs cpu 5.405e-04, gpu-run done in 26s exit 0,
HEALTHY card 0 before and after, RUN_EXIT:0. The strings
"XMX did not launch", "16 appended", and "10 queries" are not
in the log. Neighbor-page and id lines were not re-listed.
Not a serve and not an ITL. No xe-reset line in the extract.

Verdict: OK on the max_abs lines, the PASS lines, health, and
RUN_EXIT 0. UNVERIFIED the three prose phrases as log text.

Suggested fix: quote the printed line. The test still does not
call QsaRunner.forward.

### F89 -- C13 exit 124 is not in the surviving bench log

Claim: the 11:22 C13 heading. Health before rc 0. cpu-run
--quiet exited 124 at 90s. The script printed "FAIL python
timed out". No kernel ran. Health after rc 0. gpu-run 129s
exit 124.

Evidence: a search of the flashnext tree found no "FAIL python
timed out" and no "exit 124" in the bench log. logs/c13-bench.log
was overwritten. The file now is a later run: gpu-run acquired
card 0, xpu-health HEALTHY, HEALTH_BEFORE:0, several "OK exact"
rows, then ModuleNotFoundError: No module named gguf in
c13_bench.py load_real_experts. cpu-run done in 13s exit 1.
HEALTH_AFTER:0. gpu-run done in 43s exit 1. That later run is
not in the journal.

Verdict: UNVERIFIED the exit-124 claim, because the log it
names was replaced. The surviving run is a real exit 1 with
health before and after, and it is not a kernel timing.

Suggested fix: keep the timed-out log under a distinct name.
Install gguf in the bench environment before the next card run.

### F90 -- C7 DP=2 death is a host global OOM

Claim: the 11:32 C7 heading. HEALTH_RC 0 at 11:12:39. Serve
11:13:08. Both workers placed 1200 tensors at 11:17:22. No
Python traceback. Kernel 11:17:41 oom-kill CONSTRAINT_NONE
global_oom pid 3258155, total-vm 129572252 kB, anon-rss
46915752 kB, file-rss 12 kB. Mem-Info active_anon 70711248 kB,
inactive_anon 26628912 kB, active_file 0, gpu_active 22211584
kB, all_unreclaimable, Normal free about 195 MB. Reaped
11:17:44. SERVE_RC 1 at 11:18:06. Died while copying
per_layer_token_embd.weight, 28800138240 bytes. No greedy.
No xe-reset.

Evidence: logs/c7-fn-window-oom.out is the seven-line window:
HEALTH_RC:0, serve 11:13:08, serve exited before ready, stop
11:18:06, SERVE_RC:1. logs/c7-fn-serve-oom.log shows both
workers placed 1200 tensors at 11:17:22, then EngineCore_DP1
at 11:17:44 "WorkerProc initialization failed" and
RuntimeError engine core initialization failed. The worker
root frame is not in that file. /var/log/kern.log at
11:17:41 matches the oom-kill line, the pid, total-vm,
anon-rss, file-rss, active_anon, inactive_anon, active_file
0, gpu_active 22211584kB, all_unreclaimable? yes, and Node 0
Normal free 194852kB. Reaped 11:17:44. The kernel log and the
serve log do not name per_layer_token_embd.weight. That name
and the 28800138240-byte size are the shard description in
reports/1C7_gguf_graph_ep.md. No /v1/models line. No xe-reset
string.

Verdict: OK on the host OOM, the counters, and SERVE_RC 1.
UNVERIFIED that the in-flight tensor was the PLE table. Not
an XPU wedge and not a greedy result. The serve never became
ready.

Suggested fix: the memmap view is the right next step. Say
"inferred next tensor" until a loader line names it.

### F91 -- PLE memmap commit exists; the CPU test log does not

Claim: the second 11:32 C7 heading. Fork 0a6170da8c, not
pushed. Host tables skip _owning_uint8 and raise on a
non-memmap source. cpu-run exit 0 in 12 s. HOST_TABLE_TEST_OK.
PY_COMPILE_OK. Both shard-2 PLE tensors are C-contiguous uint8
memmaps of 28800138240 bytes. MemAvailable about 108 GiB.
Greedy 323 / 667 is not measured.

Evidence: fork b70/flashnext is 0a6170da8c, ahead 18 of
origin/main, status ahead and not a push. The commit touches
serve_loader.py, ple.py, loader.py, and a mapping test.
_host_table_payload uses torch.from_numpy on a C-contiguous
uint8 view, walks .base for a memmap, and raises if the walk
fails. _owning_uint8 is still the path for other payloads.
A scan of /tmp, the flashnext logs, and the build tree found
no file containing HOST_TABLE_TEST_OK. MemAvailable was not
sampled from a saved file. No second serve log exists.

Verdict: OK that the commit matches the stated guard and is
unmeasured on a GPU. UNVERIFIED HOST_TABLE_TEST_OK,
PY_COMPILE_OK, and the 108 GiB figure. The 323 / 667 compare
did not run.

Suggested fix: save the cpu-run stdout next to the commit.
Do not treat the commit as a serve.

### F92 -- E2 fourth serve died before a prefill number

Claim: the 11:35 E2 heading. Server up 11:33:27. Identity and
both Paris calls passed, TTFT 99 ms and 70 ms. BMM_1 returned
no usage. Engine died at 11:33:38 in _group_slots at torch.cat,
level_zero error 40, before any expert bmm. Split log has the
config only. H1 took card 1 at 11:33:44. No xe-reset.

Evidence: logs/e2-driver.log. PHASE 11:33:27 server up.
first_ok true. COHERENCE 99.349 and 70.052, landmarks text.
BMM_1 EMPTY. PHASE 11:33:38 diag rc 1, split log lines 11,
FAILED. The driver traceback is EngineDeadError only. No e2
log contains _group_slots, torch.cat, or error 40. The split
file is 0 bytes after the next launch. H1's health gpu-run
finished at 11:34:13 and took 29s, which is 11:33:44 by
subtraction. The stamp is not printed. No xe-reset string.
A fifth serve prints PHASE 11:35:47 start and was still
waiting on weights at the last driver line. That run is not
scored. vllm-fn-g4-kv-c1 is named in h1c-m3-serve.log as the
card-1 holder from 11:34:44.

Verdict: OK on identity, coherence, the empty BMM_1, and diag
rc 1. UNVERIFIED the _group_slots / torch.cat / error-40 site.
Not a prefill tok/s. The fifth serve is in progress in the
driver snapshot.

Suggested fix: preserve the worker frame and the split log
before the next truncation. empty_cache next to the KV pool
stays off until that frame is on disk.

### F93 -- C12 no-root-sync graph times match

Claim: the 11:35 C12 heading. sha256 b3c15e14, 10165216 bytes.
Health before and after. Eager failures=0 covered=12. Fair
us_graph 180.17 / 267.04 / 385.36 / 515.13 and 191.86 /
280.62 / 402.88 / 560.55. Cosine 0.99999. About 4 percent
under the 11:02 rows. Ratio about 1.60x and 1.97x. gpu-run
84s exit 0. No xe-reset.

Evidence: c12/fair4.log. HEALTHY card 0 before and after.
VERDICT failures=0 and covered=12. Q8_0 mismatches=0. OURS
lines match those microseconds. Cosine is 0.999992 to
0.999999. max_abs on gu=17 T=1 is still 1.4664e+01 and
max_rel_bulk on gu=17 T=2 is still 1.1534, the same values
as gates.log. gpu-run done in 84s exit 0.
c12/_C.c12.noroot.so and the live plugin _C.c12.so are both
10165216 bytes, sha256
b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979.
180.17/112.89 is 1.596. 515.13/261.64 is 1.969.
(186.95-180.17)/186.95 is 3.6 percent. The IQ2 T=4 pair is
about 2 percent (571.82 to 560.55). Type names are still not
on the OURS lines. Strata remains the old citation.

Verdict: OK on the capture, the microseconds, health, and
exit 0. The "about 4 percent" title is the IQ3 rows, not
every row. Bulk relative error is unchanged. Not an ITL.
Shipped _C.so was not replaced by this file.

Suggested fix: keep this binary as a graph candidate only
for schedule time. The numeric gap to the reference row is
unchanged.

### F94 -- H1 M3 bench hit a server whose alias is Ornith

Claim: no journal heading owns this yet. The 11:20 window
later printed a bench.

Evidence: logs/h1c-m3-window.out. Health-before rc 0 at
11:34:13, HEALTHY cards 0 and 1. Serve line 11:34:13. READY
11:38:03. Greedy texts both equal the C4 loop string,
same true, match_c4 true. BENCH_RC=0 GREEDY_RC=0 at
11:38:08. Window text: identity first hotschmoe-dev,
first_ok true, alias_ok false. Coherence ok true.
C1 itl_p50 16.51, p90 16.78, p99 16.95, 60.62 tok/s,
TTFT 87.19, n=64. C2 p50 17.37 and 17.34. Stop 11:38:08.
logs/h1c-m3-bench.json matches those figures.
identity.ids is hotschmoe-dev then
ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.
expect_alias is qwen3.6-35b-a3b-ud-iq3-xxs. alias_ok false.
base_url is http://127.0.0.1:18115. The serve log is five
lines: the intended command is Qwen3.6-35B-A3B, GGUF,
FULL_DECODE_ONLY, port 18115, served names hotschmoe-dev
then the Qwen alias. It then waits because card 0 is held
by C12 run_fair.sh since 11:34:13 and card 1 is held by
vllm-fn-g4-kv-c1 since 11:34:44. No "Application startup
complete". No HEALTH_AFTER. No SERVE_RC. No xe-reset.
No DEVICE_LOST.

Verdict: WRONG to treat 60.62 tok/s as the Qwen M3 graph
serve. The first id is hotschmoe-dev, and the second id is
a different model. The launcher had not acquired the cards.
Health after is absent. The text match and the JSON
arithmetic are real for whichever server answered on 18115.

Suggested fix: fail the window when alias_ok is false.
Do not start the bench until gpu-run prints the acquire
for vllm-fn-h1-m3. Record health after.

### F95 -- hygiene at 11:40 UTC

Claim: no new staged binary and no secret file. Fork not
pushed. No host UMD change in the logs checked.

Evidence: git diff --cached in /mnt/vm_8tb/github/b70_ai_things
is empty. git status still shows the pre-existing modified
vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so.
New untracked paths checked for size are source files and two
small directories (guard_candidate 16K, ornith15_s2b 44K).
No status line matched a secret-like name. Fork
b70/flashnext is 0a6170da8c, ahead 18. Docker command lines
still bind libze_loader.so.1.28.2. .gitignore was not
re-opened.

Verdict: OK on staging. The push-allreduce .so remains a
dirty worktree binary and is not a new staged object.

Suggested fix: leave that .so unstaged. Do not push.

### F96 -- new plugin code has no host sync in the files scanned

Claim: review of plugins added this pass for races, strides,
bounds, and host sync inside capture.

Evidence: dequant_wide.cpp, c13_bind.cpp, and qsa_xmx.cpp have
no synchronize, .item, or wait. dequant_wide.hpp is
declarations only and documents dst_mode 0, 1, and 2.
_host_table_payload refuses a non-contiguous or non-memmap
PLE source instead of copying. moe_ep.cpp and copy_batch.cpp
were not re-walked; F76 still covers those waits. A full
stride proof of the wide dequant kernel was not done.

Verdict: OK that the three new C++ files scanned do not add
a host sync. UNVERIFIED a bounds or stride bug in
dequant_wide.cpp beyond that scan. The PLE path is a loader
change, not a graph op.

Suggested fix: none from this scan. The next pass should
read the wide kernel's row index against rows and cols if
that bench starts producing numbers.

## Pass 9 -- 2026-10-20 12:16 UTC

CPU only. Cursor is the 12:11 G4 heading and the 12:07 H1c
heading (JOURNAL lines 3596 and 3613). F1-F96 are unchanged.
The C12 35B graph bench finished and is not in the journal.

### F97 -- C13 first wide-dequant bench

Claim: the 11:35 C13 heading. Bit-exact vs prefill_dequant,
0 mismatches, including fused layout. Event GB/s IQ3_XXS
285.0 (old 256.2), IQ3_S 286.3, IQ2_XXS 272.9, IQ2_XS 275.0,
IQ2_S 279.2, Q2_0 270.5 (old 259.2), IQ4_NL 280.9. bf16 copy
1154.7. Small fast layer max_rel 1.19e-2. Fused synth max_rel
1.8625, frac_gate 0.9476, worst got -2.06 ref 1.77. Then
gguf import died. gpu-run exit 1. Health before and after.

Evidence: logs/c13-bench.log. HEALTH_BEFORE:0 and
HEALTH_AFTER:0, both HEALTHY card 0. Exact rows mismatches=0
for types 18, 42, 20, 16, 17, 22, 21 and fused-layout E=4.
DQ lines match the GB/s figures, including DQ_OLD 256.2 and
259.2. COPY gate_bf16 GBps=1154.7. STAT fast max_rel=1.1900e-02.
STAT fused max_rel=1.8625e+00, frac_gate=0.9476,
maxrel_got=-2.0558, maxrel_ref=1.7732. ModuleNotFoundError
gguf. cpu-run exit 1 in 13s. gpu-run exit 1 in 43s. The
type numbers are not spelled as ggml names on the exact rows.
The DQ lines do name the types.

Verdict: OK on the bandwidth, the mismatch counts, health,
and the gguf failure. Not an ITL. The fused relative error
is not a tight pass.

Suggested fix: keep the gguf import on the path that already
finished the bandwidth rows, and print type names on the
exact lines.

### F98 -- C13 ESIMD library linked

Claim: the 11:43 C13 heading. Build exit 0 in 96s.
_C.c13.so 2822840 bytes. C13_BUILD_OK. Bench queued.

Evidence: logs/c13-build2.log ends with that size, the
11:43 timestamp, C13_BUILD_OK, and cpu-run done in 96s
exit 0. c13-bench2.log is the later run, owned by F104.
The queued sentence was true when written.

Verdict: OK. A link is not a bandwidth result.

Suggested fix: none.

### F99 -- H1 M3 first window benched G4's Ornith server

Claim: the 11:42 H1c heading. Port 18115 answered while
gpu-run was still waiting. Alias was the Ornith GPTQ id.
C1 ITL 16.51 ms. alias_ok false. Text was the C4 loop.
Container never started. Discarded.

Evidence: this is the same window as F94.
logs/h1c-m3-bench-falseport.json identity.ids is
hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2,
alias_ok false. C1 itl_p50_ms 16.5068. Greedy texts are the
C4 loop string. logs/h1c-m3-serve-falseport.log never
acquires; it waits on C12 and on vllm-fn-g4-kv-c1.
logs/g4-t2-serve-c1-copy.txt is that Ornith serve on port
18115, ready 11:38:02. No DEVICE_LOST in the preserved
window.

Verdict: OK. Those ITL numbers are not the EP sidecar.
F94 stands.

Suggested fix: the heading's ready check is the right
gate. Keep alias_ok false as a hard stop.

### F100 -- G4 copy T2 restored the second turn

Claim: the 11:45 G4 heading. pass true. host_delta 3063,
device_delta 0, computed 18, cached 3063. Control TTFT
0.414 s, restored 0.124 s. Load 131235840 bytes in
0.009678 s = 13.560 GB/s, one load. Store 1369210880 bytes
in 0.298174 s = 4.592 GB/s. The 11:38:42 scrape is
440729600 during SIGTERM. Health after timed out at 90 s
because E2 held card 1 from 11:38:53. Identity Ornith.
No EngineDead.

Evidence: logs/g4-t2-turn-c1-copy.json. pass true,
greedy_match true, text_match true, host_delta 3063,
device_delta 0, computed 18, cached 3063, control ttft
0.4136, restored ttft 0.1242, load_bytes 131235840,
load_time_s 0.009678177. Identity is hotschmoe-dev then
the Ornith alias. logs/g4-t2-serve-c1-copy.txt: port
18115, copy queue group 1, store metrics 1369210880 bytes
and 0.298174322 s at 11:38:32, then 440729600 at 11:38:42
with SIGTERM. Running: 0 at 11:38:12. Driver
T2COPY_DRIVER_EXIT 124. Health-after file shows E2's
vllm-fn-lowbit-e2 since 11:38:53 and a 90 s timeout.
131235840/0.009678177 is 13.560 GB/s.
1369210880/0.298174322 is 4.592 GB/s.

Verdict: OK. H2D on this restore is above 7.5. The
pre-shutdown D2H counter is not. Exit 124 is the health
timeout, not a failed restore. Not a decode-ITL gate.

Suggested fix: quote the 90 s timeout next to pass=true
so the two exits stay distinct.

### F101 -- G4 card 1 health after the copy T2

Claim: the 11:56 G4 heading. HEALTHY, exit 0, 18 s,
11:56:16. No serve.

Evidence: logs/g4-t2-health-after-c1-copy-late.txt.
xpu-health HEALTHY card 1, gpu-run done in 18s exit 0.
File mtime is 2026-10-20T11:56:16. The file does not name
card 0's holder.

Verdict: OK on card 1 health. The card-0 sentence is
UNVERIFIED in this file.

Suggested fix: none for the health line.

### F102 -- C12 per-EC graph is slower; fused binary is installed

Claim: the 11:50 C12 heading. Per-EC sha256 cd2882dc,
11297864 bytes. Fair us_graph 185.76 / 320.79 / 378.78 /
595.23 and 197.79 / 329.97 / 403.33 / 626.23. gpu-run
107s exit 0. Health before and after. Installed file is
b3c15e14, 10165216 bytes, copied onto _C.so, _C.graph.so,
and _C.c12.so. ITL not measured in this heading.

Evidence: c12/fair5.log. HEALTHY card 0 before and after.
VERDICT failures=0 covered=12. Q8_0 mismatches=0. OURS
lines match those microseconds. Cosine 0.999992 to
0.999999. max_rel_bulk on gu=17 T=2 is still 1.1534.
gpu-run done in 107s exit 0.
c12/_C.c12.ecsplit.so is 11297864 bytes and its sha256
starts with cd2882dce7964770. _C.so, _C.graph.so, and
_C.c12.so are each 10165216 bytes, sha256
b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979.
Type names are still not on the OURS lines. The 35B ITL
is F111, not this heading.

Verdict: OK on the microseconds, health, exit 0, and the
install hashes. Separate entry-width launches are slower
than fair4 at T=1, T=2, and T=4. Bulk relative error is
unchanged. Not an ITL by itself.

Suggested fix: keep fair4's fused schedule as the installed
graph binary. Do not read this row as a serve result.

### F103 -- C13 ESIMD kernel hits an IGC fault

Claim: the 11:46 C13 heading. First wide kernel JIT failed
with IGC Internal Compiler Error, floating point exception.
Health after rc 0. gpu-run exit 1 in 31s. No bandwidth.

Evidence: logs/c13-bench2.log. HEALTH_BEFORE:0, wants_fast=1,
RuntimeError with that IGC sentence, cpu-run exit 1 in 3s,
HEALTH_AFTER:0, gpu-run done in 31s exit 1.

Verdict: OK. No bandwidth number exists. Not a serve.

Suggested fix: keep this binary off the serve path.

### F104 -- E2 official matmul bench is 1961 tok/s at 8k

Claim: the 11:55 E2 heading. Server up 11:53:47. Identity
hotschmoe-dev. Coherence Paris. Warm step forward 4479 ms,
linear_attn 243, self_attn 126, experts 4067, shared 35.
Bench rc 0: 1017 tok TTFT 2945 ms = 345 tok/s; 8185 tok
TTFT 4173 ms = 1961 tok/s; 32762 tok TTFT 9664 ms = 3390
tok/s. C1 ITL 11.34 / 11.54 / 11.70 ms, 89.9 tok/s. C2
p50 15.76 both streams. BMM trial died at y.index_add_
in expert_prefill_bmm, level_zero error 40. Health card 1
rc 0 at 11:55:30.

Evidence: logs/e2-bench.json tag e2-int3-matmul-fdo.
identity first_ok and alias_ok true. Coherence ok and same.
Prefill rows: 1017 tokens, ttft 2945.24, 345.30 tok/s;
8185 tokens, ttft 4173.35, 1961.25 tok/s; 32762 tokens,
ttft 9663.61, 3390.24 tok/s. Each prefill row has
n_chunks=1. C1 itl 11.339 / 11.541 / 11.700, decode_tok_s
89.912. C2 p50 15.760 and 15.761. logs/e2-driver.log:
server up 11:53:47, diag rc 0, bench rc 0, BMM_1 EMPTY,
bmm trial rc 1, health rc 0 at 11:55:30.
logs/e2-fwd-split.log step 2: forward 4479.28, linear_attn
243.42 n=30, self_attn 126.26 n=10, experts 4066.91,
shared_expert 34.64. logs/e2-serve.log at 11:54:33:
expert_prefill_bmm calls y.index_add_ and Level Zero
error 40 UR_RESULT_ERROR_OUT_OF_RESOURCES. The client
json does not say the 32762-token prefill was two chunks.

Verdict: OK on identity, coherence, the bench numbers,
the split step, the index_add frame, and health after.
5000 tok/s is not met. The two-chunk sentence is
UNVERIFIED. The BMM trial is not a serve path.

Suggested fix: keep the official bench next to the split
log. Leave grouped bmm off until error 40 is explained.

### F105 -- C13 block-write outputs do not match

Claim: the 11:55 C13 heading. Generic IQ4_NL 3x64
mismatches=0. Tuned mismatches: IQ3 128x2560 314956,
IQ3 2560x2560 6299964, Q2 4x640 1875, Q2 8x640 3757,
IQ4 64x256 15126, IQ2_XXS 75878, IQ2_XS 78065, IQ2_S
78257, IQ3_S 78729, fused 12599687. failures=10.
Health before and after. gpu-run exit 1.
_C.c13.so was 2647704 bytes.

Evidence: logs/c13-bench3.log. Those mismatch counts and
VERDICT failures=10. type=20 rows=3 cols=64 mismatches=0.
HEALTH_BEFORE:0 and HEALTH_AFTER:0. gpu-run done in 39s
exit 1. logs/c13-build3.log shows _C.c13.so 2647704 at
11:51 and C13_BUILD_OK. The exact lines use type numbers.
A later build4 linked 2643608 bytes. logs/c13-bench4.log
is still one wait line on C12's container. That later
bench has no number.

Verdict: OK that the block-write kernel failed exact
match and the process exited 1 with health on both sides.
Not a bandwidth result. The SLM rebuild is not measured
in the surviving bench4 log.

Suggested fix: do not install this .so. Keep the mismatch
list next to the rebuild.

### F106 -- G4 T3 r1 restore counts passed, greedy did not

Claim: the 12:04 G4 heading. cached 3063, computed 18,
host 3063, device 0, restored TTFT 0.146 s, control
0.415 s. Load 131235840 bytes in 0.009508 s = 13.803
GB/s, one load. greedy_match false. win_n=68. p50
17.3 to 19.0 ms. p99 17.7 and 32.4 to 51.0. Health
after HEALTHY, exit 0, 12:04:39. No EngineDead.

Evidence: logs/g4-t3-itl-c1-r1.json. pass false.
host_delta 3063, device_delta 0, cached 3063, computed
18, restored ttft 0.14600, control ttft 0.41495,
greedy_match false. active-0 and active-1 n_window=68.
p50_base about 0.01733 and 0.01735, p50_window about
0.01900. p99_base 0.03242 and 0.01772. p99_window
0.05101. logs/g4-t3-serve-c1-r1.txt load_bytes
131235840, load_time 0.009507656, size_count 1.
131235840/0.009507656 is 13.803 GB/s.
logs/g4-t3-driver-c1-r1.txt: client rc 1 at 12:04:11,
health after rc 0, HEALTHY card 1, gpu-run 13s exit 0,
T3_DRIVER_EXIT 1 at 12:04:39. Identity Ornith, port
18112. The quiet-lock wait names C12's bench on port
18104.

Verdict: OK. Token accounting and the copy rate match.
Greedy did not. The driver exit is 1. Health after ran.
This is not a passing ITL gate.

Suggested fix: keep pass=false in the heading result
line. The 12:11 rerun is F109.

### F107 -- H1 M3 capture dies on Level Zero 44

Claim: the 12:07 H1c heading. Health before 11:55:06.
EP warmup err 0. Split through layer 39. FULL capture
raises level_zero error 44 inside moe_ep from
profile_cudagraph_memory. SERVE_RC 1 at 11:57:49.
No DEVICE_LOST. No xe-reset. New _C.so sha256 e783539c,
481240 bytes. A recording stream does not host-wait.

Evidence: logs/h1c-m3-window.out. HEALTHY cards 0 and 1,
health-before rc 0 at 11:55:06, SERVE_DIED_BEFORE_READY,
SERVE_RC=1 at 11:57:49, no DEVICE_LOST, no xe-reset.
logs/h1c-m3-serve-capture44.log: EP WARMUP moe attempt=1
layer=0 err0=0 err1=0, split layer 39, then
profile_cudagraph_memory calls moe_ep and Level Zero
error 44 UR_RESULT_ERROR_UNSUPPORTED_FEATURE. The
window's health-after line is started and has no HEALTHY
result. The 12:08 queue stamp is not in that file.
plugins/b70_ep_xpu/b70_ep_xpu/_C.so is 481240 bytes,
sha256
e783539c4fa49f774a115a816209201a189a78d8d2c051f5b56c30575eb2c870.
moe_ep.cpp stream_recording() sets cap_on and the enqueue
path returns before q1->wait() and q0().wait() when the
queue is recording. Other functions in the file still
call wait. CUDAGraphMode.NONE was not re-read.

Verdict: OK on the error site, the warmup line, SERVE_RC
1, and the new binary hash. UNVERIFIED the 12:08
health-after queue and the CUDAGraphMode.NONE sentence.
Not an M3 ITL. The recording guard is in the source and
was not run.

Suggested fix: record health after when the queue clears.
The next window has to show the container acquire and
alias_ok before any ITL.

### F108 -- G4 T3 r2 gaps match; greedy breaks at token 50

Claim: the 12:11 G4 heading. cached 3063, computed 18,
host 3063, device 0. Load 131235840 bytes in 0.009544 s
= 13.750 GB/s. TTFT 0.144 s. Greedy matched 50 of 64,
then bigram (18, 2195). Gaps match the printed lists.
Baseline p50 17.4 ms. Health after HEALTHY, 15 s,
12:11:17.

Evidence: logs/g4-t3-itl-c1-r2.json. mismatch_at=50,
greedy_match false, pass false, host_delta 3063,
device_delta 0, restored ttft 0.14410, n_out 64.
p50_base 0.017414 and 0.017362. gaps_ms are the two
lists in the heading, to three decimals. Identity
hotschmoe-dev then the Ornith alias.
logs/g4-t3-serve-c1-r2.txt load_time 0.009544218,
load_bytes 131235840, which is 13.750 GB/s.
logs/g4-t3-driver-c1-r2.txt launch 12:07:41, port 18112,
client rc 1, health after rc 0, HEALTHY, gpu-run 15s
exit 0, T3_DRIVER_EXIT 1 at 12:11:17. control_ttft_s
in the json is 2.235, which the heading does not quote.
The ordered pair (18, 2195) is not a short json field.
The integer 2195 is in the file. The 83 ms gap is in
the list. The log does not label it as the 18-token
prefill.

Verdict: OK on the counts, the load rate, the gap list,
mismatch_at=50, and health after. The run's pass flag
is false. The bigram pair and the 83 ms label are
UNVERIFIED as log fields. Control TTFT on this run is
2.235 s.

Suggested fix: quote control TTFT next to restored TTFT.
Keep pass=false.

### F109 -- C12 35B graph bench is real and unjournaled

Claim: no journal heading owns c12/bench_35b.json. The
11:50 heading said the ITL was not measured yet.

Evidence: c12/serve_driver.log. SERVE_SO sha256
b3c15e14, the installed _C.graph.so. Health before
HEALTHY card 0. MODEL_IDS hotschmoe-dev and
qwen3.6-35b-a3b-ud-iq3-xxs. Bench identity first_ok
and alias_ok true. cpu-run done in 246s exit 0.
BENCH_RC=0. c12/bench_35b.json: C1 itl_p50 11.0188 ms,
p90 11.2288, p99 11.4693, 90.893 tok/s, TTFT 169.32 ms,
n=64. Coherence ok and same. Prefill 1017 tokens
177.43 tok/s TTFT 5731.68 ms; 8185 tokens 175.43 tok/s
TTFT 46656.35 ms; 32762 tokens 173.05 tok/s TTFT
189322.73 ms. The driver also prints C2 p50 16.61 ms
on both streams, with stream0 p99 75.01. Health after
has started and is waiting because card 0 is still
held by vllm-fn-qwen36-gguf-c0. No HEALTHY-after line.

Verdict: the identity, coherence, and bench numbers
match the files. Health after did not finish. This is
not a closed serve result until teardown is in the log.
Prefill is about 175 tok/s on this run. Do not promote
it against the Strata prefill row.

Suggested fix: stop the container, then record health
after, before quoting the ITL as qualified.

### F110 -- hygiene at 12:16 UTC

Claim: no new staged binary. Fork not pushed.

Evidence: git diff --cached is empty. git status still
shows the pre-existing modified
libxpu_push_ar_graph.so. The new C13 sources are
untracked text. Fork b70/flashnext is 0a6170da8c,
ahead 18. The installed gguf _C.so and _C.graph.so
are not status lines. No secret-like name in the
status list checked.

Verdict: OK on staging. The installed graph libraries
are a worktree change under the plugin tree and are
not staged.

Suggested fix: leave the push-allreduce .so unstaged.
Do not push.

## Pass 10 -- 2026-10-20 12:49 UTC

CPU only. Cursor is the 12:40 G4 heading. F1-F110 are unchanged.
A later cycle driver finished at 12:48:14 and is not in the journal.
The C12 35B health-after line is still the wait from F109.

### F111 -- 32 blocks refuse max_len 32768

Claim: the 12:16 G4 heading. ValueError. 32 blocks are 0.64 GiB.
A 32768 context wants 0.77 GiB. Estimated fit is 27200. Health
after HEALTHY, 12 s, 12:16:04. No xe-reset.

Evidence: logs/g4-t3-serve-c1-cycle-maxlen.txt. Port 18113,
num_gpu_blocks_override=32, served names hotschmoe-dev then the
Ornith alias. ValueError text is 32768, 0.77 GiB, 0.64 GiB, and
estimated maximum model length 27200. logs/g4-t3-cycle-driver-c1-maxlen.txt:
gpu-run exit 1, health after rc 0, HEALTHY card 1, health gpu-run
12s exit 0, T3_CYCLE_DRIVER_EXIT 1 at 12:16:04. No xe-reset string.
(32-7)*1088 is 27200. The engine never reached /v1/models.

Verdict: OK. This is an init refusal, not a restore and not an ITL.

Suggested fix: none. The 20480 retry is the next heading.

### F112 -- r1 cycle greedy matches and the restore gate does not

Claim: the 12:20 G4 heading. Eight restores. cached 0 on five and
1088 on three. Park tail end=1519. Actives ended at 17407 and
16319. Greedy matched. Health after HEALTHY at 12:20:33.
No G4 load tail.

Evidence: logs/g4-t3-cycle-c1-r1.json. Eight turns. cached_tokens
is 0 on turns 0-4 and 1088 on turns 5-7. greedy_match true on all
eight. cycle pass is false. Each turn has cached_ok false and
ok false. Identity is hotschmoe-dev then the Ornith alias.
logs/g4-t3-serve-c1-cycle-r1.txt has queued tail end=1519 at
12:19:37, end=17407 at 12:19:52, and end=16319 at 12:19:56.
Driver exit 1 at 12:20:33, HEALTHY card 1. The same serve log's
metrics line has kv_offload_load_bytes=86671360 and load_time
0.00639073 s.

Verdict: OK on the 5/3 cached split, the three end tokens, greedy
true, and health. The cycle pass flag is false. WRONG that there
was no load: the counter moved 86.7 MB.

Suggested fix: quote load_bytes. Keep cycle pass=false next to
greedy_match true. A text match on a full recompute is not a
restore.

### F113 -- r2 cached split is four and four

Claim: the 12:28 G4 heading. One active queued end=17407 before
the first restore. Five restores cached 0 and three cached 1088.
No load tail. Health after HEALTHY at 12:28:14.

Evidence: logs/g4-t3-serve-c1-cycle-r2.txt has queued tail
end=17407 at 12:27:33 and end=1519 at 12:27:00. The driver has
no per-turn clock, so "before the first restore" is not a
driver field. logs/g4-t3-cycle-c1-r2.json has eight turns:
cached 0 on turns 0-3 and 1088 on turns 4-7. That is four and
four, not five and three. greedy_match is true on all eight.
cycle pass is false. Metrics kv_offload_load_bytes=346685440,
load_time 0.025204 s. Driver exit 1 at 12:28:14, HEALTHY card 1.

Verdict: WRONG on the five-and-three count. OK on health and on
the end=17407 line. The load counter is not zero. Greedy true
does not make the cycle pass.

Suggested fix: recount the json. Print load_bytes. The r1 5/3
split does not carry forward.

### F114 -- 8 blocks refuse max_len 4096

Claim: the 12:33 G4 heading. ValueError. Available KV 0.15 GiB,
4096 tokens need 0.21 GiB, estimated max len 1088. Health after
HEALTHY, 13 s, 12:33:27. (32-7)*1088 is 27200. (8-7)*1088 is 1088.

Evidence: logs/g4-t3-serve-c1-cycle-8block.txt ValueError text
matches 4096, 0.21 GiB, 0.15 GiB, and estimated length 1088.
logs/g4-t3-cycle-driver-c1-8block.txt: health gpu-run 13s exit 0,
HEALTHY card 1, T3_CYCLE_DRIVER_EXIT 1 at 12:33:27. Both
products of 1088 are exact. No /v1/models line.

Verdict: OK. Not a GPU fault and not a restore.

Suggested fix: none.

### F115 -- r3 waiter samples and the block remainder

Claim: the 12:40 G4 heading. From 12:38:49 to 12:39:39 the engine
had Running 1, Waiting 1, prompt throughput 0, KV 58 to 64
percent. Tail end=18071. Restores cached 0 or 1088. No load tail.
Health after HEALTHY at 12:40:29. 26400 tokens is 25 blocks,
6000 is 6, and 7 blocks remain.

Evidence: logs/g4-t3-serve-c1-cycle-r3.txt. The 12:38:49 sample
is Running 1, Waiting 1, prompt throughput 1665.4 tok/s, KV
48.4%. Later samples in that window are prompt throughput 0.0
with KV 58.1%, 61.3%, and 64.5%, still Running 1 and Waiting 1.
A later sample is Running 1 and Waiting 0. Queued tail end=18071
at 12:39:48. Driver restores: cached 0 on four turns and 1088
on four turns. Metrics load_bytes=346685440, load_time 0.025158 s.
Driver exit 1 at 12:40:29, HEALTHY card 1. 26400/1088 ceilings
to 25. 6000/1088 ceilings to 6. 32-25-6 is 1, not 7. The 7 is
the reserve in (32-7)*1088. The log does not print "11 free" or
"needs 15". 64.5% of 32 blocks is about 11 blocks free, and
16000/1088 ceilings to 15.

Verdict: OK that a waiter sat beside a runner and that restores
were only full misses or one block. The 12:38:49 sample itself
still has prompt throughput. WRONG that 7 blocks remain after
prompts of 25 and 6 blocks. The load counter is not zero.
"About 11 free and 15 needed" is arithmetic on the 64.5% line,
not a log field.

Suggested fix: say one block is left after 25+6 on a 32-block
pool, and keep the 7-block reserve as a separate sentence.
Quote load_bytes.

### F116 -- a later cycle finished after the 12:40 heading

Claim: no journal heading owns the driver that launched at
12:44:16.

Evidence: logs/g4-t3-cycle-driver-c1.txt. Launch 12:44:16, port
18113, health before ok, READY 12:46:55. The captured tail had
cycle turns 6 and 7 with cached=1088 and device delta 1088,
then CYCLE_RESULT pass=False. Health after rc 0, HEALTHY card 1,
gpu-run 14s exit 0, T3_CYCLE_DRIVER_EXIT 1 at 12:48:14. The
restore histogram was not re-counted after the file grew.

Verdict: UNVERIFIED as a full restore table. The exit, the pass
flag, and health after are in the file. Not an ITL. Do not fold
it into F115.

Suggested fix: preserve this driver under its own name before
the next launch.

### F117 -- hygiene at 12:49 UTC

Claim: no new commit and no staged binary since F110.

Evidence: git diff --cached is empty. Fork b70/flashnext is
still 0a6170da8c, ahead 18. These five headings do not name a
new plugin source file. c12/serve_driver.log is still 31 lines
and still waiting on its own container for health after.

Verdict: OK on staging. The C12 teardown from F109 is unchanged.

Suggested fix: none for git. Stop the C12 container before
treating that bench as closed.

### F118 -- 12:48 G4 r4, 90.3 percent and end=26911

Claim: at 12:47:30 the engine had Running 1, Waiting 1, KV 90.3
percent, and prompt throughput 2640 tok/s. The runner queued
end=26911 at 12:47:34. Restores were cached 0, then device 1088.
No load tail. Health after was HEALTHY at 12:48:14. KV cache is
27870 tokens. Prompts were 26400 and 6000 with decode budget 512.
About 3 blocks were free, so the 6000-token partner did not fit.

Evidence: logs/g4-t3-serve-c1-cycle-r4.txt line 251 is
12:47:30, prompt 2640.0 tok/s, Running 1, Waiting 1, XPU KV
90.3%. Line 253 is 12:47:34 end=26911 tail=799 full_keys=24.
Line 125 is "XPU KV cache size: 27,870 tokens" for max 27,000.
Line 287 at 12:47:40 is Running 2, Waiting 0, KV 45.2%.
logs/g4-t3-cycle-driver-c1-r4.txt: identity hotschmoe-dev then
ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2, block=1088. Cycle
turns 0-3 are cached=0 host=0 device=0. Turns 4-7 are
cached=1088 host=0 device=1088 computed=441. Every turn has
ok=False and cached_ok=False. CYCLE_RESULT pass=False. Health
after rc 0, HEALTHY card 1, T3_CYCLE_DRIVER_EXIT 1 at
12:48:14. Serve metrics have store_bytes up to 2211348480 and
no kv_offload_load_bytes key. g4-t3-cycle-c1.json is absent.
The string 26400 is not in the serve log. The only "6000" hit
is inside a libc map address. 26911 = 26400 + 512 - 1, and
tail 799 = (26400 - 24*1088) + 511. (1 - 0.903) * 32 = 3.10.
This preserved driver is the run F116 saw under the live name.

Verdict: OK on the 12:47:30 sample, end=26911, the 27870-token
line, identity, and health after. WRONG as a restore summary:
device 1088 is only on the four cached=1088 turns, host stays
0, and pass is false. UNVERIFIED that load_bytes was 0, and
UNVERIFIED that the waiter was 6000 tokens. The 26400 length
fits the end, tail, and full_keys arithmetic and is not printed.
Not an ITL. Not a restore pass.

Suggested fix: quote pass=False, the 4/4 split, and host=0.
Keep the 12:47:40 Running 2 line next to the wait. Preserve the
cycle json so load_bytes can be checked.

### F119 -- 12:50 C12 35B graph ITL, journal of the F109 bench

Claim: health before HEALTHY card 0 in 23 s. Identity
hotschmoe-dev first, alias qwen3.6-35b-a3b-ud-iq3-xxs.
Coherence OK/OK with both texts " Paris.\nThe capital of France
is Paris.\nThe capital of France is". C1 ITL p50 11.019 ms,
p90 11.229, p99 11.469, 90.89 tok/s, TTFT 169.3 ms, n=64. C2
p50 16.606 / 16.607 ms, tok/s 55.24 / 62.08, stream 0 p99 75.01
ms. Prefill 177.4, 175.4, and 173.1 tok/s at 1017, 8185, and
32762 tokens. MTP off. cpu-run 246 s exit 0. BENCH_RC=0.
Container docker-stopped. Post-health had not run. C1 matches
C7's 11.11 ms. The 4 percent fair-layer gain does not move
this ITL.

Evidence: c12/bench_35b.json measured C1 itl_p50_ms
11.018843040801585, p90 11.228842800483108, p99
11.469298214651644, decode_tok_s 90.89271395464053, ttft_ms
169.31546805426478, n_chunks 64. C2 stream0 itl_p50_ms
16.60554698901251, tok/s 55.24457023209998, p99
75.0051165185869. Stream1 itl_p50_ms 16.60694449674338, tok/s
62.0815018303185. Coherence texts match the heading, ok and
same true, verdicts OK/OK. identity first_ok and alias_ok
true. mtp.enabled false. c12/serve_driver.log prints the same
C1 row as 11.02 / 11.23 / 11.47 / 90.89 / 169.32, prefill
177.43 / 175.43 / 173.05, health before done in 23 s, HEALTHY
card 0, cpu-run 246 s exit 0, BENCH_RC=0. The docker command
has FULL_DECODE_ONLY, FLASH_ATTN, mode 0, capture sizes
1,2,3,4,8, max_model_len 33792, gpu 0.90, max_num_seqs 4,
B70_EAGER_QK=0, B70_GGUF_REF=0, B70_XPU_CG_RECLAIM=500, and
SERVE_SO b3c15e14. At 12:55 the driver file is still 31 lines,
mtime 12:03:38, and still ends on the health-after wait. That
container name is not in docker ps. Pids 3278640 and 3278743
are not running. No HEALTHY-after line exists. The C7 11.11 ms
figure is an earlier journal citation, not a row in this json.
There is no second 35B ITL on the old graph.so in this entry.
reports/1C12_moe_decode2.md contains one U+2026.

Verdict: OK on identity, coherence, MTP off, the rounded
latency numbers, and health before. The raw prefill rates are
177.43, 175.43, and 173.05. WRONG to treat the serve as closed:
health after never printed. UNVERIFIED that post-health queued
behind H1. UNVERIFIED that the fair-layer gain did not move
35B ITL, because this is one bench. Not a Strata comparison.
The strata_dd block in the json is still the old citation.

Suggested fix: record a HEALTHY-after line before calling the
bench closed. Replace the ellipsis in 1C12_moe_decode2.md with
three ASCII dots.

### F120 -- 12:53 G4 r5, 1024 partner, end=26979

Claim: at 12:52:24 Running 1, Waiting 1, KV 90.3 percent. The
runner queued end=26979 at 12:52:33. Restores cached 0, then
device 1088. No load tail. Health after HEALTHY at 12:53:45.
Prompts 26400 and 1024, decode budget 580. Card 1 was taken by
H1 after this health check.

Evidence: logs/g4-t3-serve-c1-cycle-r5.txt line 249 is
12:52:24, prompt 3388.1 tok/s, Running 1, Waiting 1, KV 90.3%.
Line 251 is 12:52:33 end=26979 tail=867 full_keys=24. Line 254
at 12:52:34 is Running 2, Waiting 0, KV 29.0%. KV cache size
line is 27,870 tokens. Store bytes are present (max
1820098560) and kv_offload_load_bytes is absent.
logs/g4-t3-cycle-driver-c1-r5.txt: launch 12:49:21, port
18113, health before ok, READY 12:51:54, identity
hotschmoe-dev then the Ornith alias, block=1088. Turns 0-3
cached=0 device=0 host=0. Turns 4-7 cached=1088 device=1088
host=0 computed=441. All eight ok=False and cached_ok=False.
CYCLE_RESULT pass=False. Health after rc 0, HEALTHY card 1,
T3_CYCLE_DRIVER_EXIT 1 at 12:53:45. 26979 = 26400 + 580 - 1.
tail 867 = (26400 - 24*1088) + 579. kvoffload/t3_test.py
mtime 12:49:11 has active_n = [26400, 1024] and
active_decode = 580, which is before READY. The serve log
does not print 1024. The driver does not name an H1 lease
after the health check.

Verdict: OK on the 12:52:24 sample, end=26979, identity, and
health after. The partner was admitted by 12:52:34. WRONG as
a restore summary, same 4/4 split as F118, pass false. 
UNVERIFIED that load_bytes was 0. The 1024 length is the
source constant, not a log field. UNVERIFIED that H1 took
card 1 after the check. Not an ITL. Not a restore pass.

Suggested fix: quote pass=False and the 12:52:34 Running 2
line. Name the H1 lease line if card 1 moved.

### F121 -- hygiene at 12:54 UTC

Claim: no new staged binary since F117.

Evidence: git diff --cached --stat had line count 0. git
status still marks only
vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so
as a modified binary. t3_test.py is modified text. The fork
tip was not re-read after F117.

Verdict: OK on staging. The C12 report ellipsis is in F119.

Suggested fix: leave the push-allreduce .so unstaged. Do not
push.

## Pass 11 -- 2026-10-20 13:32 UTC

Cursor: headings after the 12:56 G2b milestone, including the 12:58
and 13:02 G4 entries that sit above that milestone. CPU only.

### F122 -- 13:01 H1c M3 FULL_DECODE_ONLY

Claim: both cards, port 18115, container vllm-fn-h1-m3,
FULL_DECODE_ONLY, FLASH_ATTN, max_len 33792, seqs 4, gpu 0.90,
capture sizes 1,2,3,4,8, B70_XPU_CG_RECLAIM=500, B70_EP=1.
EP _C.so sha e783539c, 481240 bytes. _C.ep.so prefix 55c154ee.
World size 1. Health before and after HEALTHY. Identity
hotschmoe-dev then qwen3.6-35b-a3b-ud-iq3-xxs. Greedy texts match
the C4 string, match_c4 true. C1 ITL p50 11.36 ms, p90 11.62,
p99 12.35, 89.50 tok/s, TTFT 142.82 ms, n=63. C2 15.31 ms / 64
tokens / 71.14 tok/s and 15.74 ms / 40 tokens / 65.97 tok/s.
GATE_RC=0. SERVE_RC=0. No error 44. KV 14.72 GiB, 630487 tokens,
18.66x. Capture keys 4,3 then 4,3,2,1. No xe-reset.

Evidence: logs/h1c-m3-window-r2.out and h1c-m3-bench.json.
Health-before rc 0 at 12:55:14 after a wait that names the C12
container. READY 12:56:40. Identity ids are hotschmoe-dev then
the qwen alias, first_ok and alias_ok true. Coherence ok, same,
verdicts OK/OK, both texts the C4 string. match_c4 true in
h1c-m3-greedy.json. C1 measured itl_p50_ms 11.359105934388936,
p90 11.61808893084526, p99 12.35144732054323, decode_tok_s
89.49583501515579, ttft_ms 142.81834405846894, completion_tokens
63, n_chunks 62. C2 stream0 itl_p50 15.310816001147032,
completion 64, tok/s 71.14396741041058, ttft 243.01356601063162.
Stream1 itl_p50 15.741902985610068, completion 40, tok/s
65.96864075053873, json ttft_ms 242.35916906036437. The window
prints ttft_ms=242.36. GATE_RC=0 and
SERVE_RC=0 at 12:56:44 and 12:56:53. Health-after HEALTHY both
cards, rc 0 at 12:57:23. Serve log: world_size=1, B70_EP=1,
B70_XPU_CG_RECLAIM=500, capture sizes [1,2,3,4,8], profile FULL
keys 4 and 3, real FULL keys 4, 3, 2, 1, zero "error 44" and
zero "Level Zero". KV line is 14.72 GiB and 630,487 tokens,
18.66x at 33,792. EP _C.so sha256 is
e783539c4fa49f774a115a816209201a189a78d8d2c051f5b56c30575eb2c870,
481240 bytes. _C.ep.so is the gguf plugin file, 8294752 bytes,
sha256 55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0.
The earlier capture-44 window.out now ends HEALTHY at 12:53:25
with done rc=1. 1.05 * 11.11 = 11.665 and 11.359 / 11.11 = 1.022.
The window text does not name C13. The greedy json has no
temperature field.

Verdict: OK on identity, coherence, health both sides, the
rounded ITL rows, the capture keys, the KV line, and the library
hashes. n=63 is completion_tokens; n_chunks is 62. The 11.11,
16.51, 16.65, and 121.40 figures are earlier journal citations.
The 121.40 run was eager DP=2 at max_len 2048 with the collective
counter on, so it is not a matched A/B with this M3. UNVERIFIED
that the queue also waited on C13, and UNVERIFIED that this
greedy call set temperature 0 and 16 tokens. Not a Strata
prefill result. This bench has no prefill section.

Suggested fix: quote n_chunks 62 next to n=63. Keep the eager
121.40 comparison labeled with max_len 2048.

### F123 -- 13:02 C13 one-layer prefill bench

Claim: _C.c13.so 2635416 bytes at 12:59. Health before and after
rc 0. Dequant GB/s IQ3_XXS 285.0 (old 255.8), IQ3_S 286.4,
IQ2_XXS 272.6, IQ2_XS 273.9, IQ2_S 279.0, Q2_0 270.3 (old 258.8),
IQ4_NL 281.0, copy 1147. Layer ms T=4096 fast 40.13, seq 40.13,
C10 bmm 45.12, mmax 109. T=12288 fast 56.97, seq 56.92, C10 bmm
66.16, mmax 287. Profile group 1.80, dq 8.94+7.11+7.15, pad 2.89,
bmm_gu 8.45, silu 1.50, bmm_dn 4.67, unpad 3.15, reduce 11.47.
Dispatch mismatches 0. GEMM-only 3775.74 ms and 3255 tok/s.
Scratch two-layer 9.375 GiB. failures=1 is the strict x1 check.
Synthetic fused max_rel 1.96.

Evidence: logs/c13-bench5.log. Those GB/s, layer, profile, and
DQ_OLD lines match the log. Copy is 1147.4. HEALTH_BEFORE:0 and
HEALTH_AFTER:0, both HEALTHY card 0. cpu-run exit 1, gpu-run
exit 1, VERDICT failures=1. The only FAIL line is "FAIL fast
real x1 max_rel=3.6478e-01". Fused real x1 is a STAT line,
max_rel 1.9925, frac_gate 1, sig_max_rel 0, lane values
1.3378e-04 and -1.3479e-04. BUDGET prints gemm_ms=3775.5 and
gemm_tok_s=3255. two_GiB=9.3750. GATE_THRESHOLD unchanged 5e-2.
LAYER lines say overlap=1. Fast 56.97 versus seq 56.92. Dispatch
mismatches=0. _C.c13.so mtime 12:59:33, 2635416 bytes, sha256
prefix ffb059d0494ace0d. Daily _C.so is still 10165216 bytes,
prefix b3c15e14. REAL blk.36 gshape=(2560, 640, 512), gate=18,
up=18, down=42.

Verdict: OK on the GB/s, the layer milliseconds, the profile,
health, the 9.375 GiB scratch, and failures=1 as a count.
WRONG that gemm time is 3775.74 ms; the log says 3775.5.
WRONG to pin the single FAIL on the fused strict row; that FAIL
is the fast x1 row. The fused lane magnitudes are about 1e-4 and
frac_gate is 1. Not an ITL. Not a model prefill.

Suggested fix: quote gemm_ms=3775.5 and the FAIL fast real x1
line. Keep 3255 tok/s labeled as the budget GEMM rate.

### F124 -- 12:58 and 13:02 G4 r6, one host hit

Claim: health-before took card 1 at 12:58:00. Turn 0 passed with
host=1519, device=0, greedy mismatch null. A load tail end=1519
local=0 keys=5 shares the second with queued end=25999. Turns
1-3 missed. Health after HEALTHY at 13:02:54.

Evidence: logs/g4-t3-cycle-driver-c1-r6.txt launches 12:58:00,
health before ok, READY 13:00:45, identity hotschmoe-dev then
the Ornith alias. Turn 0 ok=True, host=1519, device=0,
computed_ok and cached_ok true, mismatch_at null, ttft 36.943 s.
Turns 1-3 are host 0 device 0. Turn 4 is host 1088 device 0.
Turns 5-7 are host 0 device 1088. All of 1-7 are ok=False.
CYCLE_RESULT pass=False. The next line is
"error name 'active_n' is not defined". The r6 json repeats that
error and pass false. Serve log 13:02:14 has queued tail
end=25999 tail=975 full_keys=23 and load tail end=1519 local=0
keys=5. The 13:02:05 sample is Running 1 Waiting 2. Health after
rc 0, HEALTHY card 1, driver exit 1 at 13:02:54.
25999 = 24000 + 2000 - 1.

Verdict: OK on the launch stamp, turn 0's client gate, the
same-second load and end=25999, identity, and health after.
The 12:58 "not ready" note matches this launch before READY.
Turns 4-7 also missed, and the json records the NameError.
Device delta on the passing turn is 0. Cycle pass is false.
Not an ITL.

Suggested fix: quote pass=False, the NameError, and turns 4-7.
Do not read device 0 as a device restore.

### F125 -- 13:15 G4 r7, long prefill then one load

Claim: READY 13:06:31. small_n=522 big_n=0. From 13:07:09 to
13:08:09 Running 1 Waiting 2, KV 83.9 to 93.5 percent. At
13:08:12 stored end=26799 and loaded tail end=1519 local=0
keys=5. Turn 0 host 1519 device 0 greedy match. Turns 1-7
missed as host 0 or host/device 1088. Health after rc 0 at
13:08:51. NameError active_n. No xe-reset.

Evidence: logs/g4-t3-cycle-driver-c1-r7.txt. READY 13:06:31.
small_n=522 big_n=0. Turn 0 ok=True host=1519 device=0. Turns
1-3 host 0 device 0. Turn 4 host 1088 device 0. Turns 5-7 host
0 device 1088. All of 1-7 ok=False. CYCLE_RESULT pass=False,
then the active_n NameError. Health after rc 0, HEALTHY card 1,
exit 1 at 13:08:51. Serve samples: 13:07:09 is Running 1
Waiting 1 KV 93.5%. Waiting 2 starts at 13:07:19. Later KV
values are 93.5, 83.9, 87.1, and 90.3. At 13:08:12 the log says
"queued tail" end=26799 tail=687 full_keys=24 and "load tail"
end=1519 local=0 keys=5. No xe-reset line. 26799 = 24000 + 2800
- 1. Identity is hotschmoe-dev then Ornith.

Verdict: OK on READY, the active counts, turn 0, the 13:08:12
pair, the NameError, and health. WRONG that the whole
13:07:09-13:08:09 window is Waiting 2; the 13:07:09 sample is
Waiting 1. WRONG that every miss is host 0 or host/device 1088;
turn 4 is host 1088 and device 0. Cycle pass is false. The
passing turn still has device delta 0.

Suggested fix: split the 13:07:09 sample from the Waiting 2
stretch, and quote turn 4's host 1088 device 0.

### F126 -- 13:22 G4 r8, four first-visit loads

Claim: health before ok and health after rc 0. Actives 17 and 8
at the start, 287 and 296 at the end, streams_alive true.
Running 3, KV 41.9 percent. Turns 0-3 load tail end=1519
local=1088 keys=4, host 431, device 1088, greedy match. Turns
4-7 host 0, device 1088, computed 441. Launch short prompt 256.

Evidence: logs/g4-t3-cycle-driver-c1-r8.txt launch line
short_prompt=256, health before ok, identity hotschmoe-dev then
Ornith. small_n=17 big_n=8. Turns 0-3 ok=True host=431
device=1088 cached=1519 computed=10. Turns 4-7 ok=False host=0
device=1088 computed=441. g4-t3-cycle-c1-r8.json cycle pass
false, streams_alive true, active_tokens [287, 296]. All eight
turns have greedy_match true, including the four failures.
Serve log has four "load tail" lines end=1519 local=1088
keys=4, then queued end=1560. Engine 13:19:14 Running 3 Waiting
0 KV 41.9%. Health after rc 0, HEALTHY card 1, exit 1 at
13:19:50. t3_test.py mtime 13:22:25 is after this exit, so the
4000-token constant in the current source is not this run's
saved file.

Verdict: OK on the 4/4 split, the load lines, the active token
counts, identity, and health. Greedy is also true on the failed
second visits. Cycle pass is false. Not an ITL.

Suggested fix: say greedy_match stayed true on turns 4-7 while
host_ok and cached_ok were false.

### F127 -- 13:27 G4 r9, cache flags pass, one greedy miss

Claim: eight load tails, local=1088 keys=4. Turns 0-3 host 431
device 1088 cached 1519 computed 10. Turns 4-7 host 472 device
1088 cached 1560 computed 10. Greedy exact on turns 0-6. Turn 7
matched 29 of 32, then 40822 against gold 14308. TTFT 0.126 to
0.139 s. active tokens 289 and 298. streams_alive true. Running
3, KV 41.9 percent. Health before ok, health after rc 0.
Second visit adds the 32-token reply and 9 new tokens.

Evidence: logs/g4-t3-cycle-driver-c1-r9.txt launch 13:22:45
short_prompt=256 extended=1, READY 13:25:19, identity
hotschmoe-dev then Ornith. small_n=17 big_n=8 at restore start.
Turns 0-6 ok=True. Turn 7 ok=False mismatch_at=29, host=472,
device=1088, cached_ok and computed_ok true. Cached and computed
match the heading. TTFT values run from 0.126 to 0.139. json
active_tokens [289, 298], streams_alive true, pass false.
Turn 7 restored_ids at the mismatch is 40822 and gold_ids is
14308. The next two also differ: 198 vs 25033, and 14306 vs
198. Gold prefix repeats 79593, 198, 14306, 14307, 14308.
Gold index 30 is 25033. Serve log has eight load tails, four
end=1519 and four end=1560, all local=1088 keys=4. Engine
13:25:45 Running 3 KV 41.9%. Prompt 1570 - 1529 - 32 = 9.
Health after rc 0, HEALTHY card 1, exit 1 at 13:26:22.
Source mtime 13:22:25, before this launch, contains
active_prompt=256 and active_decode=4000.

Verdict: OK on the cache flags, the eight loads, the token ids,
the TTFT span, identity, and health. CYCLE_RESULT pass=False
because turn 7 greedy fails. Host and cached flags are true on
that turn too. The 5-token period covers the prefix and index
29. Index 30 is outside that period. Not a whole-cycle pass.

Suggested fix: keep pass=False beside the cache-flag sentence,
and quote the two tokens after 40822.

### F128 -- C12 35B driver health-after closed

Claim: at pass 10 the C12 driver still had no HEALTHY-after line.

Evidence: c12/serve_driver.log is now 38 lines, mtime 12:54:44
UTC. After the old wait it records health probe card 0 OK,
HEALTHY card 0, gpu-run done in 13 s exit 0, SERVE_DONE,
DRIVER_RC=0. BENCH_RC=0 is unchanged above that.

Verdict: OK that the driver now has health after. F119's
"still 31 lines" was the earlier read. This does not change the
prefill rates or make a Strata comparison.

Suggested fix: cite SERVE_DONE and DRIVER_RC=0 when calling that
bench closed.

### F129 -- hygiene and the new host waits

Claim: no staged binary. New plugin edits do not put a queue
wait inside the EP capture path.

Evidence: git diff --cached --stat line count 0. Fork
b70/flashnext is 0a6170da8c, ahead 18. The only modified binary
in git status is the pre-existing push-allreduce .so. No
secret-like names. moe_ep.cpp returns before s.q1->wait() and
q0().wait() when cap_on or stream_recording() is set. It does
printf and fflush when an implicit capture starts. New
torch.xpu.synchronize calls in the diff are in bench_copy.py
and test_qsa_batch.py. batch.py adds ok.item() next to an
existing length.item().

Verdict: OK on staging. The EP waits are outside the capture
return. The new .item() is a host read in the QSA batch helper.
This pass did not trace that helper into a graph replay.

Suggested fix: keep device .item() off a captured replay path.
Leave the push-allreduce .so unstaged.

### F130 -- 13:33 G4 FULL_DECODE_ONLY T2

Claim: card 1, port 18116, 45 blocks, max_len 32768, max_num_seqs
2, FULL_DECODE_ONLY, capture sizes 1,2,3,4,8. Capture FULL 2/2 in
1 s. pass=True. host 3063, device 0, computed 18, greedy and text
match. Load tail end=3063 local=0 keys=6. Load 131235840 bytes in
0.009645 s, 13.607 GB/s. Restored TTFT 0.120 s, control 0.412 s.
Health after HEALTHY in 12 s. The 45-block pool held 39852 tokens.
The profiler wanted 244 blocks. Ornith text
max_position_embeddings is 262144.

Evidence: logs/g4-t2-graph-driver-c1.txt. Launch 13:27:41, port
18116, flags .so, health before ok, READY 13:30:15. Identity
hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.
pass=True, host=3063, device=0, computed=18, greedy=True, client
rc 0. Health after rc 0, HEALTHY card 1, gpu-run 12 s exit 0,
T2_GRAPH_DRIVER_EXIT 0 at 13:31:19. json restored ttft_s
0.12025012902449816, control_ttft_s 0.411765166092664,
load_bytes 131235840, load_time_s 0.009644792, text_match true,
greedy_match true, cached_ok true. 131235840 / 0.009644792 =
13.607 GB/s using 1e9. Serve line 1 is blocks=45 max_len=32768
seqs=2 mode=FULL_DECODE_ONLY. Capture bars are 2/2, finished in
1 sec, twice. CUDAGraphMode.FULL_DECODE_ONLY: (2, 0). Capture
sizes in the command are [1,2,3,4,8]. Load tail at 13:30:42 is
end=3063 local=0 keys=6. KV line is 39,852 tokens. Override is
num_gpu_blocks=244 replaced by 45. config.json for this
checkpoint has max_position_embeddings 262144. That string is
not in the serve log. No error 44 and no xe-reset in the serve
log. docker ps at the addendum showed vllm-fn-g4-kv-c1 up.

Verdict: OK on the client gate, identity, the load bytes, the
TTFT rounding, the KV lines, and health. device_delta is 0, so
the device counter did not move. The load itself is non-zero.
Only two graphs were captured. The 262144 figure is the
checkpoint config, not a serve-log field. Not a cycle pass and
not a Strata comparison.

Suggested fix: keep device=0 next to pass=True. Say the capture
that ran was 2/2 even though the size list includes 8.

## Pass 12 -- 2026-10-20 14:08 UTC

Cursor: the 13:42 G4 heading after the pass 11 addendum. CPU only.

### F131 -- 13:42 G4 Ornith 200063-token restore

Claim: card 1, port 18117, PIECEWISE, max_len 204800, profiled
pool, 4 slots, 32 GiB host, P1=200000. KV 257312 tokens, 1.26x
at 204800. pass=True. host 200063, device 0, computed 18, greedy
and text match. Load tail end=200063 local=0 keys=187. Load
4164321280 bytes in 0.299164 s = 13.920 GB/s. Restored TTFT
0.833 s, control 84.788 s. Health after HEALTHY in 12 s.
MemAvailable minimum 78051432 kB. The copy is under a 0.42 s
target. This is Ornith, not Flash-Next.

Evidence: logs/g4-t4-driver-c1.txt. Launch 13:34:31, port 18117,
max_len=204800, p1=200000, blocks=profiled, flags .so. Identity
hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.
pass=True, host=200063, device=0, computed=18, greedy=True,
client rc 0. Control ttft 84.788 s, prompt 200081. Restored
ttft 0.833 s, cached 200063, computed 18. Turn 1 ttft 84.339 s,
prompt 200000. Fillers B and C and the cold request are prompt
30464. Health after rc 0, HEALTHY card 1, gpu-run 12 s exit 0,
T4_DRIVER_EXIT 0 at 13:40:58. json load_time_s 0.299163593,
load_bytes 4164321280, text_match true, greedy_match true,
expect.replay 0, replay_ok true. 4164321280 / 0.299163593 =
13.920 GB/s using 1e9. Serve banner is PIECEWISE, blocks=0,
slots=4, seqs=2, cpu_bytes=34359738368. KV line is 257,312
tokens, 1.26x at 204,800. Load tail 13:40:19 end=200063 local=0
keys=187. Park evict 13:40:26 end=30464 n=1. Serve log also has
end=200144, which is 200081+64-1. floor(200063/1088)=183 and
200063-183*1088=959. g4-t4-mem-c1.txt is MemAvailable kB from
the launcher awk, minimum 78051432. 78051432/1024/1024=74.43.
No xe-reset in the driver. 0.42 is not in BRIEF.md. 0.299 is
under 0.42. The restored json ttft_s is 0.8326086619636044.

Verdict: OK on identity, the client gate, the load bytes, the
KV line, the evict time, health, and the MemAvailable minimum.
device_delta is 0. The mode is PIECEWISE, not the FULL capture
in F130. The 183/959 split is arithmetic on block 1088; the
printed key count is 187. The 0.42 s figure is the report's
target, not a BRIEF number, and the measured copy is under it.
Not a Flash-Next 200K result.

Suggested fix: keep device=0 and PIECEWISE next to pass=True.
Quote keys=187 separately from the 183-block remainder.

### F132 -- hygiene at 14:08 UTC

Claim: no new staged binary since F129.

Evidence: git diff --cached --stat line count 0. Fork
b70/flashnext is still 0a6170da8c, ahead 18. git status still
marks only the pre-existing push-allreduce .so as a modified
binary. The 13:42 heading does not name a new plugin source.

Verdict: OK on staging.

Suggested fix: leave the push-allreduce .so unstaged. Do not
push.

## Pass 13 -- 2026-10-20 14:50 UTC

Cursor: JOURNAL.md through the 14:45 C7b retry heading (4059 lines
at the read). The 14:45 eager serve was still loading. No bench JSON.

### F133 -- 14:36 Flash-Next eager window config

Claim: fork 0a6170da8c, locks empty, MemAvailable 118492772 kB,
hotschmoe-dd inactive, frontdoor left on 18080, no P2P, no B70_QSA.
Window logs/c7-fn-window.sh is DP=2 TP=1, expert parallel,
enforce-eager, language-model-only, max_model_len 2048, docker
memory 100g, SYCL_CACHE_PERSISTENT=0, B70_GGUF_SO=_C.graph.so at
10165216 bytes. Greedy temperature 0, enable_thinking false, refs
323 and 667. Watchdog stops the container under 8 GiB. Result line
is only "window open".

Evidence: fork log -1 is 0a6170da8c, ahead 18. _C.graph.so is
10165216 bytes, sha256 b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979,
mtime 11:22:28 UTC. The script matches that serve shape, the greedy
body, and the 8388608 kB watchdog. It exits 2 if either gpu.lock
owner file is non-empty, and it reached health and serve.
c7b-fn-window-ram.out: collective health 14:36:05, HEALTH_RC 0,
MEM_AVAILABLE_KB 118396268, serve 14:36:44. The RSS trace at
14:36:05 is avail_kb=118485716. 118492772 is only in the journal
(96504 kB above the window line, 7056 kB above the RSS line).
The health file was read before the retry overwrote it:
COLLECTIVE_HEALTH_OK world_size=2 p2p=0, HEALTHY, gpu-run 39s
exit 0, cards 0 and 1. A 14:42 unit read showed hotschmoe-dd
inactive, frontdoor active, and 18080 listening. The preserved
serve command has no B70_QSA and no CCL_TOPO_P2P_ACCESS=1.
SYCL_CACHE_PERSISTENT=1 from the harness and =0 from the window
are both on that docker line; the window flag is the later one.
served-model-name lists hotschmoe-dev first. There is no READY
line and no c7-fn-greedy.json.

Verdict: OK on the script, the library, the fork, health-before,
and the later unit read. UNVERIFIED the exact 118492772 kB.
The prepared greedy did not run. This heading is not an ITL.

Suggested fix: quote MEM_AVAILABLE_KB from the window log. Pass
SYCL_CACHE_PERSISTENT once.

### F134 -- 14:41 load abort

Claim: MEM_AVAILABLE_KB 118396268 at 14:36:44. Engine
0.32.1rc1.dev94+g5fd62688f, Qwen4ExpForConditionalGeneration, DP=2
EP linear 256/512. Placed 1100 of 1224 tensors at 14:40:47. No
host-table line, no Python traceback. ABORT_LOW_RAM avail_kb=7309964
at 14:40:53. SERVE_RC 137. No new kernel oom-kill after the 11:17
one. MemAvailable returned to about 118 GiB. No xe-reset. O3 then
held card 0 for a 73 s profile and returned HEALTHY. Shard 1 is
43.81 GiB (144 expert tensors 39.97 GiB, 887 other 3.72 GiB, 192
shared-expert tensors 0.11 GiB, largest 0.439 GiB). The dim-0 slice
kept the full expert Storage. Docker cgroup was 84.59 GiB at 1000
tensors and AnonPages was 85399492 kB. Not a coherence result. The
PLE view was not what filled the host.

Evidence: c7b-fn-window-ram.out has those MEM, ABORT, and SERVE_RC
lines. 7309964 is under 8388608. c7b-fn-serve-ram.log has the engine
string twice, Qwen4ExpForConditionalGeneration, tensors=1224 twice,
and both workers "placed 1100 tensors" at 14:40:47. "Expert
parallelism" / "strategy: linear" / "256/512" occurs once (the DP0
line). Traceback, host-table, ple_, xe-reset, error 40, error 44,
and Out of memory are absent. gpu-run done in 265s exit 137.
kern.log's last oom-kill is 2026-10-20 11:17:41. journalctl -k
since 14:00 has no oom line. The RSS trace ends at 14:41:09 with
avail_kb=119235164, which is 113.71 GiB, then SAMPLER_DONE.
The 14:42:13 lock file was "O3 pid=3327438 since=2026-10-20 14:41:09
cmd: ./run_prof.sh". A C7 session tool stdout at 14:42:23 says
"xpu-health: HEALTHY (cards 0)" and "gpu-run: [O3] done in 73s
(exit 0)". That stdout is not copied under logs/. It names card 0
only. 144+887+192=1223, not the log's 1224 tensors. 39.97+3.72+0.11
=43.80, not 43.81. Those GiB lines are not in the serve log, the
RSS log, or the window log. RSS docker lines near the 1000-tensor
mark (14:40:27) are 80.16GiB at 14:40:32 and 84.52GiB at 14:40:44.
AnonPages samples are 81293680 and 85921508. 85399492 is not one
of them. mapping.py's clone edit is mtime 14:42:59, after the
14:41:10 serve exit. The committed function returned packed[start:end].
The 16 GiB Strata sentence is not in this window's logs.

Verdict: OK on the abort, the placement counter, the missing
PLE line, the missing traceback, the 11:17 oom being the last one,
the O3 card-0 tool stdout, and "not a coherence result". Live
/v1/models did not run. WRONG on "about 118 GiB" and on 84.59 GiB
at 1000 tensors. UNVERIFIED the shard table, 85399492, and the
Strata 16 GiB sentence. The abort script has no health-after.
Device delta is not involved. Not an ITL and not a Strata match.

Suggested fix: quote the RSS line 119235164 kB and the 84.52GiB
docker sample with its timestamp. Save the shard tally in a file.
The next window's own health-before is the both-card check.

### F135 -- 14:45 clone retry, still loading

Claim: slice_expert_tensor clones a partial expert slice and
ep_size 1 still returns the original tensor. CPU check
EP_SLICE_STORAGE_OK. Not committed. MemAvailable 118622492 kB.
Both locks empty. Same eager window. Leases requested, window open.

Evidence: the uncommitted mapping.py clones when numel or data_ptr
differs, so a full dim-0 slice of a contiguous owner is returned
as the same tensor. git status shows the file modified. Fork tip
is still 0a6170da8c. /tmp/c7b-ep-slice-test.py (mtime 14:44:50)
prints EP_SLICE_STORAGE_OK and imports /fork/vllm/.../mapping.py.
No stdout file was saved beside it. The venv python is a broken
symlink, so this pass did not re-run the import. The new RSS
trace at 14:45:27 is avail_kb=118622460, 32 kB under the journal
figure. c7b-fn-window.out, read at 14:49, says health 14:45:27,
HEALTH_RC 0, MEM_AVAILABLE_KB 118262704, serve 14:46:02, and has
no READY line. 118262704 is over the script's 104857600 gate.
The health file now on disk is this launch: COLLECTIVE_HEALTH_OK
p2p=0, HEALTHY, gpu-run 35s exit 0, cards 0 and 1. It replaced
the 14:36 health file. At 14:46 both lock files were the C7
docker run. No greedy JSON.

Verdict: OK that the retry launched, passed the logged RAM gate,
and had health-before. UNVERIFIED the exact 118622492 kB and that
a process printed EP_SLICE_STORAGE_OK. Not an identity check and
not a coherence result. The serve was still loading at 14:49.

Suggested fix: save the CPU test stdout. Quote the window's
MEM_AVAILABLE_KB. Leave the result for the log that contains READY
or the abort line.

### F136 -- uncommitted expert clone and EP map

Claim: the 14:45 heading describes the clone. The working tree
also adds ep_register_map, which the headings do not mention.

Evidence: mapping.py clone is load-time host code, not a captured
op. A partial slice gets its own storage. ep_size 1 does not clone,
so a full-range view of a larger storage still pins that storage.
ep_register_map (ops.py, bind.cpp, moe_ep.cpp, ep_kernels.cpp)
checks xpu, int32, contiguous, equal non-empty length, and layer
range, then stores the two data pointers. The dispatch kernel
writes ids0[i]=map0[id] and peer_ids[i]=map1[id] when
0<=id<nexp, else -1. It does not check that the looked-up slot
is -1 or inside the local expert count, and both maps may be
non-negative for one id. enqueue_moe throws on the host if exactly
one pointer is set, before the launch. The warmup call passes
nullptr. No new queue wait was added on the forward path. The
pointers are not copied into State-owned memory.

Verdict: OK that the clone matches the heading for a contiguous
owner, and that the new forward path adds no replay host sync.
The map slot values are missing a bounds check, and a temporary
map tensor would dangle, including inside a later capture.

Suggested fix: copy both int32 tables into allocations owned by
State at register time. Reject a value that is not -1 and not in
the local expert range, and reject a token that is active on both
sides. Clone on ep_size 1 when the storage is larger than the tensor.

### F137 -- hygiene at 14:50 UTC

Claim: no new staged binary. The 14:33 commit 3dcfca6c is the
tree that the later headings sit on.

Evidence: git diff --cached --stat line count 0. git status
porcelain was 14 lines. The only modified binary is the
pre-existing vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so.
Untracked paths checked are small (the largest file seen was
postprocess_plans.json at 12949 bytes). No secret-like names.
Commit 3dcfca6c is 40 files. None of those paths present in the
worktree were over 1 MB. Fork remains 0a6170da8c, ahead 18, with
the uncommitted mapping.py edit. The EP sources are unstaged text.
_C.graph.so is still the b3c15e14 binary from F133.

Verdict: OK on staging. The commit message rounds figures already
audited in earlier passes. This pass does not reopen them.

Suggested fix: leave the push-allreduce .so unstaged. Do not
commit the map path until the slot check and the owned copy exist.

### F138 -- 14:45 retry died on XPU OOM after placement

Claim: the 14:45 heading says the window is open. It does not claim
a greedy result. The log then closed.

Evidence: c7b-fn-window.out now ends "serve exited before ready",
stop 14:49:26, SERVE_RC 1, WINDOW_DONE. No ABORT_LOW_RAM line.
c7-fn-serve.log is this launch (the 14:36 copy is c7b-fn-serve-ram.log).
Both workers placed 1224 tensors at 14:48:51. The kinds dict sums
to 1224 (14 types), skipped_mtp 0, q8_dequant 0, arch qwen4exp.
Both log "host table per_layer_token_embd.weight memmap view
bytes=28800138240" and "installed PLE hash on 1 modules".
28800138240 bytes is 26.822 GiB. serve_loader.py:296 warns twice
that the NumPy array is not writable. Then both workers raise
torch.OutOfMemoryError in embedding.py:77 at packed.to(device):
tried 418.00 MiB. GPU 0 free 200.77 MiB of 31.89 GiB, PyTorch
allocated 31.10 GiB, reserved-unallocated 360.06 MiB. GPU 1 free
202.14 MiB, same allocated and reserved figures. EngineCore wraps
that as "Engine core initialization failed" and gpu-run exits 1
in 199s. The RSS minimum is 40042444 kB at 14:49:08, while the
workers were already shrinking. The lowest sample before that
drop is 43846676 kB at 14:48:44. The trace returns to 118916024
kB at 14:49:20. No READY chat and no greedy JSON. The window
script does not run health after the exit. At 14:51 card 0 was
held by O3 (pid 3341460, ./run_mb.sh) and card 1 was empty.
No xe-reset call is in the window script.

Verdict: the 14:45 "window open" line was true when written.
The run is not a coherence result and not an ITL. Identity
(/v1/models) did not answer. Placement and the PLE memmap line
did run. The failure is XPU OOM on a 418 MiB copy, not the host
watchdog from F134. Health-after is missing.

Suggested fix: preserve this serve log before the next window
truncates c7-fn-serve.log. Run both-card health after the exit
when the leases are free. The 418 MiB packed.to(device) needs
free GPU memory, or it has to stay a host view.






