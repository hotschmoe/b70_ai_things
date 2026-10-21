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

## Pass 14 -- 2026-10-20 15:34 UTC

CPU only. No GPU, no commit. Cursor is the journal after the 14:53
G2b milestone: the 14:49 through 15:13 C7b headings, the 14:55 H1d
heading, and the 15:40 O3 heading. The 15:13 graph window had already
exited when this pass read it.

### F139 -- 14:49 C7b heading repeats the XPU OOM

Claim: the PLE memmap held, then both workers OOM'd on the bf16
placeholder. Host MemAvailable bottomed at 40042444 kB (anon
13164496 kB). No kernel oom-kill. Packed experts are about 20 GiB
per rank. O3 used card 0 for 75 s and left it HEALTHY, then started
another card-0 run at 14:53:52.

Evidence: the XPU OOM, the 418 MiB packed.to(device), the 31.10 GiB
allocated figure, SERVE_RC 1, and the 40042444 kB sample are the
F138 logs (c7b-fn-serve-xpuoom.log, c7b-rss-prev.log). F138 already
separates that 14:49:08 sample from the load-time low, 43846676 kB
at 14:48:44. journalctl -k since 14:49:00 has no "oom" line
(oom_count 0). The logs directory has no "20 GiB" expert total.
The only hit is a comment in c7-fn-window.sh about a host watermark.
No log contains "done in 75" or "14:53:52".

Verdict: OK that this heading is the F138 XPU OOM, and that the
40042444 / 13164496 pair is the trace minimum. WRONG as a placement
watermark. UNVERIFIED the 20 GiB expert figure, the 75 s duration,
and the 14:53:52 start. Not an ITL.

Suggested fix: quote 43846676 kB at 14:48:44 as the pre-teardown
low. Leave the 20 GiB and the O3 times out until a log has them.

### F140 -- 14:54 open, 15:02 KV budget short

Claim: placeholders dropped. Both ranks placed 1224. PLE memmap
28800138240. Model load about 125 s and 24.9 GiB. Available KV
-1.53 GiB. util 0.90 requests about 28.70 GiB and the non-KV
footprint is about 30.23 GiB. No XPU OOM. Host peak AnonPages
51190084 / MemAvailable 45549788 around 800 tensors. Steady
AnonPages 12252612 and workers summed 9206088 kB (about 4.4 GiB
each). The window parent died and timeout pid 3345213 kept the
serve. Pre-serve MemAvailable 118062644 (14:54) and 118704776
(15:02).

Evidence: c7b-fn-window-kvshort.out is four lines: health 14:54:56
HEALTH_RC 0, MEM_AVAILABLE_KB 118704776, serve 14:55:31. It has no
SERVE_RC and no WINDOW_DONE. c7b-fn-serve-kvshort.log: both workers
placed 1224 at 14:58:31, both memmap bytes=28800138240, model load
24.9 GiB in 124.284167 s and 125.110653 s, Available KV cache
memory: -1.53 GiB once, ValueError "No available memory for the
cache blocks" four times, OutOfMemory 0, xe-reset 0. gpu-run tag
is [C7], done in 259s exit 1. Non-default args: enforce_eager,
served_model_name hotschmoe-dev then qwen3.8-flash-next-iq3-xxs,
data_parallel_size 2, enable_expert_parallel, gpu_memory_utilization
0.9, max_model_len 2048, max_num_seqs 2. max_num_batched_tokens is
absent. c7b-rss-run3.log starts at 14:58:45, after placement. It
has 14:59:25 avail_kb=48461628 anon_kb=12252612 workers_n_rsskb=2
9206088. It does not contain 51190084 or 45549788. 9206088/2 is
4.39 GiB. 0.90*31.89 = 28.701 and 28.701+1.53 = 30.231, using the
F138 capacity 31.89, which this log does not reprint. The later
eager log reports a 30.3 GiB device total. No log contains
118062644 or pid 3345213.

Verdict: OK on placement, the memmap, 24.9 GiB, the two load times,
-1.53 GiB, the ValueError on both engines, no XPU OOM string, the
steady anon/RSS pair, and window MEM 118704776. The 28.70 / 30.23
pair is that reconstruction, not a new measurement. UNVERIFIED
118062644, 51190084, 45549788, and pid 3345213. The short window
file is consistent with a parent that never wrote the stop lines.
Not an ITL. No greedy. No health-after.

Suggested fix: preserve the stop lines in the window file. Sample
RSS from process start if the placement peak is going to be quoted.
Say that 31.89 GiB is the prior OOM total.

### F141 -- 14:55 H1d EP sidecar, second library

Claim: first link sha 308cd69d, 495104 bytes, exit 0 in 45 s, stored
the caller pointer. Second link sha 5a0a047c, 499200 bytes, exit 0
in 39 s, owns both maps, rejects a bad, duplicate, both, or neither
slot, and the kernel clamps a bad slot to -1. _C.ep.so stays
55c154ee / 8294752. The graph library exports b70_moe_layer and
b70_reserve_decode_queue. test_placement.py printed PLACEMENT_OK
and PLE_GUARD_OK. Profile counts and MemAvailable 118655020 are as
written. The serve script was not launched. Phase B has not started.

Evidence: h1d-ep-build.log exit 0 in 45 s, size 495104, and does
not print a sha. h1d-ep-build2.log exit 0 in 39 s, size 499200.
The file now on disk is 499200 bytes, sha256
5a0a047c67fc192588007e93b69c22a8ed9c781d2a891056dff200cceab067ac,
mtime 14:55:20. _C.ep.so is 8294752 bytes, sha256
55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0,
mtime 08:35. moe_ep.cpp ep_register_map mallocs own0/own1, copies
the caller pointers to host vectors, rejects nexp outside 1..4096,
rejects a slot outside the local pack, a duplicate, or on0==on1,
requires the counts to match e0 and e1, then copies the host
vectors to the owned device pointers. The .wait() calls are in
register. ep_kernels.cpp dispatch clamps a looked-up slot into
[0, n0) and [0, e1), else -1. The forward tail returns before
q0().wait() when cap_on or recording. test_placement.py contains
the two print strings. No stdout file under logs has PLACEMENT_OK
or PLE_GUARD_OK. No h1 serve log. h1d-gpu-wait.log shows the wait,
then C7b holding both cards at 15:04:52. A memcpy throw before the
free can leak the new device pointers. That is load-time, not the
F136 dangling map.

Verdict: OK on the second library identity, the build exits, the
source fix for F136, and the unchanged _C.ep.so hash. UNVERIFIED
the overwritten first sha 308cd69d (the size 495104 is in the
log). UNVERIFIED that a process printed PLACEMENT_OK, and
UNVERIFIED the profile integers and 118655020. Phase B has no
greedy and no ITL in these logs. The symbol check on the graph
library is F144, because that file was replaced at 15:19.

Suggested fix: save the test stdout and the first-link sha in the
build log. Free own0/own1 if the memcpy throws. Do not serve this
sidecar until Phase B has identity and coherence.

### F142 -- 15:10 eager DP=2 coherent at util 0.98

Claim: health-before OK, MemAvailable 118088084, model load 24.9 GiB,
Available KV 0.89 GiB, KV cache 6144 tokens, concurrency 3.00x,
kv lcm block size 1600. HTTP ready 15:09:33. Greedy temperature 0,
enable_thinking false, prompts 25 tokens, texts "323" and "667",
4 completion tokens, both match. SERVE_RC 0. Served names
hotschmoe-dev then qwen3.8-flash-next-iq3-xxs. Ready RSS and the
15:08:08 anon peak are as written. This is the first coherent
Flash-Next IQ3_XXS eager serve.

Evidence: c7b-fn-eager98-window.out has those health, READY, CHAT,
FN_GREEDY_DONE, and SERVE_RC 0 lines. The 15:03 heading's
118285456 is not in the logs. c7b-eager98-greedy.json matches both
rows. The JSON has no temperature field. The script sends model
hotschmoe-dev, temperature 0, enable_thinking false. The ready
curl discards /v1/models (`>/dev/null`). Serve log non-default
args list hotschmoe-dev first, enforce_eager, util 0.98,
max_num_batched_tokens 2048, max_num_seqs 2, DP=2, expert parallel.
Engine v0.32.1rc1.dev94+g5fd62688f. Load 24.9 GiB in 124.138689 s
and 124.665807 s. Available KV 0.89 GiB once. KV size 6,144 and
3.00x twice. "kv lcm block sizes 1600" twice. Also "block size to
64 for FLASH_ATTN" and "attention block size to 1600 tokens".
Application startup complete. OutOfMemory 0, error 44 0, xe-reset 0.
gpu-run [C7b] done in 270s exit 0. c7b-coll-health-eager98.txt:
COLLECTIVE_HEALTH_OK p2p=0, HEALTHY, gpu-run 35s exit 0. No
health-after command in the window. c7b-rss-eager98-ready.txt
matches 47736880, 12847624, RSS 4562356 and 4645336, PSS 4316028
and 4398933, Anonymous 4136256 and 4219228, Shared_Clean 424892
and 425088. c7b-rss-eager98.log peak anon_kb=53938904 at 15:08:08.
12847624 kB is 12.25 GiB. 53938904 kB is 51.44 GiB.

Verdict: OK on health-before, the configured name order, the chat
accepting hotschmoe-dev, the two matches, the token counts,
SERVE_RC 0, the KV lines, and the RSS numbers. UNVERIFIED the live
/v1/models body and health-after. The 15:03 MemAvailable is
UNVERIFIED. Not an ITL. Not a Strata comparison. The 15:02 run is
not a matched speed pair: util 0.9 versus 0.98, and
max_num_batched_tokens 2048 is only in the 0.98 args. Same
checkpoint path and the same two prompt strings. The 0.9 run never
answered.

Suggested fix: save the /v1/models body. Run collective health
after the window exits. Keep this result labeled coherence.

### F143 -- 15:13 graph window died on a capture D2H

Claim: the 15:13 heading says the FULL_DECODE_ONLY window is open,
MemAvailable 117855916. It does not claim a greedy result.

Evidence: c7b-fn-graph-window.out: health 15:13:23 HEALTH_RC 0,
MEM_AVAILABLE_KB 118052504 (the journal figure is not in the file),
serve 15:13:58, "serve exited before ready", stop 15:18:34,
SERVE_RC 1, WINDOW_DONE. A same-sized copy is
c7b-fn-graph-window-ple-d2h.out and c7b-graph-serve-ple-d2h.log.
Health file: COLLECTIVE_HEALTH_OK p2p=0, HEALTHY, gpu-run 35s
exit 0. No health-after. Serve log: both workers placed 1224,
memmap 28800138240, load 24.9 GiB in 143.359477 s and 143.529377 s.
Capturing CUDA graphs (FULL) 0/2 twice. "XPU kernel not compiled
for this config" four times. Available KV count 0. OutOfMemory 0.
xe-reset 0. error: 44 count 8. At 15:18:14 Worker_DP1 ple.forward
(model.py line 433 in the traceback) is `input_ids_cpu =
input_ids.to("cpu")`, then RuntimeError level_zero error 44
(UR_RESULT_ERROR_UNSUPPORTED_FEATURE). Both workers show that
frame. Engine core init failed. gpu-run [C7b] done in 272s exit 1.
Args: served name hotschmoe-dev first, DP=2, EP, util 0.98,
max_num_batched_tokens 2048, max_num_seqs 4, FLASH_ATTN,
cudagraph_mode FULL_DECODE_ONLY, capture sizes 1,2,3,4,8,
enforce_eager false. o3/fair_quiet.log records this container
holding card 0 since 15:13:58, so the window had started before
the 15:19 library install.

The file on disk now is not the file that crashed. model.py mtime
is 15:24:36 and model_runner.py mtime is 15:24:16, both uncommitted.
Current forward() does the .to("cpu") only in the else of
`_xpu_stream_capturing()` (line 555). stage_graph_rows (line 400)
still does the D2H, and stage_ple_cudagraph calls it. model_runner
calls that hook before run_fullgraph when the mode is FULL and the
run is not a dummy. The docstring says host work stays outside the
graph. The crashing frame was `in forward`, not that hook. The new
guard has not been through a successful FULL capture.

Verdict: the 15:13 "window open" line was true when written. The
run died during FULL capture on a host D2H, Level Zero 44, before
ready. No identity body, no greedy, no health-after, no xe-reset.
UNVERIFIED MemAvailable 117855916. Not an ITL. This is not the
15:10 eager result: mode is FULL versus enforce-eager, and this
one failed. AGENTS.md's xe-reset sentence is for a crashed TP>1
attempt. This was DP=2. The missing post-health is still a gap
before the next two-card serve.

Suggested fix: keep the D2H in the pre-replay hook and fail closed
if the capture detector returns false. Do not replay until a
capture completes and both-card health is HEALTHY. Preserve the
ple-d2h log copies.

### F144 -- hygiene, and _C.graph.so replaced at 15:19

Claim: no new staged binary. The serves named _C.graph.so.

Evidence: git diff --cached --stat line count 0. git status
porcelain was 33 paths. The only modified binary is the
pre-existing vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so.
vllm/flashnext/.gitignore ignores *.so, and check-ignore matches
the EP _C.so and _C.graph.so. Untracked files seen are text; the
largest new plugin source was sched_moe.hpp at 46847 bytes. Fork
tip remains 0a6170da8c, ahead 18, with uncommitted embedding.py,
linear.py, mapping.py, xpu/model.py, and model_runner.py.
_C.c12.so is still 10165216 bytes, sha256
b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979.
At this read, _C.so, _C.graph.so, and _C.o3.so are the same
10135104 bytes, sha256
be76661bd120aa2f6c7523ae3c08813ba56d6114660f3d7f31183c32e5e861eb.
_C.o3.so mtime is 15:05:46. _C.so and _C.graph.so mtime is
15:19:23, after the graph gpu-run exited at 15:18:30. nm -D on
the new _C.graph.so still defines b70_moe_layer and
b70_reserve_decode_queue. The 14:55, 15:02, 15:10, and 15:13
docker lines set B70_GGUF_SO to that path. None of those logs
records a digest. Pass 13 hashed the path as b3c15e14 at 14:50.
No log between 14:50 and 15:18 records a rewrite.

Verdict: OK on staging. The path the closed windows named was
replaced after they exited, so the digest they mapped is not the
file now on disk. UNVERIFIED that those processes still had the
14:50 b3c15e14 bytes. The new digest is the O3 install in F145.

Suggested fix: print sha256 of B70_GGUF_SO into each window log
before serve. Keep _C.c12.so as the revert copy, which it still is.

### F145 -- 15:40 O3 decode-gap heading

Claim: misaligned weight loads were the gap. Fair graph IQ3_S/Q2_0
T=1..4 went from 180.1/267.0/385.8/515.8 us to 68.6/100.3/141.4/178.6.
IQ2_XS/IQ4_NL went from 192.2/281.4/403.8/565.4 to
81.4/117.4/168.3/216.7. Device-only, ours is 1.07x Strata IQR2 at
T=1 and 1.01x at T=4. Gate/up 123.1 us versus Strata 45.3, then
44.4 and down 16.5. Gates failures=0 and health HEALTHY before and
after each GPU run. The 35B UD-IQ3_XXS FULL serve has identity ok,
coherence OK, C1 ITL p50 11.02 to 9.34 ms (107.0 tok/s), C2 16.61
to 12.60 ms/stream. Installed libraries share sha be76661b....e861eb.
The note says the C7b graph window had not started at the 15:19
install, so it will load the O3 binary.

Evidence: o3/fair_quiet.log prints the eight after us_graph values
under `=== FAIR _C.o3.so ===` (68.64, 100.32, 141.41, 178.59 and
81.36, 117.39, 168.33, 216.73) and the eight before values under
`=== FAIR _C.so ===` (180.10, 267.01, 385.82, 515.81 and 192.19,
281.41, 403.84, 565.37). Printed cos and max_abs match across those
two sections to the printed digits. max_rel_bulk is still high
(0.17 to 1.12). The log waited on the C7b graph container that had
held card 0 since 15:13:58, then xpu-health HEALTHY before and
after, gpu-run 45s exit 0. mb1.log has b70_sched_gu_k<21> avg_us
123.11. mb2.log has 44.42 and down 16.49. This pass did not open
the Strata unitrace file, so 45.3 us is the report's figure.
gates.log has four VERDICT failures=0 lines, covered=12,
GRAPH_PROBE_OK, and HEALTHY before and after that command.
o3/models.json first id is hotschmoe-dev, alias
qwen3.6-35b-a3b-ud-iq3-xxs, root the 35B-A3B path, max_model_len
33792. o3/serve_driver.log: SERVE_SO is the full be76661b hash,
health-before HEALTHY card 0, identity first_ok, coherence OK with
the Paris texts, C1 itl_p50=9.34 tok/s=107.03, C2 12.59 and 12.60.
bench_35b.json p50 is 9.335601, mean 9.343331, decode_tok_s
107.028 (1000/mean, not 1000/p50). C2 completion_tokens are 46 and
40, and stream 0 p99 is 50.48. c12/serve_driver.log has C1 11.02,
p90 11.23, p99 11.47, tok/s 90.89, C2 16.61/16.61 and 55.24/62.08,
prefill 1024 tok/s 177.43. The O3 prefill line is 177.14. This O3
run did not repeat the C12 8192 and 32768 prefills. Health-after
in serve_driver first says card 0 busy because the serve container
still holds it, then HEALTHY on the retry. The three installed
libraries match the printed hash (F144). The C7b note does not:
the graph window started at 15:13:58 and exited at 15:18:34, and
the new _C.graph.so mtime is 15:19:23.

Verdict: OK on the fair us_graph table, the 123.11 / 44.42 / 16.49
plugin times, the gate verdicts, the 35B identity and coherence,
the C1/C2 driver lines against c12/serve_driver.log and
bench_35b.json, and the installed hash. The 1.07x and 1.01x ratios
are the report's kernel sums against its IQR2 column (68.3/63.7
and 174.1/171.7), not the fair-graph 68.64/112.89 ratio, which is
0.61x. UNVERIFIED the Strata 45.3 us cell and literal bit-identity
beyond the printed cos/max_abs. WRONG that the C7b graph window
had not started at 15:19 and would load the O3 binary. That window
died on the previous file (F143). The 9.34 ms result is the 35B
UD-IQ3_XXS card-0 serve, not the Flash-Next DP=2 window. C2 did
not both finish 64 tokens. The serve's first health-after attempt
was blocked by its own lease.

Suggested fix: correct the C7b note. The graph window already
exited on Level Zero 44 before the install. Quote C2 as 12.59 and
12.60, and say 107.03 tok/s is 1000/mean. Stop the container before
the health-after probe so the first attempt is the one that counts.

### F146 -- 15:19 C7b heading matches the graph failure

Claim: weights loaded, then FULL capture failed on both workers.
level_zero error 44 at Qwen4ExpPLE.forward input_ids.to("cpu"),
inside torch.cuda.graph during profile_cudagraph_memory. SERVE_RC 1
at 15:18:34. MemAvailable 118052504. No greedy, no ITL, no
DEVICE_LOST. A retry at 15:22 found card 0 held by O3 pid 3358701
since 15:19:48, container vllm-fn-qwen36-gguf-c0, port 18104, and
did not start a second serve.

Evidence: c7b-graph-serve-ple-d2h.log stack is
determine_available_memory, profile_cudagraph_memory,
capture_model, cudagraph_utils.py capture line 502
`forward_fn(CUDAGraphMode.NONE)`, then model.py:433
`input_ids.to("cpu")`, error 44. Current cudagraph_utils.py still
has that call inside `with torch.cuda.graph(...)`. The log string
"torch.cuda.graph" is absent. DEVICE_LOST count is 0.
profile_cudagraph_memory count is 16. The preserved window out has
MEM_AVAILABLE_KB 118052504 and SERVE_RC 1 at 15:18:34. o3/serve_driver.log
shows card 0 held by O3 pid 3358701 since 15:19:48 in container
vllm-fn-qwen36-gguf-c0. No file under logs/ contains 3358701 or a
15:22 retry line.

Verdict: OK on the capture frame, the D2H, error 44, SERVE_RC 1,
the 118052504 kB figure, and no DEVICE_LOST string. The mode
argument is NONE while the stream is recording, which is why a
host copy fails here. UNVERIFIED a distinct 15:22 C7b retry log.
The holder identity matches the O3 35B serve, which is the F145
run, not a second Flash-Next serve. Eager coherence remains F142.
Not an ITL.

Suggested fix: keep the preserved ple-d2h log. The staged-row
retry is F147 and was still open at this read.

### F147 -- 15:34 staged-row graph window is open

Claim: the retry pins B70_GGUF_SO to _C.c12.so (sha256 b3c15e14,
10165216 bytes) and leaves the O3 _C.graph.so in place. PLE gather
is staged outside the graph. MemAvailable 117593984. Both locks
empty. Port 18104 free. Leases requested.

Evidence: c7b-fn-graph-window.sh line 13 sets SO to _C.c12.so and
passes it as B70_GGUF_SO. _C.c12.so still hashes to
b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979
and is 10165216 bytes (F144). The new window out, read while the
serve was starting, says health 15:34:42 HEALTH_RC 0,
MEM_AVAILABLE_KB 117993036, serve 15:35:19. It has no READY line
and no SERVE_RC. 117593984 is not in that file. The script comment
says O3 replaced _C.graph.so at 15:19, which matches F144. This
pass did not re-check the lock files or the port.

Verdict: OK that the script pins the C12 binary and that a new
window has started with health-before OK. UNVERIFIED the journal
MemAvailable 117593984 (the window file says 117993036). Outcome,
identity, and coherence are not in yet. Not an ITL.

Suggested fix: quote 117993036 kB from the window file. Print the
sha of _C.c12.so into the window log before serve. Leave the
result for the log that contains READY or SERVE_RC.

## Pass 15 -- 2026-10-20 16:11 UTC

CPU only. No GPU, no commit. Cursor is the journal after the 15:36
G2b milestone, including the 15:39 C7b result that was inserted
above that milestone, through the 16:10 headings.

### F148 -- 15:39 staged PLE, FA decode kernel missing

Claim: the 15:34 window died during FULL capture at 15:39:23.
SERVE_RC 1. MemAvailable 117993036. B70_GGUF_SO is _C.c12.so sha
b3c15e14. The PLE host copy did not fire. FlashAttention paged
decode is not compiled for 16,256,64,false,false,false. The
PyTorch fallback calls cu_seqlens_q.cpu() inside the graph and
Level Zero returns error 44. No DEVICE_LOST. No greedy and no ITL.
The installed package is vllm-xpu-kernels 0.1.15.4. The conf line
exists at HEAD 19c6ddc and is absent from tag v0.1.15.

Evidence: c7b-fn-graph-window-fa-missing.out has health 15:34:42
HEALTH_RC 0, MEM_AVAILABLE_KB 117993036, serve 15:35:19, stop
15:39:35, SERVE_RC 1. c7b-graph-serve-fa-missing.log places 1224
and memmaps 28800138240 on both workers, loads 24.9 GiB, then
Capturing CUDA graphs (FULL) 0/2. The config string
16,256,64,false,false,false is in the log. input_ids.to count is
0. The stack is capture_model, cudagraph_utils.py:502
forward_fn(CUDAGraphMode.NONE) inside torch.cuda.graph, then
flash_attn_interface.py _fallback_varlen_attn
cu_seqlens_q.cpu().tolist(), error 44, at 15:39:23. DEVICE_LOST
count 0. OutOfMemory 0. gpu-run done in 255s exit 1. B70_GGUF_SO
is _C.c12.so. The host venv METADATA is Version 0.1.15.4, mtime
2026-10-19 22:31:15. The serve log does not print that version.
vllm-xpu-kernels HEAD is 19c6ddc and
paged_decode_default.conf:129 is that line. git grep of tag
v0.1.15 returns no hits. No health-after in the window file.

Verdict: OK on the capture path, the missing specialization, the
fallback D2H, error 44, SERVE_RC 1, the 117993036 kB figure, the
c12 path, and the conf line. The 15:34 journal MemAvailable
117593984 remains the F147 mismatch. The worker error time is
15:39:23 and the window stop is 15:39:35. Not an ITL. No greedy.
Health-after is missing.

Suggested fix: keep this log. The side library is F149. Do not
call capture passed until a window reaches READY.

### F149 -- 16:02 Xe2 paged-decode side library

Claim: BUILD_OK 16:00:05 in 59s. libattn_kernels_xe_2.so is
11944232 bytes with OFFLOAD_DEVICE_CODE 10813072 bytes. The
enabled trait is decode_policy_q16_h256_p64. The abi3 module
NEEDs only that Xe2 library. The shared 0.1.15.4 wheel mtime is
still 2026-10-19 22:31.

Evidence: c7b-fa-build.log ends BUILD_OK 16:00:05 and gpu-run
done in 59s exit 0. The side so is 11944232 bytes, mtime 15:59:59.
readelf section OFFLOAD_DEVICE_CODE is size 0xa4fe90, which is
10813072. The so contains the filename
paged_decode_kernel_template_q16_h256_p64_fff.cpp. It does not
contain the literal decode_policy_q16_h256_p64. readelf -d on
_vllm_fa2_C.abi3.so NEEDs libattn_kernels_xe_2.so and also
libze_loader, libtorch, libc10, libsycl, libstdc++, and libc.
The venv copy of libattn_kernels_xe_2.so is 306868136 bytes,
mtime 2026-10-19 22:31:14. The side build's own METADATA is
Version 0.1.16 under c7b-fa-kernel.

Verdict: OK on the build exit, the two sizes, the template
filename, and that the venv so was not replaced. WRONG that the
abi3 module NEEDs only the Xe2 library. UNVERIFIED the exact
trait identifier decode_policy_q16_h256_p64. The side library
has not been in a serve log yet.

Suggested fix: say the abi3 NEEDs the Xe2 library among its
other NEEDED entries. Quote the template filename. Serve it
from the overlay without replacing the venv wheel, which this
build did.

### F150 -- X1 proxy baseline and the served name

Claim: the 15:58 window died on `leanup: command not found`
after health-before, using _C.o3.so be76661b. The sidecar CPU
check printed FLASHNEXT_PAYLOAD_AUDIT_OK and
WIDE_FC_REFERENCE_OK [2,4,2560]. The retry measured a no-MTP
baseline: identity first hotschmoe-dd, coherence pass, 4 x 96
greedy ids, C1 p50 9.27 ms and 108.24 tok/s, C2 p50
12.48/12.38 ms and 82.34/74.93 tok/s. Post-health HEALTHY.
This is not Flash-Next performance. Claude consults returned
unrecognized_model.

Evidence: x1/window-mtp0-harness-error.out shows the be76661b
hash, health HEALTHY card 0, then line 42 `leanup: command not
found`. x1/sidecar-cpu-r2.log prints
X1_FLASHNEXT_PAYLOAD_AUDIT_OK and
X1_WIDE_FC_REFERENCE_OK torch.Size([2, 4, 2560]), cpu-run exit
0 in 34s. x1/models-mtp0.json and bench-mtp0.json first id is
hotschmoe-dd, alias qwen3.6-35b-a3b-ud-iq3-xxs-mtp0, first_ok
true. Coherence verdicts OK, Paris texts. MTP enabled false.
C1 p50 9.274712 ms, decode_tok_s 108.235. C2 p50 12.476716 and
12.379291, tok/s 82.339 and 74.927. C2 completion_tokens are
44 and 61, and stream 1 p99 is 59.42. greedy-mtp0.json has 4
rows of 96 token ids and no match field. window-mtp0.out ends
BENCH_RC=0 and X1_WINDOW_DONE ready=1, with a HEALTHY probe
after the bench. fable51.txt and claude55.txt both say
unrecognized_model. The root is Qwen3.6-35B-A3B, not the
Flash-Next HF tree.

Verdict: OK on the broken first script, the sidecar tokens
(with the X1_ prefix), the bench numbers, coherence, MTP off,
and health around that card-0 window. The first id matches the
heading and does not match the track rule that the first
served id is hotschmoe-dev. C2 did not both return 96 tokens.
Not a Flash-Next ITL. The 9.27 ms figure is this 35B proxy.

Suggested fix: put hotschmoe-dev first and keep the research
alias second. Quote C2 completions 44 and 61 and the stream-1
p99 59.42 next to the p50.

### F151 -- X2 header budget

Claim: routed bytes 42912972800, trunk 4115863040, DP weights
per card 25572349440, sidecar cards 15682250240 and
31346585600. BF16 KV 24576 bytes/token, INT8 with per-64 FP16
scales 12672, indexer 768. Corrected linear duplicates are
186040320 bytes, not 1.42 GiB. CPU test printed
CPU_ALIAS_FORWARD_OK, STREAM_ARENA_OWNERSHIP_OK, and
REGISTER_OK. No GPU savings yet. The 16:10 card-0 allocator
probe was pending at this read.

Evidence: x2/headers.json fields: trunk_bytes 4115863040,
dp2_weight_bytes_per_card 25572349440, ep_weight_bytes
[15682250240, 31346585600]. shard_device_bytes 47028835840
minus trunk_bytes is 42912972800, and that integer occurs
once in the file. kv_bytes_per_token bf16 24576, strata_int8
12672, qsa_indexer_bf16 768. old_full_dense_duplicate_bytes
129024000 plus old_merged_dense_placeholder_bytes 57016320
is 186040320. The string 1.42 is not in the x2 files read.
cpu-test.log has CPU_ALIAS_FORWARD_OK and
STREAM_ARENA_OWNERSHIP_OK. register.log has REGISTER_OK.
The 16:10 heading still said RESULT pending.

Verdict: OK on the header integers and the two CPU tokens.
The 186040320 figure is that sum. UNVERIFIED the previous
1.42 GiB estimate as a logged number. No GPU measurement in
these files. Not an ITL.

Suggested fix: name the two fields that sum to 186040320.
Leave the allocator probe for its own log.

### F152 -- X3 one-card op timings

Claim: card-1 T=12288 medians: Flash-Next GDN conv/chunk
6.699 ms, full attention 23.875 ms, HC mix 12.187, combine
2.679, PLE hash 903.32, gather 938.01, whole 990.09. Mode-0
follow-up: GDN conv 6.725, fused norm 0.989, attention
24.155, PLE whole 983.97 to 79.47 with the numpy hash
(902.72 to 2.91). Expert blk.36 56.89 to 47.08. Reduction
max_abs 1.91e-6. Post-health HEALTHY. CPU hash X3_HASH_OK.
The _C.x3wide.so build is 898512 bytes, exit 0. The 1.38 s
figure is a full-model sum, not a measured TTFT. An early
GDN core was 7.47 ms.

Evidence: x3/ops-card1.json event_ms: flashnext.gdn_conv_chunk
6.698802, full_attention 23.874896, hc_mix 12.1870, hc_combine
2.6792, ple_hash 903.3217, ple_gather 938.0136, ple_whole
990.0934. ops-card1.log health HEALTHY before and after; the
docker command exited 1 in 76s. x3/ops-rest.json:
flashnext.gdn_conv_chunk 6.724948, gdn_norm_gate 0.988646,
full_attention 24.155312, ple_whole 983.971693,
ple_whole_numpy 79.465860, ple_hash 902.714635,
ple_hash_numpy 2.909453, experts.36 56.89151 and triton_reduce
47.080104. 35b.gdn_norm_gate there is 0.70875. ops-rest.log
gpu-run exit 0 in 78s and HEALTHY before and after.
ops-basic2.log contains 7.47. wide-build.log contains 898512
and exit 0. ops-rest.log has no 1.91e-6 and no allclose.
X3_HASH_OK is in plugins/.../x3_hash_check.py and in the
journal. No stdout log under x3 contains it. PLE whole delta
is 904.5 ms. This pass did not recompute 1.38 s.

Verdict: OK on the card1 and rest medians, both health pairs,
the 7.47 citation, and the wide so size. The card1 docker
exit 1 is a Python failure around a run that still wrote the
json; the rest run is the exit 0. UNVERIFIED X3_HASH_OK
actually printed, the 1.91e-6 reduction check, and the 1.38 s
sum. These are single-op medians at T=12288, not a serve TTFT
and not an ITL. The 16:08 heading's 35B A/B serve was still
a request, not a result, at this read.

Suggested fix: save the hash-check stdout and the reduction
allclose line. Keep the 1.38 s labeled as a sum of these
medians until a serve measures TTFT.

### F153 -- O3 stage 2 K-quant times, and the closeness claim

Claim: T=1 graph us old to new: Q2_K 485.5 to 49.3, Q3_K 268.2
to 79.4, Q4_K 96.8 to 59.2, Q5_K 119.4 to 78.7, Q6_K 581.9 to
88.4, IQ4_XS 110.5 to 81.8. T=4: 1179 to 126, 685 to 196, 252
to 149, 302 to 193, 1397 to 213, 280 to 199. New vs old cos
1.000000 and the result is bit-close. Gates failures=0 and
HEALTHY before and after. 35B C1 p50 9.30 ms (stage 1 was
9.34) and C2 12.53 ms/stream. Installed _C.so, _C.graph.so,
and _C.o3b.so are sha e9e5282d...91befa1, 10221368 bytes.
_C.o3.so stays be76661b.

Evidence: o3/kq2.log us_graph T=1 old/new: 485.46/49.31,
268.20/79.43, 96.75/59.16, 119.42/78.66, 581.89/88.42,
110.46/81.82. T=4: 1179.34/125.97, 684.84/196.43,
251.63/148.67, 302.13/193.21, 1396.57/213.08, 280.41/198.70.
Every old row has cos=0.000000. Every new row has
cos=1.000000. New max_abs is 1.22e-4 (Q2_K), 3.9e-3 (Q3_K),
1.0 (Q4_K), 8 (Q5_K), 144 (Q6_K), and 128 (IQ4_XS) at T=1.
gates_b.log contains failures=0 four times and HEALTHY twice.
The fair quartet 68.4/99.8/141.4/176.8 was not pulled from
that file in this pass. o3/bench_35b_o3b.json identity first
is hotschmoe-dev, first_ok true, coherence OK. C1 p50
9.298447. C2 p50 12.529993 and 12.539181, completions 46 and
40. The three installed libraries hash to
e9e5282d9d4cc3def64d17516e847021d623735fde357fef690208ab291befa1
and are 10221368 bytes. _C.o3.so is still be76661b,
10135104 bytes. _C.c12.so is still b3c15e14.

Verdict: OK on the microbench times, the 35B identity and
coherence, the C1 p50, and the installed hash. The new rows'
cos field is 1.000000. That is not bit-close for Q4_K and
above: max_abs is 1 to 144. The old rows' cos field is 0.
C2 is 12.530 and 12.539, not one shared 12.53, and the two
streams again stopped at 46 and 40 tokens. UNVERIFIED the
fair quartet and the bandwidth rows. Not a Flash-Next ITL.
The 9.30 ms result is the 35B card-0 serve.

Suggested fix: publish max_abs next to cos for each type.
Do not say bit-close for Q6_K while max_abs is 144. Quote
both C2 streams.

### F154 -- hygiene at 16:11 UTC

Claim: no new staged secret or untracked binary from this
pass's status check. Other agents committed.

Evidence: git diff --cached --stat line count 0. porcelain
was 13 paths. The only path flagged by a binary suffix is
still vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so.
vllm/flashnext/.gitignore still ignores *.so. Repo HEAD is
952f1971. The two parents on git log are 387bd818 and
ad52e8ef. Fork tip is still 0a6170da8c, ahead 18. This pass
did not review those commit diffs. The X1 bench's first id
is the F150 name issue, and it is not staged as a secret.

Verdict: OK on an empty index and no new staged binary.
The commits are other agents' work. This pass does not
re-audit them.

Suggested fix: leave the push-allreduce so unstaged. Keep
hotschmoe-dev first on the next proxy serve.
## Pass 16 -- 2026-10-20 16:53 UTC

Cursor: JOURNAL headings after the 16:12 G2b milestone through the
16:50 X1 collective heading. The 16:43 O3 entry is still an
announcement (s3a.log exists; no RESULT heading). The 16:43 X2
aliases arm was still pending. window-flash-mtp0.out was reopened
at 16:50:05 for the control retry and has no RESULT yet.

### F155 -- C7b 16:18 Flash-Next IQ3_XXS FULL graph serve

Claim: logs/c7b-fn-graph-window.sh served Flash-Next IQ3_XXS,
TP=1 DP=2 EP, FULL_DECODE_ONLY, FLASH_ATTN, reclaim 500, capture
1,2,3,4,8, max_model_len 2048, seqs 4, util 0.98, _C.c12.so, FA
overlay. READY 16:15:28, SERVE_RC 0, WINDOW_DONE 16:16:18. Greedy
323 and 667 match. First id hotschmoe-dev. C1 ITL p50 81.75 ms
(14.88 tok/s). C2 p50 33.20 and 33.14 ms (18.83 tok/s per stream).
Prefill 1017 tokens at 62.5 tok/s. KV 4352 and 4608 tokens,
available KV 0.64 GiB. AnonPages 14085920 kB. Worker RSS 4930408
and 4938344 kB. Capture finished with no missing-kernel line and
no Level Zero error 44. Post-health COLLECTIVE_HEALTH_OK.

Evidence: logs/c7b-fn-graph-window.out has HEALTH_RC 0,
MEM_AVAILABLE_KB 117876416, READY 16:15:28, both CHAT rows match,
GREEDY_RC 0, BENCH_RC 0, stop stamp 16:16:06, SERVE_RC 0,
WINDOW_DONE with no clock on that line. The string 16:16:18 is
absent. logs/c7b-bench-graph.json identity ids are hotschmoe-dev
then qwen3.8-flash-next-iq3-xxs, first_ok true, coherence ok.
decode_c1 itl_p50_ms 81.754281, itl_mean_ms 67.196293,
decode_tok_s 14.881773, completion_tokens 64. C2 p50 33.204481
and 33.144879, decode_tok_s 18.831348 and 18.832733. Prefill
built 1017, prefill_tok_s 62.511842. mtp.enabled false.
logs/c7b-graph-serve.log (52362 bytes at this read) has
served_model_name hotschmoe-dev first, cudagraph_mode
FULL_DECODE_ONLY, capture sizes [1, 2, 3, 4, 8], util 0.98,
B70_GGUF_SO=_C.c12.so, PYTHONPATH starting at c7b-fa-overlay.
Capturing CUDA graphs (FULL) reaches 100 percent, then "Graph
capturing finished". Counts: error 44 = 0, Paged decode kernel
not compiled = 0, Traceback = 0, DEVICE_LOST = 0,
OutOfMemoryError = 0. KV lines are 4,352 and 4,608 tokens.
Available KV cache memory 0.64 GiB. The docker command carries
SYCL_CACHE_PERSISTENT=1 from the wrapper and a later
SYCL_CACHE_PERSISTENT=0 from the script. logs/c7b-rss-graph-ready.txt
matches AnonPages 14085920 and both worker RSS figures.
logs/c7b-coll-health-graph-after.txt is COLLECTIVE_HEALTH_OK
p2p=0 HEALTHY, gpu-run exit 0. There is no c7b-health-after.txt.

Verdict: OK on identity, coherence, greedy, capture completion,
KV, RSS, and the rounded bench fields. The serve is a real
Flash-Next graph window and the rates are the ones in the JSON.
14.88 tok/s is decode_tok_s from the mean ITL (1000/67.196).
1000/p50 is 12.23 tok/s. The out file stamps stop at 16:16:06.
This is far from the Strata reference printed in the same JSON
(C1 7.09 ms). 32k was not run.

Suggested fix: stamp WINDOW_DONE from the out file and label
14.88 tok/s as the mean rate.

### F156 -- X1 proxy MTP depth 2

Claim: matched card-0 proxy, same library as the no-MTP baseline.
Identity and coherence pass, health before and after HEALTHY.
C1 chunk ITL p50 18.29 ms, 143.12 tok/s = 6.99 effective ms/token
(baseline 9.24). C2 chunk p50 28.66/28.42 ms, 91.38/84.28 tok/s
= 10.94/11.86 effective. Acceptance 114/148 = 0.77027. Three
forced-96 prompts match. The hash-table prompt first differs at
token 82, after shared EOS at token 78 (id 248044). 367 tokens
match through the first EOS.

Evidence: x1/window-mtp2.out pins _C.o3.so sha be76661b, card-0
health HEALTHY before the bench and again after H1e released
card 0, X1_WINDOW_DONE ready=1. GREEDY_RC=3 from
x1_compare_greedy.py, then BENCH_RC=0. x1/models-mtp2.json and
x1/bench-mtp2.json first id is hotschmoe-dd, alias
qwen3.6-35b-a3b-ud-iq3-xxs-mtp2, root
/mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B. first_ok is true
because expect_first is hotschmoe-dd. Coherence ok. C1 p50
18.294401, n_chunks 37, completion_tokens 96, decode_tok_s
143.123730. C2 p50 28.661820 and 28.420415, decode_tok_s
91.375966 and 84.279185, completion_tokens 46 and 40. MTP
accepted_during_bench 114, draft_tokens 148, rate
0.7702702702702703. x1/greedy-eos-gate.json: three rows
identical_forced_96 true; hash-table row identical_through_eos
true, identical_forced_96 false, baseline_end 79. Token ids:
index 78 is 248044 on both sides, first mismatch is index 82.
3*96 + 79 = 367. 1000/143.12373 = 6.986. The 9.24 baseline
matches 1000/108.235 from the F150 no-MTP bench, which is the
effective rate there.

Verdict: WRONG against the hotschmoe-dev rule. The harness
expected hotschmoe-dd and reported first_ok on that expectation.
OK on the chunk times, acceptance, EOS index, and the 367 count.
Chunk ITL is per chunk (37 chunks for 96 tokens on C1). This is
the 35B proxy. It is a latency measurement on that proxy.

Suggested fix: put hotschmoe-dev first in --served-model-name
and in the bench --model flag. Keep reporting chunk ITL and
effective ms/token as separate fields.

### F157 -- X1 Flash-Next control segfault

Claim: both ranks placed weights, load 23.97 GiB/card in 122-124 s.
The first XCCL all-gather in profiling segfaulted in
getSortedImages / PersistentDeviceCodeCache::getItemFromDisc /
allgatherv_ll_ring. No ready, no greedy, no ITL. C7b disables
persistent SYCL cache and this harness left it enabled.

Evidence: x1/flash-mtp0.log and the preserved copy
x1/flash-mtp0-sycl-cache-crash.log are 95088 bytes. Both workers
log "Model loading took 23.97 GiB" at 122.257 s and 124.058 s,
placed 1224 tensors. The segfault stack contains
getSortedImages, PersistentDeviceCodeCache::getItemFromDisc, and
allgatherv_ll_ring. READY count in that log is 0. DEVICE_LOST
count is 0. The docker command has only
SYCL_CACHE_PERSISTENT=1, B70_GGUF_SO=_C.o3.so, FA overlay first
on PYTHONPATH, served-model-name hotschmoe-dd then a detailed
IQ3 alias, max-num-batched-tokens 512, max-num-seqs 2, util
0.98, FULL_DECODE present, enforce-eager absent.
CCL_TOPO_P2P_ACCESS is absent. x1/flash-models-mtp0.json is 0
bytes. Preserved x1/window-flash-mtp0-sycl-cache-crash.out
(11230 bytes) has pre-health HEALTHY on cards 0 and 1,
SERVE_NOT_READY, RuntimeError cancelled, gpu-run exit 1 in
215 s, then post-health HEALTHY on cards 0 and 1, and
X1_FLASH_WINDOW_DONE ready=0. The live window-flash-mtp0.out
was replaced at 16:50:05 by the retry and is a different file.

Verdict: OK that this arm produced no ITL and that the stack
names the persistent device-code cache. The launch requested
hotschmoe-dd first; /v1/models never came back. The finished
window's post-health is both cards, which completes the 16:42
"card 0 passed, post-health finishing" sentence. The cache flag
differs from C7b (C7b's later --env is 0). The library and
shape also differ: this arm is _C.o3.so, batch 512, seqs 2;
C7b was _C.c12.so, batch 2048, seqs 4.

Suggested fix: keep the preserved crash log as the evidence
file. On the retry, set the served name to hotschmoe-dev and
record the library and batch next to the cache flag.

### F158 -- X2 allocator probes and the eager DP baseline

Claim: split setter leaves 67108864 reserved for 2097152 live
bytes. Totals 34242297856. First arm driver 137662464 and
profiler 202674176; second driver delta 0 and profiler
65011712. Native total 34242297856 versus vLLM total
32530182144, free gap 1712115712, and a 0.98 budget delta of
1677873398. UR-off keeps the 64 MiB reservation; expandable
segments use 20 MiB. The eager DP arm reached READY 16:36:22
as hotschmoe-dd, greedy 323/667, owned storage 26656237720,
allocated 26741253120, embedding BF16 1271398400, QSA
placeholder 39321600, GDN b/a 17694720, actual aliases
57016320, peak reserved 29345447936, post-profile reserved
28087156736, headroom 1258291200, overcount 79198208, largest
free block 835928064. One fork line passes quant_config to
embed_tokens, expected save 998246400, and was not in this arm.

Evidence: x2/allocator-probe.log shows both the default and
max_split_size_mb:20 arms with allocated 2097152, reserved
67108864, total 34242297856. First current_profiler 202674176
and reserve_adjusted 137662464. Second current_profiler
65011712 and reserve_adjusted 0. Card-0 health after is
HEALTHY. x2/info-probe.log contains both totals. Their
difference is 1712115712. 1712115712 * 0.98 = 1677873397.76,
which rounds to 1677873398. Card-1 health after is HEALTHY.
x2/allocator-ur-off.log still has reserved 67108864 and one
arm with reserved 20971520. x2/current/models.json first id is
hotschmoe-dd, second qwen3.8-flash-next-iq3-xxs, root
hf-flashnext. x2/current/greedy.json matches 323 and 667.
x2/current-summary.json total_owned_storage 26656237720,
after_profile torch_allocated 26741253120, torch_memory
28087156736, torch_peak 28078742528, reserved_slack
1345903616, possible_double_count 1337489408, embedding weight
category 1271398400. x2/current/484-weights.json embedding
parameter is 1271398400 bytes bf16. Twelve index_qk_proj
parameters at 3276800 sum to 39321600. Thirty-six in_proj_ba
parameter sizes of 491520 sum to 17694720, and the a.dense plus
b.dense shards sum to the same 17694720. Those two shard sums
are 57016320. Shard storage ids differ from the parameter
storage ids (same-storage count 0). operations reserved max is
29345447936. 29345447936 - 28087156736 = 1258291200.
1337489408 - 1258291200 = 79198208. The allocator-after-profile
file has one inactive block of size 835928064, and that is the
max inactive size (255 inactive blocks). 273152000 is in
x2/headers.json and is absent from the weight dump.
1271398400 - 273152000 = 998246400. The uncommitted fork diff
adds quant_config=self.quant_config at the embed constructor in
vllm/models/qwen4_exp/xpu/model.py, and embedding.py empties
layer.weight when packed is set. That drop is load time.

Verdict: OK on the probe integers, the greedy match, the owned
storage, the embedding byte count, the 57016320 shard sum, the
peak reserved, and the derived headroom. WRONG on the served
name: first id is hotschmoe-dd. The dense shard storages are
separate allocations in this dump. The packed size and the
998246400 save were not measured on this arm. This window is a
memory and coherence run. It has no ITL.

Suggested fix: serve hotschmoe-dev first on the aliases arm.
Report 57016320 as the extra dense-shard bytes. Measure the
embedding size again after the quant_config constructor is in
the process that loads the model.

### F159 -- X3 grouped-expert times and the census tight check

Claim: Flash-Next T=12288 conv 2.206 ms, delta 4.633 ms, whole
6.747 ms. Direct grouped 41.801 ms versus triton slot-reduce
47.215 ms. Tiles 48.213/53.769/61.171 ms. Peak deltas 7.018 GiB
direct versus 7.604 slot, tiles 2.168/1.318/0.894 GiB. The
later census says direct full-E and same-type tiling passed
1e-7 absolute / 1e-5 relative, mixed 17/16/20 max_abs 0.00133,
the 5e-2 reference passed, and post-health was HEALTHY.

Evidence: x3/checks-tile.json event_ms values are conv
2.206302, delta 4.632656, whole 6.747032, grouped 41.801354,
triton 47.215417, tiles 48.213438, 53.769479, 61.170990.
peak_allocated minus base_allocated is 7.017523, 7.603919,
2.168093, 1.318361, 0.894472 GiB. x3/census.log prints
MOE_REFERENCE_OK 17 17 20 max_abs 0.000697777 and
MOE_REFERENCE_OK 17 16 20 max_abs 0.001329824, then
MOE_GROUPED_REFERENCE_OK for 17 17 20 tiles 0 and 2 and for
17 16 20 tile 0. The next line is torch.testing.assert_close
atol=1e-7 rtol=1e-5, which raises AssertionError. Mismatched
elements 1952/163840. Greatest absolute difference
0.001329824. Greatest relative difference 13.610393. gpu-run
exit 1. Post-health card 1 HEALTHY. X3_OPS_DONE_RC=1. The
string 5e-2 is absent. The 16:13 "27.52 GiB then a 6 GiB bmm"
sentence was not re-read from x3/serve.log in this pass.

Verdict: OK on the checks-tile event_ms and GiB deltas, and on
the 0.00133 rounding of 0.001329824. WRONG that this census
passed the 1e-7 / 1e-5 check. That assert is the process
failure. The MOE_REFERENCE_OK lines did print, with max_abs
under 0.05, and the literal 5e-2 constant is not in the log.
These rows are per-op event times from checks-tile.json.

Suggested fix: keep the 5e-2 OK token and record the assert
exit 1 beside it. Leave the tight tolerance in place for
unchanged geometry, and treat the mixed-type 0.00133 gap as
an open mismatch.

### F160 -- H1e EP flag timeouts

Claim: the H1e retry ended with a layer-2 EP flag timeout, and
h1e-health-after.txt is HEALTHY on both cards.

Evidence: logs/h1e-fn-window-flag2.out (serve 16:23:56, stop
16:27:52) has SERVE_RC 1, "serve exited before ready",
NOT_READY skip greedy, health before and after HEALTHY on
cards 0 and 1. The serve tail prints "EP flag timeout layer=2
err0=0 err1=11" and RuntimeError: ep flag timeout. Sidecar
_C.so sha in that window is 5a0a047c; MoE pin _C.o3.so is
be76661b. logs/h1e-fn-window-flag12.out starts 16:42:21, stops
16:47:03, SERVE_RC 1, again not ready. Its serve log prints
"EP flag timeout layer=2 err0=12 err1=0". The sidecar sha
printed for flag12 is 1a134e8e. logs/h1e-health-after.txt is
HEALTHY on cards 0 and 1. No greedy and no bench JSON.

Verdict: OK for the layer-2 timeout, the non-ready exit, and
the both-card health after. Flag12 is a second failure with a
different sidecar hash and swapped err ranks. Neither window
is an ITL. There is still no H1e RESULT heading of its own;
the 16:29 X2 note matches flag2.

Suggested fix: give flag12 its own RESULT heading with the new
sidecar sha and err0=12 err1=0, and keep hotschmoe-dev as the
first served name when a retry reaches ready.

### F161 -- X1 collective health, first log and r2

Claim: the 16:45 check ran eager all-reduce successfully and
then Inductor failed with missing ocloc. The 16:50 retry with
ocloc on PATH returned COLLECTIVE_HEALTH_OK world_size=2
shape=4x5120 compiled_iterations=10 p2p=None, exit 0, with no
P2P override.

Evidence: x1/collective-default.log has no line that records an
eager all-reduce pass. It records COLLECTIVE_HEALTH_FAIL on
both ranks, type InductorError, FileNotFoundError for ocloc,
and "bin/xpu-collective-health.py FAILED". The word "correct"
in that file is inside a CCL topology warning.
CCL_TOPO_P2P_ACCESS is absent. SYCL_CACHE_PERSISTENT=0 is on
the command. x1/collective-default-r2.log contains
"COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120
compiled_iterations=10 p2p=None", ocloc count 0, FAILED count
0, gpu-run exit 0, cpu-run exit 0. P2P assignment count 0.

Verdict: WRONG that the first log shows an eager all-reduce
pass. OK that the first log dies on missing ocloc, and OK that
r2 matches the 16:50 success line. r2 is a collective health
result with no decode bench attached.

Suggested fix: point the 16:45 RESULT at COLLECTIVE_HEALTH_FAIL,
and keep the OK line on the r2 log only.

### F162 -- X1 PLE graph rejection probe

Claim: X1_PLE_XPU_GRAPH_REJECTION_REORDER_OK. Conv graph
0.095748 ms over 50 replays. Probe exit 0. Health before
HEALTHY; post-health later HEALTHY.

Evidence: x1/ple-graph.log prints the OK token and
"X1_PLE_CONV_GRAPH_MS 0.09574792". x1/ple-health-before.log
and x1/ple-health-after.log are card-0 HEALTHY. The harness
x1_ple_graph.py calls torch.xpu.synchronize and output.cpu()
around the compare, and end.synchronize() on the timing event.
This pass did not find a .cpu() inside the capture block.

Verdict: OK. The printed time rounds to 0.095748 ms. This is
a one-card graph replay of the conv primitive, recorded in
the probe log.

Suggested fix: none on the recorded token. Keep the CPU compare
outside the captured region.

### F163 -- hygiene at 16:53 UTC

Claim: no new staged secrets or binaries. The fork gained the
embed quant_config line and was not committed.

Evidence: repo HEAD 952f1971. git diff --cached is empty.
git status --porcelain is 59 paths. The only porcelain path
with a binary suffix is the pre-existing modified
vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so.
vllm/flashnext/.gitignore line 2 is *.so. No porcelain name
matches a secret pattern. embedding-reference.npz lives under
/mnt/vm_8tb/b70/build/vllm-flashnext/x2/, outside the git
index. Fork HEAD remains 0a6170da8c, ahead 18, with a dirty
tree: embedding.py, linear.py, mapping.py, moe.py, model.py,
mtp.py, model_runner.py, and three smaller files, 367
insertions. The embed constructor change is the single added
quant_config argument. This pass did not line-review moe.py.

Verdict: OK on an empty index and no new staged binary. The
fork edit is local and uncommitted. G2b did not commit.

Suggested fix: leave the push-allreduce so unstaged. Review
the moe.py hunk before any fork commit.


## Pass 17 -- 2026-10-20 17:38 UTC

Cursor: JOURNAL.md headings after the 16:55 G2b milestone, through
the 17:32 X2 BF16 sidecar entry and the 17:33 X3 card1-gap note.
At append the last heading was: ## 2026-10-20 17:33 X3 -- requested card1 gap after current X1
The 17:33 X3 note is a lease request. It has no new measurement.
x1/flash-mtp2.log is 74042 bytes, flash-models-mtp2.json is 1008 bytes, and flash-bench-mtp2.json exists. No MTP2 bench JSON is on disk, so this pass records no MTP2 ITL.
CPU only. No GPU, no commit, no systemd, no DD action.

### F164 -- H1e attempt 4 sidecar graph serve

Claim: logs/h1e-bench-graph.json is a finished Flash-Next sidecar
FULL graph serve. C1 ITL p50 is 60.156 ms, mean 54.368 ms, decode
18.393 tok/s, TTFT 264.67 ms. C2 p50 is 32.061 ms on both streams.
Prefill is 106.47 tok/s. Identity is hotschmoe-dev. Greedy 323/667
passed. KV is 12.06 GiB and 84,224 tokens. SERVE_RC, GREEDY_RC, and
BENCH_RC are 0. Health ran before and after.

Evidence: logs/h1e-bench-graph.json tag ep-graph-2048, port 18116,
model hotschmoe-dev. identity ids are hotschmoe-dev then
qwen3.8-flash-next-iq3-xxs, first_ok true, expect_first
hotschmoe-dev. coherence ok/same/mentions_paris, verdicts OK/OK.
decode_c1 measured: ttft_ms 264.67142, prompt 24, completion 64,
n_chunks 64, itl_p50_ms 60.155783, p90 89.402292, p99 98.675549,
mean 54.367635, decode_tok_s 18.393296, prefill_tok_s 90.678472.
decode_c2 p50 32.060791 and 32.061022, means 56.303794 and
56.409440, decode_tok_s 17.760793 and 17.727529, completion 64.
prefill built 1017/1024, ttft_ms 9551.768, prefill_tok_s
106.472434. mtp.enabled false. The same JSON stores a Strata
night-final citation c1_itl_ms 7.092199 and c1_tok_s 141.0.
60.156/7.092 = 8.482. 32.061/10.870 = 2.950.
1000/60.156 = 16.624, so 18.393 is decode_tok_s, the
mean rate, and is a different number from 1000/p50.
logs/h1e-models.json first id hotschmoe-dev, root hf-flashnext,
max_model_len 2048. logs/h1e-greedy.json has two match-true rows
and the strings 323 and 667. logs/h1e-kv.txt: Available KV 12.06
GiB, XPU KV 84,224 tokens, concurrency for 2,048 is 41.12x, lcm
block 1600. 84224/400000 = 0.21056, so a "0.211 of 2x200K"
sentence is that ratio on this max_len-2048 pool.
logs/h1e-fn-window-ok.out: sha 2e22f57e once, 5a0a047c and
1a134e8e absent, HEALTH_BEFORE_RC:0, READY 17:00:34, GREEDY_RC:0,
BENCH_RC:0, SERVE_RC:0, HEALTH_AFTER_RC:0, HEALTHY twice, no
xe-reset string. logs/h1e-fn-serve-ok.log (72925 bytes): max_len
2048, data_parallel_size=1 once, enable_expert_parallel absent,
oneCCL absent, error 44 absent, key=8 absent, Graph capturing
finished twice, model load 14.73 GiB and 131.079943 s. No separate
h1e-health-after-ok.txt. This serve is world size 1. It is a
different recipe from the C7b DP=2 graph serve.

Verdict: OK on the JSON numbers, identity, greedy rows, window
health, and SERVE_RC 0. No speed promotion. The Strata 7.09 ms
figure in the JSON is a stored citation. The 84,224-token line is
the max_len-2048 pool.

Suggested fix: keep this sidecar result separate from C7b and from
the X1 batch-512 serve. Report decode_tok_s and 1000/p50 as two
different rates.

### F165 -- X1 Flash-Next PLE-ring control

Claim: x1/flash-bench-mtp0.json C1 p50 is 40.23 ms, decode 23.98
tok/s, 84 completion tokens. C2 p50 is 44.35/44.18 ms and decode
is 22.74 tok/s per stream. Coherence passed. The window finished
ready=1 with both-card health.

Evidence: x1/flash-bench-mtp0.json model hotschmoe-dd. identity ids
are hotschmoe-dd then
qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-vllm-rt-mtp0. first_ok is
true because expect_first is hotschmoe-dd. coherence OK/OK, same,
mentions Paris. C1: ttft 609.596, prompt 24, completion 84,
n_chunks 83, p50 40.225736, p90 56.641009, p99 67.697716, mean
42.217610, decode_tok_s 23.975661, prefill_tok_s 39.370. C2 p50
44.347567 and 44.177473, means 43.978406 and 43.973624,
decode_tok_s 22.738432 and 22.740905, completion 96, n_chunks 96.
mtp.enabled false. 1000/40.226 = 24.860, so 23.976 is
decode_tok_s. x1/flash-models-mtp0.json first id hotschmoe-dd,
root hf-flashnext, max_model_len 2048. x1/window-flash-mtp0.out
(9654 bytes): HEALTHY twice, GREEDY_RC=0, BENCH_RC=0,
X1_FLASH_WINDOW_DONE, ready=1. x1/flash-mtp0.log: PLE position
ring layers=1 width=1 on two workers, Model loading took 23.97 GiB
twice, Graph capturing finished 4 times, error 44 absent,
_C.o3.so once, SYCL_CACHE_PERSISTENT=1 once and =0 once. The log
does not print a library sha. The greedy JSON is a token dump.
GREEDY_RC=0 is the window flag. This pass did not find a 323/667
match field in that JSON. The live flash-mtp0.log name is this
success log. The earlier crash remains in
x1/flash-mtp0-sycl-cache-crash.log.

Verdict: OK on the bench numbers, coherence, capture, and both-card
health in the window. WRONG on the served name: the first id is
hotschmoe-dd, and the campaign rule requires hotschmoe-dev first.
first_ok is true only because the harness expected hotschmoe-dd.
No speed promotion. Library, batch, and DP/EP differ from F164 and
from the C7b DP=2 serve, so the three p50 numbers are three recipes.

Suggested fix: put hotschmoe-dev first in --served-model-name and
set the harness expect_first to the same id. Keep n_chunks 83 beside
completion 84.

### F166 -- X1 int32 PLE stage failure and CPU fixture

Claim: the cache-disabled control died in PLE warmup on an index-put
dtype mismatch, before ready. Both ranks had loaded. A CPU fixture
then exited 0. No target benchmark.

Evidence: x1/flash-mtp0-int32.log contains the sentence "Index put
requires the source and destination dtypes match" 8 times
and the path mtp_ple.py. "23.97 GiB" occurs 2 times.
"PLEHistory" occurs 0 times. x1/window-flash-mtp0-int32.out:
HEALTHY twice, SERVE_NOT_READY, ready=0. x1/ple-cpu-int32.log ends
with X1_PLE_REJECTION_AND_SLOT_REORDER_OK and cpu-run exit 0 in 11s.
The same OK token also exists in older ple-cpu logs, so the token
alone does not identify this fixture. The later PLE control in F165
reached ready=1, which shows this dtype failure was cleared before
that ready. This pass did not line-review the int64 cast.

Verdict: OK that this attempt produced the dtype exception and no
ready, with both-card health around a failed serve. The CPU fixture
exit 0 is real for ple-cpu-int32.log. No ITL. The journal name
PLEHistory.stage is absent in this log;
the counted stack file is mtp_ple.py.

Suggested fix: line-review the stage-entry cast before treating it
as capture-safe. Cite ple-cpu-int32.log, not only the shared OK token.

### F167 -- X2 aliases memory arm

Claim: packed embedding, owned storage 25600975000 bytes per card,
available KV 3687273595 / 3709179003 versus current 960932742 /
982829958, BF16 bounds 137600 / 139200 tokens, greedy 323/667,
2047-token prefills, READY 16:53:38, HEALTHY both cards 16:56:58.
First id hotschmoe-dd. No speed claim.

Evidence: x2/aliases-summary.json workers 492 and 493 each have
total_owned_storage 25600975000. The digit string 273152000 occurs
2 time(s) in that summary. x2/aliases/
492-budget.json available_kv_bytes 3687273595. 493-budget.json
available_kv_bytes 3709179003. x2/aliases/models.json first id
hotschmoe-dd, root hf-flashnext, max_model_len 2048.
x2/aliases/greedy.json has two match-true rows for 323 and 667.
x2/aliases/prefill.json has two rows with prompt_tokens 2047,
completion_tokens 1, finish_reason length. x2/window.log has
READY=1 at 2026-10-20T16:53:38Z and RUN_DONE aliases at
2026-10-20T16:56:58Z. x2/aliases/health-after.txt reached
xpu-health HEALTHY (cards 0 1), gpu-run exit 0. x2/embedding-reference.log
is one line: EMBED_REFERENCE_OK rows=8 values=20480 qtype=21.
A search of x2 json/log/txt under 2 MB found 960932742 and
982829958 only in x2/journal-before.txt, which is a copy of the
journal sentence. 137600 inside 492-allocator JSON is a substring
of an address (18446697181137600512), not a token field. 139200
was not found as a labeled token bound.

Verdict: OK on owned storage, the two available-KV integers, greedy,
prefills, the READY stamp, and both-card health. WRONG on the first
id hotschmoe-dd. UNVERIFIED on the "versus current" KV pair and on
the 137600/139200 token bounds. The one-line embedding check records
20480 values and qtype 21. No ITL. max_model_len is 2048.

Suggested fix: write the token ceiling and the current-arm KV bytes
into the result JSON. Keep hotschmoe-dev as the first served id.

### F168 -- X2 FP8 per-token-head capacity

Claim: FP8 KV at max_len 200000 measured available KV about 5.37e9
and 5.39e9 bytes, page 3261440, block 3136, 137 pool blocks, and a
much smaller transient headroom than the aliases arm. Greedy
coherence was rejected. No speed claim.

Evidence: x2/target-fp8_per_token_head/405-budget.json
available_kv_bytes 5365499003, consumed 28189855744,
peak_headroom_bytes 2097152. 423-budget.json available_kv_bytes
5387396219, consumed 28167958528, peak_headroom_bytes 2097152.
405-profile.json transient_peak_headroom 2097152. The aliases
492-profile transient_peak_headroom is 1260388352. 405-kv-spec.json
and 423-kv-spec.json page_bytes 3261440 and block_size 3136
(3136*1040 = 3261440). 405-kv-initialized.json blocks 137.
models.json first id hotschmoe-dd, max_model_len 200000.
greedy.json: prompt "17 * 19" expected "323" match false; prompt
"23 * 29" expected "667" match true. The failed row text is not
ASCII and is omitted here. health-after.txt waited on X1 and X3,
then xpu-health HEALTHY (cards 0 1), exit 0. The digit string
5190451200 was not in the x2 result files scanned this pass.
x2/target-fp8/ is an earlier failed CLI attempt (models.json empty,
window READY=0 at 17:06:36) and is a different directory.

Verdict: OK on the capacity integers and on the greedy mismatch.
WRONG on the first id hotschmoe-dd. The coherence reject stands.
UNVERIFIED on 5190451200. No speed result and no 200K generation
quality result. The BF16 follow-up that was only "prepared" at
17:26 is the later sidecar arm in F173.

Suggested fix: keep this arm as a capacity probe. Publish a saved
derivation if 5190451200 is still cited. First served id
hotschmoe-dev.

### F169 -- X2 note on the H1e 12.06 GiB pool

Claim: 12.06 GiB is about 12.95 GB and can hold two 200K requests
that need 10341580800 bytes. The max_len-2048 capacity line is not
the long-context ceiling.

Evidence: 12.06 * 1024**3 / 1e9 = 12.9493, so "about 12.95 GB"
is the decimal-GB conversion of the logged 12.06 GiB. A search of
json/log/txt/out under the build tree, skipping venv, cache, and the
vllm source tree, found zero files containing 10341580800. The same
digit string is also absent from the later sidecar artifacts. F164
already shows the 84,224-token line is the max_len-2048 pool.

Verdict: OK on the unit conversion. UNVERIFIED on 10341580800 as a
saved result. The note correctly refuses to treat the short pool as
a measured 200K miss. It does not create a 200K ITL.

Suggested fix: save capacity.py stdout next to the journal note.

### F170 -- X3 census-r2 timings

Claim: census-r2 exited 0 with card1 health. HC mix 12.149 to 4.909
ms, combine 3.560 to 1.517 ms. 35B experts cross over at 8192 tokens
(29.818 to 13.776 ms), while 1024 fused is faster than dequant
(10.741 versus 12.771). Real router weights land in a 110-122 ms
skew fallback. A host wait and 513 scalar conversions were removed.

Evidence: x3/census-r2.log JSON rows: flashnext.hc_mix_reference
event_ms 12.148541, flashnext.hc_mix_fused 4.909323,
flashnext.hc_combine_reference 3.56, flashnext.hc_combine_fused
1.516562. 35b.experts.T8192.dq_off 29.818282 and T8192.dq_0
13.77599. T1024.dq_off 10.741406 and T1024.dq_0 12.771198.
HC_MIX_OK and HC_COMBINE_OK at max_abs 0. MOE_REFERENCE_OK max_abs
0.000421591. AssertionError count 0. gpu-run ops exit 0, then
xpu-health HEALTHY (cards 1), X3_OPS_DONE_RC=0. The CONFIG line
uses _C.c12.so. Skew rows record event_ms from 110.369 (layer 0)
through 121.780 (layer 12). Individual event_samples go higher
(one layer-35 sample is 129.852). The names in the file are dq_off
and dq_0. This pass did not confirm in source that dq_off means
fused and dq_0 means dequant. The host-wait and "513 CPU scalar
conversions" sentence was not line-reviewed. x3/census.log remains
the earlier exit-1 gate and is a different file from census-r2.log.

Verdict: OK on the event_ms values, the exit 0, and card1 health,
with the journal rounding. The 110-122 ms sentence matches the
recorded event_ms column. UNVERIFIED on the dq_off/dq_0 label
mapping and on the host-wait removal. No ITL. Distinct from the
F159 census failure.

Suggested fix: cite dq_off and dq_0 in the timing sentence. Keep
the max sample separate from the recorded event_ms.

### F171 -- X3 expert-bucket probe

Claim: layers 0/3/36 fallback 113.132/116.275/118.323 ms, untiled
buckets 60.707/58.890/60.462 ms, tile64 63.583/67.339/64.852 ms.
Peak incremental allocation 6.438 GiB falls to tile64
1.411/1.442/1.473 GiB. Original atol=rtol=0.05 gate passed.
Cosines 1.0/1.0/0.999719, worst max_abs 0.047396 at layer 36.
GPU exit 0 in 76 s. Card1 health before and after. Not bit-exact.

Evidence: x3/bucket.json event_ms: layer0 bucket0.tile0 113.132032,
bucket1.tile0 60.707188, bucket1.tile64 63.582656; layer3
116.274948, 58.89026, 67.339271; layer36 118.323021, 60.462448,
64.852343. Incremental bytes (peak_allocated minus base_allocated)
are 6912580096 on each bucket0 row, which is 6.438 GiB. tile64
increments are 1.411, 1.442, and 1.473 GiB. x3/bucket.log:
BUCKET_NUMERIC_OK layer 0 cos 1.0 max_abs 0.000959381, layer 3
cos 1.0 max_abs 0.000456132, layer 36 cos 0.999718904 max_abs
0.047395751. gpu-run done in 76s exit 0. Pre and post xpu-health
HEALTHY (cards 1). X3_OPS_DONE_RC=0. The strings atol, rtol, and
0.05 do not occur in bucket.log. bucket.layer3.bucket0.tile0 and
bucket.layer36.bucket0.tile0 carry payload tile 64, while the row
name says tile0 and the journal calls those rows fallback.
bucket.layer0.bucket0.tile0 has an empty buckets object.

Verdict: OK on the event_ms values, the incremental GiB arithmetic,
the cosine/max_abs lines, exit 0, and card1 health. The worst
max_abs 0.047396 is under 0.05, and the journal already refuses a
bit-exact claim. UNVERIFIED that the logged gate used atol=rtol=0.05,
because that threshold is not in the log. UNVERIFIED that the layer
3 and layer 36 slow rows are an untiled fallback, because their
payload tile field is 64. No ITL.

Suggested fix: print the numeric tolerance next to BUCKET_NUMERIC_OK.
Make the row name and the payload tile field agree before calling a
row fallback.

### F172 -- X3 QSA helper note

Claim: a CPU reading of the QSA helper finds per-token unused work
and a host M-RoPE loop. An opt-in path is proposed. No serving
quality claim.

Evidence: the 17:31 journal entry states the code observations and
says no QSA serving quality claim is authorized by the helper tests
alone. This pass did not line-review prefill.py or the model hook.
No QSA result JSON was checked in as a completed measurement.

Verdict: OK that the journal itself withholds a serving-quality
claim. UNVERIFIED on the helper-cost description. No ITL.

Suggested fix: measure the old and new helpers on card 1 and keep
the stock serving path at QSA0 until those gates exist.

### F173 -- X2 BF16 sidecar short gates

Claim: eager BF16 sidecar, max_len 200000, seq 2, READY 17:29:39.
Owned bytes 15710974104 / 31346585600. Card0 non-KV 18245832704,
KV budget 15311619195, 389 blocks of 39321600 bytes. Two streams
ceiling 300800 tokens each. 2x200K needs 10341580800 bytes.
Embedding 20480 values bit-exact. Greedy 323 and 667 pass. Two
2047-token prefills pass. The pool fits 2x200K. This is not 200K
coherence or latency. Post-health was still stopping the server.

Evidence: x2/window.log LEASE_ACQUIRED sidecar 17:25:45Z and
READY=1 17:29:39Z. x2/sidecar/107-budget.json available_kv_bytes
15311619195, consumed_bytes 18245832704. 107-profile.json
non_kv_cache_memory 18245832704, weights_memory 15761584640.
107-kv-initialized.json blocks 389. 107-kv-spec.json page_bytes
3276800 and block_size 1600 on the layer specs opened, dtype
bfloat16 on full attention, kv_quant_mode 0. 107-embedding-check.json
bad 0, rows 8, values 20480. models.json first id hotschmoe-dd,
max_model_len 200000, root hf-flashnext. greedy.json match true for
323 and for 667. prefill.json two rows, prompt 2047, completion 1.
serve.log EngineCore pid 107: Available KV 14.26 GiB, XPU KV
593,893 tokens, maximum concurrency for 200,000 tokens per request
2.97x, lcm block 1600. Enforce eager is logged. Graph capturing
count is 0. error 44 count is 0. The sha string 2e22f57e occurs 0
times in that serve log. 15710974104 occurs in no scanned result
file. 31346585600 occurs as xpu:1/other_buffers in
sidecar-summary.json and as ep_weight_bytes index 1 in headers.json.
headers ep_weight_bytes is 15682250240 and 31346585600.
300800 and 10341580800 and 39321600 were not found in the sidecar
json/log set. 389 * 39321600 = 15296102400, which is not
15311619195. 593893 / 2 = 296946.5, which is not 300800.
health-after.txt now contains HEALTHY.
Saved artifacts are named 107. The journal labels 138/374 were not
tied to a worker-pid field in the files opened here. The serve log
does contain those digit strings.

Verdict: OK on non-KV 18245832704, KV budget 15311619195, block
count 389, embedding bad=0, greedy, prefills, READY 17:29:39, and
the engine pool line (593,893 tokens, 2.97x at 200,000). That
engine line supports a pool large enough for two 200K requests as
a capacity print. WRONG on the first id hotschmoe-dd. WRONG to
pair 15710974104 with 31346585600 as the saved owned-byte fields;
the saved ep_weight pair is 15682250240 / 31346585600.
UNVERIFIED on 300800, 10341580800, 39321600, the H1e binary pin,
and post-health. No 200K coherence result and no latency result.

Suggested fix: quote ep_weight_bytes and the engine 593,893-token
line in the journal. Store any 39321600 derivation beside
page_bytes 3276800. Finish post-health before calling the arm
closed. First served id hotschmoe-dev.

### F174 -- hygiene and still-open serves

Claim: no new staged secrets or binaries. The fork remains local
and uncommitted.

Evidence: repo HEAD 952f1971. git diff --cached is empty.
git status --porcelain is 67 paths. Binary-suffix porcelain
paths: ['vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so']. Secret-pattern porcelain
paths: (none). Fork ## b70/flashnext...origin/main [ahead 18], HEAD
0a6170da8c. git diff --stat tail:  10 files changed, 375 insertions(+), 42 deletions(-). moe.py was not
line-reviewed this pass. x1/flash-mtp2.log is 74042 bytes, flash-models-mtp2.json is 1008 bytes, and flash-bench-mtp2.json exists. No MTP2 bench JSON is on disk, so this pass records no MTP2 ITL. x2/bf16-window.log was
empty when first listed. Porcelain rose from 59 at pass 16.
vllm/flashnext/.gitignore was not re-read this pass.

Verdict: OK on an empty index and no new secret path. The only
binary-suffix porcelain path is the pre-existing push-allreduce
library. The fork edit is uncommitted. MTP2 has no bench yet.

Suggested fix: leave the push-allreduce so unstaged. Review the
moe.py hunk before any fork commit. Read the MTP2 serve only after
it has a bench JSON or a journal RESULT.


### F175 -- MTP2 bench, sidecar health, QSA timings

Claim: the Pass 17 header and F174 say the MTP2 bench is absent. That
clause was already false when those sentences were written. The 17:39
X1 entry then reported the bench: acceptance 171/222 = 77.03%, C1
33.04 tok/s and chunk p50 85.20 ms, C2 30.40/31.24 tok/s and chunk
p50 83.37/83.53 ms, GREEDY_RC 3, BENCH_RC 0, EOS gate failed, READY
17:35:42. The 17:39 X3 entry reports QSA helper times and says
post-health was pending. The 17:39 X2 entry says sidecar post-health
was HEALTHY at 17:37:17.

Evidence: x1/flash-bench-mtp2.json tag X1-flashnext-full-mtp2, port
18121. C1 ttft 662.741, prompt 24, completion 84, n_chunks 32, p50
85.198412, mean 81.039059, decode_tok_s 33.038628. C2 p50 83.369975
and 83.531668, n_chunks 39 and 38, completion 96, decode_tok_s
30.403959 and 31.235475. 171/222 = 0.770270, which rounds to 77.03%.
1000/85.198 = 11.737, so 33.039 tok/s is the token rate and the p50
is the chunk interval. 84/32 = 2.625 tokens per chunk. Identity ids
start with hotschmoe-dd, expect_first hotschmoe-dd, first_ok true.
Coherence OK/OK. x1/flash-greedy-mtp2.json identical false,
first_mismatch [17, None, None, None], four rows of 96 tokens,
concurrent_identical false, concurrent lengths 87 and 96.
x1/window-flash-mtp2.out has GREEDY_RC=3, BENCH_RC=0, one HEALTHY
line for cards 0 and 1, and no second HEALTHY line. The window has
no READY stamp. x1/flash-mtp2.log has 41 lines stamped 17:35:42,
and that stamp is the route-registration burst, including Route
/v1/models. Graph capturing finished 4 times, error 44 count 0,
PLE width=3, gpu-run exit 0 in 323s. The indexes 4 and 72 from the
C2 sentence are not fields in the greedy JSON.
x2/window.log has RUN_DONE sidecar 2026-10-20T17:37:17Z.
x2/sidecar/health-after.txt has card 0 OK, card 1 OK, xpu-health
HEALTHY (cards 0 1), gpu-run exit 0 in 33s. x2/bf16-released was
absent at the check.
x3/qsa.json event_ms: qsa_append_token_loop 2060.655209,
qsa_prefill_chunk 5340.441771, qsa_model_hook_append 22042.110261,
qsa_model_hook_append_fast 3455.148282, indexer norm/M-RoPE
4719.806093, fast indexer 0.795469. 3455.148 + 5340.442 = 8795.590
ms, which is 8.796 s. 12 * 8.796 s = 105.55 s. Those two sums are
arithmetic on the rows, not extra timers. x3/qsa.log: pre-health
HEALTHY card 1, gpu-run done in 93s exit 0, QSA_PREFILL_NUMERIC_OK
index_max_abs 0.0, then post-health HEALTHY card 1 and
X3_OPS_DONE_RC=0. The journal text said post-health was pending.
The log now has the later HEALTHY line. No page/tail/dead/length
equality token was in the OK lines read here.
Headings after the 17:39 X3 entry were present and are not given their own measured verdict here: ## 2026-10-20 17:39 X3 -- real Flash-Next candidate gated behind proxy; ## 2026-10-20 17:41 O3 -- card-0 gates window for stage 3 (about 7 minutes); ## 2026-10-20 17:41 X2 -- sidecar teardown and collective health complete

Verdict: WRONG on the empty-bench clause in the Pass 17 header and
F174. OK on the MTP2 rates, acceptance, GREEDY_RC=3, BENCH_RC=0,
and first_mismatch [17, None, None, None]. WRONG served name
hotschmoe-dd. UNVERIFIED on C2 mismatch indexes 4 and 72. The
17:35:42 clock is the serve-log route burst. MTP2 post-health is
absent from the window. No speed promotion. Sidecar post-health is
OK, so the F173 UNVERIFIED post-health clause is closed by this
file. QSA event_ms values are OK. The 8.796 s and 105 s lines are
sums. QSA card1 post-health is OK in qsa.log. No QSA serving ITL.
The X2 17:39 entry is a lease hold. The bf16-released path was
absent at the check.

Suggested fix: quote first_mismatch and the chunk-versus-token rate
in the X1 note. Serve hotschmoe-dev first. Keep QSA0 as the serving
candidate until a real serve is measured. Treat F175 as the MTP2
file record.



### F176 -- X2 collective close and the 593893 / 300800 arithmetic

Claim: the 17:41 X2 entry says both-card post-health was HEALTHY at
17:37:17, compiled collective health passed, and WINDOW_DONE was
17:40:05. It also says the engine 593,893-token line is
389/(125+6)*200000, and that a two-stream inversion with a null
reservation yields 300800 tokens per stream. The 17:39 X3 and 17:41
O3 entries are prepared windows, not results.

Evidence: x2/window.log has RUN_DONE sidecar 2026-10-20T17:37:17Z and
WINDOW_DONE 2026-10-20T17:40:05Z. x2/collective-after.txt (1946 bytes)
has one COLLECTIVE_HEALTH_OK line: world_size=2 shape=4x5120
compiled_iterations=10 p2p=0. Then xpu-collective-health HEALTHY and
gpu-run exit 0 in 33s. FAILED count 0. 389/(125+6)*200000 =
593893.12977, which rounds to the serve log's 593,893 tokens.
(389 - 1 - 2*6) * 1600 / 2 = 300800 exactly. A null-only subtraction,
(389-1)*1600/2, is 310400, so the 300800 figure also uses six blocks
per stream on top of the null block. The 17:18 note is where that
six-block reserve was stated. x3's 17:39 entry says the full model
is prepared and not running. O3's 17:41 entry says the next O3 entry
will carry the result. No O3 gate output was opened for this finding.
No newer non-G2b heading was present after the 17:41 G2b addendum.

Verdict: OK on WINDOW_DONE 17:40:05, the collective-after health
line, and the 593,893 rounding. OK on 300800 as the null-plus-six
arithmetic above. The 17:41 wording names the null reserve and does
not restate the six blocks in that sentence. The X3 and O3 entries
have no measured ITL. No speed promotion.

Suggested fix: print (389-1-2*6)*1600/2 beside 300800 so the six
GDN blocks stay visible. Keep the collective line as health, not
as decode latency.



## Pass 18 -- 2026-10-20 18:16 UTC

Cursor: JOURNAL.md after the 17:42 G2b close, through the 18:15 X3
load note. CPU only. No GPU, no commit, no DD action.
No heading after the 18:15 X3 load note was present at this append.

### F177 -- O3 shared-expert fusion microbench

Claim: gates_e failures=0, card 0 healthy, and shared fusion cuts
about 45 us. T=1 times 115.2 to 68.5 us and 124.9 to 81.4 us. T=4
times 217.0 to 171.9 and 254.5 to 210.7. Sign table 66.3/174.8 is
slower. Stage-2 microbench and a 510 GB/s floor are also stated.

Evidence: o3/gates_e.log TIME rows: gu=21 dt=42 T=1 115.22 to 68.46
us, T=4 216.99 to 171.88 us. gu=17 dt=20 T=1 124.89 to 81.38 us, T=4
254.50 to 210.74 us. Saved column is 46.76, 45.11, 43.51, and 43.76 us.
Sign-table graph rows are 66.31 and 174.81 us, slower than 61.55 and
163.11 on the paired mb_s3 rows. failures=0 occurs 5 times. xpu-health
HEALTHY (cards 0) before and after. gpu-run exit 0 in 246s. Cosines on
the OK rows are at least 0.999998. Graph bench max_abs on one row is
1.4664e+01, so the cosine gate is not bit-exact. The strings 70.3,
189.9, 43.9, 510, 528, and 361 do not occur in this log.

Verdict: OK on the TIME before/after rows, the sign-table graph times,
failures=0, and card-0 health. UNVERIFIED on the stage-2 70.3/189.9
pair, the 30.4-vs-43.9 load split, the 510 GB/s floor, and the ISA
528-to-361 count. No serving ITL. The quant names IQ3_S/Q2_0 are not
printed on the TIME rows; the log keys are gu and dt.

Suggested fix: cite gu/dt beside the quant names, and point the
stage-2 and bandwidth sentences at the log lines that contain them.

### F178 -- X3 35B matched prefill arm

Claim: same 35B checkpoint, flag 0 to flag 1, 1024 TTFT 6025.851 to
675.427 ms and 169.93 to 1516.08 tok/s. At 8181 tokens, 45445.784 to
2388.216 ms and 180.02 to 3425.57 tok/s (19.03x). Warm 12277 is
2456.050 ms / 4998.68 tok/s. The first 12277 is 52254.904 ms and is
a JIT spike, with expert span 50304.7 ms. Identity passes. Post-health
is HEALTHY.

Evidence: x3/serve-ab.json identity ids are hotschmoe-dd then
qwen3.6-35b-a3b-ud-iq3-xxs. expect_first is hotschmoe-dd and first_ok
is true. Both arms have coherence OK/OK. matched_long_prompt_equal is
true. Arm 0: 1024 tokens, ttft 6025.851431, prefill_tok_s 169.934492;
8181 tokens, ttft 45445.784428, prefill_tok_s 180.016697. Arm 1: 1024,
ttft 675.427260, prefill_tok_s 1516.077394; 8181, ttft 2388.216451,
prefill_tok_s 3425.568900; then 12277 ttft 52254.904232 / 234.944455,
then 12277 ttft 2456.049912 / 4998.676916.
45445.784/2388.216 = 19.029. x3/proxy-spans.jsonl contains
ep_compute 50304.732871 once. The string _pad_tokens is absent there.
x3/serve-window-r2.log has two card-1 HEALTHY probes and
X3_SERVE_DONE_RC=0. This alias is the 35B proxy, not Flash-Next.
The E2 1961 reference was not re-opened.

Verdict: OK on the prefill numbers and on card-1 health in the window.
WRONG on the campaign served name: the first id is hotschmoe-dd.
The cold 52254 ms row is in the JSON before the 2456 ms row. The
50304.7 ms span exists. UNVERIFIED that the span is the _pad_tokens
JIT and UNVERIFIED the E2 1961 comparison. No decode ITL.

Suggested fix: serve hotschmoe-dev first on this proxy too. Keep the
52254 ms row out of the warm-speed sentence, as the journal already does.

### F179 -- X1 primitive shape probes

Claim: row comparisons and native GDN, packed Q8, and FA probes are
bit-identical. The two logs exit 0 after 21 s and 16 s.

Evidence: x1/shape-probe.log and shape-probe-r2.log. Every printed op
has unequal 0 and max_abs 0.0, including gdn_final_ssm, packed_q8_gemv
rows 2/3/6, and fa_spec_vs_sequential at past 16/64/128/512/1536.
gpu-run exit 0 in 21 s and 16 s. Neither log contains HEALTHY.

Verdict: OK on zero mismatches and the two exit times. These logs do
not record health. No ITL.

Suggested fix: none on the numeric rows. Keep them separate from the
later expert-row probe, which does not match exactly.

### F180 -- X2 DP BF16 200K capacity and collective close

Claim: eager DP BF16, max_len 200000, READY 17:50:28, owned
25600975000 bytes per card, KV budgets 5365490811 / 5387383931,
136 / 137 blocks of 39321600 bytes, ceilings 206400 / 208000,
greedy 323/667, embedding exact, prefills pass. Post-health HEALTHY
17:53:11. Collective OK and WINDOW_DONE 17:58:02. Storage falls
1055262720 bytes per card. Not a 200K quality or speed result.

Evidence: x2/target-auto/423-budget.json available_kv_bytes
5365490811 and blocks 136. 405-budget.json available_kv_bytes
5387383931 and blocks 137. Both weight files contain unique_bytes
25600975000. kv-spec page_bytes is 3276800 and block_size is 1600.
39321600 does not occur in the target-auto json/log set. 206400 occurs
only inside an address fragment 18446720640015663104. 208000 does not
occur. (136-1-6)*1600 = 206400 and (137-1-6)*1600 = 208000, using the
same null-plus-six reserve as F176. Profile original_transient and
torch_peak_increase are 336241664. transient_peak_headroom is 2097152.
models.json first id hotschmoe-dd, root hf-flashnext, max_model_len
200000. greedy.json match true for 323 and 667. Both embedding checks
are bad 0, values 20480. prefill.json has two 2047+1 rows.
bf16-window.log READY=1 17:50:28Z, RUN_DONE target-auto 17:53:11Z,
WINDOW_DONE 17:58:02Z. health-after.txt is HEALTHY cards 0 and 1,
exit 0. The 17:53:11 stamp is RUN_DONE, not a clock inside the health
file. target-auto/collective-after.txt has COLLECTIVE_HEALTH_OK
world_size=2 shape=4x5120 compiled_iterations=10 p2p=0, HEALTHY,
exit 0. x2/collective-after.txt is now byte-identical to that file
(6330 bytes). The 1946-byte sidecar file cited by F176 is no longer
at that path. window.log still has WINDOW_DONE 17:40:05Z for the
sidecar. 26656237720 - 25600975000 = 1055262720. The subtrahend is
the F158 current-arm figure. 1055262720 is not stored in target-auto.

Verdict: OK on unique_bytes, the two KV budgets, block counts 136 and
137, greedy, embedding, prefills, READY, both-card health, and the
collective line. WRONG first id hotschmoe-dd. WRONG to call 39321600
the spec page size. UNVERIFIED 206400/208000 as stored fields; they
match the null-plus-six derivation. The 1055262720 drop is that
subtraction, not a new counter. No 200K generation and no latency.
F176's sidecar bytes are preserved in the 17:40:05 window line, not
in the replaced collective-after.txt.

Suggested fix: store page_bytes 3276800 and the ceiling formula in
the result JSON. Keep per-arm collective logs, as the 17:58 note
already says. First served id hotschmoe-dev.

### F181 -- X3 collective R1 timeout and R2 timings

Claim: R1 produced no transfer timings and the wrapper exited 124.
R2 measured uneven RS about 93.25/93.50 ms versus balanced RS about
3.08 ms, and projects 4.49 s across 48 layers. Health passed around
R2. No ITL.

Evidence: x3/r1-collective-budget.log has no X3_COLLECTIVE timing
row, ends in SignalException signal 15, and gpu-run exit 124 in 191s.
x3/collective-budget.log R2 rows, event_ms: dispatch sizes [12288, 1]
7.665677 and 7.746823; reduce-scatter 93.250572 and 93.496406.
Balanced sizes [6144, 6144] dispatch 3.209844 and 3.215000;
reduce-scatter 3.082552 and 3.067292. 93.496406*48/1000 = 4.488 s.
X3_COLLECTIVE_HEALTH_OK before and after, compiled_iterations=10.
gpu-run exit 0 in 28s. The word HEALTHY is absent from that log.

Verdict: OK on the R2 event_ms values, the 4.49 s projection from the
slower rank, the R1 exit 124, and the compiled health tokens. The
per-card HEALTHY line is not in collective-budget.log. These are
collective probes, not decode latency.

Suggested fix: keep the 93 ms uneven RS separate from any serve ITL.
Print the rank event_ms rather than only the rounded pair.

### F182 -- X1 no-MTP repeat gates fail

Claim: two exact gates in one load both fail. Pass 1 mismatches at
tokens 4 and 72. Pass 2 versus pass 1 mismatches at 4, and the
concurrent rows differ at 2 and 72. GREEDY rc is 3. No latency rerun.
READY 18:06:10. Server stopped 18:07:15.

Evidence: x1/flash-repeat-pass1.json first_mismatch [4, 72, None, None]
in prompt order Paris, Fibonacci, count, hash. identical false.
x1/flash-repeat-pass2.json first_mismatch [4, None, None, None].
Token compare of pass 2 against pass 1: Paris first diff index 4,
and the other three C1 prompts have zero diffs. Concurrent Paris
first diff index 2. Concurrent Fibonacci first diff index 72.
x1/window-flash-repeat.out has REPEAT_PASS_1_RC=3 and
REPEAT_PASS_2_RC=3, and the dumped identity id is hotschmoe-dd.
x1/flash-repeat.log contains 18:07:15 ten times, Graph capturing
finished 4 times, error 44 count 0, and PLE width=1 twice.
18:06:10 and the word READY do not occur in that log or the window.
The window's last lines show post-health waiting on an O3 container.
hotschmoe-dev count in the serve log is 0.

Verdict: OK that both gates exit 3 and that the recorded mismatch
indexes match the token lists above. WRONG served name hotschmoe-dd.
UNVERIFIED the READY 18:06:10 clock. The 18:07:15 stop stamp is in
the serve log. No latency number was claimed and none is taken from
this window. MTP is off on this repeat, so the mismatch is not only
an MTP2 effect.

Suggested fix: journal the first_mismatch arrays. Serve hotschmoe-dev
first. Record post-health when the wait finishes.

### F183 -- X1 packed expert probe and the reduction-order note

Claim: dense rows are bit-identical. Eight expert layers differ in
FP32 by at most 7.45e-9 to 1.49e-8. Graph replay matches eager.
Card 1 health passed. A later CPU note says the grouped reduction
order can change with other tokens in the batch.

Evidence: x1/packed-shape-probe.log dense target rows have max_abs
0.0 for qtypes 21, 23, 14, 12, and 13. Expert layers 0, 1, 3, 8, 12,
24, 36, and 47 have max_abs from 7.450581e-09 to 1.490116e-08, with
unequal counts in the thousands. Every target_moe_graph_two_calls
replay has unequal 0. gpu-run exit 0 in 34s. The probe log does not
contain HEALTHY. Fork moe.py has zero .cpu(, .item(, .tolist(, and
synchronize hits. sched_moe.hpp line 1057 was not opened this pass.

Verdict: OK on the dense zeros, the expert max_abs range, the graph
replays, and exit 0. UNVERIFIED on card-1 health and on the
reduction-order mechanism. The journal already refuses full-model
causal attribution. No ITL.

Suggested fix: attach the health log to the probe. Review the plan
walk before changing FMA order.

### F184 -- X3 full load dies on a missing embedding reference

Claim: both DP ranks loaded 23.92 GiB and then exited because
x3/embedding-reference.npz was missing. No request ran. No OOM and
no device loss.

Evidence: x3/load1-flashnext-serve.log. Both workers log
"Model loading took 23.92 GiB" at about 155 s. Both then raise
FileNotFoundError for
/mnt/vm_8tb/b70/build/vllm-flashnext/x3/embedding-reference.npz.
READY count 0. error 44 count 0. OutOfMemory count 0. DEVICE_LOST
count 0. Traceback count 7. hotschmoe-dev count 0. hotschmoe-dd
count 4. No bench JSON is cited for this load.

Verdict: OK that the load reached 23.92 GiB and then failed on the
missing npz, with no ready and no device-loss string. No ITL. The
served-name string in the log is hotschmoe-dd.

Suggested fix: the journal already says the harness should check the
reference before GPU load. Keep hotschmoe-dev first on the retry.

### F185 -- O3 fuse1 35B bench file, no journal RESULT yet

Claim: no journal RESULT yet claims this file. The file itself is a
finished card-0 35B bench.

Evidence: o3/bench_35b_o3f_fuse1.json tag o3f-fuse1. identity ids
hotschmoe-dev then qwen3.6-35b-a3b-ud-iq3-xxs. expect_first
hotschmoe-dev. first id is hotschmoe-dev. coherence OK/OK.
mtp.enabled false. C1 ttft 168.271, prompt 24, completion 63,
n_chunks 62, p50 9.256293, mean 9.263198, decode_tok_s 109.723819.
C2 p50 11.792350 and 11.800019, decode_tok_s 88.322085 and
87.087215, completions 46 and 40, n_chunks 45 and 39.
1000/9.256293 = 108.035, so decode_tok_s is not 1000/p50.
o3/serve_driver_o3f_fuse1.log has HEALTHY (cards 0) twice and
BENCH_RC=0. o3/serves_f.log is 0 bytes. The 18:01 O3 journal entry
says the serve result belongs in the next O3 entry. This alias is
the 35B proxy.

Verdict: OK on the bench fields and on card-0 health in the driver
log. The campaign first id is hotschmoe-dev here. There is no
journal sentence yet to match. This is not a Flash-Next ITL and not
a closed promotion. Launch flags beyond the tag were not re-read.

Suggested fix: journal this bench with the 35B alias visible, and
do not mix 9.256 ms with Flash-Next decode numbers.

### F186 -- hygiene and still-queued windows

Claim: no new staged secret or binary. O4 and the QSA batch note are
queues. The fork remains uncommitted.

Evidence: repo HEAD 5c6edd0a. git diff --cached is empty. Porcelain
is 20 paths. The only binary-suffix porcelain path is
vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so.
Fork HEAD is still ahead 18, with 11 files changed, 384 insertions,
43 deletions. moe.py has no .cpu(, .item(, or .tolist(. The 18:14 O4
entry and the 18:12 QSA-batch entry say their results are pending.
No heading after the 18:15 X3 load note was present at this append.

Verdict: OK on an empty index and no new staged binary. HEAD moved
since pass 17; this pass did not review that commit. Queued windows
have no ITL.

Suggested fix: leave the push-allreduce so unstaged. Review the new
fork insertions before a fork commit.


## Pass 19 -- 2026-10-20 18:57 UTC

CPU-only. Headings after the 18:17 pass-18 milestone through the 18:53
O4 profile and the 18:52 X3 capture note. No GPU, no commit.

### F187 -- X1 sched probes and slot-order numeric gate

Claim: sched1 has 10 of 50 BF16 differences (11 unequal entries).
sched0 is exact in FP32 and BF16. The slot-order kernel's 63 rows are
exact. The slot library sha is bd22ebbd. The CPU build finished in
128s. Card 1 was HEALTHY before and after the slot probe. The 18:18
note said schedule post-health was still queued.

Evidence: x1/packed-shape-sched1.log has 50 JSON rows. bf16_unequal
sums to 11 and is nonzero on 10 rows. FP32 unequal is nonzero on 16
rows. x1/packed-shape-sched0.log has 50 rows and every unequal and
bf16_unequal is 0. x1/slot-probe.log has 63 rows; unequal,
bf16_unequal, max_abs, and bf16_max_abs are all 0. The built file
x1/slot-reduce/b70_gguf_xpu/_C.slot.so is 10298592 bytes and sha256
bd22ebbd40c4c6c4b0e2513be99306fe8927834e6ecdf9beeccdfd52bcda6426.
That sha is not printed in slot-probe.log. x1/slot-reduce-build.log
ends "cpu-run: [X1] done in 128s (exit 0)". slot-health-before.log
and slot-health-after.log are HEALTHY on card 1, gpu-run exit 0 in
12s and 15s. packed-shape-schedule-health-after.log is now HEALTHY
on card 1, gpu-run exit 0 in 14s, after waiting on O4.

Verdict: OK for the numeric gate and for card-1 health around those
probes. The schedule post-health file has since completed. This is
not a serving ITL. The sched_moe.hpp reduction-order mechanism stays
UNVERIFIED (F183).

Suggested fix: print the library sha in the probe log. Do not treat
the 63 exact rows as a Flash-Next decode result.

### F188 -- X1 slot-control cancel and the 18:46 watcher

Claim: per-card health passed at 18:38:44, the wrapper exited 143,
and no model loaded. The 18:46 restart is only a watcher.

Evidence: x1/window-slot-before-x3-health.out contains
"xpu-health: HEALTHY (cards 0 1)" and gpu-run exit 0. It does not
contain the clock 18:38:44. x1/slot-queue-before-x3-health.log says
card 0 was busy, held by O4 since 2026-10-20 18:38:44 running
bin/xpu-health, then "Terminated" on flock and "gpu-run: timed out
after 7200s on card 0". The string 143 is absent from the x1 slot
logs under 200 KB. The wall gap from 18:38 to the 18:41 cancel is
minutes, so the 7200s line is not a measured wait. x1/window-slot-mtp0.out
is 58 bytes: "Tue Oct 20 18:48:06 UTC 2026" and
"REUSING_RECORDED_POST_HEALTH". No tokens and no latency.

Verdict: OK that the both-card window is healthy and that no slot
serve result exists. UNVERIFIED for exit 143. UNVERIFIED that
18:38:44 is X1's health-pass clock; that stamp is O4's lease.
No MTP2 and no speed.

Suggested fix: write the wrapper exit code into the preserved queue
log, and cite the HEALTHY line's own clock.

### F189 -- X3 QSA query-batch timings

Claim: the chunk median 5352.194 ms falls to 397.849 / 293.954 /
269.644 ms at batch 256 / 1024 / 4096, and the 1024 path keeps a
0.492 GiB peak delta.

Evidence: x3/qsa-batch.json event_ms and peak_allocated-base_allocated:
token loop 2086.939584 ms and 1024 bytes; prefill chunk 5352.194323 ms
and 528482304 bytes (0.492188 GiB); batch256 397.849271 ms and the
same 0.492188 GiB; batch1024 293.953959 ms and the same 0.492188 GiB;
batch4096 269.643750 ms and 838991872 bytes (0.781372 GiB).

Verdict: OK on those timings and bytes. Exact token match for this
micro is UNVERIFIED from the JSON, which has no EXACT line. Not a
serving ITL.

Suggested fix: keep the EXACT_OK lines in the same artifact as these
rows. The 4096 peak is the larger one.

### F190 -- X3 QSA append, aligned path, and stage split

Claim: native append drops 2065.480 ms to 4.318 ms. The aligned path
drops append 1991.643 ms to 26.698 ms and the query chunk 5337.157 ms
to 252.253 ms, with exact prefixes, a 256 MiB scratch cap, stage
microseconds through 283.858, and about 3.36 s across 12 layers.
That projection is not real-model TTFT. Post-health is HEALTHY.

Evidence: x3/qsa-append.json: token loop 2065.479583 ms; native
batches 4.318386 ms with peak delta 264316416 bytes (0.246164 GiB).
x3/qsa-aligned.json: token loop 1991.643281 ms; native batches
26.698386 ms with peak delta 268708352 bytes (0.250254 GiB); prefill
chunk 5337.156875 ms and batch1024 252.252604 ms, each with peak
delta 528482304 bytes (0.492188 GiB). x3/qsa-stage-events.json:
index_scores_batch 47.702345, topk_batch 48.056512, union_build
0.596405, union_attend_xmx 151.963646, gate_batch 3.975573,
write_step_batch 1.420001, whole_event_ms 283.857552, glue_event_ms
30.143070. qsa-append.log and qsa-aligned.log each contain EXACT_OK
five times and HEALTHY twice. 12 * (26.698386 + 252.252604) / 1000
is 3.347 s. 12 * 283.857552 / 1000 is 3.406 s. Neither stored field
is 3.36. The 256 MiB cap is not a JSON field; the native delta is
0.250 GiB.

Verdict: OK on the medians, the stage split, the exact-prefix
strings, and the journal's own statement that this is a component
budget rather than TTFT. The ~3.36 s figure is an approximate
projection, not a measured serve timer. The 18:30 sentence that
post-health was still waiting is older than the HEALTHY lines now
in the logs. Not a Flash-Next ITL.

Suggested fix: store the 12-layer sum in the JSON if it will be
cited, and label it as a projection.

### F191 -- X3 R3 startup stall and the health that followed

Claim: R3 made no API readiness and no device-loss. Embedding
reference passes on both ranks. The saved rank0 stack is
"_Group starts.cpu()". Rank1 is in all_gather. At 18:46 and 18:47
the compiled post-health was still queued. The 18:52 note is capture
preparation only.

Evidence: x3/flashnext-serve-r3.log is 50539 bytes. READY count 0,
OutOfMemory count 0, DEVICE_LOST count 0, "23.92" count 2,
hotschmoe-dev count 1, hotschmoe-dd count 3. There is no /v1/models
body. x3/flashnext-memory-r3/490-embedding-check.json and
493-embedding-check.json are each bad 0, rows 8, values 20480.
x3/r3-stack-rank0.txt has no "cpu" substring. The frame is
b70_gguf_xpu/prefill_dq.py:162 via serve_prefill.py, inside
x3_prefill_profile.py during the memory profile. Rank1's stack
contains all_gather_into_tensor (distributed_c10d.py:5195) via
xpu_communicator.py:218 all_gather, and also has no "cpu"
substring. x3/flashnext-health-after-r3.log waited on O3, then
HEALTHY cards 0 and 1, gpu-run exit 0 in 27s.
x3/flashnext-collective-after-r3.log contains
"COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120 compiled_iterations=10 p2p=None"
and gpu-run exit 0 in 27s. HEALTHY count in that collective log is
0. No file matching qsareal exists under the build tree. No R4
bench JSON was opened.

Verdict: OK that startup did not reach READY, that both embedding
checks are bad 0, that rank1 is in all_gather, and that native
post-health is HEALTHY on both cards. WRONG for the saved rank0
location: the dump does not show _Group starts.cpu(). The compiled
collective file is now OK with p2p=None; the 18:46 and 18:47
"still queued" sentences predate that line. Identity is
UNVERIFIED because READY is 0. Both name strings appear, and the
first id was not established. No request and no ITL.

Suggested fix: attach the stack sample that actually contains
.cpu() if that frame is the claimed stall, and do not treat this
startup log as an identity check.

### F192 -- O4 base versus ringmlock PLE profile

Claim: sidecar Flash-Next, FULL_DECODE_ONLY, max_len 2048, no MTP,
B70_QSA=0. Base C1 step p50 is 80.39 ms, made of PLE gather 53.3 ms,
graph device 24.11 ms, and about 3 ms host, with input-id D2H at
20 ms. C2 base same_text is false. ringmlock mlocks 26.82 GiB in
22.9 s, MemAvailable 61.8 to 33.7 GB, C1 step p50 26.72 ms (gather
0.18, graph 23.81, host about 2.9), C2 step 32.14 ms, same_text
true. Gates 323/667 pass on both and health is HEALTHY around.
PLE faults were about 2/3 of the H1e C1 ITL.

Evidence: o4/_C.o4pin-o3f.so sha256 starts 7389c1333c7f0512.
plugins/b70_ep_xpu/b70_ep_xpu/_C.so sha256 starts 2e22f57ea997a5c2.
Both run windows print those prefixes. Both run logs contain
B70_QSA=0 and speculative_config=None. The ring log contains
B70_PLE_MLOCK=1 and B70_MTP_RT set to the strata-ab mtp-bf16 tree.
Both arms load hf-flashnext and the IQ3_XXS GGUF, tp 1, max_model_len
2048, FULL_DECODE_ONLY, FLASH_ATTN. Served-name kwargs list
hotschmoe-dev first. port=none, so there is no /v1/models body.
Profiler SUMMARY n_graph_steps is 94 on each series. Base C1 p50:
step_ms 80.391, ple_gather_ms 53.295, replay_dev_ms 24.113,
ple_d2h_ids_ms 20.205, prep_ms 3.124. Gather min 0.183, max 85.863.
Ring C1 p50: step_ms 26.72, gather_ms 0.179, replay_dev_ms 23.81.
Ring C2 step_ms p50 32.139. Base c2 same_text=False. Ring
c2 same_text=True. gates.json match true for 323 and 667 on both,
and both profile logs say gates_ok=True. The ring log line is
"PLE mlock: 26.82 GiB resident in 22.9 s; MemAvailable 61823904 ->
33723224 kB", which is 61.824 -> 33.723 decimal GB. Base window:
HEALTHY cards 0 and 1 before (28s, exit 0) and after (27s, exit 0),
RUN_RC 0. Ring health-after.txt is only "card 0 busy (held by: O3
... since=2026-10-20 18:51:38 cmd: .../bin/xpu-health --card 0)".
The ring window has no HEALTHY line and no HEALTH_AFTER. Ring
kwargs also set enable_prefix_caching False and mamba_cache_mode
align; the base kwargs omit both. No majflt counter is in the run
log. 53.295/80.391 = 0.663. 26.72 - 23.81 - 0.179 = 2.731, and
that residual is not a stored field.

Verdict: OK on the library hashes, the checkpoint, B70_QSA=0, MTP
off, the SUMMARY percentiles, the gates, the mlock line, and base
health before and after. The configured first name is hotschmoe-dev.
HTTP identity did not run. WRONG as written for "health HEALTHY
around" on the ringmlock arm: post-health has not produced a
HEALTHY line. UNVERIFIED for a major-fault count and for the causal
claim that D2H waits on the previous graph; the 20.205 ms p50 is
real. The arms are not a single-flag A/B because ring also changes
prefix caching and mamba cache mode. The gather drop 53.295 -> 0.179
is still specific to that bucket. These step_ms figures are an
in-process profiler, not the harness ITL. They do not replace the
closed H1e 60.156 ms or C7b 81.754 ms results. The two-thirds ratio
matches this profile's own step p50, not the closed H1e harness ITL.

Suggested fix: finish ring post-health and record it in the window.
Note the extra engine kwargs. Keep 26.72 ms labeled as profiler
step_ms.

### F193 -- pin_host_table is load-time mlock

Claim: B70_PLE_MLOCK=1 mlocks the IQ4_NL memmap via pin_host_table.

Evidence: fork vllm/models/qwen4_exp/gguf/ple.py pin_host_table
(about lines 161-203), called from install_host_table only when
B70_PLE_MLOCK is 1. It returns false on a zero-byte table and when
MemAvailable minus the table would fall under the 8 GiB floor. It
page-aligns the start downward, calls madvise WILLNEED, then mlock.
The function has no .cpu(), .item(), or synchronize. The ring log
prints the mlock line at 18:50:16, before the EP graph-finalize
lines. HostPackedTable.data is a uint8 CPU tensor of shape
[n_rows, row_bytes] built without a copy. pin_host_table does not
check is_contiguous(). The madvise return code is ignored.

Verdict: OK that this path is opt-in and runs at install time, so
it is not a host sync inside a captured graph op. UNVERIFIED that
every table tensor is contiguous; a gapped view would mlock the
wrong span. No race inside the decode graph was shown from this
function.

Suggested fix: reject a non-contiguous table before mlock, and check
the madvise return the same way as mlock.

### F194 -- hygiene, and windows that are still open

Claim: the repo index has no new secret or binary. R4, the real
activation capture, and the full X1 slot serve have no result yet.

Evidence: HEAD 5c6edd0a. git diff --cached is empty. Porcelain
is 28 paths. Binary-suffix porcelain rows:  M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so.
vllm/flashnext/.gitignore still lists *.so. No qsareal artifact
exists. x1/window-slot-mtp0.out has no serve. Ring post-health is
the wait in F192. gates_f.log was not parsed. Fork diff stat tail:
 12 files changed, 511 insertions(+), 45 deletions(-). This pass read pin_host_table only; moe.py's earlier
zero .cpu/.item/.tolist count was not re-opened.

Verdict: OK on an empty index and no new staged binary beyond the
already-known push-allreduce .so. The 18:52 X3 entry is preparation
and has no measurement to check. No O4 or QSA number above is a
Flash-Next harness promotion.

Suggested fix: leave the push-allreduce so unstaged. Finish ring
post-health before citing that arm as closed.

## Pass 20 -- 2026-10-20 19:31 UTC

CPU-only. Headings after the 18:57 pass-19 milestone through the
19:25 X3 note. No GPU, no commit.

### F195 -- X1 slot serve stalls after greedy

Claim: the slot-order no-MTP serve reached READY at 18:55:33, recorded
six exact token sequences with GREEDY_RC 0, then stalled on the first
benchmark coherence request. Rank0 blocked in mtp_ple.py line 60 token
D2H. Rank1 was at the next CPU DP padding sync. No latency JSON.
Post-health was queued. Fresh R3 health passed before acquisition
at 18:51:56.

Evidence: x1/slot/flash-models-mtp0.json mtime is 18:55:33 UTC and the
serve log READY count is 0. The saved /v1/models body lists
hotschmoe-dd first, then the IQ3_XXS alias, max_model_len 2048.
flash-greedy-mtp0.json has 4 rows plus 2 concurrent rows and no match
field. The window records GREEDY_RC=0. bench_latency.py raises
ConnectionResetError and the window records BENCH_RC=1. No bench JSON
is in x1/slot/. CONTROL_REPEAT_RC=1. x1/slot-hang-rank0.txt active
frame is mtp_ple.py:60 stage(), and the current source line 60 is
ids.detach().to(device="cpu"). Rank1's active frame is cpu_all_reduce_
inside sync_cudagraph_and_dp_padding, called from _dummy_run /
execute_dummy_batch. The string 18:51:56 is not in the window. The
window now also contains HEALTHY cards 0 and 1, gpu-run exit 0 in
28s, and X1_FLASH_WINDOW_DONE. Graph capturing finished on both ranks.
speculative_config=None. error 44, DEVICE_LOST, and OutOfMemory are 0.

Verdict: OK that greedy was record-only, the bench died with no
latency JSON, and rank0's saved frame is the token D2H. WRONG for the
campaign name: the first id is hotschmoe-dd. UNVERIFIED for a harness
READY line and for the 18:51:56 clock. Rank1 is in the DP padding
sync, and that sample is a dummy run rather than a proven later real
step. The 19:01 "post-health queues" sentence is older than the
HEALTHY line now in the window. Not an ITL. The arm stays rejected.

Suggested fix: serve hotschmoe-dev first. Write the API-ready clock
from a READY line, not only the models-file mtime.

### F196 -- PLE CPU barrier is outside graph replay

Claim: after optional PLE staging, every DP rank enters a CPU barrier
before model submission. The barrier is outside graph replay. The next
fenced control was prepared, not measured.

Evidence: vllm/v1/worker/gpu/model_runner.py calls prepare_mtp_ple only
when it is not a dummy run, then, if the attribute exists and the DP
world is larger than 1, calls torch.distributed.barrier on
dp_group.cpu_group. That barrier is before run_fullgraph. Dummy ranks
skip the D2H and still enter the barrier. stage() still does the
.host .to(cpu) and .tolist() work before the barrier. x1/window-slot-fenced-mtp0.out
starts 19:16:51 UTC, reuses recorded post-health, records GREEDY_RC=0,
and stops on the bench_latency.py launch. First id is again
hotschmoe-dd. flash-bench-mtp0.json does not exist. The window is
2055 bytes.

Verdict: OK that the new barrier is a CPU collective outside captured
replay and that idle ranks are included. The D2H host sync remains in
stage() on the real rank. UNVERIFIED that the fence makes the serve
finish: the fenced bench has no JSON yet, so this pass does not call
it a second stall or a pass. Name is WRONG on the fenced greedy body.
No ITL.

Suggested fix: do not claim the fence fixed serving until the bench
JSON and the repeat gate exist. Keep hotschmoe-dev first.

### F197 -- X3 R4 inner-group stall and post-health

Claim: R4 rank0 is in inner _Group(E5, M245760) after outer E256.
Rank1 waits in Gloo before RS8. Gate/up type is 16 and down is 42.
No API result and no device loss. Native and compiled post-health pass.

Evidence: x3/r4-stack-rank0.txt active frame is prefill_dq.py:162
_Group.__init__ with locals e=5 and m=245760. The caller moe_fast frame
has type_gu=16 and type_down=42. A caller _bucketed frame has locals
gt=16, dt=42, and e=256. The dump has no "Gloo" string and no literal
"E5". Rank1's active frame is distributed_c10d.py barrier inside
_prefill_ready with op "RS", seq 8, rows 24576, sizes [12288, 12288].
flashnext-serve-r4.log READY count is 0, DEVICE_LOST 0, OutOfMemory 0,
hotschmoe-dd count 3, hotschmoe-dev count 0. Embedding checks 422 and
423 are bad 0, rows 8, values 20480. flashnext-health-after-r4.log is
HEALTHY cards 0 and 1, gpu-run exit 0 in 27s. The collective log is
COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120 compiled_iterations=10
p2p=None, gpu-run exit 0 in 31s. The R4 window file is the line
READY_TIMEOUT.

Verdict: OK on the inner-group frame, the type 16/42 locals, rank1's
RS sequence-8 CPU barrier, both embedding checks, and both post-health
results. UNVERIFIED that the barrier backend string is Gloo and that
the outer E256 group "already completed"; e=256 is a caller local, not
a completion timer. Identity did not run. No benchmark and no ITL.

Suggested fix: quote the py-spy locals rather than a Gloo label the
dump does not print.

### F198 -- X3 packed-gather reuse oracle exits on contiguity

Claim: GU stride 844800 and down stride 460800 are byte-exact for
torch and Triton. Torch medians are .144 / .183 / .163 ms and Triton
.208 / .204 / .210 ms. Grouping is .295 ms and the subset is .226 ms.
The first oracle then exits 1 on a q-contiguous assertion, before
candidate comparison. Final health was still queued.

Evidence: x3/reuse.log PACKED_GATHER_EXACT 0 and 1 record stride
(844800, 1); row 2 records (460800, 1). event_ms medians are torch
0.143854, 0.183072, 0.163073 and Triton 0.207917, 0.203594, 0.209948.
native_inner_E5_M245760 event_ms is 0.295364. subset_E5_M245760 is
0.225625. REUSE_FULL_SHAPE_EXACT prints 245760 5. The log then raises
RuntimeError: q must be contiguous. gpu-run and cpu-run both exit 1.
A HEALTHY card 1 line is before the probe. No post-health line follows
the exit. x3/reuse-r2.log is 0 bytes.

Verdict: OK on the strides, the rounded medians, the exact-gather
lines, and exit 1 at that RuntimeError. Those gathers are a card-1
microbench, not a serving ITL. Post-health for this failed probe is
not in reuse.log. The 19:32 "RESULT: pending" sentence is older than
this log; the later 19:25 note matches the file.

Suggested fix: make the reference tensor contiguous in the probe and
record post-health even when the process exits 1.

### F199 -- O3 gates_g has not started its kernels

Claim: card-0 gates for _C.o3g.so sha 38f7176a were queued. The result
belongs in the next O3 entry.

Evidence: o3/gates_g.log is one wait line: card 0 busy, held by O4
since 19:15:33, command the ut2 unitrace docker. No TIME row and no
sha string. No _C.o3g.so was found under the flashnext build tree.

Verdict: OK that the gates run had not produced kernel results at this
read. UNVERIFIED for sha 38f7176a. No ITL.

Suggested fix: print the library sha on the first line of gates_g.log
once the run starts.

### F200 -- ringmlock post-health landed; ut2 profile has no journal

Claim: none in the journal for these two file updates. The 18:53 entry
left ring post-health without a HEALTHY line (F192) and queued a
unitrace breakdown.

Evidence: o4/run-ringmlock/health-after.txt now ends HEALTHY cards 0
and 1, gpu-run exit 0 in 34s, after the O3 wait. o4/run-ut2/profile.log
has no matching journal RESULT. Served-name kwargs list hotschmoe-dev
first. Gates 323/667 match and gates_ok=True. c2 same_text=True.
SUMMARY n_graph_steps 30: C1 step_ms p50 39.174, replay_dev_ms 36.156,
gather_ms 0.193, ring_d2h_ms 34.477. C2 step_ms p50 44.041, replay
40.423. A traced series has step p50 52.561 and replay 40.513
(n_graph_steps 22). RUN_RC 0 at 19:21:23. The ut2 window has no
HEALTHY line. This is an in-process profiler under unitrace, 32
tokens, not the 94-step ringmlock window.

Verdict: OK that ringmlock post-health is now HEALTHY on both cards.
The ut2 numbers match that profile.log and were not in a journal
sentence at this append. They are not a harness ITL and do not replace
the ringmlock C1 step p50 26.72 ms. replay_dev 36.156 is slower than
that run's 23.81 ms, and ring_d2h 34.477 is still a large host bucket.
ut2 post-health is UNVERIFIED from the window.

Suggested fix: journal the ut2 summary as a unitrace window, separate
from the 26.72 ms step, and attach its health result.

### F201 -- hygiene

Claim: no new staged secret or binary.

Evidence: HEAD 5c6edd0a. Cached diff is empty. Porcelain is 40
paths. Binary-suffix porcelain rows: [' M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so']. Fork shortstat: 12 files changed, 521 insertions(+), 45 deletions(-).
vllm/flashnext/.gitignore still lists *.so.

Verdict: OK on an empty index. The only binary porcelain path is the
known push-allreduce .so. The fork grew past the pass-19 511-line
stat; this pass reviewed the model_runner barrier and not the rest
of that diff.

Suggested fix: leave the .so unstaged.

## Pass 21 -- 2026-10-20 20:05 UTC

CPU-only. Headings after the 19:31 pass-20 milestone through the two
20:00 notes. No GPU, no commit.

### F202 -- X3 reuse-r2 and the paged FA probes

Claim: reuse-r2 exits 0, strided gathers are exact, the E5 subset
max abs is .000923, residual mean .838073 ms times 48 is 40.2275 ms,
and FA KV write .198125 ms times 12 is 2.3775 ms. Paged block64 is
24.884844 ms and fails a tight SDPA gate on 32/393216 elements, max
abs .0048828125. The r2 probe is 25.047500 and 24.564740 ms, byte-exact
across page sizes, and passes the FP32 slices. Twelve layers are
294.777 ms. The dense-GQA sum is 2344.349 ms.

Evidence: x3/reuse-r2.json event_ms is 0.838073 (count 48) and
0.198125 (count 12). 0.838073*48 = 40.227504. 0.198125*12 = 2.3775.
The log has PACKED_GATHER_EXACT, REUSE_NUMERIC_OK torch and triton
0.0009230077, gpu-run exit 0, and HEALTHY card 1 before and after.
The ".05" hit in that log is "pad=0.05ms", not a tolerance. x3/ops-rest.log
has flashnext.full_attention event_ms 24.155312. x3/paged-fa.json
event_ms is 24.884844 and wall_ms 25.621199. paged-fa.log exits 1
with "Mismatched elements: 32 / 393216" and greatest absolute
difference 0.0048828125, then HEALTHY card 1. paged-fa-r2.json is
25.0475 and 24.56474. All six PAGED_FA_FP32_REFERENCE lines are
tight_pass True. RMS values are 0.000494094, 2.645e-05, and 1.878e-05.
PAGED_FA_LAYOUT_EXACT 1600. Body exit 0 and HEALTHY card 1. The first
slice max_abs is 0.015625, so that pass is not atol 0.002 alone.
12*24.56474 = 294.77688. The string 2344.349 is not in these logs.

Verdict: OK on the reuse medians, the exact-gather lines, the card-1
health, the paged timings, the 32/393216 rejection, and the r2
tight_pass flags. The 40.23 ms, 2.38 ms, and 294.777 ms figures are
projections, not extra timers. UNVERIFIED for a 0.05 reference
tolerance and for 2344.349. Not a serving ITL.

Suggested fix: store the 12-layer and 48-layer products in the JSON,
and cite the unpaged 24.155 ms row from ops-rest.log.

### F203 -- loaded EP completes; R5 loses the device at AG 75

Claim: the 22 GiB loaded-EP oracle finishes all 16 per-rank
combinations, RS rows match, and collective health passes. R5 then
hits UR_RESULT_ERROR_DEVICE_LOST at 19:51:31 in rank1
_prefill_return during AG75, before API readiness. Weights were 1224
tensors and the embedding check passed. Post-health was pending in
the 19:52 note and is described as passed in the 20:00 note.

Evidence: x3/loaded-ep.log has 32 X3_LOADED_BEGIN, 32 X3_LOADED_OK,
and 32 X3_LOADED_RS_EXACT lines. X3_LOADED_READY prints 23958388736
bytes (22.3125 GiB). Collective lines before and after use
compiled_iterations=10. gpu-run exit 0. loaded-ep-health-after.log is
HEALTHY cards 0 and 1, exit 0 in 27s. The separate collective log is
COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120 p2p=None, exit 0 in
26s. loaded-ep-parsed.json has 24 rows, not 32, because some log
lines are concatenated. flashnext-serve-r5.log loads 1224 tensors.
Embedding checks 517 and 518 are bad 0. READY count is 0.
hotschmoe-dd count is 3 and hotschmoe-dev count is 0. models json is
0 bytes. AG seq=75 has enter 2 and ready 2 and return 0. The max seq
is 75. Rank1's traceback contains _prefill_return and
UR_RESULT_ERROR_DEVICE_LOST at 19:51:31. flashnext-health-after-r5.log
is now HEALTHY cards 0 and 1, exit 0 in 27s. The R5 collective log is
COLLECTIVE_HEALTH_OK with p2p=None, exit 0 in 31s.

Verdict: OK that the loaded oracle exited 0 with RS_EXACT lines and
that R5 died with device loss before READY. The 16-per-rank count
matches 32 RS_EXACT lines, not the 24-row parsed JSON. Identity did
not run. The dd strings are present with no dev id. No ITL and no
2000 tok/s. The 19:52 "health pending" sentence is older than the
HEALTHY files.

Suggested fix: split concatenated X3_LOADED lines before parsing, and
do not treat the startup log as an identity check.

### F204 -- producer fixture exits on the router reshape

Claim: the first ready-ep body exits before the producer loop because
the byte-backed BF16 router cannot reshape to 512x2560. CPU
decode-policy checks passed. Post-health was queued.

Evidence: x3/ready-ep.log raises RuntimeError: shape '[512, 2560]' is
invalid for input of size 2621440, on both ranks, at the reshape of
blk.3.ffn_gate_inp.weight. 512*2560 = 1310720 and 2621440 is twice
that, which matches a byte view of a BF16 tensor. gpu-run exit 1.
ready-ep-health-after.log has HEALTHY count 0. This pass did not find
an "8+8" string in that log.

Verdict: OK that the failure is the reshape and not a device-loss
line. UNVERIFIED for the CPU decode-policy pass from this log. The
post-health file has not reached HEALTHY. No speed claim.

Suggested fix: view the GGUF bytes as BF16 before reshape, and keep
the policy check in the same log as the GPU body.

### F205 -- O4 f1 profiler step and the HC microbench

Claim: f1 C1 step p50 falls from 26.72 to 20.81 ms and the card-0
graph from 23.81 to 18.27 ms. C2 step falls from 32.14 to 24.65 ms
and the graph from 28.61 to 21.41. Gates pass, c2 same_text is true,
and health is HEALTHY. The HC graph microbench is 0 ulp and
7.876 to 4.027 ms at T=1. The unitrace census is 4827 kernels and a
24.15 ms sum.

Evidence: o4/run-f1/profile.log SUMMARY n_graph_steps 94: C1 step_ms
p50 20.808, replay_dev_ms 18.267, sample_dev_ms 0.829; C2 step_ms
24.653, replay_dev_ms 21.409. Gates 323/667 match, c2 same_text=True,
gates_ok=True. hotschmoe-dev count 1 and hotschmoe-dd count 0 in that
profile. The window HEALTHY count is 2. 20.808-18.267 = 2.541.
o4/_C.o4pin-o3g.so sha256 starts 38f7176ac290cb26. o4/b70_ep_C.o4.so
sha256 starts 4189799b340e8060. o4/hctest.log reports max_ulp=0 at
T=1, T=2, and T=4, and graph times 7.876 to 4.027 ms at T=1 and
9.694 to 4.171 ms at T=2. That log has no HEALTHY line. The string
4827 in x3/ops-rest.log is not a kernel census; the kernel-sum claim
was not in a saved census file this pass read.

Verdict: OK on the f1 profiler percentiles, the gates, same_text, the
two library hashes, and the HC ulp and graph times. The ~2.5 ms host
figure is step minus replay. These step_ms values are an in-process
profiler, not a harness ITL, and they do not replace C7b or H1e.
UNVERIFIED for the 4827-kernel census and for a HEALTHY line around
the HC microbench.

Suggested fix: write the census totals to a small text file beside
run-ut2, and keep 20.81 ms labeled as profiler step_ms.

### F206 -- O3 stage-3 35B serve and a partial gates table

Claim: _C.o3g.so sha 38f7176a, gates failures=0, and the 35B fused
serve has C1 ITL p50 8.76 ms (116.2 tok/s) and C2 11.12/11.17 ms.
Identity and coherence pass. Several microbench pairs are quoted,
including IQ3_S/Q2_0 70.3 to 61.1 us and reduce T=4 6.55 to 3.90 us.

Evidence: bench_35b_o3g_fuse1.json first id is hotschmoe-dev,
expect_first hotschmoe-dev, first_ok true, alias
qwen3.6-35b-a3b-ud-iq3-xxs. coherence ok true, same true,
mentions_paris true. mtp.enabled false. C1 itl_p50_ms 8.760302,
decode_tok_s 116.158972, n_chunks 62, completion 63. C2 itl_p50_ms
11.117689 and 11.166312. gates_g.log contains failures=0 six times
and HEALTHY three times. b70_sched_red_k avg_us 6.55 and 3.90 are
present. The string 70.3 is absent. The o3g sha matches the o4 copy
hashed in F205; gates_g.log does not print it. This pass did not
read serve_driver_o3g_fuse1.log for an xpu-health line.

Verdict: OK on the 35B bench identity, coherence, MTP off, and the
ITL percentiles. This is the 35B proxy, not Flash-Next. OK that
failures=0 is in the gates log. UNVERIFIED for the 70.3 us arm and
for the serve's own health line. The 6.55 and 3.90 us values are in
the log without the T=4 label on the extracted line.

Suggested fix: keep the 8.76 ms result labeled 35B, and quote only
microbench rows that the gates log prints in full.

### F207 -- the second slot arm still stalls; eventless sha is the failed build

Claim: the CPU barrier did not fix slot-order serving. Both ranks
logged instantiate done 500, then the active rank blocked on PLE D2H.
The eventless probe exits 127 on replay_prologue. Its sha256 is
bc88f344. Card 1 health is HEALTHY around that probe. The build took
10s.

Evidence: x1/fenced-hang-rank0.txt is still mtp_ple.py:60 stage()
inside execute_model. Rank1 is in distributed barrier at
model_runner.py:2033, called from _dummy_run. The fenced serve log
says "re-instantiate every 500 replays". The string "done 500" is
absent. window-slot-fenced-mtp0.out now has BENCH_RC=1,
CONTROL_REPEAT_RC=1, and HEALTHY cards 0 and 1. The preserved failed
library libx1_eventless.so.rng-symbol-failure sha256 starts
bc88f344391143c9. The current libx1_eventless.so sha256 starts
b82405b40a13457c and does not match the journal sha.
eventless-probe-rng-symbol-failure.log contains 127 and
replay_prologue. Health before and after are HEALTHY card 1, exit 0
in 20s and 14s. eventless-build.log ends X1_EVENTLESS_BUILD_OK and
cpu-run exit 0 in 10s.

Verdict: OK that the fenced arm's saved stacks are still D2H versus
the new CPU barrier, and that the bench result is BENCH_RC=1. WRONG
for the exact words "instantiate done 500"; the log line is the
500-replay reclaim policy. OK that bc88f344 is the preserved failed
library, not the rebuilt file. No matched latency. The served name
on this arm remains hotschmoe-dd, as in F195.

Suggested fix: quote the reclaim log line as written, and print the
sha of the library the probe actually loaded.

### F208 -- hygiene

Claim: no new staged secret or binary.

Evidence: HEAD 18a01296. Cached diff is empty. Porcelain is 25
paths. Binary porcelain rows: ['M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so']. Fork shortstat: 12 files changed, 629 insertions(+), 47 deletions(-).
vllm/flashnext/.gitignore still lists *.so. HEAD moved from the
pass-20 value 5c6edd0a; this pass did not review that commit.

Verdict: OK on an empty index and the same push-allreduce .so. The
fork diff grew again. This pass did not line-review hc_decode.py.

Suggested fix: leave the .so unstaged.
## Pass 22 -- 2026-10-20 20:46 UTC

Evidence window: journal headings after the 20:05 pass-21 milestone, through the 20:42 X5 fallback note. CPU only. No GPU command was run. Headings that appeared after that 20:42 note at write time: 2026-10-20 20:44 X1 -- final health pass; handoff complete; 2026-10-20 20:45 F1 -- sidecar prefill built; first two-card window queued; 2026-10-20 20:44 X5 -- first BF16 GDN numeric state tolerance fails.

### F209 -- R5 allocator headroom records

Claim: the R5 memory profile minimum free is 461045760 bytes; the last record is allocated 26700187136, reserved 32359055360, free 461168640; the peak is 29548236288. No reset. The completion-wait stack alone is not a data-readiness cause.

Evidence: x3/flashnext-memory-r5/518-operations.json is a list of 295 records with fields allocated, free, name, peak, reserved, total. The minimum free is 461045760 and occurs once, on language_model.model.layers.36.mlp.shared_expert_gate. The last record is name profile-end with allocated 26700187136, reserved 32359055360, free 461168640, peak 29548236288. That peak value is also the maximum peak in the file (254 records carry it). Reserved 32359055360 occurs 246 times, so it is not a unique last snapshot. Free 461168640 occurs 10 times. The file contains no reset action.

Verdict: OK on the four byte counts and on the minimum-free and last-record fields. The peak figure is the max peak field, and it repeats. Not a cause finding and not an ITL.

Suggested fix: cite the record name with each byte count. Do not treat the repeated reserved or peak values as a single sample.

### F210 -- eventless short probe and the r2 health pin

Claim: r3 ran 2066 graph replays, RSS rose 39504 KiB, the old dispatcher changes 15216 BF16 elements with max abs 0.0029296875, canonical Q8 is exact, and captured T17 is exact. r2 leased card 1 but omitted --card 1, exited 143, and has no numeric result. The revised shim is b82405b4.

Evidence: x1/eventless-probe-r3.log line X1_EVENTLESS_REPLAY_AND_RNG_OK 2066 rss_kb 663904 703408 delta 39504. Following JSON rows: canonical 0, rows 9 and rows 17, unequal 15216, max_abs 0.0029296875; canonical 1, rows 9 and rows 17, unequal 0, max_abs 0.0. Then X1_CANONICAL_Q8_DISPATCH_OK, gpu-run exit 0, X1_PROBE_RC=0. Health before and after is bin/xpu-health --card 1, HEALTHY (cards 1), 15s exit 0. The log does not contain the words T7vsT9. x1/eventless-probe-r2.log invokes bin/xpu-health with no --card while the acquire line is card(s) 1, and the probe printed both cards HEALTHY. The r2 file ends at the loader check. It has no replay line and no exit code. The current shim sha was hashed in pass 21 as b82405b4 and was not re-hashed here.

Verdict: OK on the r3 counts, the RSS delta, the 15216 / 0.0029296875 pair, the canonical zero diffs, card-1 health, and the r2 missing --card 1. UNVERIFIED on the r2 exit 143 and on the label T7vsT9. Not a serving ITL. The RSS delta is not a leak qualification.

Suggested fix: keep r3 as the numerical record. Print the r2 gpu-run exit line if that code is cited again.

### F211 -- eventless full control harness

Claim: FULL_DECODE_ONLY DP2/EP2 control reached READY 20:13:49. C1 median chunk ITL is 43.016 ms. C2 is 45.299 and 45.606. same=false. Repeat Paris differs at index 4 and the other three C1 rows match. C2 differs at 1. Exactness fails. Diagnostic only, not a promotion. The orchestrator compared this 43 ms with O4 f1 20.8 ms.

Evidence: x1/eventless-canonical/flash-bench-mtp0.json tag X1-flashnext-full-mtp0. decode_c1 measured itl_p50_ms 43.015575502067804, itl_mean_ms 43.76410179283106, decode_tok_s 23.128433590222244, n_chunks 83, completion_tokens 84, ttft_ms 392.0271350070834. C2 itl_p50_ms 45.29905691742897 and 45.60551198665053, decode_tok_s 21.841854404994173 and 20.069014658778816, completions 96 and 96. 1000/43.0156 is 23.25, so 23.128 is the JSON decode_tok_s field. identity ids start with hotschmoe-dd, expect_first hotschmoe-dd, first_ok true. coherence ok false, same false, mentions_paris true. flash-models-mtp0.json first id is hotschmoe-dd. control-repeat-diff.json rows are first_mismatch 4 then three same true; concurrent_rows are both first_mismatch 1. The JSON does not name Paris or Fibonacci. window-eventless-control.out: HEALTHY cards 0 1 before and after, GREEDY_RC=0, BENCH_RC=3, CONTROL_REPEAT_RC=3, X1_FLASH_WINDOW_DONE ready=1. flash-mtp0.log READY count is 0. Route lines are stamped 20:13:48 and the next line is Application startup complete. served_model_name is hotschmoe-dd first, data_parallel_size 2, enable_expert_parallel true, speculative_config None, quantization b70_gguf, prefix caching false. The log contains FULL_DECODE and Capturing CUDA graphs (FULL) 2/2. oneCCL warnings are present. A later x1/eventless-health-after.log is card 1 only because card 0 was busy with X3; the control window itself has both cards.

Verdict: OK on the ITL figures, the repeat indexes, same=false, GREEDY_RC=0, BENCH_RC=3, and both-card health in the control window. WRONG on the campaign name: the first id is hotschmoe-dd. WRONG on the literal READY 20:13:49: there is no READY line, and the route clock is 20:13:48. The harness ready=1 flag is not that line. This is a finished diagnostic with a failing coherence gate, not a promotion. It is not a matched A/B against O4 f1: DP2/EP2 harness chunk ITL versus a sidecar profiler step_ms, and this arm fails identity and coherence. F205 still holds f1 at 20.808 ms.

Suggested fix: keep the first served id hotschmoe-dev. Cite Application startup complete at 20:13:48, not a READY token. Do not compare this ITL with an O4 step_ms.

### F212 -- MTP registry spin and the cancelled retry

Claim: the cold MTP2 child spun at 100 percent CPU in libze_intel_gpu.so+0x8c1075 with no Python interpreter, before weight load. Shutdown hit a SYCL queue-destructor fault, exit 139, then both cards were HEALTHY. The retry was stopped exit 143 in 18s during compiled pre-health. Model-info priming did not run. No greedy, acceptance, or latency.

Evidence: x1/window-eventless-mtp2.out pre-health HEALTHY cards 0 1, SERVE_NOT_READY, gpu-run done in 252s exit 139, then post-health HEALTHY cards 0 1 in 26s exit 0. flash-mtp2.log traceback is KeyboardInterrupt terminated while the parent is in registry.py inspect_model_cls / subprocess communicate, followed by ur_queue queueFinish and sycl queue_impl destructor frames. flash-models-mtp2.json is 0 bytes. startup-api-stack.txt shows served-model-name hotschmoe-dd first, data-parallel-size 2, speculative MTP num_speculative_tokens 2, and the parent stack in registry.py. startup-registry-stack.txt is the line "Error: Failed to find a python interpreter in the .data section". startup-registry.perf has zero occurrences of 8c1075. startup-registry.strace is 0 bytes. x1/window-eventless-primed-mtp2.out pre-health HEALTHY cards 0 1, SERVE_NOT_READY, gpu-run done in 18s exit 143. No bench JSON was produced by that retry. The primed models file is 0 bytes.

Verdict: OK on exit 139, the registry parent stack, the missing Python interpreter in the child stack file, the SYCL destructor frames, the later both-card HEALTHY line, exit 143 in 18s, and the absence of a latency result. UNVERIFIED on the libze+0x8c1075 offset. WRONG on the campaign name in the attempted command (hotschmoe-dd first). Not an ITL. Eventless preload is not qualified.

Suggested fix: cite the stack files and the window exit codes. Drop the offset until a perf or debugger line prints it.

### F213 -- O4 f2, f3, and the on-disk f4 profile

Claim: f2 C1 step is 20.92 ms with graph 18.32 ms and C2 24.59 ms, neutral versus the o3g pin. f3 adds B70_EP_FUSE_SHARED=1 with b70_ep_C.o4b.so sha 8489056a and reports C1 19.99 (graph 17.36) and C2 23.57 (graph 20.37). Gates pass and health is HEALTHY around every window. A kernel census of 3038 and GEMV times 19.7 / 13.4 us were also claimed. f4 was described as the next B70_PLE_FAST=1 arm, with 300 random cases hashing equal, and had no RESULT numbers in the journal yet.

Evidence: o4/run-f2/profile.log SUMMARY c1 step_ms p50 20.924, replay_dev_ms p50 18.322, sample_dev_ms p50 0.829, n_graph_steps 94; c2 step_ms p50 24.587, replay_dev_ms p50 21.39, n_graph_steps 94. GATES 323/667 match, c2 same_text=True, gates_ok=True. Served name in kwargs is hotschmoe-dev first. FULL_DECODE_ONLY sidecar. run.log loads o4/b70_ep_C.o4.so and o4/_C.o4pin-o3g.so. Env has B70_PLE_MLOCK=1, B70_DECODE_HC=triton, B70_DECODE_LEAN=1, B70_EP_SHARED_OVERLAP=1. B70_EP_FUSE_SHARED and B70_PLE_FAST are absent. Window HEALTH_AFTER HEALTHY cards 0 1. There is no health-before file for f2. o4/run-f3/profile.log c1 step 19.985 replay 17.362; c2 step 23.571 replay 20.368; n_graph_steps 94; same gates, same_text, gates_ok. run.log loads o4/b70_ep_C.o4b.so and _C.o4pin-o3g.so with B70_EP_FUSE_SHARED=1. Hashed this pass: b70_ep_C.o4b.so sha256 starts 8489056aa6c3de51, 523640 bytes; _C.o4pin-o3g.so still 38f7176ac290cb26. f3 health-before and health-after are HEALTHY cards 0 1. The f3 window header also prints 2e22f57ea997a5c2, which is not the loaded o4b file. o4/run-f4/profile.log, with no journal RESULT sentence, c1 step_ms p50 19.775 replay_dev_ms 17.348; c2 step 23.145 replay 20.382; n_graph_steps 94; gates match; same_text true; gates_ok true. run.log sets B70_PLE_FAST=1 and B70_EP_FUSE_SHARED=1 and loads o4b plus o3g. f4 health-before and health-after are HEALTHY cards 0 1, after exit 0. The string 3038 was not recomputed. o4/w8.log contains "300" only inside a linux-vdso address. No 300-case hash log was found.

Verdict: OK on the f2 and f3 SUMMARY p50s, gates, same_text, hotschmoe-dev, the o4b sha, and f3 before/after health. f2 has an after-health line only, so "around every window" is too strong for that arm. OK that f4 exists on disk as another sidecar profiler: C1 step 19.775 is 0.210 ms under f3 and is the same metric class. UNVERIFIED on the 3038-kernel census, the GEMV microseconds, and the 300-case hash. None of f2, f3, or f4 is a harness ITL and none replaces C7b 81.754 ms or H1e 60.156 ms.

Suggested fix: journal the f4 SUMMARY explicitly. Keep calling these step_ms. Recompute a kernel census from the trace before citing 3038.

### F214 -- 48-round producer, then R6 OOM

Claim: the 48-round producer test passed. 24 GiB ballast per rank. Every first and last reduce-scatter row was exact. Eager and 10 compiled iterations passed before and after. Decode 8+8 and global 16 guards passed. Allocator cap restored. Dynamic GC 0.9 was accepted at round 24. Body exit 0. Low driver free on rank 1 was 659468288 bytes. Native post-health passed. The separate compiled post-check was cancelled. R6 was queued, not reported as a result.

Evidence: x3/ready-ep-rank0.json and rank1.json each have 48 rows, iterations 0 through 47. Phase baseline is iterations 0-23 and gc0.9 is 24-47. ready-ep-r3.log has X3_LOADED_READY 0 and 1 with 26105872384 bytes (24.3125 GiB, not 24*1024**3). X3_COLLECTIVE_HEALTH_OK before and after for ranks 0 and 1, compiled_iterations=10. X3_PROFILE_ALLOC fraction=0.9 restored=1.0 and X3_PROFILE_ALLOC_RESTORE_OK on both ranks. X3_ALLOC_GC_CONFIG at the iteration-24 boundary. X3_READY_EP_DONE 1 and 0. gpu-run done in 60s exit 0. Rank1 free 659468288 appears on 20 rows including iteration 47. Rank1 minimum free in the JSON is 659423232 at iteration 13. Rank0 minimum free is 81920 at iteration 13. The log contains zero occurrences of RS_, exact, mismatch, 8+8, or global. One iteration-23 line is two JSON objects concatenated. ready-ep-r3-health-after.log is HEALTHY cards 0 1, 27s exit 0. ready-ep-r3-collective-after.log starts with card 0 busy under O4 f4; this pass did not see an OK line there. x3/flashnext-serve-r6.log later shows served_model_name hotschmoe-dd first, data_parallel_size 2, placed 1224 tensors, then torch.OutOfMemoryError allocating 960.00 MiB on GPU 0 (1.28 GiB free, 28.70 GiB allowed), rank1 gloo connection closed, Engine core initialization failed, gpu-run done in 259s exit 1. READY count 0, hotschmoe-dev count 0, DEVICE_LOST count 0. models json was 0 bytes at the earlier inventory.

Verdict: OK on 48 iterations, the GC switch at 24, in-body compiled before/after, exit 0, the 659468288 rank1 reading, and native post-health. WRONG if 24 GiB is read as exact: the loaded byte count is 26105872384 (24.3125 GiB). 659468288 is not the minimum free. UNVERIFIED on reduce-scatter exactness and on the 8+8 / global 16 guards. The R6 serve is an on-disk failure the 20:32 queue note did not claim: exit 1, OOM, first id hotschmoe-dd, no ready. Not an ITL.

Suggested fix: print the loaded byte count. Point the exactness claim at a max-abs line. Treat R6 as a failed startup until a new serve reaches /v1/models.

### F215 -- X5 IQ3 proxy failed before ready; inventory matches

Claim: the BF16 D3 library built with sha 2bb3661f. The first IQ3 proxy arm loaded 733 tensors and then capture failed because the c7b FA overlay lacks the 35B paged config, with a CPU-reference D2H UNSUPPORTED_FEATURE during capture. No API readiness and no latency. Teardown completed. No DEVICE_LOST. The three GGUF headers are 14.069 / 18.209 / 22.663 GB, 753 tensors each, 20 MTP tensors under blk.40, nextn_predict_layers 1. The 20:42 note that the old fp16 wrapper computes projections before rejecting BF16 was not checked in source.

Evidence: x5/gdn-bf16/libb70_gdn_xpu.so sha256 starts 2bb3661f2b445ba5, 326576 bytes. x5/ud-iq3_xxs-mtp0/health-before.txt HEALTHY card 1, 19s exit 0. serve.log served_model_name hotschmoe-dd then qwen3.6-35b-a3b-ud-iq3_xxs-x5-mtp0, speculative_config None, b70 GGUF placed 733 skipped_mtp 20. Capture raises RuntimeError "Paged decode kernel not compiled for this configuration" from c7b-fa-overlay flash_attn_interface.py. UNSUPPORTED_FEATURE count 2. DEVICE_LOST count 0. READY count 0. Application startup count 0. hotschmoe-dev count 0. gpu-run done in 94s exit 1. An earlier import of b70_gdn_xpu raised ModuleNotFoundError: No module named torch, and a later line says b70_d3 gdn fused decode installed spec=0, then a bf16 layout skip. x5/quant-inventory.json has three entries: UD-IQ3_XXS 14069266720 bytes (14.069 GB), UD-IQ4_XS 18209036576 (18.209 GB), UD-Q4_K_M 22663387424 (22.663 GB). Each has 753 tensors and 20 mtp records, all named under blk.40, and metadata qwen35moe.nextn_predict_layers 1. Trunk type counts, using the standard ggml enum: IQ3 type 14 (Q6_K) 245; IQ4 and Q4 type 8 (Q8_0) 251. d3_gdn.py was not line-reviewed. Post-health for this failed arm was not confirmed.

Verdict: OK on the library sha, placed 733, the paged-kernel capture error, UNSUPPORTED_FEATURE present, exit 1, no DEVICE_LOST, and the inventory sizes and counts. WRONG on the campaign name (hotschmoe-dd first). The arm is a 35B proxy startup failure, not a Flash-Next ITL and not a match to the O3 BF16 8.760 ms bench. UNVERIFIED on the duplicate-projection code claim and on post-health.

Suggested fix: serve hotschmoe-dev first. Drop the c7b overlay on the proxy, which the journal already says was removed from X5 only. Record a health-after line for the exit-1 teardown.

### F216 -- X4 dry gate, the EP speculator skip, and hygiene

Claim: eight CPU failure-injection tests pass. Four localmaxxing dry-run plans have canSubmit false and the dry wrapper rejects measure-local. No live latency. O5 hooks.py skips SpeculatorCudaGraphManager so a draft graph key does not submit the target card-1 graph. The repo has no staged secrets or new large binaries.

Evidence: x4/cpu-tests.txt says Ran 9 tests in 4.631s, OK. The names are six GateHTTPTests and three TimingTests. x4/plans has four JSON files (no-spec and mtp, code-v1 and reasoning-v1). Each contains one canSubmit false and zero canSubmit true. x4/dry-guard.txt is the argparse error that --measure-local is not allowed with --dry-run. live-gate.log and live-models.json were 0 bytes. plugins/b70_ep_xpu/b70_ep_xpu/hooks.py sets _b70_ep_local when the class name is SpeculatorCudaGraphManager and then returns the original replay, before replay_prepare. The shown skip has no .cpu, .item, or synchronize. method.py, ep_api.hpp, moe_ep.cpp, prefill_ep.cpp, ple_fast.py, and hc_decode.py were not line-reviewed. At write time git HEAD is c1fd67c9, cached diff is empty, porcelain rows 54. Binary porcelain rows: [' M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so']. vllm/flashnext/.gitignore still lists *.so. No porcelain path name matched secret, credential, .env, id_rsa, or token. The only porcelain path over 1 MB found by a size scan was JOURNAL.md. Fork b70/flashnext HEAD 0a6170da8c shortstat: 12 files changed, 693 insertions(+), 47 deletions(-). This pass did not review commit c1fd67c9 or fork commit 0a6170da8c.

Verdict: WRONG on the count eight: the test log says 9 tests OK. OK on four plans canSubmit false, the dry-run rejection of measure-local, and an empty live gate. OK that the speculator-manager skip is present and sits outside replay_prepare. The key-collision behavior itself was not executed here. OK on an empty index and no secret-named paths. The tracked push-allreduce .so remains modified and unstaged.

Suggested fix: cite 9 tests. Leave the .so unstaged. Review ple_fast.py and the new prefill EP sources before treating them as qualified.

### F217 -- headings that landed while pass 22 was written

Claim: X1's final both-card health passed in 27s (the 26s figure was a typo). The closeout window exit 1 is ready=0, not a health failure. reports/1X1_mtp.md records original Flash-Next MTP2 acceptance 171/222 = 77.03 percent and an aggregate 528/700 = 75.43 percent. F1 built b70_ep_C.f1.so sha 3d7b231c and queued a two-card prefill window with no result yet. The X5 BF16 GDN probe failed 12/2097152 state elements at atol 2e-5, greatest delta 3.974745e-5, with post-health still pending at that stamp.

Evidence: x5/gdn-bf16/probe-r1-failed.log assert_close rtol=0.005 atol=2e-5, mismatched elements 12 / 2097152, greatest absolute difference 3.97474505007267e-05 at index (0, 31, 46, 7), gpu-run done in 19s exit 1. No DEVICE_LOST line in that file. health-before.txt and health-after.txt are each HEALTHY card 1 (13s and 15s, exit 0). The after file was still queued when the journal sentence was written. x5/ud-iq3_xxs-mtp0/health-after.txt is HEALTHY card 1, 20s exit 0, which F215 had not yet seen. x3/flashnext-health-after-r6.log is HEALTHY cards 0 1, 29s exit 0, after the R6 exit 1. f1/b70_ep_C.f1.so sha256 starts 3d7b231cbf5ebedb, 584224 bytes. f1/run-p1/health-before.txt is 119 bytes and still waiting on card 0; HEALTHY count 0. prefill_ep.cpp was not line-reviewed. A search of recent x1 logs did not find a saved "done in 27s" HEALTHY line. x5/gdn-bf16/health-after.txt shows X1 holding card 1 for bin/xpu-health since 20:43:39. reports/1X1_mtp.md contains "171/222=77.03%" and "acceptance528/700=75.43%". 171/222 matches F175. 528/700 was not re-summed from a bench JSON here. o5/run-m0/profile.log has hooks installed at 20:45:22 and served_model_name hotschmoe-dev first, and no SUMMARY line yet.

Verdict: OK on the GDN mismatch counts, the printed delta, probe exit 1, and card-1 health before and after. OK that the IQ3 proxy and R6 now have post-health HEALTHY lines. OK that the F1 library sha matches and that its window has not reported a TTFT. UNVERIFIED on the X1 27s both-card closeout and on "window exit 1 for ready 0". OK that the X1 report text contains the two acceptance strings; the 75.43 percent aggregate was not recomputed. The FP16 5.85e-5 citation was not re-opened. O5 m0 has not produced a step time. None of these is a harness ITL.

Suggested fix: point the X1 closeout sentence at the health log path. Leave F1 and O5 m0 as pending until their windows print a result.

### F218 -- X4 recounts nine tests; R6 OOM is not device loss

Claim: the current X4 cpu-tests file is nine passing tests, and the earlier "eight" sentences described an earlier stage. vLLM ModelCard.created changes on every GET. R6 used fraction 0.90, GC 0.9, and util 0.95, passed AG 11, then rank 0 OOMed at the expert-down bmm for a 960 MiB output (E8, cap 24576, H2560 BF16) with 1.28 GiB driver free. Post-health was pending. No DEVICE_LOST. A pad-row cap and an optional BF16 dense return were added and are not yet measured.

Evidence: x4/cpu-tests.txt still ends with Ran 9 tests and OK, matching the file read for F216. No earlier eight-test transcript was retained under that path. protocol.py was not opened. x3/flashnext-serve-r6.log contains OutOfMemoryError for 960.00 MiB and "1.28 GiB is free" (F214). DEVICE_LOST count is 0. gpu-run exit 1. This pass did not extract an "AG 11" line, the strings 24576 or E8, or the served fraction and util from that log. flashnext-health-after-r6.log is already HEALTHY cards 0 1, 29s exit 0, so the 20:47 "post-health pending" sentence is behind the file. The new pad-row and BF16-return code was not line-reviewed. No probe result for the queued card-1 check is in this finding.

Verdict: OK that the current CPU file is 9 tests and that the R6 log is an OOM with no DEVICE_LOST line. UNVERIFIED that an earlier eight-test run existed, that created changes on every GET, and that the OOM was specifically AG 11 / E8 / cap 24576. The pending post-health sentence is stale: the health file is HEALTHY. Not an ITL.

Suggested fix: keep the nine-test log. Cite the OOM line and the health-after file together. Review the pad-row split before the next full reload.

## Pass 23 -- 2026-10-20 21:22 UTC

Evidence window: journal headings after the 20:48 pass-22 close, through the 21:13 X5 coordination note. CPU only. Titles past the 21:10 F1 result at write time: none.

### F219 -- X5 BF16 GDN r2 tolerance gate

Claim: the same 2bb3661f binary, seeds 0/1/2, passes with convolution exact and 32 captured replays exact. Core max is 3.05e-5 to 6.10e-5, output max 0.015625, cosine 0.9999988 to 0.9999990. SSM overall max is 1.50e-4 to 2.45e-4, mean 1.31e-6 to 1.60e-6, cosine 1.0. Post-health was queued, then reported HEALTHY. This is not model qualification.

Evidence: x5/gdn-bf16/r2/probe.log passed true, gpu-run exit 0 in 21s. Seed cores max_abs 6.1035e-05, 6.1035e-05, 3.0518e-05. Output max_abs 1.5625e-02 on all three. Cosine 0.999998927, 0.999998808, 0.999999046. SSM max_abs 2.3164e-04, 2.4485e-04, 1.5016e-04; mean_abs 1.4679e-06, 1.6038e-06, 1.3133e-06; cos 1.0. conv_exact true and replay_exact true on all three rows. The JSON has no replay-count field of 32. r2/health-after.txt is HEALTHY card 1, 18s exit 0. The library sha was hashed in pass 22 as 2bb3661f and was not re-hashed.

Verdict: OK on the three-seed maxima, the output max, the cosines, conv_exact, replay_exact, exit 0, and card-1 post-health. UNVERIFIED on the numeral 32. Not a serving ITL and not stock bit parity.

Suggested fix: print the replay count next to replay_exact.

### F220 -- O4 f5 split-K profiler and the GEMV microbench

Claim: f4 C1 step is 19.78 ms (graph 17.35, host stage about 0.2 ms) and C2 23.15. f5 C1 is 19.04 ms (graph 16.65) and C2 22.48. Gates and c2 same_text pass. f5 C2 token ids match f4; f5 C1 diverges at token 49/96 because T=1 used ksplit 16. Card-1 bf_gemv is 13.8-15.5 us versus oneDNN 19.7 us at 0 ulp. An int8 variant was 12.1 us and was not used. The JIT build segfaulted. Host idle is still about 1.6 ms.

Evidence: f4 numbers are the pass-22 profile: step 19.775 / replay 17.348 and C2 step 23.145. o4/run-f5/profile.log c1 step_ms p50 19.042, replay_dev_ms 16.653, n_graph_steps 94; c2 step_ms p50 22.475, replay_dev_ms 19.707. Gates 323/667 match, same_text true, gates_ok true, RUN_RC 0. Served name hotschmoe-dev first. Env B70_PLE_FAST=1, B70_DECODE_HC_GEMV=1, B70_EP_FUSE_SHARED=1, loaded b70_ep_C.o4b.so and _C.o4pin-o3g.so. Window has HEALTH_AFTER HEALTHY cards 0 1 and no HEALTH_BEFORE line. text-c2.json is byte-identical between f4 and f5. text-c1.json lists are length 96 and the first differing index is 49 (47 ids differ after that). The f5 run log does not contain the string ksplit. o4/bf.log down M=1: ks=4 15.47 us ulp_max=0, ks=8 20.62 us ulp_max=0, ks=16 13.80 us ulp_max=0, oneDNN 19.74 us. bf.log HEALTHY count is 0. No 12.1 us line and no segfault line were read. No 0.2 ms or 1.6 ms field is in the f5 SUMMARY (prep_ms p50 is 1.477; step minus replay is 2.389).

Verdict: OK on the f5 SUMMARY p50s, gates, same_text, the C2 id match, and C1 index 49. OK that ks=16 is 13.80 us and ks=4 is 15.47 us against oneDNN 19.74 us at ulp 0. The stated 13.8-15.5 band omits ks=8 at 20.62 us, which is also ulp 0. UNVERIFIED on ksplit 16 as the cause, the 0.2 ms and 1.6 ms host split, the int8 12.1 us, the JIT segfault, and microbench health. f5 is a sidecar profiler step_ms, not a harness ITL, and not a replacement of C7b or H1e.

Suggested fix: quote ks=8 as well. Put the host-stage split in the SUMMARY if it is a claimed bucket.

### F221 -- X3 bounded row-cap probe

Claim: peak incremental memory fell from 3.37 GiB to 1.375 GiB, max padded rows 49152, reference max abs under 3e-10. Hot latency went from 25.0 ms to 33.8 ms. Layer 3 with 24 GiB ballast finished in 33.558 ms with 5.85 GB driver free. Dense BF16 roundtrip went from 10.779 ms to 4.555 ms, peak 0.986 to 0.469 GiB, bit-exact. Card-1 health passed. Not a 2000 tok/s claim.

Evidence: x3/bounded.log layer0 unbounded peak_operation_allocated 3768032256 minus base_allocated 143237120 is 3624795136 bytes (3.376 GiB). The row-capped peak minus base is 1477932032 bytes (1.376 GiB). max_padded_rows 49152 on layers 0, 3, and 36. max_abs is 4.6566e-10, 2.3283e-10, and 1.7462e-10. Unbounded layer0 event_ms is 25.097; row-capped event_ms is 33.864. loaded24GiB event_ms is 33.557813. BOUNDED_LOADED_OK free 5853351936 bytes (5.853 GB decimal). Dense event_ms 10.779271 and 4.555261. Health HEALTHY card 1 before and after the body. The strings 0.986 and 0.469 are absent. R7's serve log was already accepting HTTP 200s with an engine logger near 1229 prompt tok/s; that figure is not a probe.json TTFT and this pass did not close R7 identity, coherence, or post-health.

Verdict: OK on the byte-derived 3.376 to 1.376 GiB drop, 49152 padded rows, 33.558 ms as rounding of 33.557813, the 5.853 GB free field, the dense event times, and card-1 health. WRONG on max abs under 3e-10: layer 0 is 4.66e-10. The 25.0 to 33.8 ms pair is the 25.097 and 33.864 event_ms values, rounded. UNVERIFIED on the 0.986 and 0.469 GiB peaks and on bit-exact for the dense pair (those strings are absent; the pad rows have max_abs, not a bit-exact flag). Not an ITL.

Suggested fix: cite layer 0 max_abs 4.66e-10. Keep the R7 engine-logger rate out of the prefill table until probe.json exists.

### F222 -- F1 p1 sidecar prefill and a short decode

Claim: health before HEALTHY. KV 396,288 tokens is 2.69x at 147456, peak activation 2.22 GiB, card 1 free 1.19 GiB after layer 0. Identity hotschmoe-dev. Gates 323/667. Warm TTFT: 1K 811 ms = 1256 tok/s, 8K 4.65 s = 1761, 32K 21.98 s = 1491, 128K 117.2 s = 1118, with run 0 within 3-20 percent. The 32K math prompt is 32788 tokens and returns 323/667. Decode C1 ITL p50 is 18.45 ms and Paris coherence is ok/same. Zero EP-PF flag timeouts, SERVE_RC 0. The path is 10-16x the H1e 106 tok/s citation and below 2000 and below Strata ~2200.

Evidence: f1/run-p1/models.json first id hotschmoe-dev, alias qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-f1-prefill, max_model_len 147456. probe.json identity first_ok and alias_ok true, gates match, coherence ok true same true mentions_paris true, ok true. Warm (run 1) rows: target 1024 ttft 811.323 ms built 1019 prefill_tok_s 1255.97; target 8192 ttft 4649.75 ms built 8187 rate 1760.74; target 32768 ttft 21979.37 ms built 32762 rate 1490.58; target 131072 ttft 117232.42 ms built 131066 rate 1118.00. Run 0 rates are 996.89, 1619.26, 1456.65, 1119.27. The 1K pair differs by 26 percent of the warm TTFT. long_coherence_built is 32762, not 32788, and both math gates match. decode_c1 itl_p50_ms 18.44944094773382, n_chunks 64, completion_tokens 64, decode_tok_s 54.130. 1000/18.449 is 54.20, so 54.13 is the mean token rate. C2 p50 22.684 and 22.685. Serve log KV cache size 396,288 tokens, 2.69x at 147,456. Peak activation in that same memory line is 2.22 GiB. Card 1 free after layer 0 is 1.19 GiB. Window HEALTHY cards 0 1 before the body, PROBE_RC 0, SERVE_RC 0. health-after.txt is still the wait line for O5 m2 and has HEALTHY count 0. speculative_config None. EP library sha 3d7b231c was hashed in pass 22. The x3wide sha 98c87510 was not hashed. No separate "flag timeout" counter was read.

Verdict: OK on identity, gates, coherence, the warm prefill rates, KV size, 2.22 GiB, 1.19 GiB, SERVE_RC 0, and the 18.449 ms harness ITL on this 64-token sidecar decode. WRONG on prompt 32788 (the file says 32762) and on run 0 staying inside 3-20 percent (the 1K cold TTFT is 26 percent above the warm one). The 10-16x band matches the warm rates divided by the H1e 106 tok/s citation; the lengths are not the H1e 1017-token prompt except near 1K. Prefill stays under 2000 tok/s. The 18.45 ms figure is a real harness ITL and is still far from the Strata 7.09 ms citation. It is a 64-token sidecar measurement, not the C7b DP2 recipe. Post-health has not printed HEALTHY. UNVERIFIED on zero EP-PF flag timeouts and on sha 98c87510.

Suggested fix: cite built token counts and run 0 beside run 1. Wait for the health-after HEALTHY line before calling teardown fully closed.

### F223 -- hygiene, and claims left unchecked

Claim: no new X5 speed number. WikiText fixture hashes were recorded. X4's current file is nine CPU tests. The repo index is clean.

Evidence: the 20:55, 21:03, 21:09, and 21:13 notes do not publish a new latency. The WikiText SHA strings were not re-hashed against the parquet or tokenizer. x4/cpu-tests.txt still says Ran 9 tests OK (F216, F218). At write time HEAD is c1fd67c9, cached diff empty, porcelain rows 62. Binary rows: [' M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so']. Fork HEAD 0a6170da8c shortstat: 12 files changed, 693 insertions(+), 47 deletions(-). This pass did not line-review prefill_ep.cpp, bf_gemv, or ple_device.py. O5 m2 was holding card 0 in the F1 health-after wait; its profile SUMMARY was not read.

Verdict: OK on an empty index and the same unstaged push-allreduce .so. UNVERIFIED on the WikiText hashes. No speed promotion from the coordination notes.

Suggested fix: store the fixture hashes next to a one-line sha256 of the parquet so the next pass can recompute them.



## Pass 24 -- 2026-10-20 22:06 UTC

Journal titles after the 21:22 pass-23 milestone, checked against raw files: X5 21:26 lease note (no latency), X4 21:29 and 21:30 live gate, X5 21:33 stall stop, X3 21:36 and 21:55 R7 prefill, X5 21:37 and 21:39 eventless stock control, X3 21:40 and 21:47 health notes, X5 21:49 queue note, X5 21:58 BF16 GDN proxy, F1 21:58 p2, O5 22:02 MTP2. O4 f6 has artifacts and no journal RESULT in this window. An X5 IQ4 container was visible only as a lease holder.

### F224 -- X3 R7 dense-GQA prefill is 2615 input tok/s, and the served name is hotschmoe-dd

Claim: real Flash-Next C1 exceeds 2000. Plain C1 TTFT 4758.596/4574.296/4698.163 ms, median 2615.490 input tok/s, range 2582-2686. C2 plain 5052.789/9892.520 ms (2431.924/1242.151 tok/s). The note calls primary id hotschmoe-dd correct. Math 323/667. Long prompts 12415 tok. Native and compiled post-health closed. Native QSA at 2000 tok/s stays open, projected near 6.8 s.

Evidence: x3/flashnext-serve.json and x3/flashnext-probe-r7.log. Three profiled=false pad rows, prompt_tokens 12288, completion_tokens 1, itl_p50_ms null: ttft 4758.596285 prefill_tok_s 2582.274113; ttft 4574.296289 prefill_tok_s 2686.314839; ttft 4698.163446 prefill_tok_s 2615.490104. The median rate is 2615.490 and the median TTFT is 4698.163. C2 plain rows, also 12288 tokens: 5052.789108 ms / 2431.924178 tok/s and 9892.519894 ms / 1242.150648 tok/s. Long coherence prompt_tokens 12415, texts 323 and 667. Short C1 and C2 texts 323 and 667. identity.ids is ["hotschmoe-dd", "qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-x3-prefill"], first_ok true, expect_first hotschmoe-dd. Serve log: data_parallel_size 2, max_model_len 12800, max_num_batched_tokens 12288, max_num_seqs 2, gpu_memory_utilization 0.95, quantization b70_gguf, speculative_config None, served_model_name starts with hotschmoe-dd. The only READY hit is VLLM_ENGINE_READY_TIMEOUT_S. Application startup complete is present. DEVICE_LOST 0. OutOfMemory 0. The one hotschmoe-dev hit is inside a quoted F1 docker command. flashnext-health-after-r7.log: HEALTHY cards 0 1, done in 29s, exit 0. flashnext-collective-after-r7.log: COLLECTIVE_HEALTH_OK world_size=2 shape=4x5120 compiled_iterations=10 p2p=None, done in 33s, exit 0. flashnext-window-r7.log is X3_FLASHNEXT_DONE_RC=0. There is no flashnext-health-before-r7.log. Engine-logger lines near 1228 tok/s are interval averages, not these TTFTs.

Verdict: OK on the three plain C1 rates, the C2 pair, 12288 prompt tokens, the 12415-token math, 323/667, both post-health results, and the collective line. WRONG on calling hotschmoe-dd the correct primary for this campaign. The harness expect_first is hotschmoe-dd and first_ok is true; the campaign rule still requires hotschmoe-dev first. This field is prefill_tok_s. itl_p50_ms is null. It is a real dense-GQA DP2 prefill measurement above 2000 input tok/s at 12288 tokens, and it is not a harness decode ITL. It does not replace C7b 81.754 ms or H1e 60.156 ms. C2 is asymmetric (2432 vs 1242 tok/s). The 6.8 s native-QSA figure is the journal's cross-run projection, not a TTFT in this JSON. No health-before file for the R7 serve.

Suggested fix: serve hotschmoe-dev first. Report the three plain rows as prefill TTFT and prefill_tok_s. Keep native QSA, balanced C2, and 200K on the open list.

### F225 -- X3 component times and the restated 35B dispatch table

Claim: profiled C1 pad model 6827.370 ms and client 7016.851 ms. Native RS 4844.158 ms versus pad 421.479 ms. Profiled C2 all-gather 20.4-20.5 s. The loaded AG schema is three inputs. Captured QSA 3648.721 ms goes to 171.049 ms with exact output. Union mean 3509.986 max 4853. Indexer Q/K 0.676 ms plus norm/rope 2.151 ms. Glue times: fusedQK 2.205, QKVcat 2.917, sharedcat 0.2335, nativeSilu 0.1601, block1600 KVwrite 0.2147, sample/logits/argmax 0.5613 ms. Fused versus eager Q/K maxabs 0.015625 and the gate/value layout is exact. The 35B same-checkpoint 8181-token dispatch remains 180.02 to 3425.57 tok/s.

Evidence: the pad profiled X3_PREFILL record is pid 518, T=12288, wall_ms 6993.557785, model_total 6827.369896, ep_reduce_scatter 421.478806. The matching client row is ttft 7016.851336, profiled true. The native T=12288 record is pid 517, wall_ms 12900.623, ep_reduce_scatter 4844.158016, client ttft 12949.024. The two profiled C2 records have ep_allgather 20542.915 and 20402.795 ms; their client TTFTs are 26710 and 26675 ms. The payload shapes are [[12288, 2560], [12288, 512], [12288]] with dtypes bfloat16, bfloat16, bool. qsa-real-r7.log: flashnext.qsa_prefill_chunk event_ms 3648.720729, flashnext.qsa_prefill_batch1024 event_ms 171.048542, union mean 3509.986328125 max 4853. indexer_q event_ms 0.393802 plus indexer_k 0.282604 is 0.676406. indexer_query_norm_mrope event_ms 2.150625. The QSA body finished gpu-run 51s exit 0, then card 1 HEALTHY in 15s exit 0. That log has no max_abs field and no exact-output string. complete.json event_ms: fused_qk 2.205, attention_qkv_cat 2.91677, shared_gu_cat 0.23349, shared_silu_native 0.160104, kv_write_block1600 0.21474. sample_position_gather 0.152448 plus logits_float 0.142656 plus greedy_argmax 0.266197 is 0.561301. complete.log records FUSED_QK_EAGER_DIFFERENCE 0.015625 twice and FUSED_QK_GATE_VALUE_LAYOUT_EXACT, then gpu-run 53s exit 0 and card 1 HEALTHY in 15s. The same log also has two earlier "timed out after 21600s" lines. serve-ab.json arm 0 at prompt_tokens 8181 is ttft 45445.784428 and prefill_tok_s 180.016697. Arm 1 is ttft 2388.216451 and prefill_tok_s 3425.568900. That file's first id is hotschmoe-dd. The string for a 390 MB capture save was not found.

Verdict: OK on model_total 6827.370, client TTFT 7016.851, the two reduce-scatter group times, the 20.40-20.54 s all-gather group times, the three-input payload, the QSA event pair, the union stats, the indexer sum, the glue event times and the 0.5613 sum, maxabs 0.015625, and the 35B 8181-token rates. WRONG if 6827.370 is read as the client TTFT, or if 20.4 s is read as the C2 client TTFT. UNVERIFIED that the QSA output is exact, and UNVERIFIED the 390 MB gloss. The 35B table is proxy prefill with first id hotschmoe-dd. This pass did not reopen that arm's health file.

Suggested fix: label model_total, client TTFT, and ep_allgather as separate fields. Point the exact-output sentence at a max_abs line.

### F226 -- X4 live proxy gate fails closed

Claim: identity, alias, 323, 667, and C4 Paris pass. The second C1 round times out. C2 is not reached. Both long checks fail HTTP 400 because prompts 8192 and 32768 exceed server 4096. Body 21:27:40.596 to 21:28:47.376, 67s, exit 3. No ITL and no promotion.

Evidence: x4/proxy-live-gate.json started_at 2026-10-20T21:27:40.596442+00:00, finished_at 2026-10-20T21:28:47.375971+00:00, which is 66.78 s. Top-level pass is false. identity pass true, greedy_17x19 actual 323, greedy_23x29 actual 667, c4_paris pass true with 16 completion tokens, identity_after pass true, exact_repeat pass false with error "timed out", long_8192 and long_32768 errors are HTTP 400 and quote maximum context length 4096. repeat_rounds contains only c1_1, four arrays of 96 token ids. First model id is hotschmoe-dd. Second id is qwen3.6-35b-a3b-ud-iq3_xxs-x5-mtp0. x4/live-gate.log says cpu-run done in 67s exit 3. The gate JSON has no itl field.

Verdict: OK on the pass map, the four 96-id arrays, the timeout, both HTTP 400s, exit 3, and the 67s clock. WRONG if that identity pass is read as the campaign name. First id is hotschmoe-dd. This is the 35B proxy on port 18125 with max_model_len 4096. No ITL and no promotion. The stall cause is still unlocated. Source-manifest equality is not a field in this gate JSON.

Suggested fix: keep the result as a failed exact-repeat on a 4096-context proxy.

### F227 -- X5 eventless IQ3 stock control

Claim: C1 p50/p90/p99 is 8.83/9.04/9.17 ms and 115.11 tok/s. C2 p50 is 11.14/11.02 ms and stream0 p99 is 71.60 ms. PPL is 9.09554 on 2044 tokens, corpus SHA prefix efd86268...e17c46. Identity and coherence pass. The six-request repeat is exact. Card 1 health is HEALTHY before and after. Source before and after is unchanged. eventless-vs-initial-stock.json rows=true.

Evidence: x5/ud-iq3_xxs-mtp0-gdn0-ev1/bench.json. decode_c1 itl_p50_ms 8.832132, itl_p90_ms 9.044415, itl_p99_ms 9.168432, itl_mean_ms 8.829550, decode_tok_s 115.112707, n_chunks 62, completion_tokens 63, prompt_tokens 24. 1000/8.832132 = 113.22, so 115.11 is decode_tok_s. C2 itl_p50_ms 11.136452 and 11.019069. Stream 0 itl_p99_ms 71.595598. mtp.enabled false. identity ids are hotschmoe-dd then qwen3.6-35b-a3b-ud-iq3_xxs-x5-mtp0, first_ok true, expect_first hotschmoe-dd. coherence ok, same, and mentions_paris are true. models.json max_model_len 4096 and root /mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B. repeat.json identical true, concurrent_identical true, first_mismatch [None, None, None, None], row lengths 96/96/96/79 and concurrent 96/96. quality.json perplexity 9.09554145257705, scored_tokens 2044, corpus_sha256 efd86268d53e85efd67b67fcdf4846c7a4c56bc57cff41d00e3d7e4b57e17c46. The fixture block repeats revision b08601e04326c79dfdd32d625aee71d232d685c3, parquet SHA 5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91, tokenizer SHA 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42, and token-id SHA ec73501f08894e863f56b1f5a3f3cf1915d49a24299f00ff27ee7e45f713c225. Those files were not re-hashed here. health-before HEALTHY card 1, 14s exit 0. health-after HEALTHY card 1, 16s exit 0. Window rc values: GREEDY 0, REPEAT 0, BENCH 0, QUALITY 0. X5_WINDOW_DONE ready=1 is a harness flag. serve.log READY count is 0, Application startup complete is present, X1_EVENTLESS_REPLAY_ACTIVE pid=176, and b82405b4 occurs once. The word reclaim is absent. eventless-vs-initial-stock.json is exactly {"rows": true, "concurrent_rows": true}. The earlier stall arm health-after is HEALTHY card 1, 14s exit 0. Its source-before and source-after SHA maps match. The ev1 maps do not: gguf/linear.py is 47cc631935c79d10 before and ebb1d9c96fcda79b after. The 21:39 note says source before and after was unchanged.

Verdict: OK on the percentiles, decode_tok_s 115.11, the C2 p50s, stream0 p99 71.60, the PPL, 2044 tokens, the corpus prefix, repeat identity, both card-1 health lines, the compare booleans, worker 176, and the stall-arm source match. WRONG on the campaign name, because the first id is hotschmoe-dd. WRONG on "source before/after unchanged" for the ev1 window, because linear.py changed. 1000/p50 is 113.22. This is a 35B IQ3 proxy, not Flash-Next, and not a matched copy of the O3 o3g 8.760 ms figure. The b82405b4 prefix is in the log and was not re-hashed. The WikiText strings match the 21:08 note and were not recomputed. UNVERIFIED that reclaim was 0.

Suggested fix: record the linear.py change on the ev1 window. Label 115.11 as decode_tok_s.

### F228 -- X5 BF16 GDN proxy versus the stock control

Claim: C1 p50/p90/p99 is 8.69/8.89/9.02 ms and 116.83 tok/s. C2 p50 is 10.74/10.77 ms and p99 is 11.51/11.22 ms. PPL is 9.09554145 on 2044 tokens. Token parity with the stock control passes. FUSED_GDN_ACTIVE=1. Sources are unchanged. The new GDN library 2bb3661f is the only change. Post-health is still queued. The move from 8.83 ms to 8.69 ms is modest.

Evidence: x5/ud-iq3_xxs-mtp0-gdn1-ev1/bench.json. C1 itl_p50_ms 8.689149, itl_p90_ms 8.891074, itl_p99_ms 9.023454, decode_tok_s 116.832567, n_chunks 62, completion_tokens 63, prompt_tokens 24. 1000/8.689149 = 115.086. C2 itl_p50_ms 10.744196 and 10.768632. C2 itl_p99_ms 11.508576 and 11.219513. First id hotschmoe-dd, first_ok true, expect_first hotschmoe-dd. coherence ok, same, and mentions_paris are true. mtp.enabled false. quality perplexity 9.09554145257705, scored_tokens 2044, same corpus SHA as the ev1 control. repeat.json identical true and first_mismatch all None, lengths 96/96/96/79. stock-parity.json identical true, concurrent_identical true, first_mismatch all None. window.log contains FUSED_GDN_ACTIVE=1. serve.log does not. The prefix 2bb3661f occurs 0 times in both logs. health-before HEALTHY card 1, 21s exit 0. health-after.txt exists and its HEALTHY count is 0. Source SHA maps are equal inside the gdn1 arm, with linear.py ebb1d9c96fcda79b. The ev1 arm started on linear.py 47cc631935c79d10. Window rc values for greedy, repeat, bench, quality, and stock parity are 0. serve.log READY count is 0.

Verdict: OK on the percentiles, decode_tok_s 116.83, the shared PPL float, within-arm source equality, FUSED_GDN_ACTIVE=1 in the window log, the parity booleans, and card-1 pre-health. WRONG on the campaign name. WRONG that the GDN library was the only source change against the stock control, because linear.py changed between ev1 start and this arm. Post-health has not printed HEALTHY. UNVERIFIED that this process loaded SHA 2bb3661f. This is a 35B proxy, not Flash-Next. The 8.69 versus 8.83 ms pair is not a single-variable A/B. 1000/p50 is 115.09.

Suggested fix: wait for the HEALTHY line. Put the linear.py hash beside the GDN swap before treating 0.14 ms as a GDN-only delta.

### F229 -- F1 p2 prefill budget, and p1 post-health did complete

Claim: warm bf16 profile-off TTFT is 8K 4.49 s = 1822 tok/s at chunk 2048, 4.31 s = 1899 tok/s at chunk 4096, and 32K 21.3 s = 1540 tok/s plus 20.3 s = 1614 tok/s. fp32 8K with profile on is 4.62 s = 1773 tok/s. Gates 323/667 pass. The 12288-token rows read TTFT None. SERVE_RC is 0. Card 1 experts take 4.0-4.1 s, about 86 ms/layer, and card 0 waits 4.4-4.6 s. p1 health-after did not run. p2 health-before covers post-p1.

Evidence: f1/run-p2/probe.json. First id hotschmoe-dev, alias qwen3.8-flash-next-ista-gsq-rco-iq3-xxs-f1-prefill, first_ok true. Gate texts 323 and 667 match. Profile-off bf16 rows, itl null: chunk 2048 built 8186 ttft 4492.888596 rate 1821.990424; chunk 4096 built 8186 ttft 4309.802405 rate 1899.391023; chunk 2048 built 32763 ttft 21277.492319 rate 1539.796115; chunk 4096 built 32763 ttft 20295.969230 rate 1614.261415. The profile-on fp32 8K row is built 8186, ttft 5594.692550, rate 1463.172449. The profile-on bf16 8K row is ttft 4618.314948, rate 1772.507958. Four target-12288 rows are built 12283 with ttft_ms null. long_coherence texts match 323 and 667, usage prompt_tokens is 32788, and long_coherence_built is 32762. gates_by_arm has three bf16 keys and no fp32 key. window.log has SERVE_RC:0 and PROBE_RC:0. health-before is HEALTHY on cards 0 and 1, 26s exit 0. p2 health-after HEALTHY count is 0. EP-PF PROFILE T=12288 card-1 expert totals are 4004.9, 4033.5, 4126.0, and 4151.0 ms. Card-0 wait+add on those rows is 4261.5, 4378.0, 4496.7, and 4616.4 ms. Card-0 experts on T=12288 are 1168.3 to 1315.5 ms. The 1650.2 ms expert row is T=8186. 4151/48 = 86.48. f1/run-p1/health-after.txt now ends HEALTHY on cards 0 and 1, done in 31s exit 0, and it contains no parse-error line. The 63 MB and 11 GB/s gloss is not on the profile rows. The printed EP-PF rows do not carry an expert-count field.

Verdict: OK on identity hotschmoe-dev, the four warm bf16 profile-off rates, SERVE_RC 0, the bf16 gates, the null 12288 TTFTs, and p2 pre-health. WRONG that 4.62 s / 1773 tok/s is the fp32 arm. That row is profile-on bf16. The fp32 profile-on 8K row is 5.595 s and 1463 tok/s. WRONG that p1 health-after did not run. The file shows HEALTHY on both cards in 31s. The 4.0-4.1 s band misses 4151.0 ms, and the 4.4-4.6 s wait band misses 4261.5 and 4378.0 ms. These are prefill rates. The best warm rate here is 1899 tok/s, still under 2000. Not a harness decode ITL. p2 post-health has not printed HEALTHY. UNVERIFIED the byte-rate gloss and the 138/374 split as fields of these rows. Usage 32788 versus built 32762 is the same recording split as F222; the gate text still matches.

Suggested fix: label the 4.62 s row as bf16 profile-on. Cite p1 health-after as HEALTHY. Quote the T=12288 card-1 range as 4.005 to 4.151 s.

### F230 -- O5 MTP2 effective ms/token is not wall clock and not a promotion

Claim: m0b is C1 19.79 ms/tok and C2 24.32. m2 is C1 13.04 and C2 20.88, accept 72.4 percent and 59.5 percent, step p50 32.1, verify graph 24.5 at T=3, propose 3.70. m2v is C1 13.01 and C2 20.65, accept 68.2 percent and 55.9 percent, step 30.8, verify 24.5, propose 2.37. Effective ms/token equals the decode window divided by 127. Gates pass. Health is HEALTHY before and after every window. Speedup is 1.52x at C1 and 1.16x at C2. Parity fails at 111/57/none/24. Draft vocab is 40525 rows. The result is far from 7.09 ms.

Evidence: o5/run-m0b/profile.log PHASE c1 mean 19.7865 and c2 mean 24.3235. Gates text 323 and 667 match. served_model_name lists hotschmoe-dev first. Capture sizes in that kwargs line are [1, 2, 3, 4, 8]. o5/run-m2/profile.log PHASE c1 mean 13.04475 accept_mean 0.723625, c2 mean 20.883 accept_mean 0.59515, capture sizes [1, 2, 3, 4, 6, 8], speculative num_speculative_tokens 2. o5/run-m2v/profile.log PHASE c1 mean 13.0055 accept_mean 0.68185, c2 mean 20.6535 accept_mean 0.55865. graph_nt p50 is 1 on m0b and 3 on m2 and m2v. First C1 stream replay_dev p50 is 17.424, 24.478, and 24.477. propose_dev p50 is 3.696 and 2.361. step_ms p50 on that stream is 19.827, 32.082, and 30.737. 32.082 - 24.478 - 3.696 = 3.908. 3.696 - 2.361 = 1.335. 19.7865/13.04475 = 1.517. 24.3235/20.883 = 1.165. tokens JSON wall_s/127 means are m0b C1 21.963, m2 C1 15.306, and m2v C1 15.236 ms. m2 stream 0 wall_s 2.1078 gives 16.597 ms/tok against logged eff 14.448. eff matches mean step_ms times n_graph_steps, divided by 127. It does not match wall_s/127. m0b and m2 health-after files are HEALTHY on cards 0 and 1, 28s and 27s, exit 0. m2v health-after is a wait line and its HEALTHY count is 0. None of run-m0b, run-m2, or run-m2v has a health-before file, and none of those window logs contains HEALTHY. run-m2v/run.log says "b70 MTP draft head: 40525 of 248320 vocab rows". No compare artifact contains the index list 111/57. Hits on "111" in the token JSON are token ids.

Verdict: OK on the PHASE means, the acceptance rates, the gates, hotschmoe-dev, graph width 1 versus 3, the 40525-row line, and m0b/m2 post-health. WRONG that effective ms/token is wall_s/127. The table value 13.04 is the mean of the logged eff field. Wall-clock C1 means are 21.96, 15.31, and 15.24 ms per 127 tokens. WRONG that every arm used capture sizes 1,2,3,4,6,8. m0b omits 6. WRONG that every window was HEALTHY before and after. m2v after-health has not printed HEALTHY, and none of the three arms has a health-before file. UNVERIFIED the parity indices 111/57/none/24 and the claim that m2 C2 matches m0b C2 on all four prompts. This is an in-process sidecar metric, not a harness itl_p50_ms. Both the eff means and the wall means stay above the Strata 7.09 ms citation. The 1.52x ratio is the eff ratio. The wall ratio of m0b C1 to m2 C1 is 21.963/15.306 = 1.435. Not a promotion while parity is unproven and m2v teardown is open.

Suggested fix: publish wall_s/127 beside eff. Attach the compare JSON before citing mismatch indices. Close m2v health-after.

### F231 -- hygiene, and results left open

Claim: the new HEAD records the campaign speeds, including O4 decode C1 at 19.04 ms. The index has no secrets or new binaries.

Evidence: HEAD is 21c5878ed06a14614d7c27979fb9edfabd2080dc. The subject says "O4 decode C1 19.04 ms" and "Flash-Next prefill 2615 tok/s at 12288". Cached diff is empty. Porcelain is 16 rows. The only binary row is " M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so". No secret-like porcelain names. vllm/flashnext/.gitignore still lists *.so. Fork branch b70/flashnext remains 0a6170da8c, with shortstat 12 files changed, 727 insertions(+), 48 deletions(-). Added lines in the fork linear.py diff contain no synchronize, .item(), or .cpu(). prefill_ep.cpp was not line-reviewed. Not graded from a journal RESULT this pass: o4/run-f6 profile and gates, f1/run-p2 health-after (HEALTHY count 0), and the X5 IQ4 container seen only as a lease holder in the m2v health wait. WikiText strings inside quality.json were not recomputed from the parquet.

Verdict: OK that the only binary porcelain row is the known push-allreduce .so and that .gitignore ignores *.so. The commit subject calls the F220 profiler step_ms of 19.042 a decode C1 result. That number is not a harness ITL. The 2615 tok/s half matches F224 as prefill_tok_s and still has first id hotschmoe-dd. This audit made no host change.

Suggested fix: keep 19.04 ms labeled profiler step_ms wherever that commit is quoted.


## Pass 24 addendum -- 2026-10-20 22:08 UTC

The 21:58 O4 f6 note and the 22:03 X5 health note were already on disk when the pass-24 milestone was appended. They were listed as deferred there. This addendum checks them. F224-F231 are unchanged.

### F232 -- O4 f6 device-PLE profiler step is 17.434 ms

Claim: f6 C1 step goes from 19.04 ms to 17.43 ms, graph 16.55 ms, GPU idle about 0.05 ms per step. C2 goes from 22.48 ms to 20.82 ms. Gates 323/667 pass, c2 same_text, and C1 and C2 token ids match f5. c2 also matches f4. Health before and after is HEALTHY. The served path is device PLE with no host sync.

Evidence: o4/run-f6/profile.log SUMMARY c1 step_ms p50 17.434, replay_dev_ms p50 16.55, replay_host_ms p50 13.191, gap_ms p50 0.307, sample_dev_ms p50 0.824, n_graph_steps 94, n_steps 96. SUMMARY c2 step_ms p50 20.822, replay_dev_ms p50 19.658, n_graph_steps 94. 17.434 - 16.55 = 0.884. Gates text 323 and 667 match, c2 same_text=True, gates_ok=True. served_model_name lists hotschmoe-dev first. window RUN_RC:0 and HEALTH_AFTER_RC:0. health-after.txt is HEALTHY on cards 0 and 1, done in 29s exit 0. There is no health-before.txt, and the window log HEALTH_BEFORE count is 0. o4/run-f6-fail1/health-after.txt is HEALTHY on cards 0 and 1, 29s exit 0. text-c1.json and text-c2.json are byte-identical between run-f5 and run-f6 (534 and 1068 bytes). text-c2.json is also byte-identical to run-f4. text-c1.json is not byte-identical to run-f4. The profile log contains neither "0.05" nor "idle". gap_ms p50 is 0.307. The window log contains B70_PLE_DEVICE 0 times. The 26.82 GiB pin, the 122 s load, and the gemv_out bitwise rows were not opened. ple_device.py was not line-reviewed.

Verdict: OK on C1 step_ms p50 17.434, replay_dev 16.55, C2 step_ms p50 20.822, n_graph_steps 94, gates, c2 same_text, hotschmoe-dev, the f5 token-file match, the f4 c2 match, and health-after on both cards. The 19.04 to 17.43 and 22.48 to 20.82 sentences match F220's f5 profiler figures rounded against these p50s. WRONG that this successful window shows health before and after. Only health-after is in the f6 window. The failed first attempt has its own HEALTHY after file. WRONG if 0.05 ms is read from this profile: gap_ms p50 is 0.307 and step minus replay_dev is 0.884. replay_host_ms p50 is 13.191, so the replay bucket still has a large host component. UNVERIFIED the no-host-sync code claim, the env flag, the pin size, and the gemv_out equality. This is profiler step_ms, not a harness itl_p50_ms. It is not a decode promotion past the Strata 7.09 ms citation.

Suggested fix: label 17.43 ms as step_ms p50. Cite gap_ms 0.307 and replay_host_ms 13.191 beside any idle claim. Add the missing health-before file.

### F233 -- X5 IQ3 post-health closed after F228, and IQ4 pre-health is HEALTHY

Claim: the IQ3 GDN arm post-health is HEALTHY in 21s. IQ4 pre-health is HEALTHY. No new speed number.

Evidence: x5/ud-iq3_xxs-mtp0-gdn1-ev1/health-after.txt now contains xpu-health HEALTHY card 1 and gpu-run done in 21s exit 0. At the F228 read the same file still had HEALTHY count 0. x5/ud-iq4_xs-mtp0-gdn1-ev1/health-before.txt is HEALTHY card 1, done in 13s exit 0. A health-after.txt in that same directory is also HEALTHY card 1, done in 13s exit 0. The 22:03 note does not publish an IQ4 bench, and this addendum does not grade one.

Verdict: OK that the IQ3 card-1 post-health line is now HEALTHY in 21s, which closes the open item in F228. OK that the IQ4 pre-health file is HEALTHY on card 1. No new latency. The campaign name and the linear.py mismatch in F228 are unchanged. The extra IQ4 health-after file is a health result only.

Suggested fix: keep IQ4 latency off the record until its bench JSON is cited.


## Pass 25 -- 2026-10-20 22:41 UTC

Journal after the 22:08 addendum: X5 teardown and IQ4/Q4 numbers, F1 p3 OOM, X5 source correction, late-load r1/r3, and the CPU replay-binding notes through 22:40.

### F234 -- four X5 eventless 35B serves exit 139 after request checks

Claim: stock IQ3, fused IQ3, and IQ4 finish request gates and then exit 139 in queueFinish/queueRelease during finalizer teardown. The parent force-kills the worker about 5s after shutdown done. IQ4 C1 p50 is 11.02 ms, C2 p50 is 12.87/12.79 ms, PPL is 8.70159088, and the repeat passes. Q4_K_M C1 p50 is 10.96 ms, C2 p50 is 12.94/12.96 ms, PPL is 8.51682205 on 2044 tokens. Q4 is the selected four-bit candidate. A later note adds Q4 as the fourth exit-139 serve. Native post-health passes. No promotion.

Evidence: each serve.log ends gpu-run exit 139 and contains queueFinish, queueRelease, and __cxa_finalize beside a sycl::_V1::queue vector destructor. Shutdown-done to force-kill timestamps are 5s on stock IQ3, IQ4, and Q4, and 4s on fused IQ3 (21:56:40 to 21:56:44). The words stream-pool are absent. Window logs still show client rc 0 and X5_WINDOW_DONE ready=1. IQ4 bench itl_p50_ms is 11.015152, C2 12.873609 and 12.787370, decode_tok_s 92.437, 1000/p50 = 90.784. PPL 8.701590882651692, scored_tokens 2044, same corpus prefix efd86268. Repeat identical true, first_mismatch all None. Q4 itl_p50_ms 10.957061, C2 12.944775 and 12.955760, decode_tok_s 92.930, 1000/p50 = 91.265. PPL 8.516822052463134, scored 2044, same corpus prefix. Repeat identical true. All four first ids are hotschmoe-dd with first_ok true and expect_first hotschmoe-dd. Coherence ok/same/mentions_paris true. mtp.enabled false. Card-1 health-before and health-after are HEALTHY on all four arms. DEVICE_LOST count is 0. Serve READY count is 0 aside from no clock line; Application startup complete is present.

Verdict: OK on exit 139, the queueFinish/queueRelease/__cxa_finalize stack, the IQ4 and Q4 percentiles, both PPL values, 2044 tokens, repeat identity, and card-1 health. WRONG on the campaign name for every arm. The window rc does hide the 139, as the note says. UNVERIFIED the label "torch stream-pool"; the stack names a sycl::queue vector. The fused-IQ3 force-kill gap is 4s on the timestamps, not 5s. These are 35B proxy harness ITLs, not Flash-Next, and teardown failure blocks a stability promotion. 1000/p50 is 90.78 on IQ4 and 91.27 on Q4. Q4's lower PPL and similar p50 match the selection sentence as a request-level comparison only.

Suggested fix: record gpu-run exit 139 next to the p50. Keep hotschmoe-dev first. Do not promote a recipe whose serve log ends 139.

### F235 -- X5 source hashes and the late-load stub

Claim: stock IQ3 EV1 linear.py changed 47cc6319 to ebb1d9c9. IQ4 ple_device.py changed fafd61fe to 06ba1db9. Fused IQ3 and Q4 are stable inside their windows. The IQ3 0.14 ms gap is not GDN-only. r1 exits 127 because libintlc.so.5 is missing, and both health checks pass. r2 exits 126 on script permission. The clang stub is libc-only. The r2 build succeeds in 21s.

Evidence: source SHA maps match F227 and F228 for linear.py, and IQ4 ple_device.py is fafd61fe20a8 before and 06ba1db939fc after. Fused IQ3 and Q4 have no changed SHA keys. probe-r1/probe.log contains the libintlc.so.5 load error and gpu-run exit 127. exit-code.txt is 127. health-before and health-after are HEALTHY on card 1. launch-probe-r2.log says Permission denied on x5_lazy_window.sh and does not print 126. build-r2.log exits 0 in 21s and its NEEDED set includes libsvml, libirng, libimf, and libintlc.so.5. build-r4-clang.log NEEDED line is only libc.so.6, exit 0.

Verdict: OK on the two source diffs, the two stable arms, and the agreement with F227/F228. OK on r1 exit 127, the missing libintlc, and both health lines. OK that the clang stub log is libc-only. The permission failure is real; the numeral 126 is not in the launch log. The 21s success is build-r2, which is not the libc-only stub. No new serving speed in this note.

Suggested fix: cite build-r4-clang.log for the libc-only NEEDED line, and keep build-r2 separate.

### F236 -- F1 p3 tile-32 prefill dies with Level Zero error 40

Claim: p2 health-after is HEALTHY. p3 health-before is HEALTHY and READY is 22:22:36. The first 8186-token fast:32 request sees card 1 E=374, slots=64340, max_rows=5428, buckets=9. A 32-expert tile in an 8192-row bucket pads to 262144 rows and about 1.3 GB, and card 1 has 1.19 GiB free. Level Zero error 40, HTTP 500, serve rc 0.

Evidence: f1/run-p2/health-after.txt is HEALTHY on cards 0 and 1, 33s exit 0. p3 health-before and health-after are HEALTHY on cards 0 and 1, 28s and 26s, exit 0. models.json and probe.json first id is hotschmoe-dev. Gates 323 and 667 match. The failing row is arm profile=1 ret=bf16 c1=fast:32, built 8186, prompt_tokens null, prefill_tok_s null, wall_ms 2112.250511, then HTTP 500. Serve log: card 1 census E=374, T=8186, slots=64340, max_rows=5428, buckets=9, tile=32, max_padded_rows=32256. Error text is level_zero error 40 UR_RESULT_ERROR_OUT_OF_RESOURCES. Window SERVE_RC:0 and PROBE_RC:1. The only serve.log READY hit is VLLM_ENGINE_READY_TIMEOUT_S. Window READY:1 22:22:36 matches probe t0. String 262144 is absent. A separate line says card1 free=1.19 GiB at T=12288, not on the T=8186 census line. 32*8192 = 262144, and 262144*2560*2 is 1.342e9 bytes, which is a derivation rather than a logged allocation.

Verdict: OK on identity, gates, both health files, p2 post-health, the census fields, error 40, HTTP 500, and SERVE_RC 0. WRONG that the serve log shows an engine READY at 22:22:36. That clock is the harness flag at probe start. WRONG that this request's logged pad is 262144 rows. max_padded_rows is 32256. The 1.3 GB figure is not a logged byte count. The 1.19 GiB free line is real and is stamped T=12288. The 2112 ms wall has null token counts and is not a prefill rate. Not a decode ITL. The run died before the 12288 and 32768 rows.

Suggested fix: quote max_padded_rows 32256. Call 22:22:36 probe t0, not engine READY. Drop 2112 ms from any rate table.

### F237 -- X5 r3 stops in replay, and the helper on disk is a direct jmp

Claim: probe-r3 enters the stub, spins in the first graph.replay at x1_eventless_probe.py line 30, before X1_EVENTLESS_REPLAY_ACTIVE, and exits 143. Native unwind fails UNW_EBADREG. Post-health was queued. The helper wrapper is a vtable load plus jmp *0x28, an infinite tail call. A later rebuild should call XPUGraphImpl::replay directly.

Evidence: probe-r3/python-stack.txt shows replay at torch/xpu/graphs.py:107 called from x1_eventless_probe.py:30, under x5_lazy_probe.py. probe.log contains X5_LAZY_EVENTLESS_STUB_ACTIVE, gpu-run done in 99s exit 143, and zero hits for X1_EVENTLESS_REPLAY_ACTIVE, 2066, and UNW_EBADREG. exit-code.txt is 143. health-after HEALTHY count was 0 at this read. objdump of libx5_eventless_helper.so symbol b70_x5_eventless_impl is `e9 db fa ff ff` then `jmp 2370 <_ZN2at3xpu12XPUGraphImpl6replayEv>`. That is a direct jmp to the mangled replay symbol, not a vtable indirect jmp.

Verdict: OK on the Python stack, the stub-active line, exit 143, and the absence of X1_EVENTLESS_REPLAY_ACTIVE in the probe log. UNVERIFIED UNW_EBADREG, which is not in probe.log. The current helper does not contain the vtable sequence in the note. That sequence was not saved beside the note, so the earlier binary is UNVERIFIED. The file now on disk already jumps straight at XPUGraphImpl::replay. No numerical pass and no speed. Post-health had not printed HEALTHY.

Suggested fix: keep the objdump of the failing binary next to the vtable sentence. Re-disassemble before calling r5 qualified.

### F238 -- hygiene, and the Q8 byte estimate

Claim: Q4 shared pack is refused because the loader dequants Q8_0. There are 257 Q8 block projections, 2871001088 BF16 bytes versus 1525219328 packed bytes. That is storage accounting, not a latency. The repo has no new secret or binary.

Evidence: those three numerals were not found in x5 logs or JSON at this pass. HEAD is 6d08fbf4. Cached diff is empty. Porcelain is 27 rows. The only binary row is the known push-allreduce .so. No secret-like names. vllm/flashnext/.gitignore still lists *.so. Fork HEAD remains 0a6170da8c, shortstat 12 files changed, 749 insertions(+), 48 deletions(-).

Verdict: UNVERIFIED the 257-tensor and byte totals. OK that no speed was claimed for that accounting. OK that porcelain still has only the one known binary and no secret names. This audit made no host change.

Suggested fix: write the tensor count and the two byte totals into the log line that computed them.

## Pass 26 -- 2026-10-20 23:27 UTC

### F239 -- O4 f7 profiler step is 16.681 ms, and the host bucket is still 12.31 ms

Claim: f7 is f6 plus B70_GEMV_BF16=1. Gates pass, C2 same_text, and C1/C2 token ids match f6. C1 step is 16.68 ms with graph 15.80 ms. C2 step is 19.92 ms with graph 18.78 ms. Health is HEALTHY around both runs. The GPU is fully busy.

Evidence: o4/run-f7/profile.log C1 step_ms p50 is 16.681, mean 16.548, replay_dev_ms p50 15.8, replay_host_ms p50 12.31, gap_ms p50 0.317, exec_ms p50 13.908, n_graph_steps 94, n_steps 96. C2 step_ms p50 is 19.92, replay_dev_ms p50 18.781, replay_host_ms p50 15.375, gap_ms p50 0.372, exec_ms p50 16.988. gates.json matches 323 and 667. profile.log says c2 same_text=True and gates_ok=True. text-c1.json and text-c2.json are each 96 ids and byte-identical to o4/run-f6. The sidecar kwargs list served_model_name with hotschmoe-dev first. There is no /v1/models body in the run directory. run-f7 and run-f7ut have no health-before.txt. Both health-after.txt files contain one HEALTHY line for cards 0 and 1. The f7 after-probe waited on other holders and then gpu-run exited 0 in 28s. Window RUN_RC is 0.

Verdict: OK for the step_ms and replay_dev_ms figures, the gates, the token-id match, and the kwargs name. OK that health-after is HEALTHY. WRONG that health was taken around the run: the before file is absent. WRONG that the GPU is fully busy: the same C1 summary has replay_host_ms p50 12.31, and step_ms minus replay_dev_ms is 0.881. exec_ms p50 13.908 is a different field from the 16.681 step. This is an in-process profiler step_ms. It is a harness ITL only if a bench JSON itl_p50_ms says so, and this run has no such field. It stays above the Strata 7.09 ms citation. No promotion.

Suggested fix: publish step_ms, replay_dev_ms, and replay_host_ms together, and add a health-before file for the same window.

### F240 -- f7ut kernel sum is 16.539 ms over a 22.38 ms span; top-k log is 0.281 vs 0.548; f8 has unjournaled steps

Claim: card 0 runs 2259 kernels/step, down from 4827 at ringmlock. The kernel sum is 16.54 ms, including lm_head 0.77, gemm 2.30, dense GGUF 5.1, bf_gemv 1.51, MoE about 3.1, TopK 0.61, GDN 0.72, ep_wait 0.34, q8 0.30, and casts 0.33. Triton top-k is 0.268 ms versus 0.551 ms over 48 calls, with the same expert ids on 600 random cases and weights within 1-3 ulp. f8 was still queued.

Evidence: o4/run-f7ut/step-dump.txt has 2259 parsed rows. The duration column sums to 16538.5 us, which is 16.5385 ms. The last timestamp is 22.380 ms, matching the untraced f7ut C1 step_ms p50 of 22.38, not the f7 16.681 ms run. bf_gemv_k is 97 calls and 1.515 ms. gemm_kernel is 219 calls and 2.296 ms. TopKGating is 48 calls and 0.610 ms. GDN conv plus delta is 0.716 ms. b70_q8_quant_k is 0.297 ms. ep_wait is 48 calls and 0.233 ms. One b70_kreg_k<13> call is 769.75 us. b70_kreg_k plus b70_dense_k sums to 5.087 ms, and that bucket includes the 0.770 ms call. The listed b70_sched kernels sum to about 3.20 ms. o4/run-ut2/trace-summary.txt contains the standalone token 4827 on long steps such as 36.8/4827. o4/run-ringmlock steps JSON has no standalone 4827. o4/topk.log says "48 calls: xpu 0.548 ms triton 0.281 ms", id_mismatch 0/200, w_mismatch 200/200 or 187/200, and max ulp 1, 2, or 3. gpu-run exited 0. The numerals 0.268, 0.551, and 600 are not in that log. o4/run-f8/profile.log C1 step_ms p50 is 16.43, replay_dev_ms 15.593, replay_host_ms 12.298. C2 step_ms p50 is 19.7, replay_dev_ms 18.575. Gates match. Token ids match f7. health-after is HEALTHY. There is no health-before. The kwargs name starts with hotschmoe-dev. No journal RESULT heading states the 16.43 figure.

Verdict: OK for 2259 kernels and the 16.5385 ms duration sum on the f7ut dump. OK for bf_gemv 1.51, gemm about 2.30, TopK 0.61, GDN 0.72, q8 0.30, and the 0.770 ms lm_head call. The 97/48/72 gemm split is not labeled in the dump. WRONG to place 4827 on ringmlock; that count is in the ut2 trace summary. WRONG for ep_wait 0.34 against the ep_wait column sum 0.233. WRONG for the top-k pair and the 600-case count; the saved log is 0.281 ms triton versus 0.548 ms xpu on 200 rows, with ulp at most 3 and weights not bitwise. The 460 GB/s figure is not in the dump. The f8 queue sentence is stale. f8's 16.43 ms is a profiler step_ms with replay_host_ms still 12.298. It is not a harness ITL and not a promotion.

Suggested fix: cite step-dump.txt for the 16.539 ms busy sum and 22.380 ms span, cite topk.log's 0.281/0.548 line, and journal f8's step_ms next to replay_host_ms before anyone uses 16.43.

### F241 -- F1 p4 prefill table matches probe.json; the 2280 heading is high; three cells clear 2000

Claim: one p4 load, READY 22:43:05, health before and after HEALTHY, gates 323/667 on every arm. Profile-off warm rates are fast:32/16384 at 2223/1961/1936, fast:16/16384 at 1946/1805/1794, fast:32/32768 at 2278/2040/1991, and fused 4096 at 1811/1624/1610 tok/s on built 8186/12283/32763. The recipe band is 2040-2280 tok/s. Card-1 census is 374 experts, 64204 slots, hottest 5426, 9 buckets, 13 calls, tile 32. Card 1 has 1.19 GiB free and no OOM at maxpad 32768.

Evidence: f1/run-p4/probe.json t0 is 22:43:05. Identity ids start with hotschmoe-dev, first_ok true. models.json first id is hotschmoe-dev, max_model_len 147456, root hf-flashnext. Profile-off prefill_tok_s values are 2222.954, 1960.760, 1935.843, 1945.604, 1804.809, 1793.827, 2277.886, 2039.846, 1990.741, 1810.957, 1623.988, and 1609.543. Each tok/s equals built/ttft_s. The journal seconds are the ttft values rounded to 0.01 s. itl_p50_ms is null on every row. gates_by_arm matches 323 and 667 on all four profile-off arms. Long coherence matches 323 and 667, usage prompt_tokens 32788, long_coherence_built 32762. health-before and health-after are HEALTHY for cards 0 and 1, gpu-run exit 0 in 29s and 27s. Window READY:1 22:43:05 equals t0. serve.log contains one READY hit, on the docker line as VLLM_ENGINE_READY_TIMEOUT_S, and it also contains Application startup complete. Census line: E 374, T 8186, slots 64204, max_rows 5426, buckets 9, calls 13, tile 32, row_limit 16384, max_padded_rows 16320. card1 free=1.19 GiB is stamped T=12288. OUT_OF_RESOURCES count is 0. The 2.25-2.5 s, 4.0-4.1 s, 1.27-1.55 s, 0.34 s, and 0.27 s profile budget is not in probe.log or serve.log.

Verdict: OK for the rounded table, the identity, the gates, both health files, the census line, and the absence of an OOM. WRONG to read READY 22:43:05 as an engine READY clock; it is the probe t0. WRONG for the heading upper bound 2280: the JSON cell is 2277.886, and the body already says 2278. The cells at or above 2000 are 2222.954, 2277.886, and 2039.846. The 32763-token cell is 1990.741. These are prefill_tok_s values. X3's 2615 at 12288 stays the higher dense-GQA prefill citation. The expert-time budget and the 1.7 s leftover are UNVERIFIED. Not a harness decode ITL.

Suggested fix: title the measured band 2039.846-2277.886 and keep the sub-2000 cells in the same table. Call 22:43:05 the probe t0.

### F242 -- X5 r5 replay exits 0; the vtable bytes are in a different helper; stream-pool is still absent

Claim: probe-r5 passes 2066 exact replays and RNG, RSS 708988 to 751284 KiB, process exit 0 in 15s, with the helper sha b2668e59 and the libc-only stub 425788b2. Pre-health is HEALTHY and post-health later is HEALTHY. reports/1X5_proxy35b.md says the r3 failure disassembly confirmed a virtual call, and that the four early serves died in Torch stream-pool destruction.

Evidence: x5/eventless-lazy/probe-r5/probe.log prints X1_EVENTLESS_REPLAY_AND_RNG_OK 2066 rss_kb 708988 751284 delta 42296, then X5_LAZY_PROBE_NORMAL_PYTHON_EXIT, then gpu-run done in 15s exit 0. exit-code.txt is 0. health-before and health-after are HEALTHY for card 1, each gpu-run exit 0 in 15s. SHA256 of libx5_eventless_helper.so is b2668e593652a74eec46be295958ad6d0ff5b21b50591b28713e0ae641f8c4a7. SHA256 of libx5_eventless_stub.so is 425788b21e86565665ddef1d5f72581c9e62fd25c70d26ffaca7560462e46d6a, and build-r5 records NEEDED libc.so.6. objdump of b70_x5_eventless_impl in that helper is e9 db fa ff ff, a direct jmp to XPUGraphImpl::replay. helper-r5-virtual-call.so is a different file, sha 01dd90ac5ad4d3194d704c0658c3da9b2c3338aca9360ef5c5cbcacac505dd78, and its same symbol is mov (%rdi),%rax; jmp *0x28(%rax). probe-r3's logs do not contain that disassembly. A search of x5 text logs found zero stream-pool, stream_pool, or StreamPool hits.

Verdict: OK for the r5 exit, the 2066 count, the RSS delta, both card-1 health files, and both hashes. The passing binary is the direct jmp. The vtable sequence is real in helper-r5-virtual-call.so and is not stored in the probe-r3 logs, so the report's "confirmed in probe-r3" locator is UNVERIFIED. WRONG for the stream-pool wording. This probe loaded no model. It is not a serving ITL and not a speed.

Suggested fix: attach the virtual-call objdump to helper-r5-virtual-call.so, and drop stream-pool unless a log line contains it.

### F243 -- Q4 late-load serve exits 0, first id is hotschmoe-dd, and the C2 repeat fails at index 34

Claim: q4-lazy-control has a clean serve exit 0, no worker force-kill, HEALTHY pre/post checks, and unchanged source hashes. C1 p50 is 10.9098 ms, C2 is 12.9485/13.0183, PPL is 8.51682205. C1 repeat is exact. C2 Paris differs at token 34 and Fibonacci is exact. A later old-baseline comparison passes all 6 streams. Window gate_rc is 3. The recipe is not qualified. The deferred trace and candidate controllers exited 3 without GPU.

Evidence: x5/q4-lazy-control/bench.json tag X5-q4-lazy-control, mtp.enabled false, first id hotschmoe-dd, expect_first hotschmoe-dd, first_ok true, max_model_len 4096, root /mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B. C1 itl_p50_ms is 10.909757926128805, decode_tok_s 93.25107012164695, n_chunks 57, completion 58. C2 itl_p50_ms is 12.948463438078761 and 13.01828701980412. quality.json perplexity is 8.516822052463134, scored_tokens 2044, corpus_sha256 starts efd86268d53e85ef. serve-exit-code.txt is 0. serve.log ends gpu-run done in 155s exit 0, with Application startup complete, zero READY hits, zero force-kill, and zero exit 139. health-before and health-after are HEALTHY for card 1, 13s and 21s, exit 0. source-before.json equals source-after.json. Window REPEAT_RC=3, BENCH_RC=0, QUALITY_RC=0, STOCK_PARITY_RC=0, gate_rc=3. repeat.json identical is false and concurrent_identical is false, while first_mismatch is [None, None, None, None]. Concurrent Paris token ids differ from the sequential Paris ids at index 34, and Fibonacci matches. stock-parity.json identical and concurrent_identical are true, which is 4 sequential rows plus 2 concurrent rows. Inside that same file, concurrent Paris still differs from sequential Paris at index 37. x5/q4-packed-q8-launch-r2.log says the candidate was cancelled before GPU and prints no exit code. The trace exit is covered in F245; it is not an exit 3.

Verdict: OK for the p50 pair, the PPL, the corpus prefix, serve exit 0, card-1 health, source-before equals source-after, gate_rc 3, and the index-34 Paris mismatch. OK that the journal refuses qualification. WRONG for the campaign name: the first /v1/models id is hotschmoe-dd. The report documents that name as the proxy primary. The first_mismatch array does not store 34, so that field alone hides the failure. The stock-parity flags pass against the baseline, and they do not mean C1 and C2 Paris tokens match each other. UNVERIFIED that a controller exited 3 without using the GPU. This window is separate from F234's Q4 serve exit 139. It is a 35B proxy harness ITL. It is not matched to O3's 8.760 ms and not a promotion. 1000/10.9098 is 91.66, so 93.251 is decode_tok_s.

Suggested fix: record concurrent_first_mismatch 34 next to REPEAT_RC=3, and put hotschmoe-dev first in /v1/models.

### F244 -- F1 p5 profile-off prefill clears 2000 at 8K, 12K, and 32K; post-health is still a wait; decode ITL is 18.467 ms

Claim: p5 health before and after is HEALTHY. Gates 323/667 pass on all 4 arms. Profile-off push=main / push=side TTFT is 1.33/1.34 s (763 tok/s) at 1018, 3.57/3.57 s (2295/2293) at 8186, 5.97/6.00 s (2059/2046) at 12283, 16.33/16.34 s (2006/2005) at 32763, and 93.5/93.5 s (1402) at 131067. Decode C1 ITL p50 is 18.47 ms and C2 is 22.73/22.73 ms, with O4 f4 unchanged. READY is 23:10:50. The native library sha starts 790230cb. 2000 tok/s holds at 8K, 12288, and 32K. 128K is 1402.

Evidence: f1/run-p5/probe.json ok true, t0 23:10:50, first id hotschmoe-dev, first_ok true. Profile-off prefill_tok_s is main 763.434, 2295.512, 2058.690, 2006.145, 1401.885 and side 761.368, 2293.806, 2045.880, 2004.532, 1401.818, on built 1018, 8186, 12283, 32763, 131067. ttft_ms for those ten rows is 1333.449, 3566.089, 5966.416, 16331.320, 93493.432 and 1337.068, 3568.742, 6003.775, 16344.461, 93497.883. itl_p50_ms is null on the prefill rows. decode_c1 measured itl_p50_ms is 18.466882989741862, decode_tok_s 54.075154961856555, completion_tokens 64, n_chunks 64. decode_c2 itl_p50_ms is 22.734455997124314 and 22.726351977325976. coherence ok, same, and mentions_paris are true. Top-level gates match 323 and 667. gates_by_arm has three keys, all matching; profile=1 push=main is not a separate key. Long coherence matches, usage prompt_tokens 32788, built 32762. health-before is HEALTHY for cards 0 and 1, gpu-run exit 0 in 29s. At this read, f1/run-p5/health-after.txt is a one-line gpu-run wait behind O4 f9 (vllm-fn-o4-f9) and contains zero HEALTHY lines. Window READY:1 23:10:50 equals t0. The only serve.log READY hit is VLLM_ENGINE_READY_TIMEOUT_S. Application startup complete is present. SERVE_RC and PROBE_RC are 0. SHA256 of f1/_C.f1bucket.so starts 790230cb027b56d5. Sampled EP-PF lines at T=12288 show c0_dispatch_push about 272 ms, c0_shared about 94 ms, c0_wait_add about 365 ms, c1_experts about 2028-2355 ms, and c0_between_layers values from 1925 ms up to 7909 ms. The string .cpu() is not in the serve log.

Verdict: OK for the profile-off tok/s table within ordinary rounding, including side 761 rather than a second 763, and for 1401.885/1401.818 at 128K. OK that 8K, 12283, and 32763 profile-off cells are above 2000 on both push modes, and that 1018 is about 763. OK for identity, coherence, the recorded gates, and the library sha prefix. WRONG that post-health is already HEALTHY. Pre-health is HEALTHY. WRONG to read 23:10:50 as an engine READY clock. The 18.467 ms figure is a harness itl_p50_ms on 64 completion tokens. O4 f4's closed profiler step was 19.775 ms, so "f4 unchanged" does not make 18.47 an f4 step_ms. decode_tok_s is 54.075. That decode rate stays far from the Strata 141 tok/s citation. The prefill rates do not replace X3's 2615 at 12288. The .cpu() cause of host_c0 is UNVERIFIED. p6 is queued in the journal and is not a result here.

Suggested fix: leave post-health open until health-after.txt prints HEALTHY, and label 18.467 ms as the 64-token sidecar harness ITL.

### F245 -- unitrace session x5q4-lazy-trace exits 1 before load; repo hygiene

Claim: the q4-lazy-trace diagnostic exits 1 before model load because unitrace rejects session x5q4-lazy-trace. Pre-health is HEALTHY and post-health is queued. There is no latency. Canonical MoE is wired default off. The repo has no new secret or binary from this audit.

Evidence: x5/q4-lazy-trace/run.log line 46 is "Invalid value for --session: x5q4-lazy-trace", and the same log's help text says the session argument is an alphanumeric string. gpu-run done in 27s exit 1. process-exit-code.txt is 1. No itl or tok/s line is in that log. health-before is HEALTHY for card 1, 15s exit 0. health-after is a wait behind O4 and contains zero HEALTHY lines. source-before equals source-after. x5/q4-lazy-trace-r2 has a health-before wait and no bench file. harness/x5_overlay/sitecustomize.py calls x5_canonical_moe.install only when B70_X5_CANONICAL is 1. The wrapper chunks T>8 TP1 apply by 8, returns early for T<=8 or a non-None expert_map, and has no synchronize, .item(), or .cpu(). _store_shared assigns _output[_output_idx] and does not increment the index, which is what the wrapper's read expects. Fork xpu_communicator.py adds torch.xpu.synchronize under B70_PREFILL_EP_DRAIN before dist.barrier and under B70_PREFILL_EP_SYNC on the prefill return. HEAD is 929563aa. Cached diff is empty. Porcelain is 20 rows. Binary or secret-like porcelain rows:  M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so. vllm/flashnext/.gitignore still lists *.so. Fork HEAD is 0a6170da8c on b70/flashnext, 12 files changed, 789 insertions(+), 49 deletions(-). Deferred headings at append: none.

Verdict: OK for exit 1, the invalid session name, no latency, and pre-health HEALTHY. OK that post-health is still queued rather than passed. OK that canonical install is opt-in and unwired unless B70_X5_CANONICAL=1. The new synchronize calls are opt-in host fences on the prefill EP path. They are not an unconditional sync inside decode graph replay. OK that this audit made no host change. Q8 byte totals from F238 stay UNVERIFIED.

Suggested fix: pass a session string of letters and digits, and keep the synchronize env flags off any FULL decode capture.
## Pass 27 -- 2026-10-21 00:11 UTC

### F246 -- O4 f8 and f9 profiler steps are 16.43 ms and 16.295 ms; replay_host stays about 12.3 ms

Claim: f8 is f7 plus B70_DECODE_TOPK=1 and B70_LINEAR_CAT=1. f9 adds B70_DENSE_GEMV=1 and B70_EP_BF16_OUT=1 with o4/b70_ep_C.o4c.so sha 10495833, and that sum equals an fp32 add then a cast. Gates pass, C2 same_text, and C1 and C2 token ids match f7 over 96 tokens. f8 C1 is 16.43 ms with graph 15.59 ms and C2 is 19.70 ms. f9 C1 is 16.30 ms with graph 15.44 ms and C2 is 19.53 ms. Health is HEALTHY. Int8 HC GEMVs floor near 12 us (down 11.9 vs bf16 15.5, up 12.0 vs oneDNN 13.4), and a 4-deep unroll was reverted.

Evidence: o4/run-f8/profile.log C1 step_ms p50 is 16.43, replay_dev_ms p50 15.593, replay_host_ms p50 12.298, gap_ms p50 0.316, exec_ms p50 13.863, n_graph_steps 94, n_steps 96. C2 step_ms p50 is 19.7, replay_dev_ms 18.575, replay_host_ms 15.441, gap_ms 0.365, exec_ms 17.02. o4/run-f9/profile.log C1 step_ms p50 is 16.295, replay_dev_ms 15.435, replay_host_ms 12.289, gap_ms 0.32, exec_ms 13.863. C2 step_ms p50 is 19.526, replay_dev_ms 18.367, replay_host_ms 15.153, gap_ms 0.37, exec_ms 16.721. Both runs print c2 same_text=True and gates_ok=True. gates.json matches 323 and 667. text-c1.json and text-c2.json are each a one-element list whose inner list has 96 ids and is byte-identical to o4/run-f7. Sidecar kwargs served_model_name starts with hotschmoe-dev. There is no /v1/models body. Both health-after files are HEALTHY for cards 0 and 1, gpu-run exit 0 in 26s and 29s. Neither run has health-before.txt. SHA256 of o4/b70_ep_C.o4c.so starts 1049583379fc2d88 and the file is 554080 bytes. w8f.log contains "down M=1 ks=8 11.89 us". w8e.log contains "up M=1 onednn bf16 13.41 us". No w8 log line pairs 11.9 us with bf16 15.5 or 12.0 us with oneDNN 13.4, and no unroll-revert line was found.

Verdict: OK for the step_ms and replay_dev_ms rounding, the gates, the 96-id match, the kwargs name, the sha prefix, and health-after. WRONG that health was taken around the runs: the before file is absent. WRONG that these steps show a busy GPU: C1 replay_host_ms p50 is 12.298 and 12.289. The "equal to fp32 add then cast" sentence and the exact 11.9/15.5 and 12.0/13.4 pairing are UNVERIFIED. These are in-process profiler step_ms values. They are harness ITLs only if a bench JSON itl_p50_ms says so. They stay above the Strata 7.09 ms citation. No promotion.

Suggested fix: publish step_ms with replay_dev_ms and replay_host_ms, and add a health-before file for the same window.

### F247 -- f10 finished at 16.31 ms and f11 at 15.698 ms; neither has a journal RESULT; f11 post-health is open

Claim: the 23:35 note queued f10 (B70_EP_Q0=150) and f11 (B70_DECODE_HC_Q8=1). No RESULT heading gives their step times.

Evidence: o4/run-f10/run.log sets B70_EP_Q0=150, B70_DENSE_GEMV=1, B70_EP_BF16_OUT=1, and B70_EP_SO to o4/b70_ep_C.o4c.so. It does not set B70_DECODE_HC_Q8. profile.log C1 step_ms p50 is 16.31, replay_dev_ms 15.455, replay_host_ms 12.072, exec_ms 13.583, n_graph_steps 94, n_steps 96. C2 step_ms p50 is 19.544, replay_dev_ms 18.377, replay_host_ms 15.23. gates match 323 and 667. c2 same_text=True. Token files match f7, inner length 96. health-after is HEALTHY for cards 0 and 1, gpu-run exit 0 in 35s. There is no health-before. o4/run-f11/run.log adds B70_DECODE_HC_Q8=1 on that same flag set. profile.log C1 step_ms p50 is 15.698, replay_dev_ms 14.85, replay_host_ms 11.316, exec_ms 12.893. C2 step_ms p50 is 18.926, replay_dev_ms 17.864, replay_host_ms 14.183, exec_ms 15.956. Gates match, c2 same_text=True, token files match f7, and RUN_RC is 0. Kwargs served_model_name starts with hotschmoe-dev on both runs. f11 health-after is a gpu-run wait with zero HEALTHY lines. There is no health-before.txt.

Verdict: OK that the raw files contain those step_ms values and that f11's 96 ids still match f7 with the Q8 flag on. The 23:35 queue sentence is stale for both tags. WRONG to treat either step as a harness ITL or as a promotion. f10 C1 16.31 ms is flat beside f9's 16.295 ms, and replay_host_ms is still 12.072. f11 C1 15.698 ms still has replay_host_ms 11.316 and no closed pre-health. No promotion.

Suggested fix: journal f10 and f11 step_ms next to replay_host_ms, and leave f11 post-health open until health-after.txt prints HEALTHY.

### F248 -- X5 r2 paused-unitrace step is 11.748 ms with an 8.183 ms host gap; the API 10.91 ms figure is a different file

Claim: q4-lazy-trace-r2 exits 0. Arithmetic 323/667 passes. C1 and C2 differ at index 26. Twenty-three decode-like spans have 1185 kernels, span p50 11.92 ms, busy 10.70 ms. GEMMs sum 6.188 ms. Cat is 0.304 ms, casts about 0.232 ms, routed GU 0.864 ms, down about 0.631 ms, GDN 0.338 ms, and head Q6K 0.709 ms. With collection paused, C1 step is 11.748 ms and replay is 11.093 ms, and C2 step is 12.959 ms and replay is 11.864 ms. The host gap of 8-9 ms is an asynchronous completion wait. The API control ITL of 10.91 ms is separate. Profile exit 0 skips finalizers. Pre and post health are HEALTHY.

Evidence: unitrace-session.txt is x5q4lazytracer2. process-exit-code.txt is 0. gates.json passes 323 and 667. batch-parity.json is two rows, each identical false with first_mismatch 26. trace-summary.txt says 23 decode-like steps with 1185 kernels, span p50 11.92 ms, busy p50 10.70 ms. Kernel rows: gemm_kernel 6.188 ms n=390; b70_sched_gu_k<12> 0.864 ms n=40; b70_kreg_k<14, 1, 4> 0.709 ms n=1; b70_sched_down_k<13> 0.569 ms n=37; b70_sched_down_k<14> 0.062 ms n=3; launch_decode<false> 0.338 ms n=30; CatArrayBatchedCopy 0.304 ms n=110; UnrolledElementwise CopyScalar float 0.232 ms n=41. summary-c1.json step_ms p50 is 11.748, replay_dev_ms 11.093, replay_host_ms 0.352, gap_ms 8.183, exec_ms 2.038, n_steps 96, n_graph_steps 94. summary-c2.json step_ms p50 is 12.959, replay_dev_ms 11.864, replay_host_ms 0.358, gap_ms 9.407. summary-c1-traced.json step_ms p50 is 25.147 on 24 steps, which is the traced summary and not the 11.748 figure. profile.log records unitrace pause before the untraced summaries. run.log gpu-run exits 0 in 143s, with zero force-kill and zero exit 139. The substring finaliz has zero hits. health-before and health-after are HEALTHY for card 1, each gpu-run exit 0 in 15s. This directory has no itl_p50_ms field.

Verdict: OK for the exit, the gates, the mismatch index, the kernel rows, the paused step_ms pair, and both card-1 health files. The down sum 0.569+0.062 is 0.631. The 0.338 ms row is named launch_decode, and the 0.709 ms row is named b70_kreg_k<14>, so the GDN and Q6K labels are not printed. The "skips finalizers" sentence is UNVERIFIED. gap_ms 8.183 and 9.407 match the 8-9 ms host gap; calling that gap an asynchronous completion wait is an interpretation, and replay_host_ms is 0.352. The 10.91 ms API ITL is the F243 q4-lazy-control bench, a different file. This is a 35B proxy trace. It is a Flash-Next harness ITL only if a bench JSON says so. No promotion.

Suggested fix: cite launch_decode and b70_kreg_k by the names in trace-summary.txt, and keep the 10.91 ms ITL on the q4-lazy-control bench.

### F249 -- packq8-canon C1 ITL is 15.650 ms, the first id is hotschmoe-dd, and the repeat passes

Claim: qwen36-35b-udq4km-packq8-canon-mtp0 has an exact C1 and C2 repeat, identity and coherence, exit 0, and pre and post health. Prefix PPL is 8.50822826. C1 p50 is 15.6498 ms and C2 is 17.5697/17.5643. That is slower than the 10.91 ms control, so the speed recipe is rejected. The DP4A library sha is 51f304b431db82cd6cde1b6591c5caa19dd4184766a018eef615dac2068ce17c from a 29s CPU build. No DD, reset, or commit.

Evidence: bench.json tag X5-qwen36-35b-udq4km-packq8-canon-mtp0, model hotschmoe-dd, mtp.enabled false, expect_first hotschmoe-dd, first_ok true. coherence ok, same, and mentions_paris are true. C1 itl_p50_ms is 15.649834997020662, decode_tok_s 65.01586614411876, n_chunks 56, completion_tokens 57, ttft_ms 81.75224298611283. C2 itl_p50_ms is 17.569674062542617 and 17.564272973686457. quality.json perplexity is 8.508228260209568, scored_tokens 2044, corpus_sha256 starts efd86268d53e85ef. repeat.json identical and concurrent_identical are true, first_mismatch is None four times. models.json first id is hotschmoe-dd, second id qwen3.6-35b-a3b-ud-q4_k_m-x5-mtp0, max_model_len 4096, root /mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B. serve-exit-code.txt is 0. serve.log has Application startup complete, zero READY hits, zero force-kill, zero exit 139, and gpu-run exit 0 in 124s. Window gate_rc is 0 with REPEAT_RC, BENCH_RC, and QUALITY_RC all 0. dd-state.txt is inactive. health-before and health-after are HEALTHY for card 1, 15s exit 0. source-before.json equals source-after.json. x5/q8-fast/build-r1.log prints that sha and cpu-run exit 0 in 29s. The packq8 serve log does not print 51f304.

Verdict: OK for the ITL pair, the PPL, the corpus prefix, the repeat, serve exit 0, and card-1 health. OK that the author rejects this recipe against the 10.91 ms control, and that control is the F243 q4-lazy-control bench, a different recipe, so the slowdown is a direction and not a one-knob A/B. WRONG for the campaign name: the first /v1/models id is hotschmoe-dd. The harness expect_first matches that name. 1000/15.6498 is about 63.9, so 65.016 is decode_tok_s. The 51f304 sha is the separate CPU build. This serve is not shown to have loaded that .so. This is a 35B proxy harness ITL. It stays above the Strata 7.09 ms citation. No promotion.

Suggested fix: put hotschmoe-dev first in /v1/models, and keep the DP4A sha on the build log until a serve line prints it.

### F250 -- fastq8 C1 ITL is 10.225 ms on the 35B proxy; the 34.7 percent drop changes more than the new .so

Claim: qwen36-35b-udq4km-fastq8-canon-mtp0 passes identity, coherence, an exact C1 and C2 repeat, exit 0, and pre and post health. The probe is 10/10 with FP32 maxabs 0, an exact BF16 epilogue, batch M 1/2/3/4/8, and captured replay, exit 0, health around it. Cached projection N=12288 K=2048 is M1 255.47 to 40.15 us and M3 184.60 to 31.12 us, and those are hot-loop times. Full-model C1 p50 is 10.2246 ms and C2 is 11.5000/11.4909. PPL 8.50822826 is unchanged. Source hashes are stable inside each window and the fork hash matches across them. C1 improves 34.7 percent versus the generic Q8 15.6498 ms. It is a finite Q4 control with no DD promotion.

Evidence: bench.json C1 itl_p50_ms is 10.224584955722094, decode_tok_s 99.8052754562704, n_chunks 56, completion_tokens 57. C2 itl_p50_ms is 11.500005959533155 and 11.490878998301923. mtp.enabled false. first id hotschmoe-dd, expect_first hotschmoe-dd, first_ok true, max_model_len 4096, root the 35B A3B path. coherence ok, same, and mentions_paris are true. repeat identical and concurrent_identical are true. quality.json perplexity is 8.508228260209568, scored_tokens 2044, corpus prefix efd86268d53e85ef. That PPL scalar matches the packq8 file, and the two quality.json sha256 prefixes differ (5c4ccdb08fa5f029 versus ff33544673a28639). serve-exit-code.txt is 0. serve.log gpu-run exits 0 in 128s, Application startup complete, zero READY, zero force-kill, zero exit 139. The docker line sets B70_X5_Q8_FAST=1 and B70_X5_SO to x5/q8-fast/libb70_x5_xpu.so. It does not print sha 51f304. Window gate_rc is 0. dd-state.txt is inactive. health-before and health-after are HEALTHY for card 1, 16s and 15s, exit 0. source-before equals source-after inside the window. fork_head is 0a6170da8ca9991972322fa1ca1d27674f0e75cd in both the packq8 and fastq8 source files. The sha maps share 36 keys, differ on sitecustomize.py and x5_serve_proxy.sh, and the fastq8 map adds libb70_x5_xpu.so, x5_q8_fast.py, and q8_linear.cpp. The only environment key added is B70_X5_Q8_FAST. probe-r1/result.json has 10 rows, every max_abs 0.0, batch_exact true, graph_exact true. The N=12288 K=2048 M=1 timings are 0.255471 ms and 0.040148 ms. M=3 is 0.184602 ms and 0.031122 ms. process-exit-code.txt is 0. probe.log prints X5_Q8_PROBE_NORMAL_EXIT and gpu-run exit 0 in 36s. Probe health-before and health-after are HEALTHY for card 1, 19s and 14s. The result file does not contain the strings FP32 or BF16. q8_linear.cpp dp4() multiplies four int8 lanes with scalar products. The source has no synchronize, .item(), or .cpu(). TORCH_CHECK requires M in 1..8, K divisible by 32, and contiguous 2D inputs. The dispatch switch has no default after that check.

Verdict: OK for the harness ITL pair, the 34.7 percent arithmetic on itl_p50 against 15.649834997020662, the PPL scalar, the repeat, serve exit 0, card-1 health, the 10 zero max_abs rows, and the microsecond conversion of the M1 and M3 timings. OK that the author refuses a DD promotion. WRONG for the campaign name: the first id is hotschmoe-dd. The 34.7 percent figure also rides on sitecustomize.py, x5_serve_proxy.sh, and B70_X5_Q8_FAST, so it is wider than a single .so swap. The FP32 and BF16 labels on the probe are UNVERIFIED. The DP4A name is a label; the checked source dot is scalar. 1000/10.2246 is about 97.80, so 99.805 is decode_tok_s. This is a 35B proxy harness ITL. 10.225 ms stays above the Strata 7.09 ms citation. No promotion.

Suggested fix: put hotschmoe-dev first, and state the 34.7 percent figure as an itl_p50 ratio against packq8 with the two changed harness files named.

### F251 -- F1 p6 profile-off prefill peaks at 2527.287 tok/s; decode ITL is 18.414 ms; the 3 percent X3 comparison is high

Claim: p6 health before and after is HEALTHY. Gates 323/667 pass on every arm. Decode C1 ITL p50 is 18.41 ms and C2 is 22.73/22.74 ms. The clean profile-off table is c0 fast plus c1 native:32 at 919/2438/2150/2091, c0 native:64 plus c1 native:32 at 820/2319/2527/2336, and the fast control at 717/2217/2027/2004 tok/s on 1018/8186/12283/32763. READY is 23:51:12. The native library sha starts 790230cb. The 12288 native chunk reaches 2527 tok/s, within 3 percent of X3's 2615, while keeping O4's decode. Profile at 12288 shows host_c1 0.28-0.34 s (6 ms/layer, was 40-48), card-1 device 1.74-2.0 s, card-0 wait+add 1.8-2.0 s, card-0 experts 1.19-1.28 s, and between-layers 1.92-2.07 s. The Paris pair is same=false with one token different at position 14, and that prompt stays off the prefill path because T<=8. Zero flag timeouts. p7 is queued.

Evidence: f1/run-p6/probe.json ok true, t0 23:51:13, first id hotschmoe-dev, first_ok true. models.json first id is hotschmoe-dev, max_model_len 147456, root hf-flashnext. Profile-off prefill_tok_s for c1 native:32 is 918.995, 2437.554, 2149.821, 2090.759. For c0 native:64 and c1 native:32 it is 819.602, 2318.729, 2527.287, 2336.018. For c0 fast and c1 fast:32 it is 717.332, 2216.821, 2027.122, 2003.873. ttft_ms for the 2527.287 cell is 4860.152 on built 12283. The four 12283 rows have n_chunks 0 and an empty first_token, and prefill_tok_s still equals built/ttft_s. itl_p50_ms is null on the prefill rows. decode_c1 itl_p50_ms is 18.41406198218465, decode_tok_s 54.22095131057519, completion_tokens 64, n_chunks 64. decode_c2 itl_p50_ms is 22.72903500124812 and 22.739444975741208. gates and the three profile-off gates_by_arm entries match texts 323 and 667. Profile=1 is not a gates_by_arm key. Long coherence matches 323 and 667, usage prompt_tokens 32788, built 32762. coherence ok is false, same is false, mentions_paris is true, verdicts are OK and OK. The two texts differ by an inserted "a" ("with 2" versus "with a 2"). No token-index field stores 14. health-before and health-after are HEALTHY for cards 0 and 1, gpu-run exit 0 in 27s and 28s. Window READY:1 is 23:51:12. The serve.log READY hit is the docker environment name VLLM_ENGINE_READY_TIMEOUT_S. Application startup complete is present. SERVE_RC and PROBE_RC are 0. The docker line sets B70_PREFILL_WIDE_SO to f1/_C.f1bucket.so. That file's sha256 starts 790230cb027b56d5. The serve log prints "native bucketed op" for c1 tile 32 maxpad 32768 and for c0 tile 64 maxpad 65536. The two T=12288 EP-PF rows have host_c1 325.4 and 279.2 ms, c1_experts 2004.5 and 1739.2 ms, c0_wait_add 1998.0 and 1779.3 ms, c0_experts 1276.6 and 1190.7 ms, and c0_between_layers 1920.8 and 2470.9 ms. p5 T=12288 host_c1 values span about 1906 to 2294 ms. window.log prints "decode c1 itl p50=None" on the same minute the JSON stores 18.414. No "flag timeout" string is in the window. f1/run-p7/serve.log is 0 bytes.

Verdict: OK for the rounded clean table, identity, the three profile-off gate pairs, long coherence, both health files, the library sha, and the 18.414 ms JSON ITL. WRONG that every arm has a gate record: profile=1 is absent from gates_by_arm. WRONG to read 23:51:12 as an engine READY clock. WRONG that one maxpad of 32768 covers the native path: the c0 native line says maxpad 65536. WRONG that between-layers stays inside 1.92-2.07 s: one T=12288 row is 2470.9 ms. WRONG that 2527 is within 3 percent of X3's 2615.490: the gap is 3.37 percent, and X3 was a dense-GQA prefill with first id hotschmoe-dd. The 18.414 ms figure is a 64-token harness itl_p50. decode_tok_s is 54.221. It is the same class as p5's 18.467 ms and is f9's 16.295 ms profiler step only if a bench JSON says so. The position-14 and T<=8 attribution is UNVERIFIED. p7 has no serve body. The 2527.287 figure is prefill_tok_s. No promotion of decode toward the Strata 141 tok/s citation.

Suggested fix: title the 12288 cell 2527.287 prefill_tok_s, record c0 maxpad 65536 beside c1 maxpad 32768, and keep the between-layers sample 2470.9 ms in the band.

### F252 -- p5 post-health is now HEALTHY; profile flush sync is gated; repo hygiene

Claim: F244 left f1/run-p5/health-after.txt as a wait. The prefill profile path now calls torch.xpu.synchronize on cards 0 and 1. The new X5 Q8 plugin has no host sync. The repo has no new secret or binary from this audit.

Evidence: f1/run-p5/health-after.txt contains "xpu-health: HEALTHY (cards 0 1)" and "gpu-run: [F1] done in 33s (exit 0)". plugins/b70_ep_xpu/b70_ep_xpu/prefill_ep.py lines 344-345 call torch.xpu.synchronize(0) and synchronize(1) inside _flush_profile, which runs only when _PROF is set and the layer is the last layer. _PROF follows B70_EP_PREFILL_PROFILE or the ctl profile flag. plugins/b70_x5_xpu has no synchronize, .item(), or .cpu() match. bucket_moe.cpp and w8_gemv.cpp have none either. HEAD is 929563aa. Cached diff is empty. Porcelain is 36 rows. Binary or secret-like porcelain rows:  M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so. vllm/flashnext/.gitignore still lists *.so: true. Fork HEAD is 0a6170da8ca9 on b70/flashnext, 12 files changed, 789 insertions(+), 49 deletions(-). This audit made no host change.

Verdict: OK that p5 post-health is now HEALTHY for cards 0 and 1. That closes the open wait in F244 without changing F244's earlier read. The new synchronize pair is a profile-end fence on the prefill path. It is an unconditional sync inside decode graph replay only if a FULL decode capture runs with the profile flag on. The Q8 op checked here has no host sync. OK that this audit made no host change.

Suggested fix: keep B70_EP_PREFILL_PROFILE off any FULL decode capture.
## Pass 28 -- 2026-10-21 00:45 UTC

### F253 -- O4 f10 stays 16.31 ms and f11 is 15.698 ms; health-after is HEALTHY and health-before is absent

Claim: f10 is the f9 recipe plus B70_EP_Q0=150 and does not move latency, so Q0 stays 138. f11 adds B70_DECODE_HC_Q8=1. f10 C1 is 16.31 ms and C2 is 19.54 ms. f11 C1 is 15.70 ms with graph 14.85 ms and C2 is 18.93 ms with graph 17.86 ms. Gates pass, C2 same_text, and token ids match f10 over 96 tokens. Health is HEALTHY around both runs. The Q8 copies add 0.67 GB on card 0. The GEMV uses ksplit 8 and short-K 8 lanes.

Evidence: o4/run-f10/profile.log C1 step_ms p50 is 16.31, replay_dev_ms 15.455, replay_host_ms 12.072. C2 step_ms p50 is 19.544, replay_dev_ms 18.377. o4/run-f11/profile.log C1 step_ms p50 is 15.698, replay_dev_ms 14.85, replay_host_ms 11.316. C2 step_ms p50 is 18.926, replay_dev_ms 17.864, replay_host_ms 14.183. Both gates match 323 and 667 and print c2 same_text=True. text-c1.json and text-c2.json are byte-identical between f10 and f11, inner length 96. f10 run.log sets B70_EP_Q0=150 and does not set B70_DECODE_HC_Q8. f11 run.log sets B70_DECODE_HC_Q8=1. Both health-after files are HEALTHY for cards 0 and 1. f11's after-probe is gpu-run exit 0 in 27s. Neither run has health-before.txt. Both run logs say "Model loading took 15.66 GiB". w8_gemv.cpp accepts ksplit 8 via launch<8> and ksplit -8 via launch_short<8>. The f11 log does not print the ksplit argument.

Verdict: OK for the step_ms rounding, the gates, and the 96-id match between f10 and f11. WRONG that health was taken around the runs: the before file is absent. f11 health-after is HEALTHY at this read, which closes the wait left in F247. WRONG that these logs show a +0.67 GB card-0 increase: both Model loading lines say 15.66 GiB. The runtime ksplit value is UNVERIFIED. C1 replay_host_ms is 12.072 and 11.316, so the GPU is not full. These are profiler step_ms values. They stay above the Strata 7.09 ms citation. No promotion.

Suggested fix: publish replay_host_ms with the step, and log the resident-memory delta that produced 0.67 GB.

### F254 -- IQ3, IQ4, and Q4 fast-Q8 ITLs match the journal; every first id is hotschmoe-dd

Claim: the matched no-MTP quant choice is IQ3 C1 8.0471 ms, C2 10.2258/10.2725, PPL 9.05563298; IQ4 C1 10.8541, C2 11.5288/11.5106, PPL 8.76477959; Q4 C1 10.2246, C2 11.5000/11.4909, PPL 8.50822826. Request gates and exit 0 pass. Source hashes match inside each window. IQ3 pre and post health are HEALTHY. The 00:19 note left IQ4 post-health queued, and the 00:38 note says all three now have pre and post health. Q4 wins the 4-bit choice on C1 and prefix PPL. IQ3 is the fastest lower-bit option.

Evidence: IQ3 bench.json itl_p50_ms is 8.047066046856344, decode_tok_s 126.34435231051334, n_chunks 62, completion_tokens 63. C2 itl_p50_ms is 10.225807956885546 and 10.272542014718056. PPL is 9.055632975432564. IQ4 C1 itl_p50_ms is 10.854145046323538, decode_tok_s 93.75515967603972, completion_tokens 57. C2 is 11.528771079611033 and 11.510595912113786. PPL is 8.764779589576639. Q4 is the F250 fastq8 bench: C1 10.224584955722094, C2 11.500005959533155 and 11.490878998301923, PPL 8.508228260209568. All three have mtp.enabled false, first id hotschmoe-dd, expect_first hotschmoe-dd, first_ok true, max_model_len 4096, coherence ok, repeat identical, scored_tokens 2044, and corpus prefix efd86268d53e85ef. serve-exit-code.txt is 0 on each, with Application startup complete, zero READY hits, zero exit 139, and zero force-kill. gpu-run exits 0 in 129s, 123s, and 128s. Window gate_rc is 0 on each. source-before equals source-after inside each window. Docker B70 flags match except B70_GGUF_PATH, which is UD-IQ3_XXS, UD-IQ4_XS, and UD-Q4_K_M. IQ3 also prints STOCK_PARITY_RC=0. Health-before and health-after are HEALTHY for card 1 on all three. dd-state.txt is inactive on each.

Verdict: OK for the three ITL pairs, the three PPL scalars, the shared corpus, the within-window source match, serve exit 0, and card-1 health on both sides. The 00:38 health sentence closes the 00:19 IQ4 wait. WRONG for the campaign name on all three: the first /v1/models id is hotschmoe-dd. The GGUF paths differ, so this is a quant comparison and not one checkpoint. IQ3's C1 completion is 63 tokens and the other two are 57, so the p50 samples are not the same length. 1000/8.0471 is about 124.3, so 126.344 is decode_tok_s. Q4's C1 10.225 ms is lower than IQ4's 10.854 ms and its PPL is lower, which is the comparison the journal uses. IQ3's 8.047 ms is the lowest ITL of the three and stays above the Strata 7.09 ms citation. These are 35B proxy harness ITLs. No promotion.

Suggested fix: put hotschmoe-dev first, and record C1 completion_tokens next to each p50.

### F255 -- F1 p7 profile-off prefill peaks at 2511.124 tok/s; post-health is still a wait

Claim: no journal RESULT states p7. The p6 note queued p7 as the 1K through 128K qualification for c0=auto:64 and c1=native:32.

Evidence: f1/run-p7/probe.json ok true, t0 00:20:59, first id hotschmoe-dev, first_ok true. models.json first id is hotschmoe-dev, max_model_len 147456, root hf-flashnext. The profile-off arm is "c0=auto:64 c1=native:32". prefill_tok_s is 923.837 at built 1018, 2433.644 at 8186, 2511.124 at 12283, 2342.246 at 32763, and 1586.657 at 131067. The 12283 row has n_chunks 0 and prefill_tok_s still equals built/ttft_s (ttft_ms 4891.434). itl_p50_ms is null on the prefill rows. decode_c1 itl_p50_ms is 18.49067909643054, decode_tok_s 54.22725399310854, n_chunks 64, completion_tokens 64. decode_c2 itl_p50_ms is 22.696322994306684 and 22.70782506093383. Top-level gates and long coherence match 323 and 667, usage prompt_tokens 32788, built 32762. gates_by_arm has only the profile-off arm. coherence ok is false, same is false, mentions_paris is true, verdicts are OK and OK, and the texts differ by an inserted "a". health-before is HEALTHY for cards 0 and 1, gpu-run exit 0 in 34s. p7 health-after is a gpu-run wait with zero HEALTHY lines. Window READY:1 is 00:20:56. serve.log has one READY hit and Application startup complete. SERVE_RC is 0.

Verdict: OK that the raw probe contains those prefill_tok_s values and the 18.491 ms harness ITL, with identity hotschmoe-dev and pre-health HEALTHY. There is no journal RESULT to match. WRONG to read 00:20:56 as an engine READY clock. The 2511.124 cell is prefill_tok_s on built 12283. It is below p6's 2527.287 cell and is not X3's 2615. decode_tok_s is 54.227. The same=false Paris pair is the same text split recorded on p6. Post-health is not closed in the sentence above. No promotion.

Suggested fix: journal the profile-off table with replay of post-health, and keep 2511.124 labeled prefill_tok_s.

### F256 -- O4 s1's saved bench is marked wrong-server; MTP2 has no ITL yet; repo hygiene

Claim: the 00:15 note left s1 and s2 pending. The 00:38 note says MTP2 pre-health has been waiting since 00:25:57 behind the O4 s1 serve, and it asks for a card gap. It states no ITL. The repo has no new secret from this audit.

Evidence: o4/serve-s1 has bench-WRONG-SERVER-f1.json and no bench.json. The window bench command uses http://127.0.0.1:18140. Its greedy line lists the f1-prefill alias. The window prints alias_ok=False and "decode C1 itl_p50=18.58 ms", then BENCH_RC:0. The renamed JSON has alias_ok true, ids hotschmoe-dev and the o4 alias, and decode_c1 itl_p50_ms 16.218937933444977. models.json max_model_len is 147456. health-after has zero HEALTHY lines. x5/qwen36-35b-udq4km-fastq8-canon-mtp2: no bench.json. health-before HEALTHY=True. files: dd-state.txt, health-before.txt, models.json, serve.log, source-before.json, started, window.log. HEAD is 929563aa. Cached diff is empty. Porcelain is 41 rows. Binary or secret-like porcelain rows:  M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so. vllm/flashnext/.gitignore still lists *.so: true. Fork HEAD is 0a6170da8ca9 on b70/flashnext, 12 files changed, 789 insertions(+), 49 deletions(-). This audit made no host change.

Verdict: OK that the journal does not publish an s1 or MTP2 ITL. WRONG to read the window's BENCH_RC 0 as an O4 recipe ITL: that print saw the F1 alias and 18.58 ms, while the renamed JSON stores 16.219 ms and a different alias_ok. The two artifacts disagree, and post-health is open. The 00:38 MTP2 wait sentence is stale if health-before is already HEALTHY; the directory still has no bench ITL. OK that this audit made no host change. No promotion.

Suggested fix: delete or quarantine the wrong-server JSON, and publish an s1 ITL only from a bench whose /v1/models alias matches the O4 serve.
## Pass 29 -- 2026-10-21 01:30 UTC

### F257 -- O5 n0/n2 eff on the f9 sidecar, and the oneDNN HC-down rows

Claim: the 00:50 note says prefill is identical and the first T=1 vs T=3 divergence is an HC mix whose input matches. It says every dense GGUF GEMV is row-invariant, oneDNN BF16 F.linear is row-variant for the HC down shape (K=10240) at M>=4, and for a fused [324, 10240] at M=3, while b70_o4 bf_gemv stays invariant. It says n0 is 16.29 / 20.44 ms/tok and n2 is 11.62 / 17.87 with accept 68.4/65.5 percent, step 27.6, verify graph 22.6, propose 3.55, gates pass, and HEALTHY before and after. Parity is C2 4/4 over 128 tokens and C1 3/4, with the hash-map prompt differing at token 58. The Q5_K head census is 782 / 1652 / 2132 us.

Evidence: tokens-n0.json and tokens-n2.json both gate-match 323 and 667. Served-name kwargs in profile.log start with hotschmoe-dev. PHASE lines: n0 C1 mean 16.291 and C2 mean 20.4415; n2 C1 mean 11.62325 accept_mean 0.68435 and C2 mean 17.8695 accept_mean 0.6549. Mean of C1 step p50 is 16.359 on n0 and 27.626 on n2. Mean of C1 replay_dev p50 is 15.526 on n0 and 22.548 on n2. n2 propose p50 mean is 3.555. C1 token lists: 3 of 4 identical over 128 ids, and the hash-map prompt first differs at index 58. C2: 4 of 4 identical over 128. n0 health-before and health-after both contain xpu-health: HEALTHY. n2 has no health-before.txt. n2 health-after is HEALTHY. gemv_bench.json BF16-onednn blk.14.hc_attn_down N=320 K=10240 has invariant false at M=4, 6, and 8. The same tensor on blk.15 is invariant false only at M=6 and M=8; M=4 is true. bench.log records bf_gemv_invariant true on those rows. No row has N=324 and K=10240. run-p0b/probe.log FIRSTDIFF names layers.0.mlp.gate for the request hash and layers.5.linear_attn.in_proj_ba for row0. gemv_bench.json Q5_K output.weight M=1/3/4 is 782.34 / 1652.05 / 2132.5 us. These are in-process eff ms/tok values, not harness itl_p50.

Verdict: OK on the n0/n2 means, the acceptance percents, the 128-token parity, the gates, the head census, and layer-14 oneDNN variance at M>=4. WRONG that both arms are HEALTHY before and after: n2 has no health-before. WRONG that every HC-down row is variant at M>=4: layer 15 M=4 is still invariant. UNVERIFIED for the fused [324, 10240] sentence and for the HC-mix input-identical sentence; the FIRSTDIFF line does not say that. No promotion. 11.623 ms/tok is still slower than the Strata 7.09 ms citation, and it is not a harness ITL.

Suggested fix: split the health sentence per arm, and quote the FIRSTDIFF module name instead of an HC-mix sentence until that row is in the hash file.

### F258 -- O5 kreg A/B is bitwise identical; M=1 latency is not flat on every row

Claim: the 01:15 note says _C.o5b.so sha starts 04ea8891, every output hash matches o4g, every kernel stays row-invariant, and the cited shapes drop from 1638 to 944, 2118 to 1101, 4614 to 1820, with M=1 and M=2 unchanged at 781 and 1068. IQ4_XS M=3 drops 140 to 68. Q5_K 2560x6144 drops 78 to 48 and 116 to 66. It says M<=2 is unchanged so the patch is safe to merge. It also says n3v C1 is 11.26 ms and C2 is 18.73, with verify 25.0 ms, propose 3.3 ms, and acceptance 57.8 percent.

Evidence: sha256 of o5/gguf-src/b70_gguf_xpu/_C.o5b.so is 04ea88914585fb55375fee6cb49f2edb939acc0362fba381198cdf9e477a4396. bench-o4g.json and bench-o5b.json have 84 shared keys and 84 identical h fields, and every invariant flag is true. Q5_K output.weight 248320x2560 us is M=1 781.06 to 781.0, M=2 1067.51 to 1067.91, M=3 1638.12 to 944.54, M=4 2118.43 to 1101.05, M=8 4613.54 to 1819.93. IQ4_XS 12288x2560 M=3 is 139.9 to 68.37. Q5_K ssm_out 2560x6144 M=3 is 77.57 to 48.14 and M=4 is 116.2 to 66.18. Q4_K attn_qkv 10240x2560 M=1 is 96.0 to 68.22, so at least one M=1 row moved. All six gemv health files contain HEALTHY on card 1 with exit 0. The patch harness/patches/o5_kreg_multicol.patch has no synchronize, .cpu(), or .item() call. The NCOLS>=3 branch is compile-time and the else branch keeps the old loop. Repo plugins/b70_gguf_xpu/csrc/kreg.hpp does not contain that branch. n3v PHASE C1 mean is 11.26225 and C2 mean is 18.728, accept_mean 0.5782. C1 replay p50 mean is 25.09 and propose p50 mean is 3.298. kwargs name starts with hotschmoe-dev. n3v health-before is missing. n3v health-after now contains HEALTHY. These us figures are kernel calls, and the n3v figures are eff ms/tok.

Verdict: OK on the sha prefix, the 84/84 hashes, the cited us rows within rounding, the card-1 GEMV health pairs, and the n3v means. WRONG as a measured claim that every M<=2 call is unchanged: the Q4_K qkv M=1 call moved from 96.0 us to 68.22 us. The patch text does keep the old loop when NCOLS is at most 2. No promotion, and this audit did not merge the patch.

Suggested fix: cite the hash match as the numeric check, and list M=1 rows whose timings moved before calling the single-token path unchanged. Keep the serve A/B in front of any repo merge.

### F259 -- X5 Q4 MTP2 repeats, and the Paris on/off gate fails

Claim: the 00:48 note says same-load C1 and C2 repeats pass, on/off parity fails Paris at C1 index 57 and C2 index 34, the other four streams match, diagnostic C1 effective is 6.597 ms (151.58 tok/s) with wire p50 17.515, C2 effective is 9.244 and 9.392, accept is 90/120, and identity, coherence, source, exit 0, and pre/post health pass. The verdict is not qualified.

Evidence: bench.json decode_c1 itl_p50_ms is 17.514917068183422 and decode_tok_s is 151.57913794802053. 1000/151.57913794802053 is 6.597. C2 decode_tok_s values are 108.17516502307348 and 106.4729062323292, which are 9.244 and 9.393 ms effective. mtp accepted_during_bench is 90 of 120. identity ids are hotschmoe-dd then the mtp2 alias, first_ok true, expect_first hotschmoe-dd. coherence ok, same, and mentions_paris are true. repeat.json identical and concurrent_identical are true. parity-analysis.json rows index 0 first_mismatch is 57 and concurrent_rows index 0 first_mismatch is 34; the other four streams are identical. source-before equals source-after. serve-exit-code is 0. dd-state is inactive. health-before and health-after both contain xpu-health: HEALTHY (cards 1) and exit 0. This is a 35B proxy. The wire p50 is a chunk ITL (n_chunks 22 on a 56-token completion).

Verdict: OK on the numbers, the repeat pass, the Paris mismatches, the source match, the exit, and both health files. WRONG campaign name: the first id is hotschmoe-dd. The 6.597 ms figure is 1000/decode_tok_s, not the wire p50. The journal's own verdict not-qualified stands. Not a Flash-Next promotion.

Suggested fix: keep the gate failed, and publish a campaign ITL only when the first /v1/models id is hotschmoe-dev.

### F260 -- X5 noSpec packet rates match the median samples; spec-r1 is 21 exact rows

Claim: the 00:55 note says reasoning is 96.059 tok/s (10.410 ms) and code is 96.255 tok/s (10.389 ms), with TTFT 561.07 and 450.31 ms, both dry-runs valid and verified, all 576 baseline token ids match, and pre/post health pass. The 01:03 note says 21 BF16 cases are bit-exact, maxabs is 0, 8 replays are bit-exact, the probe body is 17 s, exit is 0, and pinned health is HEALTHY.

Evidence: localmaxxing iteration-3 sample.json for reasoning has tok_s_out 96.05906677806044, effective_itl_ms 10.410261452054796, and ttft_ms 561.071356. The code sample has 96.25450261916497, 10.389124381604699, and 450.311742. dry-run.txt for both suites prints valid true, issues empty, and verified true, with specDecoding false. validation.json submitted is false. lmx repeat.json token ids equal the canon-mtp0 repeat.json on all 6 streams of 96 ids, which is 576 ids. lmx health-before and health-after both contain HEALTHY and exit 0. The harness bench in the same directory is a different measurement: itl_p50_ms 10.257977992296219 and decode_tok_s 99.63697676522722. First id is hotschmoe-dd. spec-r1/numeric.json has 21 rows, widths 3 and 4, and every core_exact, out_exact, states_exact, final_conv_exact, and replay_exact flag is true, with core_maxabs 0.0 and state_maxabs 0.0. probe.log prints X5_GDN_SPEC_PROBE_PASS 21 and gpu-run done in 17s exit 0. spec-r1 health-before and health-after both contain HEALTHY (cards 1) and exit 0. The parent gdn-bf16/numeric.json is an older file whose out max_abs is 1.5625e-02, and its probe.log ends exit 1. spec-r1/numeric.json does not contain a count of 8 replays.

Verdict: OK on the packet rates, the 576-id match, the dry-run flags, spec-r1 exactness, the 17 s exit 0, and the health files that belong to those runs. WRONG campaign name on the lmx serve. UNVERIFIED that the replay count is 8. The 96 tok/s figures are 35B packet rates, not a Flash-Next harness ITL. No promotion.

Suggested fix: cite spec-r1/numeric.json for the exactness claim, and keep the parent numeric.json labeled as the earlier failed probe.

### F261 -- fused-spec matches noSpec on C1; the BI noSpec control is slower than 10.225 ms

Claim: the 01:09 note says fused spec captured 3/6 rows, all 4 C1 streams match noSpec and repeat, C2 Paris differs at 34, diagnostic C1 effective is 6.632 ms, accept is 91/120, and post-health was still queued. The 01:17 note says the BI noSpec control has C1 p50 11.2989 ms, C2 12.7707 and 12.7679, PPL 8.52856158, repeat pass, exit 0, and post-health queued, and that it is slower than the selected 10.225 ms.

Evidence: fusedspec parity token ids versus the canon-mtp0 repeat.json are identical on all 4 C1 streams of 96 ids. C2 Paris first differs at index 34. The Fibonacci concurrent stream is identical. fusedspec repeat.json token ids equal fusedspec parity.json on all 6 streams. bench.json accepted_during_bench is 91 of 120. decode_tok_s is 150.7805288712081, and 1000/that is 6.632. wire itl_p50_ms is 17.70823507104069. First id is hotschmoe-dd. source-before equals source-after. serve exit is 0. health-before and health-after both contain HEALTHY, so the queued post-health sentence is stale. The serve log shows spec=1 fused calls at ntok 6 and ntok 3. It does not print the fraction 3/6. BI bench.json C1 itl_p50_ms is 11.298915022052824. C2 is 12.770730012562126 and 12.767868931405246. quality.json perplexity is 8.528561575107465, scored_tokens 2044, corpus prefix efd86268d53e85ef. repeat identical and concurrent_identical are true. coherence same is true. source files match. serve exit is 0. health-before and health-after both contain HEALTHY, so that queued sentence is also stale. First id is hotschmoe-dd. The selected Q4 harness itl from the earlier canon window is 10.224584955722094. Both of these windows are 35B proxy runs.

Verdict: OK on the fusedspec token compare, the 6.632 diagnostic, the 91/120 accept, and the BI p50, C2, and PPL figures. WRONG campaign name on both serves. UNVERIFIED for the 3/6 capture sentence. The post-health sentences are stale because both health-after files are HEALTHY now. Not a qualification and not a Flash-Next promotion.

Suggested fix: keep the C2 Paris mismatch as a failed gate, and record post-health from the HEALTHY files.

### F262 -- O4 s2 harness ITL is 15.571 ms; s1's directory still mixes two servers

Claim: the 01:20 note says the first s1 bench hit F1 and was discarded, and a hand rerun against the O4 alias measured C1 ITL p50 16.22 ms (61.7 tok/s) and C2 20.26/20.25, with prefill 151 tok/s. It says s2 C1 ITL p50 is 15.57 ms (64.0 tok/s) and C2 is 19.84/19.84, identity is hotschmoe-dev plus the o4 alias, gates are 323/667, and health is HEALTHY before and after both. It compares 60.2 ms to 15.57 ms and cites Strata no-spec 15.1 ms. PLE pin is 87.8 s. The 01:23 note says the mixed GDN CPU test passed in a 16 s container after a host exit 127, with no speed claim.

Evidence: serve-s2 window and bench.json agree. C1 itl_p50_ms is 15.57096210308373, p90 is 15.971000352874398, decode_tok_s is 63.99091285745765, n_chunks is 64, mtp.enabled is false. C2 itl_p50_ms is 19.840828026644886 and 19.841279019601643. models.json ids are hotschmoe-dev then the o4 alias, max_model_len 2048. greedy.json matches 323 and 667 with first_ok true. coherence ok and same are true. The bench port in the window is 18147. health-before.txt is absent. health-after is HEALTHY on cards 0 and 1, exit 0. The same window prints the Strata reference as 7.09 ms and 141 tok/s. serve-s1 bench.json C1 itl_p50_ms is 16.218937933444977, decode_tok_s is 61.69024273119305, C2 is 20.25533898267895 and 20.24738397449255, and prefill_tok_s is 151.24743904083832, with alias_ok true and the o4 alias. greedy.json for s1 also names the o4 alias and matches 323/667. window.log for s1 still prints alias_ok false, the f1-prefill alias, C1 18.58 ms, and prefill 994.15 tok/s on port 18140. models.json for s1 still lists the f1-prefill alias and max_model_len 147456. s1 health-before.txt is absent. s1 health-after is HEALTHY. serve-s1.log prints PLE device table pinned in 87.8 s. o4_serve.sh defaults PORT to 18147 and greps the o4 alias before it prints READY. It writes health-after only. gdn-mixed-cpu.log is exit 127. gdn-mixed-cpu-r2.log prints X5_GDN_MIXED_CPU_PASS and done in 16s exit 0, and the docker line does not mount /dev/dri. HEAD is 929563aa. Cached diff is empty. Porcelain is 47 rows. Binary porcelain row:  M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so. The recipe env files are flag lists with no key, token, or password name. vllm/flashnext/.gitignore still lists *.so: true. Fork HEAD is 0a6170da8ca9 on b70/flashnext, 12 files changed, 789 insertions(+), 49 deletions(-). This audit made no host change.

Verdict: OK that s2's bench, window, greedy file, and models file describe one O4 serve, and that 15.571 ms and 19.841 ms are that serve's harness itl_p50 on 64 chunks. OK on the s1 bench.json rounding and on the 87.8 s PLE line. WRONG that health is HEALTHY before and after: neither serve directory has health-before.txt. UNVERIFIED as a closed s1 qualification: models.json and window.log still record the F1 server while bench.json records the o4 alias. The 60.2 to 15.57 comparison is not a matched A/B, and the 15.1 ms Strata citation is not the 7.09 ms line in the window. s2 at 15.571 ms is about 64 tok/s, far from 141 tok/s. OK on the mixed-GDN CPU pass and the 16 s exit 0. No promotion.

Suggested fix: refresh serve-s1/models.json from the same process that wrote bench.json, add a health-before probe to o4_serve.sh, and keep 15.571 ms labeled as a 64-token harness ITL with no health-before.
### F263 -- X5 BI MTP2 matches noSpec on C1; Paris still fails

Claim: the 01:28 note says all 4 C1 streams match the baseline and the repeat, C2 Paris fails the repeat at index 66 and the baseline at index 34, Fibonacci matches, diagnostic C1 effective is 7.647 ms, C2 effective is 10.067 and 9.301, accept is 93/128 = 72.66 percent, source is stable, exit is 0, and post-health was queued. The verdict is not qualified.

Evidence: qwen36-35b-udq4km-fastq8-bi-mtp2/bench.json decode_c1 decode_tok_s is 130.770078735233, and 1000 divided by that is 7.647. C2 decode_tok_s values are 99.33667054491666 and 107.51554883958136, which are 10.067 and 9.301 ms effective. Wire itl_p50_ms is 19.753099069930613 on C1. mtp accepted_during_bench is 93 of draft_tokens 128, rate 0.7265625. Against the bi-mtp0 repeat.json, all 4 C1 streams match in both parity.json and repeat.json. C2 Paris differs from that baseline at index 34 in both files. The Fibonacci stream matches. repeat.json versus parity.json differs at index 66 on Paris only. source-before equals source-after. serve-exit-code is 0. health-before and health-after both contain xpu-health: HEALTHY, so the queued post-health sentence is stale. The first id is hotschmoe-dd. This is a 35B proxy diagnostic.

Verdict: OK on the effective rates, the 93/128 accept, and the token indexes. WRONG campaign name. The post-health sentence is stale. 7.647 ms is 1000/decode_tok_s, and the wire p50 is 19.753 ms. The journal's not-qualified verdict stands. No Flash-Next promotion.

Suggested fix: keep the Paris gate failed, and do not cite 7.647 ms as a harness ITL.

## Pass 30 -- 2026-10-21 02:15 UTC

### F264 -- mixed noSpec matches its own solos; mixed MTP2 still fails Paris

Claim: the 01:33 note says the mixed noSpec path matches on C1 and C2 repeat, both C2 responses match their solo outputs, all 4 solo outputs match the prior BI control, PPL is unchanged at 8.52856158, C1 p50 is 11.3582 ms, C2 is 12.7837 and 12.7484 ms, source is stable, exit is 0, pre-health passed, and post-health was queued. The 01:51 note says the mixed MTP2 arm fails C2 Paris at repeat index 43 and parity index 41, all 4 solo comparisons pass, and source, exit 0, and native pre and post health pass.

Evidence: bi-mixed-mtp0 bench.json C1 itl_p50_ms is 11.358243995346129, p90 is 11.561907909344882, p99 is 11.763032078742981, decode_tok_s is 89.78440211195391, n_chunks is 59. C2 itl_p50_ms is 12.783745012711734 and 12.748392997309566. identity ids are hotschmoe-dd then the x5-mtp0 alias, first_ok true, expect_first hotschmoe-dd. coherence ok and same are true. mtp.enabled is false. quality.json perplexity is 8.528561575107465, scored_tokens 2044, corpus_sha256 efd86268d53e85ef, the same PPL and corpus as bi-mtp0. source-before.json equals source-after.json. serve-exit-code is 0. health-before and health-after both contain HEALTHY and exit 0, so the queued post-health sentence is stale. repeat.json identical and concurrent_identical are true. All 4 solo rows match bi-mtp0 repeat.json over 96 ids. Within this run, both concurrent rows match their solo prompts. Versus bi-mtp0, concurrent Paris still differs at index 34; the note does not claim that C2 matches the prior control. bi-mixed-mtp2 C1 wire itl_p50_ms is 19.901745952665806 and decode_tok_s is 130.78762621710737. C2 wire itl_p50_ms is 23.903592955321074 and 23.78734201192856. accepted_during_bench is 93 of 128. First id is hotschmoe-dd. source files match. serve exit is 0. Both health files are HEALTHY. Against mixed-mtp0, all 4 solo rows match in parity.json and in repeat.json. Concurrent Paris differs at index 43 in repeat.json and at index 41 in parity.json. Those two captures of the MTP2 run differ from each other at index 41 on Paris only. Fibonacci concurrent matches.

Verdict: OK on the noSpec p50s, the PPL, the solo match, the within-run C2-to-solo match, the exit, and both health files. WRONG campaign name: the first id is hotschmoe-dd. The post-health sentence is stale. OK on the MTP2 Paris indexes 43 and 41 and on the 4 solo matches. The wire p50 is a chunk ITL. Both windows are 35B proxy runs, slower than the selected Q4 harness itl 10.225 ms on the noSpec arm. Not qualified and not a Flash-Next promotion.

Suggested fix: keep the mixed MTP2 Paris gate failed, and publish a campaign ITL only when the first /v1/models id is hotschmoe-dev.

### F265 -- b2 eff means match; the named GDN divergence site does not

Claim: the 01:55 O5 note says b2 C1 is 10.20 ms/token (8.48-11.54) and C2 is 13.57 ms/stream, verify T=3 is 19.8 ms (n2 22.6, T=1 control 15.5), C2 verify T=6 is 26.5 ms (n2 35.2), propose is 3.50 ms, host gap is about 1.0 ms, acceptance is 337/430 = 78.4 percent then 270/430 = 62.8 percent, and 1.41 tokens are accepted per step. It says the kreg fix is worth 1.4 ms/token at C1 and 4.3 ms/token at C2. It says the GDN flag has no effect on MTP verify batches, and that p0c/p2c put the first T=1 vs T=3 divergence at a GDN layer whose linear_attn.norm input equals the core output.

Evidence: run-b2 tokens and profile agree. C1 eff_ms_per_tok values are 10.907, 9.896, 8.476, and 11.539. The phase mean is 10.2045. C2 values are 13.386 and 13.756, phase mean 13.571. C1 replay_dev p50 values are 19.75, 20.037, 19.843, and 19.823. C2 replay_dev p50 values are 26.499 and 26.675. Propose p50 values are 3.492 to 3.504. step p50 minus replay_dev p50 minus propose p50 is about 1.02 to 1.05 on the four C1 prompts. Metrics are spec_decode_num_drafts 430 and spec_decode_num_accepted_tokens_per_pos 337 and 270. 337/430 is 0.7837 and 270/430 is 0.6279. (337+270)/430 is 1.4116. Gates match 323 and 667. kwargs served_model_name starts with hotschmoe-dev. Versus the n2 phase means 11.62325 and 17.8695, the gaps are 1.419 ms and 4.299 ms. The n2 C2 replay p50 mean is 35.367, so 35.2 is a loose round. recipe.env differs from run-n2/recipe.env only by B70_GGUF_SO pointing at _C.o5b.so and by B70_GDN_SPEC_DECODE=1. In the fork _xpu_ops.py the flag branch runs only when num_prefills is 0, num_spec_decodes is 0, and num_decodes is greater than 0. An MTP verify batch does not enter it. The branch then allocates torch.ones and does not check that each sequence has one token. p0c and p2c FIRSTDIFF solo vs dup/trip name language_model.model.layers.0.mlp.gate and language_model.model.layers.5.linear_attn.in_proj_qkvz. Neither log says linear_attn.norm input equals core output. health-before.txt is absent. health-after is HEALTHY on cards 0 and 1, exit 0. The note does not claim health. These are in-process eff ms/tok, not a harness ITL.

Verdict: OK on the eff means, the replay and propose figures, the 337/430 and 270/430 rates, the 1.41 sum, and the arithmetic gap versus n2. OK that the flag branch does not run when num_spec_decodes is already non-zero, so the MTP verify comparison's remaining recipe difference is the o5b library. WRONG on the named divergence site: the probe logs name mlp.gate and in_proj_qkvz, not linear_attn.norm. The 1-token claim is not enforced by the branch. No promotion. 10.2045 ms/token is still above the 7.09 ms Strata citation.

Suggested fix: cite in_proj_qkvz for the probe, and gate the spec-kernel reroute on a checked query length of 1 before allocating torch.ones.

### F266 -- b0 matches b2 on all 8 streams; k=3 misses one C1 prompt

Claim: the 02:12 note says b0 versus b2 is token-identical on all 8 streams over 128 tokens, b0 is 16.20 and 20.19 ms/token and the GDN flag costs nothing, b0 versus b3v matches 7 of 8 with the hash-map C1 stream differing at token 58, b3v is 9.59 and 13.77 ms/token with verify T=4 at 21.1 ms, propose 3.3 ms, and acceptance 74.2/56.3/42.4 percent. It says the default FA split differs at past 12000 and batch-2 differs at past 5000, and that B70_FA_KV_SPLITS 16 or 32 is bitwise equal on every case from 16 through 60000, with card-1 health HEALTHY before and after.

Evidence: tokens-b0.json and tokens-b2.json each have 8 streams of 128 ids, and all 8 pairs are identical, including 4 C1 prompts and 4 C2 prompt streams. Gates match 323 and 667 on both. b0 phase means are 16.19525 and 20.188. kwargs name starts with hotschmoe-dev and the sidecar is mtp0. b0, b2, and b3v have no health-before.txt. b0 health-after is HEALTHY on cards 0 and 1, exit 0. b3v C1 phase mean is 9.59125 and C2 is 13.773. C1 replay_dev p50 values average 21.183. Propose p50 values are about 3.31 to 3.34. Accepted-per-position counts are 285, 216, and 163 over 384 drafts, which are 74.22, 56.25, and 42.45 percent. Versus b0, 7 streams match over 128 ids and the C1 hash-map stream first differs at index 58. The 40525-row draft-vocab count was not opened. fa-0.log at past 12000 has verify_rows_equal all false and verify_max_abs 0.000244140625. At past 5000, batch2_row0_equal is false. fa-16.log and fa-32.log are equal with max_abs 0 at the seven logged past values 16, 900, 1700, 5000, 12000, 30000, and 60000. fa/health-before.txt and health-after.txt are HEALTHY on card 1, exit 0. The overlay passes _FIXED_KV_SPLITS into the spec-decode varlen call and into the q_len 1 path. b0's recipe is the same file as b2, so it changes both the o5b library and the flag relative to the earlier n0 MTP-off arm.

Verdict: OK on the 8-of-8 token match, the b0 and b3v eff means, the token-58 miss, the acceptance arithmetic, and the seven FA sample points. OK that the FA card-1 health files are HEALTHY before and after. The sidecar arms have no health-before. UNVERIFIED that the flag alone costs nothing: there is no MTP-off arm that flips only that flag. UNVERIFIED for a 40525-row vocab count. The FA sentence is the seven logged pasts, not every integer length. These are in-process eff ms/tok and a KV microbench, not a harness ITL. No promotion.

Suggested fix: keep the k=3 hash-map miss open, and add a health-before file on the sidecar arms before calling the pair qualified.

### F267 -- s3 harness ITL is 15.422 ms; the 2 percent Strata claim is not the window line

Claim: the 01:55 O4 note says f12 C1 step is 15.49 ms (graph 14.60) and C2 is 18.79, tokens match f11 over 96, and gates pass. It says s3 identity is hotschmoe-dev plus the o4 alias, greedy is 323/667, coherence is OK/OK, C1 ITL p50 is 15.42 ms (p90 15.64, p99 15.81, 64.9 tok/s), C2 is 19.68/19.69, and health is HEALTHY before and after. It says IQ3_S GEMV is 52.7, 65.7, and 50.5 us, the row kernel is 31.8, 37.4, and 21.9 us and bitwise equal, and other types are slower on the row kernel. It says s3 is 2 percent above Strata no-spec at 15.1 ms, and f13 was queued.

Evidence: run-f12 SUMMARY c1 step_ms p50 is 15.491, replay_dev_ms p50 is 14.601, replay_host_ms p50 is 11.186, n_steps 96, n_graph_steps 94. c2 step_ms p50 is 18.787. text-c1.json and text-c2.json are byte-identical to run-f11. Gates match 323 and 667. kwargs name starts with hotschmoe-dev. health-before.txt is absent. health-after is HEALTHY, exit 0. serve-s3 bench.json C1 itl_p50_ms is 15.422048047184944, p90 is 15.638708742335439, p99 is 15.80667945323512, decode_tok_s is 64.94014635514438, n_chunks is 64. C2 itl_p50_ms is 19.681554986163974 and 19.691923982463777. identity ids are hotschmoe-dev then the o4 alias, alias_ok true. models.json max_model_len is 2048. greedy matches 323 and 667. coherence ok and same are true. mtp.enabled is false. Prefill built 1017 at 150.77694049873728 tok/s. health-before.txt is absent. health-after is HEALTHY on cards 0 and 1, exit 0. serve.log contains one Application startup complete line and no READY line outside the engine-timeout variable. The window READY stamp is the o4_serve.sh alias grep. The same window prints Strata as 141.0 tok/s and 7.09 ms. 15.422/7.09 is 2.18x, not 2 percent. 1000/15.42205 is 64.84, so 64.9 is decode_tok_s. Workspace commit 62cf19945a62 subject cites this 15.42 ms figure and Strata 66.2, which is 1000/15.1, not the window's 141 tok/s. o4_dense_bench.py prints the row-kernel line and then the tensor header. The row-kernel line before attn_gate IQ3_S is 31.83 us bitwise True, before attn_q IQ3_S is 37.35 us bitwise True, and before ssm_out IQ3_S is 21.89 us bitwise True. The GEMV lines are 52.71, 65.73, and 50.48 us. attn_output Q4_K is 27.12 us on GEMV and 26.17 us on the row kernel, so the row kernel is not slower for every other type. dense2.log ends gpu-run done in 39s exit 0 after an opening card-1 busy wait. run-f13, which the note had queued, has since finished its profile: c1 step_ms p50 15.243, replay_dev p50 14.397, c2 step_ms p50 18.684, tokens byte-identical to f12, gates match, health-before HEALTHY. f13 health-after at this write is HEALTHY exit 0. No later journal heading claims the f13 numbers. f12 and f13 step_ms values are profiler steps. s3 is a 64-token harness ITL.

Verdict: OK on the f12 step, the token match, the s3 itl percentiles, the identity, the greedy and coherence files, and the three IQ3_S row-kernel times and bitwise flags. WRONG that s3 health is HEALTHY before and after: health-before.txt is absent. WRONG that s3 is 2 percent above the window's Strata line. The 15.1 ms citation is not in that window. WRONG as a blanket that other types are slower on the row kernel. s3 at 15.422 ms is about 65 tok/s, with the right campaign name, and it is not a promotion past 7.09 ms. f13's 15.243 ms is an in-process step with no journal claim. health-after is HEALTHY.

Suggested fix: cite the window's 7.09 ms line when comparing s3, and do not treat the f13 profiler step as a harness ITL.

### F268 -- ragged R1 fails the 8-replay check; R2 has no result yet

Claim: the 01:51 note gives SHA 34241ad3ef8c9e79c39692c090848e4b628d3e6d20d34c1d1aac48b027de8e77 for the new ragged library and says the probe was queued. The 02:02 note says the stock approximation gate passes, width-3 two-sequence cases pass, and at 3 sequences with lengths 3,1,2 the eager checks are exact while the 8-replay combined check fails, exit 1, with native pre and post health HEALTHY. It gives R2 SHA 248e6b78d150ef59f79ca0b726a696a10d39c6c9c3b9fd13c1fd26445c77b4be and says the card-1 probe was queued. The 02:10 and 02:06 notes announce localmaxxing priority and contain no bench numbers.

Evidence: x5/gdn-ragged/libb70_gdn_xpu.r1.so sha256 is 34241ad3ef8c9e79c39692c090848e4b628d3e6d20d34c1d1aac48b027de8e77, size 385392. libb70_gdn_xpu.so sha256 is 248e6b78d150ef59f79ca0b726a696a10d39c6c9c3b9fd13c1fd26445c77b4be, size 385440. r1/probe-source.py replays the captured graph 8 times and then sets one replay_exact flag. r1/probe.log reaches AssertionError on width 3, lengths 3,1,2, seed 0, step 0, with core_exact, out_exact, state_exact, conv_exact, and padding_exact true and replay_exact false, then gpu-run done in 19s exit 1. Earlier width-3 two-sequence rows print replay_exact true. The stock record in the log has passed true and out max_abs 1.5625e-02. r1 health-before and health-after are HEALTHY on card 1, exit 0. r2/probe.log has since exited 1; that result is F269. r2 health-after is HEALTHY exit 0. r2 health-before is HEALTHY on card 1, exit 0. The kernel cpp was not found to contain a host synchronize call under gdn-ragged. No model speed is claimed. The 02:10 and 02:06 notes have no bench JSON. lmx-l1 holds a fork-status snapshot, not a measured serve. A later health wait shows L1 holding card 1 for xpu-health; that is not a bench result. HEAD is 62cf19945a62bd96f17c1df0f484a8a6f2308b1e. Cached diff is empty. Porcelain is 20 rows. Binary porcelain row:  M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so. vllm/flashnext/.gitignore lists *.so: true. Fork HEAD is 0a6170da8ca9 on b70/flashnext, 13 files changed, 834 insertions(+), 49 deletions(-). This audit made no host change.

Verdict: OK on both library hashes, the R1 exit 1, the replay_exact failure, the 8-replay loop in the retained probe source, and both R1 health files. The R2 eager mismatch is F269. OK that the orchestrator and L1 notes in this window state no completed measurement. No speed promotion.

Suggested fix: keep the R1 replay_exact failure on record, and use F269 for the R2 eager core/output miss.

### F269 -- R2 still fails 3-sequence eager core and output; post-health is already HEALTHY

Claim: the 02:15 note says lengths 3,1,2 now fail eager core and output, while SSM, conv, and canary stay exact, persistent scratch does not fix it, the probe exits 1, post-health was queued, and reduce_128 returns parts[0] after a barrier in a way that lets the next reduction overwrite that slot before every subgroup has read the q sum.

Evidence: r2/probe.log AssertionError is width 3, lengths 3,1,2, seed 0, step 0, with core_exact false, out_exact false, state_exact true, conv_exact true, and padding_exact true. gpu-run done in 22s exit 1. Earlier width-3 two-sequence rows print the eager and replay flags true. The stock rows still show passed true. health-before and health-after both contain HEALTHY on card 1 and exit 0, so the queued post-health sentence is stale. gdn_decode.r2.cpp reduce_128 writes the 4-part sum to parts[0], barriers, then returns parts[0] with no barrier after that load. The q sum and the k sum at the call sites both pass sm + kSlmParts, so the next call reuses the same slot. No model bench JSON is in the r1 or r2 directories.

Verdict: OK on the eager miss, the exact SSM/conv/canary flags, the 22 s exit 1, and the source shape of reduce_128. The post-health sentence is stale. The probe shows the failure mode changed from R1's replay-only miss to an eager core/output miss. That does not by itself prove the missing post-load barrier is the cause. No model speed claim.

Suggested fix: barrier after every thread has loaded parts[0], or return a register copy that the next call cannot overwrite, and rerun the same 3-sequence row.

## Pass 31 -- 2026-10-21 02:51 UTC

### F270 -- R3 SHA matches and all 76 ragged rows are exact; post-health is already HEALTHY

Claim: the 02:17 note says the R3 build is a read-complete barrier in reduce_128 only, SHA 84dad4e82fa110fae57849d370fd3151ab2c09552bb83bf11b37dc94c613d130, CPU build 50 s exit 0, and R2 post-health is HEALTHY. The 02:31 note says all 76 eager, rejection, page, inactive, and 8-replay cases are bit-exact, including the 3-sequence row, stock approximation is 3 seeds and 32 replays, exit is 0, pre-health is HEALTHY, and post-health was pending.

Evidence: libb70_gdn_xpu.so sha256 is 84dad4e82fa110fae57849d370fd3151ab2c09552bb83bf11b37dc94c613d130, size 385440. The diff of gdn_decode.r2.cpp against gdn_decode.cpp is only the reduce_128 tail: it copies parts[0] into a register and barriers again before returning that register. r3/probe.log prints X5_GDN_RAGGED_NORMAL_EXIT_PASS 76 and gpu-run done in 19s exit 0. All 76 width rows have every exact flag true, including the lengths 3,1,2 row that R2 failed. stock-numeric.json has 3 seeds, passed true, and replay_exact true on the printed rows. The current probe source loops 8 replays, not 32. The 50 s build line was not opened. r3 health-before and health-after are both HEALTHY on card 1, exit 0, so the pending post-health sentence is stale. r2 health-after is also HEALTHY. No model bench is in the r3 directory.

Verdict: OK on the SHA, the one-function barrier diff, the 76 exact rows, the 19 s exit 0, the 3 stock seeds, and both health files. UNVERIFIED for a 50 s build time and for 32 stock replays. The post-health sentence is stale. This is a numeric kernel probe, not a serving ITL. No promotion.

Suggested fix: cite probe.log for the 76-row pass, and do not describe post-health as pending.

### F271 -- s4 harness ITL is 15.247 ms; health-before is absent

Claim: the 02:30 note says f13 C1 step is 15.24 ms (graph 14.40) and C2 is 18.68, tokens match f12, and gates pass. It says s4 identity is hotschmoe-dev plus the o4 alias, greedy is 323/667, coherence is OK/OK, C1 ITL p50 is 15.25 ms (p90 15.48, p99 16.26, 65.6 tok/s), C2 is 19.62/19.62, and health is HEALTHY before and after. It says C1 is 1 percent above Strata no-spec at 15.1 ms, and f14 was queued.

Evidence: run-f13 SUMMARY c1 step_ms p50 is 15.243 and replay_dev_ms p50 is 14.397. c2 step_ms p50 is 18.684. text files are byte-identical to run-f12. Gates match 323 and 667. serve-s4 bench.json C1 itl_p50_ms is 15.247206902131438, p90 is 15.481600444763899, p99 is 16.259598562028263, decode_tok_s is 65.59842990649202, n_chunks is 64. C2 itl_p50_ms is 19.618978956714272 and 19.616755074821413. identity ids are hotschmoe-dev then the o4 alias, alias_ok true. models.json max_model_len is 2048. greedy matches 323 and 667. coherence ok and same are true. mtp.enabled is false. Prefill built 1017 at 150.8618095482403 tok/s. health-before.txt is absent. health-after is HEALTHY on cards 0 and 1, exit 0. serve.log has one Application startup complete line and no READY line outside the engine-timeout variable. The window READY stamp is the alias grep. The same window prints Strata as 141.0 tok/s and 7.09 ms. 15.247/7.09 is 2.15x. 15.247/15.1 is about 1.0 percent, but 15.1 is not the line in the window. run-f14, which the note had queued, has since finished its profile: c1 step_ms p50 15.222, replay_dev p50 14.343, c2 step_ms p50 18.613, tokens byte-identical to f13, gates match, health-before HEALTHY. f14 health-after at this write is wait_or_incomplete. No journal heading claims the f14 numbers. f13 and f14 step_ms values are profiler steps. s4 is a 64-token harness ITL.

Verdict: OK on the f13 step, the token match, the s4 percentiles, the identity, and the greedy and coherence files. WRONG that s4 health is HEALTHY before and after: health-before.txt is absent. WRONG that s4 is 1 percent above the window's Strata line. s4 at 15.247 ms is about 66 tok/s, with the right campaign name, and it is not a promotion past 7.09 ms. f14 is an in-process step with no journal claim.

Suggested fix: cite the window's 7.09 ms line when comparing s4, and add a health-before probe to o4_serve.sh.

### F272 -- L1 W1 packet medians match 120.9 and 122.1; the saved files say not submitted

Claim: the 02:45 note says the frozen engine is worktree commit cfd2b6ffdb8f on b70/lmx-20261021, max_model_len is 131072, health is HEALTHY before and after, and the .so files and frozen-repo are unchanged. It says the gate passed identity, 323/667, C4 Paris, and the 8K and 32K needles, with C1 repeat bitwise and C2 prompt 3 diverging at token 80. It says reasoning is 120.9 tok/s (TTFT 719 ms) and code is 122.1 tok/s (TTFT 577 ms), 512 tokens, and that both packets were submitted and APPROVED with verifiedRun true. It says w1 and w1b failed because the frozen copy pointed at live-repo d3/libdequant paths.

Evidence: vllm-lmx-20261021 HEAD is cfd2b6ffdb8fc2a96c09712e7d04c946b149ca95 on b70/lmx-20261021. The shared fork HEAD is still 0a6170da8ca9 on b70/flashnext. models.json ids are hotschmoe-dd then the ud-iq3_xxs alias, max_model_len 131072. health-before and health-after are HEALTHY, exit 0. so-before.sha256 equals so-after.sha256. frozen-repo.sha256 and frozen-repo.final.sha256 share 351 paths and 7 of those paths changed hash, including models.yaml and several harness scripts. gate.json pass is false. The identity, greedy 323, greedy 667, c4_paris, long_8192, and long_32768 checks pass. exact_repeat has two comparisons with first_mismatch 80 and the rest null. gate-note.txt attributes that miss to the C2 batch. Payload tokSOut is 120.9 and 122.1, outputTokens 512. Validation medians are tokSOut 120.87984504820012 and 122.10281679444456. Reasoning payload ttftMs is 719.22. Code payload ttftMs is 576.73, while the validation median ttft is 575.284. validation.json submitted is false in both suites, and the cited run ids are not in lmx-l1. w1 serve.log ends in ModuleNotFoundError for b70_gdn_xpu. w1b serve.log raises dequant_bf16 is not registered. The KV-token count 569740 and the tree-hash equality were not recomputed. First id is hotschmoe-dd. These are 35B packet rates, not a Flash-Next harness ITL.

Verdict: OK on the worktree commit, the shared fork staying put, the 131072 context, both health files, the .so hash match, the listed gate checks, the token-80 mismatches, and the 120.9 and 122.1 payload rates. WRONG campaign name. WRONG that frozen-repo is unchanged: 7 manifest paths changed. WRONG against the saved validation files that the packets were submitted. UNVERIFIED for APPROVED and verifiedRun, for the 577 ms code TTFT as a median, and for the d3/libdequant path sentence. Not a Flash-Next promotion.

Suggested fix: store the submit response beside validation.json, and keep the first id hotschmoe-dev on any Flash-Next submission.

### F273 -- pass 31 hygiene

Claim: no host change by this audit, and no new secret or binary was staged.

Evidence: HEAD is 62cf19945a62bd96f17c1df0f484a8a6f2308b1e. Cached diff is empty. Porcelain is 33 rows. Binary porcelain row:  M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so. vllm/flashnext/.gitignore lists *.so: true. Fork HEAD is 0a6170da8ca9 on b70/flashnext, 13 files changed, 856 insertions(+), 52 deletions(-). This audit made no host change.

Verdict: OK. The only binary porcelain row is the known push-allreduce library.

Suggested fix: none.

### F274 -- IQ3 mixed-ragged C1 ITL is 8.1166 ms; the served first id is hotschmoe-dd

Claim: the 02:53 note says the fixed IQ3 no-spec window has C1 p50 8.1166 ms, C2 10.2861 and 10.3282, PPL 9.05563298, and that exact repeat, stock token parity, both solo-vs-C2 outputs, source, identity, coherence, exit 0, and pre and post health all pass.

Evidence: bench.json C1 itl_p50_ms is 8.116608951240778, n_chunks 62, decode_tok_s 125.12948731014187. C2 itl_p50_ms is 10.286088450811803 and 10.328217991627753. quality.json perplexity is 9.055632975432564. mtp.enabled is false. coherence.ok and coherence.same are true. models.json ids are hotschmoe-dd then the ud-iq3_xxs alias, max_model_len 4096. identity.expect_first is hotschmoe-dd and first_ok is true. repeat.json identical is true, first_mismatch is four nulls, and concurrent_identical is true. stock-parity.json identical is true and concurrent_identical is true, with four null mismatches. source-before.json equals source-after.json. serve-exit-code.txt is 0. health-before and health-after are HEALTHY, exit 0, 419 bytes each. This is a 35B proxy harness ITL.

Verdict: OK on the rounded latencies, the PPL, the repeat and stock flags, both health files, and exit 0. WRONG campaign name: the first id is hotschmoe-dd. Not a Flash-Next promotion.

Suggested fix: keep hotschmoe-dev as the first served id on this campaign, and keep this window labeled as the 35B proxy.

### F275 -- f14 step is 15.22 ms and both health files are HEALTHY; the 15.1 ms parity line is not the window

Claim: the 03:00 note says f14 C1 step is 15.22 ms (graph 14.34) and C2 is 18.61, tokens match f13, gates pass, and health is HEALTHY. It says the server C1 ITL p50 is 15.25 ms (65.6 tok/s) and C2 is 19.62, versus H1e 60.2 and Strata no-spec 15.1 ms, parity within 1 percent. The 03:20 note says f15 is pending, with o4e.so sha prefix a2b44212 and a router path of about 10 us to about 4 us.

Evidence: run-f14 c1 step_ms p50 is 15.222 and replay_dev p50 is 14.343. c2 step_ms p50 is 18.613. Those tokens and gates were already byte-matched to f13 in the prior pass. f14 health-before and health-after are now both HEALTHY on cards 0 and 1, exit 0. The 15.25 ms and 65.6 tok/s figures are the s4 harness row already recorded: itl_p50_ms 15.247206902131438 and decode_tok_s 65.59842990649202, with C2 19.618978956714272 and 19.616755074821413. The s4 window prints Strata as 141.0 tok/s and 7.09 ms. 15.247/7.09 is 2.15x. 15.247/15.1 is about 1.0 percent, and 15.1 is not the line in that window. H1e 60.2 is a different serve. b70_ep_C.o4e.so sha256 starts a2b442128205634a. run-f15 has no profile. Its health-before file is a wait on card 0, not HEALTHY, and window.log only prints library prefixes and MEM_AVAILABLE_KB. reports/2O4_flashnext_decode.md exists and was not line-audited.

Verdict: OK on the f14 step rounding and on both f14 health files. OK that s4 is 15.247 ms. WRONG that this is parity within 1 percent of the window Strata line, and the 60.2 to 15.25 arrow is not a matched A/B. UNVERIFIED for the 10 us to 4 us router claim. f15 is not a result. No promotion past 7.09 ms.

Suggested fix: cite the window 7.09 ms line, and do not publish the router microbench until f15 has a profile and closed health.

### F276 -- W2 packet rates match 97.0 and 97.2; staged responses are APPROVED while the run file still says not submitted

Claim: the 03:02 note says W2 uses engine cfd2b6ff, max_model_len 65536, KV 161792, health HEALTHY before and after, and unchanged .so, engine, and frozen tree. It says identity, 323, 667, and the 8K and 32K needles pass, C4 Paris fails only because the golden text is from another quant, and the second C2 round prompt 0 diverges at token 34. It says reasoning is 97.0 tok/s (TTFT 563 ms) and code is 97.2 (449 ms), and that ids cmvhix9xu033yrf01fma4faax and cmvhiyocc0341rf01lm0dd1k9 are APPROVED with verifiedRun true.

Evidence: engine.txt is cfd2b6ffdb8fc2a96c09712e7d04c946b149ca95. models.json first id is hotschmoe-dd, max_model_len 65536. server-config kv_cache_size_tokens is 161792. health-before and health-after are HEALTHY, exit 0. so-before.sha256 equals so-after.sha256. The pinned GDN library in that list is gdn-bf16 sha 2bb3661f, not the R3 ragged sha 84dad4e8. gate.json top pass is false. identity, greedy 323, greedy 667, long_8192, and long_32768 pass. c4_paris fails; the stored text starts "Paris, a city renowned for its iconic landmarks". exact_repeat has one mismatch, round c2_2 prompt 0 at token 34, and the other 15 comparisons are null. Payload tokSOut is 97.0 and 97.2. Reasoning payload ttftMs is 562.58 (median 558.954). Code payload ttftMs is 448.62 (median 448.622). Both run-dir validation.json files have submitted false and verification "pending localmaxxing dry-run validation; no badge claimed". The staged submit json files for those two ids have status APPROVED and verifiedRun true. The same directory now also holds the W1 ids from the 02:45 note with status APPROVED and verifiedRun true. No frozen-repo manifest is in the W2 run directory. The cited earlier packet 96.1/96.3 was not opened.

Verdict: OK on the commit, context, KV count, both health files, the .so match, the listed gate checks, the token-34 miss, and the 97.0 and 97.2 payload rates. The 563 ms and 449 ms lines are rounds of the payload TTFTs. OK that the staged responses say APPROVED and verifiedRun true. The run-dir validation files still say not submitted. WRONG campaign name. UNVERIFIED that the frozen tree is byte-unchanged, that 96.1/96.3 is the prior packet, and that the token-34 miss is the R3 reduction race. These are 35B packet rates, not a Flash-Next harness ITL.

Suggested fix: copy the submit response next to the run validation.json, and keep the first id hotschmoe-dev.

### F277 -- IQ4 mixed-ragged C1 ITL is 10.9075 ms; recipe and lmx-local are the only source-hash changes

Claim: the 03:02 note says IQ4 C1 p50 is 10.9075 ms, C2 is 11.6587 and 11.6468, PPL is 8.76477959, both concurrent responses match solo, and identity, coherence, exact repeat, source, exit 0, and pre and post health pass. It says the only IQ3-to-IQ4 source change is x5_recipe and x5_lmx_local, with model and kernel hashes matched.

Evidence: bench.json C1 itl_p50_ms is 10.907506919465959, n_chunks 56, decode_tok_s 93.7021683512854. C2 is 11.658658972010016 and 11.646786006167531. quality.json perplexity is 8.764779589576639. mtp.enabled is false. coherence.ok and coherence.same are true. repeat.json identical is true, four null mismatches, concurrent_identical true. source-before equals source-after. The IQ3 and IQ4 source sha256 maps have 56 paths in common and no path unique to either file. Two values differ: x5_recipe_35b.sh and x5_lmx_local.py. serve-exit-code.txt is 0. health-before is HEALTHY (396 bytes) and health-after is HEALTHY (419 bytes). models.json first id is hotschmoe-dd, max_model_len 4096. There is no stock-parity.json in this directory. This is a 35B proxy harness ITL.

Verdict: OK on the latencies, the PPL, the repeat flags, the two-file source diff, exit 0, and both health files. WRONG campaign name. Not a Flash-Next promotion.

Suggested fix: label the window as the 35B proxy and keep the first id hotschmoe-dev.

### F278 -- eager solo-vs-dup hashes are not bitwise identical; b2v eff is 9.8995 / 13.225 ms and the 430 denominator is not in the log

Claim: the 03:06 note says eager T=1 versus T=4 is bitwise identical on every module of every shared position from 28 positions to 89, so the b3v token-58 split exists only under FULL graphs. It says Q8_0 KREG q_proj drops from 146-168 us to 56-62 us and o_proj from 136-161 us to 35 us, with rel err at most 1.9e-7 and row invariance. It says b2v is C1 9.90 ms/token and C2 13.23, versus b2 10.20 / 13.57, with acceptance 335/430 and 263/430.

Evidence: position-aligned h0 comparison of hashes-p0d.json gives solo vs dup 72 shared positions and 72 positions with a hash mismatch, including position 28 on linear_attn.norm (nt 1 vs 2). dup vs trip is 72 positions and 0 mismatches. hashes-p3d.json solo vs dup is 28 shared positions and 28 mismatches. solo vs trip shares 13 positions and all 13 mismatch. The probe.log FIRSTDIFF lines are the same mismatches, step-aligned. No token-58 census is in probe.log. p0d and p3d have no health-before file; both health-after files are HEALTHY. bench-k0 q_proj us is 145.89, 153.39, 160.88, 167.86 and o_proj is 135.59, 143.28, 152.19, 161.26. bench-k1 q_proj is 55.72, 62.81, 58.17, 59.44 and o_proj is 35.40, 36.71, 35.50, 35.66. All 36 k1 rows are invariant, and the largest max_rel is 1.8450487004884053e-7. Both q8 health files are HEALTHY, exit 0. b2v profile PHASE means are 9.8995 and 13.225. Metrics are drafts 440, draft tokens 880, accepted tokens 598, and accepted_tokens_per_pos 335 and 263. 335+263 is 598. The number 430 is not in profile.log. b2v health-before is absent and health-after is HEALTHY. These are in-process figures, not harness ITLs. The FA capture-split mechanism was not re-read this pass.

Verdict: WRONG that every shared-position module hash is bitwise identical. OK on the Q8 bands as rounded edges, with k1 q_proj reaching 62.81 and k1 o_proj reaching 36.71, and OK that max_rel stays under 1.9e-7. OK on the b2v means and on the numerators 335 and 263. WRONG that the denominator in this log is 430. UNVERIFIED for the capture-time split sentence and for the claim that token 58 diverges only under FULL graphs. No promotion.

Suggested fix: print the position-aligned census, and cite drafts 440 or draft tokens 880 as the denominator.

### F279 -- Q4 mixed-ragged C1 ITL is 10.3716 ms; post-health is now HEALTHY and verified=true is not in the validation file

Claim: the 03:21 note says Q4 C1 p50 is 10.3716 ms, C2 is 11.5693 and 11.5573, PPL is 8.50822826, both C2 responses match solo, and identity, coherence, exact repeat, source, and exit 0 pass. It says reasoning is 94.8 tok/s and code is 94.9, dry-run valid true, verified true, issues empty, no submit, and that Q4 dominates IQ4. It says post-health has been waiting since 03:11.

Evidence: bench.json C1 itl_p50_ms is 10.371617041528225, n_chunks 56, decode_tok_s 98.4098816931264. C2 is 11.569287977181375 and 11.557296034879982. quality.json perplexity is 8.508228260209568, against IQ4 8.764779589576639. mtp.enabled is false. coherence.ok and coherence.same are true. repeat.json identical is true, four null mismatches, concurrent_identical true. source-before equals source-after. serve-exit-code.txt is 0. health-before was HEALTHY. health-after is now HEALTHY, exit 0, 7416 bytes, so the waiting sentence is stale. Validation medians are tokSOut 94.75425748299934 and 94.9330940766703, submitted false, issues empty, valid true, verification "pending localmaxxing dry-run validation; no badge claimed". models.json first id is hotschmoe-dd, max_model_len 4096. Prefix bytes behind the two PPL files were not diffed. This is a 35B proxy harness ITL.

Verdict: OK on the latencies, the PPL values, the repeat flags, exit 0, and both health files. OK that Q4's C1 and stored PPL beat this IQ4 pair. WRONG campaign name. WRONG that the validation file says verified true. The no-submit flag matches. Not a Flash-Next promotion.

Suggested fix: drop verified=true unless the verification string says so, and record that post-health has closed.

### F280 -- W3 Flash-Next packet rates are 61.3 and 61.5 tok/s; first id is hotschmoe-dd and W3b has no result

Claim: the 03:30 note says W3 uses cfd2b6ff and the O4 s4 recipe at max_model_len 131072, KV 491905, max_num_seqs 4, and batched 2048. It says health is HEALTHY before and after, integrity is unchanged, and the gate passes identity, 323, 667, and the 8K and 32K needles. It says C1 repeats are bitwise and C2 prompt 0 differs from C1 at token 2 in both rounds. It says reasoning is 61.3 tok/s (TTFT 2315 ms) and code is 61.5 (1455 ms), inter-chunk mean 16.3 ms rising from 15.9 to 16.7, compared with s4 at 15.25 ms over 64 tokens. It says the packet is not submitted and W3b is still running.

Evidence: engine.txt is cfd2b6ffdb8fc2a96c09712e7d04c946b149ca95. models.json first id is hotschmoe-dd, then the flash-next iq3 alias, max_model_len 131072. server-config model is the Flash-Next HF tree, kv_cache_size_tokens 491905, max_num_seqs 4, max_num_batched_tokens 2048. health-before and health-after are HEALTHY on cards 0 and 1, exit 0. so-before.sha256 equals so-after.sha256. gate identity, greedy 323, greedy 667, long_8192, and long_32768 pass. exact_repeat fails. The only mismatches are c2_1 prompt 0 and c2_2 prompt 0, both first_mismatch 2 against the c1_1 baseline. Payload tokSOut is 61.3 and 61.5. Reasoning payload ttftMs is 2315.24 (median 2313.517). Code payload ttftMs is 1455.34 (median 1455.485). Both validation.json files have submitted false. No labeled inter-chunk mean of 16.3, 15.9, or 16.7 was found outside sample streams. s4 remains the 64-token harness row at max_model_len 2048. w3b health-before is a wait, health-after is absent, and gate.json is absent. 1000/61.5 is about 16.3 ms. The Strata reference on the s4 window is 7.09 ms.

Verdict: OK on the commit, the 131072 context, the KV count, both W3 health files, the .so match, the listed gate checks, the token-2 indexes, the 61.3 and 61.5 rates, and not-submitted. WRONG campaign name. UNVERIFIED for the 15.9 to 16.7 ramp and for byte equality of the two C2 outputs. W3b is not a result. This packet is slower than the s4 15.247 ms row and is not a matched A/B with it. No promotion.

Suggested fix: serve hotschmoe-dev first, and leave W3b uncited until its gate and both health files close.

### F281 -- pass 32 hygiene

Claim: no host change by this audit, and no new secret or large binary was staged. The new EP dispatch path was skimmed for a host sync inside the kernel.

Evidence: HEAD is 2302f40bb59064de61fd4c480a5efbe46c6ea1b9. Cached diff is empty. Porcelain is 25 rows. Binary porcelain row: M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so. vllm/flashnext/.gitignore lists *.so. Fork HEAD is 0a6170da8ca9 on b70/flashnext, 13 files changed, 856 insertions(+), 52 deletions(-). The validate notes name the --api-key flag and do not hold a key value. ep_dispatch checks the expert id before indexing and does not call a host wait. ep_touch_pages and ep_store_flag_pages do call queue wait; those are setup helpers. The w8_gemv diff adds no .wait. This audit made no host change and did not commit the HEAD move.

Verdict: OK. The only binary porcelain row is the known push-allreduce library. No capture-time host sync was found in ep_dispatch on this skim.

Suggested fix: none.

### F282 -- MTP2 wire ITL is 17.934 ms; 6.6182 ms is 1000/decode_tok_s, and post-health is still a wait

Claim: the 03:28 note says all 576 raw token ids across 4 C1 and 2 C2 match the no-spec baseline, the same-load repeat passes, and ragged ntok 1/2 at slot (2,3) uses the fused kernel with no old fallback. It says C1 effective is 6.6182 ms (wire p50 17.9341) and C2 effective is 9.9137 and 8.5302 (wire 23.0351 and 23.0406), with bench accept 91/120. It says canonical rates are 147.1 and 141.8 tok/s, accept 85.714 and 80.203 percent, mean accepted length 2.714 and 2.604, local valid with ranking_eligible false, the API dry-run completed, and there is no submit. Post-health is queued.

Evidence: parity.json was compared with the no-spec greedy.json from qwen36-35b-udq4km-fastq8-canon-mixed-ragged-mtp0. identical is true, concurrent_identical is true, and first_mismatch is four nulls. The file holds 4 rows and 2 concurrent rows of 96 token ids, 576 ids in total. repeat.json uses this serve's own greedy.json as baseline and has the same flags. bench.json C1 itl_p50_ms is 17.93412899132818, n_chunks 22, decode_tok_s 151.09871264823215. 1000/151.09871264823215 is 6.6182. C2 wire is 23.035096004605293 and 23.04057206492871, and 1000 over those decode_tok_s values is 9.9137 and 8.5302. The window mtp line prints accepted_during_bench 91 and draft_tokens_during_bench 120. Payload tokSOut is 147.1 and 141.8. specAcceptanceRate is 0.8571428571428571 and 0.8020304568527918. specMeanAcceptedLength is 2.7142857142857144 and 2.6040609137055837. validation.json ranking_eligible is false, submitted is false, and the verification string is the pending dry-run line. The dry-run header says local validation only and that the API dry-run still requires an api key. models.json first id is hotschmoe-dd. source-before equals source-after. serve exit is 0. health-before is HEALTHY, 387 bytes, exit 0. health-after is now HEALTHY, 6647 bytes, exit 0, so the queued post-health sentence is stale. The window prints FUSED_GDN_ACTIVE=1. serve.log has shared-expert FUSED lines at T=1 and T=2. No slot (2,3) or ragged-fallback line was found. The window Strata line is 7.09 ms. The window also prints n=57 while bench n_chunks is 22.

Verdict: OK on the 576-id parity and the same-load repeat flags, the wire percentiles, the 91/120 accept count, the packet rates, ranking_eligible false, and no submit. The 6.6182 ms figure is 1000/decode_tok_s, not the harness ITL. WRONG campaign name. WRONG that this dry-run header shows a completed API dry-run. UNVERIFIED for the slot (2,3) fused-kernel sentence. Post-health has since closed, so the queued sentence is stale. Not a promotion. The wire C1 ITL is 17.934 ms.

Suggested fix: cite 17.934 ms as the harness ITL. Post-health is already HEALTHY.

### F283 -- p7 prefill table matches the probe, and post-health is now HEALTHY; only one gate arm is stored

Claim: the 00:40 note says p7 health is HEALTHY before and after, both arms pass 323/667, clean TTFT is 924, 2434, 2511, 2342, and 1587 tok/s at the listed builds, and decode C1 ITL p50 is 18.49 ms with C2 22.70/22.71. It says the 12288 profile is push 0.27 s, card-0 experts 1.02-1.24 s, wait+add 1.3-1.6 s, card-1 experts 1.71-2.03 s, host_c1 0.26-0.37 s, and between-layers 1.92 s rising to 7.9 s.

Evidence: probe.json prefill_tok_s values were already 923.837, 2433.644, 2511.124, 2342.246, and 1586.657. decode_c1 itl_p50_ms is 18.49067909643054 and the C2 pair is 22.696322994306684 and 22.70782506093383. gates_by_arm has one arm, the profile-off line, not two. health-before and health-after are now both HEALTHY, exit 0. The 12288 profile rows show dispatch_push about 272 ms, c0_experts about 1.01-1.06 s, wait_add about 1.30-1.60 s, and c1_experts about 1.71-2.02 s. host_c1 on those rows is about 0.26-0.31 s. The 1.24 s expert line and the 0.37 s host_c1 line are the T=12283 row. between-layers on the first T=12288 row is 1920.5 ms and a later T=12288 row is 7882.5 ms. This decode row is not the s4 harness ITL of 15.247 ms. The READY 00:20:56 clock was already not an engine READY line.

Verdict: OK on the prefill rates, the 18.491 ms decode ITL, and both health files. WRONG that two gate arms are stored. The profile band is OK if the T=12283 row is included; the strict T=12288 host_c1 cap is 0.31 s. UNVERIFIED that this decode equals the older f4 row. These are prefill rates plus a 64-token decode ITL. No promotion.

Suggested fix: say one stored arm, and keep 2511 labeled prefill_tok_s.

### F284 -- MTP3 wire C1 ITL is 17.853 ms; 5.6736 ms is 1000/decode_tok_s, and verified=true is not the validation string

Claim: the 03:40 note says all five final windows pass identity, coherence, repeat, source, exit 0, and pre and post health. It says MTP3 C1 effective is 5.6736 ms and C2 is 11.0375/9.0111, with wire 17.8525 and 27.5076/27.5768, accept 98/159. It says both MTP depths match all 576 no-spec token ids, the three Q4 source maps match, canonical rates are 181.7 and 163.5 tok/s, acceptance is 79.912 and 67.653 percent, and both MTP3 dry-runs have verified true with ranking_eligible false. It says the controller finished at 03:37:12Z and DD is inactive.

Evidence: bench.json C1 itl_p50_ms is 17.852537508588284, n_chunks 19, decode_tok_s 176.25523301627436. 1000/176.25523301627436 is 5.6736. C2 wire is 27.50762295909226 and 27.576814987696707, and 1000 over those rates is 11.0375 and 9.0111. accepted_during_bench is 98 and draft_tokens_during_bench is 159. parity.json baseline is the no-spec greedy.json, identical true, concurrent_identical true, four null mismatches, 4 rows plus 2 concurrent rows of 96 ids. repeat.json matches this serve's own greedy.json. The no-spec, MTP2, and MTP3 source sha256 maps are equal, 56 paths. Payload tokSOut is 181.7 and 163.5. specAcceptanceRate is 0.7991169977924945 and 0.6765285996055227. specMeanAcceptedLength is 3.3973509933774833 and 3.029585798816568. validation.json ranking_eligible is false, submitted is false, and the verification string is the pending dry-run line. The dry-run header says the API dry-run still requires an api key. The no-spec validation files have ranking_eligible true and submitted false. health-before and health-after are HEALTHY, 387 bytes, exit 0. models.json first id is hotschmoe-dd. final-fixed-gdn-launch.log prints X5_FINAL_ALL_DONE 2026-10-21T03:37:12Z. hotschmoe-dd.service is inactive. The wire C1 ITL is slower than the no-spec wire 10.372 ms. Summary c1_effective_ms for the no-spec rows is 1000/decode_tok_s, not the wire p50.

Verdict: OK on the wire numbers, the 576-id parity, the source-map match, the packet rates, ranking_eligible false, no submit, both health files, the controller stamp, and DD inactive. The 5.6736 ms figure is 1000/decode_tok_s, not the harness ITL. WRONG campaign name. WRONG that the validation string or the dry-run header says an API verified badge. MTP3's effective C1 is faster than MTP2's effective C1, and MTP2's effective C2 is faster than MTP3's. No promotion.

Suggested fix: cite 17.853 ms as the harness ITL, and keep verified true off the pending validation string.

### F285 -- f14 replay means are not a steady 14.31 to 14.46 ms rise; La, Lb, and Lc have no result

Claim: the 03:55 note says the f14 card-0 graph grows from 14.31 to 14.46 ms over 96 tokens, about 1.5 us per KV token, the same slope as a 0.8 ms rise over 512 tokens. It says a fused HC mix drops 48 mixes from 1.64 to 1.22 ms at T=1. La, Lb, and Lc are queued. The verdict is pending.

Evidence: steps-c1.jsonl has 95 replay_dev_ms samples. The 16-step means are 14.334, 14.314, 14.254, 14.346, 14.424, and 14.461. The 14.314 and 14.461 ends match the cited rounding, and the third bucket is 14.254, so the series dips. 0.15 ms over about 96 tokens is about 1.6 us per token. The 1.64 to 1.22 line is not in harness/o4_hcq8_fuse_test.py. No La, Lb, or Lc profile directory was graded. The L1 0.8 ms rise was not a labeled mean in the W3 window.

Verdict: OK that the later replay bucket ends near 14.31 and 14.46. WRONG as a steady rise across the six buckets. UNVERIFIED for the 1.64 to 1.22 mix and for the L1 0.8 ms series. La, Lb, and Lc are not results. No promotion.

Suggested fix: print all six bucket means, and wait for a closed La window before citing the mix.

### F286 -- W3b packet rates match 61.3 and 61.5 at max_len 40960; the 66.2 tok/s bar is not the window Strata line

Claim: the 03:52 note says W3b at max_model_len 40960, KV 450560, has health HEALTHY before and after, the same gate shape as W3, and C2 divergence on prompt 1 at token 1 in both rounds. It says reasoning is 61.3 tok/s (TTFT 2315 ms) and code is 61.5 (1448 ms), with per-64-token means 15.92 to 16.73 ms and 15.81 to 16.60 ms, within 0.05 ms of the 131072 window. It says Flash-Next is not submitted because 61.3 is below Strata 66.2.

Evidence: models.json first id is hotschmoe-dd, max_model_len 40960. server-config kv_cache_size_tokens is 450560. health-before and health-after are HEALTHY, exit 0. so-before equals so-after. engine.txt is cfd2b6ffdb8fc2a96c09712e7d04c946b149ca95. gate identity, 323, 667, and both needles pass. exact_repeat fails at c2_1 prompt 1 and c2_2 prompt 1, both first_mismatch 1. Payload tokSOut is 61.3 and 61.5. Reasoning payload ttftMs is 2315.49. Code payload ttftMs is 1448.3, while the validation median ttft is 1915.169. Both validation.json files have submitted false. The s4 window Strata line is 141.0 tok/s and 7.09 ms. 66.2 tok/s is 1000/15.1, and 15.1 is not that line. The 15.92 and 16.73 substrings occur inside stream JSON and were not found as a labeled bucket mean. The directory flashnext-iq3_xxs-NOT-SUBMITTED exists.

Verdict: OK on the context, KV count, both health files, the .so match, the gate indexes, and the 61.3 and 61.5 payload rates. The 1448 ms line is the payload TTFT, not the median. WRONG campaign name. WRONG that 66.2 tok/s is the window Strata rate. UNVERIFIED for the per-64-token means and for equality within 0.05 ms of the 131072 series. Not submitted matches the validation files. No promotion.

Suggested fix: compare 61.3 tok/s with the window 141 tok/s line, and store the bucket means beside the payload.

### F287 -- pass 33 hygiene

Claim: no host change by this audit, and no new secret or large binary was staged.

Evidence: HEAD is 2d5ff2a1ad7335dde315c0c3e3fc8c127397ac8f. Cached diff is empty. Porcelain is 14 rows. Binary porcelain row: M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so. vllm/flashnext/.gitignore lists *.so. Fork HEAD is 0a6170da8ca9 on b70/flashnext, 13 files changed, 856 insertions(+), 52 deletions(-). This audit made no host change and did not stop or start the daily service.

Verdict: OK.

Suggested fix: none.

### F288 -- merged m1 C1 harness ITL is 15.201 ms with first id hotschmoe-dev; it is not a matched s4 A/B

Claim: the 04:05 note says merged m1 health is HEALTHY before and after, identity is hotschmoe-dev plus the o4 alias, greedy is 323/667, the 32K filler is 323/667, and bench coherence is ok and same. It says bench C1 ITL p50 is 15.20 ms (65.8 tok/s) and C2 is 19.54/19.55, probe C1 is 15.21 and C2 is 19.52/19.54, unchanged from s4 at 15.25/19.62. Warm prefill is 920, 2448, 2397, and 2349 tok/s. Bench prefill is 751 and 2048 tok/s. KV is 356207 tokens, 2.72x at 131072. SERVE_RC is 0. Library prefixes are 790230cb and 249cf94e.

Evidence: bench.json C1 itl_p50_ms is 15.200939029455185, decode_tok_s 65.82580979614309, n_chunks 64. C2 is 19.53622093424201 and 19.550336990505457. identity ids are hotschmoe-dev then the o4 alias, first_ok true, alias_ok true. coherence ok and same are true. mtp.enabled is false. probe C1 itl_p50_ms is 15.205517993308604 and C2 is 19.524318049661815 and 19.537833984941244. Probe prefill_tok_s is 920.151 at built 1018, 2447.816 at 8186, 2396.977 at 12283, and 2349.438 at 32763. Bench prefill_tok_s is 751.338 at built 1017 and 2048.230 at built 8185. Probe gates and the 32788-token long check both match 323 and 667. health-before and health-after are HEALTHY, exit 0. models.json max_model_len is 131072. serve.log prints KV 356,207 tokens and 2.72x, and one Application startup complete line. The window READY stamp is 03:55:59. SERVE_RC in the window is 0. _C.f1bucket.so sha256 starts 790230cb and b70_ep_C.f1c.so starts 249cf94e. s4 was max_model_len 2048 at 15.247 ms. 15.201 is still about 2.1x the window Strata line of 7.09 ms.

Verdict: OK on the harness percentiles, the warm prefill rates, the bench prefill rates, identity, gates, both health files, the KV count, and the library prefixes. The s4 comparison is close on the C1 number and is not a matched max_len A/B. No promotion past 7.09 ms.

Suggested fix: cite s4 as a different context length, and keep 15.201 ms as the harness ITL.

### F289 -- q1 and q2 have empty serve logs; the queued bucket2 sha prefix matches

Claim: the 04:30 note says q1 and q2 are queued and the result is pending. It cites _C.f1bucket2.so sha prefix 958e5d06 and a prior QSA helper time of about 2.4 s per 12288 chunk.

Evidence: merged-q1/serve.log and merged-q2/serve.log are both 0 bytes. No bench or health file is in either directory. _C.f1bucket2.so sha256 starts 958e5d06. The 2.4 s figure was not opened.

Verdict: OK that q1 and q2 are not results, and OK on the bucket2 sha prefix. UNVERIFIED for the 2.4 s helper time. No promotion.

Suggested fix: leave q1 and q2 uncited until bench.json and both health files exist.

### F290 -- recipe c PHASE means match; the c2v denominator is 440; b2v is not a full match

Claim: the 04:20 note says c3v C1 is 9.19 ms/token and C2 is 13.43, with acceptance 284/218/161 of 384, propose 3.3 to 2.2 ms, and verify at T=4 is 21.2 ms. It says c2v C1 is 9.66 and C2 is 12.89, with 336/261 of about 430. It says c0 is 16.21 and 20.23. It says c3v equals b0 on all 8 streams, c0 differs from c3v on the hash-map C1 stream at token 58, that token is 1287 or 17839 in the named arms, every other stream matches in every arm, and k=2 parity of b0 against b2 passes.

Evidence: run-c3v profile.log PHASE c1 eff_ms_per_tok mean is 9.19125 and c2 mean is 13.427. METRIC drafts are 384, draft tokens 1152, accepted 663, per-position 284, 218, 161. run-c2v PHASE c1 mean is 9.6565 and c2 mean is 12.8945. METRIC drafts are 440, draft tokens 880, accepted 597, per-position 336, 261. run-c0 PHASE c1 mean is 16.208 and c2 mean is 20.226. b3v propose_dev_ms medians on the four C1 step files, dropping rows with step_ms at or above 80, are 3.329, 3.335, 3.309, and 3.329. c3v propose_dev_ms medians on ntok 4 rows are 2.204, 2.209, 2.184, and 2.200. c3v replay_dev_ms medians on those rows are 21.034, 21.409, 21.377, and 21.163. Token files: c3v matches b0 on four C1 streams of 128 ids and on both C2 entries. Each C2 entry holds two length-128 sequences, so the stored set is 8 sequences. The only c0 versus c3v C1 difference is stream 1 at index 58: 1287 versus 17839. At that index, n0, b3v, and c0 hold 1287, and n2, n3v, b0, b2, b2v, and c3v hold 17839. b0 matches b2 on every stored sequence. b2v C1 stream 3 first differs from b0 at index 81, 1876 versus 6321. No health-before file is present for c3v, c2v, or c0. Each health-after file is HEALTHY. The served-name string in the profiler kwargs starts with hotschmoe-dev. These means are profiler eff_ms_per_tok, not harness ITL.

Verdict: OK on the PHASE means, the 284/218/161 of 384 counts, the 336 and 261 numerators, the rounded propose and replay figures, the index-58 split, c3v equaling b0 on the 8 sequences, and b0 equaling b2. WRONG that the c2v denominator is about 430, because drafts are 440. WRONG that every other stream matches in every named arm, because b2v stream 3 differs at index 81. Pre-health is UNVERIFIED because the file is absent. No promotion. 9.19125 ms is not under the window Strata line of 7.09 ms, and it is not harness ITL.

Suggested fix: cite drafts 440 for c2v, and do not call b2v a full token match.

### F291 -- FA graph verify rows match decode-graph rows; fixed splits also match eager past 2048

Claim: the 04:20 note says verify rows equal decode graph rows at every past for auto and fixed splits, and fixed splits also make graph equal eager above the 2048 capture bound, where auto differs.

Evidence: fag-0-3, fag-0-4, fag-16-3, and fag-16-4 record cap_maxk 2048. graph_verify_vs_graph_decode is true on every row. For splits 0, graph_decode_vs_eager_decode is false at past 2500 and at the last past, which is 3196 when ql is 3 and 3195 when ql is 4. For splits 16, graph_decode_vs_eager_decode is true on every row, including past 2500 and the last past. health-before2.txt and health-after2.txt both contain HEALTHY, and window2 reports RC 0. This is a card-1 kernel replay, not a serving ITL.

Verdict: OK. The ql=4 series ends at past 3195.

Suggested fix: none.

### F292 -- c3vq ran out of memory on card 1; health-after is HEALTHY

Claim: the 04:20 note says c3vq runs out of memory on card 1 because 412 experts do not fit, and that Q0 cannot go much below 138.

Evidence: run-c3vq/run.log raises torch.OutOfMemoryError on XPU, tried to allocate 364.00 MiB. GPU 1 total capacity is 31.89 GiB, of which 322.47 MiB is free. 31.00 GiB is allocated by PyTorch, and 156.30 MiB is reserved. Split lines say E=512, n0=100, n1=412. Window RUN_RC is 1. health-after is HEALTHY. There is no health-before file. profile.log does not contain the OOM line. The 138 figure was not derived from these logs.

Verdict: OK on the OOM and the 412 expert split. UNVERIFIED for the 138 clause. Not a speed result.

Suggested fix: none.

### F293 -- q1 failed in _rope_ at line 175 with Level Zero error 44; both health files are HEALTHY

Claim: the 04:40 note says QSA registered, the KV block size stayed 64 for FLASH_ATTN, the private path ran, _rope_ failed with Level Zero error 44 because of a host sync inside capture, SERVE_RC is 1, and health-after was still queued.

Evidence: merged-q1/window.log has HEALTH_BEFORE_RC 0, serve exited before ready, BENCH_RC 2, and SERVE_RC 1. serve.log contains the line b70_ep: B70_QSA=1 registered; MTP layers will be marked draft, and the line Setting kv cache block size to 64 for FLASH_ATTN backend(s). The traceback goes through model_hook.py line 675 _forward_private and line 175 _rope_, then RuntimeError level_zero error 44 UR_RESULT_ERROR_UNSUPPORTED_FEATURE. The serve log contains zero copies of the string host sync, zero copies of qsa_cover, and zero copies of uint8. models.json is 0 bytes. health-before.txt and health-after.txt both end HEALTHY with exit 0. In model_hook.py, line 175 is torch.arange on the tensor device. The comment about .item() synchronizing is in _note_mrope, around line 478, and that function does not call .item().

Verdict: OK on the failed serve, the private-path frames, error 44, and both health files. The health-after queued sentence is stale. UNVERIFIED that error 44 is a host sync. The F289 empty-log sentence for q1 is stale and was not rewritten. No bench and no /v1/models identity. No promotion.

Suggested fix: cite health-after as HEALTHY, and do not call line 175 a host sync unless a host operation is shown on that line.

### F294 -- c0r same-load repeat flips index 58; d32v has a profiler result

Claim: the 04:20 note queued c0r to settle whether the hash-map token flip is cross-load only, and it queued d32v. It had not reported either result.

Evidence: run-c0r profile.log PHASE c1 mean is 16.19325 and c2 mean is 20.2295. REPEAT c1 versus c1r is True, False, True, False. Token c1 versus c1r differs on stream 1 at index 58, 1287 versus 17839, and on stream 3 at index 81, 1876 versus 6321. Gates match 323 and 667. RUN_RC is 0. health-after is HEALTHY. There is no health-before file. The kwargs served-name string starts with hotschmoe-dev. run-d32v PHASE c1 mean is 9.2755, median 9.7275, accept_mean 0.561525, and c2 mean is 12.896. REPEAT is True, False, True, True. Gates match 323 and 667. health-after is HEALTHY. There is no health-before file. d32v versus b0 first differs on C1 stream 1 at index 110, 694 versus 321. Index 58 on that stream is in the 17839 group. Both runs are profiler eff_ms_per_tok, not harness ITL.

Verdict: OK that both runs have finished and that the 04:20 queued wording is stale for them. The c0r same-load mismatch shows index 58 can flip inside one load, 1287 versus 17839, so a cross-load-only reading is not supported. Pre-health is UNVERIFIED. No promotion. 9.2755 ms and 16.19325 ms are not harness ITL and are not under 7.09 ms.

Suggested fix: cite the same-load index-58 mismatch, and keep these means labeled as profiler eff.

### F295 -- X6 CPU logs pass; consultations did not log in; the 1GiB estimate was not measured

Claim: the 04:26 and 04:33 notes say the nested Attention binding was repaired, CPU tests print QSA_SERVE_OK and QSA_SPEC_OK, the hybrid slope is 1184 bytes per token, per-token .item() was removed from mRoPE bookkeeping, and both Claude consultations failed with unrecognized_model and not logged in. The 04:44 note says scratch is allocated during the dummy profile, a 12K prefill could need more than 1GiB of extra workspace, two warmups remain, research aliases were added, the stable DD entry is still an older 27B stack, q2 holds both cards, and the graph probe still yields. It claims no latency.

Evidence: x6/cpu-serve.log ends with QSA_SERVE_OK and contains the inner bind line for layers.3.self_attn.attn. x6/cpu-spec.log ends with QSA_SPEC_OK and contains PASS hybrid byte slope. tests/test_kv_spec.py checks page_size_bytes equals 1184 for that slope name, on a block_size 1 spec. harness/x6_serve.sh sets cudagraph_num_of_warmups to 2. model_hook.py _note_mrope does not call .item(). The nearby comment says an .item() there used to synchronize. .item() calls remain in serve.py and batch.py. consult-fable-5.1.log and consult-claude-5.5.log each contain unrecognized_model and the text Not logged in. No responding model id is present. graph-window.log contains only X6_YIELD lines, 71 of them. evals/configs/models.yaml label hotschmoe-dd is family qwen3.8-27b. New labels flashnext-ista-iq3-xxs-x6-qsa-research and flashnext-ista-iq3-xxs-x6-dense-research exist. The scanned entry bodies do not set primary_client_id, and the notes name hotschmoe-dd. prompts.log prints X6_REFERENCE_PROMPTS_READY with three lengths. reference-prompts.json was not read line by line. The dummy-profile allocation and the 1GiB sentence were not found in a measured log.

Verdict: OK on the two OK lines, the 1184 assertion, the warmup count of 2, the consult failures, the yield log, the 27B DD family, and the absence of a latency claim. UNVERIFIED for the dummy-profile scratch story and the 1GiB gap. No GPU serve and no /v1/models identity. q1 already showed that a CPU QSA_SERVE_OK does not mean a captured engine bind.

Suggested fix: keep the 1GiB figure uncited until a profile log records the workspace bytes, and serve the research alias with first id hotschmoe-dev.

### F296 -- pass 34 hygiene

Claim: this pass made no host change, did not stop or start the daily service, and did not stage secrets or a new binary.

Evidence: HEAD is 2d5ff2a1ad7335dde315c0c3e3fc8c127397ac8f. Cached diff shortstat is empty. Porcelain is 33 rows. Binary-looking porcelain rows: M vllm/contrib/vllm_push_allreduce/prebuilt/libxpu_push_ar_graph.so. Secret-like porcelain names: none. Untracked paths at or above 1000000 bytes: none. vllm/flashnext/.gitignore lists *.so: yes. Fork HEAD is 0a6170da8ca9 on b70/flashnext, 13 files changed, 856 insertions(+), 52 deletions(-). systemctl --user is-active hotschmoe-dd printed inactive. This audit made no host change and did not stop or start the daily service.

Verdict: OK.

Suggested fix: none.

### F297 -- q2 harness C1 ITL is 15.233 ms with first id hotschmoe-dev; post-health status is live

Claim: no journal heading has cited a q2 speed number. F289 said the q2 serve log was empty. The bench is now on disk.

Evidence: bench.json C1 itl_p50_ms is 15.233078971505165, decode_tok_s 65.84819237901634, n_chunks 64. C2 itl_p50_ms is 19.4929379504174 and 19.522273913025856. identity ids are hotschmoe-dev then the o4 alias, first_ok true, alias_ok true, expect_first hotschmoe-dev. coherence ok and same are true. mtp.enabled is false. models.json first id is hotschmoe-dev and max_model_len is 131072. Bench prefill_tok_s is 659.182 at built 1017. Probe prefill rows: built 8186 target 8192 prefill_tok_s 2056.349 n_chunks 1; built 12283 target 12288 prefill_tok_s 1949.099 n_chunks 0. Probe gates match 323 and 667, and probe identity first_ok is true. The window GREEDY record matches 323 and 667. The window Strata line is C1 141.0 tok/s, itl 7.09 ms. serve.log contains 1 Application startup complete line. The serve.log READY hits are on the launch line, not a separate engine READY record: -e VLLM_NO_USAGE_STATS=1 -e VLLM_ENGINE_READY_TIMEOUT_S=1200 -e SYCL_CACHE_PERSI. The window stamp is READY 04:48:01. health-before contains HEALTHY. health-after does not. Its last line is: gpu-run: card 0 busy (held by: O5 pid=4013157 since=2026-10-21 04:48:34 cmd: docker run --rm --init --name vllm-fn-o5-p0g --user 1000:1000 --group-add 991 --ipc. Window flags: SERVE_RC=SERVE_RC:0, BENCH_RC=BENCH_RC:0, HEALTH_BEFORE_RC=HEALTH_BEFORE_RC:0, HEALTH_AFTER_RC=absent, WINDOW_DONE=absent. m1 at max_len 131072 was harness C1 15.201 ms. 15.233 ms is not under 7.09 ms.

Verdict: OK on the harness percentiles, identity, coherence, mtp off, the bench prefill rate, and health-before. Post-health is not closed. The 12283 probe row stores a prefill rate with n_chunks 0, so that rate is a weaker record than the 8186 row. F289 is stale about the empty q2 serve log and was not rewritten. No promotion.

Suggested fix: keep 15.233 ms as the harness ITL, and cite post-health only after the file ends HEALTHY.

### F298 -- X6 Strata reference client is prepared; the GPU reference has not run

Claim: the 04:50 note says the Strata reference client is prepared on CPU, does not start the daily service, and has not run. It claims no timing comparison.

Evidence: harness/x6_strata_reference.py contains StrataEngine and does not contain systemctl. harness/x6_strata_window.sh says it does not start or alter hotschmoe-dd, writes pre and post health files, and sets an 8m timeout. It does not contain systemctl. x6/strata-tokenizer.log is the line X6_STRATA_TOKENIZER_OK [4096, 16384, 32768]. x6 has no strata-health file in this listing: no.

Verdict: OK that the GPU reference has not run and that the scripts do not call systemctl. The tokenizer line is a CPU check, not a generate result and not a latency. No promotion.

Suggested fix: none.

### F299 -- Lc profiler step mean is 15.696 ms; the 0.54 ms gap is not harness ITL

Claim: the 04:52 note says La reasoning step mean is 16.25 ms with buckets 15.86 to 16.66, the card-0 graph grows 14.98 to 15.82 while sample stays 0.83, Lb mean is 16.24 with buckets 15.82 to 16.74 and first token diffs at 96 and 20, and Lc reasoning mean is 15.70 with buckets 15.28 to 16.16, code 15.58, and C2 20.08, with token diffs at 96 and 84. It keeps route plus Q8F for a 0.54 ms gain. Gates are 323 and 667, and health is HEALTHY.

Evidence: SUMMARY step_ms means are La c1 16.248, Lb c1 16.238, Lc c1 15.696, Lc code 15.582, and Lc c2 20.078. Printed step bucket medians match 15.86 to 16.66, 15.82 to 16.74, and 15.28 to 16.16. La replay bucket medians on graph steps go from 14.98 to 15.83, and the index-441 bucket is 15.82. Sample bucket medians stay 0.83. 16.238 minus 15.696 is 0.542. Token first diffs are index 96 and index 20 for La versus Lb, and index 96 and index 84 for Lb versus Lc. Gates match 323 and 667 on La, Lb, and Lc. La health-before and health-after contain HEALTHY. Lb and Lc have no health-before file, and their health-after files contain HEALTHY. The kwargs served-name string starts with hotschmoe-dev. run-Ld has a profile header and no SUMMARY. These are profiler step_ms values, not harness ITL.

Verdict: OK on the step means, the printed buckets, the token-diff indexes, the gates, and the 0.54 ms rounding. The cited graph end of 15.82 is the index-441 replay median; the last bucket is 15.83. The streams diverge, so the 0.54 ms gap is not a same-token comparison. Pre-health for Lb and Lc is UNVERIFIED. Ld is not a result. No promotion. 15.696 ms is not under 7.09 ms.

Suggested fix: cite the last replay bucket as 15.83, and keep the 0.54 ms figure labeled as a profiler step-mean gap.
