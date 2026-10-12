# Paired original MTP OFF target versus accepted normal52

CONFIG -> Live V3 natural first-case OFF: same pinned source52 native target,
materialized0, spec2, pref512, context4096, FP16 KV, short-read0/noBorrow,
mirrorT1off. Reference is source52-native2-fairness-v1-run-v1 serial long A,
P2048/GEN512. No live producer/source mutation or runtime action in this audit.

COMMAND -> Source/metadata reads of both original recipe.json files plus the
actual52 generate/session/prefill/verifier sources and paired_greedy_v1.hpp.
No provider/GGUF reads, kernel build, GPU, new capture or numerical execution.

RESULT -> Root observed first102 output IDs equal; token103 differs (17295
versus71367). Source shows TWO distinct unresolved issues:

1. Missing fresh secondary state reset is a correctness gap independently of
   this divergence. generate3769 zeros only the primary during construction.
   The stage session_init3792 carves pointers; the stage loop ends3819 without
   session_zero. session.cpp159-169 shows GDN/PLE zero belongs to session_zero,
   not session_init. Normal public fresh request reset10040-10056 resets BOTH
   owners and persistent QSA metadata; the one-shot hook bypasses it. Neither
   Prefill.run nor Verifier.run initializes a new session. Same-instance forced
   greedy controls can pass while starting from that same unproved state.
   NEW V5 source proposal now restores exact normal reset before binding.
   That does not prove the actual allocator returned nonzero bytes or that
   this caused token103; no such bytes were captured.

2. Reference execution/configuration is not matched beyond that reset. Normal
   has batch2, cache3, PIN1536, expert-cache auto and no original drafter.
   Current OFF has batch0/cache0/no PIN/fixed6500 plus same-resident original
   2,775,258,112-byte drafter. Materialized0 is the SAME, not a mode1 mismatch.
   Other differences include explicitly pinned CPU thread env and request/
   decode/observer flags. No native CPU expert execution is inferred from them.

Normal logical cuts are generated at generate10593-10626 from root/system
boundary, PIN, last-turn boundary and n-1. With this actual prompt they are
24/1536/2041/2047. generate10643 routes each part through read_part because
short-read=0; read_part10462 calls the same sp.run for an inactive paired
population. Each part restarts the local chunk count at512 and records its
checkpoint only AFTER Prefill.run completes. Thus the physical target chunk
shapes are 24; 512+512+488; 505; 6. New one-shot PP directly calls
first_prefill.run(prompt,2047,0) and actual stdout records512+512+512+511.
These are the same 2047 positions but different operator batch grouping.

Native HC normalization/projection remains the native F32/FMA/Q8 contract
(prefill2863-2878), with precision views at downstream boundaries2881 onward.
GDN qkv/z/out use native_proj and alpha/beta use bf16_proj2970-2978 with actual
T; QSA and expert batching also consume T. Different GEMM/batching shapes are
a possible arithmetic source, not an observed explanation of the output
change. Do not claim generic CUDA TF32 behavior from translated comments.

No missing generic post-prefill verifier commit was found: Prefill directly
updates carved recurrent state; intermediate stage dispatch4435 pipelines the
same positions, LAST callbacks4480 onward wait completed rows, and public
Prefill.run4680 drains the pipeline. It updates PLE previous IDs4506-4507.
The one-shot explicitly runs the final prompt row at2047, commits1 and waits;
normal also begins decode with p=n-1 and x=last prompt token10697-10713.

Normal batch admission then transfers all stage checkpoint/KV state into
BSlots (copy_to_slot8856 onward) and decodes with run_slot_rows3134; the
one-shot continues global ver.run2095. Both recurse to LAST stages, but that
is another actual dispatch difference requiring a matched target check, not
a guarantee of bitwise equivalence. Existing source57 matched history is
reference evidence, not automatically a proof for this changed one-shot path.

VERDICT -> First fix the explicit fresh-state gap, then rerun bounded control
and first-case against the same target. Keep profile/chunk/slot differences
separate. ON/OFF equality, if achieved, means lossless to THIS matched target;
it cannot establish equality to the earlier normal-serving reference. No
quantization/method/tolerance changes or performance promotion are justified.
