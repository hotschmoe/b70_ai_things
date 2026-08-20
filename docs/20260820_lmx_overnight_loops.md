# LocalMaxxing overnight -- loop ledger

Standing feedback for `docs/20260820_lmx_overnight_plan.md`.
Newest loop at the **bottom**. Do not rewrite old loops.

Runtime status: `/mnt/vm_8tb/b70/lmx_overnight/STATUS` (not git).

---

## NEXT PICK (keep this line true)

P1 when Q8_0 GGUF >= 25000 MB:
`bash llamacpp/sweep_obliterated_q8.sh`
W1 is DONE (65.08, G1 GO). If P1 still
blocked, O2 Ornith GRAPH-safe INT4 dtype
(L65 Half!=BF16). Fetch:
/mnt/vm_8tb/b70/lmx_overnight/p1_hf_q8.log

---

## LOOP 0 -- 2026-08-20T0705Z -- contract + 2026.08.19 patch wire, no GPU

Picked: author overnight plan/ledger/dead-ends; copy cookbook
  2026.08.19 patches into `vllm/patches/cookbook/`; apply them
  from `apply_mtp_patches.py`; `DRAFT_INT4=1` on launch.sh;
  add `sweep_lmx_w1_draftint4.sh`. Start 30m durable scheduler.
Why this, not the other open row: LocalMaxxing review said
  draft-INT4 is the 27B W4A16 unlock (83.7 -> 112.7) and S1 on
  this box was 47.58 without that overlay. Ckpt and f01e24f6
  image are already local. Cards free.
GPU: none this fire (parent). Lease free both cards. DD PARKED.
Command: docs + patch copy + launch/apply edits.
Log: n/a
Result: contract LOOPING. Next pick W1 GPU.
Verdict: GO (plan + protocol). No number moved.
Changed beliefs: 30m is a check-in. W1 serve may span two fires.
  Do not start a second serve. Hold Ornith 34.9 and k1bar 31.9
  until something beats them on their own metric.
Next pick: W1 sweep on card 0.
Do not: start DD; P2P; emul NVFP4; demote W8A8; 4x B70; kill a
  healthy W1 serve at 30m.
Restore: DD PARKED. Cards free for W1.
JOURNAL: ### 2026-08-20au

## LOOP 0b -- 2026-08-20T0740Z -- add Pliny OBLITERATED Q8_0 to the loop

Picked: add P1. Fetch Q8_0 GGUF only (~29 GB). Write SYCL
  serve + sweep. Q8 fused doors ON, Q4K OFF. Manifest id
  qwen3.8-27b/obliterated-q8. No GPU serve until file complete.
Why this, not the other open row: operator asked to add and
  optimize the Pliny Q8 that just landed. W1 stays in queue.
GPU: none (fetch is CPU). Cards still free. DD PARKED.
Command:
  hf download OBLITERATUS/Qwen3.8-27B-OBLITERATED \
    Qwen3.8-27B-OBLITERATED-Q8_0.gguf \
    --local-dir models/files/qwen3.8-27b/obliterated-q8
  llamacpp/serve_qwen38_obliterated_q8.sh
  llamacpp/sweep_obliterated_q8.sh
Log: /mnt/vm_8tb/b70/lmx_overnight/p1_hf_q8.log
Result: download started. Serve scripts landed. Next GPU
  is P1 when GGUF >= 25 GB, else W1.
Verdict: GO (added). No tok/s yet.
Changed beliefs: Q8_0 is weight-only (~W8A16), not our
  W8A8-INT8 XMX path. Historical B70 Q8_0 was slow vs Q4_K_M.
  Profile Q8_DOORS before writing new GEMV.
Next pick: P1 sweep if GGUF complete, else W1.
Do not: fetch whole OBLITERATUS repo; vLLM P2P; demote
  31.9/34.9; start DD; 4x B70.
Restore: DD PARKED.
JOURNAL: ### 2026-08-20av

## LOOP 1 -- 2026-08-20T0715Z -- W1 draft-INT4 G1 GO 65.08

Picked: W1 3.8 GPTQ-Int4 MTP4 + cookbook 2026.08.19
  draft-INT4 on f01e24f6, 1x B70, G1 + phase_bench
  p512/g128 n=5 vs S1 47.58.
Why this, not the other open row: P1 GGUF was
  incomplete (~15.6/29 GB). W1 was the first
  unblocked GPU row; ckpt and image local.
GPU: card 0 via sweep gpu-run. Card 1 free.
  Container lmx_w1_d38. DD PARKED. P2P off.
Command:
  bash vllm/cookbook_campaign/sweep_lmx_w1_draftint4.sh
Log: /mnt/vm_8tb/b70/lmx_overnight/w1_serve_20260820T0709Z.log
  G1 /mnt/vm_8tb/b70/lmx_overnight/w1_g1_20260820T0709Z.log
  bench /mnt/vm_8tb/b70/lmx_overnight/w1_phase_20260820T0709Z.json
Result: patches applied. MTP 0.85->0.22 GB, LM
  head 2.54->0.66 GB. Graph 5s. G1 Paris/391.
  median post_first **65.08** (52.84-81.91)
  vs S1 47.58. Spec 546/984, mean accept 2.22.
  No DEVICE_LOST. Serve stopped after bench.
Verdict: GO. New W1 hold 65.08. Not W8A8.
  Board 112.65 still ceiling.
Changed beliefs: dense 3.8 GPTQ + draft-INT4
  GRAPH-boots on float16 (unlike Ornith L65
  Half!=BF16). Overlay is a real decode lift
  on this box (+36.8% vs S1), still short of
  SergiioB 112.7.
Next pick: P1 if GGUF >=25 GB else O2.
Do not: publish 65.08 as W8A8; demote 31.9
  or 34.9; start DD; P2P; emul NVFP4; 4x B70.
Restore: DD PARKED. Cards free. W1 container
  removed.
JOURNAL: ### 2026-08-20aw
