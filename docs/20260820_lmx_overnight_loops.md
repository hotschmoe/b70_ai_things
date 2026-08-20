# LocalMaxxing overnight -- loop ledger

Standing feedback for `docs/20260820_lmx_overnight_plan.md`.
Newest loop at the **bottom**. Do not rewrite old loops.

Runtime status: `/mnt/vm_8tb/b70/lmx_overnight/STATUS` (not git).

---

## NEXT PICK (keep this line true)

W1 -- 3.8 GPTQ-Int4 MTP4 + cookbook 2026.08.19
draft-INT4 on f01e24f6, 1x B70, p512/g128 n=5
vs S1 47.58. Sweep:
`bash vllm/cookbook_campaign/sweep_lmx_w1_draftint4.sh`

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
