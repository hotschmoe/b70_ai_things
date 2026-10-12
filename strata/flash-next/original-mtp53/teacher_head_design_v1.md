# Teacher-prefill-only original head skip

CONFIG -> Frozen genuine paired V6 original MTP natural P2048/GEN512 controller.
New default-OFF STRATA_ORIGINAL_MTP53_TEACHER_HEAD_SKIP=0/1 only. Original34 weight
reserve, allocation, original quantization, target model and decode are unchanged.

COMMAND -> Root overlays two new versioned headers into the actual V6 recipe.
Each teacher row still prepares, records original EH/HC/QSA/FFN/next_hidden,
commits both banks and hidden, and drains exactly as before. Only final HC head
read and original own Q6 vocabulary projection are omitted on 2047 teacher rows.
Final MAIN prompt T1 run and advance_actual remain unchanged and compute a fresh
original head before the first proposal. All decode catchup calls retain the
record default compute_head=true. Existing final target/queue shutdown is intact.

RESULT -> Head scratch/logits are separate from next_hidden and all bank state.
The next teacher front consumes actual MAIN hidden and known next prompt token;
neither logits nor argmax is used. Head-ready is cleared before every record and
first proposal refuses stale logits. Host-only counters report teacher rows,
actual queued head calls/skips, committed cells and freshness before first argmax.
Expected OFF:2047 teacher/2048 calls/0 skips; ON:2047 teacher/1 call/2047 skips;
committed cells2048 in both. Successful counters require existing drains/Exit0.

VERDICT -> Source-only candidate. Root must freshly compile and compare complete
512 output histories, positions/acceptance, same residency, identity, health and
normal retirement, then measure matched prompt latency. No numeric/speed claim.
No per-layer observer, new workspace, query, barrier or decode shortcut is added.
