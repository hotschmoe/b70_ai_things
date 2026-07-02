# Fallback: per-node consume-ack (only if full-capture serves but garbles under CONCURRENT load)

Codex flagged a cross-forward payload-lifetime race: a fast rank could reuse (overwrite) a payload
slot in forward K+1 before the slow rank's forward-K reduce consumed it. My analysis (2026-07-02):
the autoregressive loop + in-order per-rank queue + per-node AR coupling make it effectively
impossible in decode -- rank B's reduce[node] follows its own push[node] immediately on its in-order
queue (only a spin between), and rank A cannot complete a full forward + sample (to reach forward
K+1 push[node]) in that window, because rank A's forward-K completion itself depends on rank B's
forward-K push[node]. The single-stream serve + tightly-barriered microbench both stay coherent.

IF concurrent/soak load ever shows intermittent "!" garbage (not the fixed within-graph aliasing),
apply this per-node consume-ack. Fully device-side, replay-safe:

In do_ar_graph_spin, add a consume page and gate the PUSH on the peer's previous reduce:
- Layout: my_flags (page A), my_counts (page B: push/expect), my_consumed (page C: consume acks).
  seq region grows to 384KB; seq_base carves the last 384KB.
- reduce kernel epilogue (after reading local slot): c = my_consume_ct[node]+1; my_consume_ct[node]=c;
  store c -> peer_consumed[node] (release, system scope).  // "I consumed my slot[node] use #c"
- push kernel PROLOGUE (before copying to peer slot): want = my_push_ct[node] (the count this push
  will become minus 1 = previous use); acquire-spin my_consumed[node] >= want.  // peer freed the slot
  First use: my_consumed[node]==0, want==0 -> no wait.

Cost: doubles the tiny single_task flag traffic; adds a wait in the push path that is already
satisfied in the common case (only blocks under the pathological lap). No host round-trips.
