# S94 complete-resident caller-domain correction (source only)

CONFIG -> S93 same551cca binary flag0/1, QSA0/Q6center0, INT8/two200k,
batch2. ON three4K requests and first32K match output/work but selection
marker is absent; root stopped remaining requests and closed normal teardown/
post-health. Preserve partial INACTIVE evidence; no candidate speed summary.

COMMAND -> Source/config/identity proof, not a new runtime field readback:
- generate.cpp472 defaults kv_resident0; actual saved argv has NO
  --kv-resident. generate2351 passes0 to qsa_set_kv_resident.
- layer.cpp718..727 defaults KvPlan.mode0, slots=pages, then returns when
  g_kv_resident<=0. state_init793..794 assigns st.kv_mode/st.n_slots;
  st.n_pages850 is pages. --kv int8 sets st.kv_int8. Therefore staged=
  (st.kv_mode==1) is FALSE in the real S93 configuration: the guard is
  sufficient to explain no marker. Missing KV-stream log is corroboration,
  not the proof. Existing prompt-XMX defaultsOFF and S90 actually recorded
  12264 full32 fallback calls, establishing source-path reachability.
- mode0 layer966..971 creates table[i]=i, uploads and WAITS. QsaState owns
  the table/pools for the session lifetime. qsa_state_zero resets mapping
  only in mode1; qsa_kv_resolve1033 returns for mode!=1. No mode0 eviction/
  ring remapping occurs. conversation_kv_restore copies authoritative arrays;
  only mode1 reset/mode2 ring restoration affect mappings (217..234), so
  mode0 identity remains. This is source evidence, not a new cache test. Nonelastic slots>=pages imply physically allocated
  capacity for the complete mapping. mode0 virtual elastic capacity alone
  is insufficient: new guard explicitly excludes kv_elastic>=0. Actual
  batch2 also makes the generate3935 kv-grow enabling predicate false.
- qsa_attn_pools1021..1028 passes state.page_table and its INT8 K/V/scales.
  prefill3301 selects these DIRECT pools for mode0. Existing append3126 on
  m.cs publishes this chunk before attention; past valid history is the
  same state the old attention path consumed. Top-k/cell bounds and input
  validity remain existing contracts, not a new arbitrary-memory promise.

RESULT -> One caller-only patch replaces staged eligibility with staged OR
(mode0 && nonelastic && n_pages>0 && n_slots>=n_pages). Existing INT8,
nb32,cap2051, first-host-width2051 guards remain; monotonic width proves all
32 are full. Mode2 rings and partial/virtual resident pools stay excluded.
No GPU table scan, new copy/event/wait/allocation or altered pool pointer.
Same original call offsets/stride, wrapper, marker, template, kernel tags,
attention object and archive, math and SLM as551cca. OFF selected wrapper
still returns directly to unchanged old API; only a CPU eligibility test
changes. Existing header's staged-only label is overnarrow historical S93
comment; leave it unchanged to avoid a pointless attention-TU rebuild. The
API receives the same complete identity-mapped, published pool contract.

VERDICT -> SOURCE PROPOSAL ONLY, no apply/build/GPU/models. Identity mapping
is source/config-derived, not newly read back; no general nonidentity or
unfilled-pool validity claim. Current fullresident serving is now eligible
by source, but future actual ON marker and output/work/health gates still
required. Future build should touch ONLY prefill.cpp, libstrata_prefill.a
and mainAOT/link; keep attention member/archive exact and reuse its existing
binding/ISA evidence, verifying relink binding as appropriate. No new
primitive matrix or repeated cap/36case campaign. New full main executable
is not the old551cca GPU binary; compare only matched same-new-binary arms.
