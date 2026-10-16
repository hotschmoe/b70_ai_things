# S143 default-OFF three-row QSA commit grouping (source only)

CONFIG -> S141 private passed only runtime n_steps3 (accepted prefix1/2/3),36 eager/captured A/B/A phases, full state/guards/finite/read-only, normal health. Production eligibility is therefore groupLen==3 ONLY. All other lengths preserve old loop. This does not generalize n_steps2/4/5/6 from accepted-prefix tests.

COMMAND -> External source candidate only; no live edit/apply/build/GPU. Root applies candidate.patch after complete source review and S142 closure. verify.cpp ONLY: process-constant const bool reads existing env_on("STRATA_QSA_COMMIT_GROUP_STEPS") before capture. Unset/'0' OFF; '1' ON. For eligible3rows, existing append_steps raw first row, commitpositions+2, stride1,n_steps3,base0. Tail restore remains BEFORE branch. All gamma/epsilon/buffers/shapes/capacity/RoPE/queue arguments unchanged. No device code or header/parser implementation changes.

RESULT -> Projected verify SHA c28de288061beae19949982ff3de7cf0a54fcae4d23f61e3a0374f90aeb435c6 from original e97ce93a3434fa1d4e9a3a3dce99a3fd41d86aac8662fbd3c959a3c78ce0d4b2. OFF else body is exact original after removing4 added indentation spaces. Full inverse patch restores original bytes. GDN,position records,PLE,KV,snapshots,queue/wait,eviction/restore contracts unchanged. Three-row rejected positions remain -1 and device loop skips them; complete stale pooled/dead/blockpos side effects retained.

HOST marker exact: strata QSA commit grouped steps: enabled=1 host-selection-only

Unique static identifier qsa_commit_group_steps_logged avoids the previously observed sibling logged-name collision. Emits once only at first selected three-row capture (startup allowed); OFF emits none. This proves host selection only, not per-layer/owner shapes, graph execution, or all GPU dispatch. Harness needs fresh sameNEWbinary OFF+ON, no S139/S142 crossbinarycontrol substitution; all other experimental flags0.

VERDICT -> Source ready for root review, main application/build PENDING. Existing configured Ninja target should require exactly verify.cpp.o,libstrata_engine.a,strata IF no other dependency changed; root actual dryplan must establish this, no assumed three-action PASS. No configure/kernel-archive/mainbackup/cachewarm/ISA certification. Existing stable offlinecache/precise/e41/BMG-G31/SG32 flags and native/HC/attention sources preserved. Expected build recipe separately supplied. Main numerical/output/work/capture/health remains to qualify actual integration. S141 private whole-chain timing is no serving forecast.
