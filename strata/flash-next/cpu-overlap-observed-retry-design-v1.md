# Same-recipe CPU continuation with passive swap observation

CONFIG -> Exact e633 continuation recipe and observer V3 0b4fa9e0, both source
identities retained. Original failed screen95049 and its raw evidence remain.
COMMAND -> bin/gpu-run python3 strata/flash-next/run_cpu_overlap_observed_retry_v1.py --output NEW_ROOT --expected-source-plan-sha256 REVIEWED_PLAN_SHA
RESULT -> Root3 CPU controls check exact original recipe/new output, all child
processes inherit exclusion fd8/9, and reviewed observer plan pin. Source only;
no actual idle/runtime observation or model retry has executed. Parent retains
one pair exclusion lease across idle30 and model/observer children. No devices
are granted, no model settings/guard changes. On screen terminal the passive
observer receives SIGTERM. The model screen retains Docker cleanup ownership;
parent waits for its exit before releasing exclusion. Both final readers must
pass before orchestration PASS, which does not establish causal attribution.
VERDICT -> Peer lifecycle/error review pending. Start only after the currently
live H46 native4 parent and its children are terminal. An observer timeout or
failure is preserved; it is not permission to weaken the strict host guard.
