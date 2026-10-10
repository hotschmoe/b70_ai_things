# Source37 fresh cacheOFF serial successor and first-job readonly recovery

CONFIG

Frozen harness40/source37 native2 paired ON selected four consumed-prefix jobs.
Its first serial job was captured completely, then the cacheON extractor rejected
committed_live(false,true,false). The actual source37 generate.cpp call uses
!cancelled for chain_updated and prompt_cache>0 && req_ckpt for live_reusable.
Thus complete fresh cacheOFF means a chain update with no reusable live chain.
The original failed parent and all raw records remain immutable.

COMMAND

CPU controls:
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s strata/flash-next -p test_batch_serial_source37_cpu_v1.py -q

Root-only readonly recovery, after source review:
python3 strata/flash-next/adjudicate_serial37_cacheoff_first_v1.py --output NEW_EXTERNAL_RECEIPT.json

Root-only future preparation uses batch_serial_source37_v1.py prepare with the
same genuine harness40 source37 preparation arguments, plus optional
--first-job-adjudication NEW_EXTERNAL_RECEIPT.json. Without that argument every
selected group job is freshly executed. With it, exactly the named first four-job
group can select indices 1,2,3 after recomputing the exact saved recovery proof.
Run qualify_batch_serial_source37_v1.py --plan NEW_PLAN --output NEW_RUN_ROOT.
The parent acquires both leases and preserves V40 strict health, original journal
rc/error/argv/logSHA, child interpreter/PID/EOF/log hashes, post-terminal full4,
and both exact third-shard page guards. No native math or SDK source changes.

RESULT

Twelve CPU controls pass using tiny mock records and an ordinary subprocess text
producer. No agent native/model execution. The new controller nests the complete
V40 source preparation unchanged, with a distinct controller and parent41.
The concrete serial driver uses absent-PIN fresh GEN1 and strict cacheOFF numeric
extraction. Actual selected indices/count/budgets are explicit separate fields;
the nested original full-group budget remains historical preparation provenance.
Report plan SHA and actual full49 comparison counts are produced explicitly.

Readonly recovery pins the exact original failed parent/result/raw/command/log
artifacts. It requires the unique cacheON-extractor error, parent failure, child1,
normal engine0/removal, no force/interruption, actual original health/journal and
new4/page closure. It compares every actual first-job head and layer row exactly.
It neither invents a missing child report nor overwrites parent failure: its
conservative completion boundary is the actual saved parent child-terminal epoch.
Its legacy PIN0 exception is explicitly historical and grants no absent-PIN,
complete-group, cache, full-model-math, quality, or latency qualification.

VERDICT

Source/CPU ready for independent review and root-only readonly recovery. Future
remaining-three execution is independently source/owned-lifecycle qualified.
A new final private reader must deliberately compose the saved first-job legacy
exception and the three actual absent-PIN jobs; frozen private reader V1 cannot
silently accept a new serial controller or an incomplete four-job group.
