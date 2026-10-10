# V2 HC projection canonical-count admission

CONFIG -> unchanged frozen CPPv1 f53ce82ca026bf9326ffe3ef001e908cf1971a6db506254e584f634938d8cdb6
and unchanged synthetic operands/public HC kernels. New V2 fixture producer,
collector,tests and plan; all V1 source/plan/runtime evidence preserved.

COMMAND -> PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s
strata/flash-next -p 'test_hc_projection_arithmetic35_fixture_cpu_v2.py'.

RESULT -> 14 CPU tests PASS. Canonical geometry independently computes analytic
output12 plus heldout output24 =36 F32 words across13 cases and2 negatives.
V1 metadata incorrectly said38. Root reported all13 V1 GPU cases bitwise equal
and2 effective negatives; the V1 parent must remain failed on its wrong count,
with raw output/health/source evidence retained. This source agent performed
no actual hosthelper/compile/GPU/Docker/model payload/runtime/git execution.

VERDICT -> frozen V2 source ready; no arithmetic change or old proof transfer.
corpus_counts() derives counts from canonical fixture M*T, never a mocked target.
source_binding() requires plan cases/output/negative counts match that computation
before host admission/export or GPU execution. input_binding() requires manifest
computed counts and canonical raw operands; collect() additionally requires
actual sum(words)==computed output count. Unit test changes plan count to38
and proves failclosed before any host/helper export. WholeT2 shape/raw and all
existing receipt/library/source/negative/mismatch gates remain unchanged.

ROOT-only API names are unchanged in hc_projection_arithmetic35_fixture_v2.py:
prepare(output,host_run,helper,build_root),input_binding(package),collect(package,
raw,log). Use NEW packages/compile/parent outputs. Source33review supplies NEWV2
metadata/compile/runtime wrappers and plans; shared CPP remainsV1. Do not rewrite
or finalize V1 receipts. Expected data stays outside GPU input-only mount.
Actual leaf/lifecycle/intrinsic/model qualification remains false in this source
plan; future root actual evidence is separate. No tolerance relaxation, fullHC,
exp/rsqrt,original fidelity,concurrency or speed claim is introduced.
