# CPU swap attribution observer V2 closure

CONFIG

NEW successor to source-only V1/401996; V1 has never executed and stays intact.
All unchanged e633 CPU screen settings/guards and the preserved 95049 failure
remain exact. V1 design describes measurement fields, bounds and limitations.

V2 additionally stores the exact original idle receipt path/SHA. The read-only
runtime consumer reexecutes that entire idle raw/source closure and checks its
actual finish before runtime start with the original300-second freshness bound.
The producer also rehashes the complete small source closure before terminal
export. No cached digest, global guard relaxation or inference change occurs.

COMMAND

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_cpu_swap_attribution_cpu_v2

Root uses observe_cpu_swap_attribution_v2.py instead of V1 for both idle30 and
runtime modes. All arguments and exclusion-lease rules from V1 design apply.
Use NEW separate output trees; do not rewrite earlier failed or source evidence.
Read-only admission: observe_cpu_swap_attribution_v2.finalized_binding(ROOT).

RESULT

11 tiny CPU controls pass, including original10 and explicit successor idle/
post-source contract control. Actual host sampling/Docker/model work is absent.
V1 producer admitted the full idle tree at startup; its read-only runtime
consumer lacked that explicit path association. V2 closes the saved association
and full producer post-source check before any actual observer execution.

VERDICT

Source-only ready for independent review, not actual observation qualification.
Passed diagnostic closure does not establish swap causality, whole16-case
coverage, successful screen inference, cache/math/quality or performance.
