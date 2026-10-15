# S65 exact IQ3_S C6 SWAR primitive record

Recorded experiment sources, not a new serving or general benchmark interface.
Root reviewed exact source and actual Opus/Fable source reviews before build.
The private candidate changes only packed sign reconstruction; original weight
load expressions and arithmetic/geometry remain. Main Strata source unchanged.

Raw preparation/build/runtime evidence:
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/iq3s-dense-oracle-s65-prep/
Actual execution controller and root reports: sibling iq3s-dense-s65-execution/.

Recorded commands: original preparation build.sh once through bin/cpu-run;
actual execution iq3s-dense-s65-execution/ROOT_RUN.sh once. Original preparation
ROOT_RUN.sh/leased_run.sh were NOT run; recorded versions here are corrected
execution controllers. Their absolute original paths, SHA pins, unique Docker
name and refuse-overwrite output directory are intentional historical bindings.
Do not run these copies against existing result directories.

The build reuses S60's compiler-produced common image wrapper and frozen host
archives; the adapter has its own normal SYCL compilation/device prelink before
ordinary host linking. No custom registration or binary restoration.

One card0 quiet window: exact public/private outputs, graph replays, guards,
immutable buffers, actual DOT bindings, health and final saved bytes pass.
Gate projection improves about2.5percent; ssm_out regresses about1.3percent,
eight balanced nested pairs each in one process. Projection only, not serving.
Keep global SWAR unintegrated; no immediate server test or duplicate run.
See docs/20261015_strata_iq3s_dense_swar_probe.md from the repository root.
