# Q1 decode selector serving diagnostic

Default B70_Q1_SELECTOR=native leaves the QSA selector untouched. The
explicit fused mode selects the measured q1_fused_r2 component through
q1_selector_api.py in the API process and spawned workers. This wrapper
imports only the standard library before worker setup. Compilation mode0
and FULL_DECODE_ONLY are the measured baseline; Dynamo is not qualified.

Only QsaRunner._decode authorizes the overlay, and only its exact
persistent QsaBatch scores/ids/step buffers qualify. Prefill explicitly
clears that scope, including tiny tails that call run_decode. Each live
batch owns one selector/workspace, warmed before graph capture. Changed
buffers fail closed; no global cache retains transient prefill tensors.

The finite window preserves the usual GPU leases, native pre/post health,
identity, arithmetic, short decode and fixed4K/16K/32K C1/C2 reference
probes. Actual server configuration, sources, SOs, fixture IDs and native
engine teardown are recorded. The180s output budget applies to a whole
fixture phase, whose historical native32K duration was about139s. It is
a finite failure bound, not proof of a deadlock.

selector-outcome.json separates operational closure from strict repeat
and Strata text-reference gates. The native controls already fail those
strict gates; operational success does not qualify fidelity or stability.
No shelf promotion or external submission is part of these windows.

Run each arm only under root coordination, with both cards reserved:

    B70_Q1_SELECTOR=native B70_Q1_FUSED_SO=/absolute/measured/libq1_fused_r2.so TAG=fresh-native bash vllm/flashnext/harness/q1_selector_window.sh
    B70_Q1_SELECTOR=fused B70_Q1_FUSED_SO=/absolute/measured/libq1_fused_r2.so TAG=fresh-fused bash vllm/flashnext/harness/q1_selector_window.sh

Both matched arms must use the same sources, library, fixtures and runtime
configuration. Freeze campaign harness/plugins/registry until both close.
Keep Strata DD down between tests, as directed by the user.
