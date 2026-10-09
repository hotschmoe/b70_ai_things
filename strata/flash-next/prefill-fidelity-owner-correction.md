# Prefill fidelity owner correction0017

CONFIG -> frozen full-engine v2 source plus0017, pinned compiler image,
actual SYCL include order, no devices exposed. Existing0013..0016 patches,
v1/v2 plans, failed source tree and failed build receipt remain unchanged.

COMMAND -> python3 strata/flash-next/compile_prefill_fidelity_owner.py
The controller acquires the pair lease, copies failed source into a new tree,
checks/applies0017, then independently compiles actual prefill.cpp and verify.cpp.
It mounts source read-only and exposes no GPU devices. It records complete
commands, source/object hashes and consumed header hashes.

RESULT -> both translation units exit0: prefill31.930seconds and
verify31.831seconds. Receipt:
/mnt/vm_8tb/b70/build/strata-prefill-owner0017-4yog27pw/compile-receipt.json.
The explicit structural check confirms no observer fields in PeerPrefill and
all three observer fields in Prefill::Impl following its Gemm and owned fields.

VERDICT -> PASS for these actual object compilations. Full linking, GPU
execution, teardown and paired same-version diagnostic-off/on equivalence
remain required. The earlier helper-only tests and generate.cpp compilation
failed to detect0013 placing the fields in the first owned-vector owner.

Patch0017 only relocates the declarations; it adds no math/reset/cache option
and leaves observer default-off behavior as defined by0013. Its SHA256 is
f92143d988517cc32ac60e32d864d9816186d82a46515a25bf3455d3822ba6b8.
New prefix-full-state-engine-build-plan-v3.json appends0017 to pristine v2
SHA827f3631b3feb857d6a91d76cab060c5b373832934b17e82dce0cc349e124639.
The v3 SHA256 is
7f2751dee26dd7fa2d1f3c1ebd4ba12e5069367482a1d346ffabd43fa86a56e3.

Use the existing build_native_hc_engine.py --plan workflow with this new v3
plan for the next full build in a fresh source/output tree. The local
prefill-fidelity-owner-correction-receipt.json retains the full CPU compiler
receipt and independent declaration-owner check.
