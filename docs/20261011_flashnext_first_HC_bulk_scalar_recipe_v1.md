# Bulk first-HC versus executed scalar bytes

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Scalar66797 reporte62ef355 is preserved. Recompute original embedding
and firstHC with explicit CPU ELF83c289, then require all SIX F32 fields and
packet byte-identical to that scalar result. No captured scalar/native operand
enters compute; saved arrays are comparison outputs only. No full48 run.

COMMAND: Root waits for current47 model terminal, supplies a fresh passing
complete four-shard identity receipt, then runs:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/compare_original_first_hc_bulk_v1.py --actor-root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source44-num38-v2-run-v1 --model-identity ACTUAL_FRESH_FULL4_JSON --expected-identity-sha256 ACTUAL_FRESH_IDENTITY_SHA --bulk-elf /mnt/vm_8tb/b70/build/hc-rows-dot32-cpu-v1/hc-rows.so --expected-bulk-sha256 83c289ccd11ef6fed8cb9c0563328795f1fbd665110edace7efcfb4fc710fb47 --scalar-root /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/original-first-HC-F32-NUM38-v1 --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/original-first-HC-bulk-vs-scalar-v1
```

ELF hash/ABI and exact saved scalar report/field hashes check before original
payload work. The unchanged firstHC controller calls OriginalTensorRows and
OriginalStorageProjector, consumes historical NUM38 targets without recursive
current admission, and owns ONE fresh postCPU full4/pages close in finally.
Mismatch or failed computation still closes and retains failure. New private
compute namespace replaces exactly the sole HC-read call with the opt-in bulk
backend. Scalar source/e62e evidence and all other arithmetic stay unchanged.

RESULT: Two tiny controls PASS0.001: actual CLI argument dispatch without a
provider, and byte match/mutation refusal. Actual original bulk compute pending.
No model/provider/GPU work by the author; no elapsed speedup claim.

VERDICT: Only after actual seven-output bitwise success, proceed to a new
original full48 own-state candidate using the same explicit rows backend.
The existing OwnedOriginalNativeHcBulkF32 exposes this opt-in compute adapter,
but its RouteAware base still guards old source35 three-CPP hashes and cache3
schedule. Those must be deliberately associated with the actual current source
and requested config, not bypassed or presented as current47 proof. A bounded
prefix1 has only verifier rows; own zero recurrent/QSA/PLE state and all48
primary original operands must remain. Current mode1 indexer storage and other
F32/integer-dot/nonlinear frontiers need honest source routing. Do not reuse
device-RS/captured-input diagnostic lanes as the primary reference. First prove
the fast HC transport; then seal the smallest actual full48 source port.
Host sqrtf/expf and device mix contraction remain unresolved, and neither CPU
transport parity nor head NMSE establishes full-model math qualification.
