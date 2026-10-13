CONFIG

Runtime-only successor to the failed V1 startup. Immutable original export ed27d9f88d436ca3686a313cb2e741bcb201f23c265983c02af24a1b88db4ad2, NoDri build e827a0e99fc3c35b1aee711e1b2611b3d26b33b4fb7bf4c64eb5cdaa0707ac35, ELF 6dfc0d1d0d43b51ecb8f6841dda29eb8feae8bf6ccfb2032c929d0025ed98543. Only startup setvars guard changes; all fixture arithmetic, original packed bytes, owner/health/kernel/source and output gates remain unchanged. Build source joins exact original V1 plan, new runtime source is independently pinned.

COMMAND

    bin/gpu-run --card 0 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/run_prefill_gdn_projection_operator_v2.py runtime --build /mnt/vm_8tb/b70/build/prefill-gdn-projection-operator-build-v1 --build-receipt-sha256 e827a0e99fc3c35b1aee711e1b2611b3d26b33b4fb7bf4c64eb5cdaa0707ac35 --elf-sha256 6dfc0d1d0d43b51ecb8f6841dda29eb8feae8bf6ccfb2032c929d0025ed98543 --inputs /mnt/vm_8tb/b70/build/prefill-gdn-original-weights-v1 --inputs-report-sha256 ed27d9f88d436ca3686a313cb2e741bcb201f23c265983c02af24a1b88db4ad2 --expected-pci 0000:0b:00.0 --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/prefill-gdn-projection-operator-v2

RESULT

V1 runtime85835 failed before binary execution: empty stdout/stderr, setvars already initialized returned3 under set-e; failed report2188cc70a2baca6f379cd944e9cf88740942bbf87cac191d24656504baf074e1 remains unchanged. Root reported health/kernel normal. Two targeted CPU tests passed: actual emitted shell skips initialized setter returning3; an uninitialized setter failure remains fatal. Exact undo verifies only guard/runtime version/build-plan association changes.

VERDICT

Ready runtime retry source. No rebuild, export or whole-model hash rerun. No GPU execution by author; operator fidelity/performance remain pending actual execution.
