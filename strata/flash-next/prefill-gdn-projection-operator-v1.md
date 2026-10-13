CONFIG

Original layer0 Q8_0 QKV [10240,2560], Z [6144,2560], output [2560,6144], original F32 alpha/beta narrowed by the loader's BF16 RNE, original F32 dt/ssm_a. Source62 receipt 613694ec6fda4b0136a090406ec5f8c40070e31d967a9af726d308bab4ca48db. Candidate helper from source plan 8f54b6369742f7e87eb4e67e708cb2196567a5c1bce92052791efb130b51d59e. Synthetic F32 operands independent of native captures. T=1,7,8,9; candidate chunks at8, decoder one-row baseline. Original default exact MMVQ is required, never set by fixture.

Five complete F32 outputs QKV/Z/decay/beta/block must match bitwise. The actual final-row input and output Q81 scratch packets must match. Two graphs replay both original and changed owner inputs. Packed weights remain byte-identical and all allocations have canaries. Exact source-derived post false/true variants preserve half bytes and verify F32 publication; they do not qualify conv or recurrent arithmetic. GPU drain precedes graph deletion and buffer freeing; failed drain retains resources and exits86. Root uses card0/full DRI/mask0, expected PCI0000:0b:00.0.

COMMAND

Run the export only after clean timing ends, with a fresh identity supplied by root. It reads original packed weights, never native projection inputs, then performs one mandatory fresh closing full4 scan. Replace SHA placeholders with hashes of actual named receipts, not guessed values.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/export_prefill_gdn_original_weights_v1.py --model-identity FRESH_FULL4.json --identity-sha256 ACTUAL_SHA --output /mnt/vm_8tb/b70/build/prefill-gdn-original-weights-v1
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/prepare_prefill_gdn_projection_operator_v1.py --output /mnt/vm_8tb/b70/build/prefill-gdn-projection-operator-inputs-v1
    bin/gpu-run --card 0 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/run_prefill_gdn_projection_operator_v1.py build --prepared /mnt/vm_8tb/b70/build/prefill-gdn-projection-operator-inputs-v1 --recipe-sha256 ACTUAL_RECIPE_SHA --output /mnt/vm_8tb/b70/build/prefill-gdn-projection-operator-build-v1
    bin/gpu-run --card 0 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 strata/flash-next/run_prefill_gdn_projection_operator_v1.py runtime --build /mnt/vm_8tb/b70/build/prefill-gdn-projection-operator-build-v1 --build-receipt-sha256 ACTUAL_BUILD_REPORT_SHA --elf-sha256 ACTUAL_ELF_SHA --inputs /mnt/vm_8tb/b70/build/prefill-gdn-original-weights-v1 --inputs-report-sha256 ACTUAL_EXPORT_REPORT_SHA --expected-pci 0000:0b:00.0 --output /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/prefill-gdn-projection-operator-v1

Compiler argv is reconstructed from the actual source62 verifier Ninja command: O3/C++20/precise, original subgroup32/per-kernel flags, correct divide/sqrt at device link. It uses unchanged original libstrata_kernels/core archives, actual image/compiler/package bytes, NoDri2CPU4GiB, root inherited FD8 and normal owned EOF/removal. Runtime has per-card pre/post health, kernel fault predicate, source/input pre/post hashes and typed owned terminal. No engine rebuild, no weights mount at compilation, no API/serving authority.

RESULT

Source construction and three tiny CPU controls only. No compiler, weight export or GPU execution by author. Actual compiled/runtime receipts are pending root execution.

VERDICT

Ready source candidate for projection/post-store operator parity. This is not fullGDN/full48 fidelity, throughput qualification or a2000token/s claim. Candidate wholeT recurrence and conv remain unchanged but must be validated separately with an actual model capture. The old AB observer reads m.ab; candidate fused AB does not produce that buffer, so old AB observations cannot be reused as candidate evidence.
