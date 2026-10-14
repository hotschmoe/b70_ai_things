# Root-owned HC T2 paired original MTP compile

CONFIG -> Frozen paired compile plan8be12152 and actual prepared recipe
fa613f41. Same original SDK52, V6 fresh-reset/reference-prefix Main and
HC callerV2 candidate; no additional expert-pair optimization. NoDri2CPU6GiB,
card0FD8 lease. Root owns compiler/runtime; no author compile/GPU/model reads.

COMMAND -> Direct bounded compiler invocation from repository:

    bin/gpu-run --card 0 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/vm_8tb/github/b70_ai_things/strata/flash-next python3 strata/flash-next/build_hc_t2_paired_mtp_v1.py --compile-root /mnt/vm_8tb/b70/build/hc-t2-paired-mtp-compile-inputs-v1 --recipe-sha256 fa613f414ac1d6b96d8d576132537eb9d556db8cfafc7644346f01930ad2da2f

Output must remain the original EMPTY prepared directory. Preserve failed
attempts rather than overwrite. No nested GPU launcher and no --rm: direct
owned Docker foreground inheritsFD8; parent retains original process/lease
through exact terminal inspection/removal/absence/EOF, retries ambiguous
retirement, and latches SIGINT/SIGTERM/SIGHUP. Actual no-device, RO SDK/overlay,
2CPU6GiB/pids256/network-none and original label/image/command are checked.

RESULT -> Pending root actual compiler. Four fresh Main/verify/projection/
composition objects, exact frozen production flags/includes. Copies of TWO
archives remove exactly one verify.cpp.o and exactly one each HC projection/
composition member; before/after entire ordered member rosters are checked.
Four new objects precede archive lookup in the original link recipe. Original
SDK/inputs/archives/includes/compiler/packages are hashed before/after; image
compiler/version and package census must match original SDK recorded bytes.
Receipt binds all4objects, both filtered archives, new paired ELF, raw stdout/
stderr/command/source/normal ownership closure. Existing operator PASS is not
represented as a model pass. One targeted CPU check exercises actual recipe,
both exact archive filters and new-object-first linkage, no broad test suite.

VERDICT -> Source ready only. Actual new compiler receipt/ELF and subsequent
paired model guards are mandatory; no source52/previous MTP runtime transfer.
