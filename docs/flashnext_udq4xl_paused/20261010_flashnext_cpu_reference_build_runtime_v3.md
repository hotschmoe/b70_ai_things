# Flash-Next actual CPU reference build/runtime evidence


## 2026-10-10 - Actual V3 CPU build and intended-runtime checks

CONFIG -> Exact verified3579-file V2 source snapshot, V3 configure recipe,
pinned compiler/runtime image39992d70, CPU-only backend, two cores/8GiB,
networknone; no model mount or device grants. All six declared targets built.

COMMAND -> Root64755 configure/build ends exit0 with normal owned removal.
Its post-build host ldd reader then fails because host libgomp.so.1 is absent.
Preserve original receipt. NEW all-six ELF closure35997 checks the actual pinned
CPU container dependencies, exact CMake cache and source pre/post bindings.
NEW smoke14260 executes server/completion --version and test-quantize-fns in
same intended runtime with clean environment, 2GiB cap and no model/devices.

RESULT -> Actual compile/link succeeded. Original parent remains FAILED with
errors ['AssertionError: '] and empty ELFs; it must never be called a passing
whole-build receipt. Host ldd also omitted server from its loop, so the NEW
closure explicitly derives all six targets from --target, not a numeric slice.
Supplemental closure PASS all6, including libgomp resolved inside pinnedimage;
all exact OFF/ON CMake values, source completeness, ELF/dependency bytes and
normal terminal/removal confirmed. NEW sidecar pins original failed receipt,
missing-libgomp ldd file, exact bound logs and supplemental receipt. CPU runtime
smoke PASS all three commands, quantization tests exit0, no GPU/model execution.
Preserved root source scripts live alongside receipts for replay/inspection.

Receipt directory:
`/mnt/vm_8tb/b70/build/flashnext-cpu-build-v3-20261010/`

- Original failed receipt.json SHA256:
  2df28740819c1b0869395fb8ff52d000cf167e4993b434a3e76f5fc48a5969ac.
- NEW elf-closure-v1.json SHA256:
  26e460378f77e5e6ae0e76d1293d1694991a491483cf628c9d37f096a9f6bfb6.
- NEW host-loader-failure-binding-v1.json SHA256:
  e2126b4e67bb51548f5b1647bb1d08e9b50ba139146de6bebaa0d821b7f8b18e.
- NEW cpu-runtime-smoke-v1/receipt.json SHA256:
  2cead24f1c108f07fb36c9feefb4c06aae2ecd1f5ecd3e2a137f74433a981937.

VERDICT -> Fresh CPU executables and intended runtime are now usable for a
coordinator-owned exclusive-RAM functional pilot. A failed historical host
loader reader does not require recompilation when the intended runtime and
all source/ABI identities are independently established. No model inference,
quality, whole-model math, native bitwise authority or speed proof yet. Pilot
must admit ONLY this exact closed failure plus passing all6/runtime/source
bindings; arbitrary failed builds remain rejected. No stack/package changes.
