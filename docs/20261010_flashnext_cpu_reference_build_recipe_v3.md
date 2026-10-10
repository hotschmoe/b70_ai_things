# CPU reference build recipe V3

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: V3 changes only the configure recipe from V2. Remove
FETCHCONTENT_FULLY_DISCONNECTED and FETCHCONTENT_UPDATES_DISCONNECTED: with
all fetch features disabled, this source does not consume either variable.
V2 attempt2 correctly stopped before compilation on the unused-option guard,
exit65, owned container removed. Its receipt/logs remain historical at
/mnt/vm_8tb/b70/build/flashnext-cpu-build-v2-attempt2-20261010.
Do not relabel V1/V2 preparation or configuration failures as V3 evidence.

No producer, source corpus, identity, reference arithmetic, pilot or tolerance
change. Reuse the actual V2 snapshot only with its original source-receipt.json,
HEAD/index/dirty-patch identity and complete path/hash/mode manifest. V3 pins
the same source corpus3579files /98156639bytes, manifest SHA
b216da5229b0a38268cb2f115bec941ae8e486579a95ef18e825bf7e591cf1d1,
and the existing V2 producer18810ac7621134e1b303cd8b506229c4bd3cc061e3bf2ef08355f96f025d9c1d.
The original source receipt remains bound to V2; a NEW build receipt must bind
that original receipt SHA plus the V3 recipe SHA. Never rewrite the snapshot
receipt's plan. The209 essential architecture/CMake files remain mandatory.

COMMAND (ROOT, immediately before NEW build configure):

```
PYTHONDONTWRITEBYTECODE=1 python3 llamacpp/flash-next/prepare_cpu_reference_source_v2.py \
  --plan llamacpp/flash-next/cpu-reference-source-build-plan-v3.json \
  --check-snapshot <actual-verified-V2-snapshot-source-directory>
```

Use V3 JSON configure/build argv in a NEW owned output; never reuse the failed
CMakeCache. All V2 container controls remain: image39992, /usr/bin/env explicit
entrypoint and -i clean argv, GCC/G++, network none, no GPU devices/socket/group
or model mount, parallel2, memory8g/swap8g, cpus2, pid256, empty cwd, no install.
Actual fetch controls LLAMA_LLGUIDANCE=OFF, GGML_OPENMP_FETCH=OFF,
LLAMA_BUILD_UI=OFF and LLAMA_USE_PREBUILT_UI=OFF remain. No dependency download
or package installation is allowed; missing local dependencies fail.
All GPU/RPC/DL/repack/BLAS/LLAMAFILE exclusions remain unchanged.
Require no unused options, CPU-only backend configuration, pinned ELF/current
runtime dependencies and backend registry before model access. Associate both
configure and build log SHAs on failure or success, and retain owned terminal
exit/removal. Follow V2 pilot instructions for exclusive RAM, original four
shard identities/pages, primary/secondary aliases, exact no-thinking template,
greedy two-task natural-finish64 controls and final-head/raw-ID capture.

RESULT: CPU-only recipe comparison verifies exactly two removed arguments;
all other V2 plan fields remain equivalent except version/history/design/test
metadata. Actual V3 configure/build/model inference remains unexecuted here.

VERDICT: V3 is ready for ROOT's fresh CPU-only build. A successful build would
not qualify original/native full48 numerical fidelity, quality or speed.


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
