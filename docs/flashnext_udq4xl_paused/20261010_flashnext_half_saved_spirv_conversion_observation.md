# Saved post-link SPIRV half conversion observation

CONFIG -> Reviewed704f recipe, genuine db27 saved bitcode/symbols, exact observed
original translator options. Six direct/graph-wrapper modules. Fresh translation
is separate from original executed ELF and original runtime JIT ISA.

COMMAND -> Actual65718 under card0 lease/pin, terminal0/PASS60s. Capture binding
ceb15c5f0f2f95690bad1b21cc1714afff080ccfc5fab697ef4c48047fd5afa9.

RESULT -> Direct Half37Expression(module33) and graph wrapper(module21) contain
32-bit float Load, FConvert to TypeFloat16, FConvert back to TypeFloat32 and
32-bit Store. Both results have FPRoundingMode0 decorations. Materialized
store modules36/24 contain FConvert32-to16 and16-bit Store; load modules38/25
contain16-bit Load, FConvert16-to32 and32-bit Store. All six current text/code
artifacts and original input/module/source hashes retained; helper not executed.

VERDICT -> The fresh post-link translation retains the declared conversion
roundtrip; it does not show frontend deletion. Runtime expression identity versus
materialized rounding is already observed, but original embedded image/JIT ISA
and execution cause remain unproven. No model/reference/tolerance modification
or broad original-model fidelity claim. Next extraction must distinguish original
executed ELF from fresh/offline artifacts and pin actual tools/target/IGC.

Runtime artifacts: /mnt/vm_8tb/b70/build/half37-spirv-text-v3-20261010/
