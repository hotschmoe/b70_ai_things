SOURCE ONLY standalone S138 HOST checker V2; no wrapper/build changes.
V1 LC-only assumption caught before execution; preserve V1. Root builds via
its manual-host derivative, so no automatic metadata gate delays compilation.
After actual build root may invoke under CPU lease/e41/noDRI:

python3 HOSTCHECK_V2/host_check.py ACTUAL_NATIVE_OBJECT NEW_OUTPUT_DIRECTORY

V2 resolves actual R_X86_64_32/32S/64 SECTION+ADDEND strings using readelf
section names/raw offsets; observed S137 form .rodata.str1.1+0x2a42 supported.
Repeated references deduplicate by section/value. LC-symbol fallback remains.
No PC-relative section addend correction is guessed. Original narrow selected
HOST disassembly/logging+parser guard checks/Info capture unchanged; missing
Info stays UNKNOWN. No device image/ISA, main/archive/model scan or GPU.
Only source AST/ASCII validation performed; no object or tool execution.
