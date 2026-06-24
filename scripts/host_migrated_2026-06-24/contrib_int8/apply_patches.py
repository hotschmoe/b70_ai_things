import sys, os, shutil, vllm
BASE = os.path.dirname(vllm.__file__)   # the ACTUAL imported vllm (resolves editable /workspace too)
# copy the kernel class into the real scaled_mm package (reliable BASE, no shell path capture)
CLS_DST = os.path.join(BASE, "model_executor/kernels/linear/scaled_mm/xpu_int8.py")
shutil.copyfile("/mnt/vm_8tb/b70/contrib_int8/xpu_int8.py", CLS_DST)
print("copied class ->", CLS_DST)
V = os.path.join(BASE, "model_executor/kernels/linear/__init__.py")
print("patching:", V)
src = open(V).read()

# 1) import the kernel class right after the scaled_mm.xpu import block
anchor = ("from vllm.model_executor.kernels.linear.scaled_mm.xpu import (\n"
          "    XPUFp8BlockScaledMMKernel,\n"
          "    XPUFP8ScaledMMLinearKernel,\n)")
if anchor not in src:
    sys.exit("FAIL: scaled_mm.xpu import anchor not found")
add = ("\nfrom vllm.model_executor.kernels.linear.scaled_mm.xpu_int8 import (\n"
       "    XPUInt8ScaledMMLinearKernel,\n)")
src = src.replace(anchor, anchor + add, 1)

# 2) register XPU in _POSSIBLE_INT8_KERNELS (insert before the dict's closing brace)
key = "_POSSIBLE_INT8_KERNELS"
i = src.find(key)
if i < 0:
    sys.exit("FAIL: _POSSIBLE_INT8_KERNELS not found")
j = src.find("\n}", i)
if j < 0:
    sys.exit("FAIL: end of _POSSIBLE_INT8_KERNELS dict not found")
if "PlatformEnum.XPU" in src[i:j]:
    sys.exit("FAIL: XPU already present in INT8 registry")
src = src[:j] + "\n    PlatformEnum.XPU: [XPUInt8ScaledMMLinearKernel]," + src[j:]

# 3) harden the chooser: possible_kernels[...] -> .get(...) with a clear error
sub = "    platform_kernels = possible_kernels[current_platform._enum]"
if sub not in src:
    sys.exit("FAIL: chooser subscript not found")
repl = ("    platform_kernels = possible_kernels.get(current_platform._enum)\n"
        "    if not platform_kernels:\n"
        "        raise ValueError(\n"
        "            \"No ScaledMM linear kernels registered for platform \"\n"
        "            f\"{current_platform._enum}.\")")
src = src.replace(sub, repl, 1)

open(V, "w").write(src)
print("PATCHED linear/__init__.py: import + XPU registry + .get() hardening OK")
