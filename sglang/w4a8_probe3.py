#!/usr/bin/env python3
# W4A8 gate-3: introspect auto_round_kernel int8 paths (woqgemm_s8, igemm_s8s8s32)
# and QuantLinearGPTQ forward -- learn the exact API to wire a fused int4w x int8a.
import inspect, torch
import auto_round_kernel as ark
from auto_round_kernel import qlinear

def show(obj, name):
    print(f"\n===== {name} =====")
    try: print("signature:", inspect.signature(obj))
    except Exception as e: print("sig:", e)
    try:
        src=inspect.getsource(obj)
        print(src[:2600])
    except Exception as e: print("src:", e)

print("ark module file:", getattr(ark,'__file__',None))
print("ark attrs:", [a for a in dir(ark) if not a.startswith('__')])
for n in ("woqgemm","woqgemm_s8","igemm_s8s8s32","woq_linear","repack_quantized_weight"):
    o=getattr(ark,n,None)
    if o is not None: show(o, "ark."+n)
    else: print(f"\nark.{n}: ABSENT")

# QuantLinearGPTQ forward path
show(qlinear.QuantLinearGPTQ.forward, "QuantLinearGPTQ.forward")
# any s8/int8 QuantLinear class?
print("\nqlinear classes:", [c for c in dir(qlinear) if 'Quant' in c or 's8' in c.lower() or 'S8' in c])
