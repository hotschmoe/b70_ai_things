#!/usr/bin/env python3
"""Level Zero peer access precheck (zeDeviceCanAccessPeer both directions) for the B70 pair. Enumeration only: no
context, no memory, no kernels. NOT RUN by the prep agent. Run under: bin/gpu-run python3 -I ze_p2p_precheck.py
(whole box lease; do not set ZE_AFFINITY_MASK, both cards must be visible)."""
import ctypes as C, struct
ze = C.CDLL("libze_loader.so.1")
def chk(r, what):
    if r: raise SystemExit(f"{what} failed: 0x{r:x}")
chk(ze.zeInit(1), "zeInit")                      # ZE_INIT_FLAG_GPU_ONLY
n = C.c_uint32(0); chk(ze.zeDriverGet(C.byref(n), None), "zeDriverGet")
drv = (C.c_void_p * n.value)(); chk(ze.zeDriverGet(C.byref(n), drv), "zeDriverGet")
for di, d in enumerate(drv):
    k = C.c_uint32(0); chk(ze.zeDeviceGet(d, C.byref(k), None), "zeDeviceGet")
    devs = (C.c_void_p * k.value)(); chk(ze.zeDeviceGet(d, C.byref(k), devs), "zeDeviceGet")
    info = []
    for dev in devs:
        buf = C.create_string_buffer(1024); struct.pack_into("<I", buf, 0, 0x3)   # ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
        chk(ze.zeDeviceGetProperties(C.c_void_p(dev), buf), "zeDeviceGetProperties")
        dev_id = struct.unpack_from("<I", buf, 24)[0]; name = buf.raw[112:112 + 256].split(b"\0")[0].decode()
        info.append((dev, dev_id, name))
    print(f"driver {di}: {len(devs)} devices")
    for i, (_, did, name) in enumerate(info): print(f"  dev{i}: deviceId=0x{did:x} {name}")
    for i, (a, _, _) in enumerate(info):
        for j, (b, _, _) in enumerate(info):
            if i == j: continue
            can = C.c_uint8(0); chk(ze.zeDeviceCanAccessPeer(C.c_void_p(a), C.c_void_p(b), C.byref(can)), "zeDeviceCanAccessPeer")
            print(f"  zeDeviceCanAccessPeer(dev{i} -> dev{j}) = {can.value}")
