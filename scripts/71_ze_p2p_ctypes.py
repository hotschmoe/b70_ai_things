#!/usr/bin/env python3
# RAW Level-Zero P2P probe via ctypes -> libze_loader.so. Bypasses torch/oneCCL entirely and asks the
# driver directly: can B70 dev0 peer-access dev1? This is the authoritative answer to P2P_GPU F.3/F.4
# ("no userspace P2P enablement matrix found for B70"). Calls: zeInit, zeDriverGet, zeDeviceGet,
# zeDeviceCanAccessPeer (both directions), zeDeviceGetP2PProperties (ACCESS/ATOMICS flags), and a
# best-effort IPC-handle open (zeMemAllocDevice -> zeMemGetIpcHandle -> zeMemOpenIpcHandle on the peer).
# Run under a matrix of env vars (71_run_ze_matrix.sh) to see which settings flip peer access on.
import ctypes as C, os, sys

ZE_RESULT_SUCCESS = 0
ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES = 0x9  # from ze_api.h ze_structure_type_t

def load():
    for name in ("libze_loader.so.1", "libze_loader.so"):
        try:
            return C.CDLL(name)
        except OSError:
            continue
    print("FATAL: cannot dlopen libze_loader.so(.1)"); sys.exit(2)

class ze_p2p_props(C.Structure):
    _fields_ = [("stype", C.c_int), ("pNext", C.c_void_p), ("flags", C.c_uint32)]
class ze_ipc_handle(C.Structure):
    _fields_ = [("data", C.c_char * 64)]

def main():
    print("=== env of interest ===")
    for k in ("ZE_AFFINITY_MASK","ZE_FLAT_DEVICE_HIERARCHY","NEOReadDebugKeys","EnableCrossDeviceAccess",
              "EnableP2P","CCL_TOPO_P2P_ACCESS","SYCL_UR_USE_LEVEL_ZERO_V2"):
        v = os.environ.get(k)
        if v is not None: print(f"  {k}={v}")
    ze = load()
    r = ze.zeInit(C.c_uint32(0))
    print(f"zeInit -> 0x{r & 0xffffffff:x} ({'OK' if r==0 else 'ERR'})")
    if r != 0: return 1
    # drivers
    n = C.c_uint32(0)
    ze.zeDriverGet(C.byref(n), None)
    print(f"drivers: {n.value}")
    drivers = (C.c_void_p * n.value)()
    ze.zeDriverGet(C.byref(n), drivers)
    devs = []
    for di in range(n.value):
        dn = C.c_uint32(0)
        ze.zeDeviceGet(drivers[di], C.byref(dn), None)
        arr = (C.c_void_p * dn.value)()
        ze.zeDeviceGet(drivers[di], C.byref(dn), arr)
        for k in range(dn.value):
            devs.append((di, arr[k]))
        print(f"  driver[{di}] devices: {dn.value}")
    print(f"TOTAL devices visible to L0: {len(devs)}")
    if len(devs) < 2:
        print("NEED >=2 devices for P2P probe (ZE_AFFINITY_MASK may be hiding one)"); return 1
    # canAccessPeer matrix (use first two)
    d0 = devs[0][1]; d1 = devs[1][1]
    print("=== zeDeviceCanAccessPeer (THE answer) ===")
    for (a, b, lbl) in ((d0, d1, "dev0->dev1"), (d1, d0, "dev1->dev0")):
        val = C.c_uint8(0)
        rr = ze.zeDeviceCanAccessPeer(a, b, C.byref(val))
        print(f"  {lbl}: result=0x{rr & 0xffffffff:x} canAccess={bool(val.value)}")
    print("=== zeDeviceGetP2PProperties (flags: bit0=ACCESS bit1=ATOMICS) ===")
    props = ze_p2p_props(); props.stype = ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES; props.pNext = None
    rr = ze.zeDeviceGetP2PProperties(d0, d1, C.byref(props))
    print(f"  result=0x{rr & 0xffffffff:x} flags=0x{props.flags:x} "
          f"ACCESS={'Y' if props.flags & 1 else 'N'} ATOMICS={'Y' if props.flags & 2 else 'N'}")
    # best-effort IPC handle path (what oneCCL drmfd uses). Needs a context + device alloc.
    print("=== IPC handle export/import probe (best-effort) ===")
    try:
        # zeContextCreate(driver, &desc, &ctx); desc = {stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC(0x16),pNext,flags}
        class ctx_desc(C.Structure):
            _fields_ = [("stype", C.c_int), ("pNext", C.c_void_p), ("flags", C.c_uint32)]
        cd = ctx_desc(); cd.stype = 0x16; cd.pNext = None; cd.flags = 0
        ctx = C.c_void_p()
        rc = ze.zeContextCreate(drivers[0], C.byref(cd), C.byref(ctx))
        print(f"  zeContextCreate -> 0x{rc & 0xffffffff:x}")
        if rc == 0:
            # zeMemAllocDevice(ctx, &devdesc, size, align, hDevice, &ptr)
            class dev_mem_desc(C.Structure):
                _fields_ = [("stype", C.c_int), ("pNext", C.c_void_p), ("flags", C.c_uint32), ("ordinal", C.c_uint32)]
            dm = dev_mem_desc(); dm.stype = 0x15; dm.pNext = None; dm.flags = 0; dm.ordinal = 0
            ptr = C.c_void_p()
            ra = ze.zeMemAllocDevice(ctx, C.byref(dm), C.c_size_t(1<<20), C.c_size_t(64), d0, C.byref(ptr))
            print(f"  zeMemAllocDevice(dev0,1MB) -> 0x{ra & 0xffffffff:x}")
            if ra == 0:
                ipc = ze_ipc_handle()
                rg = ze.zeMemGetIpcHandle(ctx, ptr, C.byref(ipc))
                print(f"  zeMemGetIpcHandle -> 0x{rg & 0xffffffff:x}")
                if rg == 0:
                    pout = C.c_void_p()
                    ro = ze.zeMemOpenIpcHandle(ctx, d1, ipc, C.c_uint32(0), C.byref(pout))
                    print(f"  zeMemOpenIpcHandle(on dev1) -> 0x{ro & 0xffffffff:x} "
                          f"({'PEER MAP OK' if ro==0 else 'peer-open FAILED'})")
    except AttributeError as e:
        print(f"  IPC probe skipped (symbol missing): {e}")
    except Exception as e:
        print(f"  IPC probe error: {type(e).__name__}: {e}")
    print("DONE_ZE_PROBE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
