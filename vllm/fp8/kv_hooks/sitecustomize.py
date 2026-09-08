"""Opt-in hooks inherited by spawned vLLM workers, never enabled by default."""
import os

if os.environ.get('B70_KV_MODE') in ('record', 'load'):
    try:
        import kv_calibration_hook
        kv_calibration_hook.install()
    except BaseException:
        import traceback
        traceback.print_exc()
        # sitecustomize exceptions would otherwise be swallowed by Python.
        os._exit(78)

if os.environ.get('B70_OFFLOAD_GROUP_FIX') == '1':
    try:
        import kv_offload_group_fix
        kv_offload_group_fix.install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)

if os.environ.get('B70_OFFLOAD_TRACE') == '1':
    try:
        import kv_offload_trace
        kv_offload_trace.install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)
