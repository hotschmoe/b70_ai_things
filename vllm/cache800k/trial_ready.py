#!/usr/bin/env python3
"""Wait for the validated trial frontdoor; performs no generation."""
import time
import urllib.request

deadline = time.monotonic() + 8500
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=3) as response:
            if response.status == 200:
                print('Validated FP8 trial frontdoor ready', flush=True)
                break
    except OSError:
        pass
    time.sleep(2)
else:
    raise RuntimeError('trial readiness deadline expired')
