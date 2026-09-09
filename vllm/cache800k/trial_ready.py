#!/usr/bin/env python3
"""Wait for the validated trial frontdoor; performs no generation."""
import time
import os
import select
import sys
import urllib.request

pidfd = os.pidfd_open(int(sys.argv[1]))
poller = select.poll()
poller.register(pidfd, select.POLLIN)
deadline = time.monotonic() + 8500
while time.monotonic() < deadline:
    if poller.poll(0):
        raise RuntimeError('trial launcher exited before readiness')
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
