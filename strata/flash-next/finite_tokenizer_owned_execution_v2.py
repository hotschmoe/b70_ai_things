"""Root-only CPU tokenizer Docker client, kept alive until creation settles."""
import signal
import subprocess
import time
from pathlib import Path
from cpu_retry_owned_drain_v1 import Launches, enable_subreaper
from observe_cpu_swap_attribution_v4 import inspect


def execute(command, output, owned, timeout=120):
    output = Path(output)
    stop = []
    previous = {}
    process = None
    launches = None
    error = None
    subreaper = enable_subreaper()
    name = command[command.index('--name') + 1]

    def interrupted(sig, frame):
        stop.append(sig)

    def stop_owned():
        obj = inspect(name)
        if obj is not None:
            owned(obj)
            if obj['State']['Running']:
                row = subprocess.run(['docker', 'stop', '--time', '10', name], capture_output=True, text=True, timeout=30)
                if row.returncode != 0:
                    raise ValueError('Exact owned tokenizer stop failed')

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[sig] = signal.signal(sig, interrupted)
    started = time.time()
    try:
        with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            launches = Launches(process, output)
            while process.poll() is None:
                if stop or time.time() - started >= timeout:
                    error = 'Owned tokenizer interrupted/deadline; actual Docker creation/exit still required'
                    # Never kill the pending Docker client and guess that its
                    # daemon-side create request was canceled.
                    stop_owned()
                time.sleep(.05)
            return_code = process.wait()
        descendants = launches.retire(stop_owned)
        return {'return_code': return_code, 'error': error,
                'command': list(command),
                'client_pid': process.pid, 'client_started_epoch': started,
                'client_terminal_epoch': time.time(), 'stop_signals': stop,
                'subreaper': subreaper, 'launch_descendants': descendants,
                'Docker_daemon_async_creation_time_bound_claimed': False}
    finally:
        # Exceptions from inspection/ownership do not authorize orphaning the
        # client. Retain this root producer until its actual launch settles.
        if process is not None:
            while process.poll() is None:
                try:
                    stop_owned()
                except BaseException as exc:
                    error = str(exc)
                time.sleep(.1)
            process.wait()
            if launches is not None and launches.thread.is_alive():
                while True:
                    try:
                        launches.retire(stop_owned)
                        break
                    except BaseException:
                        time.sleep(.1)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
