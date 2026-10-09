"""Owned VTune software sampling of a supervised server child, never PID 1."""
import json
from pathlib import Path
import subprocess
import time
import uuid

VTUNE = '/opt/intel/oneapi/vtune/2026.2/bin64/vtune'


class CpuProfile:
    def __init__(self, container, output):
        self.container = container
        self.output = Path(output)
        self.process = None
        self.log = None
        self.stopped = False
        self.owner = uuid.uuid4().hex
        self.attached_epoch = None

    def start(self):
        probe = "import glob,os,json; p=[]\nfor x in glob.glob('/proc/[0-9]*/exe'):\n try:\n  if os.readlink(x).endswith('/llama-server'): p.append(int(x.split('/')[2]))\n except OSError: pass\nprint(json.dumps(p))"
        pids = json.loads(subprocess.check_output(['docker', 'exec', self.container, 'python3', '-c', probe], timeout=20))
        if len(pids) != 1 or pids[0] == 1:
            raise RuntimeError('Software sampling requires one supervised llama-server child, not PID1')
        command = ['docker', 'exec', '-e', 'B70_CPU_PROFILE_OWNER=' + self.owner, self.container, VTUNE, '-collect', 'hotspots', '-knob', 'sampling-mode=sw',
                   '-knob', 'enable-stack-collection=true', '-target-pid', str(pids[0]),
                   '-result-dir', '/results/cpu-profile', '-duration', '2700']
        (self.output / 'cpu-profile-command.json').write_text(json.dumps(command, indent=2) + '\n')
        self.log = (self.output / 'cpu-profiler.log').open('w')
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        return pids[0]

    def ready(self):
        if self.process.poll() is not None:
            raise RuntimeError('Software profiler exited before attachment')
        ready = 'vtune: Collection started.' in (self.output / 'cpu-profiler.log').read_text(errors='replace')
        if ready and self.attached_epoch is None:
            self.attached_epoch = time.time()
        return ready

    def assert_active(self):
        if self.process.poll() is not None:
            raise RuntimeError('Software collection ended before diagnostic requests completed')

    def owned_processes(self):
        script = "import glob,json; p=[]\nfor x in glob.glob('/proc/[0-9]*/environ'):\n try:\n  if b'B70_CPU_PROFILE_OWNER=" + self.owner + "\\x00' in open(x,'rb').read(): p.append(int(x.split('/')[2]))\n except OSError: pass\nprint(json.dumps(p))"
        return json.loads(subprocess.check_output(['docker', 'exec', self.container, 'python3', '-c', script], timeout=20))

    def stop(self):
        if self.stopped or self.process is None:
            return
        errors = []
        try:
            with (self.output / 'cpu-profiler-stop.log').open('w') as log:
                status = subprocess.run(['docker', 'exec', self.container, VTUNE, '-r', '/results/cpu-profile', '-command', 'stop'],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=60).returncode
            if status:
                errors.append('stop exit ' + str(status))
            if self.process.wait(timeout=120):
                errors.append('collector exit nonzero')
        except Exception as error:
            errors.append(type(error).__name__)
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
        self.log.close()
        if errors:
            raise RuntimeError('Software profiler cleanup failed: ' + ', '.join(errors))
        remaining = self.owned_processes()
        if remaining or 'Collection detached.' not in (self.output / 'cpu-profiler.log').read_text(errors='replace'):
            raise RuntimeError('Software collector detachment is not confirmed')
        self.stopped = True
        (self.output / 'cpu-profile-coverage.json').write_text(json.dumps({'attached_epoch': self.attached_epoch,
              'detached_epoch': time.time(), 'owned_processes_remaining': remaining, 'detachment_confirmed': True}, indent=2) + '\n')
        with (self.output / 'cpu-profiler-report.log').open('w') as log:
            status = subprocess.run(['docker', 'exec', self.container, VTUNE, '-report', 'hotspots',
                                     '-r', '/results/cpu-profile', '-format', 'csv', '-csv-delimiter', 'comma',
                                     '-group-by', 'function,module', '-column', 'CPU Time',
                                     '-report-output', '/results/cpu-hotspots.csv'], stdout=log, stderr=subprocess.STDOUT, timeout=60).returncode
        if status:
            raise RuntimeError('Software profiler report failed')
