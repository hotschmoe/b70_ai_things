"""Root-only subreaper/session containment and exact CPU-container retirement."""
import ctypes
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
import observe_cpu_swap_attribution_v5 as observer
require = observer.require
screen = observer.screen


def enable_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(36, 1, 0, 0, 0) == 0, 'Required child subreaper installation failed')
    current = ctypes.c_int()
    require(libc.prctl(37, ctypes.byref(current), 0, 0, 0) == 0 and current.value == 1,
            'Child subreaper installation not observed')
    return {'subreaper': True, 'owner_pid': os.getpid()}


def process_row(path):
    path = Path(path)
    text = (path / 'stat').read_text()
    stop = text.rfind(')')
    require(stop > 0, 'Malformed owned process identity')
    fields = text[stop + 2:].split()
    return {'pid': int(path.name), 'ppid': int(fields[1]), 'pgrp': int(fields[2]),
            'session': int(fields[3]), 'start_ticks': int(fields[19]),
            'state': fields[0]}


def session_rows(session, proc='/proc'):
    result = []
    for path in Path(proc).iterdir():
        if not path.name.isdigit():
            continue
        try:
            row = process_row(path)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if row['session'] == session:
            result.append(row)
    return result


class Launches:
    """All original CPU launch descendants share an isolated owned session.

    Subreaping provides actual waitpid handles after producer exit. A metadata
    polling record alone never supplies terminal authority.
    """
    def __init__(self, producer, output):
        self.producer = producer
        self.output = Path(output)
        self.seen = {}
        self.errors = []
        self.stop = threading.Event()
        self.lock = threading.Lock()
        try:
            row = process_row('/proc/' + str(producer.pid))
        except FileNotFoundError:
            require(producer.poll() is not None, 'Missing live original producer identity')
            # Popen actually reaped this owned child. Its start_new_session
            # source contract still requires a full subreaper/session census;
            # missing producer metadata supplies no descendant absence.
            row = None
        if row is not None:
            require(row['session'] == row['pgrp'] == producer.pid and row['ppid'] == os.getpid(),
                    'CPU producer must be original isolated-session child')
        self.initial = row
        self.thread = threading.Thread(target=self.watch, daemon=False)
        self.thread.start()

    def sample(self):
        rows = session_rows(self.producer.pid)
        with self.lock:
            for row in rows:
                key = (row['pid'], row['start_ticks'])
                self.seen[key] = row
        return rows

    def watch(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except BaseException as exc:
                self.errors.append(type(exc).__name__ + ': ' + str(exc))
            self.stop.wait(.025)

    def retire(self, census_callback):
        require(self.producer.poll() is not None, 'Original CPU producer must be joined before descendant drain')
        adopted = {}
        # Never use a timer expiry as absence. Keep exclusion until every
        # isolated-session launch child is actually reaped.
        while True:
            rows = self.sample()
            children = [r for r in rows if r['pid'] != self.producer.pid]
            if not children:
                break
            # A Docker client may still be between its owned Popen and daemon
            # creation. Do not kill that client and guess the request vanished.
            # Let it finish; retire its exact created container when observable.
            census_callback()
            for row in children:
                require(row['ppid'] == os.getpid(), 'Owned orphan not adopted by original subreaper')
                try:
                    now = process_row('/proc/' + str(row['pid']))
                except FileNotFoundError:
                    continue
                require(now['start_ticks'] == row['start_ticks'] and now['session'] == self.producer.pid,
                        'Owned descendant identity changed before retirement')
                waited, status = os.waitpid(now['pid'], os.WNOHANG)
                if waited:
                    adopted[str(waited)] = {'start_ticks': row['start_ticks'], 'wait_status': status,
                                           'actually_reaped': True, 'finished_epoch': time.time()}
            time.sleep(.05)
        self.stop.set()
        self.thread.join()
        require(not session_rows(self.producer.pid), 'Final owned launch session not empty')
        return {'producer_pid': self.producer.pid, 'producer_session': self.producer.pid,
                'initial_process_metadata_observed': self.initial is not None,
                'tracked': list(self.seen.values()), 'adopted_children': adopted,
                'tracking_errors': self.errors, 'launch_session_empty': True,
                'complete_process_ancestry_from_polling_claimed': False,
                'subreaper_wait_and_session_census_required': True}


def command_binding(path, root, plan, pid):
    original = Path(path)
    require(original.is_file() and not original.is_symlink(), 'Original command must be regular nonsymlink')
    require(original.absolute() == original.resolve(), 'Original command ancestor/path alias refused')
    path = original.resolve()
    root = Path(root).resolve()
    require(path.is_relative_to(root) and not path.is_symlink(), 'Command path outside owned screen')
    command = observer.read_unique(path)
    require(type(command) is list and all(type(x) is str for x in command), 'Exact original command argv required')
    name = command[command.index('--name') + 1]
    if re.fullmatch(r'b70-cpu-overlap-screen-' + str(pid) + r'-[0-7]-[01]', name):
        expected = observer.expected_command(path, plan, pid)
        image = plan['image']
    elif re.fullmatch(r'b70-cpu-overlap-wrapper-' + str(pid) + r'-[0-9]+', name):
        require(path == root / 'server-wrapper-preflight/command.json', 'Wrong original preflight command path')
        recipe = dict(plan, memory_cap_bytes=2 << 30)
        expected = screen.server_command(name, path.parent, recipe, Path(plan['build_root']), [], Path(screen.__file__).resolve(), None)
        expected[expected.index('--network') + 1] = 'none'
        expected[expected.index('--cpus') + 1] = '2'
        image = plan['image']
    elif re.fullmatch(r'b70-cpu-overlap-tokenizer-' + str(pid) + r'-[0-9]+', name):
        require(path.parent == root and path.name in ('tokenizer-prepare-command.json', 'tokenizer-decode-command.json'), 'Wrong original tokenizer command path')
        source = Path(plan['tokenizer_source'])
        expected = ['docker', 'run', '--name', name, '--label', 'b70.api-overlap.cpu-screen=' + plan['runner_sha256'], '--network', 'none', '--read-only', '--memory', '512m', '--memory-swap', '512m', '--cpus', '1', '--pids-limit', '128', '--user', str(os.getuid()) + ':' + str(os.getgid()), '--entrypoint', '/usr/bin/env', '-v', str(Path(screen.__file__).resolve()) + ':' + screen.CONTAINER_RUNNER + ':ro', '-v', str(root / 'source-plan.snapshot.json') + ':/pilot-plan.json:ro', '-v', str(source / 'tools/strata_tokenizer.py') + ':/native-tools/strata_tokenizer.py:ro', '-v', str(source / 'serve/frontend.py') + ':/native-serve/frontend.py:ro', '-v', plan['tokenizer_pack'] + ':/tokenizer:ro', '-v', str(root) + ':/results:rw', plan['tokenizer_image'], '-i', 'PATH=/usr/bin:/bin', 'LANG=C', 'LC_ALL=C', 'PYTHONDONTWRITEBYTECODE=1', '/opt/b70-c1-python/bin/python', screen.CONTAINER_RUNNER, '--inside-tokenizer', '/pilot-plan.json', '--token-output', '/results/' + ('tokenizer-fixtures.json' if path.name.startswith('tokenizer-prepare') else 'decoded-output.json')]
        if path.name.startswith('tokenizer-decode'):
            expected += ['--decode-input', '/results/output-id-requests.json']
        image = plan['tokenizer_image']
    else:
        raise ValueError('Foreign original launch name refused')
    require(command == expected, 'Original owned exact CPU recipe differs')
    return command, name, image


def recipe_gate(obj, command, image, label, tokenizer_image_config=None):
    name = command[command.index('--name') + 1]
    require(screen.owned(obj, name, image, label), 'Foreign image/label/name refused')
    if 'b70-cpu-overlap-tokenizer-' not in name:
        screen.runtime_recipe_gate(obj, command)
        return
    host, config = obj['HostConfig'], obj['Config']
    at = command.index(image)
    require(config['Image'] == image and config['Cmd'] == command[at + 1:] and config['Entrypoint'] == ['/usr/bin/env'], 'Tokenizer actual image/argv differs')
    require(host['NetworkMode'] == 'none' and host['ReadonlyRootfs'] is True and host['Memory'] == host['MemorySwap'] == 512 << 20 and host['NanoCpus'] == 10**9 and host['PidsLimit'] == 128, 'Tokenizer actual CPU limits differ')
    require(config['User'] == command[command.index('--user') + 1] and not host.get('Devices') and not host.get('DeviceRequests') and not host['Privileged'] and not host.get('GroupAdd'), 'Tokenizer device/user/group scope differs')
    require(type(tokenizer_image_config) is dict and config['WorkingDir'] == tokenizer_image_config['Config']['WorkingDir'], 'Tokenizer image-default cwd differs')
    expected = sorted(tuple(command[i + 1].rsplit(':', 2)) for i, value in enumerate(command) if value == '-v')
    actual = sorted((r['Source'], r['Destination'], 'rw' if r['RW'] else 'ro') for r in obj['Mounts'] if r['Type'] == 'bind')
    require(actual == expected and len(actual) == len(obj['Mounts']), 'Tokenizer exact mounts differ')


def image_binding(image):
    result = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, text=True, timeout=20)
    require(result.returncode == 0 and not result.stderr.strip(), 'Pinned CPU image inspect failed')
    objects = json.loads(result.stdout)
    require(type(objects) is list and len(objects) == 1 and objects[0]['Id'] == image,
            'Actual tokenizer image ID differs')
    require(type(objects[0]['Config']['WorkingDir']) is str, 'Actual image-default cwd unavailable')
    return objects[0]


def census(root, plan, pid):
    root = Path(root)
    paths = list(root.glob('case*-repeat*/command.json'))
    paths += [p for p in (root / 'server-wrapper-preflight/command.json', root / 'tokenizer-prepare-command.json', root / 'tokenizer-decode-command.json') if p.exists()]
    commands = {}
    for path in paths:
        command, name, image = command_binding(path, root, plan, pid)
        require(name not in commands, 'Duplicate original owned container name')
        commands[name] = (command, image, path)
    query = ['docker', 'ps', '-a', '--filter', 'label=b70.api-overlap.cpu-screen=' + plan['runner_sha256'], '--format', '{{.Names}}']
    result = subprocess.run(query, capture_output=True, text=True, timeout=20)
    require(result.returncode == 0 and not result.stderr.strip(), 'Exact ownership census failed')
    relevant = [name for name in result.stdout.splitlines() if re.fullmatch(r'b70-cpu-overlap-(?:screen|wrapper|tokenizer)-' + str(pid) + r'-.+', name)]
    require(all(name in commands for name in relevant), 'Owned launch lacks original recipe snapshot')
    rows = []
    tokenizer_image = None
    for name, (command, image, path) in commands.items():
        obj = observer.inspect(name)
        before = obj
        recovered = False
        if obj is not None:
            if name.startswith('b70-cpu-overlap-tokenizer-') and tokenizer_image is None:
                tokenizer_image = image_binding(image)
            recipe_gate(obj, command, image, plan['runner_sha256'], tokenizer_image)
            recovered = True
            if obj['State']['Running']:
                stopped = subprocess.run(['docker', 'stop', '--time', '10', name], capture_output=True, text=True, timeout=30)
                require(stopped.returncode == 0, 'Owned late-launch stop failed')
            obj = observer.inspect(name)
            require(obj is not None, 'Owned stopped container inspection disappeared')
            recipe_gate(obj, command, image, plan['runner_sha256'], tokenizer_image)
            require(obj['State']['Running'] is False, 'Owned late launch still running')
            removed = subprocess.run(['docker', 'rm', name], capture_output=True, text=True, timeout=20)
            require(removed.returncode == 0, 'Owned late-launch removal failed')
        require(observer.inspect(name) is None, 'Exact final absence missing')
        rows.append({'name': name, 'command': command, 'command_sha256': observer.sha(path),
                     'inspection_before': before, 'terminal': None if obj is None else obj['State'],
                     'late_launch_recovery_performed': recovered, 'actual_absence_verified': True})
    return {'rows': rows, 'finished_epoch': time.time(), 'all_owned_launches_absent': True,
            'tokenizer_image_observation': tokenizer_image,
            'recovery_performed': any(r['late_launch_recovery_performed'] for r in rows),
            'Docker_daemon_async_creation_time_bound_claimed': False,
            'original_screen_qualification_transferred': False}


def tree_view(root):
    return {str(p.relative_to(root)): {'sha256': observer.sha(p), 'bytes': p.stat().st_size}
            for p in Path(root).rglob('*') if p.is_file()}


def original_artifact_changes(root):
    root = Path(root)
    path = root / 'report.json'
    require(path.is_file(), 'Original producer report missing after ownership retirement')
    report = observer.read_unique(path)
    expected = report['artifact_sha256']
    actual = {str(p.relative_to(root)): observer.sha(p) for p in root.rglob('*')
              if p.is_file() and p != path}
    changed = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
    return {'original_report_sha256': observer.sha(path), 'original_passed': report['passed'],
            'changed_paths': changed, 'original_artifacts_unchanged': not changed,
            'original_failed_result_rewritten': False}
