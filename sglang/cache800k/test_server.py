"""CPU-only lifecycle regression tests; no Docker or GPU commands execute."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('campaign_sglang_server', Path(__file__).with_name('server.py'))
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class LifecycleTest(unittest.TestCase):
    def exercise(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'arm'
            commands = []
            class Process:
                exited = False
                def poll(self):
                    return 1 if self.exited else None
                def wait(self, timeout):
                    return 0
            process = Process()
            def run(command, **kwargs):
                commands.append(command)
                if command == ['cpu-probe']:
                    (out / 'STOP').touch()
                    if mode == 'crash':
                        process.exited = True
                    if mode == 'timeout':
                        raise subprocess.TimeoutExpired(command, 1)
                    return subprocess.CompletedProcess(command, 1 if mode in ('crash', 'failed') else 0)
                return subprocess.CompletedProcess(command, 0)
            def start(*args, **kwargs):
                (out / 'jobs/01.json').write_text(json.dumps({'command': ['cpu-probe'], 'timeout': 1}))
                if mode == 'stop_crash':
                    process.exited = True
                    (out / 'STOP').touch()
                return process
            argv = ['server', '--out', str(out), '--served-model', 'test-int4-fp16', '--leased']
            with patch('sys.argv', argv), patch.object(server.subprocess, 'run', run), patch.object(server.subprocess, 'Popen', start), patch.object(server.urllib.request, 'urlopen', return_value=io.BytesIO(b'{"data":[{"id":"test-int4-fp16"}]}')), patch.object(server.time, 'sleep'), patch.object(server.signal, 'signal'):
                result = server.main()
            failed = mode != 'success'
            self.assertEqual(result, int(failed))
            self.assertEqual((out / 'exit.rc').read_text(), str(int(failed)) + '\n')
            self.assertEqual((out / 'failure.txt').exists(), failed)
            reset = [i for i, command in enumerate(commands) if any('xe-reset' in part for part in command)]
            self.assertEqual(len(reset), int(failed))
            health = [i for i, command in enumerate(commands) if command[0].endswith('xpu-health') or command[0].endswith('xpu-collective-health')]
            self.assertEqual(len(health), 4)
            if reset:
                stop = next(i for i, command in enumerate(commands) if command[:2] == ['docker', 'stop'])
                self.assertLess(stop, reset[0])
                self.assertLess(reset[0], health[-2])
            if mode != 'stop_crash':
                self.assertEqual((out / 'jobs/01.done').read_text(), ('124' if mode == 'timeout' else '1' if failed else '0') + '\n')

    def test_failed_probe_stop_race_with_crashed_server(self):
        self.exercise('crash')

    def test_failed_probe_before_server_exit(self):
        self.exercise('failed')

    def test_timeout_stop_race(self):
        self.exercise('timeout')

    def test_exit_and_stop_during_startup(self):
        self.exercise('stop_crash')

    def test_successful_probe_and_owned_stop(self):
        self.exercise('success')


if __name__ == '__main__':
    unittest.main()
