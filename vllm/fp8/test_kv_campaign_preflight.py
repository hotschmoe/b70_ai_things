"""CPU-only lifecycle tests: distinguish probe refusal from hardware failure."""
import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import kv_campaign_server as server


class PreflightTest(unittest.TestCase):
    def exercise(self, first_collective_rc, guard_refusal=False):
        commands = []
        collective_count = 0
        def run(command, **kwargs):
            nonlocal collective_count
            commands.append(command)
            if Path(command[0]).name == 'xpu-collective-health':
                collective_count += 1
                if collective_count == 1:
                    message = ('xpu-collective-health: refusing P2P=1 without I_KNOW_P2P_WEDGES=1\n'
                               if guard_refusal else '=== collective probe\nworker failed after device setup\n')
                    kwargs['stdout'].write(message)
                    return SimpleNamespace(returncode=first_collective_rc)
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = root / 'config'
            cfg.mkdir()
            (cfg / 'Config.json').write_text(json.dumps({'Env': [], 'Cmd': [
                '--served-model-name', 'test', '--kv-cache-dtype', 'auto', '--speculative-config', '{}']}))
            (cfg / 'Mounts.json').write_text('[]')
            # Every subprocess is mocked; no lease or device operation occurs.
            with patch.object(server.subprocess, 'run', run), patch.object(server.subprocess, 'Popen') as popen, patch('sys.argv', [
                    'server', '--preservation', str(cfg), '--out', str(root / 'result'), '--leased']):
                self.assertEqual(server.main(), 1)
                popen.assert_not_called()
            resets = [cmd for cmd in commands if any(Path(word).name == 'xe-reset' for word in cmd)]
            self.assertEqual(len(resets), int(not guard_refusal))
            self.assertEqual(collective_count, 2)  # pre plus mandatory post
            self.assertEqual((root / 'result/exit.rc').read_text().strip(), '1')

    def test_refusal_does_not_reset_healthy_hardware(self):
        self.exercise(2, guard_refusal=True)

    def test_actual_failure_still_resets_before_post_health(self):
        self.exercise(1)

    def test_inconclusive_device_attempt_still_requires_recovery(self):
        self.exercise(2)

    def test_stop_during_preflight_does_not_launch_and_still_checks_health(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cfg = root / 'config'; cfg.mkdir()
            (cfg / 'Config.json').write_text(json.dumps({'Env': [], 'Cmd': [
                '--served-model-name', 'test', '--kv-cache-dtype', 'auto', '--speculative-config', '{}']}))
            (cfg / 'Mounts.json').write_text('[]')
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                (root / 'result/STOP').touch()
                return SimpleNamespace(returncode=0)
            with patch.object(server.subprocess, 'run', run), patch.object(server.subprocess, 'Popen') as popen, \
                    patch('sys.argv', ['server', '--preservation', str(cfg), '--out', str(root / 'result'), '--leased']):
                self.assertEqual(server.main(), 0)
                popen.assert_not_called()
            self.assertEqual(sum(Path(c[0]).name == 'xpu-health' for c in commands), 2)
            self.assertEqual(sum(Path(c[0]).name == 'xpu-collective-health' for c in commands), 2)
            self.assertFalse(any('xe-reset' in word for c in commands for word in c))

    def test_backend_death_racing_stop_still_recovers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cfg = root / 'config'; cfg.mkdir()
            (cfg / 'Config.json').write_text(json.dumps({'Env': [], 'Cmd': [
                '--served-model-name', 'test', '--kv-cache-dtype', 'auto', '--speculative-config', '{}']}))
            (cfg / 'Mounts.json').write_text('[]')
            commands = []; state = {'dead': False}
            child = SimpleNamespace(poll=lambda: 1 if state['dead'] else None, wait=lambda **_: 1)
            def run(command, **kwargs):
                commands.append(command)
                return SimpleNamespace(returncode=0)
            def ready(*args, **kwargs):
                state['dead'] = True
                (root / 'result/STOP').touch()
                return io.BytesIO(b'{"data":[{"id":"test"}]}')
            with patch.object(server.subprocess, 'run', run), patch.object(server.subprocess, 'Popen', return_value=child), \
                    patch.object(server.urllib.request, 'urlopen', ready), \
                    patch('sys.argv', ['server', '--preservation', str(cfg), '--out', str(root / 'result'),
                                       '--served-model', 'test', '--leased']):
                self.assertEqual(server.main(), 1)
            self.assertEqual(sum(any(Path(w).name == 'xe-reset' for w in c) for c in commands), 1)
            self.assertEqual(sum(Path(c[0]).name == 'xpu-health' for c in commands), 2)


if __name__ == '__main__':
    unittest.main()
