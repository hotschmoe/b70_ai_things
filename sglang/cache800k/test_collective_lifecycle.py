"""CPU-only tests of timeout, teardown and recovery ordering."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import collective_lifecycle as lifecycle


class LifecycleTests(unittest.TestCase):
    def exercise(self, failed=False, remaining='', inspect_rc=0, recovery_rc=0, post_rc=0):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            calls = []
            def run(command, destination, timeout):
                calls.append(command)
                destination.write_text(remaining if destination.name == 'remaining-container.log' else '')
                if destination.name == 'remaining-container.log':
                    return inspect_rc
                if destination.name == 'recovery.log':
                    return recovery_rc
                if destination.name.startswith('post-'):
                    return post_rc
                return 0
            rc = lifecycle.finish(out, 'test-control', failed, run)
            return rc, calls

    def test_success_removes_before_health_without_reset(self):
        rc, calls = self.exercise()
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][:3], ['docker', 'rm', '-f'])
        self.assertEqual(len(calls), 4)
        self.assertTrue(calls[2][0].endswith('/xpu-health'))

    def test_failed_control_recovers_before_both_post_checks(self):
        rc, calls = self.exercise(failed=True)
        self.assertEqual(rc, 1)
        self.assertIn('B70_XE_RESET_UNDER_LEASE=1', calls[2])
        self.assertTrue(calls[3][0].endswith('/xpu-health'))
        self.assertTrue(calls[4][0].endswith('/xpu-collective-health'))

    def test_unverified_teardown_never_resets_or_probes(self):
        for options in ({'remaining': 'container-id'}, {'inspect_rc': 124}):
            rc, calls = self.exercise(failed=True, **options)
            self.assertEqual(rc, 1)
            self.assertEqual(len(calls), 2)

    def test_recovery_or_post_health_failure_stays_failed(self):
        self.assertEqual(self.exercise(failed=True, recovery_rc=1)[0], 1)
        self.assertEqual(self.exercise(post_rc=1)[0], 1)

    def test_outer_timeout_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(lifecycle.subprocess, 'run', side_effect=subprocess.TimeoutExpired('docker', 240)):
                self.assertEqual(lifecycle.execute(['docker'], Path(directory) / 'log', 240), 124)

    def test_command_uses_exact_image_and_subgroup_source(self):
        name, command = lifecycle.docker_command(Path('/raw'), {'CCL_TOPO_P2P_ACCESS': '0'})
        self.assertIn(lifecycle.IMAGE, command)
        self.assertIn('CCL_TOPO_P2P_ACCESS=0', command)
        self.assertIn('/raw/collective_control.py:/control.py:ro', command)
        self.assertIn('--nproc-per-node=2', command)
        self.assertNotIn('--rm', command)  # Lifecycle explicitly owns removal.

    def test_queued_job_success_and_failure_with_stop_race(self):
        for control_rc in (0, 1, 124):
            with self.subTest(control_rc=control_rc), tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / 'arm'
                reference = Path(directory) / 'reference.json'
                reference.write_text(json.dumps({'env': {'CCL_TOPO_P2P_ACCESS': '0'}}))
                digest = hashlib.sha256(Path(lifecycle.__file__).with_name('collective_control.py').read_bytes()).hexdigest()
                def execute(command, destination, timeout):
                    destination.write_text('')
                    if destination.name == 'pre-xpu-collective-health.log':
                        (out / 'jobs/01.json').write_text(json.dumps({'command': ['fresh-subgroup-control'], 'timeout': 240}))
                    if destination.name == 'control.log':
                        self.assertEqual(timeout, 240)
                        (out / 'STOP').touch()
                        return control_rc
                    return 0
                argv = ['control', '--out', str(out), '--reference', str(reference),
                        '--source-sha256', digest, '--leased']
                with patch('sys.argv', argv), patch.object(lifecycle, 'execute', execute), patch.object(lifecycle, 'finish', return_value=int(bool(control_rc))) as finish, patch.object(lifecycle.signal, 'signal'), patch.object(lifecycle.time, 'sleep'):
                    self.assertEqual(lifecycle.main(), int(bool(control_rc)))
                self.assertEqual((out / 'jobs/01.done').read_text(), str(control_rc) + '\n')
                self.assertEqual(finish.call_args.args[2], bool(control_rc))


if __name__ == '__main__':
    unittest.main()
