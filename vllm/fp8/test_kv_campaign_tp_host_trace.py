"""CPU-only argv, activation and refusal tests; every subprocess is mocked."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import kv_campaign_server as server

ROOT = Path(__file__).resolve().parent


class TPHostTraceIntegrationTests(unittest.TestCase):
    def command(self, extra):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cfg = root / 'config'; cfg.mkdir()
            out = root / 'result'
            (cfg / 'Config.json').write_text(json.dumps({'Env': [
                'CCL_TOPO_P2P_ACCESS=0', 'PYTHONPATH=/kept/hooks', 'KEEP_FIXTURE=1'],
                'Cmd': ['--tensor-parallel-size', '2', '--served-model-name', 'test',
                        '--kv-cache-dtype', 'auto', '--speculative-config', '{}']}))
            (cfg / 'Mounts.json').write_text('[]')
            def stop_in_preflight(*args, **kwargs):
                (out / 'STOP').touch()
                return SimpleNamespace(returncode=0)
            with patch.object(server.subprocess, 'run', stop_in_preflight), \
                    patch.object(server.subprocess, 'Popen') as popen, \
                    patch('sys.argv', ['server', '--preservation', str(cfg),
                                       '--out', str(out), '--leased', *extra]):
                self.assertEqual(server.main(), 0)
                popen.assert_not_called()
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertIn('kv_hooks/b70_tp_host_trace.py', manifest['source_sha256'])
            return [part.replace(str(root), '<TEMP>') for part in manifest['command']]

    def test_defaults_preserve_prechange_command_bytes(self):
        expected = json.loads((ROOT / 'testdata/tp_host_trace_default_commands.json').read_text())
        for hook in ('none', 'load'):
            with self.subTest(hook=hook):
                self.assertEqual(self.command(['--hook', hook]), expected[hook])

    def test_profiles_force_entry_and_forward_environment_with_hook_none(self):
        for profile in ('phase', 'phase-mrv1'):
            command = self.command(['--tp-host-trace', profile, '--p2p', '0',
                                    '--tp-host-trace-max-events', '321'])
            self.assertEqual(command[command.index('--entrypoint') + 1], '/opt/venv/bin/python')
            self.assertIn('/kv-source/kv_campaign_entry.py', command)
            self.assertIn('PYTHONPATH=/kv-source/kv_hooks:/kept/hooks', command)
            self.assertIn('B70_KV_MODE=none', command)
            self.assertIn('B70_TP_HOST_TRACE_DIR=/kv-campaign/tp-host-trace', command)
            self.assertIn('B70_TP_HOST_TRACE_PROFILE=' + profile, command)
            self.assertIn('B70_TP_HOST_TRACE_MAX_EVENTS=321', command)
            self.assertIn('KEEP_FIXTURE=1', command)

    def test_guards_refuse_before_lease_or_subprocess(self):
        cases = [[], ['--p2p', '1'], ['--p2p', '0', '--tensor-parallel-size', '1'],
                 ['--p2p', '0', '--packaged-hooks'],
                 ['--p2p', '0', '--tp-host-trace-max-events', '0']]
        for extra in cases:
            with self.subTest(extra=extra), \
                    patch('sys.argv', ['server', '--preservation', '/unused', '--out', '/unused',
                                       '--tp-host-trace', 'phase', *extra]), \
                    patch.object(server.os, 'execv') as execute, \
                    patch.object(server.subprocess, 'run') as run, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    server.main()
                self.assertEqual(exc.exception.code, 2)
                execute.assert_not_called(); run.assert_not_called()

    def test_optin_import_target_preserves_existing_hooks(self):
        path = ROOT / 'kv_hooks/sitecustomize.py'
        for enabled in (False, True):
            env = {'B70_KV_MODE': 'load', 'B70_OFFLOAD_TRACE': '1'}
            if enabled:
                env['B70_TP_HOST_TRACE_DIR'] = '/private'
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(sys, 'meta_path', list(sys.meta_path)):
                result = runpy.run_path(str(path))
            targets = result['TARGETS']
            self.assertEqual('vllm.v1.worker.gpu_model_runner' in targets, enabled)
            self.assertEqual(targets['vllm.model_executor.layers.attention.attention'], ['kv_calibration_hook'])
            self.assertEqual(targets['vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler'], ['kv_offload_trace'])


if __name__ == '__main__':
    unittest.main()
