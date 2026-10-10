"""CPU-only parser, ownership, and actual tiny orphan waitpid controls."""
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import cpu_retry_owned_drain_v1 as drain
import observe_cpu_swap_attribution_v4 as observer


class Controls(unittest.TestCase):
    def test_exact_absence_both_actual_capitalizations(self):
        for text in ('Error response from daemon: No such object: owned\n', 'error: no such object: owned\n', 'Error: No such object: owned\n'):
            self.assertIsNone(observer.inspection_result('owned', Mock(returncode=1, stdout='[]\n', stderr=text)))

    def test_absence_never_arbitrary_failure_or_other_name(self):
        for rc, out, err in ((1, '[]', 'permission denied'), (1, '[]', 'error: no such object: foreign'),
                             (125, '[]', 'error: no such object: owned'), (1, '{}', 'error: no such object: owned'),
                             (1, '', 'error: no such object: owned\nconnection failure')):
            self.assertRaises(ValueError, observer.inspection_result, 'owned', Mock(returncode=rc, stdout=out, stderr=err))

    def test_success_exact_named_object_and_no_stderr(self):
        obj = dict(Name='/owned')
        self.assertEqual(observer.inspection_result('owned', Mock(returncode=0, stdout=json.dumps([obj]), stderr='')), obj)
        self.assertRaises(ValueError, observer.inspection_result, 'owned', Mock(returncode=0, stdout='[]', stderr=''))

    def test_original_failed_artifact_mutation_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'server.log'
            log.write_text('old')
            (root / 'report.json').write_text(json.dumps({'passed': False, 'artifact_sha256': {'server.log': observer.sha(log)}}))
            log.write_text('late launch wrote here')
            result = drain.original_artifact_changes(root)
            self.assertEqual(result['changed_paths'], ['server.log'])
            self.assertFalse(result['original_passed'])
            self.assertFalse(result['original_failed_result_rewritten'])

    def test_foreign_recipe_never_stopped(self):
        command = ['docker', 'run', '--name', 'b70-cpu-overlap-screen-123-0-0']
        for obj in (dict(Name='/foreign', Image='image', Config={'Labels': {'b70.api-overlap.cpu-screen': 'label'}}),
                    dict(Name='/b70-cpu-overlap-screen-123-0-0', Image='wrong', Config={'Labels': {'b70.api-overlap.cpu-screen': 'label'}}),
                    dict(Name='/b70-cpu-overlap-screen-123-0-0', Image='image', Config={'Labels': {'b70.api-overlap.cpu-screen': 'foreign'}})):
            self.assertRaises(ValueError, drain.recipe_gate, obj, command, 'image', 'label')

    def test_actual_orphan_adopted_and_waited_without_killing_launch(self):
        script = '''import subprocess,sys,time,tempfile,json
import cpu_retry_owned_drain_v1 as d
d.enable_subreaper()
with tempfile.TemporaryDirectory() as root:
 p=subprocess.Popen([sys.executable,'-c',"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(.5)']);time.sleep(.15)"],start_new_session=True)
 launches=d.Launches(p,root);p.wait();calls=[]
 result=launches.retire(lambda:calls.append(1));assert result['launch_session_empty'];assert calls;assert result['adopted_children'];assert all(r['actually_reaped']for r in result['adopted_children'].values())
 print('tiny owned orphan adopted+actually waited')
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_immediate_exit_still_runs_session_drain(self):
        script = '''import subprocess,sys,tempfile
import cpu_retry_owned_drain_v1 as d
d.enable_subreaper()
with tempfile.TemporaryDirectory() as root:
 p=subprocess.Popen([sys.executable,'-c','raise SystemExit(3)'],start_new_session=True)
 assert p.wait()==3
 launches=d.Launches(p,root);result=launches.retire(lambda:None)
 assert result['launch_session_empty'];assert result['initial_process_metadata_observed']is False
 print('actual Popen joined; separate session census complete')
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_passive_observer_failure_does_not_stop_inference(self):
        source = (Path(__file__).parent / 'run_cpu_overlap_observed_retry_v3.py').read_text()
        self.assertIn('unchanged inference continues under its own guards', source)
        tree = ast.parse(source)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        loops = [n for n in ast.walk(main) if isinstance(n, ast.While) and 'cpu.poll() is None' == ast.unparse(n.test)]
        self.assertEqual(len(loops), 1)
        self.assertNotIn('SIGTERM', ast.unparse(loops[0]))
        self.assertIn('start_new_session=True', source)
        self.assertIn('launches.retire(actual_census)', source)

    def test_frozen_actual_tokenizer_prepare_decode_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(runner_sha256='label', tokenizer_source='/tiny/source',
                        tokenizer_pack='/tiny/tokenizer', tokenizer_image='image', source_bindings={})
            obj = dict(Name='/b70-cpu-overlap-tokenizer-123-456', Image='image',
                       Config={'Labels': {'b70.api-overlap.cpu-screen': 'label'}},
                       State={'Running': False, 'ExitCode': 0, 'OOMKilled': False},
                       HostConfig={'Devices': [], 'DeviceRequests': None})
            for decode in (None, root / 'output-id-requests.json'):
                with patch.object(drain.screen.os, 'getpid', return_value=123), patch.object(drain.screen.time, 'time_ns', return_value=456), patch.object(drain.screen, 'inspect', return_value=obj), patch.object(drain.screen.subprocess, 'run', return_value=Mock(returncode=0, stdout='', stderr='')), patch.object(drain.screen, 'read', return_value={'tokenizer_file_sha256': {}}):
                    drain.screen.metadata_tokenizer(plan, root, decode)
                path = root / ('tokenizer-prepare-command.json' if decode is None else 'tokenizer-decode-command.json')
                command, name, image = drain.command_binding(path, root, plan, 123)
                self.assertEqual(image, 'image')
                if decode:
                    self.assertEqual(command[-1], '/results/output-id-requests.json')
                command[command.index('--network') + 1] = 'host'
                path.write_text(json.dumps(command))
                self.assertRaises(ValueError, drain.command_binding, path, root, plan, 123)

    def test_original_preflight_recipe_exact_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'server-wrapper-preflight'
            out.mkdir()
            plan = dict(image='image', build_root='/tiny/build', runner_sha256='label', memory_cap_bytes=116 << 30)
            command = drain.screen.server_command('b70-cpu-overlap-wrapper-123-456', out, dict(plan, memory_cap_bytes=2 << 30), Path(plan['build_root']), [], Path(drain.screen.__file__).resolve(), None)
            command[command.index('--network') + 1] = 'none'
            command[command.index('--cpus') + 1] = '2'
            path = out / 'command.json'
            path.write_text(json.dumps(command))
            self.assertEqual(drain.command_binding(path, root, plan, 123)[0], command)
            target = out / 'actual.json'
            path.rename(target)
            path.symlink_to(target)
            self.assertRaises(ValueError, drain.command_binding, path, root, plan, 123)

    def server_fixture(self, root):
        path = root / 'case0-repeat0/command.json'
        path.parent.mkdir()
        plan = dict(image='image', runner_sha256='label', build_root='/tiny/build', memory_cap_bytes=116 << 30)
        lock = observer.read_unique(observer.HERE / 'model-lock.json')
        shards = [observer.ROOT / lock['destination'] / row['path'] for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')]
        command = drain.screen.server_command('b70-cpu-overlap-screen-123-0-0', path.parent, plan, Path(plan['build_root']), shards, Path(drain.screen.__file__).resolve(), None)
        path.write_text(json.dumps(command))
        at = command.index('image')
        mounts = []
        for i, value in enumerate(command):
            if value == '-v':
                source, destination, mode = command[i + 1].rsplit(':', 2)
                mounts.append(dict(Source=source, Destination=destination, RW=mode == 'rw', Type='bind'))
        obj = dict(Name='/b70-cpu-overlap-screen-123-0-0', Image='image',
                   Config=dict(Image='image', Labels={'b70.api-overlap.cpu-screen': 'label'}, Cmd=command[at + 1:], Entrypoint=['/usr/bin/env'], User=command[command.index('--user') + 1], WorkingDir='/results/empty'),
                   HostConfig=dict(NetworkMode='host', Memory=116 << 30, MemorySwap=116 << 30, NanoCpus=8 * 10**9, PidsLimit=256, Devices=[], DeviceRequests=None, Privileged=False, GroupAdd=None),
                   Mounts=mounts, State=dict(Running=True, ExitCode=0, OOMKilled=False))
        return plan, command, obj

    def test_actual_server_command_and_late_owned_container_retired(self):
        import copy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, command, obj = self.server_fixture(root)
            terminal = copy.deepcopy(obj)
            terminal['State']['Running'] = False
            with patch.object(observer, 'inspect', side_effect=[obj, terminal, None]), patch.object(drain.subprocess, 'run', return_value=Mock(returncode=0, stdout='b70-cpu-overlap-screen-123-0-0\n', stderr='')) as run:
                result = drain.census(root, plan, 123)
                self.assertTrue(result['recovery_performed'])
                self.assertEqual(run.call_args_list[1].args[0], ['docker', 'stop', '--time', '10', 'b70-cpu-overlap-screen-123-0-0'])
                self.assertEqual(run.call_args_list[2].args[0], ['docker', 'rm', 'b70-cpu-overlap-screen-123-0-0'])
            bad = copy.deepcopy(obj)
            bad['HostConfig']['Devices'] = [{'PathOnHost': '/dev/dri'}]
            with patch.object(observer, 'inspect', return_value=bad), patch.object(drain.subprocess, 'run', return_value=Mock(returncode=0, stdout='', stderr='')) as run:
                self.assertRaises(ValueError, drain.census, root, plan, 123)
                self.assertEqual(run.call_count, 1)

    def test_unknown_label_census_name_cannot_be_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = dict(runner_sha256='label')
            with patch.object(drain.subprocess, 'run', return_value=Mock(returncode=0, stdout='b70-cpu-overlap-screen-123-7-1\n', stderr='')):
                self.assertRaises(ValueError, drain.census, Path(directory), plan, 123)

    def test_tokenizer_groups_and_image_default_cwd_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(runner_sha256='label', tokenizer_source='/tiny/source', tokenizer_pack='/tiny/tokenizer', tokenizer_image='image', source_bindings={})
            base = dict(Name='/b70-cpu-overlap-tokenizer-123-456', Image='image', Config={'Labels': {'b70.api-overlap.cpu-screen': 'label'}}, State={'Running': False, 'ExitCode': 0, 'OOMKilled': False}, HostConfig={'Devices': [], 'DeviceRequests': None})
            with patch.object(drain.screen.os, 'getpid', return_value=123), patch.object(drain.screen.time, 'time_ns', return_value=456), patch.object(drain.screen, 'inspect', return_value=base), patch.object(drain.screen.subprocess, 'run', return_value=Mock(returncode=0, stdout='', stderr='')), patch.object(drain.screen, 'read', return_value={'tokenizer_file_sha256': {}}):
                drain.screen.metadata_tokenizer(plan, root)
            command, name, image = drain.command_binding(root / 'tokenizer-prepare-command.json', root, plan, 123)
            obj = dict(base)
            at = command.index(image)
            obj['Config'] = dict(Image=image, Labels=base['Config']['Labels'], Cmd=command[at + 1:], Entrypoint=['/usr/bin/env'], User=command[command.index('--user') + 1], WorkingDir='/default')
            obj['HostConfig'] = dict(NetworkMode='none', ReadonlyRootfs=True, Memory=512 << 20, MemorySwap=512 << 20, NanoCpus=10**9, PidsLimit=128, Devices=[], DeviceRequests=None, Privileged=False, GroupAdd=None)
            obj['Mounts'] = []
            for i, value in enumerate(command):
                if value == '-v':
                    source, destination, mode = command[i + 1].rsplit(':', 2)
                    obj['Mounts'].append(dict(Source=source, Destination=destination, RW=mode == 'rw', Type='bind'))
            image_config = dict(Id=image, Config={'WorkingDir': '/default'})
            drain.recipe_gate(obj, command, image, 'label', image_config)
            obj['HostConfig']['GroupAdd'] = ['foreign']
            self.assertRaises(ValueError, drain.recipe_gate, obj, command, image, 'label', image_config)
            obj['HostConfig']['GroupAdd'] = None
            obj['Config']['WorkingDir'] = '/foreign'
            self.assertRaises(ValueError, drain.recipe_gate, obj, command, image, 'label', image_config)


if __name__ == '__main__':
    unittest.main()
