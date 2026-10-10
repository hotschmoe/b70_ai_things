"""Tiny sixteen-case source-shaped identity/decode controls, no model runtime."""
import copy
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
import finite_cpu_identity_v3 as gate


class Controls(unittest.TestCase):
    def fixture(self, root):
        def put(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        source = root / 'source'
        source.write_text('synthetic source receipt')
        binary = root / 'tiny-binary'
        binary.write_bytes(b'Tiny CPU identity only')
        interpreter = root / 'tiny-interpreter'
        interpreter.write_bytes(b'Tiny interpreter identity only')
        sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        build = root / 'build'
        elf = {'path': str(binary), 'sha256': sha(binary), 'dependencies': [{'path': '/tiny/library', 'sha256': 'CPU-test-only', 'bytes': 1}]}
        for name, value in [('receipt.json', {'passed': False}), ('elf-closure-v1.json', {'ELFs': {'llama-server': elf}}),
                            ('host-loader-failure-binding-v1.json', {}), ('cpu-runtime-smoke-v1/receipt.json', {})]:
            put(build / name, value)
        put(root / 'strata/flash-next/model-lock.json', {'files': [{'path': 'UD-Q4_K_XL/shard' + str(i)} for i in range(4)]})
        template = 'tiny original template'
        plan = {'build_root': str(build), 'source_receipt': str(source), 'model_root': '/tiny/model',
                'research_alias': 'source-only-model-alias', 'server_argv': ['--n-gpu-layers', '0'],
                'screen_port': 28671, 'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest()}
        inputs = [dict(messages=[{'role': 'user', 'content': 'tiny ' + str(i)}], rendered='rendered ' + str(i), ids=[10, i]) for i in range(8)]
        fixture = {'fixtures': {'warm': inputs[:2], 'target': inputs[2:]}, 'tokenizer_file_sha256': {'tiny': 'CPU-test-only'}}
        report = {'producer_pid': 123, 'producer_interpreter': str(interpreter), 'producer_interpreter_sha256': sha(interpreter),
                  'producer_argv': [str(interpreter), 'source-only-runner'], 'cases': [],
                  'build_binding': {'original_build_receipt_passed': False, 'receipt_sha256': sha(build / 'receipt.json'),
                                    'supplement_sha256': sha(build / 'elf-closure-v1.json'), 'source_receipt_sha256': sha(source),
                                    'supplemental_all_six_passed': True, 'host_loader_failure_sidecar_sha256': sha(build / 'host-loader-failure-binding-v1.json'),
                                    'runtime_smoke_receipt_sha256': sha(build / 'cpu-runtime-smoke-v1/receipt.json')}}
        for i in range(16):
            case, repeat = divmod(i, 2)
            response = {'tokens': [20, 21], 'content': 'tiny response ' + str(case)}
            report['cases'].append(dict(case=case, repeat=repeat, response=response))
            directory = root / ('case%d-repeat%d' % (case, repeat))
            original = inputs[case]
            put(directory / 'models.json', {'data': [{'id': 'hotschmoe-dd', 'aliases': ['hotschmoe-dd', plan['research_alias']]}]})
            put(directory / 'props.json', {'default_generation_settings': {'n_ctx': 2048}, 'total_slots': 1, 'model_alias': 'hotschmoe-dd', 'model_path': '/tiny/model/UD-Q4_K_XL/shard0', 'chat_template': template})
            put(directory / 'template-request.json', {'model': 'hotschmoe-dd', 'messages': original['messages'], 'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_format': 'none'})
            put(directory / 'rendered.json', {'prompt': original['rendered']})
            put(directory / 'tokenize-request.json', {'content': original['rendered'], 'add_special': True, 'parse_special': True})
            put(directory / 'accepted-input-ids.json', original['ids'])
            put(directory / 'completion-request.json', {'model': 'hotschmoe-dd', 'prompt': original['ids'], 'cache_prompt': False, 'return_tokens': True, 'stream': False, 'temperature': 0, 'seed': 1234, 'n_predict': 64, 'repeat_penalty': 1, 'samplers': ['temperature']})
            put(directory / 'launch.json', {'binary': str(binary), 'ELF': elf, 'argv': ['-m', '/tiny/model/UD-Q4_K_XL/shard0', *plan['server_argv'], '--port', '28671']})
            for name in ('runtime-binding.json', 'runtime-post-binding.json'):
                put(directory / name, {'binary_sha256': elf['sha256'], 'dependencies': elf['dependencies'], 'cwd_empty': True, 'actual_GPU_touch': False, 'environment': {'OMP_NUM_THREADS': '8'}})
        requests = [dict(case=r['case'], repeat=r['repeat'], tokens=r['response']['tokens'], content=r['response']['content']) for r in report['cases']]
        put(root / 'output-id-requests.json', requests)
        decoded = {'fixtures': inputs, 'tokenizer_file_sha256': fixture['tokenizer_file_sha256'], 'decoded_outputs': [dict(case=r['case'], repeat=r['repeat'], text=r['content'], matches=True) for r in requests]}
        put(root / 'decoded-output.json', decoded)
        put(root / 'tokenizer-fixtures.json', {'fixtures': inputs, 'tokenizer_file_sha256': fixture['tokenizer_file_sha256']})
        return plan, report, fixture, types.SimpleNamespace(ROOT=root, sha=sha)

    def test_all16_decode_and_runtime_identity_join(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, report, fixture, runner = self.fixture(root)
            self.assertEqual(gate.admit(root, report, plan, fixture, runner)['actual_independent_decode_response_count'], 16)

    def test_hashed_files_still_require_original_predicates(self):
        for name in ('models.json', 'props.json', 'template-request.json', 'rendered.json', 'tokenize-request.json',
                     'completion-request.json', 'launch.json', 'runtime-binding.json', 'runtime-post-binding.json'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan, report, fixture, runner = self.fixture(root)
                path = root / 'case7-repeat1' / name
                path.write_text('{}')
                with self.assertRaises((ValueError, KeyError)):
                    gate.admit(root, report, plan, fixture, runner)

    def test_response_decode_and_build_interpreter_mutations_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, report, fixture, runner = self.fixture(root)
            bad = copy.deepcopy(report)
            bad['cases'][-1]['response']['tokens'] = [999]
            self.assertRaises(ValueError, gate.admit, root, bad, plan, fixture, runner)
            bad = copy.deepcopy(report)
            bad['build_binding']['supplement_sha256'] = 'wrong'
            self.assertRaises(ValueError, gate.admit, root, bad, plan, fixture, runner)
            bad = copy.deepcopy(report)
            bad['producer_interpreter_sha256'] = 'wrong'
            self.assertRaises(ValueError, gate.admit, root, bad, plan, fixture, runner)
            path = root / 'decoded-output.json'
            decoded = json.loads(path.read_text())
            decoded['decoded_outputs'][-1]['matches'] = 1
            path.write_text(json.dumps(decoded))
            self.assertRaises(ValueError, gate.admit, root, report, plan, fixture, runner)
