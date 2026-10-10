"""Reexecute saved model/template/runtime and all16 independent decode joins."""
from pathlib import Path
from serial37_canonical_json_v3 import canonical, read_unique


def require(ok, message):
    if not ok:
        raise ValueError(message)


def admit(root, report, plan, fixture, runner):
    root = Path(root)
    rows = fixture['fixtures']['warm'] + fixture['fixtures']['target']
    require(len(rows) == 8 and len(report['cases']) == 16, 'Complete actual corpus/response roster required')
    output_requests = [{'case': row['case'], 'repeat': row['repeat'],
                        'tokens': row['response']['tokens'], 'content': row['response']['content']}
                       for row in report['cases']]
    require(canonical(read_unique(root / 'output-id-requests.json')) == canonical(output_requests),
            'Actual independent decode input must equal all16 original responses')
    decoded = read_unique(root / 'decoded-output.json')
    expected_decoded = [{'case': row['case'], 'repeat': row['repeat'], 'text': row['content'], 'matches': True}
                        for row in output_requests]
    require(canonical(decoded['decoded_outputs']) == canonical(expected_decoded),
            'Actual independently decoded output IDs/text must match all16 responses')
    rendered = [{k: row[k] for k in ('messages', 'rendered', 'ids')} for row in rows]
    require(canonical(decoded['fixtures']) == canonical(rendered)
            and canonical(read_unique(root / 'tokenizer-fixtures.json')['fixtures']) == canonical(rendered),
            'Independent tokenizer/template corpus changed')
    for name in ('tokenizer-fixtures.json', 'decoded-output.json'):
        data = read_unique(root / name)
        require(data['tokenizer_file_sha256'] == fixture['tokenizer_file_sha256'], 'Actual decode tokenizer bytes differ')
    supplement = read_unique(Path(plan['build_root']) / 'elf-closure-v1.json')
    build = read_unique(Path(plan['build_root']) / 'receipt.json')
    expected_build = {'original_build_receipt_passed': build['passed'],
                      'receipt_sha256': runner.sha(Path(plan['build_root']) / 'receipt.json'),
                      'supplement_sha256': runner.sha(Path(plan['build_root']) / 'elf-closure-v1.json'),
                      'source_receipt_sha256': runner.sha(plan['source_receipt']),
                      'supplemental_all_six_passed': True,
                      'host_loader_failure_sidecar_sha256': runner.sha(Path(plan['build_root']) / 'host-loader-failure-binding-v1.json'),
                      'runtime_smoke_receipt_sha256': runner.sha(Path(plan['build_root']) / 'cpu-runtime-smoke-v1/receipt.json')}
    require(canonical(report['build_binding']) == canonical(expected_build), 'Actual original build/supplement/smoke/source report binding changed')
    require(type(report['producer_pid']) is int and report['producer_pid'] > 0
            and runner.sha(report['producer_interpreter']) == report['producer_interpreter_sha256']
            and report['producer_argv'][0] == report['producer_interpreter'], 'Actual original producer interpreter/PID identity changed')
    elf = supplement['ELFs']['llama-server']
    require(runner.sha(elf['path']) == elf['sha256'], 'Actual qualified CPU binary changed')
    lock = read_unique(runner.ROOT / 'strata/flash-next/model-lock.json')
    shards = [Path(plan['model_root']) / row['path'] for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')]
    require(len(shards) == 4, 'Exact original model shard roster required')
    for row in report['cases']:
        case, repeat = row['case'], row['repeat']
        directory = root / ('case' + str(case) + '-repeat' + str(repeat))
        original = rows[case]
        models = read_unique(directory / 'models.json')
        require(len(models.get('data', [])) == 1 and models['data'][0]['id'] == 'hotschmoe-dd'
                and set(models['data'][0]['aliases']) == {'hotschmoe-dd', plan['research_alias']},
                'Actual saved API primary/research model identity changed')
        props = read_unique(directory / 'props.json')
        require(props['default_generation_settings']['n_ctx'] == 2048 and props['total_slots'] == 1
                and props['model_alias'] == 'hotschmoe-dd' and props['model_path'] == str(shards[0]),
                'Actual saved context/model/slot properties changed')
        import hashlib
        require(hashlib.sha256(props['chat_template'].encode()).hexdigest() == plan['chat_template_sha256'],
                'Actual saved GGUF template differs from original export')
        body = {'model': 'hotschmoe-dd', 'messages': original['messages'],
                'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_format': 'none'}
        require(canonical(read_unique(directory / 'template-request.json')) == canonical(body)
                and read_unique(directory / 'rendered.json') == {'prompt': original['rendered']},
                'Actual saved template request/rendered result changed')
        require(read_unique(directory / 'tokenize-request.json') == {'content': original['rendered'], 'add_special': True, 'parse_special': True}
                and read_unique(directory / 'accepted-input-ids.json') == original['ids'], 'Actual saved token request/accepted IDs changed')
        expected_request = {'model': 'hotschmoe-dd', 'prompt': original['ids'], 'cache_prompt': False,
                            'return_tokens': True, 'stream': False, 'temperature': 0, 'seed': 1234,
                            'n_predict': 64, 'repeat_penalty': 1, 'samplers': ['temperature']}
        require(canonical(read_unique(directory / 'completion-request.json')) == canonical(expected_request),
                'Actual saved sampling request changed')
        launch = {'binary': elf['path'], 'ELF': elf, 'argv': ['-m', str(shards[0]), *plan['server_argv'], '--port', str(plan['screen_port'])]}
        require(canonical(read_unique(directory / 'launch.json')) == canonical(launch), 'Actual CPU ELF/argv launch changed')
        for name in ('runtime-binding.json', 'runtime-post-binding.json'):
            runtime = read_unique(directory / name)
            require(runtime['binary_sha256'] == elf['sha256'] and canonical(runtime['dependencies']) == canonical(sorted(elf['dependencies'], key=lambda x: x['path']))
                    and runtime['cwd_empty'] is True and runtime['actual_GPU_touch'] is False,
                    'Actual saved intended-runtime ELF/dependencies/scope changed')
            require(runtime['environment'].get('OMP_NUM_THREADS') == '8'
                    and not any(k.startswith(('GGML_', 'LLAMA_ARG_', 'ONEAPI_', 'ZE_', 'SYCL_')) for k in runtime['environment']),
                    'Actual saved CPU runtime environment differs')
    return {'actual_saved_model_template_runtime_joins_reexecuted': True,
            'actual_independent_decode_response_count': 16,
            'original_response_or_model_proof_transferred': False}
