"""Explicit current C137 prepared gate; original producer identity retained."""
from c1_serve_controller_combined_v137 import *
from operation_pack_hash_witness_v2 import consume_prepared_pack
def validate_prepared(directory, *, pack_epoch=None):
    m = read(directory / 'prepared.json')
    require(m.get('combined_generation') == combined_generation_gate(Path(m['engine_receipt']).parent), 'Combined generation source chain changed')
    require(m.get('runtime_python_source_count') == 6 and read(directory / 'artifact-identity.json')['runtime']['python_sources'] == PYTHON_SOURCE_SHA, 'Manifest must bind actual6 runtime Python files')
    require(m['launch_allowed'], 'Preparation has unresolved runtime/model/upload lifecycle gates')
    require(m.get('trace_contract_sha256') == sha(REPO / 'strata/flash-next/c1_trace_contract.py'), 'C1 completion contract changed or older prepared generation')
    require(m['controller_sha256'] == sha(Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/c1_serve_controller_combined_v137.py')) and m['trace_sha256'] == sha(REPO / 'strata/flash-next/c1_api_trace.py'), 'Qualification controller/tracer changed')
    require(sha(m['engine_receipt']) == m['engine_receipt_sha256'] and sha(m['executable']) == m['executable_sha256'], 'Engine build generation changed')
    require(sha(m['pack_receipt']) == m['pack_receipt_sha256'], 'Pack receipt changed')
    require(sha(directory / 'artifact-identity.json') == m['artifact_manifest_sha256'] and sha(directory / 'server-config.json') == m['config_sha256'], 'Launch configuration changed')
    require(sha(m['runtime_receipt']) == m['runtime_receipt_sha256'], 'Runtime receipt changed')
    require(sha(m['runtime']['packages_path']) == m['runtime']['packages_sha256'], 'Runtime package/library census changed')
    for row in m['model_shards']:
        require(row['verified_sha256'] == row['sha256'] and stat_signature(row['path']) == row['stat'], 'Selected model was not fully hashed or changed since hashing')
    require(original_page_sentinel(m['model_shards']) == m['source_page_sentinel'], 'Original page sentinel contract changed')
    for name, expected in read(m['engine_receipt'])['patched_source_sha256'].items():
        require(sha(Path(m['source']) / name) == expected, 'Frozen source changed: ' + name)
    consume_prepared_pack(m, pack_epoch)
    upload = m['upload_lifecycle']
    require(sha(upload['path']) == upload['sha256'] and sha(upload['oracle']) == upload['oracle_sha256'], 'Upload lifecycle evidence changed')
    require(upload['logical_free_parser_sha256'] == sha(REPO / 'strata/flash-next/parse_usm_logical_free_trace.py'), 'Logical-free evidence parser changed')
    upload_gate(upload['path'], upload['oracle'], m['engine_receipt'], m['pack_receipt'])
    cfg = read(directory / 'server-config.json')
    baseline_profile_gate(cfg['args'], cfg['env'])
    segmented_profile_gate(PROFILES[m['profile']], read(m['engine_receipt']), cfg['args'], cfg['env'])
    return m
