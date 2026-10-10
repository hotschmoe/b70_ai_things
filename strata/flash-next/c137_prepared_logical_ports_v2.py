# NEW mandatory immutable logical-byte port of c137_prepared_pack49_v1.py
from upload_logical_port_identity_v2 import original_producer_file, original_module_file
'Proposed explicit SDK+pack prepared admission port; unintegrated.'
from c137_prepared_pack48_v1 import *
from c137_sdk37_logical_ports_v2 import combined_generation_gate, upload_gate
from sdk37_witness_scope_v1 import require_prepared

def validate_prepared(directory, *, pack_epoch, sdk_epoch, logical_epoch, logical_roster):
    require_prepared(sdk_epoch, Path(directory) / 'prepared.json')
    m = read(directory / 'prepared.json')
    require(m.get('combined_generation') == combined_generation_gate(Path(m['engine_receipt']).parent, sdk_epoch=sdk_epoch), 'Combined generation source chain changed')
    require(m.get('runtime_python_source_count') == 6 and read(directory / 'artifact-identity.json')['runtime']['python_sources'] == PYTHON_SOURCE_SHA, 'Manifest must bind actual6 runtime Python files')
    require(m['launch_allowed'], 'Preparation has unresolved runtime/model/upload lifecycle gates')
    require(m.get('trace_contract_sha256') == sha(REPO / 'strata/flash-next/c1_trace_contract.py'), 'C1 completion contract changed or older prepared generation')
    require(m['controller_sha256'] == sha(Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/c1_serve_controller_combined_v137.py')) and m['trace_sha256'] == sha(REPO / 'strata/flash-next/c1_api_trace.py'), 'Qualification controller/tracer changed')
    require(sha(m['engine_receipt']) == m['engine_receipt_sha256'] and sdk_epoch.digest_for('SDK37:strata', m['executable'], m['executable_sha256']) == m['executable_sha256'], 'Engine build generation changed')
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
    upload_gate(upload['path'], upload['oracle'], m['engine_receipt'], m['pack_receipt'], sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    cfg = read(directory / 'server-config.json')
    baseline_profile_gate(cfg['args'], cfg['env'])
    segmented_profile_gate(PROFILES[m['profile']], read(m['engine_receipt']), cfg['args'], cfg['env'])
    return m
