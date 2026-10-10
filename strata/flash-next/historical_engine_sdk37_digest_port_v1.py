"""Proposed explicit SDK37 hash/magic port, all other gates retained."""
from batch_proofs_pack48_v1 import *
from c137_sdk37_digest_ports_v1 import combined_generation_gate
from sdk37_witness_scope_v1 import require_engine
def engine_binding(engine, lane='source37', *, sdk_epoch):
    require_engine(sdk_epoch, engine)
    spec = lane_contract(lane)
    source = read(ROOT / spec['plan'])
    require(sha(ROOT / spec['plan']) == spec['sha256'], 'Explicit lane SDK source plan changed')
    r = read(engine / 'receipt.json')
    require(r.get('build_rc') == 0 and r.get('external_source_unchanged') is True and (r.get('plan_snapshot_unchanged') is True) and (r.get('plan_sha256') == spec['sha256']) and (r['image'] == source['image']), 'Actual matching whole SDK absent; foreign source lane cannot transfer')
    require(r['patched_source_sha256'] == source['expected_patched_source_sha256'] and len(r['patched_source_sha256']) == spec['source_files'], 'Actual complete source ledger differs')
    require(len(r['patches']) == spec['patches'] and [(Path(p['path']).name, p['sha256']) for p in r['patches']] == [(Path(p['path']).name, p['sha256']) for p in source['patches']], 'Actual complete ordered patch chain differs')
    for name, digest in source['expected_patched_source_sha256'].items():
        require(sha(engine / 'source' / name) == digest, 'Consumed source changed ' + name)
    require(len(source['added_header_payloads']) == spec['headers'], 'Actual complete header payload roster differs')
    for item in source['added_header_payloads']:
        require(item['sha256'] == source['expected_patched_source_sha256'][item['path']], 'Header payload contradicts final source')
    for target in source['build_targets']:
        binary = engine / 'build' / target
        require(binary.is_file() and r['binary_sha256'].get(str(binary)) == sdk_epoch.digest_for('SDK37:' + target, binary, r['binary_sha256'][str(binary)]), 'Actual ABI target missing ' + target)
        require(sdk_epoch.ELF_magic('SDK37:' + target, binary, r['binary_sha256'][str(binary)]) == b'\x7fELF', 'Mock file is not actual rebuilt ELF')
    require(len(source['build_targets']) == 8 and 'icpx --version' in r['container_script'], 'Actual full SDK command evidence absent')
    c1, _ = providers(lane)
    combined_generation_gate(engine, sdk_epoch=sdk_epoch)
    return r
