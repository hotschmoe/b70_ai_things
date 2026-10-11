"""Actual finalizer build intake; a missing current GPU0 property receipt refuses."""
import hashlib
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical
import source41_pti_runtime_v1 as original
require=original.require;regular=original.regular;sha=original.sha;inherited_env=original.inherited_env
HERE=Path(__file__).resolve().parent
SOURCE=Path('/mnt/vm_8tb/b70/build/pti-unitrace-model-finalizer-source-v1')
BUILD=Path('/mnt/vm_8tb/b70/build/pti-unitrace-model-finalizer-c388-build-v1')
BUILD_SHA='54a47028897d6fe51e35a8d98a70189fa6f5352a7c2ed80f115a453f54e20d1f'
PROPOSAL=HERE/'pti-finalizer-tool-build-proposal-v1.json';PROPOSAL_SHA='b70e97d3ed4ac953af51080e509aebe2937081703d8445216474b321aedc699a'
TOOL_SHA='d6595250aacaa82e3c330d8e235a493bf07040c9d8a08155d534a3b40fb56b2d'
LIB_SHA='f2b1e3c2048d6afdf03bb197d9c24167abc764725ba892f77b712a262dbf7178'
DEPS=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/pti-model-finalizer-no-device-metadata-v1/receipt.json')
PROPERTY=DEPS.parent.parent/'pti-finalizer-gpu0-clock-property-v1/device-clock-properties.json'
PROPERTY_SHA='d211fd32374fd9293a2d628f515e54cd6adaeabed7f8488b8b7caf0f9aabe02e'
QUERY_SHA='bd20c2db65bd94909b66a5d9d770f396755786b30b1aa4d00a1ebada0f7e8c0b'
OPTIONS=original.OPTIONS
CLOCK_ENV={**original.CLOCK_ENV,'UNITRACE_MODEL_FINALIZE_V1':'1'}
NATIVE_ENV={'STRATA_DIAGNOSTIC_PTI_FINALIZE':'1'}

def properties(value):
    fields=('descriptor_id','timer_frequency_hz','kernel_valid_bits','global_valid_bits','timer_mask','ns_per_cycle_ieee754_u64','uuid_word0_native_u64','uuid_word1_native_u64')
    require(set(value)==set(fields)and all(type(value[k])is int and value[k]>=0 for k in fields),'Typed exact actual clock property roster required')
    require(value['descriptor_id']==0 and value['timer_frequency_hz']>0 and 1<=value['kernel_valid_bits']<=64 and 1<=value['global_valid_bits']<=64 and value['timer_mask']==(1<<value['kernel_valid_bits'])-1,'Actual device timer geometry invalid');return value

def binding(path):
    path=Path(path).resolve();value=read_unique(path);require(value['runtime_ready']is True,'Actual43 PTI device-property intake still pending')
    require(type(value['schema'])is int and value['schema']==1 and value['source_root']==str(SOURCE)and value['install_root']==str(BUILD/'install')and value['metadata_cap_bytes']==512<<20 and type(value['metadata_cap_bytes'])is int,'Exact finalizer source/install/metadata cap required')
    regular(PROPOSAL,PROPOSAL_SHA,8<<20);proposal=read_unique(PROPOSAL)
    for name,want in proposal['expected_patched_source_sha256'].items():regular(SOURCE/name,want,2<<20)
    require(value['build_receipt_path']==str(BUILD/'receipt.json')and value['build_receipt_sha256']==BUILD_SHA,'Exact actual finalizer build receipt required')
    receipt=regular(BUILD/'receipt.json',BUILD_SHA,8<<20);build=read_unique(receipt['path']);require(build['passed']is True and type(build['build_rc'])is int and build['build_rc']==0 and build['error']is None and build['GPU_devices_exposed']is False and build['normal_owned_removed']is True and build['clock_source_root']==str(SOURCE)and build['clock_source_plan_sha256']==PROPOSAL_SHA and build['all_patched_source_bytes_checked_before_after']is True,'Actual finalizer build/source/normal retirement required')
    tool=regular(BUILD/'install/bin/unitrace',TOOL_SHA);injection=regular(BUILD/'install/lib/libunitrace_tool.so',LIB_SHA)
    for row in(tool,injection):
        with Path(row['path']).open('rb')as stream:require(stream.read(4)==b'\x7fELF','Actual profiler ELF required')
    dep=regular(value['dependency_receipt_path'],value['dependency_receipt_sha256'],16<<20);require(Path(dep['path'])==DEPS,'Exact actual NoDri finalizer dependency receipt required');deps=read_unique(DEPS)
    require(deps['passed']is True and deps['return_code']==0 and type(deps['return_code'])is int and deps['error']is None and deps['GPU_devices_exposed']is False and deps['normal_owned_removed']is True and deps['image']==build['image']and deps['unitrace_sha256']==TOOL_SHA and deps['injection_sha256']==LIB_SHA and len(deps['files'])==9,'Actual finalizer same-image NoDri DSO census required')
    exports=[]
    for row in deps['files']:
        require(type(row['original_container_path'])is str and row['original_container_path'].startswith('/')and type(row['exported_host_path'])is str,'Original container and exported host paths required');exports.append(regular(row['exported_host_path'],row['sha256']))
    require(Path(value['actual_device_clock_metadata_path'])==PROPERTY and value['actual_device_clock_metadata_sha256']==PROPERTY_SHA,'Exact current actual GPU0 property receipt required');device=regular(PROPERTY,PROPERTY_SHA,8<<20);record=read_unique(device['path']);property_root=Path(device['path']).parent
    require(record['passed']is True and record['ZE_AFFINITY_MASK']=='0'and record['actual_API_query_only']is True and record['model_executed']is False and record['operation_counts_transferred']is False and record['clock_calibration_qualified']is False,'Actual API-only GPU0 clock property scope required')
    require(record['receipt_sha256']==QUERY_SHA,'Exact actual query parent receipt required');query=regular(property_root/'receipt.json',QUERY_SHA,8<<20);query_body=read_unique(query['path']);require(query_body['passed']is True and type(query_body['return_code'])is int and query_body['return_code']==0 and query_body['clock_build_receipt_sha256']==BUILD_SHA and query_body['clock_source_plan_sha256']==PROPOSAL_SHA and query_body['normal_owned_removed']is True and query_body['helper_or_model_executed']is False and query_body['ZE_AFFINITY_MASK']=='0'and query_body['pre_health_rc']==query_body['post_health_rc']==0,'Actual same-finalizer clock query/owned health receipt required')
    regular(property_root/'b70_root_pti_finalizer_gpu0_clock_property_v1.py',record['source_script_sha256'],1<<20);clock_raw=regular(property_root/'unitrace/clock.1.jsonl',record['raw_clock_sha256'],8<<20);require(query_body['raw_files']['clock.1.jsonl']['sha256']==clock_raw['sha256']and query_body['raw_files']['clock.1.jsonl']['bytes']==clock_raw['bytes'],'Original actual property raw clock receipt changed')
    import json
    rows=[json.loads(line)for line in Path(clock_raw['path']).read_text().splitlines()];device_rows=[row for row in rows if row['kind']=='device_origin'];require(len(device_rows)==1,'Unique actual masked GPU0 clock descriptor required')
    clock=properties(record['clock_properties']);require(all(device_rows[0][k]==v and type(device_rows[0][k])is int for k,v in clock.items())and canonical(clock)==canonical(value['actual_device_clock_properties']),'Actual current clock raw/property/intake differs')
    return {'intake_path':str(path),'intake_sha256':sha(path),'source_root':str(SOURCE),'source_plan_sha256':PROPOSAL_SHA,'build':receipt,'tool':tool,'injection':injection,'dependencies':dep,'exported_dependencies':exports,'device_metadata':device,'clock_properties':clock,'install_root':str(BUILD/'install'),'image':build['image'],'fresh512MiB_clock_budget':True,'explicit_model_finalizer_enabled':True}

def intake(device):
    device=Path(device).resolve();record=read_unique(device)
    return {'schema':1,'runtime_ready':True,'source_root':str(SOURCE),'install_root':str(BUILD/'install'),'metadata_cap_bytes':512<<20,'build_receipt_path':str(BUILD/'receipt.json'),'build_receipt_sha256':BUILD_SHA,'dependency_receipt_path':str(DEPS),'dependency_receipt_sha256':sha(DEPS),'actual_device_clock_metadata_path':str(device),'actual_device_clock_metadata_sha256':sha(device),'actual_device_clock_properties':properties(record['clock_properties'])}
