# NEW mandatory immutable logical-byte port of serial_manifest_pack49_v1.py
from upload_logical_port_identity_v1 import original_producer_file, original_module_file
from batch_serial_source37_v3 import *
import batch40_manifest_logical_ports_v1 as origin
from serial_selection_logical_ports_v1 import selected_jobs

def manifest_binding(plan, *, pack_epoch, sdk_epoch, logical_epoch, logical_roster):
    require(plan['slots'] == 2, 'Only actual historical paired2 corpus is a prerequisite; no other source scope transfer')
    require(plan.get('serial_source37_generation') == 3 and plan['kind'] == 'serial' and (plan['driver_sha256'] == sha(Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/batch_serial_source37_v3.py'))), 'Explicit new source37 serial controller required')
    for name, want in read(SOURCE_PLAN)['files'].items():
        require(sha(ROOT / name) == want, 'Frozen serial successor dependency changed ' + name)
    old = plan['harness40_source_preparation']
    binding = origin.manifest_binding(old, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    require(old['kind'] == 'serial' and old['lane'] == 'source37', 'Genuine harness40 source37 serial preparation required')
    for key, value in old.items():
        if key != 'driver_sha256':
            require(plan[key] == value, 'Serial successor changed original preparation ' + key)
    reused = plan['reused_V40_origin_plan']
    if reused is not None:
        require(sha(reused['path']) == reused['sha256'] and canonical(read_unique(reused['path'])) == canonical(old), 'Preserved actual V40 origin file changed')
    serial.config(plan)
    selected_jobs(plan, read(Path(plan['batch_parent']) / 'child/serial-jobs.json')['jobs'], pack_epoch=pack_epoch, sdk_epoch=sdk_epoch, logical_epoch=logical_epoch, logical_roster=logical_roster)
    return binding
