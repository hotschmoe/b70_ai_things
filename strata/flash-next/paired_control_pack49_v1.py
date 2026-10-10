# NEW explicit pack epoch admission port of paired_source37_private_v4_control_v1.py
"""NEW explicit paired2 privateV4 numerical prerequisite before requested4/6.
No saved PASS transfer; actual current reader/core admissions rerun every call.
"""
from pathlib import Path
from batch_proofs_pack49_v1 import require, sha, read
HERE = Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/paired_source37_private_v4_control_v1.py').resolve().parent
READER_SHA = 'd3837b2e9b8df6836f390e346d50aa6226a90d29f26e6d0660269bc4deecd6ca'
READER_PLAN_SHA = '28e9041801d23e506d98b5369491907c1792eb5e06a780c97dee637b8429fc40'
REPORT_SHA = 'c05d9416ac7b92a78f9d770283c22dd118017a4ef6281c2740b46eb6eb4bc9f1'
KEYS = {'schema', 'reader_sha256', 'reader_source_plan_sha256', 'off_root', 'on_root', 'serial_roster_sha256', 'private_report_path', 'private_report_sha256'}

def finalized_binding(control, engine_sha256, *, pack_epoch, sdk_epoch):
    import private_reader_pack49_v1 as reader
    require(set(control) == KEYS and type(control['schema']) is int and (control['schema'] == 4), 'Exact NEW privateV4 prerequisite declaration required')
    require(control['reader_sha256'] == READER_SHA == sha(Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/validate_private_native_offon_source37_v4.py')) and control['reader_source_plan_sha256'] == READER_PLAN_SHA == sha(HERE / 'private-native-offon-source37-source-plan-v4.json'), 'Frozen exact privateV4 helper/closure changed')
    off = Path(control['off_root']).resolve()
    on = Path(control['on_root']).resolve()
    roster = reader.serial_roster_path(on)
    require(sha(roster) == control['serial_roster_sha256'], 'Actual new canonical privateV4 serial roster changed')
    from serial37_canonical_json_v3 import canonical, read_unique
    report = Path(control['private_report_path']).resolve()
    require(control['private_report_sha256'] == REPORT_SHA == sha(report), 'Exact closed actual privateV4 report required; no pending or foreign control')
    actual = reader.finalized_binding(off, on, pack_epoch=pack_epoch, sdk_epoch=sdk_epoch)
    require(canonical(actual) == canonical(read_unique(report)), 'Every current privateV4 proof field must match actual closed report')
    serial = actual['actual_complete_serial49']
    require(actual['passed'] is True and actual['schema'] == 4 and (actual['harness_generation'] == 40) and (actual['source_lane'] == 'source37') and (actual['cards'] == [0, 1]) and (actual['engine_receipt_sha256'] == engine_sha256), 'Genuine same source37 SDK paired2 proof required')
    require(actual['actual_ON_complete_raw49'] == {'vectors': 196, 'selected_jobs': 4} and len(serial['jobs']) == 4 and (len(serial['comparisons']['comparisons']) == 196) and (serial['comparisons']['passed'] is True) and (serial['actual_complete_serial49_qualified'] is True), 'Actual paired2 complete196 prerequisite missing')
    require(serial['original_failed_first_parent_passed'] is False and serial['historical_first_PIN0_exception_used'] is True and (serial['full_group_absent_PIN_scope_qualified'] is False) and (serial['actual_absent_PIN_jobs'] == 3) and (serial['historical_supervisor_EOF_producer_field_observed'] is False), 'Historical49/proper147 qualification boundary changed')
    return {'schema': 4, 'engine_receipt_sha256': engine_sha256, 'off_root': str(off), 'on_root': str(on), 'private_reader_sha256': READER_SHA, 'private_reader_source_plan_sha256': READER_PLAN_SHA, 'actual_private_off_on_binding': actual, 'actual_all49_serial': serial['comparisons'], 'actual_source40_native2_and_parent43_serial_prerequisite': True, 'full_group_absent_PIN_scope_qualified': False, 'full_model_math_qualified': False, 'latency_qualified': False}
