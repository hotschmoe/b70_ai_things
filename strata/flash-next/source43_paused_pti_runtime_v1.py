"""Fresh paused-list fixture is the runtime authority; software options only reused."""
from pathlib import Path
import source43_sampled_pti_runtime_v1 as software
from serial37_canonical_json_v3 import canonical
require=software.require
OPTIONS=software.OPTIONS
CLOCK_ENV={**software.CLOCK_ENV,'UNITRACE_RETAIN_PAUSED_REGULAR_LISTS':'1'}
NATIVE_ENV=software.NATIVE_ENV
inherited_env=software.inherited_env
BUILD=Path('/mnt/vm_8tb/b70/build/pti-unitrace-paused-regular-c388-build-v1')
TOOL_SHA='9f481a37a2884f7c7d499f33b5f41dbc00e2c0d089a9e786d059881803417b0c'
LIB_SHA='c72c6b4f42d8fe1a2c352286f5099280038b2136843924489d84a01887f43c4b'
SOURCE_SHA='42e4e3de36b3b376a254f46e0bc91648d9e4a64705421cd544e7f8ed68cd6c14'

def binding(path,expected_intake_sha256):
    from validate_pti_paused_regular_fixture_v2 import finalized_binding
    path=Path(path)
    require(path.is_absolute()and path.name=='report.json'and path.parent.resolve()==path.parent,'Original successful fixture report path required')
    proof=finalized_binding(path.parent,expected_intake_sha256)
    require(proof['fixture_success_revalidated']is True and proof['actual_retention_OFF_ON_discriminated']is True,'Actual loaded paused-list fixture prerequisite required')
    result=proof['runtime_envelope']
    require(result['intake_path']==str(path)and result['exact_intake_sha256']==expected_intake_sha256 and result['tool']['sha256']==TOOL_SHA and result['injection']['sha256']==LIB_SHA,'Exact fresh tool/fixture identities required')
    require(result['install_root']==str(BUILD/'install')and result['source_plan_sha256']==SOURCE_SHA and result['old_dependency_receipt_transferred']is False and result['paused_regular_list_retention_required']is True,'Fresh source/runtime ownership required; no old dependency receipt transfer')
    require(type(result['query_sample_every'])is int and result['query_sample_every']==512 and result['full_model_coverage_qualified']is False and result['clock_alignment_qualified']is False,'Fixture counts or alignment cannot become model authority')
    return result
