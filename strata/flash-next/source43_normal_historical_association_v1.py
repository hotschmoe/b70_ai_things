"""Named immutable normal43 evidence; no current old-reader PASS cache."""
import hashlib,stat,json
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical,unique_object,finite_constant
import source43_api_prerequisite_bytes_v1 as history
require=history.require
HERE=Path(__file__).resolve().parent
F17=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')
API_PREPARED=F17/'source43-api-mode1-baseline-v3-prepared-v1/prepared.json'
API_PREPARED_SHA='562a2a25bdc4522f377909ac37d29f5e2de4f021ea0712e20e263a727f3d4298'
NORMAL=F17/'source43-normal-cli-v1-run-v1'
REPORT_SHA='96624d15371daa621eb5e00dae2dc8fd2400ab2b63a6390e72af82fda44398b0'
PUBLIC=F17/'source43-normal-cli-v1-root-public-binding-v1.json'
PUBLIC_SHA='899efe224ae62672e50b2c1cb3857eb93941deafc97eb78c0714be7d643ce646'
PRODUCER_SOURCE_SHA='d17fcce3574d2ee81124555f83f42cd2f7f20b16e860bca970cc8264b2a21947'
MAX_FILES=2048;MAX_BYTES=16<<20
CONFIG_SHA='d3984406369a4bd0db4f82faa349b51abf502373b11cb3b4c71103825434f9f0'
def parse(raw):return json.loads(raw,object_pairs_hook=unique_object,parse_constant=finite_constant)

PLAN=HERE/'source43-normal-historical-association-source-plan-v1.json'
def source_binding():
    raw=PLAN.read_bytes();plan=parse(raw)
    for name,want in plan['files'].items():
        file=Path('/mnt/vm_8tb/github/b70_ai_things')/name
        require(hashlib.sha256(file.read_bytes()).hexdigest()==want,'Historical association source changed '+name)
    require(PLAN.read_bytes()==raw,'Historical association plan changed');return hashlib.sha256(raw).hexdigest()

def signature(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def bound(path,want=None,cap=1<<20):
    path=Path(path);require(path.is_absolute()and path.resolve()==path and not any(p.is_symlink()for p in(path,*path.parents)),'Canonical nonaliased historical input required');before=path.stat();require(stat.S_ISREG(before.st_mode)and before.st_size<=cap,'Bounded original historical file required');raw=path.read_bytes();after=path.stat();require(signature(before)==signature(after)and len(raw)==after.st_size,'Historical file changed during consumed bytes');digest=hashlib.sha256(raw).hexdigest();require(want is None or digest==want,'Historical evidence SHA changed');return raw,dict(path=str(path),sha256=digest,bytes=len(raw),stat=signature(after))

def tree_binding(root,expected):
    """Recheck precisely the actual saved byte roster, with stat/reread brackets."""
    root=Path(root);require(root.is_absolute()and root.resolve()==root and root.is_dir()and not any(p.is_symlink()for p in(root,*root.parents)),'Canonical historical evidence root required');require(type(expected)is dict and 0<len(expected)<=MAX_FILES,'Exact bounded historical roster required')
    require(not any(p.is_symlink()for p in root.rglob('*')),'Historical child aliases refused');actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()};require(actual==set(expected),'Original complete historical evidence roster changed')
    require(all(type(r['bytes'])is int and r['bytes']>=0 for r in expected.values())and sum(r['bytes']for r in expected.values())<=MAX_BYTES,'Historical total raw byte bound required')
    bindings={}
    for name,row in sorted(expected.items()):
        require(type(name)is str and not Path(name).is_absolute()and '..'not in Path(name).parts,'Relative original evidence path required');_,fresh=bound(root/name,row['sha256'],MAX_BYTES);require(fresh['bytes']==row['bytes'],'Original historical extent changed');bindings[name]=fresh
    require({str(p.relative_to(root))for p in root.rglob('*')if p.is_file()}==actual and not any(p.is_symlink()for p in root.rglob('*')),'Historical roster changed across read')
    return bindings

def envelope(record,public,report):
    require(record['role']=='normal43_mode0'and record['root']==str(NORMAL)and record['report_sha256']==REPORT_SHA and canonical(record['original_public_admission'])==canonical(public),'Exact named genuine normal43 association required')
    require(public['passed']is True and public['report_sha256']==REPORT_SHA and public['root']==str(NORMAL)and public['source_plan_sha256']==PRODUCER_SOURCE_SHA and type(public['source_generation'])is int and public['source_generation']==43 and type(public['materialized_mode'])is int and public['materialized_mode']==0 and public['OFF_equivalence_passed']is True and public['normal43_fresh_constructor_and_bounded_outputs_observed']is True,'Closed historical constructor/output proof required')
    require(report['passed']is True and report['errors']==[]and report['phase']=='off-equivalence'and report['source_plan_sha256']==PRODUCER_SOURCE_SHA and record['byte_roster']['report.json']['sha256']==REPORT_SHA,'Original closed successful normal report required')
    require(record['old_current_registry_gate_passed']is False and record['historical_completed43_source_and_lifecycle_only']is True and record['fresh_API_or_original_math_authority_transferred']is False,'Original historical false-scope envelope changed')
    return dict(schema=1,role='named immutable normal43 mode0 constructor/output history',original_public_admission=public,original_historical_record=record,current_original_normal_reader_reexecuted=False,current_public_full4_rehashed=False,current_model_identity_qualified=False,current_GPU_health_transferred=False,full_model_math_qualified=False,current_API_or_serving_qualified=False,current_SDK_and_config_must_be_independently_admitted=True,fresh_per_arm_full4_pages_health_and_retirement_required=True,historical_completed_evidence_only=True)

def binding():
    source=source_binding();api_raw,api_before=bound(API_PREPARED,API_PREPARED_SHA);api=parse(api_raw);record=api['historical_prerequisites']['normal'];public_raw,public_before=bound(PUBLIC,PUBLIC_SHA);public=parse(public_raw);report_raw,report_before=bound(NORMAL/'report.json',REPORT_SHA);report=parse(report_raw);result=envelope(record,public,report)
    before=tree_binding(NORMAL,record['byte_roster']);after=tree_binding(NORMAL,record['byte_roster']);require(canonical(before)==canonical(after),'Historical consumed bytes/stat changed across association')
    for old in(api_before,public_before,report_before):_,fresh=bound(old['path'],old['sha256']);require(fresh==old,'Historical proof input changed across raw reads')
    require(source_binding()==source,'Historical association source changed across raw reads')
    result.update(association_source_plan_sha256=source,original_API_capture_sha256=API_PREPARED_SHA,original_public_admission_sha256=PUBLIC_SHA,historical_raw_current_bindings=after,original_model_shard_files_read=False,saved_historical_artifact_bytes_read=True,new_semantic_or_model_admission_performed=False);return result

def match_current_contract(association,current_sdk_binding,current_config):
    """Field equality ONLY: caller supplies separately admitted actual current values."""
    require(association['historical_completed_evidence_only']is True and association['current_model_identity_qualified']is False,'Historical scope cannot become current authority')
    require(canonical(current_sdk_binding)==canonical(association['original_public_admission']['source43_sdk']),'Actual current43 SDK differs from historical normal constructor')
    raw,_=bound(NORMAL/'config.snapshot.json',CONFIG_SHA);expected=parse(raw);require(canonical(current_config)==canonical(expected),'Original mode0 normal configuration differs')
    return dict(historical_current_contract_fields_matched=True,current_SDK_bytes_verified_by_this_function=False,current_model_identity_qualified=False,fresh_per_arm_identity_and_runtime_gates_still_required=True)
