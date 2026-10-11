"""Named historical43 prerequisite byte association; never a current old gate."""
import hashlib
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
from source43_api_baseline_profile_v1 import require

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tree(root):
    original=Path(root).absolute();require(not original.is_symlink()and not any(p.is_symlink()for p in original.parents),'Historical root aliases refused');root=original.resolve();require(root.is_dir()and not root.is_symlink()and not any(p.is_symlink()for p in root.rglob('*')),'Exact historical prerequisite tree required');rows={}
    for p in sorted(root.rglob('*')):
        if p.is_file():
            require(p.stat().st_size<=1<<30,'Historical prerequisite file bound');before=p.stat();digest=sha(p);after=p.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Historical prerequisite changed during bytes');rows[str(p.relative_to(root))]={'sha256':digest,'bytes':after.st_size}
    require('report.json'in rows,'Original completed prerequisite report required');return rows

def capture(root,admitted,role):
    root=Path(root).resolve();before=tree(root);report=read_unique(root/'report.json');require(report['passed']is True and admitted['report_sha256']==before['report.json']['sha256'],'Actual admitted historical43 prerequisite required');require(tree(root)==before,'Historical prerequisite changed during binding');return {'role':role,'root':str(root),'report_sha256':before['report.json']['sha256'],'original_public_admission':admitted,'byte_roster':before,'old_current_registry_gate_passed':False,'historical_completed43_source_and_lifecycle_only':True,'fresh_API_or_original_math_authority_transferred':False}

def rejoin(record):
    require(record['old_current_registry_gate_passed']is False and record['historical_completed43_source_and_lifecycle_only']is True and record['fresh_API_or_original_math_authority_transferred']is False,'Historical/current scope must remain explicit');root=Path(record['root']);require(canonical(tree(root))==canonical(record['byte_roster'])and read_unique(root/'report.json')['passed']is True and sha(root/'report.json')==record['report_sha256']==record['original_public_admission']['report_sha256'],'Actual complete prior report/raw bytes missing or changed');return record
