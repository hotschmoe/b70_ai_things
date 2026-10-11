"""Explicit historical identity admission against fresh publisher association.

This returns a labelled historical projection, never a current-stat PASS.
The original receipt and its chronology remain immutable.
"""
from pathlib import Path
from postboot_original_model_association_v1 import historical_stat, require, sha
from serial37_canonical_json_v3 import read_unique
import hashlib

def identity_admission(path,lock_path,shards,after_epoch,*,model_association,original_binding):
    path=Path(path);lock_path=Path(lock_path)
    require(not path.is_symlink() and path.resolve()==path,'Original identity alias refused')
    raw=path.read_bytes();value=read_unique(path);lock=read_unique(lock_path)
    expected=[r for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')]
    require(value.get('passed') is True and value['lock_sha256']==sha(lock_path)
            and value['model_revision']==lock['revision'],'Original lock/revision differs')
    require(len(expected)==len(shards)==len(value['rows'])==4
            and value['started']>=after_epoch and value['finished']>=value['started'],
            'Original complete source chronology differs')
    for row,want,p in zip(value['rows'],expected,shards):
        require(row['path']==str(p) and Path(p).name==Path(want['path']).name,
                'Original ordered source association differs')
        require(row['passed'] is True and row['sha256']==row['expected_sha256']==want['sha256']
                and row['bytes']==want['size'] and row['stat_before']==row['stat_after'],
                'Original publisher identity differs')
        historical_stat(model_association,p,row['stat_after'])
    require(path.read_bytes()==raw,'Original source receipt changed during admission')
    historical={'path':str(path),'sha256':sha(path),'started':value['started'],
                'finished':value['finished'],'current_stat_verified':True,
                'complete_four_publisher_hashes_verified':True}
    require(original_binding==historical,'Original identity binding contradicts original receipt')
    return {'original_binding':original_binding,'historical_evidence_only':True,
            'old_current_stat_gate_passed':False,'current_runtime_qualified':False,
            'postboot_association_sha256':model_association['current_identity_sha256']}

def native_identity(path,lock,shards,recorded,*,model_association):
    """Recheck a recorded NUM10/P30 identity with explicit fresh page evidence."""
    path=Path(path);require(not path.is_symlink() and path.resolve()==path,'Original identity alias refused')
    raw=path.read_bytes();value=read_unique(path)
    expected=[r for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')]
    require(value['passed'] is True and value['model_revision']==lock['revision']
            and value['lock_sha256']==model_association['current_lock_sha256']
            and len(value['rows'])==len(shards)==len(expected)==4,'Original native identity roster differs')
    for row,want,p in zip(value['rows'],expected,shards):
        require(row['path']==str(p) and row['passed'] is True and row['bytes']==want['size']
                and row['sha256']==row['expected_sha256']==want['sha256']
                and row['stat_before']==row['stat_after'],'Original native publisher differs')
        historical_stat(model_association,p,row['stat_after'])
    require(recorded['path']==str(path) and recorded['sha256']==hashlib.sha256(raw).hexdigest()
            and recorded['model_revision']==lock['revision'],'Original native binding differs')
    from run_source_upload_oracle_full import SENTINEL_OFFSET,SENTINEL_BYTES,SENTINEL_SHA
    require(recorded['sentinel']=={'path':str(shards[2]),'offset':SENTINEL_OFFSET,
            'bytes':SENTINEL_BYTES,'sha256':SENTINEL_SHA},'Original native sentinel declaration differs')
    with Path(shards[2]).open('rb') as handle:
        handle.seek(SENTINEL_OFFSET);data=handle.read(SENTINEL_BYTES)
    require(len(data)==SENTINEL_BYTES and hashlib.sha256(data).hexdigest()==SENTINEL_SHA,
            'Fresh actual known original page differs')
    from source_page_watchdog_v3 import guard
    current=guard(shards[2]);original=recorded['current_known_pages']
    require(current['passed'] is True,'Fresh native knownpage observation failed')
    historical_stat(model_association,shards[2],original['stat_after'])
    def stable(row):
        import copy
        row=copy.deepcopy(row);row.pop('epoch',None)
        for key in ('stat_before','stat_after'):
            if key in row:row[key]=row[key][1:]
        return row
    require(stable(current)==stable(original),'Current knownpages differ beyond associated st_dev/epoch')
    require(path.read_bytes()==raw,'Original native identity changed during admission')
    return {'original_binding':recorded,'fresh_known_pages':current,
            'old_current_stat_gate_passed':False,'historical_evidence_only':True,
            'current_runtime_qualified':False}

