"""Named original three-CPP dispatch identity association; no old source PASS."""
import hashlib
from pathlib import Path
from serial37_canonical_json_v3 import canonical
from postboot_original_model_association_v1 import require
CODE=('sycl/src/program/generate.cpp','sycl/src/core/verify.cpp','sycl/src/prefill/prefill.cpp')

def stat5(path):
    s=path.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def admit(original,current,model_association):
    old=original['HC_source']['dispatch']['files'];new=current['HC_source']['dispatch']['files']
    require(set(old)==set(new)==set(CODE),'Exact original three dispatch source files required')
    old_devices={r['historical_stat5'][0] for r in model_association['mapping']}
    new_devices={r['current_stat5'][0] for r in model_association['mapping']}
    require(len(old_devices)==len(new_devices)==1,'Exact model-remount mapping required')
    rows=[]
    for name in CODE:
        before=old[name];after=new[name]
        require(set(before)==set(after)=={'path','sha256','stat5'} and
                before['path']==after['path'] and before['sha256']==after['sha256'],
                'Original dispatch source path/hash differs')
        for value in (before['stat5'],after['stat5']):
            require(type(value) is list and len(value)==5 and all(type(v) is int for v in value),
                    'Exact typed original dispatch stat5 required')
        require(before['stat5'][1:]==after['stat5'][1:] and
                before['stat5'][0] in old_devices and after['stat5'][0] in new_devices,
                'ONLY named remount device may change for dispatch source')
        path=Path(after['path'])
        require(path.is_absolute() and not path.is_symlink() and path.resolve()==path and
                path.as_posix().endswith('/'+name),'Exact nonalias source path required')
        state=stat5(path);require(state==after['stat5'] and 0<state[2]<=64<<20,
                                'Bounded current exact source stat required before read')
        raw=path.read_bytes()
        require(stat5(path)==state and len(raw)==state[2] and
                hashlib.sha256(raw).hexdigest()==before['sha256'] and
                path.read_bytes()==raw and stat5(path)==state,
                'Exact consumed original CPP bytes changed during association')
        rows.append({'path':str(path),'sha256':before['sha256'],'bytes':len(raw),
                     'historical_stat5':before['stat5'],'current_stat5':state})
    # Compare every other original source field without replacing any stat.
    def all_other_equal(a,b,route=()):
        if route in [('HC_source','dispatch','files',name,'stat5') for name in CODE]:
            return type(a) is list and type(b) is list and a[1:]==b[1:]
        if type(a) is not type(b):return False
        if isinstance(a,dict):
            return set(a)==set(b) and all(all_other_equal(a[k],b[k],route+(k,))for k in a)
        return canonical(a)==canonical(b)
    require(all_other_equal(original,current),'Non-device original source-binding field changed')
    return {'schema':1,'rows':rows,'original_binding':original,'current_binding':current,
            'only_declared_source_st_dev_associated':True,'old_current_source_gate_passed':False,
            'source_bytes_changed':False,'compiled_flags_or_mathematics_changed':False,
            'historical_GPU_health_transferred':False,'full_model_math_qualified':False}
