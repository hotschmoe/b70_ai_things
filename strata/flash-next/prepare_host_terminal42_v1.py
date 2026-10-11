"""Source-only patch42: owned H36 terminal export before normal serve _Exit."""
import difflib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261011T052714Z-k9s4qsio')
BASE=HERE/'indexer-materialized41-engine-build-proposal-v3.json';BASE_SHA='2f40973fb4c0519c1b99434a820b07dc44ada19a8c4d7fda8aecc11110c7d01a'
G='sycl/src/program/generate.cpp';H='sycl/include/strata/program/host_critical_path_trace.hpp'
PATCH=HERE/'patches/0042-explicit-owned-idempotent-host-trace-terminal.patch'
OLD='delete[] s.events;s.events=nullptr; }'
NEW='s.on.store(false,std::memory_order_release);delete[] s.events;s.events=nullptr; }'
ANCHOR='''        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(0);
'''
REPLACEMENT='''        if (strata::program::host_trace36::enabled())
            strata::program::host_trace36::exit_export();
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(0);
'''

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def changes():
    require(sha(BASE)==BASE_SHA,'Exact frozen source41 parent plan required');parent=json.loads(BASE.read_bytes());old={}
    for name,want in parent['expected_patched_source_sha256'].items():require(sha(SDK/'source'/name)==want,'Exact actual source41 bytes changed '+name)
    for name in(G,H):old[name]=(SDK/'source'/name).read_text()
    require(old[G].count(ANCHOR)==1 and old[H].count(OLD)==1,'Exact normal serve exit/header terminal point required');new={G:old[G].replace(ANCHOR,REPLACEMENT),H:old[H].replace(OLD,NEW)}
    require(new[G].replace(REPLACEMENT,ANCHOR)==old[G]and new[H].replace(NEW,OLD)==old[H],'Exact full-byte undo required');return old,new

def patch_text():
    old,new=changes();return ''.join(''.join(difflib.unified_diff(old[name].splitlines(True),new[name].splitlines(True),fromfile='a/'+name,tofile='b/'+name))for name in(G,H))
def proposal():
    _,new=changes();parent=json.loads(BASE.read_bytes());parent['patches']=[*parent['patches'],{'path':str(PATCH.relative_to(ROOT)),'sha256':sha(PATCH)}];parent['expected_patched_source_sha256']={**parent['expected_patched_source_sha256'],**{name:hashlib.sha256(body.encode()).hexdigest()for name,body in new.items()}};parent.update(status='SOURCE ONLY source42 H36 terminal lifecycle correction; fresh ABI/model/upload proof absent',source42_parent_plan_sha256=BASE_SHA,source42_parent_actual_sdk=str(SDK),source_generation=42,terminal_H36_explicit_idempotent=True,normal_fast_exit_preserved=True,PTI_destructor_flush_fixed=False,old_source41_runtime_proof_transferred=False,full_model_math_qualified=False)
    return parent
