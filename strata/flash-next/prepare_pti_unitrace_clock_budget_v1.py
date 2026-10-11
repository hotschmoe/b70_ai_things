"""SOURCE ONLY fixed512MiB clock metadata overlay over accepted bridge source."""
from pathlib import Path
import difflib,hashlib,json
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/mnt/vm_8tb/b70/build/pti-unitrace-clock-bridge-source-v1')
BASE_PLAN=ROOT/'strata/flash-next/pti-unitrace-clock-bridge-source-plan-v1.json'
BASE_PLAN_SHA='110ca7f1b376ddd253e4b47410590fb7b7900778fc03fb535c7d168a10948def'
HEADER='tools/unitrace/src/pti_unitrace_clock_export_v1.h'
OLD='value.size() > 4096 || bytes + value.size() > (8u << 20)'
NEW='value.size() > 4096 || bytes + value.size() > (512u << 20)'
METADATA_CAP=512<<20
ROW_CAP=4096

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def base_binding():
    raw=BASE_PLAN.read_bytes();require(sha(raw)==BASE_PLAN_SHA,'Frozen bridge plan changed')
    expected=json.loads(raw)['expected_reconstructed_source_sha256']
    for name,want in expected.items():require(sha((BASE/name).read_bytes())==want,'Accepted source changed '+name)
    return expected

def changes():
    base_binding();raw=(BASE/HEADER).read_bytes()
    require(raw.count(OLD.encode())==1 and NEW.encode()not in raw,'Exactly one original metadata ceiling required')
    new=raw.replace(OLD.encode(),NEW.encode())
    require(new.replace(NEW.encode(),OLD.encode())==raw,'Full byte undo failed')
    return {HEADER:new}
def patch_text():
    return ''.join(difflib.unified_diff((BASE/HEADER).read_text().splitlines(True),changes()[HEADER].decode('ascii').splitlines(True),fromfile='a/'+HEADER,tofile='b/'+HEADER))
def admission(current_bytes,row_bytes,cap=METADATA_CAP):
    """CPU reference of unchanged native byte guard; not native execution proof."""
    require(all(type(x)is int and x>=0 for x in(current_bytes,row_bytes,cap)),'Exact nonnegative byte extents required')
    require(0<cap<=METADATA_CAP and current_bytes<=cap,'Bounded CPU fixture extent required')
    require(row_bytes<=ROW_CAP and current_bytes+row_bytes<=cap,'metadata output bound exceeded')
    return current_bytes+row_bytes
if __name__=='__main__':print(patch_text(),end='')
