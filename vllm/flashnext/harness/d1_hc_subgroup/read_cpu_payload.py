"""Read trusted torch.save CPU archive bits using only stdlib; no Torch import."""
import collections
import io
import json
from pathlib import Path
import pickle
import struct
import zipfile

S=Path(__file__).resolve().parent
BUILD=Path('/mnt/vm_8tb/b70/build/vllm-flashnext/d1/hc_subgroup_r1')
P=BUILD/'probe-r2/large_gate-1-cpu-payload.pt'
class Read(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='collections'and name=='OrderedDict':return collections.OrderedDict
        if module=='torch'and name.endswith('Storage'):return name
        if module=='torch._utils'and name in ('_rebuild_tensor_v2','_rebuild_tensor'):
            return lambda storage,offset,size,stride,*rest:dict(storage=storage,offset=offset,size=size,stride=stride)
        raise ValueError((module,name))
    def persistent_load(self,item):
        kind,dtype,key,device,count=item
        assert kind=='storage'and device=='cpu'
        return dict(dtype=dtype,key=key,count=count)
with zipfile.ZipFile(P)as z:
    prefix=next(n[:-8]for n in z.namelist()if n.endswith('data.pkl'))
    x=Read(io.BytesIO(z.read(prefix+'data.pkl'))).load()
    def bit(t,h):
        assert t['storage']['dtype']=='BFloat16Storage'
        i=t['offset']+h*t['stride'][-1]
        b=z.read(prefix+'data/'+t['storage']['key'])
        return struct.unpack_from('<H',b,i*2)[0]
    rows=[]
    for h in x['h_locations']:
        outputs={n:bit(v,h)for n,v in x['outputs'].items()}
        rows.append({'h':h,'bf16_bits':{n:hex(v)for n,v in outputs.items()},
                     'all_magnitudes_exact_zero':all(v&0x7fff==0 for v in outputs.values())})
    r={'scope':'stdlib raw BF16 storage read; no CPU float arithmetic/Torch/XPU import',
       'rows':rows,'payload_bytes':P.stat().st_size,
       'strict_gate':'FAIL remains; unequal +0/-0 bits; timing skipped'}
    (BUILD/'probe-r2-offline-ported.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii')
    print(json.dumps(r))
