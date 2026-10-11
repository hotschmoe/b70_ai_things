"""Independent lazy CPU original-GGUF foundation; no Strata arithmetic imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import numpy as np

GEOMETRY={'F32':(1,4),'BF16':(1,2),'Q8_0':(32,34),'Q5_1':(32,24),
          'Q4_K':(256,144),'Q5_K':(256,176),'IQ4_NL':(32,18)}
IQ4=np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113],dtype=np.float64)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def half(raw,offset=0):return struct.unpack_from('<e',raw,offset)[0]


def decode(raw,kind,lane='original_fp64'):
    """Exact encoded real values or scalar GGML F32 dequantization storage.

    fp64 lane does NOT first round dequantized weights to F32. ggml_f32 lane
    explicitly rounds multiplication/subtraction as the official scalar decoder.
    Neither lane emulates dot-product activation quantization.
    """
    if kind not in GEOMETRY or lane not in ('original_fp64','ggml_f32'):raise ValueError('Unknown format/lane')
    elements,width=GEOMETRY[kind]
    if len(raw)%width:raise ValueError('Partial encoded block')
    if kind=='F32':out=np.frombuffer(raw,dtype='<f4').astype(np.float64)
    elif kind=='BF16':out=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    else:
        out=np.empty(len(raw)//width*elements,dtype=np.float64)
        rnd=lambda value:float(np.float32(value)) if lane=='ggml_f32' else float(value)
        for b in range(len(raw)//width):
            packet=raw[b*width:(b+1)*width];d=half(packet)
            for j in range(elements):
                if kind=='Q8_0':value=d*struct.unpack_from('b',packet,2+j)[0]
                elif kind=='IQ4_NL':value=d*IQ4[(packet[2+j%16]>>(4*(j//16)))&15]
                elif kind=='Q5_1':
                    code=((packet[8+j%16]>>(4*(j//16)))&15)|(((int.from_bytes(packet[4:8],'little')>>j)&1)<<4)
                    value=rnd(d*code)+half(packet,2)
                else:
                    group=j//32;within=j%32;scales=packet[4:16]
                    if group<4:scale=scales[group]&63;minimum=scales[group+4]&63
                    else:scale=(scales[group+4]&15)|((scales[group-4]>>6)<<4);minimum=(scales[group+4]>>4)|((scales[group]>>6)<<4)
                    offset=16 if kind=='Q4_K' else 48
                    code=(packet[offset+(group//2)*32+within]>>(4*(group%2)))&15
                    if kind=='Q5_K':code|=((packet[16+within]>>group)&1)<<4
                    value=rnd(rnd(d*scale)*code)-rnd(half(packet,2)*minimum)
                out[b*elements+j]=rnd(value)
    if not np.isfinite(out).all():raise ValueError('Nonfinite encoded values')
    return out


class OriginalGgufPostboot:
    """Bounded pread loader bound to pinned inventory, lock and full-hash receipt.

    Receipt/stat/header binding is admission evidence, not a new whole-file hash.
    MTP sidecar is excluded by exact selected shard roster.
    """
    def __init__(self,manifest,identity,max_read_bytes=None,*,model_association):
        from postboot_original_model_association_v1 import association as current_association
        from serial37_canonical_json_v3 import canonical
        fresh=current_association(identity,model_association['current_identity_path'],current_expected_sha256=model_association['current_identity_sha256'])
        if canonical(fresh)!=canonical(model_association):raise ValueError('Exact current byte association changed')
        self.manifest=json.loads(Path(manifest).read_bytes());m=self.manifest
        cap=m['max_decoded_or_read_bytes_per_call']
        if type(cap) is not int or not 0<cap<=64*1024*1024:raise ValueError('Manifest reader cap invalid')
        if max_read_bytes is None:max_read_bytes=cap
        if type(max_read_bytes) is not int or not 0<max_read_bytes<=cap:raise ValueError('Caller reader cap exceeds manifest or is invalid')
        if sha(m['inventory'])!=m['inventory_sha256'] or sha(m['lock'])!=m['lock_sha256']:raise ValueError('Pinned inventory/lock changed')
        inventory=json.loads(Path(m['inventory']).read_bytes());lock=json.loads(Path(m['lock']).read_bytes())
        receipt=json.loads(Path(identity).read_bytes())
        if receipt.get('passed') is not True or receipt.get('lock_sha256')!=m['lock_sha256'] or receipt.get('model_revision')!=lock['revision']:raise ValueError('Full identity receipt not admitted')
        self.files=[f for f in inventory['files'] if '/UD-Q4_K_XL/' in f['path']]
        if len(self.files)!=4:raise ValueError('Selected shard coverage differs')
        rows={r['path']:r for r in receipt['rows']}
        if set(rows)!={f['path'] for f in self.files}:raise ValueError('Full receipt shard coverage differs')
        expected={Path(f['path']).name:f for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')}
        self.tensors={};self.max_read_bytes=max_read_bytes
        for f in self.files:
            p=Path(f['path']);r=rows[str(p)];st=p.stat();sig=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
            if r.get('passed') is not True or r['stat_before']!=r['stat_after']:raise ValueError('Original source receipt changed')
            from postboot_original_model_association_v1 import historical_stat
            historical_stat(model_association,p,r['stat_after'])
            if sig!=next(row['current_stat5'] for row in model_association['mapping'] if row['path']==str(p)):raise ValueError('Current associated source stat changed')
            if r['sha256']!=r['expected_sha256'] or r['sha256']!=expected[p.name]['sha256'] or st.st_size!=expected[p.name]['size']:raise ValueError('Publisher identity differs')
            with p.open('rb') as stream:header=stream.read(f['header_bytes_read'])
            if hashlib.sha256(header).hexdigest()!=f['header_sha256']:raise ValueError('Actual header differs from pinned inventory')
            for t in f['tensors']:
                if t['name'] in self.tensors:raise ValueError('Duplicate role')
                if t['type'] not in GEOMETRY:raise ValueError('Unsupported actual source format')
                count,width=GEOMETRY[t['type']];shape=t['shape_ggml_order']
                if math.prod(shape)!=t['elements'] or shape[0]%count or t['packed_bytes']!=t['elements']//count*width:raise ValueError('Invalid block/role geometry')
                if t['absolute_offset']<f['tensor_data_offset'] or t['absolute_offset']+t['packed_bytes']>st.st_size:raise ValueError('Tensor extent exceeds shard')
                self.tensors[t['name']]=(f,t,sig)
        self.validate_roles()

    def validate_roles(self):
        m=self.manifest
        metadata=self.files[0]['metadata']
        for key,expected in m['geometry'].items():
            got=metadata[key]['value']
            if isinstance(got,dict):got=got.get('values')
            if got!=expected:raise ValueError('Actual qwen4exp geometry differs: '+key)
        roster={name:{'type':t['type'],'shape':t['shape_ggml_order']} for name,(_,t,_) in self.tensors.items()}
        digest=hashlib.sha256(json.dumps(roster,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=m['tensor_role_sha256'] or len(roster)!=m['tensor_count']:raise ValueError('Actual full tensor-role coverage differs')
        for layer in range(48):
            for role in ('hc_attn_norm','hc_attn_down','hc_attn_up','hc_attn_inject','hc_ffn_norm','hc_ffn_down','hc_ffn_up','hc_ffn_inject','ffn_gate_exps','ffn_up_exps','ffn_down_exps'):
                if 'blk.%d.%s.weight'%(layer,role) not in roster:raise ValueError('Missing mandatory layer role')

    def rows(self,name,indices,lane='original_fp64'):
        f,t,sig=self.tensors[name];shape=t['shape_ggml_order'];elements,width=GEOMETRY[t['type']]
        row_bytes=shape[0]//elements*width;total_rows=math.prod(shape[1:]);indices=list(indices)
        if len(indices)*row_bytes>self.max_read_bytes or len(indices)*shape[0]*8>self.max_read_bytes:raise ValueError('Bounded reference read/cache budget exceeded')
        if any(type(i) is not int or not 0<=i<total_rows for i in indices):raise ValueError('Row index invalid')
        fd=os.open(f['path'],os.O_RDONLY)
        try:
            st=os.fstat(fd)
            if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=sig:raise ValueError('Source changed before read')
            values=[]
            for i in indices:
                raw=os.pread(fd,row_bytes,t['absolute_offset']+i*row_bytes)
                if len(raw)!=row_bytes:raise ValueError('Short source read')
                values.append(decode(raw,t['type'],lane))
            st=os.fstat(fd)
            if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=sig:raise ValueError('Source changed during read')
            return np.asarray(values,dtype=np.float64).reshape(len(indices),shape[0])
        finally:os.close(fd)
