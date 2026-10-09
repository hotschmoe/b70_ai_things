"""Independent vectorized decoder; frozen scalar v1 remains triangulation oracle."""
import os
import numpy as np

GEOMETRY={'F32':(1,4),'BF16':(1,2),'Q8_0':(32,34),'Q5_1':(32,24),'Q4_K':(256,144),'Q5_K':(256,176),'IQ4_NL':(32,18)}
IQ4=np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113],dtype=np.float64)


def decode(raw,kind,lane='original_fp64'):
    if kind not in GEOMETRY or lane not in ('original_fp64','ggml_f32'):raise ValueError('Unknown independent format/lane')
    count,width=GEOMETRY[kind]
    if len(raw)%width:raise ValueError('Partial encoded block')
    if kind=='F32':out=np.frombuffer(raw,dtype='<f4').astype(np.float64)
    elif kind=='BF16':out=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    else:
        packets=np.frombuffer(raw,dtype=np.uint8).reshape(-1,width);blocks=len(packets)
        half=lambda offset:packets[:,offset:offset+2].copy().view('<f2').reshape(blocks,1).astype(np.float64)
        rnd=lambda x:x.astype(np.float32).astype(np.float64) if lane=='ggml_f32' else x
        d=half(0)
        if kind=='Q8_0':out=rnd(d*packets[:,2:].view(np.int8))
        elif kind=='IQ4_NL':
            codes=np.concatenate([packets[:,2:]&15,packets[:,2:]>>4],axis=1);out=rnd(d*IQ4[codes])
        elif kind=='Q5_1':
            low=np.concatenate([packets[:,8:]&15,packets[:,8:]>>4],axis=1)
            high=np.unpackbits(packets[:,4:8],axis=1,bitorder='little')*16
            out=rnd(rnd(d*(low|high))+half(2))
        else:
            packed=packets[:,4:16]
            scales=np.concatenate([packed[:,:4]&63,(packed[:,8:]&15)|((packed[:,:4]>>6)<<4)],axis=1).astype(np.float64)
            minimum=np.concatenate([packed[:,4:8]&63,(packed[:,8:]>>4)|((packed[:,4:8]>>6)<<4)],axis=1).astype(np.float64)
            start=16 if kind=='Q4_K' else 48
            q=packets[:,start:].reshape(blocks,4,32)
            codes=np.stack([q&15,q>>4],axis=2).reshape(blocks,8,32)
            if kind=='Q5_K':codes=codes|(((packets[:,16:48,None].transpose(0,2,1)>>np.arange(8,dtype=np.uint8)[None,:,None])&1)*16)
            out=rnd(rnd(rnd(d*scales)[:,:,None]*codes)-rnd(half(2)*minimum)[:,:,None])
    out=out.reshape(-1)
    if not np.isfinite(out).all():raise ValueError('Nonfinite encoded values')
    return out


def rows(reader,name,indices,lane='original_fp64'):
    """Use frozen v1 metadata/identity binding, fresh per-read stat and new decoder."""
    f,t,sig=reader.tensors[name];shape=t['shape_ggml_order'];elements,width=GEOMETRY[t['type']]
    indices=list(indices);row_bytes=shape[0]//elements*width;total_rows=int(np.prod(shape[1:],dtype=np.int64))
    if len(indices)*row_bytes>reader.max_read_bytes or len(indices)*shape[0]*8>reader.max_read_bytes:raise ValueError('Bounded source/read expansion exceeded')
    if any(type(i) is not int or not 0<=i<total_rows for i in indices):raise ValueError('Invalid source row')
    fd=os.open(f['path'],os.O_RDONLY)
    try:
        def stable():
            st=os.fstat(fd)
            if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=sig:raise ValueError('Source stat changed at row boundary')
        stable();raw=[]
        for i in indices:
            packet=os.pread(fd,row_bytes,t['absolute_offset']+i*row_bytes)
            if len(packet)!=row_bytes:raise ValueError('Short original row')
            raw.append(packet)
        stable();return decode(b''.join(raw),t['type'],lane).reshape(len(indices),shape[0])
    finally:os.close(fd)
